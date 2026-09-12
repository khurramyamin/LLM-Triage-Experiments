from __future__ import annotations

import csv
import gzip
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from . import constants as C


@dataclass(frozen=True)
class ContextMeta:
    context_id: str
    case_number: int
    gold_standard: str


@dataclass(frozen=True)
class OriginalPaperRow:
    case_id: str
    case_num: int
    gold_triage: str
    llm_triage: str
    race: str
    gender: str
    has_anchor: str
    has_barrier: str
    is_edge_case: str


@dataclass
class ConfigData:
    name: str
    family: str
    reasoning: str
    belief: dict[str, float] = field(default_factory=dict)
    decisions: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class StudyData:
    meta: dict[str, ContextMeta]
    configs: dict[str, ConfigData]
    original_paper: dict[str, OriginalPaperRow]
    source_rows: dict[str, int]
    belief_repetitions: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    repetition_source_rows: dict[str, int] = field(default_factory=dict)


def validate_repetition_collection(
    study: StudyData,
    required_configs: Sequence[str] = tuple(C.PAPER_CONFIG_ORDER),
) -> None:
    """Reject partially collected five-run data instead of silently mixing runs."""
    if not study.belief_repetitions:
        return
    expected_contexts = set(study.meta)
    problems: list[str] = []
    for name in required_configs:
        repetitions = study.belief_repetitions.get(name)
        if repetitions is None:
            problems.append(f"{name}: no repetition files")
            continue
        if len(repetitions) != C.BELIEF_REPETITION_COUNT:
            problems.append(
                f"{name}: {len(repetitions)}/{C.BELIEF_REPETITION_COUNT} repetitions"
            )
            continue
        missing = [
            len(expected_contexts.difference(repetition))
            for repetition in repetitions
        ]
        if any(missing):
            problems.append(f"{name}: missing per repetition {missing}")
    if problems:
        preview = "; ".join(problems[:5])
        suffix = f"; plus {len(problems) - 5} more" if len(problems) > 5 else ""
        raise ValueError(
            "Belief repetition collection is incomplete; refusing to generate "
            f"canonical ROC figures: {preview}{suffix}"
        )


def _open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


def _iter_rows(path: Path) -> Iterator[dict[str, str]]:
    with _open_csv(path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _factorial_context_id(row: dict[str, str]) -> str:
    race_code = {"White": "W", "Black": "B"}[row["race"]]
    gender_code = {"man": "M", "woman": "W"}[row["gender"]]
    modifiers = ""
    if row["has_anchor"] == "yes":
        modifiers += "A"
    if row["has_barrier"] == "yes":
        modifiers += "X"
    suffix = f"-{modifiers}" if modifiers else ""
    return f"{row['case_id']}__{race_code}{gender_code}{suffix}"


def load_belief_repetitions(
    root: Path,
    config_names: list[str],
) -> tuple[dict[str, list[dict[str, float]]], dict[str, int]]:
    """Load the last parseable belief per context from each repetition file."""
    repetitions: dict[str, list[dict[str, float]]] = {}
    source_rows: dict[str, int] = {}
    if not root.exists():
        return repetitions, source_rows

    for config_name in config_names:
        config_repetitions: list[dict[str, float]] = []
        found_file = False
        for repetition in range(1, C.BELIEF_REPETITION_COUNT + 1):
            path = root / f"repetition_{repetition}" / config_name / "results.csv.gz"
            if not path.exists():
                path = path.with_suffix("")
            values: dict[str, float] = {}
            row_count = 0
            if path.exists():
                found_file = True
                for row in _iter_rows(path):
                    row_count += 1
                    if row.get("regime") != "belief":
                        continue
                    probability = row.get("parsed_probability") or ""
                    if probability:
                        try:
                            value = float(probability)
                        except ValueError:
                            continue
                        if math.isfinite(value) and 0.0 <= value <= 1.0:
                            values[row["context_id"]] = value
                source_rows[str(path.relative_to(root))] = row_count
            config_repetitions.append(values)
        if found_file:
            repetitions[config_name] = config_repetitions
    return repetitions, source_rows


def load_study_data(
    models_dir: Path = C.MODELS_DIR,
    repetitions_root: Path | None = None,
) -> StudyData:
    meta: dict[str, ContextMeta] = {}
    configs: dict[str, ConfigData] = {}
    source_rows: dict[str, int] = {}

    for model_path in sorted(models_dir.glob("*.csv.gz")):
        row_count = 0
        for row in _iter_rows(model_path):
            row_count += 1
            model = row["model"] or model_path.name.removesuffix(".csv.gz")
            reasoning = row["reasoning_effort"]
            config_name = f"{model}_re-{reasoning}"
            config = configs.setdefault(
                config_name,
                ConfigData(name=config_name, family=model, reasoning=reasoning),
            )

            context_id = row["context_id"]
            case_number = int(row["case_number"])
            gold_standard = row["gold_standard"]
            existing = meta.get(context_id)
            if existing is None:
                meta[context_id] = ContextMeta(
                    context_id=context_id,
                    case_number=case_number,
                    gold_standard=gold_standard,
                )
            elif (
                existing.case_number != case_number
                or existing.gold_standard != gold_standard
            ):
                raise ValueError(f"Inconsistent metadata for {context_id}")

            regime = row["regime"]
            probability = row.get("parsed_probability") or ""
            decision = row.get("parsed_decision") or ""
            if regime == "belief":
                if probability:
                    config.belief[context_id] = float(probability)
            elif decision:
                config.decisions.setdefault(regime, {})[context_id] = int(float(decision))
        source_rows[model_path.name] = row_count

    original_paper: dict[str, OriginalPaperRow] = {}
    original_rows = 0
    for row in _iter_rows(C.ORIGINAL_PAPER_FILE):
        original_rows += 1
        context_id = _factorial_context_id(row)
        original_paper[context_id] = OriginalPaperRow(
            case_id=context_id,
            case_num=int(row["case_num"]),
            gold_triage=row["gold_triage"],
            llm_triage=row["llm_triage"],
            race=row["race"],
            gender=row["gender"],
            has_anchor=row["has_anchor"],
            has_barrier=row["has_barrier"],
            is_edge_case=row["is_edge_case"],
        )
    source_rows[C.ORIGINAL_PAPER_FILE.name] = original_rows

    missing_original = sorted(set(meta) - set(original_paper))
    missing_model = sorted(set(original_paper) - set(meta))
    if missing_original or missing_model:
        raise ValueError(
            "Original-paper alignment mismatch: "
            f"{len(missing_original)} missing in original, "
            f"{len(missing_model)} missing in model data"
        )
    repetition_root = C.BELIEF_REPETITIONS_DIR if repetitions_root is None else repetitions_root
    belief_repetitions, repetition_source_rows = load_belief_repetitions(
        repetition_root, list(configs)
    )
    return StudyData(
        meta=meta,
        configs=configs,
        original_paper=original_paper,
        source_rows=source_rows,
        belief_repetitions=belief_repetitions,
        repetition_source_rows=repetition_source_rows,
    )


def paper_config_names(study: StudyData) -> list[str]:
    return [name for name in C.PAPER_CONFIG_ORDER if name in study.configs]


def endpoint_gold(meta: dict[str, ContextMeta], endpoint: str) -> dict[str, int]:
    if endpoint == "primary":
        return {
            context_id: 1 if row.gold_standard == "D" else 0
            for context_id, row in meta.items()
            if row.case_number <= 30 and row.gold_standard != "C/D"
        }
    if endpoint == "expanded":
        return {
            context_id: 1 if "D" in row.gold_standard else 0
            for context_id, row in meta.items()
        }
    raise ValueError(f"Unknown endpoint: {endpoint}")


def endpoint_counts(meta: dict[str, ContextMeta], endpoint: str) -> dict[str, int]:
    gold = endpoint_gold(meta, endpoint)
    positives = int(sum(gold.values()))
    return {
        "n_contexts": len(gold),
        "n_positive": positives,
        "n_negative": len(gold) - positives,
    }


def original_paper_labels(rows: dict[str, OriginalPaperRow], endpoint: str) -> dict[str, int]:
    if endpoint == "primary":
        return {
            case_id: 1 if row.gold_triage == "D" else 0
            for case_id, row in rows.items()
            if row.case_num <= 60 and row.gold_triage != "C/D"
        }
    if endpoint == "expanded":
        return {case_id: 1 if "D" in row.gold_triage else 0 for case_id, row in rows.items()}
    raise ValueError(f"Unknown endpoint: {endpoint}")


def original_paper_decisions(rows: dict[str, OriginalPaperRow]) -> dict[str, int]:
    return {case_id: 1 if row.llm_triage == "D" else 0 for case_id, row in rows.items()}
