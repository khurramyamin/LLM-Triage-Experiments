"""Build the public ``data/`` folder from the raw sweep outputs.

Readers of the paper should not have to dig through ``revealed_preferences/``
run directories.  This script collects everything the paper figures consume
into one place:

    data/
      README.md                   (written by hand, not this script)
      models/<model>.csv          one CSV per model, containing ALL rows for
                                  that model: every reasoning effort and both
                                  experiments (utility + threshold prompting)
      original_paper/             the deployed tool's published decisions that
                                  fig1 / nature_decision_consistency compare against

Each per-model CSV is the union of the corresponding ``results.csv`` files,
with three provenance columns prepended:

    model              e.g. "gpt-5.4"
    reasoning_effort   e.g. "high"
    experiment         "utility"  = 2026-07-01_factorial sweep
                       "threshold" = 2026-07-08_threshold sweep

Failed attempts are dropped: the raw sweeps are append-only, so an item that
hit an API error / unparseable response was re-queried and appears multiple
times.  Mirroring ``analyze_sweep.load_results``, we keep — per (context_id,
regime) — only the LAST successfully parsed row (``parsed_probability`` for
the belief regime, ``parsed_decision`` for decision regimes).  Items that
never parsed are omitted; the raw attempts stay in ``revealed_preferences/``.

Run:  python3 make_data_folder.py
"""
from __future__ import annotations

import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize)

SWEEPS = [  # (experiment label, sweep dir)
    ("utility", Path("revealed_preferences/2026-07-01_factorial")),
    ("threshold", Path("revealed_preferences/2026-07-08_threshold")),
]
ORIGINAL_PAPER_FILES = [
    Path("original_paper_data/extracted/ashwinra-code-gpt-health-eval-61bce8a"
         "/data/DataExpanded_FINAL.csv"),
    Path("original_paper_data/extracted/ashwinra-code-gpt-health-eval-61bce8a"
         "/data/DataDictionary.csv"),
]
OUT = Path("data")

EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "max"]
PROVENANCE_COLS = ["model", "reasoning_effort", "experiment"]


def discover_configs() -> dict[str, list[tuple[str, str, Path]]]:
    """Return {model: [(effort, experiment, results_csv), ...]} sorted for output."""
    by_model: dict[str, list[tuple[str, str, Path]]] = {}
    for experiment, sweep in SWEEPS:
        for res in sorted(sweep.glob("*/results.csv")):
            model, effort = res.parent.name.rsplit("_re-", 1)
            by_model.setdefault(model, []).append((effort, experiment, res))
    exp_rank = {label: i for i, (label, _) in enumerate(SWEEPS)}
    for model, entries in by_model.items():
        entries.sort(key=lambda e: (exp_rank[e[1]], EFFORT_ORDER.index(e[0])))
    return by_model


def parsed_ok(row: dict) -> bool:
    """True if the LLM call produced a usable answer (same rule as analyze_sweep)."""
    field = "parsed_probability" if row["regime"] == "belief" else "parsed_decision"
    return row.get(field, "") not in (None, "")


def merge_model(model: str, entries: list[tuple[str, str, Path]]) -> Counter:
    """Write data/models/<model>.csv; return per-source kept-row counts."""
    counts: Counter = Counter()
    out_path = OUT / "models" / f"{model}.csv"
    fieldnames: list[str] | None = None
    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = None
        for effort, experiment, res in entries:
            with open(res, newline="", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                if fieldnames is None:
                    fieldnames = PROVENANCE_COLS + list(reader.fieldnames)
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()
                elif list(reader.fieldnames) != fieldnames[len(PROVENANCE_COLS):]:
                    raise SystemExit(f"schema mismatch in {res}")
                # last parsed attempt wins, in first-seen (context, regime) order
                kept: dict[tuple[str, str], dict] = {}
                for row in reader:
                    if parsed_ok(row):
                        kept[(row["context_id"], row["regime"])] = row
            for row in kept.values():
                row.update(model=model, reasoning_effort=effort,
                           experiment=experiment)
                writer.writerow(row)
            counts[(experiment, effort)] = len(kept)
    return counts


def count_rows(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # minus header


def main() -> None:
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    (OUT / "original_paper").mkdir(parents=True, exist_ok=True)

    # every (context, regime) answered = this many rows per config
    nominal = {"utility": 9 * 1248, "threshold": 7 * 1248}

    grand_total = 0
    for model, entries in sorted(discover_configs().items()):
        counts = merge_model(model, entries)
        written = count_rows(OUT / "models" / f"{model}.csv")
        if written != sum(counts.values()):
            raise SystemExit(f"row count mismatch for {model}: wrote {written}, "
                             f"kept {sum(counts.values())}")
        missing = sum(nominal[exp] - n for (exp, _), n in counts.items())
        grand_total += written
        print(f"{model:<20} {written:>6} rows kept, {missing:>3} unanswered  "
              + "  ".join(f"{exp[:4]}/{eff}={n}"
                          for (exp, eff), n in sorted(counts.items())))

    for src in ORIGINAL_PAPER_FILES:
        shutil.copy2(src, OUT / "original_paper" / src.name)
        print(f"copied {src.name}")

    print(f"\ntotal rows across models: {grand_total}")


if __name__ == "__main__":
    main()
