"""Append two extreme utilities (and their concordant thresholds) to the sweeps.

The original factorial / threshold sweeps used five cost functions spanning
(10,1) ... (1,10)  (FN/FP = 0.1 ... 10).  This script adds two *extreme* points
requested afterwards:

    (c_FP, c_FN)   FN/FP    p* = c_FP/(c_FP+c_FN)   stance
    (100,   1)     0.01     0.99                    efficiency-first (near-never refer)
    (  1, 100)   100.00     0.01                    safety-first  (almost-always refer)

For every one of the 1,248 factorial variants and every model x reasoning
configuration we elicit, for each of these two cost functions:

  * a utility-prompted admit decision   (regime ``decision_u_fp{fp}_fn{fn}``)
    -> appended to the FACTORIAL sweep directory, next to the existing five.
  * a threshold-prompted admit decision (regime ``decision_t_fp{fp}_fn{fn}``)
    -> appended to the THRESHOLD sweep directory, next to the existing five.

We deliberately do NOT re-elicit beliefs or the baseline decision; the existing
beliefs are reused for analysis.  Appends are resumable and non-destructive: we
reuse ``run_threshold_sweep.run_config_resumable`` which skips (context_id,
regime) pairs already present and flushes after every row.

Usage (one process per model family, 3 reasoning levels sequential inside):
    python run_extra_utilities.py \
        --configs gpt-5-mini:minimal,gpt-5-mini:medium,gpt-5-mini:high \
        --parallel 100 --run-tag mini
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import revealed_preferences as rp
import run_factorial_sweep as fs
import run_threshold_sweep as ts

for _name in ("httpx", "openai", "openai._base_client", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

THEIR_CSV = fs.THEIR_CSV

# The two extreme cost functions to add (c_FP, c_FN).
EXTRA_COST_FUNCTIONS: list[tuple[int, int]] = [(100, 1), (1, 100)]

# Defaults matching the existing sweep directories.
DEFAULT_FACTORIAL_ROOT = rp.OUTPUT_ROOT / "2026-07-01_factorial"
DEFAULT_THRESHOLD_ROOT = rp.OUTPUT_ROOT / "2026-07-08_threshold"


def _base_rows(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    for r in rows:
        case_id = r["case_id"]
        variant = r["variant_code"]
        context = fs.extract_context(r["prompt_text"])
        gold_triage = r["gold_triage"]
        needs_er = 1 if "D" in gold_triage else 0
        prompt_type = 1 if case_id[0] in ("E", "M") else 2
        yield context, dict(
            context_id=f"{case_id}__{variant}",
            case_number=int(r["scenario_num"]) if r.get("scenario_num") else 0,
            vignette_id=case_id,
            prompt_type=prompt_type,
            gold_standard=gold_triage,
            classification="edge" if r.get("is_edge_case") == "yes" else "clear",
            needs_er_gold=needs_er,
            variant_code=variant,
            race=r["race"],
            gender=r["gender"],
            has_anchor=r["has_anchor"],
            has_barrier=r["has_barrier"],
            their_triage=r["llm_triage"],
        )


def build_utility_items(csv_path: Path) -> list[fs._Item]:
    """Utility (cost-function) decision prompts for the extra cost functions."""
    items: list[fs._Item] = []
    for context, base in _base_rows(csv_path):
        for c_fp, c_fn in EXTRA_COST_FUNCTIONS:
            d = dict(base)
            d.update(
                regime=f"decision_{rp.cost_label(c_fp, c_fn)}",
                c_fp=c_fp, c_fn=c_fn, p_star=rp.admit_threshold(c_fp, c_fn),
                prompt_text=f"{context}\n\n{rp.build_decision_utility_question(c_fp, c_fn)}",
            )
            items.append(fs._Item(d))
    return items


def build_threshold_items(csv_path: Path) -> list[fs._Item]:
    """Threshold decision prompts for the extra cost functions' break-even p*."""
    items: list[fs._Item] = []
    for context, base in _base_rows(csv_path):
        for c_fp, c_fn in EXTRA_COST_FUNCTIONS:
            p_star = round(rp.admit_threshold(c_fp, c_fn), 2)
            d = dict(base)
            d.update(
                regime=f"decision_{rp.threshold_label(c_fp, c_fn)}",
                c_fp=c_fp, c_fn=c_fn, p_star=p_star,
                prompt_text=f"{context}\n\n{rp.build_decision_threshold_question(p_star)}",
            )
            items.append(fs._Item(d))
    return items


def parse_configs(args) -> list[tuple[str, str]]:
    if args.configs:
        pairs = []
        for pair in args.configs.split(","):
            pair = pair.strip()
            if pair:
                model, eff = pair.split(":")
                pairs.append((model.strip(), eff.strip()))
        return pairs
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    reasonings = [r.strip() for r in args.reasonings.split(",") if r.strip()]
    return [(m, e) for m in models for e in reasonings]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gpt-5-mini")
    ap.add_argument("--reasonings", default="minimal,medium,high")
    ap.add_argument("--configs", default=None)
    ap.add_argument("--endpoint", default=rp.AZURE_RESPONSES_ENDPOINT)
    ap.add_argument("--parallel", type=int, default=100)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--factorial-root", default=str(DEFAULT_FACTORIAL_ROOT))
    ap.add_argument("--threshold-root", default=str(DEFAULT_THRESHOLD_ROOT))
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    factorial_root = Path(args.factorial_root)
    threshold_root = Path(args.threshold_root)
    tag = args.run_tag or "extra"
    config_pairs = parse_configs(args)

    util_items = build_utility_items(THEIR_CSV)
    thr_items = build_threshold_items(THEIR_CSV)
    n_ctx = len({it.context_id for it in util_items})

    if args.dry_run:
        print(f"[dry-run] utility items: {len(util_items)} "
              f"({len(util_items)//max(n_ctx,1)} regimes/variant), "
              f"threshold items: {len(thr_items)} across {n_ctx} variants.")
        seen = set()
        for it in util_items + thr_items:
            if it.regime in seen:
                continue
            seen.add(it.regime)
            print("\n" + "=" * 72)
            print(f"regime={it.regime}  c_fp={it.c_fp} c_fn={it.c_fn} "
                  f"p_star={it.p_star}")
            print("-" * 72)
            print(it.prompt_text.split("\n\n", 1)[1])
        return

    # Console-handle safety on Windows when backgrounded: write to our own file.
    factorial_root.mkdir(parents=True, exist_ok=True)
    _logf = open(factorial_root / f"run_extra_{tag}.log", "a", buffering=1,
                 encoding="utf-8", errors="replace")
    sys.stdout = _logf
    sys.stderr = _logf
    logging.disable(logging.INFO)

    progress_path = factorial_root / f"progress_extra_{tag}.log"
    print(f"[{tag}] extra-utility run start {datetime.now().isoformat()}",
          file=sys.stderr)
    print(f"[{tag}] configs={config_pairs}  extra_costs={EXTRA_COST_FUNCTIONS}",
          file=sys.stderr)
    print(f"[{tag}] {len(util_items)} utility + {len(thr_items)} threshold "
          f"prompts per config.", file=sys.stderr)

    for model, eff in config_pairs:
        label = f"{model}_re-{eff}"
        for kind, items, root in (
            ("utility", util_items, factorial_root),
            ("threshold", thr_items, threshold_root),
        ):
            cfg = root / label
            cfg.mkdir(parents=True, exist_ok=True)
            print(f"\n=== [{tag}] {label} :: {kind} "
                  f"({len(items)} prompts) ===", file=sys.stderr)
            t0 = time.time()
            rpath = ts.run_config_resumable(
                items, cfg, model, args.endpoint, eff,
                args.parallel, args.api_key, progress_path)
            ts._sort_results(rpath)
            dur = time.time() - t0
            print(f"  [{tag}] {label} :: {kind} done in {dur:.0f}s",
                  file=sys.stderr)

    print(f"\n[{tag}] Done at {datetime.now().isoformat()}", file=sys.stderr)


if __name__ == "__main__":
    main()
