"""Full utility + threshold elicitation for the DeepSeek V4 models.

Repeats the FULL experiment (belief, baseline decision, seven prompted-utility
decisions, and seven concordant probability-threshold decisions) for the DeepSeek
V4 models at each of their three reasoning levels.

Cost functions (efficiency-first -> safety-first), FN/FP:
    (100,1)=.01  (10,1)=.1  (5,1)=.2  (1,1)=1  (1,5)=5  (1,10)=10  (1,100)=100

Reasoning levels (DeepSeek V4 exposes exactly these three via the chat.completions
top-level ``reasoning_effort`` parameter -- the Responses API rejects it):
    none  -> non-thinking / fast chat  (lowest)
    high  -> thinking
    max   -> maximum thinking

Prompted-utility decisions and beliefs -> the FACTORIAL sweep directory
(regimes belief, decision_baseline, decision_u_fp*_fn*).
Threshold decisions -> the THRESHOLD sweep directory (regimes decision_t_fp*_fn*).
Each (model, level) is written as ``<model>_re-<level>``.

Usage (one process per model, run concurrently):
    python run_deepseek_full.py --model DeepSeek-V4-Pro   --parallel 100 --run-tag pro
    python run_deepseek_full.py --model DeepSeek-V4-Flash --parallel 100 --run-tag flash
"""
from __future__ import annotations

import argparse
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

# Full seven cost functions (efficiency-first -> safety-first).
FULL_COST_FUNCTIONS = [(100, 1), (10, 1), (5, 1), (1, 1), (1, 5), (1, 10), (1, 100)]

FACTORIAL_ROOT = rp.OUTPUT_ROOT / "2026-07-01_factorial"
THRESHOLD_ROOT = rp.OUTPUT_ROOT / "2026-07-08_threshold"

# DeepSeek V4 reasoning levels (chat.completions reasoning_effort values).
REASONING_LEVELS = ["none", "high", "max"]

# chat.completions with reasoning_effort=max can take minutes; use a long timeout.
CLIENT_TIMEOUT = 600.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="e.g. DeepSeek-V4-Pro or DeepSeek-V4-Flash")
    ap.add_argument("--levels", default=",".join(REASONING_LEVELS),
                    help="comma-separated reasoning_effort levels (default none,high,max)")
    ap.add_argument("--endpoint", default=rp.AZURE_RESPONSES_ENDPOINT)
    ap.add_argument("--parallel", type=int, default=100)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--timeout", type=float, default=CLIENT_TIMEOUT,
                    help="per-request client timeout in seconds (default 600)")
    args = ap.parse_args()

    # Expand the cost-function set to the full seven for item construction.
    rp.COST_FUNCTIONS = FULL_COST_FUNCTIONS

    model = args.model
    tag = args.run_tag or model
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]

    # Build item sets once (reused across levels).
    util_items = fs.build_factorial_items(fs.THEIR_CSV)      # belief+baseline+7 utils
    thr_items = ts.build_threshold_items(ts.THEIR_CSV)        # 7 thresholds
    n_ctx = len({it.context_id for it in util_items})

    FACTORIAL_ROOT.mkdir(parents=True, exist_ok=True)
    THRESHOLD_ROOT.mkdir(parents=True, exist_ok=True)

    # Console-handle safety on Windows when backgrounded.
    _logf = open(FACTORIAL_ROOT / f"run_deepseek_{tag}.log", "a", buffering=1,
                 encoding="utf-8", errors="replace")
    sys.stdout = _logf
    sys.stderr = _logf
    logging.disable(logging.INFO)

    progress_path = FACTORIAL_ROOT / f"progress_deepseek_{tag}.log"
    print(f"[{tag}] DeepSeek full run start {datetime.now().isoformat()}",
          file=sys.stderr)
    print(f"[{tag}] model={model} levels={levels} costs={FULL_COST_FUNCTIONS}",
          file=sys.stderr)
    print(f"[{tag}] {len(util_items)} factorial + {len(thr_items)} threshold "
          f"prompts per level across {n_ctx} variants.", file=sys.stderr)

    for eff in levels:
        label = f"{model}_re-{eff}"
        for kind, items, root in (
            ("factorial", util_items, FACTORIAL_ROOT),
            ("threshold", thr_items, THRESHOLD_ROOT),
        ):
            cfg = root / label
            cfg.mkdir(parents=True, exist_ok=True)
            print(f"\n=== [{tag}] {label} :: {kind} ({len(items)} prompts) ===",
                  file=sys.stderr)
            t0 = time.time()
            rpath = ts.run_config_resumable(
                items, cfg, model, args.endpoint, eff,
                args.parallel, args.api_key, progress_path,
                api_mode="chat", timeout=args.timeout)
            ts._sort_results(rpath)
            cov = rp.coverage_from_results(rpath)
            (cfg / "settings.json").write_text(json.dumps({
                "model": model, "reasoning_effort": eff,
                "n_items": len(items), "n_contexts": n_ctx,
                "duration_sec": round(time.time() - t0, 1), "coverage": cov,
                "kind": kind, "full_utilities": True, "api": "chat.completions",
            }, indent=2), encoding="utf-8")
            print(f"  [{tag}] {label} :: {kind} done in {time.time()-t0:.0f}s",
                  file=sys.stderr)

        # Recover cost-ratio fits for this level's factorial config.
        fac_cfg = FACTORIAL_ROOT / label
        fits = rp.fit_from_results(fac_cfg / "results.csv")
        (fac_cfg / "fit.json").write_text(json.dumps(fits, indent=2),
                                          encoding="utf-8")

    print(f"\n[{tag}] Done at {datetime.now().isoformat()}", file=sys.stderr)


if __name__ == "__main__":
    main()
