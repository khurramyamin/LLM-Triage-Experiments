"""Full belief, utility-decision, and threshold sweeps for Claude models.

Runs Claude Fable 5 or Claude Sonnet 5 at low, medium, and high effort through
Anthropic's Messages API. Results append safely to the existing factorial and
threshold sweep roots and can be resumed after interruption.

Usage:
    set ANTHROPIC_API_KEY=...
    python run_claude_full.py --family fable --parallel 200
    python run_claude_full.py --family sonnet --parallel 200
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import revealed_preferences as rp
import run_factorial_sweep as fs
import run_threshold_sweep as ts


FULL_COST_FUNCTIONS = [
    (100, 1), (10, 1), (5, 1), (1, 1), (1, 5), (1, 10), (1, 100)
]
REASONING_LEVELS = ["low", "medium", "high"]
MODEL_SPECS = {
    "fable": ("claude-fable-5", "Claude-Fable-5"),
    "sonnet": ("claude-sonnet-5", "Claude-Sonnet-5"),
}
FACTORIAL_ROOT = rp.OUTPUT_ROOT / "2026-07-01_factorial"
THRESHOLD_ROOT = rp.OUTPUT_ROOT / "2026-07-08_threshold"
CLIENT_TIMEOUT = 600.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=MODEL_SPECS)
    parser.add_argument("--levels", default=",".join(REASONING_LEVELS))
    parser.add_argument("--parallel", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=CLIENT_TIMEOUT)
    parser.add_argument("--run-tag")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY must be set in the process environment.")

    api_model, config_model = MODEL_SPECS[args.family]
    levels = [level.strip() for level in args.levels.split(",") if level.strip()]
    invalid = set(levels) - set(REASONING_LEVELS)
    if invalid:
        raise ValueError(f"Unsupported effort levels: {sorted(invalid)}")

    rp.COST_FUNCTIONS = FULL_COST_FUNCTIONS
    factorial_items = fs.build_factorial_items(fs.THEIR_CSV)
    threshold_items = ts.build_threshold_items(ts.THEIR_CSV)
    n_contexts = len({item.context_id for item in factorial_items})
    tag = args.run_tag or args.family

    FACTORIAL_ROOT.mkdir(parents=True, exist_ok=True)
    THRESHOLD_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = open(
        FACTORIAL_ROOT / f"run_claude_{tag}.log",
        "a",
        buffering=1,
        encoding="utf-8",
        errors="replace",
    )
    sys.stdout = log_file
    sys.stderr = log_file
    logging.disable(logging.INFO)
    progress_path = FACTORIAL_ROOT / f"progress_claude_{tag}.log"

    print(
        f"[{tag}] start={datetime.now().isoformat()} api_model={api_model} "
        f"levels={levels} parallel={args.parallel}",
        file=sys.stderr,
    )
    print(
        f"[{tag}] {len(factorial_items)} factorial + "
        f"{len(threshold_items)} threshold prompts per level",
        file=sys.stderr,
    )

    for effort in levels:
        label = f"{config_model}_re-{effort}"
        for kind, items, root in (
            ("factorial", factorial_items, FACTORIAL_ROOT),
            ("threshold", threshold_items, THRESHOLD_ROOT),
        ):
            config_dir = root / label
            config_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            print(
                f"\n=== [{tag}] {label} :: {kind} ({len(items)} prompts) ===",
                file=sys.stderr,
            )
            results_path = ts.run_config_resumable(
                items,
                config_dir,
                api_model,
                "",
                effort,
                args.parallel,
                api_key,
                progress_path,
                api_mode="anthropic",
                timeout=args.timeout,
            )
            ts._sort_results(results_path)
            coverage = rp.coverage_from_results(results_path)
            (config_dir / "settings.json").write_text(
                json.dumps(
                    {
                        "model": api_model,
                        "display_model": config_model,
                        "reasoning_effort": effort,
                        "reasoning_method": "output_config.effort",
                        "thinking": (
                            "adaptive"
                            if api_model.startswith("claude-sonnet-5")
                            else "always-on adaptive"
                        ),
                        "max_tokens": 4096,
                        "n_items": len(items),
                        "n_contexts": n_contexts,
                        "duration_sec": round(time.time() - started, 1),
                        "coverage": coverage,
                        "kind": kind,
                        "full_utilities": True,
                        "api": "anthropic.messages",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"[{tag}] {label} :: {kind} done in "
                f"{time.time() - started:.0f}s",
                file=sys.stderr,
            )

        factorial_dir = FACTORIAL_ROOT / label
        fits = rp.fit_from_results(factorial_dir / "results.csv")
        (factorial_dir / "fit.json").write_text(
            json.dumps(fits, indent=2), encoding="utf-8"
        )

    print(f"\n[{tag}] done={datetime.now().isoformat()}", file=sys.stderr)


if __name__ == "__main__":
    main()
