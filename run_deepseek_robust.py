"""Robust, self-healing driver for a DeepSeek full sweep.

The plain detached runner occasionally dies from a Windows handle error when its
launching context is torn down. This driver keeps a single long-lived process
that repeatedly calls the resumable sweep for one model until every
(context, regime) pair has a parsed result, catching and retrying on any
exception. Because the underlying runner is resumable, each pass only re-fires
the still-missing prompts.

Usage:
    python run_deepseek_robust.py --model DeepSeek-V4-Flash --parallel 100 \
        --run-tag flash --api-key <KEY>
"""
from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
import time
from pathlib import Path

import revealed_preferences as rp
import run_factorial_sweep as fs
import run_threshold_sweep as ts

for _name in ("httpx", "openai", "openai._base_client", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

FULL_COST_FUNCTIONS = [(100, 1), (10, 1), (5, 1), (1, 1), (1, 5), (1, 10), (1, 100)]
FACTORIAL_ROOT = rp.OUTPUT_ROOT / "2026-07-01_factorial"
THRESHOLD_ROOT = rp.OUTPUT_ROOT / "2026-07-08_threshold"
LEVELS = ["none", "high", "max"]


def uniq_parsed(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("parsed_decision") not in (None, "") or \
               r.get("parsed_probability") not in (None, ""):
                done.add((r["context_id"], r["regime"]))
    return done


def missing_count(model: str) -> int:
    rp.COST_FUNCTIONS = FULL_COST_FUNCTIONS
    util_items = fs.build_factorial_items(fs.THEIR_CSV)
    thr_items = ts.build_threshold_items(ts.THEIR_CSV)
    total = 0
    for eff in LEVELS:
        label = f"{model}_re-{eff}"
        fdone = uniq_parsed(FACTORIAL_ROOT / label / "results.csv")
        tdone = uniq_parsed(THRESHOLD_ROOT / label / "results.csv")
        total += sum(1 for it in util_items
                     if (it.context_id, it.regime) not in fdone)
        total += sum(1 for it in thr_items
                     if (it.context_id, it.regime) not in tdone)
    return total


def run_single_chunk(model, eff, kind, parallel, api_key, timeout, progress_path):
    """Run exactly one (level, kind) config chunk, then return. Meant to be
    invoked as a short-lived child process so a wedged thread pool dies with the
    process instead of accumulating across the whole run."""
    rp.COST_FUNCTIONS = FULL_COST_FUNCTIONS
    items = (fs.build_factorial_items(fs.THEIR_CSV) if kind == "factorial"
             else ts.build_threshold_items(ts.THEIR_CSV))
    root = FACTORIAL_ROOT if kind == "factorial" else THRESHOLD_ROOT
    cfg = root / f"{model}_re-{eff}"
    cfg.mkdir(parents=True, exist_ok=True)
    rpath = ts.run_config_resumable(
        items, cfg, model, rp.AZURE_RESPONSES_ENDPOINT, eff,
        parallel, api_key, Path(progress_path), api_mode="chat", timeout=timeout)
    ts._sort_results(rpath)


# A single chunk should finish well within this wall-clock cap on a healthy
# endpoint; if it doesn't, the child is killed and the next pass retries the
# still-missing items (the runner is resumable).
CHUNK_TIMEOUT = 900.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--parallel", type=int, default=100)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max-passes", type=int, default=60)
    # Internal: run one chunk in this process then exit (used by the orchestrator).
    ap.add_argument("--single", nargs=2, metavar=("EFF", "KIND"), default=None)
    args = ap.parse_args()

    rp.COST_FUNCTIONS = FULL_COST_FUNCTIONS
    model = args.model
    tag = args.run_tag or model
    FACTORIAL_ROOT.mkdir(parents=True, exist_ok=True)
    THRESHOLD_ROOT.mkdir(parents=True, exist_ok=True)
    progress_path = FACTORIAL_ROOT / f"progress_robust_{tag}.log"

    if args.single is not None:
        eff, kind = args.single
        for _name in ("httpx", "openai", "openai._base_client", "httpcore"):
            logging.getLogger(_name).setLevel(logging.WARNING)
        logging.disable(logging.INFO)
        run_single_chunk(model, eff, kind, args.parallel, args.api_key,
                         args.timeout, progress_path)
        return

    logf = open(FACTORIAL_ROOT / f"run_robust_{tag}.log", "a", buffering=1,
                encoding="utf-8", errors="replace")
    sys.stdout = logf
    sys.stderr = logf
    logging.disable(logging.INFO)

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} [robust {tag}] {msg}"
        print(line, file=sys.stderr, flush=True)
        with open(progress_path, "a", encoding="utf-8") as pf:
            pf.write(line + "\n")

    log(f"start model={model} missing={missing_count(model)}")

    for pass_i in range(1, args.max_passes + 1):
        miss = missing_count(model)
        log(f"pass {pass_i}: {miss} missing")
        if miss == 0:
            log("COMPLETE")
            break
        for eff in LEVELS:
            for kind in ("factorial", "threshold"):
                cmd = [sys.executable, __file__, "--model", model,
                       "--parallel", str(args.parallel), "--timeout",
                       str(args.timeout), "--run-tag", tag,
                       "--single", eff, kind]
                if args.api_key:
                    cmd += ["--api-key", args.api_key]
                try:
                    proc = subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
                    proc.communicate(timeout=CHUNK_TIMEOUT)
                except subprocess.TimeoutExpired:
                    log(f"pass {pass_i} {eff}/{kind} TIMEOUT after "
                        f"{CHUNK_TIMEOUT:.0f}s -> killing child, will retry")
                    proc.kill()
                    try:
                        proc.communicate(timeout=30)
                    except Exception:  # noqa: BLE001
                        pass
                except Exception as e:  # noqa: BLE001 - self-heal, keep going
                    log(f"pass {pass_i} {eff}/{kind} ERROR: {type(e).__name__}: {e}")
                    time.sleep(5)

    # Regenerate fit.json per level from the completed factorial results.
    import json
    for eff in LEVELS:
        cfg = FACTORIAL_ROOT / f"{model}_re-{eff}"
        if (cfg / "results.csv").exists():
            fits = rp.fit_from_results(cfg / "results.csv")
            (cfg / "fit.json").write_text(json.dumps(fits, indent=2),
                                          encoding="utf-8")
    log(f"done. final missing={missing_count(model)}")


if __name__ == "__main__":
    main()
