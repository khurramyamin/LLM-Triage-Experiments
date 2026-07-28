"""Probability-threshold decision elicitation (counterpart to the utility sweep).

The revealed-preference factorial sweep (``run_factorial_sweep.py``) prompted each
model with an explicit cost function (utility) and asked it to minimise expected
cost. This script instead prompts the model with the Bayes-optimal probability
threshold p* = c_FP / (c_FP + c_FN) implied by each of those same five cost
functions, and asks the model to admit to the emergency department when its own
probability that the patient needs emergency care reaches that threshold.

For every one of the 1,248 factorial variants we elicit the 5 threshold-prompted
admit decisions only (5 prompts x 1,248 variants = 6,240 prompts per config). We
deliberately do NOT re-elicit beliefs or the baseline decision: the independently
elicited beliefs from the factorial run are reused for analysis, so the two
prompting styles (utility vs threshold) are compared on the identical beliefs.

The five thresholds map to the five cost functions as follows
(p* = c_FP / (c_FP + c_FN)):

    (c_FP, c_FN)   FN/FP    p* (shown, 2 dp)   stance
    (10, 1)        0.1      0.91               efficiency-first (refer only if near-certain)
    ( 5, 1)        0.2      0.83               efficiency-leaning
    ( 1, 1)        1.0      0.50               neutral
    ( 1, 5)        5.0      0.17               safety-leaning
    ( 1,10)       10.0      0.09               safety-first (refer even at low probability)

Output layout mirrors the factorial sweep so the analysis code can pair a
threshold regime (``decision_t_fp{c_fp}_fn{c_fn}``) with its utility counterpart
(``decision_u_fp{c_fp}_fn{c_fn}``) and with the reused beliefs.

Usage (identical CLI to run_factorial_sweep.py):
    python run_threshold_sweep.py \
        --configs gpt-5-mini:minimal,gpt-5-mini:medium,gpt-5-mini:high \
        --parallel 100 --output-dir revealed_preferences/<ts>_threshold
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

# Silence per-request httpx/openai INFO logging (see run_factorial_sweep.py).
for _name in ("httpx", "openai", "openai._base_client", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

THEIR_CSV = fs.THEIR_CSV
FORMAT_MARKER = fs.FORMAT_MARKER


def build_threshold_items(csv_path: Path) -> list[fs._Item]:
    """One threshold-decision prompt per (variant, cost function). 5 per variant."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    items: list[fs._Item] = []
    for r in rows:
        case_id = r["case_id"]
        variant = r["variant_code"]
        context = fs.extract_context(r["prompt_text"])
        gold_triage = r["gold_triage"]
        needs_er = 1 if "D" in gold_triage else 0
        prompt_type = 1 if case_id[0] in ("E", "M") else 2
        base = dict(
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

        def make(regime, question, c_fp, c_fn, p_star):
            d = dict(base)
            d.update(regime=regime, c_fp=c_fp, c_fn=c_fn, p_star=p_star,
                     prompt_text=f"{context}\n\n{question}")
            return fs._Item(d)

        for c_fp, c_fn in rp.COST_FUNCTIONS:
            p_star = round(rp.admit_threshold(c_fp, c_fn), 2)
            regime = f"decision_{rp.threshold_label(c_fp, c_fn)}"
            q = rp.build_decision_threshold_question(p_star)
            items.append(make(regime, q, c_fp, c_fn, p_star))
    return items


def _sort_results(path: Path) -> None:
    """Rewrite results.csv sorted by (context_id, regime) after an append run."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["context_id"], r["regime"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fs.FACTORIAL_FIELDS)
        w.writeheader()
        w.writerows(rows)


def load_done_keys(results_path: Path) -> set:
    """Return the set of (context_id, regime) already present in a results.csv,
    so a resumed run can skip them. A row counts as done if it has either a
    parsed decision (decision regimes) or a parsed probability (belief regime);
    only empty/failed rows are retried."""
    done = set()
    if not results_path.exists():
        return done
    try:
        with open(results_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("parsed_decision") not in (None, "") or \
                   r.get("parsed_probability") not in (None, ""):
                    done.add((r["context_id"], r["regime"]))
    except Exception:
        pass
    return done


def run_config_resumable(items, cfg: Path, model: str, endpoint: str, eff: str,
                         parallel: int, api_key, progress_path: Path,
                         api_mode=True, timeout: float = 90.0):
    """Run one config, appending each result to results.csv as it completes and
    skipping items already recorded. Crash-safe and resumable.

    api_mode: True -> Responses API; "chat" -> chat.completions;
              "anthropic" -> Anthropic Messages API.
    timeout:   per-request client timeout in seconds.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results_path = cfg / "results.csv"
    done = load_done_keys(results_path)
    todo = [it for it in items if (it.context_id, it.regime) not in done]
    label = f"{model}_re-{eff}"

    def log(msg: str):
        line = f"{datetime.now().strftime('%H:%M:%S')} [{label}] {msg}"
        print(line, file=sys.stderr, flush=True)
        with open(progress_path, "a", encoding="utf-8") as pf:
            pf.write(line + "\n")

    if not todo:
        log(f"already complete ({len(done)} rows), skipping")
        return results_path
    log(f"resuming: {len(done)} done, {len(todo)} to do")

    if api_mode == "anthropic":
        client = rp._make_anthropic_client(api_key, timeout=timeout)
    else:
        client = rp._make_client(endpoint, api_key, timeout=timeout)
    new_file = not results_path.exists()
    # line-buffered append; flush after every row so a kill loses at most 1 row
    out = open(results_path, "a", newline="", encoding="utf-8", buffering=1)
    writer = csv.DictWriter(out, fieldnames=fs.FACTORIAL_FIELDS)
    if new_file:
        writer.writeheader()

    completed = 0
    hard_deadline = max(timeout * 1.5, timeout + 60.0)
    try:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futs = {pool.submit(rp._run_one, it, client, model, 0.0, eff,
                                api_mode, hard_deadline): it
                    for it in todo}
            for fut in as_completed(futs):
                it = futs[fut]
                row = fut.result()
                for k in fs.EXTRA_FIELDS:
                    row[k] = getattr(it, k)
                writer.writerow(row)
                out.flush()
                completed += 1
                if completed % 250 == 0 or completed == len(todo):
                    log(f"{completed}/{len(todo)} (this run)")
    finally:
        out.close()
    return results_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gpt-5-mini")
    ap.add_argument("--reasonings", default="minimal,medium,high")
    ap.add_argument("--configs", default=None,
                    help="Explicit comma-separated model:effort pairs; overrides "
                         "--models/--reasonings.")
    ap.add_argument("--endpoint", default=rp.AZURE_RESPONSES_ENDPOINT)
    ap.add_argument("--parallel", type=int, default=100)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--limit-cases", type=int, default=None,
                    help="Smoke test: only the first N case_ids.")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build prompts and print a sample without any API calls.")
    args = ap.parse_args()

    if args.output_dir:
        root = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y-%m-%d-%H%M")
        root = rp.OUTPUT_ROOT / f"{ts}_threshold"
    root.mkdir(parents=True, exist_ok=True)
    tag = args.run_tag or "run"

    items = build_threshold_items(THEIR_CSV)
    if args.limit_cases:
        keep = set(sorted({it.vignette_id for it in items})[:args.limit_cases])
        items = [it for it in items if it.vignette_id in keep]
    n_ctx = len({it.context_id for it in items})

    if args.dry_run:
        print(f"[dry-run] Built {len(items)} prompts across {n_ctx} variants "
              f"({len(items)//max(n_ctx,1)} regimes each).")
        seen = set()
        for it in items:
            if it.regime in seen:
                continue
            seen.add(it.regime)
            print("\n" + "=" * 72)
            print(f"regime={it.regime}  c_fp={it.c_fp} c_fn={it.c_fn} "
                  f"p_star={it.p_star}")
            print("-" * 72)
            # print only the question part (after the shared context block)
            print(it.prompt_text.split("\n\n", 1)[1])
        return

    # Redirect stdout/stderr to a file (Windows console-handle safety).
    _logf = open(root / f"run_{tag}.log", "a", buffering=1,
                 encoding="utf-8", errors="replace")
    sys.stdout = _logf
    sys.stderr = _logf
    logging.disable(logging.INFO)

    progress_path = root / f"progress_{tag}.log"
    print(f"Built {len(items)} prompts across {n_ctx} variants "
          f"({len(items)//max(n_ctx,1)} regimes each).", file=sys.stderr)

    if args.configs:
        config_pairs = []
        for pair in args.configs.split(","):
            pair = pair.strip()
            if not pair:
                continue
            model, eff = pair.split(":")
            config_pairs.append((model.strip(), eff.strip()))
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        reasonings = [r.strip() for r in args.reasonings.split(",") if r.strip()]
        config_pairs = [(m, e) for m in models for e in reasonings]

    print(f"Threshold sweep root: {root}", file=sys.stderr)
    print(f"Configs ({len(config_pairs)}): {config_pairs} = "
          f"{len(config_pairs)*len(items)} total calls.", file=sys.stderr)

    all_fits, all_cov = {}, {}
    for model, eff in config_pairs:
        label = f"{model}_re-{eff}"
        cfg = root / label
        cfg.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {label} ({len(items)} prompts) ===", file=sys.stderr)
        t0 = time.time()
        rpath = run_config_resumable(items, cfg, model, args.endpoint, eff,
                                     args.parallel, args.api_key, progress_path)
        # ensure rows are sorted deterministically after a resumed/append run
        _sort_results(rpath)
        dur = time.time() - t0
        cov = rp.coverage_from_results(rpath)
        all_cov[label] = cov
        (cfg / "settings.json").write_text(json.dumps({
            "model": model, "reasoning_effort": eff,
            "n_items": len(items), "n_contexts": n_ctx,
            "duration_sec": round(dur, 1), "coverage": cov,
            "threshold": True,
        }, indent=2), encoding="utf-8")
        print(f"  done in {dur:.0f}s", file=sys.stderr)

    print(f"\nDone. Threshold sweep at {root}", file=sys.stderr)


if __name__ == "__main__":
    main()
