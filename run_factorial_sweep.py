"""Factorial (16-condition) belief + decision elicitation.

Extends the revealed-preference sweep from 78 neutral contexts to the full
1,248 = 78 case_ids x 16 factorial conditions used by the Nature Medicine paper
(race x gender x anchoring x access-barrier).  For every one of the 1,248
variants we elicit:

  * an independent belief  P(need ER),
  * a baseline (no-utility) admit decision,
  * 5 utility-prompted admit decisions (cost ratios 10:1 ... 1:10),

i.e. 7 prompts x 1,248 variants = 8,736 prompts per (model x reasoning) config.

The clinical context for each variant is taken verbatim from the paper's own
``prompt_text`` (so the exact race/gender/anchor/barrier wording matches what
ChatGPT Health saw); we simply swap their A/B/C/D triage instruction for our
belief / decision / utility questions.  We also carry their own ``llm_triage``
into the results so decisions can be paired variant-by-variant.

Output layout mirrors ``--sweep`` so ``analyze_sweep.py`` and
``original_paper_comparison.py`` work unchanged (context_id = case_id__variant).

Usage:
    python run_factorial_sweep.py --models gpt-5-mini --reasonings minimal,medium,high \
        --parallel 100 --output-dir revealed_preferences/<ts>_factorial
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

# Silence the per-request httpx/openai INFO logging. With thousands of calls
# piped/backgrounded, those log writes hit an invalid stderr handle on Windows
# (OSError WinError 6) and can kill the run. We track progress via a file.
for _name in ("httpx", "openai", "openai._base_client", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

THEIR_CSV = Path("original_paper_data/extracted/"
                 "ashwinra-code-gpt-health-eval-61bce8a/data/DataExpanded_FINAL.csv")
FORMAT_MARKER = "Please answer in exactly this format:"

# extra columns appended to the standard results schema
EXTRA_FIELDS = ["variant_code", "race", "gender", "has_anchor", "has_barrier",
                "their_triage"]
FACTORIAL_FIELDS = rp.RESULTS_FIELDS + EXTRA_FIELDS


def extract_context(prompt_text: str) -> str:
    """Everything before the paper's A/B/C/D format instruction."""
    return prompt_text.split(FORMAT_MARKER)[0].strip()


def build_factorial_items(csv_path: Path) -> list[dict]:
    """Return list of dicts (one per prompt) carrying PromptItem attrs + extras.

    We use plain dicts (not PromptItem) so we can attach the factorial extras;
    ``_run_one`` only needs attribute access, so we wrap each in a small object.
    """
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    items: list[_Item] = []
    for r in rows:
        case_id = r["case_id"]
        variant = r["variant_code"]
        context = extract_context(r["prompt_text"])
        gold_triage = r["gold_triage"]
        needs_er = 1 if "D" in gold_triage else 0
        prompt_type = 1 if case_id[0] in ("E", "M") else 2  # E/MH = with labs
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

        def make(regime, question, c_fp=None, c_fn=None, p_star=None):
            d = dict(base)
            d.update(regime=regime, c_fp=c_fp, c_fn=c_fn, p_star=p_star,
                     prompt_text=f"{context}\n\n{question}")
            return _Item(d)

        items.append(make("belief", rp.BELIEF_QUESTION))
        items.append(make("decision_baseline", rp.DECISION_BASELINE_QUESTION))
        for c_fp, c_fn in rp.COST_FUNCTIONS:
            regime = f"decision_{rp.cost_label(c_fp, c_fn)}"
            q = rp.build_decision_utility_question(c_fp, c_fn)
            items.append(make(regime, q, c_fp=c_fp, c_fn=c_fn,
                              p_star=rp.admit_threshold(c_fp, c_fn)))
    return items


class _Item:
    """Attribute view over a dict so rp._run_one can read fields via getattr."""
    __slots__ = ("_d",)

    def __init__(self, d: dict):
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError as e:
            raise AttributeError(name) from e


def run_config_items(items, out_dir: Path, model: str, base_url: str,
                     reasoning_effort: str, parallel: int, api_key,
                     progress_path: Path = None):
    """Like rp.run_items but preserves the factorial extra columns."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    client = rp._make_client(base_url, api_key)
    label = f"{model}_re-{reasoning_effort}"

    def log(msg: str):
        line = f"{datetime.now().strftime('%H:%M:%S')} [{label}] {msg}"
        print(line, file=sys.stderr, flush=True)
        if progress_path is not None:
            with open(progress_path, "a", encoding="utf-8") as pf:
                pf.write(line + "\n")

    results = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futs = {pool.submit(rp._run_one, it, client, model, 0.0,
                            reasoning_effort, True): it for it in items}
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            it = futs[fut]
            for k in EXTRA_FIELDS:
                row[k] = getattr(it, k)
            results.append(row)
            done += 1
            if done % 250 == 0 or done == len(items):
                log(f"{done}/{len(items)}")
    results.sort(key=lambda r: (r["context_id"], r["regime"]))
    path = out_dir / "results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FACTORIAL_FIELDS)
        w.writeheader()
        w.writerows(results)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gpt-5-mini")
    ap.add_argument("--reasonings", default="minimal,medium,high")
    ap.add_argument("--configs", default=None,
                    help="Explicit comma-separated model:effort pairs; overrides "
                         "--models/--reasonings so families with different reasoning "
                         "levels can queue in one run.")
    ap.add_argument("--endpoint", default=rp.AZURE_RESPONSES_ENDPOINT)
    ap.add_argument("--parallel", type=int, default=100)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--limit-cases", type=int, default=None,
                    help="Smoke test: only the first N case_ids.")
    ap.add_argument("--run-tag", default=None,
                    help="Suffix for run/progress log files so concurrent "
                         "processes writing to the same output dir don't collide.")
    args = ap.parse_args()

    # Resolve output root first so we can redirect all further output to a file.
    if args.output_dir:
        root = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y-%m-%d-%H%M")
        root = rp.OUTPUT_ROOT / f"{ts}_factorial"
    root.mkdir(parents=True, exist_ok=True)
    tag = args.run_tag or "run"
    progress_path = root / f"progress_{tag}.log"

    # CRITICAL: reopen stdout/stderr to a file. When the shell backgrounds this
    # command the inherited console handle becomes invalid, and the next write
    # to it raises OSError [WinError 6] and kills the process. Writing to our own
    # file makes the run independent of the console handle.
    _logf = open(root / f"run_{tag}.log", "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = _logf
    sys.stderr = _logf
    logging.disable(logging.INFO)  # belt-and-suspenders: no INFO log writes

    items = build_factorial_items(THEIR_CSV)
    if args.limit_cases:
        keep = sorted({it.vignette_id for it in items})[:args.limit_cases]
        keep = set(keep)
        items = [it for it in items if it.vignette_id in keep]
    n_ctx = len({it.context_id for it in items})
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

    progress_path = root / f"progress_{tag}.log"
    print(f"Factorial sweep root: {root}", file=sys.stderr)
    print(f"Configs ({len(config_pairs)}): {config_pairs} = "
          f"{len(config_pairs)*len(items)} total calls.", file=sys.stderr)

    all_fits, all_cov = {}, {}
    for model, eff in config_pairs:
        label = f"{model}_re-{eff}"
        cfg = root / label
        cfg.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {label} ({len(items)} prompts) ===", file=sys.stderr)
        t0 = time.time()
        rpath = run_config_items(items, cfg, model, args.endpoint, eff,
                                 args.parallel, args.api_key, progress_path)
        dur = time.time() - t0
        fits = rp.fit_from_results(rpath)
        cov = rp.coverage_from_results(rpath)
        all_fits[label] = fits
        all_cov[label] = cov
        (cfg / "fit.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
        (cfg / "settings.json").write_text(json.dumps({
            "model": model, "reasoning_effort": eff,
            "n_items": len(items), "n_contexts": n_ctx,
            "duration_sec": round(dur, 1), "coverage": cov,
            "factorial": True,
        }, indent=2), encoding="utf-8")
        print(f"  done in {dur:.0f}s", file=sys.stderr)

    rp.write_sweep_summary(root, all_fits, all_cov)
    print(f"\nDone. Sweep at {root}", file=sys.stderr)


if __name__ == "__main__":
    main()
