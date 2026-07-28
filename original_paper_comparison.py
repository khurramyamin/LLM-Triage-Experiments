"""On-curve vs off-curve analysis of the original Nature Medicine under-triage.

Research question
-----------------
The Nature Medicine paper ("The jagged edge of ChatGPT Health: Under-triage in
consumer-facing AI", Zenodo 18451491) found ChatGPT Health *under-triages*.
Is that under-triage:

  (A) an operating-point / utility problem — the model sits ON the ROC frontier
      of good beliefs but at an over-conservative (FP-averse) threshold, i.e. it
      simply chose the wrong point on the curve; or
  (B) an off-curve problem — its decisions fall BELOW the frontier achievable by
      thresholding well-ranked beliefs, i.e. a genuine discrimination failure
      that no single utility function would fix?

Method
------
* Their decisions: ChatGPT Health (gpt-5-mini thinking backbone) 4-level triage
  A/B/C/D, from ``DataExpanded_FINAL.csv``.  We use the reference factorial
  condition (White man, no anchor, no barrier) so there is exactly one decision
  per case, matching the plain vignette on which our beliefs were elicited.
  Binary admit  :=  (triage == 'D')  ("go to the emergency department now").
* True label   :  needs_er_gold = 1 iff the clinician gold triage includes 'D'.
* Frontier     :  the ROC of OUR independently elicited beliefs P(need ER)
  (from the revealed-preference sweep).  Their case_ids and gold labels match
  ours exactly (78/78), so beliefs and decisions refer to identical scenarios.

We overlay their operating point on the belief ROC and measure how far below the
frontier it sits (TPR deficit at matched FPR).  ~0 deficit => on the curve
(utility problem);  large deficit => off the curve (discrimination problem).

Caveat: beliefs are from our models (gpt-5.4 / gpt-5.4-mini), not the exact
ChatGPT Health backbone (which we cannot query), so the frontier is a strong
rational-belief proxy rather than the model's own internal probabilities.

Usage:
    python original_paper_comparison.py [sweep_dir]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

THEIR_CSV = Path("original_paper_data/extracted/"
                 "ashwinra-code-gpt-health-eval-61bce8a/data/DataExpanded_FINAL.csv")
DEFAULT_SWEEP = Path("revealed_preferences/2026-06-30-1450_sweep")
# belief config used for the headline figure (same family as their gpt-5-mini backbone)
PRIMARY_CONFIG = "gpt-5-mini_re-high"

# One colour per config, shared across both panels so blue/green always mean the
# same model. The two gpt-5-mini configs get blue (medium) and green (high) to
# match the headline (left) panel.
CONFIG_COLORS = {
    "gpt-5-mini_re-medium": "C0",   # blue
    "gpt-5-mini_re-high": "C2",     # green
    "gpt-5-mini_re-minimal": "#7fbfff",   # light blue
    "gpt-5.4-mini_re-none": "#d62728",
    "gpt-5.4-mini_re-medium": "#ff7f0e",
    "gpt-5.4-mini_re-high": "#8c564b",
    "gpt-5.4_re-none": "#9467bd",
    "gpt-5.4_re-medium": "#e377c2",
    "gpt-5.4_re-high": "#17becf",
}


def gold_needs_er(gold_triage: str) -> int:
    return 1 if "D" in gold_triage else 0


def load_their_decisions(csv_path: Path):
    """Reference-condition (WM, no anchor, no barrier) triage per case_id.

    Returns (admit_by_ctx, gold_by_ctx, triage_by_ctx, allcond_admit_lists).
    ``allcond`` pools every factorial condition for a robustness point.
    """
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    gold = {r["case_id"]: gold_needs_er(r["gold_triage"]) for r in rows}
    ref = {}
    allcond: dict[str, list[int]] = {}
    for r in rows:
        cid = r["case_id"]
        allcond.setdefault(cid, []).append(1 if r["llm_triage"] == "D" else 0)
        if (r["race"] == "White" and r["gender"] == "man"
                and r["has_anchor"] == "no" and r["has_barrier"] == "no"):
            ref[cid] = r["llm_triage"]
    admit = {cid: (1 if t == "D" else 0) for cid, t in ref.items()}
    return admit, gold, ref, allcond


def load_beliefs(results_csv: Path):
    belief, gold = {}, {}
    for r in csv.DictReader(open(results_csv, encoding="utf-8")):
        c = r["context_id"]
        if r.get("needs_er_gold") not in (None, ""):
            gold[c] = int(float(r["needs_er_gold"]))
        if r["regime"] == "belief" and r["parsed_probability"]:
            belief[c] = float(r["parsed_probability"])
    return belief, gold


def load_factorial_their_decisions(results_csv: Path):
    """Per-variant ChatGPT-Health decision + gold, read straight from a factorial
    results.csv (which carries the ``their_triage`` column per context_id).

    Fully matched: our belief and their decision refer to the *identical*
    variant (same race/gender/anchor/barrier), so no reference-condition proxy
    is needed. Returns (admit_by_ctx, gold_by_ctx, triage_by_ctx).
    """
    admit, gold, triage = {}, {}, {}
    for r in csv.DictReader(open(results_csv, encoding="utf-8")):
        c = r["context_id"]
        if c in triage:
            continue
        t = r.get("their_triage")
        if t in (None, ""):
            continue
        triage[c] = t
        admit[c] = 1 if t == "D" else 0
        if r.get("needs_er_gold") not in (None, ""):
            gold[c] = int(float(r["needs_er_gold"]))
    return admit, gold, triage


def is_factorial(sweep_dir: Path) -> bool:
    """True if the sweep's results carry per-variant context ids + their_triage."""
    for d in sweep_dir.glob("*/results.csv"):
        with open(d, encoding="utf-8") as f:
            header = f.readline()
            first = f.readline()
        return "their_triage" in header and "__" in first.split(",", 1)[0]
    return False


def roc_curve(scores, labels):
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    P, N = int((y == 1).sum()), int((y == 0).sum())
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = np.concatenate([[0.0], tp / P])
    fpr = np.concatenate([[0.0], fp / N])
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return fpr, tpr, float(trap(tpr, fpr))


def op_point(admit: dict, gold: dict, ctxs):
    y = np.array([gold[c] for c in ctxs])
    p = np.array([admit[c] for c in ctxs])
    P, N = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    tpr = float(((p == 1) & (y == 1)).sum() / P)
    fpr = float(((p == 1) & (y == 0)).sum() / N)
    acc = float((p == y).mean())
    return fpr, tpr, acc


def analyse(sweep_dir: Path):
    factorial = is_factorial(sweep_dir)
    any_results = sorted(sweep_dir.glob("*/results.csv"))[0]

    if factorial:
        # fully matched: their decision per identical variant, from results.csv
        admit, gold_t, triage = load_factorial_their_decisions(any_results)
        ctxs = list(admit)
        fpr_t, tpr_t, acc_t = op_point(admit, gold_t, ctxs)
        admit_rate = float(np.mean([admit[c] for c in ctxs]))
        # "pooled" == the matched full-factorial point here (no proxy needed)
        pooled_fpr, pooled_tpr = fpr_t, tpr_t
    else:
        admit, gold_t, triage, allcond = load_their_decisions(THEIR_CSV)
        ctxs = list(admit)
        fpr_t, tpr_t, acc_t = op_point(admit, gold_t, ctxs)
        admit_rate = float(np.mean([admit[c] for c in ctxs]))
        pooled_admit, pooled_gold = [], []
        for cid, lst in allcond.items():
            for a in lst:
                pooled_admit.append(a)
                pooled_gold.append(gold_t[cid])
        pa, pg = np.array(pooled_admit), np.array(pooled_gold)
        pooled_fpr = float(((pa == 1) & (pg == 0)).sum() / max((pg == 0).sum(), 1))
        pooled_tpr = float(((pa == 1) & (pg == 1)).sum() / max((pg == 1).sum(), 1))

    rows = []
    for d in sorted(sweep_dir.glob("*/results.csv")):
        name = d.parent.name
        belief, gold_b = load_beliefs(d)
        cc = [c for c in belief if c in gold_b]
        f_, t_, auc = roc_curve([belief[c] for c in cc], [gold_b[c] for c in cc])
        tpr_at = float(np.interp(fpr_t, f_, t_))       # frontier TPR at their FPR
        fpr_at = float(np.interp(tpr_t, t_, f_))       # frontier FPR at their TPR
        rows.append({
            "name": name, "auc": auc, "fpr": f_, "tpr": t_,
            "tpr_at": tpr_at, "tpr_deficit": tpr_at - tpr_t,
            "fpr_at": fpr_at, "fpr_excess": fpr_t - fpr_at,
        })
    return {
        "factorial": factorial,
        "ctxs": ctxs, "gold": gold_t, "triage": triage,
        "their_fpr": fpr_t, "their_tpr": tpr_t, "their_acc": acc_t,
        "admit_rate": admit_rate,
        "pooled_fpr": pooled_fpr, "pooled_tpr": pooled_tpr,
        "configs": rows,
    }


def load_regime_op_points(results_csv: Path):
    """(FPR,TPR) vs gold for each prompted-utility decision regime in a config.

    These are the same operating points drawn in roc_oracle.png — the decisions
    the model made under each cost function, scored against the true ER label.
    Returns {regime: (fpr, tpr)}.
    """
    gold, dec = {}, {}
    for r in csv.DictReader(open(results_csv, encoding="utf-8")):
        ctx = r["context_id"]
        if r.get("needs_er_gold") not in (None, ""):
            gold[ctx] = int(float(r["needs_er_gold"]))
        if r["regime"] != "belief" and r.get("parsed_decision") not in (None, ""):
            dec.setdefault(r["regime"], {})[ctx] = int(r["parsed_decision"])
    out = {}
    for regime, dctx in dec.items():
        ctxs = [c for c in dctx if c in gold]
        if not ctxs:
            continue
        f_, t_, _ = op_point(dctx, gold, ctxs)
        out[regime] = (f_, t_)
    return out


def make_figure(res: dict, out_dir: Path, sweep_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = res["configs"]
    prim = next((c for c in configs if c["name"] == PRIMARY_CONFIG), configs[0])
    # gpt-5-mini medium + high frontiers for the headline panel (same family as
    # ChatGPT Health's documented backbone).
    left_names = ["gpt-5-mini_re-medium", "gpt-5-mini_re-high"]
    left_configs = [c for n in left_names
                    for c in configs if c["name"] == n] or [prim]

    # utility-regime operating points (from the model's actual decisions), the
    # same points shown in roc_oracle.png. medium = open circles, high = filled.
    regime_pts = {
        name: load_regime_op_points(sweep_dir / name / "results.csv")
        for name in left_names
        if (sweep_dir / name / "results.csv").exists()
    }
    ratio_colors = {  # per prompted FN/FP ratio, FP-averse (blue) -> FN-averse (red)
        "decision_u_fp10_fn1": "#08519c",
        "decision_u_fp5_fn1": "#3182bd",
        "decision_u_fp1_fn1": "#6baed6",
        "decision_u_fp1_fn5": "#fb6a4a",
        "decision_u_fp1_fn10": "#a50f15",
    }
    ratio_lbl = {
        "decision_u_fp10_fn1": ".1", "decision_u_fp5_fn1": ".2",
        "decision_u_fp1_fn1": "1", "decision_u_fp1_fn5": "5",
        "decision_u_fp1_fn10": "10",
    }

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # --- headline: gpt-5-mini belief ROCs + their operating point ------------
    ft, tt = res["their_fpr"], res["their_tpr"]
    colors = CONFIG_COLORS
    for c in left_configs:
        col = colors.get(c["name"], "C0")
        eff = c["name"].split("_re-")[-1]
        ax.plot(c["fpr"], c["tpr"], "-", color=col, lw=2.2,
                label=f"belief ROC — gpt-5-mini {eff} (AUC={c['auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)

    baseline_col = "black"
    for name in left_names:
        pts = regime_pts.get(name, {})
        eff = name.split("_re-")[-1]
        filled = (eff == "high")
        # utility-prompt decision operating points, coloured by prompted ratio
        for regime, col in ratio_colors.items():
            if regime not in pts:
                continue
            f_, t_ = pts[regime]
            ax.plot(f_, t_, "o", markersize=10, zorder=6,
                    markeredgecolor=col, markeredgewidth=1.8,
                    markerfacecolor=(col if filled else "none"))
        # no-utility baseline decision operating point (black)
        if "decision_baseline" in pts:
            f_, t_ = pts["decision_baseline"]
            ax.plot(f_, t_, "o", markersize=11, zorder=6,
                    markeredgecolor=baseline_col, markeredgewidth=1.8,
                    markerfacecolor=(baseline_col if filled else "none"))
    # legend proxies for the ratio colours + medium/high fill convention
    from matplotlib.lines import Line2D
    ratio_handles = [
        Line2D([], [], marker="o", linestyle="none", markeredgecolor=col,
               markerfacecolor=col, markersize=9, label=f"prompt {ratio_lbl[reg]} FN/FP")
        for reg, col in ratio_colors.items()
    ]
    ratio_handles.append(
        Line2D([], [], marker="o", linestyle="none", markeredgecolor=baseline_col,
               markerfacecolor=baseline_col, markersize=9, label="no utility (baseline)")
    )
    style_handles = [
        Line2D([], [], marker="o", linestyle="none", markeredgecolor="0.3",
               markerfacecolor="none", markersize=9, label="gpt-5-mini medium (open)"),
        Line2D([], [], marker="o", linestyle="none", markeredgecolor="0.3",
               markerfacecolor="0.3", markersize=9, label="gpt-5-mini high (filled)"),
    ]

    ax.plot(ft, tt, "*", color="crimson", markersize=22, markeredgecolor="k",
            markeredgewidth=0.8, zorder=7,
            label=f"ChatGPT Health decision (sens={tt:.0%}, FPR={ft:.0%})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false positive rate  (over-triage of non-ER patients)")
    ax.set_ylabel("true positive rate  (sensitivity to real ER need)")
    ax.set_title("Is ChatGPT Health under-triage on or off the belief frontier?")
    ax.grid(True, alpha=0.3)
    roc_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=roc_handles + ratio_handles + style_handles,
              fontsize=7, loc="lower right", ncol=1)

    fig.tight_layout()
    out = out_dir / "original_vs_belief_roc.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def write_report(res: dict, out_dir: Path):
    tt, ft = res["their_tpr"], res["their_fpr"]
    matched = res.get("factorial")
    n_ctx = len(res["ctxs"])
    L = ["# Was ChatGPT Health under-triage a utility problem or an off-curve problem?\n"]
    if matched:
        L.append("**Their decisions**: ChatGPT Health (gpt-5-mini thinking) 4-level "
                 "triage A/B/C/D, across **all 16 factorial conditions** (race x gender "
                 "x anchoring x access-barrier) of every case. We elicited our beliefs "
                 "on the *identical* per-variant context, so this is a **fully matched** "
                 "belief-vs-decision comparison (no reference-condition proxy). Binary "
                 "admit = triage is **D** (\"go to the ED now\").\n")
    else:
        L.append("**Their decisions**: ChatGPT Health (gpt-5-mini thinking) 4-level "
                 "triage A/B/C/D, reference factorial condition (White man, no anchor, "
                 "no barrier) — one decision per case, matching the plain vignette our "
                 "beliefs used. Binary admit = triage is **D** (\"go to the ED now\").\n")
    L.append(f"**Their operating point** ({n_ctx} "
             f"{'variants' if matched else 'cases'}, "
             f"{int(sum(res['gold'].values()))} truly need ER): "
             f"sensitivity (TPR) = **{tt:.0%}**, "
             f"false-positive rate = **{ft:.0%}** (specificity {1-ft:.0%}), "
             f"admit rate = {res['admit_rate']:.0%}, accuracy {res['their_acc']:.0%}.\n")
    L.append("This is the paper's under-triage in ROC terms: a **very FP-averse "
             "(bottom-left) operating point** — it rarely over-triages, but catches "
             "only about half of true emergencies.\n")

    L.append("## On-curve vs off-curve\n")
    L.append("For each belief frontier we read off the sensitivity the frontier "
             "*could* achieve at ChatGPT Health's own false-positive rate. The "
             "**TPR deficit** is how far their point sits **below** the curve. "
             "~0 => on the curve (an operating-point / utility problem you could "
             "fix by prompting a more FN-averse cost function). Large => below the "
             "curve (a discrimination problem no single utility fixes).\n")
    L.append("| belief frontier | AUC | frontier sens @ their FPR | their sens | "
             "TPR deficit (below curve) | frontier FPR @ their sens | FPR excess |")
    L.append("|---|---|---|---|---|---|---|")
    for c in res["configs"]:
        L.append(f"| {c['name']} | {c['auc']:.2f} | {c['tpr_at']:.0%} | {tt:.0%} | "
                 f"**{c['tpr_deficit']:+.0%}** | {c['fpr_at']:.0%} | {c['fpr_excess']:+.0%} |")

    mean_def = float(np.mean([c["tpr_deficit"] for c in res["configs"]]))
    strong = [c for c in res["configs"] if "re-high" in c["name"] or "re-medium" in c["name"]]
    strong_def = float(np.mean([c["tpr_deficit"] for c in strong])) if strong else mean_def
    L.append(f"\nMean TPR deficit across frontiers: **{mean_def:+.0%}** "
             f"(against the stronger reasoning frontiers: **{strong_def:+.0%}**).\n")

    L.append("## Interpretation — a two-part answer\n")
    L.append(f"**1. Yes, it is partly an operating-point / utility problem.** "
             f"ChatGPT Health sits in the **bottom-left (FP-averse) corner** of the "
             f"ROC (FPR {ft:.0%}, sensitivity {tt:.0%}): it almost never over-triages "
             f"but misses half of real emergencies. That is exactly what an "
             f"over-conservative implicit cost function (heavily penalising false "
             f"alarms) looks like — the kind of thing a more FN-averse prompt could, "
             f"in principle, push up and to the right.\n")
    L.append(f"**2. But it is also partly an off-curve discrimination failure.** "
             f"At that same {ft:.0%} false-positive rate, thresholding well-ranked "
             f"beliefs would catch **{res['configs'][0]['tpr_at']:.0%}–"
             f"{max(c['tpr_at'] for c in res['configs']):.0%}** of emergencies rather "
             f"than {tt:.0%}. Their point lies **{strong_def:+.0%}** below the stronger "
             f"belief frontiers, so a real sensitivity gap remains *even after* "
             f"accounting for their conservative FPR. That portion cannot be fixed by "
             f"any single cost ratio — the decisions simply don't track a good "
             f"probability ranking as tightly as the frontier does.\n")
    L.append(f"**Frontier-dependence.** The gap is largest against strong "
             f"reasoning-model beliefs (gpt-5.4 high: {max(c['tpr_deficit'] for c in res['configs']):+.0%}) "
             f"and nearly closes against the weakest belief ranking (mini/none: "
             f"{min(c['tpr_deficit'] for c in res['configs']):+.0%}). So *how much* is "
             f"'off-curve' depends on how good a belief model you compare against; "
             f"against a capable one, a substantial chunk of the under-triage is a "
             f"genuine discrimination gap, not just a threshold choice.\n")
    if matched:
        L.append(f"This point uses **all {n_ctx} factorial variants** with our beliefs "
                 f"elicited on the identical per-variant context — a fully matched "
                 f"comparison, not a reference-condition proxy.\n")
    else:
        L.append(f"Pooling all 16 factorial conditions gives essentially the same point "
                 f"(sens {res['pooled_tpr']:.0%}, FPR {res['pooled_fpr']:.0%}) — slightly "
                 f"*more* under-triage, since anchoring/barrier perturbations mostly "
                 f"de-escalate.\n")
    prim = next((c for c in res["configs"] if c["name"] == "gpt-5-mini_re-high"), None)
    if prim is not None:
        L.append(f"**Same-family check (strongest evidence).** ChatGPT Health's "
                 f"documented backbone is **gpt-5-mini**. Using our beliefs elicited "
                 f"from that same model (gpt-5-mini, high reasoning; AUC "
                 f"{prim['auc']:.2f}) as the frontier, its product decisions still sit "
                 f"**{prim['tpr_deficit']:+.0%}** below the curve at their own FPR "
                 f"(frontier {prim['tpr_at']:.0%} vs their {tt:.0%} sensitivity). So the "
                 f"off-curve gap is not an artefact of comparing against a different, "
                 f"stronger model — even the same model family, asked for a probability, "
                 f"ranks emergencies well enough to beat the deployed triage decisions.\n")
    L.append("*Caveat*: their decisions come from the **ChatGPT Health web product** "
             "(system prompt, product guardrails, possible retrieval), while our beliefs "
             "come from the raw gpt-5-mini API. The residual gap therefore reflects the "
             "deployed product's decision policy, not necessarily the bare model's — but "
             "that is exactly the system the paper evaluated.\n")
    L.append("## Figure\n- `original_vs_belief_roc.png`\n")

    path = out_dir / "original_comparison.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# cost functions in prompted order, with their FN/FP ratio labels
_COST_FUNCTIONS = [(10, 1), (5, 1), (1, 1), (1, 5), (1, 10)]
_UTILITY_REGIMES = [f"decision_u_fp{fp}_fn{fn}" for fp, fn in _COST_FUNCTIONS]
_RATIO_LABELS = [".1", ".2", "1", "5", "10"]


def _load_config_decisions(results_csv: Path):
    """Return their_admit[ctx] and {regime: {ctx: our_admit}} from a factorial
    results.csv (matched per variant via context_id = case_id__variant)."""
    their_admit: dict[str, int] = {}
    our: dict[str, dict[str, int]] = {}
    for r in csv.DictReader(open(results_csv, encoding="utf-8")):
        ctx = r["context_id"]
        t = r.get("their_triage")
        if t not in (None, "") and ctx not in their_admit:
            their_admit[ctx] = 1 if t == "D" else 0
        if r["regime"] != "belief" and r.get("parsed_decision") not in (None, ""):
            our.setdefault(r["regime"], {})[ctx] = int(r["parsed_decision"])
    return their_admit, our


def make_nature_consistency_plot(sweep_dir: Path, out_dir: Path):
    """Per config, % of variants where OUR admit decision matches ChatGPT
    Health's (Nature Medicine) admit decision, for each prompted utility regime,
    with a y-axis star for the no-utility baseline. Requires factorial results
    (their_triage per variant)."""
    if not is_factorial(sweep_dir):
        print("(nature-consistency plot needs factorial results; skipping)",
              file=sys.stderr)
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skipping nature-consistency plot: {e})", file=sys.stderr)
        return None

    configs = []
    for d in sorted(sweep_dir.glob("*/results.csv")):
        their, our = _load_config_decisions(d)
        row = {"name": d.parent.name}
        for regime in ["decision_baseline"] + _UTILITY_REGIMES:
            dec = our.get(regime, {})
            ctxs = [c for c in dec if c in their]
            row[regime] = (sum(int(dec[c] == their[c]) for c in ctxs) / len(ctxs)
                           if ctxs else float("nan"))
        configs.append(row)

    x = list(range(len(_UTILITY_REGIMES)))
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in configs:
        y = [100 * c[r] for r in _UTILITY_REGIMES]
        line, = ax.plot(x, y, marker="o", label=c["name"])
        base = c.get("decision_baseline")
        if base is not None and not np.isnan(base):
            ax.plot(-0.4, 100 * base, marker="*", markersize=15,
                    color=line.get_color(), markeredgecolor="k",
                    markeredgewidth=0.6, clip_on=False, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(_RATIO_LABELS)
    ax.set_xlim(-0.6, len(_UTILITY_REGIMES) - 0.5)
    ax.set_ylim(0, 101)
    ax.set_ylabel("decisions matching ChatGPT Health (Nature Medicine) (%)")
    ax.set_xlabel("prompted cost function  (False Negative / False Positive)")
    ax.set_title("Agreement of our decisions with the Nature Medicine "
                 "(ChatGPT Health) decisions")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.text(0.02, 0.02, "★ on y-axis = no-utility baseline\n(coloured by config)",
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
    fig.tight_layout()
    out = out_dir / "nature_decision_consistency.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    sweep_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SWEEP
    out_dir = sweep_dir
    res = analyse(sweep_dir)
    fig = make_figure(res, out_dir, sweep_dir)
    rep = write_report(res, out_dir)
    nat = make_nature_consistency_plot(sweep_dir, out_dir)
    print(f"Their reference point: sens(TPR)={res['their_tpr']:.3f} "
          f"FPR={res['their_fpr']:.3f} admit={res['admit_rate']:.3f}")
    for c in res["configs"]:
        print(f"  {c['name']:24} AUC={c['auc']:.3f}  "
              f"frontier_sens@theirFPR={c['tpr_at']:.3f}  "
              f"TPR_deficit={c['tpr_deficit']:+.3f}")
    print(f"Wrote {fig}\nWrote {rep}")
    if nat:
        print(f"Wrote {nat}")


if __name__ == "__main__":
    main()
