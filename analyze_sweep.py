"""Analyse a revealed-preference sweep directory.

Reads every ``<model>_re-<effort>/results.csv`` produced by
``revealed_preferences.py --sweep`` and reports, per config:

  * the baseline implied cost ratio (no utility function given);
  * how the recovered ratio tracks the *prompted* cost ratio
    (Spearman rho over the 5 utility regimes, on log ratios);
  * "self-consistency": how often the model's actual YES/NO matches the
    expected-cost-minimising threshold rule  admit  <=>  belief >= p*,
    where p* = c_FP / (c_FP + c_FN) is implied by the prompted costs.

It also writes two figures (admit-rate curves, recovered-vs-target scatter).

Usage:
    python analyze_sweep.py <sweep_dir>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

# (c_FP, c_FN) in the order they appear in the summary, with their labels.
COST_FUNCTIONS = [(10, 1), (5, 1), (1, 1), (1, 5), (1, 10)]


def cost_label(c_fp: int, c_fn: int) -> str:
    return f"u_fp{c_fp}_fn{c_fn}"


UTILITY_REGIMES = [f"decision_{cost_label(fp, fn)}" for fp, fn in COST_FUNCTIONS]
TARGET_RATIO = {f"decision_{cost_label(fp, fn)}": fn / fp for fp, fn in COST_FUNCTIONS}
THRESHOLD = {f"decision_{cost_label(fp, fn)}": fp / (fp + fn) for fp, fn in COST_FUNCTIONS}
REGIME_ORDER = ["decision_baseline"] + UTILITY_REGIMES


def load_results(results_path: Path):
    """Return belief_by_ctx, {regime: {ctx: decision}}, gold_by_ctx."""
    belief: dict[str, float] = {}
    decisions: dict[str, dict[str, int]] = {}
    gold: dict[str, int] = {}
    with open(results_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ctx = r["context_id"]
            g = r.get("needs_er_gold")
            if g not in (None, "") and ctx not in gold:
                try:
                    gold[ctx] = int(float(g))
                except ValueError:
                    pass
            if r["regime"] == "belief":
                if r.get("parsed_probability") not in (None, ""):
                    belief[ctx] = float(r["parsed_probability"])
            elif r.get("parsed_decision") not in (None, ""):
                decisions.setdefault(r["regime"], {})[ctx] = int(r["parsed_decision"])
    return belief, decisions, gold


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation without scipy."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def _matched_pairs(belief: dict, decisions: dict, regime: str):
    """Matched (belief, decision) arrays for one regime."""
    dctx = decisions.get(regime, {})
    ps, ds = [], []
    for ctx, dec in dctx.items():
        if ctx in belief:
            ps.append(belief[ctx])
            ds.append(dec)
    return np.asarray(ps, float), np.asarray(ds, int)


def bootstrap_ratio_ci(belief: dict, decisions: dict, regime: str,
                       n_boot: int = 500, seed: int = 0):
    """Percentile 95% CI for the recovered c_FN/c_FP ratio of one regime.

    Resamples the matched (belief, decision) pairs with replacement, refits the
    logit cost function each time (via revealed_preferences.fit_cost_function),
    and returns (lo, hi) from the 2.5/97.5 percentiles of the recovered ratio.
    Bootstrap draws that are degenerate (all-admit / all-not-admit) are skipped.
    """
    from revealed_preferences import fit_cost_function
    p, a = _matched_pairs(belief, decisions, regime)
    n = len(a)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        fit = fit_cost_function(list(p[idx]), list(a[idx]))
        if not fit.get("degenerate") and fit.get("ratio_fn_fp"):
            ratios.append(fit["ratio_fn_fp"])
    if len(ratios) < max(10, n_boot // 20):
        return None
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_prop_ci(indicators, n_boot: int = 4000, seed: int = 0):
    """Percentile 95% CI for the mean of a list of 0/1 indicators.

    Resamples the indicators with replacement and returns (lo, hi) from the
    2.5/97.5 percentiles of the bootstrap distribution of the mean. Used for
    the agreement-rate figures (decision<->belief consistency and agreement
    with the deployed tool), where each independent observation is one
    per-variant decision.
    """
    a = np.asarray(indicators, dtype=float)
    n = len(a)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def analyse_config(cfg_dir: Path) -> dict:
    belief, decisions, gold = load_results(cfg_dir / "results.csv")
    fit = json.loads((cfg_dir / "fit.json").read_text(encoding="utf-8"))

    # admit rate + recovered ratio per regime
    admit_rate = {}
    recovered = {}
    p_star_est = {}          # threshold implied by the *estimated* (recovered) utility
    for regime in REGIME_ORDER:
        f = fit.get(regime, {})
        admit_rate[regime] = f.get("frac_admit")
        recovered[regime] = None if f.get("degenerate") else f.get("ratio_fn_fp")
        cfp, cfn = f.get("c_fp"), f.get("c_fn")
        if not f.get("degenerate") and cfp is not None and cfn is not None and (cfp + cfn) > 0:
            p_star_est[regime] = cfp / (cfp + cfn)
        else:
            p_star_est[regime] = None

    # monotonicity of recovered ratio vs prompted target (utility regimes only)
    tgt, rec = [], []
    for regime in UTILITY_REGIMES:
        if recovered[regime] is not None:
            tgt.append(np.log(TARGET_RATIO[regime]))
            rec.append(np.log(recovered[regime]))
    rho = spearman(tgt, rec) if len(tgt) >= 3 else float("nan")

    # admit-rate monotonicity (always defined; the most robust signal)
    ar = [admit_rate[r] for r in UTILITY_REGIMES]
    rho_admit = spearman(list(range(len(ar))), ar)

    # self-consistency: actual decision vs. threshold rule on own belief,
    # using the *prompted* threshold p* = c_FP/(c_FP+c_FN).
    consistency = {}
    for regime in UTILITY_REGIMES:
        p_star = THRESHOLD[regime]
        agree = tot = 0
        for ctx, dec in decisions.get(regime, {}).items():
            if ctx not in belief:
                continue
            rule = 1 if belief[ctx] >= p_star else 0
            agree += int(rule == dec)
            tot += 1
        consistency[regime] = agree / tot if tot else float("nan")

    # decision <-> (belief + ESTIMATED utility) consistency: does the fitted
    # utility, applied as a threshold to the elicited belief, reproduce the
    # model's actual YES/NO decisions?  (goodness-of-fit of the revealed model)
    est_consistency = {}
    for regime in REGIME_ORDER:
        p_star = p_star_est[regime]
        if p_star is None:
            est_consistency[regime] = float("nan")
            continue
        agree = tot = 0
        for ctx, dec in decisions.get(regime, {}).items():
            if ctx not in belief:
                continue
            rule = 1 if belief[ctx] >= p_star else 0
            agree += int(rule == dec)
            tot += 1
        est_consistency[regime] = agree / tot if tot else float("nan")

    return {
        "name": cfg_dir.name,
        "belief": belief,
        "decisions": decisions,
        "gold": gold,
        "p_star_est": p_star_est,
        "baseline_ratio": recovered["decision_baseline"],
        "baseline_admit": admit_rate["decision_baseline"],
        "admit_rate": admit_rate,
        "recovered": recovered,
        "rho_recovered": rho,
        "rho_admit": rho_admit,
        "consistency": consistency,
        "mean_consistency": float(np.nanmean(list(consistency.values()))),
        "est_consistency": est_consistency,
        "mean_est_consistency": float(np.nanmean(
            [est_consistency[r] for r in UTILITY_REGIMES])),
    }


def roc_curve(scores: list[float], labels: list[int]):
    """ROC (fpr, tpr) and AUC from scores vs binary labels, no sklearn.

    Sweeps the decision threshold from high to low so the curve runs from
    (0,0) to (1,1).  AUC is the trapezoidal area.
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    P = int((y == 1).sum())
    N = int((y == 0).sum())
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    order = np.argsort(-s)               # descending score
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = np.concatenate([[0.0], tp / P])
    fpr = np.concatenate([[0.0], fp / N])
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trap(tpr, fpr))
    return fpr, tpr, auc


def best_fixed_threshold(belief: dict, gold: dict):
    """Best single threshold on belief that matches gold Y (max accuracy).

    Returns (threshold, accuracy, tpr, fpr, implied_ratio_fn_fp).  This is the
    single fixed 'utility value' whose belief>=thr rule best reproduces the
    true labels — i.e. the best operating point reachable on the belief ROC.
    """
    ctxs = [c for c in belief if c in gold]
    if not ctxs:
        return (float("nan"),) * 5
    p = np.array([belief[c] for c in ctxs])
    y = np.array([gold[c] for c in ctxs])
    cands = np.unique(np.concatenate([[0.0], p, [1.0 + 1e-9]]))
    best = (0.5, -1.0, float("nan"), float("nan"), float("nan"))
    P = max(int((y == 1).sum()), 1)
    N = max(int((y == 0).sum()), 1)
    for thr in cands:
        pred = (p >= thr).astype(int)
        acc = float((pred == y).mean())
        if acc > best[1]:
            tpr = float(((pred == 1) & (y == 1)).sum() / P)
            fpr = float(((pred == 1) & (y == 0)).sum() / N)
            ratio = (1 - thr) / thr if thr > 0 else float("inf")
            best = (float(thr), acc, tpr, fpr, ratio)
    return best


def best_fixed_threshold_cost(belief: dict, gold: dict, ratio_fn_fp: float):
    """Best single threshold on belief under an asymmetric FN/FP cost ratio.

    Sweeps a fixed cut-off over the elicited beliefs and returns the one that
    minimises the cost-weighted error c_FN*FN + c_FP*FP with c_FN/c_FP =
    ``ratio_fn_fp`` (c_FP fixed at 1). ``ratio_fn_fp = 1`` reproduces the
    accuracy-maximising point of ``best_fixed_threshold``; larger ratios
    penalise missed emergencies more and push the point toward higher
    sensitivity. Returns (threshold, tpr, fpr).
    """
    ctxs = [c for c in belief if c in gold]
    if not ctxs:
        return (float("nan"), float("nan"), float("nan"))
    p = np.array([belief[c] for c in ctxs])
    y = np.array([gold[c] for c in ctxs])
    cands = np.unique(np.concatenate([[0.0], p, [1.0 + 1e-9]]))
    P = max(int((y == 1).sum()), 1)
    N = max(int((y == 0).sum()), 1)
    c_fp, c_fn = 1.0, float(ratio_fn_fp)
    best_cost = float("inf")
    best = (0.5, float("nan"), float("nan"))
    for thr in cands:
        pred = (p >= thr).astype(int)
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        cost = c_fn * fn + c_fp * fp
        if cost < best_cost:
            best_cost = cost
            tpr = float(((pred == 1) & (y == 1)).sum() / P)
            fpr = float(((pred == 1) & (y == 0)).sum() / N)
            best = (float(thr), tpr, fpr)
    return best


def operating_point(decisions_ctx: dict, gold: dict):
    """(fpr, tpr, accuracy) of a set of actual decisions against gold Y."""
    ctxs = [c for c in decisions_ctx if c in gold]
    if not ctxs:
        return float("nan"), float("nan"), float("nan")
    pred = np.array([decisions_ctx[c] for c in ctxs])
    y = np.array([gold[c] for c in ctxs])
    P = max(int((y == 1).sum()), 1)
    N = max(int((y == 0).sum()), 1)
    tpr = float(((pred == 1) & (y == 1)).sum() / P)
    fpr = float(((pred == 1) & (y == 0)).sum() / N)
    acc = float((pred == y).mean())
    return fpr, tpr, acc


def make_plots(configs: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skipping plots: {e})", file=sys.stderr)
        return

    xticklabels = [".1", ".2", "1", "5", "10"]
    x = list(range(len(UTILITY_REGIMES)))

    # 1) admit-rate curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in configs:
        y = [100 * c["admit_rate"][r] for r in UTILITY_REGIMES]
        line, = ax.plot(x, y, marker="o", label=c["name"])
        base = c["admit_rate"].get("decision_baseline")
        if base is not None:
            ax.plot(-0.4, 100 * base, marker="*", markersize=15,
                    color=line.get_color(), markeredgecolor="k",
                    markeredgewidth=0.6, clip_on=False, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(-0.6, len(UTILITY_REGIMES) - 0.5)
    ax.set_ylabel("admit rate (% YES)")
    ax.set_xlabel("prompted cost function  (False Negative / False Positive)")
    ax.set_title("ER-admit rate vs prompted cost function")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.text(0.02, 0.02, "★ on y-axis = no-utility baseline\n(coloured by config)",
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out_dir / "admit_rate.png", dpi=130)
    plt.close(fig)

    # 2) recovered vs target (log-log)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    xlims = [0.02, 20]
    # bootstrap 95% CIs for each config's recovered ratios (utility regimes)
    print("  computing bootstrap CIs for recovered ratios...", file=sys.stderr)
    ci = {}
    for c in configs:
        ci[c["name"]] = {}
        for r in UTILITY_REGIMES:
            if c["recovered"][r] is not None:
                ci[c["name"]][r] = bootstrap_ratio_ci(c["belief"], c["decisions"], r)
            else:
                ci[c["name"]][r] = None
    # y-limits adapt to the data (incl. CI bounds) so nothing is clipped
    yvals = []
    for c in configs:
        for r in UTILITY_REGIMES:
            if c["recovered"][r] is not None:
                yvals.append(c["recovered"][r])
                bounds = ci[c["name"]][r]
                if bounds is not None:
                    yvals += [bounds[0], bounds[1]]
        if c["baseline_ratio"] is not None:
            yvals.append(c["baseline_ratio"])
    ylo = min([0.02] + [v for v in yvals if v > 0]) * 0.8
    yhi = max([20] + yvals) * 1.15
    ylims = [ylo, yhi]
    ax.plot(xlims, xlims, "k--", alpha=0.5, label="identity (perfect recovery)")
    for c in configs:
        xs, ys, lo_err, hi_err = [], [], [], []
        for r in UTILITY_REGIMES:
            if c["recovered"][r] is not None:
                y = c["recovered"][r]
                xs.append(TARGET_RATIO[r])
                ys.append(y)
                bounds = ci[c["name"]][r]
                if bounds is not None:
                    lo_err.append(max(y - bounds[0], 0.0))
                    hi_err.append(max(bounds[1] - y, 0.0))
                else:
                    lo_err.append(0.0)
                    hi_err.append(0.0)
        line = ax.errorbar(xs, ys, yerr=[lo_err, hi_err], marker="o",
                           capsize=3, elinewidth=1.2, label=c["name"])
        col = line[0].get_color()
        # baseline (no utility prompted) implied ratio, drawn as a dot on the
        # y-axis in the same colour as this config's steering curve.
        base = c["baseline_ratio"]
        if base is not None:
            ax.plot(xlims[0], base, marker="*", markersize=15,
                    color=col, markeredgecolor="k",
                    markeredgewidth=0.6, clip_on=False, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    # show only the five prompted cost-function ratios as x ticks
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.2, 1, 5, 10]))
    ax.xaxis.set_major_formatter(FixedFormatter([".1", ".2", "1", "5", "10"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 1, 5, 10]))
    ax.yaxis.set_major_formatter(FixedFormatter([".1", ".2", "1", "5", "10"]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", pad=12, labelsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlabel("Prompted Costs (FN/FP)", fontsize=15)
    ax.set_ylabel("Recovered Costs (FN/FP)", fontsize=15)
    ax.set_title("Prompted vs Recovered Cost Ratios", fontsize=15)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.98, "★ on y-axis = no-utility baseline\n(coloured by config)",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out_dir / "recovered_vs_target.png", dpi=130)
    plt.close(fig)
    print(f"Wrote plots to {out_dir}", file=sys.stderr)


def make_consistency_plot(configs: list[dict], out_dir: Path) -> None:
    """Decision <-> (belief + estimated utility) consistency, per config.

    x = prompted cost function (.1,.2,1,5,10); y = % of contexts where the
    fitted-utility threshold rule on the elicited belief reproduces the model's
    actual decision.  A star on the y-axis marks the baseline (no utility
    prompted) consistency for each config, in its curve colour.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skipping consistency plot: {e})", file=sys.stderr)
        return

    x = list(range(len(UTILITY_REGIMES)))
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in configs:
        y = [100 * c["est_consistency"][r] for r in UTILITY_REGIMES]
        line, = ax.plot(x, y, marker="o", label=c["name"])
        base = c["est_consistency"].get("decision_baseline")
        if base is not None and not np.isnan(base):
            ax.plot(-0.35, 100 * base, marker="*", markersize=15,
                    color=line.get_color(), markeredgecolor="k",
                    markeredgewidth=0.6, clip_on=False, zorder=5)
    ax.plot([], [], marker="*", markersize=13, linestyle="none", color="0.4",
            markeredgecolor="k", markeredgewidth=0.6,
            label="baseline (no utility prompted)")
    ax.set_xticks(x)
    ax.set_xticklabels([".1", ".2", "1", "5", "10"])
    ax.set_xlim(-0.5, len(UTILITY_REGIMES) - 0.5)
    ax.set_ylim(0, 101)
    ax.set_ylabel("decisions matching belief + estimated utility (%)")
    ax.set_xlabel("prompted cost function  (False Negative / False Positive)")
    ax.set_title("Consistency of decisions with (elicited belief + estimated utility)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "decision_belief_consistency.png", dpi=130)
    plt.close(fig)
    print(f"Wrote consistency plot to {out_dir}", file=sys.stderr)


def make_roc_plot(configs: list[dict], out_dir: Path) -> None:
    """Oracle ROC of elicited beliefs vs true Y, with model operating points.

    One panel per config: the ROC curve of the elicited probabilities against
    the gold 'needs ER' label (the best achievable frontier using the beliefs),
    the best fixed-threshold point (a single 'utility value'), and the actual
    decision regimes as operating points (baseline + the 5 prompted utilities).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skipping ROC plot: {e})", file=sys.stderr)
        return

    n = len(configs)
    ncol = 3 if n >= 3 else n
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 6.0 * nrow),
                             squeeze=False)
    # each decision regime gets its own colour (filled circle), labelled once
    # in a shared legend so the points are readable without text annotations.
    regime_style = {
        "decision_baseline":    ("no utility (baseline)", "#000000"),
        "decision_u_fp10_fn1":  ("prompt 0.1 FN/FP ratio  (FP-averse)", "#08519c"),
        "decision_u_fp5_fn1":   ("prompt 0.2 FN/FP ratio", "#3182bd"),
        "decision_u_fp1_fn1":   ("prompt 1 FN/FP ratio", "#6baed6"),
        "decision_u_fp1_fn5":   ("prompt 5 FN/FP ratio", "#fb6a4a"),
        "decision_u_fp1_fn10":  ("prompt 10 FN/FP ratio  (FN-averse)", "#a50f15"),
    }
    for i, c in enumerate(configs):
        ax = axes[i // ncol][i % ncol]
        scores = [c["belief"][k] for k in c["belief"] if k in c["gold"]]
        labels = [c["gold"][k] for k in c["belief"] if k in c["gold"]]
        fpr, tpr, auc = roc_curve(scores, labels)
        ax.plot(fpr, tpr, "-", color="0.55", lw=2.6, zorder=2,
                label=f"belief ROC (AUC={auc:.2f})")
        ax.plot([0, 1], [0, 1], "k:", alpha=0.5, zorder=1)
        # actual decision regimes as coloured operating points
        for regime, (lbl, col) in regime_style.items():
            dctx = c["decisions"].get(regime, {})
            f_, t_, _ = operating_point(dctx, c["gold"])
            if not np.isnan(f_):
                ax.plot(f_, t_, "o", markersize=15, color=col,
                        markeredgecolor="k", markeredgewidth=0.8, zorder=5,
                        label=lbl if i == 0 else None)
        # best fixed-threshold operating point (single utility value)
        thr, acc, btpr, bfpr, ratio = best_fixed_threshold(c["belief"], c["gold"])
        if not np.isnan(bfpr):
            ax.plot(bfpr, btpr, "*", color="gold", markersize=26,
                    markeredgecolor="k", markeredgewidth=1.0, zorder=6,
                    label="best fixed utility" if i == 0 else None)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("false positive rate", fontsize=17)
        ax.set_ylabel("true positive rate", fontsize=17)
        ax.set_title(f"{c['name']}  (AUC={auc:.2f})", fontsize=18)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, alpha=0.3)
    # hide any unused panels
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    # one shared legend (built from the first panel's labelled artists)
    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=4, fontsize=16,
               frameon=True, bbox_to_anchor=(0.5, 0.0), markerscale=1.3)
    fig.suptitle("Oracle ROC of elicited beliefs vs true need-ER label\n"
                 "(coloured circles = actual decision regimes; a utility picks a point on the curve)",
                 fontsize=20)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(out_dir / "roc_oracle.png", dpi=130)
    plt.close(fig)
    print(f"Wrote ROC plot to {out_dir}", file=sys.stderr)


def fmt(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def write_report(configs: list[dict], out_dir: Path) -> Path:
    L: list[str] = ["# Revealed-preference sweep — analysis\n"]

    L.append("## 1. Default (baseline) implied preference\n")
    L.append("With **no** cost function in the prompt, the model still makes "
             "YES/NO admit decisions. Fitting those against the independently "
             "elicited beliefs recovers the model's *default* implied cost ratio "
             "c_FN/c_FP. A ratio > 1 means the model behaves as if **missing an "
             "emergency is worse** than an unnecessary ED visit.\n")
    L.append("| config | baseline c_FN/c_FP | baseline admit % |")
    L.append("|---|---|---|")
    for c in configs:
        L.append(f"| {c['name']} | {fmt(c['baseline_ratio'])} | "
                 f"{fmt(100*c['baseline_admit'],0)}% |")

    L.append("\n## 2. Does the model follow the prompted cost function?\n")
    L.append("`rho_admit` = Spearman rank corr. between the prompted FN-aversion "
             "(ordered 10:1 -> 1:10) and the realised admit rate. `rho_recovered` "
             "= same, but on the *recovered* log cost ratio. ~1.0 = the model "
             "steers exactly as instructed; ~0 = it ignores the cost function.\n")
    L.append("| config | rho_admit | rho_recovered | mean self-consistency |")
    L.append("|---|---|---|---|")
    for c in configs:
        L.append(f"| {c['name']} | {fmt(c['rho_admit'])} | "
                 f"{fmt(c['rho_recovered'])} | {fmt(100*c['mean_consistency'],0)}% |")

    L.append("\n## 3. Self-consistency by regime (actual decision vs. own-belief threshold rule)\n")
    L.append("For each prompted cost function the expected-cost-minimising rule is "
             "`admit <=> belief >= p*`, with p* = c_FP/(c_FP+c_FN). The cells show the "
             "fraction of cases where the model's actual decision matches that rule "
             "applied to its **own** elicited belief.\n")
    head = "| config | " + " | ".join(
        f"{r.replace('decision_','')} (p*={THRESHOLD[r]:.2f})" for r in UTILITY_REGIMES
    ) + " |"
    L.append(head)
    L.append("|" + "---|" * (len(UTILITY_REGIMES) + 1))
    for c in configs:
        cells = [fmt(100*c["consistency"][r], 0) + "%" for r in UTILITY_REGIMES]
        L.append(f"| {c['name']} | " + " | ".join(cells) + " |")

    L.append("\n## 4. Recovered cost ratio by regime\n")
    L.append("Target ratios: " + ", ".join(
        f"{r.replace('decision_','')}={TARGET_RATIO[r]:g}" for r in UTILITY_REGIMES) + "\n")
    head = "| config | " + " | ".join(r.replace("decision_", "") for r in UTILITY_REGIMES) + " |"
    L.append(head)
    L.append("|" + "---|" * (len(UTILITY_REGIMES) + 1))
    for c in configs:
        cells = [fmt(c["recovered"][r]) for r in UTILITY_REGIMES]
        L.append(f"| {c['name']} | " + " | ".join(cells) + " |")

    L.append("\n## 5. Decisions vs (elicited belief + ESTIMATED utility)\n")
    L.append("Here the threshold comes from the utility we **fit** to each regime "
             "(the *recovered* ratio), not the prompted one: `admit <=> belief >= "
             "p*_est`, p*_est = c_FP/(c_FP+c_FN) from the fit. The cells show how "
             "often that rule reproduces the model's actual YES/NO — i.e. how well "
             "'belief + one number' explains the decisions.\n")
    head = "| config | baseline | " + " | ".join(
        r.replace("decision_", "") for r in UTILITY_REGIMES) + " | mean(util) |"
    L.append(head)
    L.append("|" + "---|" * (len(UTILITY_REGIMES) + 3))
    for c in configs:
        base = fmt(100 * c["est_consistency"]["decision_baseline"], 0) + "%"
        cells = [fmt(100 * c["est_consistency"][r], 0) + "%" for r in UTILITY_REGIMES]
        L.append(f"| {c['name']} | {base} | " + " | ".join(cells) +
                 f" | {fmt(100*c['mean_est_consistency'],0)}% |")

    L.append("\n## 6. Oracle: elicited beliefs vs the true need-ER label\n")
    L.append("The elicited probabilities are scored against the gold `needs_er_gold` "
             "label. The ROC **AUC** is how well the beliefs alone rank who truly "
             "needs the ER. *Any* utility function is just a threshold on the belief "
             "— i.e. a point on this ROC. `best fixed utility` is the single "
             "threshold that best matches the gold labels (the most desired behaviour "
             "reachable by fixing one utility value); its implied c_FN/c_FP is shown.\n")
    L.append("| config | belief AUC | best fixed-utility acc | best threshold | "
             "implied c_FN/c_FP | baseline-decision acc |")
    L.append("|---|---|---|---|---|---|")
    for c in configs:
        thr, acc, btpr, bfpr, ratio = best_fixed_threshold(c["belief"], c["gold"])
        scores = [c["belief"][k] for k in c["belief"] if k in c["gold"]]
        labels = [c["gold"][k] for k in c["belief"] if k in c["gold"]]
        _, _, auc = roc_curve(scores, labels)
        _, _, bacc = operating_point(c["decisions"].get("decision_baseline", {}), c["gold"])
        L.append(f"| {c['name']} | {fmt(auc)} | {fmt(100*acc,0)}% | {fmt(thr)} | "
                 f"{fmt(ratio)} | {fmt(100*bacc,0)}% |")

    L.append("\n## 7. Figures\n")
    L.append("- `admit_rate.png` — admit rate vs prompted cost function (steering curves).")
    L.append("- `recovered_vs_target.png` — recovered vs prompted ratio, log-log, with identity line.")
    L.append("- `decision_belief_consistency.png` — % of decisions matching belief+estimated utility (stars = baseline).")
    L.append("- `roc_oracle.png` — ROC of elicited beliefs vs true need-ER label, with decision regimes as operating points.\n")

    path = out_dir / "analysis.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python analyze_sweep.py <sweep_dir>", file=sys.stderr)
        sys.exit(1)
    sweep_dir = Path(sys.argv[1])
    cfg_dirs = sorted(p.parent for p in sweep_dir.glob("*/results.csv"))
    if not cfg_dirs:
        print(f"No results.csv under {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    configs = [analyse_config(d) for d in cfg_dirs]
    make_plots(configs, sweep_dir)
    make_consistency_plot(configs, sweep_dir)
    make_roc_plot(configs, sweep_dir)
    report = write_report(configs, sweep_dir)

    # console digest
    print("\nconfig                     base  rho_admit  rho_rec  self-cons  est-cons   AUC")
    print("-" * 84)
    for c in configs:
        scores = [c["belief"][k] for k in c["belief"] if k in c["gold"]]
        labels = [c["gold"][k] for k in c["belief"] if k in c["gold"]]
        _, _, auc = roc_curve(scores, labels)
        print(f"{c['name']:<26}{fmt(c['baseline_ratio']):>5}"
              f"{fmt(c['rho_admit']):>10}{fmt(c['rho_recovered']):>9}"
              f"{fmt(100*c['mean_consistency'],0):>9}%"
              f"{fmt(100*c['mean_est_consistency'],0):>9}%"
              f"{fmt(auc):>6}")
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
