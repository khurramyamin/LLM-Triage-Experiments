"""Prompted-vs-recovered probability-threshold figure.

Companion to fig2_recovered_capability.png (utility prompting). Here we prompt the
model with an explicit probability threshold p* and recover the threshold implied
by its decisions given its independently elicited beliefs.

  X axis: Belief Threshold Prompted   (the p* we instructed the model to use)
  Y axis: Recovered Belief Threshold  (p* recovered from its actual decisions)

Beliefs are reused from the factorial (utility) run; the threshold decisions come
from the new threshold sweep. For each config x regime we fit the same logit cost
function used elsewhere to the (belief, threshold-decision) pairs, obtain the
recovered FN/FP ratio, and convert to a recovered threshold:

    recovered p* = 1 / (1 + recovered_ratio_fn_fp) = c_FP / (c_FP + c_FN).

Styling mirrors fig2: hue = model family, shade/line style = reasoning level,
dashed diagonal = perfect following.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import analyze_sweep as A
import revealed_preferences as rp
from make_paper_figures import (
    CONFIG_ORDER, FAMILY_ORDER, FAMILY_LABEL, FAMILY_CMAP, FAMILY_MARKER,
    FAMILY_REASONS, REASON_LABEL, TIER_LABELS, reason_rank,
    style, parse, OUT,
)

BELIEF_SWEEP = Path("revealed_preferences/2026-07-01_factorial")
THRESH_SWEEP = Path("revealed_preferences/2026-07-08_threshold")

# threshold regimes, ordered efficiency-first -> safety-first, with prompted p*
THRESH_REGIMES = [f"decision_{rp.threshold_label(fp, fn)}"
                  for fp, fn in rp.COST_FUNCTIONS]
PROMPTED_PSTAR = {f"decision_{rp.threshold_label(fp, fn)}":
                  round(rp.admit_threshold(fp, fn), 2)
                  for fp, fn in rp.COST_FUNCTIONS}


def load_beliefs(config: str) -> dict:
    """Reused independently elicited beliefs from the factorial run."""
    belief, _dec, _gold = A.load_results(BELIEF_SWEEP / config / "results.csv")
    return belief


def load_threshold_decisions(config: str) -> dict:
    """{regime: {ctx: decision}} from the threshold sweep."""
    _b, decisions, _g = A.load_results(THRESH_SWEEP / config / "results.csv")
    return decisions


def recovered_threshold(belief: dict, decisions_ctx: dict):
    """Fit the logit cost function on matched (belief, decision) pairs and return
    the recovered probability threshold p* = 1/(1+ratio). None if degenerate."""
    ps, ds = [], []
    for ctx, dec in decisions_ctx.items():
        if ctx in belief:
            ps.append(belief[ctx])
            ds.append(dec)
    if len(ds) < 5:
        return None
    fit = rp.fit_cost_function(ps, ds)
    if fit.get("degenerate") or not fit.get("ratio_fn_fp"):
        return None
    ratio = fit["ratio_fn_fp"]
    return 1.0 / (1.0 + ratio)


def build() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.55, lw=1.4, zorder=1,
            label="Perfect Following")

    for name in CONFIG_ORDER:
        cfg_dir = THRESH_SWEEP / name
        # require settings.json so we only plot fully-completed configs
        if not (cfg_dir / "results.csv").exists() or \
           not (cfg_dir / "settings.json").exists():
            continue
        belief = load_beliefs(name)
        decisions = load_threshold_decisions(name)
        st = style(name)
        xs, ys = [], []
        for regime in THRESH_REGIMES:
            dctx = decisions.get(regime, {})
            rec = recovered_threshold(belief, dctx) if dctx else None
            if rec is not None:
                xs.append(PROMPTED_PSTAR[regime])
                ys.append(rec)
        order = np.argsort(xs)
        xs = list(np.array(xs)[order])
        ys = list(np.array(ys)[order])
        ax.plot(xs, ys, marker=st["marker"], ls=st["ls"], color=st["color"],
                lw=2.0, markersize=8, markeredgecolor="k", markeredgewidth=0.5,
                label=st["label"], zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.09, 0.17, 0.5, 0.83, 0.91])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlabel("Belief Threshold Prompted", fontsize=15)
    ax.set_ylabel("Recovered Belief Threshold", fontsize=15)
    ax.set_title("Capable Models Follow The Thresholds They Are Given",
                 fontsize=15)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=10.5, loc="upper left", ncol=1,
              framealpha=0.93)

    fig.tight_layout()
    out = OUT / "fig4_recovered_threshold.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def build_defaults() -> None:
    """fig1b-style figure but for default (no-threshold) probability thresholds.

    Uses the baseline (no-utility) decisions from the factorial run: those were
    prompted without any threshold, so the threshold recovered from them is each
    model's default operating threshold. p* = 1 / (1 + recovered_ratio_fn_fp)."""
    cfg_dirs = sorted(p.parent for p in BELIEF_SWEEP.glob("*/results.csv"))
    by_name = {d.name: A.analyse_config(d) for d in cfg_dirs}

    fig, axn = plt.subplots(figsize=(8.2, 4.2))
    thresholds = []
    for n in [c for c in CONFIG_ORDER if c in by_name]:
        r = by_name[n]["baseline_ratio"]
        if r is None:
            continue
        pstar = 1.0 / (1.0 + r)
        thresholds.append(pstar)
        fam, reason = parse(n)
        axn.plot(pstar, reason_rank(fam, reason), marker=FAMILY_MARKER[fam],
                 markersize=15, color=FAMILY_CMAP[fam](0.85), markeredgecolor="k",
                 markeredgewidth=0.7, zorder=5)
    axn.axvline(0.5, color="0.6", ls="--", lw=1.0, zorder=1)
    lo = min(thresholds) * 0.7
    hi = max(thresholds) * 1.25
    axn.set_xlim(max(0.0, lo), min(1.0, hi))
    axn.set_ylim(-0.6, 3.1)
    axn.set_yticks([0, 1, 2])
    axn.set_yticklabels(TIER_LABELS, fontsize=13)
    axn.set_ylabel("Reasoning Level", fontsize=15)
    for side in ("right", "top"):
        axn.spines[side].set_visible(False)
    axn.tick_params(axis="x", labelsize=13)
    axn.set_xlabel("Default Recovered Belief Threshold", fontsize=16)
    axn.set_title("Model's Default Thresholds When None Are Specified",
                  fontsize=17, fontweight="bold", pad=10)
    axn.grid(True, axis="x", alpha=0.25)
    axn.text(max(0.0, lo) * 1.01 + 0.002, 2.82, "Safety-First", fontsize=13,
             color="0.4", ha="left", va="center")
    axn.text(min(1.0, hi) * 0.995, 2.82, "Efficiency-First", fontsize=13,
             color="0.4", ha="right", va="center")

    fam_handles = [
        Line2D([], [], marker=FAMILY_MARKER[f], linestyle="none",
               markerfacecolor=FAMILY_CMAP[f](0.85), markeredgecolor="k",
               markersize=11, label=FAMILY_LABEL[f])
        for f in FAMILY_ORDER
    ]
    axn.legend(handles=fam_handles, fontsize=13, loc="center right", ncol=1,
               framealpha=0.9, handletextpad=0.3,
               title="Model Family", title_fontsize=13)
    out = OUT / "fig4b_default_thresholds.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def build_belief_grid() -> None:
    """Nine box plots of the elicited belief distribution, one per config,
    clustered by model family (3 reasoning levels per family, in order), coloured
    by model family. On each box: a purple square marks the config's default
    recovered probability threshold p* = 1/(1+baseline_ratio), and a yellow square
    marks the best fixed threshold vs the gold clinical labels."""
    cfg_dirs = sorted(p.parent for p in BELIEF_SWEEP.glob("*/results.csv"))
    by_name = {d.name: A.analyse_config(d) for d in cfg_dirs}

    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    positions, box_data, box_colors, ylabels = [], [], [], []
    star_x, star_y = [], []      # default recovered threshold (purple)
    best_x, best_y = [], []      # best fixed threshold vs gold labels (yellow)
    pos = 0.0
    # build top-to-bottom per family with the highest reasoning tier on top, so
    # each family is clustered on the y-axis (high -> low within family).
    for fam in FAMILY_ORDER:
        for rank in (2, 1, 0):          # high, mid, low tier  (top->bottom)
            eff = FAMILY_REASONS[fam][rank]
            name = f"{fam}_re-{eff}"
            if name not in by_name:
                continue
            belief, _dec, gold = A.load_results(
                BELIEF_SWEEP / name / "results.csv")
            vals = [belief[c] for c in belief]
            positions.append(pos)
            box_data.append(vals)
            box_colors.append(FAMILY_CMAP[fam](0.75))
            ylabels.append(f"{FAMILY_LABEL[fam]}  {REASON_LABEL[eff]}")
            rr = by_name[name]["baseline_ratio"]
            if rr is not None:
                star_x.append(1.0 / (1.0 + rr))
                star_y.append(pos)
            thr = A.best_fixed_threshold(belief, gold)[0]
            if thr is not None and not np.isnan(thr):
                best_x.append(thr)
                best_y.append(pos)
            pos += 1.0
        pos += 0.8  # gap between model families

    bp = ax.boxplot(box_data, positions=positions, vert=False, widths=0.7,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=2.5,
                                    markerfacecolor="0.35",
                                    markeredgecolor="none", alpha=0.4),
                    medianprops=dict(color="k", lw=1.6))
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
        patch.set_edgecolor("k")

    # magenta = default recovered threshold, yellow = best fixed threshold
    # (magenta chosen to stay distinct from the purple DeepSeek-Flash boxes)
    ax.plot(star_x, star_y, "s", color="#e7298a", markersize=14,
            markeredgecolor="k", markeredgewidth=1.0, zorder=7, linestyle="none")
    ax.plot(best_x, best_y, "s", color="gold", markersize=14,
            markeredgecolor="k", markeredgewidth=1.0, zorder=7, linestyle="none")

    ax.set_yticks(positions)
    ax.set_yticklabels(ylabels, fontsize=11)
    ax.set_ylim(positions[0] - 0.9, positions[-1] + 0.9)
    ax.invert_yaxis()   # first config at the top
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Elicited Belief  P(needs emergency care)", fontsize=14)
    ax.set_title("Elicited Belief Distributions By Model And Reasoning Level",
                 fontsize=15, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    fam_handles = [
        Line2D([], [], marker="s", linestyle="none",
               markerfacecolor=FAMILY_CMAP[f](0.75), markeredgecolor="k",
               markersize=12, label=FAMILY_LABEL[f])
        for f in FAMILY_ORDER
    ]
    default_handle = Line2D([], [], marker="s", linestyle="none", color="#e7298a",
                            markeredgecolor="k", markersize=12,
                            label="Default recovered threshold")
    best_handle = Line2D([], [], marker="s", linestyle="none", color="gold",
                         markeredgecolor="k", markersize=12,
                         label="Best fixed threshold")
    ax.legend(handles=fam_handles + [default_handle, best_handle], fontsize=11,
              loc="lower right", framealpha=0.95)

    fig.tight_layout()
    out = OUT / "fig_belief_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    # fig4_recovered_threshold.png is now generated by make_full_utility_figures
    # (full seven-threshold version); build() is kept for ad-hoc use but is not
    # run here so it can't overwrite the canonical full-utility figure.
    build_defaults()
    build_belief_grid()
