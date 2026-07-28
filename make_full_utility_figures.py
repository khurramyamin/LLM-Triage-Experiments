"""Generate 'full utilities' versions of the utility/threshold figures.

The main paper figures (make_paper_figures.py / make_threshold_figure.py) show the
original five prompted cost ratios (FN/FP = .1, .2, 1, 5, 10) and their five
threshold counterparts.  This companion script regenerates the figures that
depend on the prompted utilities/thresholds using the FULL set of seven ratios,
i.e. adding the two extreme conditions collected later:

    FN/FP = .01  (c_FP=100, c_FN=1)   p* = 0.99   efficiency-first extreme
    FN/FP = 100  (c_FP=1, c_FN=100)   p* = 0.01   safety-first extreme

Everything is written to  nature_medicine_paper/figures/full_utilities/  so the
main figures are left untouched.  Only figures that actually plot the prompted
utilities/thresholds are regenerated:

    fig1_deployed_roc.png / _advanced.png / original_deployed_roc.png
    fig2_recovered_capability.png
    fig3_roc_grid_{gpt,deepseek,claude}.png
    fig4_recovered_threshold.png

Default-only figures (fig1b, fig4b, fig_belief_distributions) are NOT affected by
the extra utilities and are intentionally not reproduced here.

Run:  .venv\\Scripts\\python.exe make_full_utility_figures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

import analyze_sweep as A
import revealed_preferences as rp
import original_paper_comparison as O
from make_paper_figures import (
    SWEEP, CONFIG_ORDER, FAMILY_ORDER, FAMILY_LABEL, FAMILY_CMAP,
    PROVIDER_GROUPS, parse, provider_config_names, style, pretty,
)

# ---------------------------------------------------------------------------
# Expand the cost-function set to the full seven and re-derive the constants
# the analysis code reads (both modules), so analyse_config picks up the two
# extra prompted-utility regimes and the threshold figure gets the extra points.
# ---------------------------------------------------------------------------
FULL_COST_FUNCTIONS = [(100, 1), (10, 1), (5, 1), (1, 1), (1, 5), (1, 10), (1, 100)]

A.COST_FUNCTIONS = FULL_COST_FUNCTIONS
A.UTILITY_REGIMES = [f"decision_{A.cost_label(fp, fn)}" for fp, fn in FULL_COST_FUNCTIONS]
A.TARGET_RATIO = {f"decision_{A.cost_label(fp, fn)}": fn / fp
                  for fp, fn in FULL_COST_FUNCTIONS}
A.THRESHOLD = {f"decision_{A.cost_label(fp, fn)}": fp / (fp + fn)
               for fp, fn in FULL_COST_FUNCTIONS}
A.REGIME_ORDER = ["decision_baseline"] + A.UTILITY_REGIMES
rp.COST_FUNCTIONS = FULL_COST_FUNCTIONS

BELIEF_SWEEP = SWEEP
THRESH_SWEEP = Path("revealed_preferences/2026-07-08_threshold")
# The full-utility figures are now the canonical paper figures (they replaced the
# old five-utility versions), so write straight into the main figures directory.
OUT = Path("nature_medicine_paper/figures")

# Full efficiency->safety palette for the seven prompted utilities.
RATIO_COLORS = {
    "decision_u_fp100_fn1": "#08306b",
    "decision_u_fp10_fn1":  "#08519c",
    "decision_u_fp5_fn1":   "#3182bd",
    "decision_u_fp1_fn1":   "#6baed6",
    "decision_u_fp1_fn5":   "#fb6a4a",
    "decision_u_fp1_fn10":  "#a50f15",
    "decision_u_fp1_fn100": "#67000d",
}
RATIO_LBL = {
    "decision_u_fp100_fn1": ".01", "decision_u_fp10_fn1": ".1",
    "decision_u_fp5_fn1": ".2", "decision_u_fp1_fn1": "1",
    "decision_u_fp1_fn5": "5", "decision_u_fp1_fn10": "10",
    "decision_u_fp1_fn100": "100",
}

# Best fixed decision rule shown in the ROC grid (fig3): the single belief
# threshold minimising cost-weighted error for a given FN/FP ratio. We show
# three ratios in distinct square colours; .2 leans efficiency and 5 leans
# safety (yellow), spreading the points along the belief ROC.
BEST_FIXED_SQUARES = [
    (0.2, "#1b9e77"),
    (1, "#e7298a"),
    (5, "gold"),
]


# ===========================================================================
# FIGURE 1 -- deployed tool ROC with the full set of utility operating points
# ===========================================================================
def figure1() -> None:
    res = O.analyse(SWEEP)
    ft, tt = res["their_fpr"], res["their_tpr"]

    left_names = ["gpt-5-mini_re-medium", "gpt-5-mini_re-high"]
    adv_name = "gpt-5.4_re-medium"
    adv_high_name = "gpt-5.4_re-high"
    deepseek_name = "DeepSeek-V4-Pro_re-high"
    claude_name = "Claude-Fable-5_re-high"
    load_names = left_names + [adv_name, adv_high_name, deepseek_name, claude_name]
    frontier = {c["name"]: c for c in res["configs"] if c["name"] in load_names}
    regime_pts = {
        n: O.load_regime_op_points(SWEEP / n / "results.csv")
        for n in load_names if (SWEEP / n / "results.csv").exists()
    }
    fam_col = {
        "gpt-5-mini_re-medium": "#3976b9",
        "gpt-5-mini_re-high": "#08306b",
        "gpt-5.4_re-medium": "#d65a2a",
        "gpt-5.4_re-high": "#d65a2a",
        "DeepSeek-V4-Pro_re-high": "#2f8a57",
        "Claude-Fable-5_re-high": "#a50f15",
    }

    # best fixed-utility operating point (vs the gold clinical labels) per config,
    # the same square shown in the fig3 ROC grid. We use the safety-leaning
    # FN/FP = 5 cost ratio, so the point sits at the higher-sensitivity end of the
    # belief ROC that a site prioritising missed emergencies would target.
    BEST_FIXED_RATIO = 5
    best_pts = {}
    for n in load_names:
        d = SWEEP / n
        if (d / "results.csv").exists():
            cc = A.analyse_config(d)
            _thr, btpr, bfpr = A.best_fixed_threshold_cost(
                cc["belief"], cc["gold"], BEST_FIXED_RATIO)
            if not np.isnan(bfpr):
                best_pts[n] = (bfpr, btpr)

    def render_roc(names, out_name, show_style_legend, force_filled,
                   title="Why The Deployed Tool Under-Triaged",
                   show_best_fixed=False):
        fig, ax = plt.subplots(figsize=(7.6, 7.0))
        for n in names:
            c = frontier.get(n)
            if c is None:
                continue
            fam, eff = parse(n)
            ax.plot(c["fpr"], c["tpr"], "-", color=fam_col[n], lw=2.3,
                    label=f"Belief Ranking \u2014 {FAMILY_LABEL[fam]} "
                          f"{eff.capitalize()} (AUC={c['auc']:.2f})")
        ax.plot([0, 1], [0, 1], "k:", alpha=0.5)

        for n in names:
            pts = regime_pts.get(n, {})
            filled = force_filled or n.endswith("high")
            for regime, col in RATIO_COLORS.items():
                if regime not in pts:
                    continue
                f_, t_ = pts[regime]
                ax.plot(f_, t_, "o", markersize=9, zorder=6,
                        markeredgecolor=col, markeredgewidth=1.7,
                        markerfacecolor=(col if filled else "none"))
            if "decision_baseline" in pts:
                f_, t_ = pts["decision_baseline"]
                ax.plot(f_, t_, "o", markersize=10, zorder=6,
                        markeredgecolor="k", markeredgewidth=1.7,
                        markerfacecolor=("k" if filled else "none"))

        if show_best_fixed:
            for n in names:
                if n in best_pts:
                    bfpr, btpr = best_pts[n]
                    ax.plot(bfpr, btpr, "s", color="gold", markersize=15,
                            markeredgecolor="k", markeredgewidth=1.0, zorder=8)

        ax.plot(ft, tt, "*", color="crimson", markersize=24, markeredgecolor="k",
                markeredgewidth=0.9, zorder=7)

        star_handle = Line2D([], [], marker="*", linestyle="none", color="crimson",
                             markeredgecolor="k", markeredgewidth=0.6, markersize=17,
                             label=f"Deployed Tool (Sens={tt:.0%}, Over-Triage={ft:.0%})")
        ratio_handles = [
            Line2D([], [], marker="o", linestyle="none", markeredgecolor=col,
                   markerfacecolor=col, markersize=8,
                   label=f"Prompt (FN/FP = {RATIO_LBL[reg]})")
            for reg, col in RATIO_COLORS.items()
        ]
        ratio_handles.append(
            Line2D([], [], marker="o", linestyle="none", markeredgecolor="k",
                   markerfacecolor="k", markersize=8, label="No Utility (Default)"))
        if show_best_fixed:
            ratio_handles.append(
                Line2D([], [], marker="s", linestyle="none", color="gold",
                       markeredgecolor="k", markeredgewidth=0.6, markersize=9,
                       label="Best Fixed Utility"))
        style_handles = []
        if show_style_legend:
            style_handles = [
                Line2D([], [], marker="o", linestyle="none", markeredgecolor="0.3",
                       markerfacecolor="none", markersize=8,
                       label="Medium Reasoning (Open)"),
                Line2D([], [], marker="o", linestyle="none", markeredgecolor="0.3",
                       markerfacecolor="0.3", markersize=8,
                       label="High Reasoning (Filled)"),
            ]
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=16)
        ax.set_ylabel("True Positive Rate", fontsize=16)
        ax.set_title(title, fontsize=17, fontweight="bold")
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, alpha=0.3)
        roc_handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=roc_handles + [star_handle] + ratio_handles + style_handles,
                  fontsize=11.5, loc="lower right", ncol=1, framealpha=0.92)
        fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_name}")

    def render_grid() -> None:
        panel_specs = [
            ("gpt-5-mini_re-high", "GPT-5 Mini \u00b7 High", "#3976b9",
             "#f3f7fc", (0.60, 0.72)),
            ("gpt-5.4_re-high", "GPT-5.4 \u00b7 High", "#d65a2a",
             "#fcf5f1", (0.50, 0.66)),
            ("DeepSeek-V4-Pro_re-high", "DeepSeek-V4-Pro \u00b7 High",
             "#2f8a57", "#f2f8f4", (0.50, 0.84)),
                 ("Claude-Fable-5_re-high", "Claude Fable 5 \u00b7 High",
                  "#a50f15", "#fcf2f3", (0.44, 0.82)),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(18.5, 16.8))
        fig.patch.set_facecolor("#f5f4f0")
        fig.subplots_adjust(left=0.12, right=0.975, top=0.865, bottom=0.20,
                            wspace=0.33, hspace=0.48)

        for ax, (name, panel_name, line_color, panel_bg, best_text) in zip(
                axes.flat, panel_specs):
            c = frontier[name]
            pts = regime_pts[name]
            ax.set_facecolor("white")
            ax.fill_between(c["fpr"], 0, c["tpr"], color=panel_bg, zorder=0)
            ax.plot(c["fpr"], c["tpr"], color=line_color, lw=2.8, zorder=3)
            ax.plot([0, 1], [0, 1], color="#b9b3a8", ls=(0, (1.5, 3.2)),
                    lw=1.4, zorder=1)

            for regime, col in RATIO_COLORS.items():
                if regime not in pts:
                    continue
                f_, t_ = pts[regime]
                ax.plot(f_, t_, "o", markersize=12, zorder=6,
                        markeredgecolor="white", markeredgewidth=1.1,
                        markerfacecolor=col)

            if "decision_baseline" in pts:
                f_, t_ = pts["decision_baseline"]
                ax.plot(f_, t_, "o", markersize=12, color="#20242a",
                        markeredgecolor="white", markeredgewidth=1.1, zorder=6)

            if name in best_pts:
                bfpr, btpr = best_pts[name]
                ax.plot(bfpr, btpr, "s", color="#f4c542", markersize=15,
                        markeredgecolor="#20242a", markeredgewidth=1.3, zorder=8)
                ax.annotate(
                    f"Best fixed utility (FN/FP=5)\n{btpr:.0%} sensitivity\n"
                    f"{bfpr:.0%} over-triage",
                    xy=(bfpr, btpr), xytext=best_text,
                    fontsize=13, color="#25282d", fontweight="bold",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="#777169", lw=1.3,
                                    shrinkA=5, shrinkB=5),
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="#d7d2c9", lw=0.8, alpha=0.97),
                    zorder=9,
                )

            ax.plot(ft, tt, "*", color="#d83456", markersize=24,
                    markeredgecolor="#7e1830", markeredgewidth=1.0, zorder=8)
            ax.annotate(
                f"Deployed tool\n{tt:.0%} sens \u00b7 {ft:.0%} over-triage",
                xy=(ft, tt), xytext=(min(ft + 0.09, 0.67), tt - 0.03),
                fontsize=12, color="#25282d", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#777169", lw=1.3,
                                shrinkA=5, shrinkB=5),
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#d7d2c9", lw=0.8, alpha=0.97),
                zorder=9,
            )

            ax.set_xlim(-0.04, 1)
            ax.set_ylim(-0.04, 1.06)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.set_xlabel("False Positive Rate (Over-Triage)", fontsize=18,
                          labelpad=12)
            ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=18,
                          labelpad=14)
            ax.set_title(f"{panel_name} (AUC={c['auc']:.2f})", loc="left",
                         fontsize=19, fontweight="bold", pad=15)
            ax.tick_params(labelsize=15, colors="#615d56")
            ax.grid(True, color="#ddd9d1", alpha=0.65, lw=0.8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#dedad2")

        # Rounded cards around the four panels, matching the supplied layout.
        for ax in axes.flat:
            pos = ax.get_position()
            card = FancyBboxPatch(
                (pos.x0 - 0.065, pos.y0 - 0.055),
                pos.width + 0.100, pos.height + 0.105,
                boxstyle="round,pad=0.008,rounding_size=0.010",
                transform=fig.transFigure, facecolor="white",
                edgecolor="#ddd9d1", linewidth=0.9, zorder=-10,
            )
            fig.add_artist(card)

        legend_handles = [
            Line2D([], [], linestyle="none", marker=None,
                   label="Prompted utility, FN/FP ="),
        ] + [
            Line2D([], [], linestyle="none", marker="o", markersize=13,
                   markerfacecolor=col, markeredgecolor="white",
                   label=RATIO_LBL[reg])
            for reg, col in RATIO_COLORS.items()
        ] + [
            Line2D([], [], linestyle="none", marker=None, label="   "),
            Line2D([], [], linestyle="none", marker="o", markersize=13,
                   markerfacecolor="#20242a", markeredgecolor="white",
                   label="No utility (default)"),
            Line2D([], [], linestyle="none", marker="s", markersize=13,
                   markerfacecolor="#f4c542", markeredgecolor="#20242a",
                   label="Best fixed utility (FN/FP=5)"),
            Line2D([], [], linestyle="none", marker="*", markersize=17,
                   markerfacecolor="#d83456", markeredgecolor="#7e1830",
                   label="Deployed tool"),
        ]
        left_card = axes[0, 0].get_position()
        right_card = axes[0, 1].get_position()
        legend_left = left_card.x0 - 0.065
        legend_right = right_card.x1 + 0.035
        legend_width = legend_right - legend_left
        legend_box = FancyBboxPatch(
            (legend_left, 0.068), legend_width, 0.040,
            boxstyle="round,pad=0.008,rounding_size=0.006",
            transform=fig.transFigure, facecolor="white",
            edgecolor="#ddd9d1", linewidth=0.9, zorder=10,
        )
        fig.add_artist(legend_box)
        legend = fig.legend(
            handles=legend_handles, loc="lower left",
            bbox_to_anchor=(legend_left - 0.010, 0.064),
            ncol=len(legend_handles), fontsize=16.5, frameon=False,
            columnspacing=0.72, handletextpad=0.3, handlelength=0.85,
        )
        legend.get_texts()[0].set_fontweight("bold")
        legend.set_zorder(12)
        fig.add_artist(legend)
        out = OUT / "fig1_emergency_triage_performance.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.14,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"wrote {out.name}")

    render_roc(left_names, "original_deployed_roc.png",
               show_style_legend=True, force_filled=False)
    render_roc(["gpt-5-mini_re-medium"], "fig1_deployed_roc.png",
               show_style_legend=False, force_filled=True,
               title="GPT-Health Under-Triage vs Utility Prompting (GPT-5 Mini)",
               show_best_fixed=True)
    render_roc([adv_name], "fig1_deployed_roc_advanced.png",
               show_style_legend=False, force_filled=True,
               title="GPT-Health Under-Triage vs Utility Prompting (GPT-5.4)",
               show_best_fixed=True)
    render_grid()


# ===========================================================================
# FIGURE 2 -- recovered vs prompted cost ratio (full range)
# ===========================================================================
def figure2(configs_by_name: dict, config_order=None, out_dir=None,
            style_fn=None) -> None:
    if config_order is None:
        config_order = CONFIG_ORDER
    if out_dir is None:
        out_dir = OUT
    if style_fn is None:
        style_fn = style
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.2), sharex=True, sharey=True)
    # widened range to include the two extreme prompted ratios (0.01 and 100);
    # recovered ratios are clipped to [FLOOR, CEIL] for display so a near-zero
    # recovered ratio (perfect efficiency-following) still shows on the log axis.
    lims = [0.006, 170]
    FLOOR, CEIL = 0.006, 170.0
    ticks = [0.01, 0.1, 0.2, 1, 5, 10, 100]
    tick_lbls = [".01", ".1", ".2", "1", "5", "10", "100"]
    panels = [
        ("GPT models", {"gpt-5-mini", "gpt-5.4"}),
        ("DeepSeek models", {"DeepSeek-V4-Flash", "DeepSeek-V4-Pro"}),
        ("Claude models", {"Claude-Fable-5", "Claude-Sonnet-5"}),
    ]
    for ax, (title, families) in zip(axes, panels):
        if families:
            ax.plot([0.01, 100], [0.01, 100], "k--", alpha=0.55, lw=1.4,
                    zorder=1, label="Perfect Following")
        for name in config_order:
            family, _ = parse(name)
            if family not in families:
                continue
            c = configs_by_name.get(name)
            if c is None:
                continue
            st = style_fn(name)
            xs, ys, elo, ehi = [], [], [], []
            for regime in A.UTILITY_REGIMES:
                if c["recovered"].get(regime) is not None:
                    xs.append(A.TARGET_RATIO[regime])
                    recovered = c["recovered"][regime]
                    y = min(max(recovered, FLOOR), CEIL)
                    ys.append(y)
                    ci = A.bootstrap_ratio_ci(c["belief"], c["decisions"], regime)
                    if ci is not None:
                        lo = min(max(ci[0], FLOOR), CEIL)
                        hi = min(max(ci[1], FLOOR), CEIL)
                        elo.append(max(y - lo, 0.0))
                        ehi.append(max(hi - y, 0.0))
                    else:
                        elo.append(0.0)
                        ehi.append(0.0)
            ax.errorbar(xs, ys, yerr=[elo, ehi], marker=st["marker"], ls=st["ls"],
                        color=st["color"], lw=2.0, markersize=8,
                        markeredgecolor="k", markeredgewidth=0.5,
                        ecolor=st["color"], elinewidth=1.2, capsize=3,
                        capthick=1.1, label=st["label"], zorder=4)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_lbls))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(tick_lbls))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="both", labelsize=17)
        ax.set_title(title, fontsize=21, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        if families:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=11.5, loc="lower right", ncol=1,
                      framealpha=0.93)

    fig.supxlabel("Prompted Costs (FN/FP)", fontsize=23, y=0.01)
    fig.supylabel("Recovered Costs (FN/FP)", fontsize=23, x=0.025)
    fig.tight_layout(rect=(0.03, 0.05, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig2_recovered_capability.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'fig2_recovered_capability.png'}")


# ===========================================================================
# FIGURE 3 -- belief-ROC grid with the full set of utility operating points
# ===========================================================================
def figure3(configs_by_name: dict, config_order=None, out_dir=None) -> None:
    if config_order is None:
        config_order = CONFIG_ORDER
    if out_dir is None:
        out_dir = OUT
    regime_style = {
        "decision_baseline":    ("No Utility (Default)", "#000000"),
        "decision_u_fp100_fn1": ("Prompt (FN/FP = .01)", "#08306b"),
        "decision_u_fp10_fn1":  ("Prompt (FN/FP = .1)", "#08519c"),
        "decision_u_fp5_fn1":   ("Prompt (FN/FP = .2)", "#3182bd"),
        "decision_u_fp1_fn1":   ("Prompt (FN/FP = 1)", "#6baed6"),
        "decision_u_fp1_fn5":   ("Prompt (FN/FP = 5)", "#fb6a4a"),
        "decision_u_fp1_fn10":  ("Prompt (FN/FP = 10)", "#a50f15"),
        "decision_u_fp1_fn100": ("Prompt (FN/FP = 100)", "#67000d"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for slug, provider_label, families in PROVIDER_GROUPS:
        names = provider_config_names(config_order, configs_by_name, families)
        ncol = 3
        nrow = (len(names) + ncol - 1) // ncol
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(6.4 * ncol, 6.2 * nrow), squeeze=False
        )
        for i, name in enumerate(names):
            c = configs_by_name[name]
            ax = axes[i // ncol][i % ncol]
            scores = [c["belief"][k] for k in c["belief"] if k in c["gold"]]
            labels = [c["gold"][k] for k in c["belief"] if k in c["gold"]]
            fpr, tpr, auc = A.roc_curve(scores, labels)
            ax.plot(fpr, tpr, "-", color="0.55", lw=2.6, zorder=2,
                    label=f"Belief ROC (AUC={auc:.2f})")
            ax.plot([0, 1], [0, 1], "k:", alpha=0.5, zorder=1)
            for regime, (lbl, col) in regime_style.items():
                dctx = c["decisions"].get(regime, {})
                f_, t_, _ = A.operating_point(dctx, c["gold"])
                if not np.isnan(f_):
                    ax.plot(f_, t_, "o", markersize=15, color=col,
                            markeredgecolor="k", markeredgewidth=0.8, zorder=5,
                            label=lbl if i == 0 else None)
            for ratio_val, sq_col in BEST_FIXED_SQUARES:
                _bthr, btpr, bfpr = A.best_fixed_threshold_cost(
                    c["belief"], c["gold"], ratio_val
                )
                if not np.isnan(bfpr):
                    if ratio_val < 1:
                        ratio_label = ("%g" % ratio_val).lstrip("0") or "0"
                    else:
                        ratio_label = "%g" % ratio_val
                    ax.plot(
                        bfpr, btpr, "s", color=sq_col, markersize=17,
                        markeredgecolor="k", markeredgewidth=1.0, zorder=6,
                        label=(
                            f"Best Fixed Utility (FN/FP = {ratio_label})"
                            if i == 0 else None
                        ),
                    )
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("False Positive Rate", fontsize=20)
            ax.set_ylabel("True Positive Rate", fontsize=20)
            ax.set_title(f"{pretty(name)}  (AUC={auc:.2f})", fontsize=21)
            ax.tick_params(axis="both", labelsize=16)
            ax.grid(True, alpha=0.3)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=20,
                   frameon=True, bbox_to_anchor=(0.5, 0.0), markerscale=1.3)
        fig.suptitle(
            f"ROC Belief Oracle Curves - {provider_label}",
            fontsize=30,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0.11, 1, 0.95))
        out = out_dir / f"fig3_roc_grid_{slug}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


# ===========================================================================
# FIGURE 4 -- recovered vs prompted probability threshold (full range)
# ===========================================================================
THRESH_REGIMES = [f"decision_{rp.threshold_label(fp, fn)}"
                  for fp, fn in FULL_COST_FUNCTIONS]
PROMPTED_PSTAR = {f"decision_{rp.threshold_label(fp, fn)}":
                  round(rp.admit_threshold(fp, fn), 2)
                  for fp, fn in FULL_COST_FUNCTIONS}


def _recovered_threshold(belief: dict, decisions_ctx: dict):
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
    return 1.0 / (1.0 + fit["ratio_fn_fp"])


def figure4() -> None:
    fig = plt.figure(figsize=(13.2, 11.2))
    grid = fig.add_gridspec(2, 4)
    gpt_ax = fig.add_subplot(grid[0, :2])
    axes = [
        gpt_ax,
        fig.add_subplot(grid[0, 2:], sharex=gpt_ax, sharey=gpt_ax),
        fig.add_subplot(grid[1, 1:3], sharex=gpt_ax, sharey=gpt_ax),
    ]
    panels = [
        ("GPT models", {"gpt-5-mini", "gpt-5.4"}),
        ("DeepSeek models", {"DeepSeek-V4-Flash", "DeepSeek-V4-Pro"}),
        ("Claude models", {"Claude-Fable-5", "Claude-Sonnet-5"}),
    ]

    for ax, (title, families) in zip(axes, panels):
        if families:
            ax.plot([0, 1], [0, 1], "k--", alpha=0.55, lw=1.4, zorder=1,
                    label="Perfect Following")

        for name in CONFIG_ORDER:
            family, _ = parse(name)
            if family not in families:
                continue
            cfg_dir = THRESH_SWEEP / name
            if not (cfg_dir / "results.csv").exists():
                continue
            belief, _d, _g = A.load_results(
                BELIEF_SWEEP / name / "results.csv")
            _b, decisions, _g2 = A.load_results(cfg_dir / "results.csv")
            st = style(name)
            xs, ys, elo, ehi = [], [], [], []
            for regime in THRESH_REGIMES:
                dctx = decisions.get(regime, {})
                rec = _recovered_threshold(belief, dctx) if dctx else None
                if rec is not None:
                    xs.append(PROMPTED_PSTAR[regime])
                    ys.append(rec)
                    ci = A.bootstrap_ratio_ci(belief, decisions, regime)
                    if ci is not None:
                        r_lo, r_hi = ci
                        t_lo = 1.0 / (1.0 + r_hi)   # threshold decreases in ratio
                        t_hi = 1.0 / (1.0 + r_lo)
                        elo.append(max(rec - t_lo, 0.0))
                        ehi.append(max(t_hi - rec, 0.0))
                    else:
                        elo.append(0.0)
                        ehi.append(0.0)
            order = np.argsort(xs)
            xs = list(np.array(xs)[order])
            ys = list(np.array(ys)[order])
            elo = list(np.array(elo)[order])
            ehi = list(np.array(ehi)[order])
            ax.errorbar(xs, ys, yerr=[elo, ehi], marker=st["marker"], ls=st["ls"],
                        color=st["color"], lw=2.0, markersize=8,
                        markeredgecolor="k", markeredgewidth=0.5,
                        ecolor=st["color"], elinewidth=1.2, capsize=3,
                        capthick=1.1, label=st["label"], zorder=4)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0.01, 0.09, 0.17, 0.5, 0.83, 0.91, 0.99])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="both", labelsize=17)
        ax.tick_params(axis="x", labelrotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.set_xlabel("Belief Threshold Prompted", fontsize=23)
        ax.set_ylabel("Recovered Belief Threshold", fontsize=23)
        ax.set_title(title, fontsize=21, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if families:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=10.5, loc="upper left",
                      ncol=1, framealpha=0.93)

    fig.tight_layout()
    fig.savefig(OUT / "fig4_recovered_threshold.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig4_recovered_threshold.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_dirs = sorted(p.parent for p in SWEEP.glob("*/results.csv"))
    by_name = {d.name: A.analyse_config(d) for d in cfg_dirs}
    figure1()
    figure2(by_name, CONFIG_ORDER, OUT)
    figure3(by_name, CONFIG_ORDER, OUT)
    figure4()
    print(f"\nAll full-utility figures written to {OUT}")


if __name__ == "__main__":
    main()
