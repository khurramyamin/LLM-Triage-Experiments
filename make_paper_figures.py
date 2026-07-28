"""Generate the three restructured main-text figures for the Nature Medicine paper.

New ordering requested by the reviewer:
  Fig 1  (was Fig 3): why the deployed tool failed -- its operating point on the
         belief ROC frontier, PLUS a number-line sub-panel showing that different
         models adopt a wide range of *default* (unspecified) cost ratios.
  Fig 2  (was Fig 1): recovered vs prompted cost ratio, redesigned so the
         capability gradient (model family mini->not-mini, reasoning level
         minimal/none->medium->high) is immediately visible via colour + style.
  Fig 3  (was Fig 2): the belief-ROC grid, unchanged (copied over).

Run:  .venv\\Scripts\\python.exe make_paper_figures.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

import analyze_sweep as A
import original_paper_comparison as O

SWEEP = Path("revealed_preferences/2026-07-01_factorial")
OUT = Path("nature_medicine_paper/figures")

# ---------------------------------------------------------------------------
# Shared capability-gradient styling.
#   hue     = model family        (mini -> not-mini)
#   shade   = reasoning level      (minimal/none -> medium -> high)
#   line    = reasoning level      (reinforces the shade)
#   marker  = model family         (reinforces the hue, aids colour-blind)
# ---------------------------------------------------------------------------
# GPT-5.4-Mini intentionally excluded from all paper figures (per request).
# Its label/colour/marker entries are kept below but it is absent from
# FAMILY_ORDER and CONFIG_ORDER, so no figure iterates over it.
FAMILY_ORDER = [
    "gpt-5-mini", "gpt-5.4",
    "DeepSeek-V4-Flash", "DeepSeek-V4-Pro",
    "Claude-Fable-5", "Claude-Sonnet-5",
]
FAMILY_LABEL = {"gpt-5-mini": "GPT-5-Mini", "gpt-5.4-mini": "GPT-5.4-Mini",
                "gpt-5.4": "GPT-5.4",
                "DeepSeek-V4-Flash": "DeepSeek-V4-Flash",
                "DeepSeek-V4-Pro": "DeepSeek-V4-Pro",
                "Claude-Fable-5": "Claude Fable 5",
                "Claude-Sonnet-5": "Claude Sonnet 5"}
FAMILY_CMAP = {"gpt-5-mini": plt.cm.Blues, "gpt-5.4-mini": plt.cm.Greens,
               "gpt-5.4": plt.cm.Oranges,
               "DeepSeek-V4-Flash": plt.cm.Purples,
               "DeepSeek-V4-Pro": plt.cm.Greens,
               "Claude-Fable-5": plt.cm.Reds,
               "Claude-Sonnet-5": plt.cm.Greys}
FAMILY_MARKER = {"gpt-5-mini": "o", "gpt-5.4-mini": "s", "gpt-5.4": "^",
                "DeepSeek-V4-Flash": "D", "DeepSeek-V4-Pro": "P",
                "Claude-Fable-5": "X", "Claude-Sonnet-5": "v"}
PROVIDER_GROUPS = [
    ("gpt", "GPT models", ("gpt-5-mini", "gpt-5.4")),
    ("deepseek", "DeepSeek models", ("DeepSeek-V4-Flash", "DeepSeek-V4-Pro")),
    ("claude", "Claude models", ("Claude-Sonnet-5", "Claude-Fable-5")),
]

# Each family exposes exactly three reasoning levels, but the *names* differ
# across vendors (GPT: minimal/none/medium/high; DeepSeek: none/high/max).  The
# capability-gradient styling (shade + line style + row) is therefore driven by
# each level's ORDINAL RANK within its own family (0=low, 1=mid, 2=high) rather
# than by the raw reasoning string, so the three tiers line up visually.
FAMILY_REASONS = {
    "gpt-5-mini": ["minimal", "medium", "high"],
    "gpt-5.4-mini": ["none", "medium", "high"],
    "gpt-5.4": ["none", "medium", "high"],
    "DeepSeek-V4-Flash": ["none", "high", "max"],
    "DeepSeek-V4-Pro": ["none", "high", "max"],
    "Claude-Fable-5": ["low", "medium", "high"],
    "Claude-Sonnet-5": ["low", "medium", "high"],
}
REASON_LABEL = {"minimal": "Minimal", "none": "None", "medium": "Medium",
                "high": "High", "max": "Max", "low": "Low"}
SHADE_BY_RANK = [0.55, 0.75, 0.98]
LS_BY_RANK = [":", "--", "-"]
# Generic tier labels shared by all families for the reasoning-level axis.
TIER_LABELS = ["Low/None", "Medium", "High"]


def reason_rank(fam: str, reason: str) -> int:
    return FAMILY_REASONS[fam].index(reason)


CONFIG_ORDER = [
    "gpt-5-mini_re-minimal", "gpt-5-mini_re-medium", "gpt-5-mini_re-high",
    "gpt-5.4_re-none", "gpt-5.4_re-medium", "gpt-5.4_re-high",
    "DeepSeek-V4-Flash_re-none", "DeepSeek-V4-Flash_re-high",
    "DeepSeek-V4-Flash_re-max",
    "DeepSeek-V4-Pro_re-none", "DeepSeek-V4-Pro_re-high",
    "DeepSeek-V4-Pro_re-max",
    "Claude-Fable-5_re-low", "Claude-Fable-5_re-medium",
    "Claude-Fable-5_re-high",
    "Claude-Sonnet-5_re-low", "Claude-Sonnet-5_re-medium",
    "Claude-Sonnet-5_re-high",
]


def reduced_config_order():
    """Main-text config order: two configs per family, shown as a low and a high
    reasoning tier. DeepSeek's highest setting (`max`) is used as the high
    representative. Grouped by family, low then high."""
    out = []
    for fam in FAMILY_ORDER:
        low, high = MAIN_TIERS[fam]
        out.append(f"{fam}_re-{low}")
        out.append(f"{fam}_re-{high}")
    return out


# Which two reasoning levels each family contributes to the reduced main-text
# figures: (low representative, high representative). DeepSeek uses its highest
# setting, `max`, as the high representative.
MAIN_TIERS = {
    "gpt-5-mini": ("minimal", "high"),
    "gpt-5.4-mini": ("none", "high"),
    "gpt-5.4": ("none", "high"),
    "DeepSeek-V4-Flash": ("none", "max"),
    "DeepSeek-V4-Pro": ("none", "max"),
    "Claude-Fable-5": ("low", "high"),
    "Claude-Sonnet-5": ("low", "high"),
}
# Reduced set shown in the main-text capability-gradient figures (fig2, fig3).
# Figure 1b uses the full three-tier CONFIG_ORDER.
MAIN_CONFIG_ORDER = reduced_config_order()
# Row/label order for the reduced figures' reasoning axis.
MAIN_TIER_LABELS = ["Low", "High"]


def reduced_tier(name: str) -> int:
    """Display tier within the reduced set: 0 = low representative, 1 = high."""
    fam, reason = parse(name)
    low, _high = MAIN_TIERS[fam]
    return 0 if reason == low else 1


# Styling for the reduced figures: the low representative renders as the lightest
# / dotted tier and the high representative as the darkest / solid tier, so the
# two shown levels read consistently across every model family (even though
# DeepSeek's "high" is ordinally its middle level).
_REDUCED_SHADE = [SHADE_BY_RANK[0], SHADE_BY_RANK[2]]
_REDUCED_LS = [LS_BY_RANK[0], LS_BY_RANK[2]]


def reduced_style(name: str):
    fam, reason = parse(name)
    t = reduced_tier(name)
    return {
        "color": FAMILY_CMAP[fam](_REDUCED_SHADE[t]),
        "marker": FAMILY_MARKER[fam],
        "ls": _REDUCED_LS[t],
        "label": f"{FAMILY_LABEL[fam]} \u00b7 {REASON_LABEL[reason]}",
    }


def parse(name: str):
    fam, reason = name.split("_re-")
    return fam, reason


def provider_config_names(config_order, configs_by_name, families):
    """Return provider configurations grouped by the requested family order."""
    return [
        name
        for family in families
        for name in config_order
        if name in configs_by_name and parse(name)[0] == family
    ]


def style(name: str):
    fam, reason = parse(name)
    rank = reason_rank(fam, reason)
    color = FAMILY_CMAP[fam](SHADE_BY_RANK[rank])
    return {
        "color": color,
        "marker": FAMILY_MARKER[fam],
        "ls": LS_BY_RANK[rank],
        "label": f"{FAMILY_LABEL[fam]} \u00b7 {REASON_LABEL[reason]}",
    }


def pretty(name: str) -> str:
    fam, reason = parse(name)
    return f"{FAMILY_LABEL[fam]} \u00b7 {REASON_LABEL[reason]}"


# ===========================================================================
# FIGURE 1 -- deployed tool ROC  (standalone)
# FIGURE 1b -- spread of default priorities  (standalone)
# ===========================================================================
def figure1(configs_by_name: dict) -> None:
    res = O.analyse(SWEEP)
    ft, tt = res["their_fpr"], res["their_tpr"]

    # belief frontiers of the documented backbone family (gpt-5-mini med/high),
    # plus the advanced backbone (gpt-5.4 medium) for the "advanced" variant.
    left_names = ["gpt-5-mini_re-medium", "gpt-5-mini_re-high"]
    adv_name = "gpt-5.4_re-medium"
    load_names = left_names + [adv_name]
    frontier = {c["name"]: c for c in res["configs"] if c["name"] in load_names}
    regime_pts = {
        n: O.load_regime_op_points(SWEEP / n / "results.csv")
        for n in load_names if (SWEEP / n / "results.csv").exists()
    }
    ratio_colors = {
        "decision_u_fp10_fn1": "#08519c", "decision_u_fp5_fn1": "#3182bd",
        "decision_u_fp1_fn1": "#6baed6", "decision_u_fp1_fn5": "#fb6a4a",
        "decision_u_fp1_fn10": "#a50f15",
    }
    ratio_lbl = {"decision_u_fp10_fn1": ".1", "decision_u_fp5_fn1": ".2",
                 "decision_u_fp1_fn1": "1", "decision_u_fp1_fn5": "5",
                 "decision_u_fp1_fn10": "10"}

    # -----------------------------------------------------------------------
    # FIGURE 1: ROC of the deployed tool against the belief frontier.
    # We render two versions: a record with both medium/high frontiers, and the
    # paper version with only the medium frontier and all-filled markers.
    # -----------------------------------------------------------------------
    fam_col = {"gpt-5-mini_re-medium": "#3182bd", "gpt-5-mini_re-high": "#08306b",
               "gpt-5.4_re-medium": "#e6550d"}

    def render_roc(names, out_name, show_style_legend, force_filled,
                   title="Why The Deployed Tool Under-Triaged"):
        fig, ax = plt.subplots(figsize=(7.6, 7.0))
        for n in names:
            c = frontier.get(n)
            if c is None:
                continue
            fam, eff = parse(n)
            ax.plot(c["fpr"], c["tpr"], "-", color=fam_col[n], lw=2.3,
                    label=f"Belief Ranking \u2014 {FAMILY_LABEL[fam]} {eff.capitalize()} (AUC={c['auc']:.2f})")
        ax.plot([0, 1], [0, 1], "k:", alpha=0.5)

        for n in names:
            pts = regime_pts.get(n, {})
            filled = force_filled or n.endswith("high")
            for regime, col in ratio_colors.items():
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

        ax.plot(ft, tt, "*", color="crimson", markersize=24, markeredgecolor="k",
                markeredgewidth=0.9, zorder=7)

        star_handle = Line2D([], [], marker="*", linestyle="none", color="crimson",
                             markeredgecolor="k", markeredgewidth=0.6, markersize=17,
                             label=f"Deployed Tool (Sens={tt:.0%}, Over-Triage={ft:.0%})")
        ratio_handles = [
            Line2D([], [], marker="o", linestyle="none", markeredgecolor=col,
                   markerfacecolor=col, markersize=8,
                   label=f"Prompt (FN/FP = {ratio_lbl[reg]})")
            for reg, col in ratio_colors.items()
        ]
        ratio_handles.append(
            Line2D([], [], marker="o", linestyle="none", markeredgecolor="k",
                   markerfacecolor="k", markersize=8, label="No Utility (Default)"))
        style_handles = []
        if show_style_legend:
            style_handles = [
                Line2D([], [], marker="o", linestyle="none", markeredgecolor="0.3",
                       markerfacecolor="none", markersize=8, label="Medium Reasoning (Open)"),
                Line2D([], [], marker="o", linestyle="none", markeredgecolor="0.3",
                       markerfacecolor="0.3", markersize=8, label="High Reasoning (Filled)"),
            ]
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=16)
        ax.set_ylabel("True Positive Rate", fontsize=16)
        ax.set_title(title, fontsize=17,
                     fontweight="bold")
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, alpha=0.3)
        roc_handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=roc_handles + [star_handle] + ratio_handles + style_handles,
                  fontsize=12.5, loc="lower right", ncol=1, framealpha=0.92)
        fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_name}")

    # record version: both frontiers, medium=open / high=filled distinction
    render_roc(left_names, "original_deployed_roc.png",
               show_style_legend=True, force_filled=False)
    # paper version: medium frontier only, all markers filled, no style legend
    render_roc(["gpt-5-mini_re-medium"], "fig1_deployed_roc.png",
               show_style_legend=False, force_filled=True,
               title="GPT-Health Under-Triage vs Utility Prompting (GPT-5 Mini)")
    # advanced version: same layout but gpt-5.4 medium frontier + operating points
    render_roc([adv_name], "fig1_deployed_roc_advanced.png",
               show_style_legend=False, force_filled=True,
               title="GPT-Health Under-Triage vs Utility Prompting (GPT-5.4)")


# ===========================================================================
# FIGURE 1b -- default (unspecified) utilities, one row per reasoning tier
# ===========================================================================
def figure1b(configs_by_name: dict, config_order=None, out_dir=None,
             tier_fn=None, tier_labels=None) -> None:
    """Number-line of each model's default recovered FN/FP, grouped by reasoning
    tier. ``config_order`` selects which configs to show. ``tier_fn`` maps a
    config to its y-row index and ``tier_labels`` names those rows; both default
    to the full three-tier layout keyed on each level's ordinal rank within its
    family."""
    if config_order is None:
        config_order = CONFIG_ORDER
    if out_dir is None:
        out_dir = OUT
    if tier_fn is None:
        tier_fn = lambda n: reason_rank(*parse(n))  # noqa: E731
    if tier_labels is None:
        tier_labels = TIER_LABELS

    names = [n for n in config_order if n in configs_by_name]
    # map the tiers that are actually present to contiguous y rows
    ranks_present = sorted({tier_fn(n) for n in names})
    row_of = {r: i for i, r in enumerate(ranks_present)}
    top_row = len(ranks_present) - 1

    fig, axn = plt.subplots(figsize=(8.2, 4.2))
    ratios = []
    hi_edges = []
    for n in names:
        cfg = configs_by_name[n]
        r = cfg["baseline_ratio"]
        if r is None:
            continue
        ratios.append(r)
        fam, reason = parse(n)
        row = row_of[tier_fn(n)]
        ci = A.bootstrap_ratio_ci(cfg["belief"], cfg["decisions"],
                                  "decision_baseline")
        if ci is not None:
            lo, hi_ci = ci
            hi_edges.append(hi_ci)
            axn.errorbar(r, row,
                         xerr=[[max(r - lo, 0.0)], [max(hi_ci - r, 0.0)]],
                         fmt="none", ecolor=FAMILY_CMAP[fam](0.55),
                         elinewidth=1.4, capsize=3.5, capthick=1.4, zorder=4)
        axn.plot(r, row, marker=FAMILY_MARKER[fam],
                 markersize=15, color=FAMILY_CMAP[fam](0.85),
                 markeredgecolor="k", markeredgewidth=0.7, zorder=5)
    axn.axvline(1.0, color="0.6", ls="--", lw=1.0, zorder=1)
    hi = max(ratios + hi_edges) * 1.18
    axn.set_xscale("log")
    axn.set_xlim(0.9, hi)
    axn.set_ylim(-0.6, top_row + 1.1)
    axn.set_yticks(list(range(len(ranks_present))))
    axn.set_yticklabels([tier_labels[r] for r in ranks_present], fontsize=13)
    axn.set_ylabel("Reasoning Level", fontsize=15)
    for side in ("right", "top"):
        axn.spines[side].set_visible(False)
    axn.xaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 7, 10]))
    axn.xaxis.set_major_formatter(FixedFormatter(["1", "2", "3", "5", "7", "10"]))
    axn.xaxis.set_minor_locator(NullLocator())
    axn.tick_params(axis="x", labelsize=13)
    axn.set_xlabel("Default Recovered FN/FP", fontsize=16)
    axn.grid(True, axis="x", alpha=0.25)
    axn.text(0.92, top_row + 0.82, "Resource Priority", fontsize=13, color="0.4",
             ha="left", va="center")
    axn.text(hi * 0.99, top_row + 0.82, "Safety Priority", fontsize=13,
             color="0.4", ha="right", va="center")

    # family + reasoning legend
    fam_handles = [
        Line2D([], [], marker=FAMILY_MARKER[f], linestyle="none",
               markerfacecolor=FAMILY_CMAP[f](0.85), markeredgecolor="k",
               markersize=11, label=FAMILY_LABEL[f])
        for f in FAMILY_ORDER
    ]
    axn.legend(handles=fam_handles, fontsize=12, loc="center left",
               bbox_to_anchor=(1.02, 0.5), ncol=1,
               framealpha=0.9, handletextpad=0.3,
               title="Model Family", title_fontsize=13)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig1b_default_utilities.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'fig1b_default_utilities.png'}")


# ===========================================================================
# FIGURE 2 -- recovered vs prompted, capability gradient
# ===========================================================================
def figure2(configs_by_name: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.2), sharex=True, sharey=True)
    xlims = [0.02, 20]
    yvals = [
        config["recovered"][regime]
        for name in CONFIG_ORDER
        if (config := configs_by_name.get(name)) is not None
        for regime in A.UTILITY_REGIMES
        if config["recovered"][regime] is not None
    ]
    ylo = min([0.05] + [value for value in yvals if value > 0]) * 0.8
    yhi = max([20] + yvals) * 1.1
    panels = [
        ("GPT models", {"gpt-5-mini", "gpt-5.4"}),
        ("DeepSeek models", {"DeepSeek-V4-Flash", "DeepSeek-V4-Pro"}),
        ("Claude models", {"Claude-Fable-5", "Claude-Sonnet-5"}),
    ]

    for ax, (title, families) in zip(axes, panels):
        if families:
            ax.plot(xlims, xlims, "k--", alpha=0.55, lw=1.4, zorder=1,
                    label="Perfect Following")
        for name in CONFIG_ORDER:
            family, _ = parse(name)
            if family not in families:
                continue
            config = configs_by_name.get(name)
            if config is None:
                continue
            st = style(name)
            xs, ys = [], []
            for regime in A.UTILITY_REGIMES:
                if config["recovered"][regime] is not None:
                    xs.append(A.TARGET_RATIO[regime])
                    ys.append(config["recovered"][regime])
            ax.plot(xs, ys, marker=st["marker"], ls=st["ls"],
                    color=st["color"], lw=2.0, markersize=8,
                    markeredgecolor="k", markeredgewidth=0.5,
                    label=st["label"], zorder=4)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlims)
        ax.set_ylim(ylo, yhi)
        ax.xaxis.set_major_locator(FixedLocator([0.1, 0.2, 1, 5, 10]))
        ax.xaxis.set_major_formatter(
            FixedFormatter([".1", ".2", "1", "5", "10"])
        )
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 1, 5, 10]))
        ax.yaxis.set_major_formatter(
            FixedFormatter([".1", ".2", "1", "5", "10"])
        )
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
    fig.savefig(OUT / "fig2_recovered_capability.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_recovered_capability.png")


# ===========================================================================
# FIGURE 3 -- ROC grid, restyled to match the paper
# ===========================================================================
def figure3(configs_by_name: dict) -> None:
    regime_style = {
        "decision_baseline":   ("No Utility (Default)", "#000000"),
        "decision_u_fp10_fn1": ("Prompt (FN/FP = .1)", "#08519c"),
        "decision_u_fp5_fn1":  ("Prompt (FN/FP = .2)", "#3182bd"),
        "decision_u_fp1_fn1":  ("Prompt (FN/FP = 1)", "#6baed6"),
        "decision_u_fp1_fn5":  ("Prompt (FN/FP = 5)", "#fb6a4a"),
        "decision_u_fp1_fn10": ("Prompt (FN/FP = 10)", "#a50f15"),
    }
    for slug, provider_label, families in PROVIDER_GROUPS:
        names = provider_config_names(CONFIG_ORDER, configs_by_name, families)
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
            _thr, _acc, btpr, bfpr, _ratio = A.best_fixed_threshold(
                c["belief"], c["gold"]
            )
            if not np.isnan(bfpr):
                ax.plot(bfpr, btpr, "s", color="gold", markersize=20,
                        markeredgecolor="k", markeredgewidth=1.0, zorder=6,
                        label="Best Fixed Utility" if i == 0 else None)
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
        out = OUT / f"fig3_roc_grid_{slug}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.name}")

def _ratio_lbl(v: float) -> str:
    if v >= 1:
        return str(int(round(v)))
    return (("%.2f" % v).lstrip("0").rstrip("0")) or "0"


# ===========================================================================
# CONSISTENCY FIGURES -- rescoped to CONFIG_ORDER with the paper's styling
#   (decision<->belief+estimated-utility, and agreement with ChatGPT Health)
# ===========================================================================
def figure_belief_consistency(configs_by_name: dict) -> None:
    """decision_belief_consistency.png: % of decisions reproduced by the fitted
    (belief + estimated utility) threshold rule, per prompted cost ratio.  Same
    capability-gradient styling as fig2; DeepSeek included, GPT-5.4-Mini excluded."""
    names = [n for n in CONFIG_ORDER if n in configs_by_name]
    regimes = A.UTILITY_REGIMES
    x = list(range(len(regimes)))
    fig, ax = plt.subplots(figsize=(8.4, 6.2))

    def est_indicators(c, regime):
        p_star = c["p_star_est"].get(regime)
        if p_star is None:
            return []
        belief = c["belief"]
        out = []
        for ctx, dec in c["decisions"].get(regime, {}).items():
            if ctx in belief:
                rule = 1 if belief[ctx] >= p_star else 0
                out.append(int(rule == dec))
        return out

    for name in names:
        c = configs_by_name[name]
        st = style(name)
        y = [100 * c["est_consistency"][r] for r in regimes]
        elo, ehi = [], []
        for r in regimes:
            yr = c["est_consistency"][r]
            ind = est_indicators(c, r)
            ci = A.bootstrap_prop_ci(ind) if ind else None
            if ci is not None and not np.isnan(yr):
                elo.append(max(100 * yr - 100 * ci[0], 0.0))
                ehi.append(max(100 * ci[1] - 100 * yr, 0.0))
            else:
                elo.append(0.0)
                ehi.append(0.0)
        ax.errorbar(x, y, yerr=[elo, ehi], marker=st["marker"], ls=st["ls"],
                    color=st["color"], lw=2.0, markersize=7,
                    markeredgecolor="k", markeredgewidth=0.5,
                    ecolor=st["color"], elinewidth=1.0, capsize=2.5,
                    capthick=1.0, label=st["label"], zorder=4)
        base = c["est_consistency"].get("decision_baseline")
        if base is not None and not np.isnan(base):
            bind = est_indicators(c, "decision_baseline")
            bci = A.bootstrap_prop_ci(bind) if bind else None
            if bci is not None:
                ax.errorbar(-0.55, 100 * base,
                            yerr=[[max(100 * base - 100 * bci[0], 0.0)],
                                  [max(100 * bci[1] - 100 * base, 0.0)]],
                            fmt="none", ecolor=st["color"], elinewidth=1.0,
                            capsize=2.5, capthick=1.0, clip_on=False, zorder=5)
            ax.plot(-0.55, 100 * base, marker=st["marker"], markersize=9,
                    color=st["color"], markeredgecolor="k", markeredgewidth=0.6,
                    clip_on=False, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([_ratio_lbl(A.TARGET_RATIO[r]) for r in regimes])
    ax.set_xlim(-0.8, len(regimes) - 0.5)
    ax.set_ylim(0, 101)
    ax.set_ylabel("Decisions Matching Belief + Estimated Utility (%)", fontsize=13)
    ax.set_xlabel("Prompted Costs (FN/FP)", fontsize=14)
    ax.set_title("Decisions Are Consistent With Belief + Recovered Utility",
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.text(0.015, 0.03, "marker on y-axis = no-utility baseline",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.legend(fontsize=9.5, loc="center left", bbox_to_anchor=(1.02, 0.5),
              ncol=1, framealpha=0.93)
    fig.tight_layout()
    fig.savefig(OUT / "decision_belief_consistency.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote decision_belief_consistency.png")


def figure_nature_consistency() -> None:
    """nature_decision_consistency.png: % of variants where our admit decision
    matches ChatGPT Health's (Nature Medicine), per prompted cost ratio.  Same
    styling as fig2; DeepSeek included, GPT-5.4-Mini excluded."""
    regimes = O._UTILITY_REGIMES
    labels = O._RATIO_LABELS
    x = list(range(len(regimes)))
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for name in CONFIG_ORDER:
        results_csv = SWEEP / name / "results.csv"
        if not results_csv.exists():
            continue
        their, our = O._load_config_decisions(results_csv)
        st = style(name)
        y, elo, ehi = [], [], []
        for r in regimes:
            dec = our.get(r, {})
            ctxs = [c for c in dec if c in their]
            if ctxs:
                ind = [int(dec[c] == their[c]) for c in ctxs]
                yr = 100 * sum(ind) / len(ind)
                y.append(yr)
                ci = A.bootstrap_prop_ci(ind)
                if ci is not None:
                    elo.append(max(yr - 100 * ci[0], 0.0))
                    ehi.append(max(100 * ci[1] - yr, 0.0))
                else:
                    elo.append(0.0)
                    ehi.append(0.0)
            else:
                y.append(float("nan"))
                elo.append(0.0)
                ehi.append(0.0)
        ax.errorbar(x, y, yerr=[elo, ehi], marker=st["marker"], ls=st["ls"],
                    color=st["color"], lw=2.0, markersize=7,
                    markeredgecolor="k", markeredgewidth=0.5,
                    ecolor=st["color"], elinewidth=1.0, capsize=2.5,
                    capthick=1.0, label=st["label"], zorder=4)
        dec = our.get("decision_baseline", {})
        ctxs = [c for c in dec if c in their]
        if ctxs:
            ind = [int(dec[c] == their[c]) for c in ctxs]
            base = 100 * sum(ind) / len(ind)
            bci = A.bootstrap_prop_ci(ind)
            if bci is not None:
                ax.errorbar(-0.55, base,
                            yerr=[[max(base - 100 * bci[0], 0.0)],
                                  [max(100 * bci[1] - base, 0.0)]],
                            fmt="none", ecolor=st["color"], elinewidth=1.0,
                            capsize=2.5, capthick=1.0, clip_on=False, zorder=5)
            ax.plot(-0.55, base, marker=st["marker"], markersize=9,
                    color=st["color"], markeredgecolor="k", markeredgewidth=0.6,
                    clip_on=False, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.8, len(regimes) - 0.5)
    ax.set_ylim(0, 101)
    ax.set_ylabel("Decisions Matching ChatGPT Health (Nature Medicine) (%)",
                  fontsize=13)
    ax.set_xlabel("Prompted Costs (FN/FP)", fontsize=14)
    ax.set_title("Agreement Of Our Decisions With ChatGPT Health", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.text(0.015, 0.03, "marker on y-axis = no-utility baseline",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.legend(fontsize=9.5, loc="center left", bbox_to_anchor=(1.02, 0.5),
              ncol=1, framealpha=0.93)
    fig.tight_layout()
    fig.savefig(OUT / "nature_decision_consistency.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote nature_decision_consistency.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_dirs = sorted(p.parent for p in SWEEP.glob("*/results.csv"))
    configs = [A.analyse_config(d) for d in cfg_dirs]
    by_name = {c["name"]: c for c in configs}
    figure1(by_name)   # deployed-ROC panels (unchanged)
    figure1b(by_name, CONFIG_ORDER, OUT)
    figure2(by_name)
    figure3(by_name)
    figure_belief_consistency(by_name)
    figure_nature_consistency()
    # The canonical paper figures now show the FULL set of prompted utilities /
    # thresholds (seven cost ratios, incl. the .01 and 100 extremes). Regenerate
    # the utility/threshold figures via the full-utility module so it always wins,
    # regardless of which script is run. Deferred import avoids a circular import.
    import make_full_utility_figures as F
    F.main()


if __name__ == "__main__":
    main()
