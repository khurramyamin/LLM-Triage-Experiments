#!/usr/bin/env python
"""Figures for the further checks of the recovered decision rule (reads consistency.json, matched_pairs.json and
results_cache.pkl; run supplemental_consistency.py and supplemental_pairs.py first)."""
import json, pickle, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from supplemental_analysis import CONFIGS, REGIMES, CELLS
import supplemental_figures as F

N = {tuple(r["cfg"]): r for r in json.load(open(HERE / "consistency.json"))}
P = {tuple(r["cfg"]): r for r in json.load(open(HERE / "matched_pairs.json"))["by_config"]}
R = {tuple(r["cfg"]): r for r in pickle.load(open(HERE / "results_cache.pkl", "rb"))}
LAB = [F.LABEL[c] for c in CONFIGS]
RTICK = ["none", ".01", ".1", ".2", "1", "5", "10", "100"]
BLUES = LinearSegmentedColormap.from_list("b", ["#f3f7fd", "#9ec5f4", "#2a78d6", "#14457c"])
DIVER = LinearSegmentedColormap.from_list("d", ["#b1440f", "#eb6834", "#f7f6f2", "#2a78d6", "#14457c"])


def family_gaps(ax, n_cols):
    for k in (3, 6, 9, 12, 15):
        ax.axhline(k - 0.5, color="white", lw=2.2)


def heat(ax, M, cmap, norm, fmt, labels=True, text_dark=0.62):
    ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            shade = norm(v) if not isinstance(norm, TwoSlopeNorm) else abs(norm(v) - 0.5) * 2
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=5.4,
                    color="white" if shade > text_dark else F.INK)
    ax.set_yticks(range(len(LAB))); ax.set_yticklabels(LAB if labels else [], fontsize=6.4)
    ax.set_xticks(range(M.shape[1])); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    family_gaps(ax, M.shape[1])


# 1 ---------------------------------------------------------------- threshold rule vs majority baseline
def fig_baseline():
    gain = np.array([[N[c]["holdout"][g]["rule_acc"] - N[c]["holdout"][g]["majority_acc"] for g in REGIMES] for c in CONFIGS])
    bal = np.array([[np.nan if N[c]["holdout"][g]["rule_bal"] is None else N[c]["holdout"][g]["rule_bal"] for g in REGIMES] for c in CONFIGS])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.0), gridspec_kw=dict(wspace=0.06))
    heat(axes[0], gain, DIVER, TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5), lambda v: f"{v:+.2f}")
    heat(axes[1], bal, BLUES, Normalize(0.5, 1.0), lambda v: f"{v:.2f}", labels=False)
    for ax, t in zip(axes, ["Threshold rule minus majority baseline\n(absolute accuracy)",
                            "Threshold rule, balanced accuracy\n(majority baseline = 0.50)"]):
        ax.set_xticklabels(RTICK, fontsize=6.4); ax.set_title(t, fontsize=7.2, color=F.INK)
        ax.set_xlabel("Prompted FN:FP cost ratio (none = no stated priority)", fontsize=6.6, color=F.INK2)
    F.save(fig, "fig_majority_baseline")


# 2 ---------------------------------------------------------------- matched-pair referral switches
def fig_pairs():
    facs = [("race", "Black", "White"), ("gender", "Woman", "Man"),
            ("has_anchor", "Anchoring", "None"), ("has_barrier", "Barrier", "None")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 4.9), sharey=True, gridspec_kw=dict(wspace=0.12))
    for i, (ax, (fac, a, b)) in enumerate(zip(axes, facs)):
        ax.axvline(0, color=F.AXIS, lw=0.8, zorder=1)
        for cfg in CONFIGS:
            p = P[cfg][f"default|{fac}"]; y = F.Y[cfg]
            r1, r2 = 100 * p["only_l1"] / p["n"], 100 * p["only_l2"] / p["n"]
            ax.barh(y, r1, height=0.62, color=F.BLUE_LT, zorder=2)
            ax.barh(y, -r2, height=0.62, color=F.ORANGE_LT, zorder=2)
            q = p; lo, hi = 100 * q["ci"][0], 100 * q["ci"][1]
            clear = not (lo <= 0 <= hi)
            ax.plot([lo, hi], [y, y], color=F.INK, lw=0.9, solid_capstyle="butt", zorder=3)
            ax.scatter([100 * q["diff"]], [y], s=13, color=F.INK if clear else F.SURF, edgecolor=F.INK, lw=0.8, zorder=4)
        F.config_axis(ax, labels=(i == 0)); ax.grid(False)
        ax.set_xlim(-16, 16); ax.set_xticks([-10, -5, 0, 5, 10]); ax.set_xticklabels(["10", "5", "0", "5", "10"], fontsize=6.4)
        ax.set_title(f"{a} vs {b.lower()}", fontsize=7.4, pad=22)
        ax.text(0.98, 1.01, f"only {a.lower()}\nreferred", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=F.BLUE_DK)
        ax.text(0.02, 1.01, f"only {b.lower()}\nreferred", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.6, color="#b1440f")
    fig.supxlabel("Matched pairs (same vignette, other factors fixed) whose default decisions differ, % of pairs",
                  fontsize=7.2, color=F.INK2, y=0.065)
    fig.legend(handles=[Line2D([], [], marker="o", color=F.INK, markerfacecolor=F.INK, markersize=4, lw=0.9,
                               label="net difference, 95% interval excludes 0"),
                        Line2D([], [], marker="o", color=F.INK, markerfacecolor=F.SURF, markersize=4, lw=0.9,
                               label="net difference, 95% interval includes 0")],
               loc="lower center", ncol=2, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(bottom=0.135)
    F.save(fig, "fig_matched_pairs")


# 3 ---------------------------------------------------------------- all 16 factorial cells
def fig_cells():
    order = ["WM", "WW", "BM", "BW", "WM-A", "WW-A", "BM-A", "BW-A",
             "WM-X", "WW-X", "BM-X", "BW-X", "WM-AX", "WW-AX", "BM-AX", "BW-AX"]
    M = np.full((len(CONFIGS), 16), np.nan); X = np.zeros_like(M, bool)
    for i, cfg in enumerate(CONFIGS):
        inv = R[cfg]["invariance"]["decision_baseline"]; ov = inv["overall"]
        for j, cell in enumerate(order):
            v = inv["cells"].get(cell); ci = inv["cell_ci"].get(cell)
            if v and ov:
                M[i, j] = np.log2(v / ov)
                X[i, j] = ci is not None and not (ci[0] <= ov <= ci[1])
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.imshow(M, cmap=DIVER, norm=TwoSlopeNorm(vmin=-1.2, vcenter=0, vmax=1.2), aspect="auto")
    yy, xx = np.where(X)
    ax.scatter(xx, yy, s=9, facecolor="none", edgecolor=F.INK, lw=0.8, zorder=3)
    ax.set_yticks(range(len(LAB))); ax.set_yticklabels(LAB, fontsize=6.4)
    ticks = []
    for c in order:
        base, _, extra = c.partition("-")
        who = ("White " if base[0] == "W" else "Black ") + ("man" if base[1] == "M" else "woman")
        ticks.append(who)
    ax.set_xticks(range(16)); ax.set_xticklabels(ticks, fontsize=5.6, rotation=90)
    for k, t in enumerate(["no statement", "anchoring", "barrier", "anchoring + barrier"]):
        ax.text(4 * k + 1.5, -1.1, t, ha="center", va="bottom", fontsize=6.4, color=F.INK2)
        if k:
            ax.axvline(4 * k - 0.5, color="white", lw=2.2)
    for k in (3, 6, 9, 12, 15):
        ax.axhline(k - 0.5, color="white", lw=2.2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=DIVER, norm=TwoSlopeNorm(vmin=-1.2, vcenter=0, vmax=1.2))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.015, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["half", "same", "double"], fontsize=6); cb.outline.set_visible(False)
    cb.set_label("cell ratio relative to the configuration's overall ratio", fontsize=6.4, color=F.INK2)
    ax.set_ylim(len(LAB) - 0.5, -1.6)
    F.save(fig, "fig_cells")


# 4 ---------------------------------------------------------------- split vignettes and systematic misfit
def fig_misfit():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 4.6), sharey=True, gridspec_kw=dict(wspace=0.14))
    ax = axes[0]; ax.axvline(0.5, color=F.AXIS, lw=0.8, zorder=1)
    for cfg in CONFIGS:
        s = N[cfg]["splits"]
        ax.scatter([s["concordance"]], [F.Y[cfg]], s=24, color=F.BLUE, edgecolor=F.SURF, lw=0.7, zorder=3)
    F.config_axis(ax); ax.set_xlim(0.4, 0.9); ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_title("Within split vignettes:\nreferred variant has higher p", fontsize=7.2)
    ax.set_xlabel("concordance (0.5 = no relation)", fontsize=6.6)
    ax = axes[1]
    for cfg in CONFIGS:
        s = N[cfg]["splits"]; y = F.Y[cfg]
        ax.plot([s["split_closeness"], s["unanimous_closeness"]], [y, y], color=F.GRID, lw=1.4, zorder=1)
        ax.scatter([s["unanimous_closeness"]], [y], s=18, color=F.BLUE_LT, edgecolor=F.SURF, lw=0.6, zorder=2)
        ax.scatter([s["split_closeness"]], [y], s=22, color=F.BLUE, edgecolor=F.SURF, lw=0.7, zorder=3)
    F.config_axis(ax, labels=False); ax.set_xlim(0.15, 0.5); ax.set_xticks([0.2, 0.3, 0.4, 0.5])
    ax.set_title("Distance from the fitted threshold", fontsize=7.2)
    ax.set_xlabel("mean |P(refer) - 0.5| under the rule", fontsize=6.6)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=F.BLUE, markersize=4.5, label="split vignettes"),
                       Line2D([], [], marker="o", ls="", color=F.BLUE_LT, markersize=4.5, label="unanimous vignettes")],
              loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=1, fontsize=5.8, frameon=False)
    ax = axes[2]
    for cfg in CONFIGS:
        r = N[cfg]["residual"]; y = F.Y[cfg]; lo, hi = r["vignette_share_null"]
        ax.plot([lo, hi], [y, y], color=F.DEEMPH, lw=4.5, solid_capstyle="butt", zorder=1)
        ax.scatter([r["vignette_share"]], [y], s=24, color=F.ORANGE, edgecolor=F.SURF, lw=0.7, zorder=3)
    F.config_axis(ax, labels=False); ax.set_xlim(0, 0.65); ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_title("Misfit shared within a vignette", fontsize=7.2)
    ax.set_xlabel("between-vignette share of misfit", fontsize=6.6)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=F.ORANGE, markersize=4.5, label="observed"),
                       Patch(color=F.DEEMPH, label="if departures were noise (95%)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=1, fontsize=5.8, frameon=False)
    F.save(fig, "fig_misfit")


if __name__ == "__main__":
    for f in (fig_baseline, fig_pairs, fig_cells, fig_misfit):
        f(); print("wrote", f.__name__)
