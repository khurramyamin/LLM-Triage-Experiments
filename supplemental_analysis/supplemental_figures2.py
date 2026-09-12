#!/usr/bin/env python
"""Condensed figures for the supplemental analyses (overrides fig_holdout / fig_recalibration; adds fig_stability)."""
from supplemental_figures import *  # noqa: F401,F403  (module-level data, palette, helpers)
from supplemental_figures import R, J, A, Y, CONFIGS, REGIMES, RLAB, BLUE, BLUE_LT, BLUE_MID, BLUE_DK, ORANGE, AQUA, DEEMPH, SURF, AXIS, GRID, INK, INK2, config_axis, save, Line2D, Patch, plt, np


def fig_holdout():
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 4.5), sharey=True, gridspec_kw=dict(wspace=0.12, width_ratios=[1, 1, 1]))
    for i, (ax, reg, t) in enumerate(zip(axes[:2], ["decision_baseline", "decision_u_fp1_fn10"], ["No stated priority", "FN:FP = 10 (safety-weighted)"])):
        for cfg in CONFIGS:
            h = R[cfg]["holdout"][reg]; yy = Y[cfg]
            ax.plot([h["in_sample"], h["held_out"]], [yy, yy], color=BLUE_LT, lw=1.6, zorder=1)
            ax.scatter([h["in_sample"]], [yy], s=20, color=BLUE_LT, edgecolor=SURF, lw=0.8, zorder=2)
            ax.scatter([h["held_out"]], [yy], s=24, color=BLUE, edgecolor=SURF, lw=0.8, zorder=3)
        config_axis(ax, labels=(i == 0)); ax.set_xlim(0.75, 1.02); ax.set_xticks([0.8, 0.9, 1.0]); ax.set_title(t, fontsize=7.5)
        ax.set_xlabel("accuracy in predicting the model's decisions" if i == 0 else "", fontsize=7)
    ax = axes[2]
    for cfg in CONFIGS:
        vals = [R[cfg]["per_regime"][r]["auc"] for r in REGIMES if "auc" in R[cfg]["per_regime"][r]]; yy = Y[cfg]
        ax.plot([min(vals), max(vals)], [yy, yy], color=DEEMPH, lw=1.8, zorder=1)
        ax.scatter([R[cfg]["per_regime"]["decision_baseline"]["auc"]], [yy], s=24, color=BLUE, edgecolor=SURF, lw=0.8, zorder=3)
    config_axis(ax, labels=False); ax.set_xlim(0.8, 1.01); ax.set_xticks([0.8, 0.9, 1.0]); ax.set_title("Probability predicts decision", fontsize=7.5); ax.set_xlabel("AUC", fontsize=7)
    fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", markersize=5, label="unseen scenarios (panels 1–2) / no stated priority (panel 3)"),
                        Line2D([], [], marker="o", color=BLUE_LT, ls="", markersize=5, label="seen scenarios, the half used to fit the rule (panels 1–2)"),
                        Line2D([], [], color=DEEMPH, lw=1.8, label="range over the seven prompted cost ratios")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), fontsize=6.3)
    save(fig, "fig_holdout")


def fig_stability():
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 4.4), sharey=True, gridspec_kw=dict(wspace=0.14, width_ratios=[1.3, 1, 1]))
    ax = axes[0]
    for cfg in CONFIGS:
        rp = R[cfg]["repetitions"]; yy = Y[cfg]; rs = [x for x in rp["ratios"] if x]
        ax.scatter(rs, [yy] * len(rs), s=16, color=DEEMPH, edgecolor=SURF, lw=0.6, zorder=2)
        ax.scatter([rp["ratio_mean_belief"]], [yy], s=30, color=BLUE, marker="D", edgecolor=SURF, lw=0.8, zorder=3)
    config_axis(ax); ax.set_xscale("log"); ax.set_xlim(0.8, 10); ax.set_xticks([1, 2, 5, 10]); ax.set_xticklabels(["1", "2", "5", "10"])
    ax.set_title("Default ratio, five belief runs", fontsize=7.5); ax.set_xlabel("recovered FN:FP", fontsize=7)
    ax.legend(handles=[Line2D([], [], marker="o", color=DEEMPH, ls="", markersize=4.5, label="one belief run"), Line2D([], [], marker="D", color=BLUE, ls="", markersize=5, label="five-run mean")], loc="lower left", fontsize=6.3)
    ax = axes[1]
    for cfg in CONFIGS:
        ax.barh(Y[cfg], A[f"{cfg[0]}|{cfg[1]}"]["mean_within_vignette_sd"], height=0.55, color=BLUE, zorder=2)
    config_axis(ax, labels=False); ax.set_xlim(0, 0.15); ax.set_xticks([0, 0.05, 0.1, 0.15]); ax.set_title("Spread across 16 variants", fontsize=7.5); ax.set_xlabel("within-vignette SD of $p$", fontsize=7)
    ax = axes[2]
    for cfg in CONFIGS:
        ax.barh(Y[cfg], A[f"{cfg[0]}|{cfg[1]}"]["flip_rate"], height=0.55, color=ORANGE, zorder=2)
    config_axis(ax, labels=False); ax.set_xlim(0, 0.12); ax.set_xticks([0, 0.05, 0.1]); ax.set_xticklabels(["0%", "5%", "10%"]); ax.set_title("Decision noise", fontsize=7.5); ax.set_xlabel("same-$p$ variant pairs with\ndifferent default decisions", fontsize=7)
    save(fig, "fig_stability")


def fig_recalibration():
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    ax.axvline(1, color=AXIS, lw=0.8, zorder=1)
    for cfg in CONFIGS:
        rc = R[cfg]["recalibration"]; yy = Y[cfg]
        ax.plot([rc["ratio"], rc["effective_ratio"]], [yy, yy], color=BLUE_LT, lw=1.6, zorder=1)
        ax.scatter([rc["ratio"]], [yy], s=22, color=BLUE_LT, edgecolor=SURF, lw=0.8, zorder=2)
        ax.scatter([rc["effective_ratio"]], [yy], s=26, color=BLUE, edgecolor=SURF, lw=0.8, zorder=3)
    config_axis(ax); ax.set_xscale("log"); ax.set_xlim(0.1, 10); ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    ax.set_xlabel("FN:FP cost ratio implied by the default threshold")
    fig.legend(handles=[Line2D([], [], marker="o", color=BLUE_LT, ls="", markersize=5, label="nominal: on the model's own probability scale"),
                        Line2D([], [], marker="o", color=BLUE, ls="", markersize=5, label="effective: at the observed emergency frequency")], loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.03), fontsize=6.5)
    save(fig, "fig_recalibration")


if __name__ == "__main__":
    for f in (fig_holdout, fig_stability):
        f(); print("wrote", f.__name__)
