#!/usr/bin/env python
"""Invariance figure: intervals coloured by individual and multiplicity-corrected significance.

Reads invariance_p.json, written by supplemental_invariance_p.py."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from supplemental_analysis import CONFIGS
import supplemental_figures as F

P = {tuple(r["cfg"]): r["res"] for r in json.load(open(HERE / "invariance_p.json"))}
NS, SIG, BH = F.DEEMPH, "#eda100", "#b02e0c"      # validated: deutan dE 26.5, normal dE 28.7
FACS = [("race", "Black /\nWhite"), ("gender", "Woman /\nMan"),
        ("has_anchor", "Anchoring /\nnone"), ("has_barrier", "Barrier /\nnone")]


def build():
    fig, axes = plt.subplots(1, 4, figsize=(6.2, 4.6), sharey=True, gridspec_kw=dict(wspace=0.1))
    for i, (ax, (fac, title)) in enumerate(zip(axes, FACS)):
        ax.axvline(1, color=F.AXIS, lw=0.8, zorder=1)
        for cfg in CONFIGS:
            r = P[cfg][fac]; y = F.Y[cfg]
            lo, hi, pt = 10 ** r["ci"][0], 10 ** r["ci"][1], 10 ** r["point"]
            if r["p_bh"] < 0.05:
                col, lw, ms = BH, 2.2, 30
            elif not (lo <= 1 <= hi):
                col, lw, ms = SIG, 1.9, 26
            else:
                col, lw, ms = NS, 1.1, 18
            ax.plot([lo, hi], [y, y], color=col, lw=lw, solid_capstyle="round", zorder=2)
            ax.scatter([pt], [y], s=ms, color=col, edgecolor=F.SURF, lw=0.8, zorder=3)
        F.config_axis(ax, labels=(i == 0))
        ax.set_xscale("log"); ax.set_xlim(0.3, 3.5)
        ax.set_xticks([0.5, 1, 2]); ax.set_xticklabels(["0.5", "1", "2"])
        ax.set_title(title, fontsize=7.5)
    fig.supxlabel("Ratio of the two recovered FN:FP cost ratios, level 1 / level 2 "
                  "(1 = identical priorities); 95% scenario-cluster intervals",
                  fontsize=7.5, color=F.INK2, y=0.02)
    fig.legend(handles=[
        Line2D([], [], color=NS, lw=1.1, marker="o", markersize=3.4, label="interval includes 1"),
        Line2D([], [], color=SIG, lw=1.9, marker="o", markersize=4.0, label="interval excludes 1"),
        Line2D([], [], color=BH, lw=2.2, marker="o", markersize=4.4,
               label="excludes 1 after correction for the 72 tests")],
        loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.035))
    F.save(fig, "fig_invariance")
    print("wrote fig_invariance")


if __name__ == "__main__":
    build()
