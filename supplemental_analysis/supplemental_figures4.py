#!/usr/bin/env python
"""Per-model subgroup differences: six panels, three reasoning levels each, with intervals.

Reads subgroup_ci.json, written by supplemental_subgroup_ci.py."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from supplemental_analysis import FAMILIES, CONFIGS
import supplemental_figures as F   # palette, save(), rcParams

CI = {tuple(r["cfg"]): r["res"] for r in json.load(open(HERE / "subgroup_ci.json"))}
SHADE = [F.BLUE, F.ORANGE, F.AQUA]   # categorical hues in fixed order, validated for CVD separation
FACS = ["race", "gender", "has_anchor", "has_barrier"]
FLAB = ["Race", "Sex", "Anchor", "Barrier"]
XS = [0, 1, 2, 3]; XO = [5, 6, 7, 8]
REG = "decision_baseline"


def build(kind="prim", fname="fig_subgroups"):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.9), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.30, wspace=0.10))
    for ax, (model, efforts, disp) in zip(axes.ravel(), FAMILIES):
        ax.axhline(0, color=F.AXIS, lw=0.9, zorder=1)
        ax.axvline(4.0, color=F.GRID, lw=1.0, zorder=0)
        for k, eff in enumerate(efforts):
            cfg = (model, eff); res = CI[cfg]
            for xs, metric in ((XS, "sens"), (XO, "over")):
                pts, los, his = [], [], []
                for fac in FACS:
                    p, lo, hi = res[f"{kind}|{REG}|{fac}|{metric}"]["variant"]
                    pts.append(p); los.append(p - lo); his.append(hi - p)
                x = [v + (k - 1) * 0.24 for v in xs]
                ax.errorbar(x, pts, yerr=[los, his], fmt="none", ecolor=SHADE[k], elinewidth=1.0,
                            capsize=0, alpha=0.85, zorder=2)
                ax.plot(x, pts, color=SHADE[k], lw=1.2, zorder=3)
                ax.scatter(x, pts, s=13, color=SHADE[k], edgecolor=F.SURF, lw=0.6, zorder=4)
        ax.set_title(disp, fontsize=8, color=F.INK)
        ax.set_xticks(XS + XO); ax.set_xticklabels(FLAB + FLAB, fontsize=6, rotation=90)
        ax.set_xlim(-0.8, 8.8); ax.set_ylim(-0.38, 0.50)
        ax.set_yticks([-0.3, -0.15, 0, 0.15, 0.3, 0.45])
        ax.tick_params(labelsize=6.5, length=2); ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("difference between\nthe two levels", fontsize=7, color=F.INK2)
    for ax in axes.ravel():
        ax.text(1.5, 0.455, "sensitivity", fontsize=6, ha="center", color=F.MUTED)
        ax.text(6.5, 0.455, "over-triage", fontsize=6, ha="center", color=F.MUTED)
    fig.legend(handles=[Line2D([], [], color=SHADE[i], lw=1.2, marker="o", markersize=3.6,
                               label=l) for i, l in enumerate(
                   ["lowest reasoning setting", "middle setting", "highest setting"])],
               loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    F.save(fig, fname)
    print("wrote", fname)


if __name__ == "__main__":
    build("prim", "fig_subgroups")
