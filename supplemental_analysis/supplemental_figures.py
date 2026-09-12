#!/usr/bin/env python
"""Figures for the supplemental analyses. Reads the cached analysis (results_cache.pkl, results.json,
addendum.json); queries no model. Writes PDF and PNG versions into figures/."""
import json, pickle, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from supplemental_analysis import CONFIGS, FAMILIES, DISPLAY, REGIMES, RLAB, FACTORS, CELLS, D_ORIG, load_original, load_config
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
R = {tuple(r["cfg"]): r for r in pickle.load(open(HERE / "results_cache.pkl", "rb"))}
J = json.load(open(HERE / "results.json")); A = json.load(open(HERE / "addendum.json"))
scen, diag, deployed = load_original()

# palette (reference instance of the data-viz skill; validated)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_LT, BLUE_MID, BLUE_DK, ORANGE_LT = "#9ec5f4", "#3987e5", "#1c5cab", "#f7bfa3"
INK, INK2, MUTED, GRID, AXIS, SURF, DEEMPH = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff", "#b9b8b1"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("divbr", ["#1c5cab", "#6da7ec", "#f0efec", "#ee9b9a", "#c1302f"])
plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "text.color": INK,
                     "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "legend.fontsize": 7,
                     "pdf.fonttype": 42, "savefig.dpi": 200, "axes.titlecolor": INK, "axes.titleweight": "medium"})
LABEL = {cfg: f"{DISPLAY[cfg[0]]} · {cfg[1]}" for cfg in CONFIGS}
REGX = list(range(8)); REGT = [RLAB[r] for r in REGIMES]

ypos, seps, y = [], [], 0.0
for m, es, d in FAMILIES:
    for e in es:
        ypos.append(y); y += 1.0
    seps.append(y - 0.5 + 0.35); y += 0.7
seps = seps[:-1]
Y = dict(zip(CONFIGS, ypos))


def config_axis(ax, labels=True):
    ax.set_yticks(ypos)
    if labels:
        ax.set_yticklabels([LABEL[c] for c in CONFIGS])
    ax.set_ylim(max(ypos) + 0.9, -0.9)
    for s in seps:
        ax.axhline(s, color=GRID, lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", zorder=0); ax.set_axisbelow(True); ax.tick_params(length=2)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=150); plt.close(fig)


# 1 ---------------------------------------------------------------- hold-out dumbbells
def fig_holdout():
    regs = ["decision_baseline", "decision_u_fp10_fn1", "decision_u_fp1_fn1", "decision_u_fp1_fn10"]
    titles = ["No stated priority", "FN:FP = 0.1", "FN:FP = 1", "FN:FP = 10"]
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 4.6), sharey=True, gridspec_kw=dict(wspace=0.12))
    for i, (ax, reg, t) in enumerate(zip(axes, regs, titles)):
        for cfg in CONFIGS:
            h = R[cfg]["holdout"][reg]; yy = Y[cfg]
            ax.plot([h["majority"], h["held_out"]], [yy, yy], color=BLUE_LT, lw=1.6, zorder=1)
            ax.scatter([h["majority"]], [yy], s=20, color=BLUE_LT, edgecolor=SURF, lw=0.8, zorder=2)
            ax.scatter([h["held_out"]], [yy], s=24, color=BLUE, edgecolor=SURF, lw=0.8, zorder=3)
        config_axis(ax, labels=(i == 0)); ax.set_xlim(0.45, 1.03); ax.set_xticks([0.5, 0.75, 1.0]); ax.set_title(t)
    fig.supxlabel("Share of the model's own decisions reproduced on held-out scenarios", fontsize=7.5, color=INK2, y=0.02)
    fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", markersize=5, label="recovered rule, fitted on the other half of the scenarios"),
                        Line2D([], [], marker="o", color=BLUE_LT, ls="", markersize=5, label="majority-class rate on the held-out half")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    save(fig, "fig_holdout")


# 2 ---------------------------------------------------------------- belief->decision AUC
def fig_auc():
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for cfg in CONFIGS:
        vals = [R[cfg]["per_regime"][r]["auc"] for r in REGIMES if "auc" in R[cfg]["per_regime"][r]]
        yy = Y[cfg]
        ax.plot([min(vals), max(vals)], [yy, yy], color=DEEMPH, lw=1.8, zorder=1)
        ax.scatter([R[cfg]["per_regime"]["decision_baseline"]["auc"]], [yy], s=26, color=BLUE, edgecolor=SURF, lw=0.8, zorder=3)
    config_axis(ax); ax.set_xlim(0.8, 1.005); ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_xlabel("AUC of the elicited probability for predicting the model's own decision")
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", markersize=5, label="no stated priority"),
                       Line2D([], [], color=DEEMPH, lw=1.8, label="range over the seven prompted cost ratios")], loc="lower left")
    save(fig, "fig_auc")


# 3 ---------------------------------------------------------------- invariance forest
def fig_invariance():
    facs = [("race", "Black /\nWhite"), ("gender", "Woman /\nMan"), ("has_anchor", "Anchoring /\nnone"), ("has_barrier", "Barrier /\nnone")]
    fig, axes = plt.subplots(1, 4, figsize=(6.0, 4.6), sharey=True, gridspec_kw=dict(wspace=0.1))
    for i, (ax, (fac, t)) in enumerate(zip(axes, facs)):
        ax.axvline(1, color=AXIS, lw=0.8, zorder=1)
        for cfg in CONFIGS:
            d = R[cfg]["invariance"]["decision_baseline"]["diff_log10"][fac]; yy = Y[cfg]
            lo, hi = 10 ** d["ci"][0], 10 ** d["ci"][1]; pt = 10 ** d["point"]
            excl = not (lo <= 1 <= hi)
            ax.plot([lo, hi], [yy, yy], color=BLUE if excl else BLUE_LT, lw=1.4, zorder=2)
            ax.scatter([pt], [yy], s=22, color=BLUE, edgecolor=SURF, lw=0.8, zorder=3)
        config_axis(ax, labels=(i == 0)); ax.set_xscale("log"); ax.set_xlim(0.3, 3.5); ax.set_xticks([0.5, 1, 2]); ax.set_xticklabels(["0.5", "1", "2"])
        ax.set_title(t, fontsize=7.5)
    fig.supxlabel("Ratio of the two recovered FN:FP cost ratios, level 1 / level 2 (1 = identical priorities); 95% scenario-cluster intervals", fontsize=7.5, color=INK2, y=0.02)
    fig.legend(handles=[Line2D([], [], color=BLUE_LT, lw=1.4, label="interval of the ratio of ratios includes 1"), Line2D([], [], color=BLUE, lw=1.4, label="interval excludes 1")], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    save(fig, "fig_invariance")


# 4 ---------------------------------------------------------------- 16 cells strip
def fig_cells():
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for cfg in CONFIGS:
        v = R[cfg]["invariance"]["decision_baseline"]; yy = Y[cfg]
        lo, hi = v["overall_ci"]
        ax.plot([lo, hi], [yy, yy], color=BLUE, lw=5, alpha=0.18, solid_capstyle="butt", zorder=1)
        cells = [v["cells"][c] for c in CELLS if v["cells"][c]]
        ax.scatter(cells, [yy] * len(cells), s=14, color=DEEMPH, edgecolor=SURF, lw=0.6, zorder=2)
        ax.scatter([v["overall"]], [yy], s=30, color=BLUE, edgecolor=SURF, lw=0.8, zorder=3)
    config_axis(ax); ax.set_xscale("log"); ax.set_xlim(0.5, 12); ax.set_xticks([0.5, 1, 2, 5, 10]); ax.set_xticklabels(["0.5", "1", "2", "5", "10"])
    ax.set_xlabel("Recovered FN:FP cost ratio, no stated priority")
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", markersize=6, label="all 1,248 variants"),
                       Patch(color=BLUE, alpha=0.18, label="95% scenario-cluster interval"),
                       Line2D([], [], marker="o", color=DEEMPH, ls="", markersize=4.5, label="one factorial cell (78 variants), 16 per configuration")], loc="upper left")
    save(fig, "fig_cells")


# 5 ---------------------------------------------------------------- repeated belief elicitation
def fig_repetitions():
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 4.4), sharey=True, gridspec_kw=dict(wspace=0.15, width_ratios=[1.3, 1, 1]))
    ax = axes[0]
    for cfg in CONFIGS:
        rp = R[cfg]["repetitions"]; yy = Y[cfg]
        rs = [x for x in rp["ratios"] if x]
        ax.scatter(rs, [yy] * len(rs), s=16, color=DEEMPH, edgecolor=SURF, lw=0.6, zorder=2)
        ax.scatter([rp["ratio_mean_belief"]], [yy], s=30, color=BLUE, marker="D", edgecolor=SURF, lw=0.8, zorder=3)
    config_axis(ax); ax.set_xscale("log"); ax.set_xlim(0.8, 10); ax.set_xticks([1, 2, 5, 10]); ax.set_xticklabels(["1", "2", "5", "10"])
    ax.set_title("Default recovered ratio"); ax.set_xlabel("FN:FP")
    ax.legend(handles=[Line2D([], [], marker="o", color=DEEMPH, ls="", markersize=4.5, label="one of five belief runs"),
                       Line2D([], [], marker="D", color=BLUE, ls="", markersize=5, label="five-run mean probability")], loc="lower right")
    ax = axes[1]
    for cfg in CONFIGS:
        rp = R[cfg]["repetitions"]; yy = Y[cfg]
        ax.plot([min(rp["auc_expanded"]), max(rp["auc_expanded"])], [yy, yy], color=BLUE_LT, lw=2.2, zorder=1)
        ax.scatter(rp["auc_expanded"], [yy] * 5, s=12, color=BLUE, edgecolor=SURF, lw=0.5, zorder=2)
    config_axis(ax, labels=False); ax.set_xlim(0.82, 0.96); ax.set_xticks([0.85, 0.9, 0.95]); ax.set_title("AUROC, 1,248 cases"); ax.set_xlabel("per belief run")
    ax = axes[2]
    for cfg in CONFIGS:
        rp = R[cfg]["repetitions"]; yy = Y[cfg]
        ax.barh(yy, rp["mean_sd"], height=0.55, color=BLUE, zorder=2)
    config_axis(ax, labels=False); ax.set_xlim(0, 0.1); ax.set_xticks([0, 0.05, 0.1]); ax.set_title("Run-to-run spread"); ax.set_xlabel("mean SD of elicited probability")
    save(fig, "fig_repetitions")


# 6 ---------------------------------------------------------------- belief invariance across variants
def fig_belief_invariance():
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 4.4), sharey=True, gridspec_kw=dict(wspace=0.12, width_ratios=[1, 1.35, 1]))
    ax = axes[0]
    for cfg in CONFIGS:
        ax.barh(Y[cfg], A[f"{cfg[0]}|{cfg[1]}"]["mean_within_vignette_sd"], height=0.55, color=BLUE, zorder=2)
    config_axis(ax); ax.set_xlim(0, 0.15); ax.set_xticks([0, 0.05, 0.1, 0.15]); ax.set_title("Spread across the 16 variants"); ax.set_xlabel("within-vignette SD of elicited probability")
    ax = axes[1]
    keys = [("diff_black_white", "Black − White"), ("diff_woman_man", "Woman − Man"), ("diff_anchor", "Anchor − none"), ("diff_barrier", "Barrier − none"), ("diff_labs", "Labs − none")]
    M = np.array([[A[f"{cfg[0]}|{cfg[1]}"][k] for k, _ in keys] for cfg in CONFIGS])
    norm = TwoSlopeNorm(vmin=-0.09, vcenter=0, vmax=0.09)
    for i, cfg in enumerate(CONFIGS):
        for j in range(5):
            ax.add_patch(plt.Rectangle((j - 0.5 + 0.03, Y[cfg] - 0.5 + 0.03), 0.94, 0.94, color=DIV(norm(M[i, j])), lw=0))
            ax.text(j, Y[cfg], f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=5.6, color=INK if abs(M[i, j]) < 0.06 else SURF)
    config_axis(ax, labels=False); ax.grid(False); ax.set_xlim(-0.5, 4.5); ax.set_xticks(range(5)); ax.set_xticklabels([t for _, t in keys], rotation=35, ha="right")
    ax.set_title("Mean paired difference in elicited probability")
    ax = axes[2]
    for cfg in CONFIGS:
        ax.barh(Y[cfg], A[f"{cfg[0]}|{cfg[1]}"]["flip_rate"], height=0.55, color=ORANGE, zorder=2)
    config_axis(ax, labels=False); ax.set_xlim(0, 0.12); ax.set_xticks([0, 0.05, 0.1]); ax.set_title("Decision noise"); ax.set_xlabel("share of same-probability variant pairs\nwith different default decisions")
    save(fig, "fig_belief_invariance")


# 7 ---------------------------------------------------------------- misses among the 64
def fig_misses():
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 4.8), sharey=True, gridspec_kw=dict(wspace=0.12, width_ratios=[1.4, 0.9, 1]))
    ydep = max(ypos) + 1.3
    ax = axes[0]
    cols = {"E9": BLUE, "F9": BLUE_LT, "E13": ORANGE, "F13": ORANGE_LT}
    names = {"E9": "asthma, with labs (E9)", "F9": "asthma, symptoms only (F9)", "E13": "DKA, with labs (E13)", "F13": "DKA, symptoms only (F13)"}
    rows = [(Y[cfg], R[cfg]["misses"]["by_vig"], R[cfg]["misses"]["missed"]) for cfg in CONFIGS] + [(ydep, J["deployed"]["missed_by_vig"], sum(J["deployed"]["missed_by_vig"].values()))]
    for yy, by, tot in rows:
        left = 0
        for v in D_ORIG:
            if by[v]:
                ax.barh(yy, by[v], left=left, height=0.62, color=cols[v], edgecolor=SURF, lw=0.8, zorder=2); left += by[v]
        if tot:
            ax.text(left + 0.6, yy, str(tot), va="center", fontsize=6.5, color=INK2)
    config_axis(ax); ax.set_ylim(ydep + 0.9, -0.9)
    ax.axhline(ydep - 0.75, color=GRID, lw=0.6); ax.set_xlim(0, 48); ax.set_xticks([0, 16, 32, 48]); ax.set_title("Missed of 64 emergencies,\nno stated priority", fontsize=7.5)
    fig.legend(handles=[Patch(color=cols[v], label=names[v]) for v in D_ORIG], loc="lower center", ncol=2, bbox_to_anchor=(0.3, -0.06), fontsize=6.3)
    ax = axes[1]
    for cfg in CONFIGS:
        for j, v in enumerate(D_ORIG):
            b = R[cfg]["misses"]["belief_by_vig"][v][0]
            ax.add_patch(plt.Rectangle((j - 0.5 + 0.03, Y[cfg] - 0.5 + 0.03), 0.94, 0.94, color=SEQ(b), lw=0))
            ax.text(j, Y[cfg], f"{b:.2f}", ha="center", va="center", fontsize=5.8, color=SURF if b > 0.55 else INK)
    ax.set_ylim(ydep + 0.9, -0.9); ax.grid(False); ax.set_xlim(-0.5, 3.5); ax.set_xticks(range(4)); ax.set_xticklabels(D_ORIG)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0); ax.set_title("Mean elicited\nprobability", fontsize=7.5)
    ax = axes[2]
    shades = {"decision_baseline": BLUE_LT, "decision_u_fp1_fn5": BLUE_MID, "decision_u_fp1_fn10": BLUE_DK}
    lab = {"decision_baseline": "no stated priority", "decision_u_fp1_fn5": "FN:FP = 5", "decision_u_fp1_fn10": "FN:FP = 10"}
    for cfg in CONFIGS:
        m = R[cfg]["misses"]; yy = Y[cfg]
        vals = {"decision_baseline": m["fp"] / m["n_neg"], "decision_u_fp1_fn5": m["fp5"] / m["n_neg5"], "decision_u_fp1_fn10": m["fp10"] / m["n_neg10"]}
        ax.plot([min(vals.values()), max(vals.values())], [yy, yy], color=DEEMPH, lw=1.2, zorder=1)
        for reg, c in shades.items():
            ax.scatter([vals[reg]], [yy], s=20, color=c, edgecolor=SURF, lw=0.7, zorder=3)
    ax.scatter([J["deployed"]["fpr"]], [ydep], s=22, color=INK2, marker="*", zorder=3)
    config_axis(ax, labels=False); ax.set_ylim(ydep + 0.9, -0.9); ax.axhline(ydep - 0.75, color=GRID, lw=0.6); ax.set_xlim(0, 0.5); ax.set_xticks([0, 0.25, 0.5]); ax.set_xticklabels(["0%", "25%", "50%"])
    ax.set_title("Over-triage of 512\nnon-emergencies", fontsize=7.5)
    fig.legend(handles=[Line2D([], [], marker="o", color=shades[r], ls="", markersize=5, label=lab[r]) for r in shades] + [Line2D([], [], marker="*", color=INK2, ls="", markersize=6, label="deployed tool")], loc="lower center", ncol=2, bbox_to_anchor=(0.78, -0.06), fontsize=6.3)
    axes[0].set_yticks(ypos + [ydep]); axes[0].set_yticklabels([LABEL[c] for c in CONFIGS] + ["Deployed tool (published)"])
    save(fig, "fig_misses")


# 8 ---------------------------------------------------------------- three-state small multiples
def fig_threestate():
    fig, axes = plt.subplots(6, 3, figsize=(6.5, 8.2), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.45, wspace=0.12))
    series = [("D", BLUE, "definitive emergencies (192)"), ("CD", ORANGE, "edge cases (448)"), ("nonD", AQUA, "non-emergencies (608)")]
    for fi, (m, es, d) in enumerate(FAMILIES):
        for ei, e in enumerate(es):
            ax = axes[fi, ei]; cfg = (m, e)
            for g, c, _ in series:
                pts = [R[cfg]["three_state"][r][g] for r in REGIMES]
                ax.fill_between(REGX[1:], [p[1][0] for p in pts[1:]], [p[1][1] for p in pts[1:]], color=c, alpha=0.12, lw=0)
                ax.plot(REGX[1:], [p[0] for p in pts[1:]], color=c, lw=1.6, zorder=2)
                ax.errorbar([0], [pts[0][0]], yerr=[[pts[0][0] - pts[0][1][0]], [pts[0][1][1] - pts[0][0]]], color=c, lw=1.0, capsize=2, zorder=2)
                ax.scatter(REGX, [p[0] for p in pts], s=11, color=c, edgecolor=SURF, lw=0.5, zorder=3)
            ax.axvline(0.5, color=GRID, lw=0.6); ax.set_title(LABEL[cfg], fontsize=7.5, pad=3)
            ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 0.5, 1]); ax.set_xticks(REGX); ax.set_xticklabels(["none", ".01", ".1", ".2", "1", "5", "10", "100"], fontsize=6)
            ax.grid(axis="y"); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.supxlabel("Prompted FN:FP cost ratio (none = no stated priority)", fontsize=7.5, color=INK2, y=0.03)
    fig.supylabel("Share referred to emergency care", fontsize=7.5, color=INK2, x=0.04)
    fig.legend(handles=[Line2D([], [], color=c, lw=1.6, marker="o", markersize=3.5, label=l) for _, c, l in series], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995))
    save(fig, "fig_threestate")


# 9 ---------------------------------------------------------------- subgroup differences
def pooled_rates(kind, reg, level):
    tp = sum(R[cfg]["subgroups"][reg][level]["tp" if kind == "prim" else "tpe"] for cfg in CONFIGS); npos = sum(R[cfg]["subgroups"][reg][level]["npos" if kind == "prim" else "npe"] for cfg in CONFIGS)
    fp = sum(R[cfg]["subgroups"][reg][level]["fp" if kind == "prim" else "fpe"] for cfg in CONFIGS); nneg = sum(R[cfg]["subgroups"][reg][level]["nneg" if kind == "prim" else "nne"] for cfg in CONFIGS)
    return tp / npos, fp / nneg


def fig_subgroups():
    facs = [("race", "Black −\nWhite"), ("gender", "Woman −\nMan"), ("has_anchor", "Anchoring −\nnone"), ("has_barrier", "Barrier −\nnone")]
    fig, axes = plt.subplots(2, 4, figsize=(6.0, 3.6), sharex=True, sharey="row", gridspec_kw=dict(hspace=0.35, wspace=0.12))
    for ri, (kind, rl) in enumerate([("prim", "Primary endpoint\n(64 / 512)"), ("exp", "Expanded set\n(192 / 608)")]):
        for ci_, (fac, t) in enumerate(facs):
            ax = axes[ri, ci_]; l1, l2 = [f for f in FACTORS if f[0] == fac][0][1:]
            ds = [pooled_rates(kind, r, f"{fac}={l1}")[0] - pooled_rates(kind, r, f"{fac}={l2}")[0] for r in REGIMES]
            do = [pooled_rates(kind, r, f"{fac}={l1}")[1] - pooled_rates(kind, r, f"{fac}={l2}")[1] for r in REGIMES]
            ax.axhline(0, color=AXIS, lw=0.8); ax.axvline(0.5, color=GRID, lw=0.6)
            ax.plot(REGX[1:], ds[1:], color=BLUE, lw=1.6, zorder=2); ax.scatter(REGX, ds, s=11, color=BLUE, edgecolor=SURF, lw=0.5, zorder=3)
            ax.plot(REGX[1:], do[1:], color=ORANGE, lw=1.6, zorder=2); ax.scatter(REGX, do, s=11, color=ORANGE, edgecolor=SURF, lw=0.5, zorder=3)
            if ri == 0:
                ax.set_title(t, fontsize=7)
            ax.set_xticks(REGX); ax.set_xticklabels(["none", ".01", ".1", ".2", "1", "5", "10", "100"], fontsize=5.5, rotation=90)
            ax.set_ylim(-0.12, 0.12); ax.set_yticks([-0.1, -0.05, 0, 0.05, 0.1]); ax.grid(axis="y"); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if ci_ == 0:
                ax.set_ylabel(rl, fontsize=6.5)
    fig.supxlabel("Prompted FN:FP cost ratio (none = no stated priority)", fontsize=7.5, color=INK2, y=-0.01)
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=1.6, marker="o", markersize=3.5, label="difference in sensitivity (emergencies referred)"),
                        Line2D([], [], color=ORANGE, lw=1.6, marker="o", markersize=3.5, label="difference in over-triage (non-emergencies referred)")], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    save(fig, "fig_subgroups")


# 10 --------------------------------------------------------------- recalibration dumbbell
def fig_recalibration():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.4), sharey=True, gridspec_kw=dict(wspace=0.12, width_ratios=[1.5, 1]))
    ax = axes[0]; ax.axvline(1, color=AXIS, lw=0.8, zorder=1)
    for cfg in CONFIGS:
        rc = R[cfg]["recalibration"]; yy = Y[cfg]
        ax.plot([rc["ratio"], rc["effective_ratio"]], [yy, yy], color=BLUE_LT, lw=1.6, zorder=1)
        ax.scatter([rc["ratio"]], [yy], s=22, color=BLUE_LT, edgecolor=SURF, lw=0.8, zorder=2)
        ax.scatter([rc["effective_ratio"]], [yy], s=26, color=BLUE, edgecolor=SURF, lw=0.8, zorder=3)
    config_axis(ax); ax.set_xscale("log"); ax.set_xlim(0.1, 10); ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    ax.set_xlabel("FN:FP cost ratio implied by the default threshold"); ax.set_title("Nominal versus effective default priority")
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE_LT, ls="", markersize=5, label="nominal: on the model's own probability scale"),
                       Line2D([], [], marker="o", color=BLUE, ls="", markersize=5, label="effective: at the observed emergency frequency")], loc="lower left", fontsize=6.3)
    ax = axes[1]
    for cfg in CONFIGS:
        ax.barh(Y[cfg], R[cfg]["recalibration"]["ece"], height=0.55, color=ORANGE, zorder=2)
    config_axis(ax, labels=False); ax.set_xlim(0, 0.4); ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4]); ax.set_title("Calibration error (ECE)"); ax.set_xlabel("expanded set, 10 bins")
    save(fig, "fig_recalibration")


# 11 --------------------------------------------------------------- per-vignette heatmap
def fig_vignettes():
    meta, _, _ = load_config(CONFIGS[0])
    gold_of = {}
    for c, m in meta.items():
        gold_of[m["vig"]] = m["gold"]
    order = ["D", "C/D", "C", "B/C", "B", "A/B", "A"]
    vigs = sorted(gold_of, key=lambda v: (order.index(gold_of[v]), scen[v], v))
    dep_counts = {}
    for c, m in meta.items():
        dep_counts[m["vig"]] = dep_counts.get(m["vig"], 0) + deployed[c]
    regs = [("decision_baseline", "No stated priority")]
    fig, axes = plt.subplots(1, len(regs), figsize=(4.8, 7.2), squeeze=False); axes = list(axes[0])
    for ai, (ax, (reg, t)) in enumerate(zip(axes, regs)):
        cols = [[R[cfg]["per_vignette"][reg].get(v, (0, 16))[0] for cfg in CONFIGS] for v in vigs]
        names = [LABEL[c] for c in CONFIGS]
        if reg == "decision_baseline":
            cols = [row + [dep_counts[v]] for row, v in zip(cols, vigs)]; names = names + ["Deployed tool (published)"]
        M = np.array(cols, float)
        ax.imshow(M, cmap=SEQ, vmin=0, vmax=16, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=5.5)
        ax.set_title(t, fontsize=8)
        # group separators and labels
        starts = {}
        for i, v in enumerate(vigs):
            starts.setdefault(gold_of[v], []).append(i)
        for g, idx in starts.items():
            if idx[0] > 0:
                ax.axhline(idx[0] - 0.5, color=SURF, lw=2.2)
            if ai == len(regs) - 1:
                ax.text(len(names) + 0.1, (idx[0] + idx[-1]) / 2, {"D": "D: emergency", "C/D": "C/D: edge", "C": "C", "B/C": "B/C", "B": "B", "A/B": "A/B", "A": "A: home"}[g], fontsize=6, va="center", ha="left", color=INK2, rotation=90 if len(idx) > 6 else 0)
        for sp in ("top", "right", "left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(vigs))); axes[0].set_yticklabels([f"{v} · {diag[v][:30]}" for v in vigs], fontsize=4.6)
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 16)); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.02, pad=0.08, aspect=40, location="top"); cb.set_label("Variants referred to emergency care (of 16)", fontsize=7); cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
    save(fig, "fig_vignettes")


# 12 --------------------------------------------------------------- AUROC by scenario
def fig_auroc():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.4), sharey=True, gridspec_kw=dict(wspace=0.12))
    ax = axes[0]
    for cfg in CONFIGS:
        a = R[cfg]["auroc"]; yy = Y[cfg]
        ax.plot([min(a["dka"], a["asthma"]), max(a["dka"], a["asthma"])], [yy, yy], color=DEEMPH, lw=1.2, zorder=1)
        ax.scatter([a["dka"]], [yy], s=20, color=ORANGE, edgecolor=SURF, lw=0.7, zorder=3)
        ax.scatter([a["asthma"]], [yy], s=20, color=AQUA, edgecolor=SURF, lw=0.7, zorder=3)
        ax.scatter([a["primary"]], [yy], s=24, color=BLUE, edgecolor=SURF, lw=0.8, zorder=4)
    config_axis(ax); ax.set_xlim(0.86, 1.005); ax.set_xticks([0.9, 0.95, 1.0]); ax.set_title("Primary endpoint (two emergency scenarios)"); ax.set_xlabel("AUROC")
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", markersize=5, label="both scenarios (64 vs 512)"), Line2D([], [], marker="o", color=ORANGE, ls="", markersize=5, label="DKA variants only"),
                       Line2D([], [], marker="o", color=AQUA, ls="", markersize=5, label="asthma variants only")], loc="lower left", fontsize=6.3)
    ax = axes[1]
    for cfg in CONFIGS:
        a = R[cfg]["auroc"]; yy = Y[cfg]
        ax.plot([a["loo_min"], a["loo_max"]], [yy, yy], color=BLUE_LT, lw=2.4, zorder=1)
        ax.scatter([a["expanded"]], [yy], s=24, color=BLUE, edgecolor=SURF, lw=0.8, zorder=3)
    config_axis(ax, labels=False); ax.set_xlim(0.82, 0.97); ax.set_xticks([0.85, 0.9, 0.95]); ax.set_title("Expanded set (20 positive scenarios)"); ax.set_xlabel("AUROC")
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", markersize=5, label="all scenarios"), Line2D([], [], color=BLUE_LT, lw=2.4, label="range, leaving each positive scenario out")], loc="lower left", fontsize=6.3)
    save(fig, "fig_auroc")


if __name__ == "__main__":
    for f in (fig_invariance, fig_misses, fig_threestate, fig_subgroups, fig_vignettes):  # figures used in the supplemental analyses; fig_holdout and fig_stability come from supplemental_figures2.py
        f(); print("wrote", f.__name__)
