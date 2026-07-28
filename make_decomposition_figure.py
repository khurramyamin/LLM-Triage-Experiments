"""Decompose each model's default-mode decision loss into the part attributable
to belief *miscalibration* and the part attributable to a *misspecified default
utility*.

Framework (reimplemented from scratch)
--------------------------------------
There is **no external "reference" utility**.  Loss is the plain 0/1
misclassification error against the reference (clinician) labels y, i.e. a false
positive and a false negative are weighted equally:

    loss(d, y) = mean( d != y ).

A decision is  ``refer  <=>  belief >= threshold``.  We toggle two binary factors:

  * beliefs : raw elicited probability  p
              vs. cross-fitted isotonic-calibrated  c = calibrate(p; y)
  * utility : the model's DEFAULT decision threshold  p*_d = 1/(1+r_d),
              where r_d is the FN/FP cost ratio recovered from the model's own
              baseline (no-utility-prompt) decisions by revealed preference,
              vs. the BEST FIXED utility -- the threshold that minimises 0/1
              loss against y (the yellow-star operating point in fig3).

This yields four policies and four losses L(beliefs, utility):

  L00 = L(raw,        default)   the model's default-mode policy  (TOTAL loss)
  L10 = L(calibrated, default)   post-process beliefs w/ calibration, keep utility
  L01 = L(raw,        best)      keep beliefs, pick the loss-optimal utility
  L11 = L(calibrated, best)      fix both  (floor set by limited discrimination)

Because isotonic calibration is monotone it preserves the ROC, so the best
achievable loss is fixed by discrimination alone:  L11 ~= L01 = the irreducible
floor.  Calibration is only able to reduce loss through the *default* utility:
with a fixed probability threshold, correcting the probability scale moves the
default operating point along the ROC.

Exact, additive Shapley attribution of the total default loss L00
(symmetric over the two factors -- uses both of the counterfactuals above):

  irreducible      = L11
  calibration_loss = 0.5 * ((L00 - L10) + (L01 - L11))
  utility_loss     = 0.5 * ((L00 - L01) + (L10 - L11))
  L00 = irreducible + calibration_loss + utility_loss   (exactly)

A negative calibration term means calibration cannot help -- the raw beliefs,
combined with the default utility, already sit at (or better than) the point a
calibrated model would reach under 0/1 scoring (well-calibrated strong models).
Such small negatives are folded into the utility term for the stacked bar and
reported verbatim in the console table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

import analyze_sweep as A
from make_paper_figures import (
    CONFIG_ORDER, FAMILY_ORDER, FAMILY_LABEL, parse, OUT,
)

BELIEF_SWEEP = Path("revealed_preferences/2026-07-01_factorial")
# The loss decomposition is exploratory and NOT part of the paper; keep all of
# its figures in a dedicated subfolder rather than the main figures directory.
OUT_DECOMP = OUT / "decomposition"
N_FOLDS = 5
SEED = 0

# Reference utility used to *score* decisions.  (1, 1) == plain 0/1 error;
# (1, 5) weights a false negative (missed emergency) 5x a false positive.
# Set by build(); the loss/threshold helpers below read these globals so the
# whole decomposition is evaluated under one consistent reference utility.
REF_C_FP = 1.0
REF_C_FN = 1.0

# How the "corrected utility" operating point is chosen for the utility bar:
#   None      -> best fixed utility: the empirically loss-minimising threshold
#                (best point on the ROC under the reference utility; fig3 star)
#   float R   -> a FIXED utility with FN:FP = R:1, i.e. cut the raw beliefs at
#                the Bayes threshold t = 1/(1+R)  (e.g. R = 5 -> t = 0.167)
UTIL_RATIO: float | None = None


# ---------------------------------------------------------------------------
# Loss + operating points (weighted misclassification cost vs reference labels)
# ---------------------------------------------------------------------------
def ref_loss(decisions: np.ndarray, y: np.ndarray) -> float:
    """Mean expected cost per patient under the reference utility.

    A false positive costs REF_C_FP, a false negative costs REF_C_FN.  With
    REF_C_FP == REF_C_FN this reduces to the 0/1 misclassification error.
    """
    d = np.asarray(decisions, int)
    y = np.asarray(y, int)
    fp = ((d == 1) & (y == 0)).sum() * REF_C_FP
    fn = ((d == 0) & (y == 1)).sum() * REF_C_FN
    return float((fp + fn) / len(y))


def best_threshold(beliefs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Threshold on `beliefs` minimising the reference-utility loss -- the
    'best fixed utility' (yellow-star operating point).  Returns (threshold, loss)."""
    s = np.asarray(beliefs, float)
    cands = np.unique(np.concatenate([[-1e-9], s, [1.0 + 1e-9]]))
    best_t, best_l = 0.5, np.inf
    for t in cands:
        l = ref_loss((s >= t).astype(int), y)
        if l < best_l:
            best_l, best_t = l, t
    return float(best_t), float(best_l)


# ---------------------------------------------------------------------------
# Cross-fitted isotonic-regression calibration against the 0/1 labels
# ---------------------------------------------------------------------------
def crossfit_calibrate(p: np.ndarray, y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Out-of-sample isotonic-calibrated probabilities via K-fold cross-fitting.

    Each held-out fold is mapped by an isotonic regression fitted on the *other*
    folds, so the calibrated value of every case uses only labels of other cases.
    """
    rng = np.random.default_rng(seed)
    n = len(p)
    folds = rng.permutation(n) % N_FOLDS
    c = np.empty(n, dtype=float)
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        if tr.sum() < 10 or te.sum() == 0:
            c[te] = p[te]
            continue
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p[tr], y[tr])
        c[te] = iso.predict(p[te])
    return c


# ---------------------------------------------------------------------------
# Loss decomposition: two independent single-fix improvements
# ---------------------------------------------------------------------------
# All losses are scored under the reference utility (ref_loss).  Baseline is the
# model's actual default policy:
#   L00 = L(p >= t_d)   raw beliefs, recovered default utility           (baseline)
# and the two independent fixes are:
#   L10 = L(c >= t_d)   calibrate beliefs, keep the default utility
#   L01 = L(p >= t_u)   keep raw beliefs, move to the corrected utility, where
#                       the operating point t_u is either the BEST point on the
#                       ROC (UTIL_RATIO is None) or the FIXED Bayes threshold
#                       t_u = 1/(1+UTIL_RATIO) of a prescribed FN:FP utility.
# giving
#   calibration improvement = L00 - L10   (loss removed by fixing calibration)
#   utility     improvement = L00 - L01   (loss removed by fixing the utility)
def _losses(p: np.ndarray, c: np.ndarray, y: np.ndarray, t_default: float):
    d00 = (p >= t_default).astype(int)          # raw, default utility
    d10 = (c >= t_default).astype(int)          # calibrated, default utility
    L00 = ref_loss(d00, y)
    L10 = ref_loss(d10, y)
    if UTIL_RATIO is None:
        _, L01 = best_threshold(p, y)           # raw, best point on the ROC
    else:
        t_u = 1.0 / (1.0 + UTIL_RATIO)          # raw, fixed FN:FP=UTIL_RATIO utility
        L01 = ref_loss((p >= t_u).astype(int), y)
    return L00, L10, L01


def decompose_config(name: str) -> dict:
    cfg = A.analyse_config(BELIEF_SWEEP / name)
    belief, gold = cfg["belief"], cfg["gold"]
    dec_base = cfg["decisions"].get("decision_baseline", {})

    ctxs = [k for k in belief if k in gold]
    p = np.array([belief[k] for k in ctxs], dtype=float)
    y = np.array([gold[k] for k in ctxs], dtype=int)

    # default utility: threshold implied by the recovered baseline FN/FP ratio.
    r_d = cfg["baseline_ratio"]
    if r_d and r_d > 0:
        t_default = 1.0 / (1.0 + r_d)
    else:
        # degenerate baseline fit -> fall back to the threshold reproducing the
        # model's actual baseline admit rate on the raw beliefs.
        admit = cfg.get("baseline_admit") or float(
            np.mean([dec_base[k] for k in ctxs if k in dec_base] or [0.5]))
        t_default = float(np.quantile(p, 1.0 - admit)) if 0 < admit < 1 else 0.5
        r_d = (1.0 - t_default) / t_default if t_default > 0 else float("inf")

    c = crossfit_calibrate(p, y)
    L00, L10, L01 = _losses(p, c, y, t_default)
    calibration = L00 - L10
    utility = L00 - L01

    # actual default-decision loss (context for narrative (1): decision perf.)
    act = [(dec_base[k], gold[k]) for k in ctxs if k in dec_base]
    L_act = ref_loss(*map(np.array, zip(*act))) if act else float("nan")

    auc = A.roc_curve(list(p), list(y))[2]
    return dict(name=name, L00=L00, L10=L10, L01=L01,
                calibration=calibration, utility=utility,
                auc=auc, r_default=r_d, t_default=t_default, L_act=L_act)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build(c_fp: float = 1.0, c_fn: float = 1.0,
          util_ratio: float | None = None) -> None:
    global REF_C_FP, REF_C_FN, UTIL_RATIO
    REF_C_FP, REF_C_FN = float(c_fp), float(c_fn)
    UTIL_RATIO = None if util_ratio is None else float(util_ratio)

    equal = abs(REF_C_FN - REF_C_FP) < 1e-9
    ratio = REF_C_FN / REF_C_FP
    if equal:
        loss_desc = "0/1 misclassification error vs. reference labels"
        suffix = ""
    else:
        loss_desc = (f"expected cost per patient, reference utility "
                     f"FN:FP = {ratio:g}:1")
        suffix = f"_fn{ratio:g}"

    if UTIL_RATIO is None:
        util_desc = ("best fixed utility \u2014 the best operating point on the ROC "
                     "under the reference utility (fig3 star)")
        util_bar_lbl = "Improvement from correcting utility (best point on ROC)"
    else:
        t_u = 1.0 / (1.0 + UTIL_RATIO)
        util_desc = (f"a fixed FN:FP = {UTIL_RATIO:g}:1 utility "
                     f"(raw beliefs cut at t = {t_u:.3f})")
        util_bar_lbl = (f"Improvement from correcting utility "
                        f"(fixed FN:FP = {UTIL_RATIO:g}:1 threshold)")
        suffix += f"_util{UTIL_RATIO:g}"

    rows = [decompose_config(n) for n in CONFIG_ORDER
            if (BELIEF_SWEEP / n / "results.csv").exists()]

    def sort_key(name):
        fam, reason = parse(name)
        rorder = {
            "minimal": 0, "none": 0, "low": 0,
            "medium": 1, "high": 2, "max": 2,
        }[reason]
        return (FAMILY_ORDER.index(fam), rorder)
    rows.sort(key=lambda r: sort_key(r["name"]))

    labels, cal_c, uti_c, aucs = [], [], [], []
    for r in rows:
        fam, reason = parse(r["name"])
        rtxt = {"minimal": "Minimal", "none": "None", "low": "Low",
                "medium": "Medium", "high": "High", "max": "Max"}[reason]
        labels.append(f"{FAMILY_LABEL[fam]}\n{rtxt}")
        # Independent single-fix loss reductions from the default policy L00:
        #   calibration improvement = L00 - L10  (calibrate beliefs, keep utility)
        #   utility improvement     = L00 - L01  (keep beliefs, best point on ROC)
        cal_c.append(r["calibration"])
        uti_c.append(r["utility"])
        aucs.append(r["auc"])

    x = np.arange(len(rows))
    cal_c = np.array(cal_c); uti_c = np.array(uti_c)

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    w = 0.38
    c_cal, c_uti = "#2c7fb8", "#d95f0e"
    b1 = ax.bar(x - w / 2, cal_c, w, color=c_cal, edgecolor="k", lw=0.5,
                label="Improvement from correcting calibration (default utility kept)")
    b2 = ax.bar(x + w / 2, uti_c, w, color=c_uti, edgecolor="k", lw=0.5,
                label=util_bar_lbl)

    # value labels
    span = float(max(cal_c.max(), uti_c.max()) -
                 min(cal_c.min(), uti_c.min(), 0.0)) or 1.0
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            off = span * 0.015 * (1 if h >= 0 else -1)
            ax.text(bar.get_x() + bar.get_width() / 2, h + off, f"{h:+.3f}",
                    ha="center", va=va, fontsize=8, color="0.25")

    ax.axhline(0, color="k", lw=0.8)

    # family separators
    for xi in (2.5, 5.5):
        ax.axvline(xi, color="0.85", lw=1.0, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Loss reduction per patient\n"
                  f"({loss_desc})",
                  fontsize=12)
    lo = min(cal_c.min(), uti_c.min(), 0.0)
    hi = max(cal_c.max(), uti_c.max())
    ax.set_ylim(lo - span * 0.10, hi + span * 0.14)
    title = "Where the Fixable Loss Is: Correcting Calibration vs. Correcting Utility"
    if not equal:
        title += f"\n(reference utility FN:FP = {ratio:g}:1)"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=10.5, loc="upper right", framealpha=0.95)

    fig.text(0.012, -0.045,
             f"Loss = {loss_desc}. Each bar is an INDEPENDENT single-fix loss "
             "reduction from the model's actual default policy "
             "(L00 = raw elicited beliefs thresholded at the recovered default "
             "utility p*_d).\n"
             "Correcting calibration: L00 \u2212 L10, where L10 applies the SAME "
             "default utility to cross-fitted isotonic-calibrated beliefs. "
             f"Correcting utility: L00 \u2212 L01, where L01 keeps the raw beliefs but "
             f"moves to {util_desc}. "
             "A negative calibration bar means recalibration hurts under the (wrong) "
             "default utility.",
             fontsize=8.5, color="0.35", ha="left")

    fig.tight_layout()
    OUT_DECOMP.mkdir(parents=True, exist_ok=True)
    out = OUT_DECOMP / f"fig_loss_decomposition{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # console table
    print(f"\n=== reference utility FN:FP = {ratio:g}:1 ===")
    print(f"{'config':<24}{'AUC':>5}{'r_d':>6}{'p*_d':>6}{'L00':>7}"
          f"{'L10':>7}{'L01':>7}{'dCal':>8}{'dUtil':>8}")
    for r in rows:
        print(f"{r['name']:<24}{r['auc']:>5.2f}{r['r_default']:>6.2f}"
              f"{r['t_default']:>6.2f}{r['L00']:>7.3f}{r['L10']:>7.3f}"
              f"{r['L01']:>7.3f}{r['calibration']:>+8.3f}{r['utility']:>+8.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    import sys
    # CLI: [scoring FN/FP ratio] [utility FN/FP ratio]
    #   arg1  reference/scoring utility ratio   (default 1 -> 0/1 error)
    #   arg2  corrected-utility ratio           (default: best point on the ROC)
    ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    util_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else None
    build(c_fp=1.0, c_fn=ratio, util_ratio=util_ratio)
