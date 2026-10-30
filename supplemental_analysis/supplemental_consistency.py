#!/usr/bin/env python
"""Further checks of the recovered decision rule, from the elicitations shipped in data/ (no model is queried).

For all 18 configurations this computes:
  holdout   : the threshold rule against a majority-decision baseline chosen on the same fitting half and scored
              on the held-out half; absolute and balanced accuracy for every configuration and regime. The random
              half-splits replay those of supplemental_analysis.py, so the rule's held-out accuracy matches it.
  slope     : the fitted slope c_FP + c_FN alongside the break-even probability, for every regime.
  splits    : within vignettes whose 16 variants receive split default decisions, whether a higher elicited
              probability goes with referral, and how close those vignettes sit to the fitted threshold.
  residual  : whether default decisions depart from the fitted rule systematically by factor (difference in mean
              residual between levels, scenario-cluster bootstrap) or by vignette (share of the misfit shared
              within a vignette, against a parametric null).
  fixed     : with the threshold selected against the clinician labels held fixed (5:1 target), how often the
              externally computed referral changes across the five probability elicitations.
Writes consistency.json and holdout_majority_baseline.csv, slope_threshold.csv, split_vignettes.csv,
misfit_by_vignette.csv, misfit_by_factor.csv and fixed_threshold_repeats.csv.
"""
from __future__ import annotations
import csv, json, math, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from supplemental_analysis import (CONFIGS, FACTORS, REGIMES, RLAB, load_original, load_config, load_repetitions,
                                   fit_cost)

SPLITS, DRAWS = 50, 500
FACS = [f for f in FACTORS if f[0] != "prompt_type"]


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def best_threshold(p, y, ratio):
    """Threshold minimising ratio*FN + FP against the labels (the manuscript's best fixed utility)."""
    cand = np.unique(np.concatenate([[0.0], p, [1.0 + 1e-9]]))
    best, bt = float("inf"), None
    for t in cand:
        pred = p >= t
        cost = ratio * np.sum(~pred & (y == 1)) + np.sum(pred & (y == 0))
        if cost < best - 1e-12:
            best, bt = cost, t
    return float(bt)


def analyze(args):
    cfg, scen = args
    meta, beliefs, decisions = load_config(cfg)
    ctx = sorted(meta)
    scen_of = {c: scen[meta[c]["vig"]] for c in ctx}
    scenarios = sorted(set(scen_of.values()))
    orig = {c for c in ctx if scen_of[c] <= 30}
    out = {"cfg": list(cfg)}

    def pairs_of(reg, subset):
        cs = [c for c in subset if c in beliefs and c in decisions[reg]]
        return cs, np.array([beliefs[c] for c in cs]), np.array([decisions[reg][c] for c in cs])

    # ---------------------------------------------------------------- held-out rule vs majority baseline
    rng = np.random.default_rng(1000 + CONFIGS.index(tuple(cfg)))
    hold = {}
    for reg in REGIMES:
        acc_r, acc_m, bal_r, bal_m = [], [], [], []
        for _ in range(SPLITS):
            perm = rng.permutation(scenarios)
            A = set(perm[: len(perm) // 2]); B = set(perm[len(perm) // 2:])
            _, pA, aA = pairs_of(reg, [c for c in ctx if scen_of[c] in A])
            _, pB, aB = pairs_of(reg, [c for c in ctx if scen_of[c] in B])
            fA = fit_cost(pA, aA)
            if not fA or len(aB) == 0:
                continue
            rule = (pB >= fA["pstar"]).astype(int)
            maj = int(aA.mean() >= 0.5)
            acc_r.append(float((rule == aB).mean())); acc_m.append(float((aB == maj).mean()))
            if 0 < aB.mean() < 1:
                bal_r.append(0.5 * ((rule[aB == 1] == 1).mean() + (rule[aB == 0] == 0).mean()))
                bal_m.append(0.5)                       # a constant prediction always scores 0.5
        hold[reg] = dict(rule_acc=float(np.mean(acc_r)) if acc_r else None,
                         majority_acc=float(np.mean(acc_m)) if acc_m else None,
                         rule_bal=float(np.mean(bal_r)) if bal_r else None,
                         majority_bal=0.5 if bal_m else None, n=len(acc_r))
    out["holdout"] = hold

    # ---------------------------------------------------------------- slope and threshold, every regime
    slope = {}
    for reg in REGIMES:
        _, p, a = pairs_of(reg, ctx)
        f = fit_cost(p, a)
        if f:
            s = f["c_fp"] + f["c_fn"]
            slope[reg] = dict(pstar=f["pstar"], slope=s, width=2 * math.log(3) / s, ratio=f["ratio"])
    out["slope"] = slope

    # ---------------------------------------------------------------- split-decision vignettes
    reg = "decision_baseline"
    d = decisions[reg]
    f0 = fit_cost(*pairs_of(reg, ctx)[1:])
    pstar, s = f0["pstar"], f0["c_fp"] + f0["c_fn"]
    byv = defaultdict(list)
    for c in ctx:
        if c in beliefs and c in d:
            byv[meta[c]["vig"]].append((beliefs[c], d[c]))
    conc = tie = npair = 0
    split_gap, unan_gap, n_split = [], [], 0
    for v, rows in byv.items():
        ps = np.array([r[0] for r in rows]); ds = np.array([r[1] for r in rows])
        gap = float(np.abs(sig(s * (ps - pstar)) - 0.5).mean())      # 0 = right at the threshold
        if 0 < ds.mean() < 1:
            n_split += 1; split_gap.append(gap)
            P, N = ps[ds == 1], ps[ds == 0]
            for x in P:
                conc += (x > N).sum(); tie += (x == N).sum(); npair += len(N)
        else:
            unan_gap.append(gap)
    out["splits"] = dict(n_vignettes=len(byv), n_split=n_split, n_pairs=int(npair),
                         concordance=float((conc + 0.5 * tie) / npair) if npair else None,
                         frac_tied=float(tie / npair) if npair else None,
                         higher=int(conc), lower=int(npair - conc - tie), tied=int(tie),
                         split_closeness=float(np.mean(split_gap)) if split_gap else None,
                         unanimous_closeness=float(np.mean(unan_gap)) if unan_gap else None)

    # ---------------------------------------------------------------- systematic residuals
    cs, p, a = pairs_of(reg, ctx)
    fitted = sig(s * (p - pstar)); resid = a - fitted
    sc = np.array([scen_of[c] for c in cs]); vig = np.array([meta[c]["vig"] for c in cs])
    rng2 = np.random.default_rng(3000 + CONFIGS.index(tuple(cfg)))
    sidx = {x: np.where(sc == x)[0] for x in scenarios}
    fac_res = {}
    for fac, l1, l2 in FACS:
        lv = np.array([meta[c][fac] for c in cs])
        m1, m2 = lv == l1, lv == l2
        pt = float(resid[m1].mean() - resid[m2].mean())
        boot = []
        for _ in range(DRAWS):
            idx = np.concatenate([sidx[x] for x in rng2.choice(scenarios, len(scenarios))])
            b1, b2 = idx[m1[idx]], idx[m2[idx]]
            if len(b1) and len(b2):
                boot.append(resid[b1].mean() - resid[b2].mean())
        fac_res[fac] = dict(diff=pt, ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])

    def between_share(r):
        tot = float(((r - r.mean()) ** 2).sum())
        btw = sum(len(r[vig == v]) * (r[vig == v].mean() - r.mean()) ** 2 for v in np.unique(vig))
        return btw / tot if tot else 0.0
    obs = between_share(resid)
    null = [between_share(rng2.binomial(1, fitted) - fitted) for _ in range(200)]
    out["residual"] = dict(by_factor=fac_res, vignette_share=obs,
                           vignette_share_null=[float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))])

    # ---------------------------------------------------------------- fixed label-selected threshold
    runs = load_repetitions(cfg)
    fixed = {}
    if runs:
        common = [c for c in ctx if all(c in r for r in runs)]
        meanp = {c: float(np.mean([r[c] for r in runs])) for c in common}
        for endpoint in ("primary", "expanded"):
            if endpoint == "primary":
                cases = [c for c in common if c in orig and meta[c]["gold"] != "C/D"]
                y = np.array([int(meta[c]["gold"] == "D") for c in cases])
            else:
                cases = common
                y = np.array([meta[c]["er"] for c in cases])
            t = best_threshold(np.array([meanp[c] for c in cases]), y, 5.0)
            refer = np.array([[r[c] >= t for c in cases] for r in runs])          # 5 x n
            changed = (refer.min(axis=0) != refer.max(axis=0))
            sens = [float(refer[k][y == 1].mean()) for k in range(len(runs))]
            over = [float(refer[k][y == 0].mean()) for k in range(len(runs))]
            fixed[endpoint] = dict(threshold=t, n=len(cases), frac_changed=float(changed.mean()),
                                   frac_changed_pos=float(changed[y == 1].mean()),
                                   sens_range=[min(sens), max(sens)], over_range=[min(over), max(over)])
    out["fixed"] = fixed
    return out


def write_csv(name, header, rows):
    with open(HERE / name, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)


def r4(x):
    return "" if x is None else round(x, 4)


if __name__ == "__main__":
    scen, _, _ = load_original()
    with Pool(6) as pool:
        rows = pool.map(analyze, [(c, scen) for c in CONFIGS])
    json.dump(rows, open(HERE / "consistency.json", "w"), indent=1)

    write_csv("holdout_majority_baseline.csv",
              ["model", "reasoning", "fn_fp_ratio", "rule_accuracy", "majority_accuracy",
               "rule_balanced_accuracy", "majority_balanced_accuracy", "splits"],
              [[*r["cfg"], RLAB[g], r4(h["rule_acc"]), r4(h["majority_acc"]), r4(h["rule_bal"]),
                r4(h["majority_bal"]), h["n"]] for r in rows for g, h in r["holdout"].items()])
    write_csv("slope_threshold.csv",
              ["model", "reasoning", "fn_fp_ratio", "break_even_probability", "slope",
               "probability_width_25_to_75pct_referral", "recovered_ratio"],
              [[*r["cfg"], RLAB[g], r4(v["pstar"]), r4(v["slope"]), r4(v["width"]), r4(v["ratio"])]
               for r in rows for g, v in r["slope"].items()])
    write_csv("split_vignettes.csv",
              ["model", "reasoning", "vignettes", "split_vignettes", "pairs", "referred_higher", "referred_lower",
               "tied", "concordance", "split_distance_from_threshold", "unanimous_distance_from_threshold"],
              [[*r["cfg"], x["n_vignettes"], x["n_split"], x["n_pairs"], x["higher"], x["lower"], x["tied"],
                r4(x["concordance"]), r4(x["split_closeness"]), r4(x["unanimous_closeness"])]
               for r in rows for x in [r["splits"]]])
    write_csv("misfit_by_vignette.csv",
              ["model", "reasoning", "between_vignette_share", "null_2.5pct", "null_97.5pct"],
              [[*r["cfg"], r4(r["residual"]["vignette_share"]), *[r4(v) for v in r["residual"]["vignette_share_null"]]]
               for r in rows])
    write_csv("misfit_by_factor.csv",
              ["model", "reasoning", "factor", "level_1", "level_2", "residual_difference", "ci_low", "ci_high"],
              [[*r["cfg"], fac, l1, l2, r4(r["residual"]["by_factor"][fac]["diff"]),
                *[r4(v) for v in r["residual"]["by_factor"][fac]["ci"]]] for r in rows for fac, l1, l2 in FACS])
    write_csv("fixed_threshold_repeats.csv",
              ["model", "reasoning", "endpoint", "threshold", "cases", "share_changed", "share_changed_emergencies",
               "sensitivity_min", "sensitivity_max", "overtriage_min", "overtriage_max"],
              [[*r["cfg"], ep, r4(v["threshold"]), v["n"], r4(v["frac_changed"]), r4(v["frac_changed_pos"]),
                *[r4(x) for x in v["sens_range"]], *[r4(x) for x in v["over_range"]]]
               for r in rows for ep, v in r["fixed"].items()])
    print("wrote consistency.json and six CSV files for", len(rows), "configurations")
