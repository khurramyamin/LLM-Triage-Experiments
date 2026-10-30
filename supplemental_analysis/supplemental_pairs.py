#!/usr/bin/env python
"""Matched-pair comparisons of the factorial variants, from the elicitations shipped in data/ (no model is queried).

A matched pair is two variants of the same vignette that differ in one factor (race, sex, anchoring statement or
stated barrier to care) and agree in the other three. For every configuration and factor, under no stated
priority and under the 5:1 and 10:1 prompts, this records the change in elicited probability and the referral
switches in each direction: pairs in which only the first level is referred, and pairs in which only the second
is. The difference between those two shares gets a 95% interval from a cluster bootstrap over the 39 base
scenarios (500 draws), because pairs from the same scenario are not independent; the pooled interval resamples
scenarios once per draw for all 18 configurations. Among pairs whose decisions differ it also counts how often
the two variants had the same elicited probability and, when they did not, how often the referred variant had
the higher one. Writes matched_pairs.json and matched_pairs.csv.
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from supplemental_analysis import CONFIGS, FACTORS, load_original, load_config

DRAWS = 500
FACS = [f for f in FACTORS if f[0] != "prompt_type"]
REGS = {"default": "decision_baseline", "5": "decision_u_fp1_fn5", "10": "decision_u_fp1_fn10"}


def counts(args):
    """Per-scenario counts (pairs, only first referred, only second referred), probability changes and switches."""
    cfg, scen = args
    meta, beliefs, decisions = load_config(cfg)
    out = {}
    for rname, reg in REGS.items():
        d = decisions[reg]
        for fac, l1, l2 in FACS:
            others = [f for f, _, _ in FACS if f != fac]
            groups = defaultdict(dict)
            for c, m in meta.items():
                if c in beliefs and c in d:
                    groups[(m["vig"],) + tuple(m[o] for o in others)][m[fac]] = c
            per = defaultdict(lambda: np.zeros(3, int))
            dp, same_p, higher, lower = [], 0, 0, 0
            for (vig, *_), g in groups.items():
                if l1 in g and l2 in g:
                    c1, c2 = g[l1], g[l2]
                    r1, r2 = d[c1], d[c2]
                    dp.append(beliefs[c1] - beliefs[c2])
                    per[scen[vig]] += (1, int(r1 and not r2), int(r2 and not r1))
                    if r1 != r2:
                        pr, pn = (beliefs[c1], beliefs[c2]) if r1 else (beliefs[c2], beliefs[c1])
                        same_p += pr == pn; higher += pr > pn; lower += pr < pn
            dp = np.array(dp)
            out[(rname, fac)] = dict(per={int(s): v.tolist() for s, v in per.items()},
                                     dp_sum=float(dp.sum()), abs_dp_sum=float(np.abs(dp).sum()),
                                     same_p_pairs=int((dp == 0).sum()),
                                     switch_same_p=same_p, switch_higher=higher, switch_lower=lower)
    return cfg, out


if __name__ == "__main__":
    scen, _, _ = load_original()
    with Pool(6) as pool:
        rows = dict(pool.map(counts, [(c, scen) for c in CONFIGS]))
    scenarios = sorted({s for r in rows.values() for v in r.values() for s in v["per"]})
    rng = np.random.default_rng(4000)
    W = np.array([np.bincount(rng.choice(len(scenarios), len(scenarios)), minlength=len(scenarios))
                  for _ in range(DRAWS)])                                     # draws x scenarios

    def table(cfgs, rname, fac):
        M = np.zeros((len(scenarios), 3))
        for cfg in cfgs:
            for s, v in rows[cfg][(rname, fac)]["per"].items():
                M[scenarios.index(s)] += v
        n, o1, o2 = M.sum(0)
        B = W @ M
        diff = (B[:, 1] - B[:, 2]) / B[:, 0]
        x = [rows[c][(rname, fac)] for c in cfgs]
        return dict(n=int(n), only_l1=int(o1), only_l2=int(o2), diff=float((o1 - o2) / n),
                    ci=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                    mean_dp=sum(v["dp_sum"] for v in x) / n, mean_abs_dp=sum(v["abs_dp_sum"] for v in x) / n,
                    frac_same_p=sum(v["same_p_pairs"] for v in x) / n,
                    switch_same_p=sum(v["switch_same_p"] for v in x),
                    switch_higher=sum(v["switch_higher"] for v in x),
                    switch_lower=sum(v["switch_lower"] for v in x))

    res = {"pooled": {f"{r}|{f}": table(CONFIGS, r, f) for r in REGS for f, _, _ in FACS},
           "by_config": [dict(cfg=list(c), **{f"{r}|{f}": table([c], r, f) for r in REGS for f, _, _ in FACS})
                         for c in CONFIGS]}
    json.dump(res, open(HERE / "matched_pairs.json", "w"), indent=1)

    lev = {f: (l1, l2) for f, l1, l2 in FACS}
    out = []
    for label, entry in [(("all", "all"), res["pooled"])] + [((tuple(e["cfg"])), e) for e in res["by_config"]]:
        for r in REGS:
            for f, _, _ in FACS:
                t = entry[f"{r}|{f}"]
                out.append([*label, r, f, *lev[f], t["n"], round(t["frac_same_p"], 4), round(t["mean_dp"], 4),
                            round(t["mean_abs_dp"], 4), t["only_l1"], t["only_l2"], round(t["diff"], 4),
                            round(t["ci"][0], 4), round(t["ci"][1], 4), t["switch_same_p"], t["switch_higher"],
                            t["switch_lower"]])
    with open(HERE / "matched_pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "reasoning", "fn_fp_ratio", "factor", "level_1", "level_2", "pairs",
                    "share_same_probability", "mean_probability_change", "mean_absolute_probability_change",
                    "only_level_1_referred", "only_level_2_referred", "net_difference", "ci_low", "ci_high",
                    "switches_same_probability", "switches_referred_higher", "switches_referred_lower"])
        w.writerows(out)
    print("wrote matched_pairs.json and matched_pairs.csv")
