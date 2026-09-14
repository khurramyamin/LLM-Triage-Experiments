#!/usr/bin/env python
"""Bootstrap p-values and a joint Benjamini-Hochberg correction for the invariance contrasts.

Replays the exact random stream used for the cached invariance intervals (the hold-out
permutations, then the cluster-bootstrap draws), so the intervals it reproduces match the
cached ones, then extends the same stream to 5,000 draws for p-value resolution. Writes
invariance_p.json with, per configuration and factor: point, ci, bootstrap p, BH-adjusted p.
"""
from __future__ import annotations
import json, math, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from supplemental_analysis import (CONFIGS, FACTORS, REGIMES, load_original, load_config,
                               fit_cost, pct)

DRAWS, SPLITS, CI_DRAWS = 5000, 50, 500
FACS = [f for f in FACTORS if f[0] != "prompt_type"]


def analyze(args):
    cfg, scen = args
    meta, beliefs, decisions = load_config(cfg)
    ctx = sorted(meta)
    scen_of = {c: scen[meta[c]["vig"]] for c in ctx}
    scenarios = sorted(set(scen_of.values()))
    rng = np.random.default_rng(1000 + CONFIGS.index(tuple(cfg)))
    # replay: the hold-out loop draws one permutation per split per regime, and nothing else
    for _ in REGIMES:
        for _ in range(SPLITS):
            rng.permutation(scenarios)

    reg = "decision_baseline"
    cs = [c for c in ctx if c in beliefs and c in decisions[reg]]
    p = np.array([beliefs[c] for c in cs]); a = np.array([decisions[reg][c] for c in cs])
    sc = np.array([scen_of[c] for c in cs])
    lv = {}
    for fac, l1, l2 in FACTORS:
        vals = np.array([meta[c][fac] for c in cs])
        lv[(fac, l1)] = vals == l1; lv[(fac, l2)] = vals == l2
    NEEDED = [(f, l) for f, l1, l2 in FACS for l in (l1, l2)]
    point = {k: fit_cost(p[m], a[m]) for k, m in lv.items()}

    boot_lv = defaultdict(list)
    scen_idx = {s: np.where(sc == s)[0] for s in scenarios}
    for _ in range(DRAWS):
        pick = rng.choice(scenarios, size=len(scenarios), replace=True)
        w = np.zeros(len(cs))
        for s, k in Counter(pick).items():
            w[scen_idx[s]] = k
        for key in NEEDED:                                  # only the levels this test uses
            m = lv[key]
            fw = fit_cost(p[m], a[m], w[m]); boot_lv[key].append(fw["ratio"] if fw else None)

    out = {}
    for fac, l1, l2 in FACS:
        dd = [math.log10(x) - math.log10(y) for x, y in zip(boot_lv[(fac, l1)], boot_lv[(fac, l2)])
              if x and y and x > 0 and y > 0]
        pt = math.log10(point[(fac, l1)]["ratio"]) - math.log10(point[(fac, l2)]["ratio"])
        n = len(dd)
        le = sum(1 for v in dd if v <= 0); ge = sum(1 for v in dd if v >= 0)
        pval = min(1.0, 2 * min((le + 1) / (n + 1), (ge + 1) / (n + 1)))
        ci500 = list(pct([v for v, keep in zip(dd, keep500) if keep][:CI_DRAWS])) if False else list(pct(dd[:CI_DRAWS]))
        out[fac] = dict(point=pt, ci=ci500, ci_full=list(pct(dd)), p=pval, n=n)
    return {"cfg": list(cfg), "res": out}


if __name__ == "__main__":
    scen, _, _ = load_original()
    with Pool(6) as pool:
        rows = pool.map(analyze, [(c, scen) for c in CONFIGS])
    # joint Benjamini-Hochberg across all 72 contrasts
    flat = [(r["cfg"][0], r["cfg"][1], fac, r["res"][fac]["p"]) for r in rows for fac in r["res"]]
    flat.sort(key=lambda t: t[3]); m = len(flat)
    adj, prev = {}, 1.0
    for i in range(m - 1, -1, -1):
        model, eff, fac, pv = flat[i]
        prev = min(prev, pv * m / (i + 1))
        adj[(model, eff, fac)] = min(1.0, prev)
    for r in rows:
        for fac in r["res"]:
            r["res"][fac]["p_bh"] = adj[(r["cfg"][0], r["cfg"][1], fac)]
    json.dump(rows, open(HERE / "invariance_p.json", "w"))
    print(f"wrote invariance_p.json  ({m} contrasts)")
