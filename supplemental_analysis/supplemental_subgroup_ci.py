#!/usr/bin/env python
"""Per-configuration subgroup differences with paired bootstrap intervals.

For every configuration, factor and regime this computes the difference between the two
factor levels in sensitivity (emergencies referred) and over-triage (non-emergencies
referred), with a paired bootstrap interval computed two ways:
  scenario : resample base scenarios with replacement (clusters)
  variant  : resample case variants within each scenario (stratified)
Writes subgroup_ci.json. Uses only the elicitations shipped in data/.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from supplemental_analysis import CONFIGS, FACTORS, REGIMES, load_original, load_config

DRAWS = 500
FACS = [f for f in FACTORS if f[0] != "prompt_type"]


def paired_ci(by_scen, draws, rng, mode):
    """by_scen: {scenario: (s1_sum, s1_cnt, s2_sum, s2_cnt)} -> (point, lo, hi)."""
    scen = sorted(by_scen)
    A = np.array([by_scen[s] for s in scen], float)          # (S, 4)
    tot = A.sum(axis=0)
    if tot[1] == 0 or tot[3] == 0:
        return (float("nan"),) * 3
    point = tot[0] / tot[1] - tot[2] / tot[3]
    out = []
    for _ in range(draws):
        if mode == "scenario":
            pick = rng.integers(0, len(scen), size=len(scen))
            t = A[pick].sum(axis=0)
        else:   # resample variants within each scenario: binomial redraw at the observed rate
            s1 = rng.binomial(A[:, 1].astype(int), np.divide(A[:, 0], A[:, 1], out=np.zeros(len(scen)), where=A[:, 1] > 0)).sum()
            s2 = rng.binomial(A[:, 3].astype(int), np.divide(A[:, 2], A[:, 3], out=np.zeros(len(scen)), where=A[:, 3] > 0)).sum()
            t = np.array([s1, A[:, 1].sum(), s2, A[:, 3].sum()], float)
        if t[1] > 0 and t[3] > 0:
            out.append(t[0] / t[1] - t[2] / t[3])
    if not out:
        return (point, float("nan"), float("nan"))
    return (float(point), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def analyze(args):
    cfg, scen = args
    meta, _, decisions = load_config(cfg)
    ctx = sorted(meta)
    scen_of = {c: scen[meta[c]["vig"]] for c in ctx}
    orig = {c for c in ctx if scen_of[c] <= 30}
    groups = dict(
        prim=( [c for c in ctx if c in orig and meta[c]["gold"] == "D"],
               [c for c in ctx if c in orig and meta[c]["gold"] not in ("D", "C/D")] ),
        exp =( [c for c in ctx if meta[c]["gold"] == "D"],
               [c for c in ctx if meta[c]["er"] == 0] ))
    res = {}
    for reg in REGIMES:
        dd = decisions[reg]
        for kind, (pos, neg) in groups.items():
            for fac, l1, l2 in FACS:
                for metric, cases in (("sens", pos), ("over", neg)):
                    by = defaultdict(lambda: [0, 0, 0, 0])
                    for c in cases:
                        if c not in dd:
                            continue
                        i = 0 if meta[c][fac] == l1 else (2 if meta[c][fac] == l2 else None)
                        if i is None:
                            continue
                        by[scen_of[c]][i] += dd[c]; by[scen_of[c]][i + 1] += 1
                    key = f"{kind}|{reg}|{fac}|{metric}"
                    rng = np.random.default_rng(1000 + CONFIGS.index(tuple(cfg)))
                    res[key] = dict(scenario=paired_ci(by, DRAWS, rng, "scenario"),
                                    variant=paired_ci(by, DRAWS, np.random.default_rng(2000 + CONFIGS.index(tuple(cfg))), "variant"),
                                    n_scen=len(by))
    return {"cfg": list(cfg), "res": res}


if __name__ == "__main__":
    scen, _, _ = load_original()
    with Pool(6) as pool:
        rows = pool.map(analyze, [(c, scen) for c in CONFIGS])
    json.dump(rows, open(HERE / "subgroup_ci.json", "w"))
    print("wrote subgroup_ci.json for", len(rows), "configurations")
    r0 = rows[0]["res"]
    for k in ["prim|decision_baseline|race|sens", "prim|decision_baseline|race|over", "exp|decision_baseline|race|sens"]:
        v = r0[k]
        print(f"{k:42s} scenarios={v['n_scen']:3d}  point {v['scenario'][0]:+.3f}  "
              f"scenario CI [{v['scenario'][1]:+.3f},{v['scenario'][2]:+.3f}]  variant CI [{v['variant'][1]:+.3f},{v['variant'][2]:+.3f}]")
