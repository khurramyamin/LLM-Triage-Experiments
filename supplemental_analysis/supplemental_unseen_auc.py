#!/usr/bin/env python
"""AUC of the elicited probability for predicting the model's own decision, restricted to the
held-out half of each scenario split, averaged over the same 50 splits used for the hold-out test."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from supplemental_analysis import CONFIGS, REGIMES, load_original, load_config, auc

SPLITS = 50


def analyze(args):
    cfg, scen = args
    meta, beliefs, decisions = load_config(cfg)
    ctx = sorted(meta)
    scen_of = {c: scen[meta[c]["vig"]] for c in ctx}
    scenarios = sorted(set(scen_of.values()))
    rng = np.random.default_rng(1000 + CONFIGS.index(tuple(cfg)))
    out = {}
    for reg in REGIMES:
        d = decisions[reg]
        cs = [c for c in ctx if c in beliefs and c in d]
        p = np.array([beliefs[c] for c in cs]); a = np.array([d[c] for c in cs])
        sc = np.array([scen_of[c] for c in cs])
        vals = []
        for _ in range(SPLITS):
            perm = rng.permutation(scenarios)
            B = set(perm[len(perm) // 2:])                 # the held-out half
            m = np.isin(sc, list(B))
            if m.sum() and 0 < a[m].mean() < 1:
                vals.append(auc(p[m], a[m]))
        out[reg] = float(np.mean(vals)) if vals else float("nan")
    return {"cfg": list(cfg), "auc_unseen": out}


if __name__ == "__main__":
    scen, _, _ = load_original()
    with Pool(6) as pool:
        rows = pool.map(analyze, [(c, scen) for c in CONFIGS])
    json.dump(rows, open(HERE / "unseen_auc.json", "w"))
    v = [x for r in rows for x in r["auc_unseen"].values() if x == x]
    print(f"AUC on the held-out half, across {len(v)} configuration-by-regime combinations:")
    print(f"  range {min(v):.2f} to {max(v):.2f}   median {np.median(v):.2f}")
