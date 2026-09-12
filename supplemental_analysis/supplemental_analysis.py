#!/usr/bin/env python
"""Supplemental analyses (held-out validation of the recovered decision rule, invariance across
the factorial design, repeated-elicitation stability, misses by scenario, three-state referral rates, subgroup error rates).

Uses ONLY the existing elicitations shipped in this repository:
  * data/models/*.csv.gz                          (utility experiment: belief, default and cost-prompted decisions)
  * data/original_paper/DataExpanded_FINAL.csv.gz (scenario membership, clinician labels, deployed-tool decisions)
  * data/belief_repetitions/                      (five repeated belief elicitations per configuration)

No model is queried. Writes Markdown tables to supplemental_analysis/tables/, CSV supplements, results.json and results_cache.pkl.

Run from the repository root:  python supplemental_analysis/supplemental_analysis.py [--draws 500] [--splits 50] [--workers 6] [--from-cache]
"""
from __future__ import annotations
import argparse, csv, gzip, json, math, re, sys
from collections import defaultdict, Counter
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

csv.field_size_limit(sys.maxsize)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TABLES = HERE / "tables"
REP_DIR = ROOT / "data" / "belief_repetitions"

FAMILIES = [("gpt-5-mini", ["minimal", "medium", "high"], "GPT-5-mini"),
            ("gpt-5.4", ["none", "medium", "high"], "GPT-5.4"),
            ("DeepSeek-V4-Flash", ["none", "high", "max"], "DeepSeek-V4-Flash"),
            ("DeepSeek-V4-Pro", ["none", "high", "max"], "DeepSeek-V4-Pro"),
            ("Claude-Fable-5", ["low", "medium", "high"], "Claude Fable 5"),
            ("Claude-Sonnet-5", ["low", "medium", "high"], "Claude Sonnet 5")]
CONFIGS = [(m, e) for m, es, _ in FAMILIES for e in es]
DISPLAY = {m: d for m, _, d in FAMILIES}
REGIMES = ["decision_baseline", "decision_u_fp100_fn1", "decision_u_fp10_fn1", "decision_u_fp5_fn1",
           "decision_u_fp1_fn1", "decision_u_fp1_fn5", "decision_u_fp1_fn10", "decision_u_fp1_fn100"]
RLAB = {"decision_baseline": "default", "decision_u_fp100_fn1": "0.01", "decision_u_fp10_fn1": "0.1",
        "decision_u_fp5_fn1": "0.2", "decision_u_fp1_fn1": "1", "decision_u_fp1_fn5": "5",
        "decision_u_fp1_fn10": "10", "decision_u_fp1_fn100": "100"}
FACTORS = [("race", "Black", "White"), ("gender", "woman", "man"), ("has_anchor", "yes", "no"),
           ("has_barrier", "yes", "no"), ("prompt_type", "1", "2")]
FACTOR_LABEL = {"race": ("Black", "White"), "gender": ("woman", "man"), "has_anchor": ("anchor", "no anchor"),
                "has_barrier": ("barrier", "no barrier"), "prompt_type": ("with labs", "symptoms only")}
CELLS = ["WM", "WW", "BM", "BW", "WM-A", "WW-A", "BM-A", "BW-A", "WM-X", "WW-X", "BM-X", "BW-X",
         "WM-AX", "WW-AX", "BM-AX", "BW-AX"]
D_ORIG = ["E9", "F9", "E13", "F13"]


def cname(cfg):
    return f"{DISPLAY[cfg[0]]} $\\cdot$ {cfg[1]}"


# ----------------------------------------------------------------------------- data
def _open(path):
    """Open a CSV that may be stored gzipped (path.gz) or plain."""
    gz = path if path.suffix == ".gz" else path.with_name(path.name + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt", encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


SCENARIO_NAMES = {1: "TIA", 2: "Painless gross hematuria", 3: "Stable angina", 4: "Back pain + red flags", 5: "Viral pharyngitis",
    6: "Tension headache", 7: "Early appendicitis", 8: "Benign PVCs", 9: "Asthma exacerbation", 10: "Moderate hyperkalemia + CKD",
    11: "Drug-induced hyponatremia", 12: "Suspected DVT", 13: "DKA", 14: "Pyelonephritis", 15: "Hypertensive urgency", 16: "Upper GI bleed",
    17: "Exercise-induced hematuria", 18: "Subclinical hypothyroidism", 19: "Mild thrombocytopenia", 20: "Hemorrhoids", 21: "Changing mole",
    22: "ITP", 23: "Post-viral leukopenia", 24: "NAFLD", 25: "Bell's palsy", 26: "Anterior uveitis", 27: "Monoarthritis", 28: "Passive SI",
    29: "Active suicidal ideation without intent", 30: "Early psychosis", 31: "Suicidal ideation after job loss",
    32: "Suicidal ideation in a new parent", 33: "Worsening depression with daily SI", 34: "Suicidal ideation associated with alcohol use",
    35: "First-episode suicidal ideation, method thought", 36: "Acute ischemic stroke", 37: "Anaphylaxis", 38: "Bacterial meningitis",
    39: "Aortic dissection"}


def _context_meta(context_id, case_number, gold):
    """Derive the factorial metadata from a context id such as ``E10__BM-AX`` (vignette id, race/gender, anchor, barrier)."""
    vig, variant = context_id.split("__", 1)
    base, _, suffix = variant.partition("-")
    prefix = vig.rstrip("0123456789")
    return dict(vig=vig, gold=gold, er=1 if "D" in gold else 0, cls="edge" if "/" in gold else "clear",
                race="Black" if base[0] == "B" else "White", gender="man" if base[1] == "M" else "woman",
                has_anchor="yes" if "A" in suffix else "no", has_barrier="yes" if "X" in suffix else "no",
                prompt_type="1" if prefix in ("E", "MH") else "2", variant=variant, case=int(case_number))


def load_original():
    """Scenario number and short name per vignette id, and the deployed tool's published decision per context.

    Scenario numbers 1-30 are the 30 base scenarios of the original study; 31-39 are the supplementary textbook
    scenarios. Both come from the ``case_number`` column of the model files; the deployed decisions from ``their_triage``."""
    scen, diag, deployed = {}, {}, {}
    with _open(ROOT / "data/models" / f"{CONFIGS[0][0]}.csv") as fh:
        for r in csv.DictReader(fh):
            if r["regime"] != "belief" or r["experiment"] != "utility":
                continue
            vig = r["context_id"].split("__", 1)[0]
            scen[vig] = int(r["case_number"]); diag[vig] = SCENARIO_NAMES[int(r["case_number"])]
            deployed[r["context_id"]] = 1 if r["their_triage"] == "D" else 0
    return scen, diag, deployed


def load_config(cfg):
    """Return (meta, beliefs, decisions[regime]) for one configuration from its (gzipped) model CSV."""
    model, effort = cfg
    meta, beliefs, decisions = {}, {}, defaultdict(dict)
    with _open(ROOT / "data/models" / f"{model}.csv") as fh:
        for r in csv.DictReader(fh):
            if r["model"] != model or r["reasoning_effort"] != effort or r["experiment"] != "utility":
                continue
            cid = r["context_id"]
            if cid not in meta:
                meta[cid] = _context_meta(cid, r["case_number"], r["gold_standard"])
            if r["regime"] == "belief":
                if r["parsed_probability"] != "":
                    beliefs[cid] = float(r["parsed_probability"])
            elif r["regime"] in RLAB and r["parsed_decision"] != "":
                decisions[r["regime"]][cid] = int(float(r["parsed_decision"]))
    return meta, beliefs, decisions


def load_repetitions(cfg):
    runs = []
    for k in range(1, 6):
        p = REP_DIR / f"repetition_{k}" / f"{cfg[0]}_re-{cfg[1]}" / "results.csv"
        if not p.exists() and not p.with_suffix(".csv.gz").exists():
            return []
        vals = {}
        with _open(p) as fh:
            for r in csv.DictReader(fh):
                if r["regime"] == "belief" and r["parsed_probability"] != "":
                    vals[r["context_id"]] = float(r["parsed_probability"])
        runs.append(vals)
    return runs


# ----------------------------------------------------------------------------- statistics
def fit_cost(p, a, w=None):
    """Binary-logit cost fit identical to analysis.metrics.fit_cost_function (beta = 1, L-BFGS-B, x0 = (1,1)),
    optionally with per-observation weights (used for the cluster bootstrap)."""
    p = np.asarray(p, float); a = np.asarray(a, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    n = int(w.sum())
    frac = float((w * a).sum() / w.sum()) if w.sum() else float("nan")
    if len(p) < 5 or frac <= 0.0 or frac >= 1.0:
        return None

    def f(c):
        cfp, cfn = c
        z = cfn * p - cfp * (1.0 - p)
        ll = a * (-np.logaddexp(0.0, -z)) + (1.0 - a) * (-np.logaddexp(0.0, z))
        s = 1.0 / (1.0 + np.exp(-z))
        r = (a - s) * w
        return -(w * ll).sum(), np.array([(r * (1.0 - p)).sum(), -(r * p).sum()])

    res = minimize(f, x0=np.array([1.0, 1.0]), jac=True, method="L-BFGS-B", bounds=[(1e-6, None), (1e-6, None)])
    cfp, cfn = float(res.x[0]), float(res.x[1])
    return dict(c_fp=cfp, c_fn=cfn, ratio=cfn / cfp, pstar=cfp / (cfp + cfn), n=n, frac=frac)


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def pav(x, y):
    """Isotonic (non-decreasing) regression of y on x; returns (sorted unique x, fitted values)."""
    order = np.argsort(x, kind="mergesort"); x = np.asarray(x, float)[order]; y = np.asarray(y, float)[order]
    # pool identical x first
    ux, inv = np.unique(x, return_inverse=True)
    sums = np.bincount(inv, weights=y); cnt = np.bincount(inv)
    vals = list(sums / cnt); wts = list(cnt.astype(float)); starts = list(range(len(ux)))
    blocks_v, blocks_w, blocks_s = [], [], []
    for v, w, s in zip(vals, wts, starts):
        blocks_v.append(v); blocks_w.append(w); blocks_s.append(s)
        while len(blocks_v) > 1 and blocks_v[-2] > blocks_v[-1]:
            v2, w2 = blocks_v.pop(), blocks_w.pop(); blocks_s.pop()
            blocks_v[-1] = (blocks_v[-1] * blocks_w[-1] + v2 * w2) / (blocks_w[-1] + w2)
            blocks_w[-1] += w2
    fitted = np.empty(len(ux))
    for i, s in enumerate(blocks_s):
        e = blocks_s[i + 1] if i + 1 < len(blocks_s) else len(ux)
        fitted[s:e] = blocks_v[i]
    return ux, fitted


def ece(p, y, bins=10):
    p = np.asarray(p, float); y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def pct(x, lo=2.5, hi=97.5):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(x, lo)), float(np.percentile(x, hi)))


# ----------------------------------------------------------------------------- per-configuration worker
def analyze(args):
    cfg, draws, splits, scen, deployed = args
    meta, beliefs, decisions = load_config(cfg)
    ctx = sorted(meta)
    scen_of = {c: scen[meta[c]["vig"]] for c in ctx}
    scenarios = sorted(set(scen_of.values()))
    orig = {c for c in ctx if scen_of[c] <= 30}
    prim_pos = [c for c in ctx if c in orig and meta[c]["gold"] == "D"]
    prim_neg = [c for c in ctx if c in orig and meta[c]["gold"] not in ("D", "C/D")]
    exp_pos = [c for c in ctx if meta[c]["er"] == 1]; exp_neg = [c for c in ctx if meta[c]["er"] == 0]
    D_all = [c for c in ctx if meta[c]["gold"] == "D"]; CD_all = [c for c in ctx if meta[c]["gold"] == "C/D"]
    out = dict(cfg=cfg)
    rng = np.random.default_rng(1000 + CONFIGS.index(tuple(cfg)))  # deterministic per configuration

    def pairs(reg, subset):
        cs = [c for c in subset if c in beliefs and c in decisions[reg]]
        return cs, np.array([beliefs[c] for c in cs]), np.array([decisions[reg][c] for c in cs])

    # --- recovered ratio per regime (full data), in-sample agreement, majority, AUC(belief->decision)
    per_regime = {}
    for reg in REGIMES:
        cs, p, a = pairs(reg, ctx)
        f = fit_cost(p, a)
        rec = dict(n=len(cs), refer_rate=float(a.mean()) if len(a) else float("nan"))
        if f:
            rule = (p >= f["pstar"]).astype(int)
            rec.update(ratio=f["ratio"], pstar=f["pstar"], agree=float((rule == a).mean()),
                       majority=float(max(a.mean(), 1 - a.mean())), auc=auc(p, a))
        per_regime[reg] = rec
    out["per_regime"] = per_regime

    # --- hold-out by base scenario
    hold = {}
    for reg in REGIMES:
        ins, outs, maj, ratios = [], [], [], []
        for s in range(splits):
            perm = rng.permutation(scenarios); A = set(perm[: len(perm) // 2]); B = set(perm[len(perm) // 2:])
            cA, pA, aA = pairs(reg, [c for c in ctx if scen_of[c] in A]); cB, pB, aB = pairs(reg, [c for c in ctx if scen_of[c] in B])
            fA = fit_cost(pA, aA)
            if not fA or len(aB) == 0:
                continue
            t = fA["pstar"]
            ins.append(float(((pA >= t).astype(int) == aA).mean())); outs.append(float(((pB >= t).astype(int) == aB).mean()))
            maj.append(float(max(aB.mean(), 1 - aB.mean()))); ratios.append(math.log10(fA["ratio"]))
        hold[reg] = dict(n_splits=len(outs), in_sample=float(np.mean(ins)) if ins else float("nan"),
                         held_out=float(np.mean(outs)) if outs else float("nan"), held_out_sd=float(np.std(outs)) if outs else float("nan"),
                         majority=float(np.mean(maj)) if maj else float("nan"), sd_log10_ratio=float(np.std(ratios)) if ratios else float("nan"))
    out["holdout"] = hold

    # --- invariance across factorial conditions (default regime: cluster bootstrap over base scenarios)
    inv = {}
    for reg in REGIMES:
        cs, p, a = pairs(reg, ctx)
        sc = np.array([scen_of[c] for c in cs]); lv = {}
        for fac, l1, l2 in FACTORS:
            vals = np.array([meta[c][fac] for c in cs])
            lv[(fac, l1)] = vals == l1; lv[(fac, l2)] = vals == l2
        cell_mask = {cell: np.array([meta[c]["variant"] == cell for c in cs]) for cell in CELLS}
        f_all = fit_cost(p, a)
        point = {"overall": f_all}
        for key, m in lv.items():
            point[key] = fit_cost(p[m], a[m])
        cellpoint = {cell: fit_cost(p[m], a[m]) for cell, m in cell_mask.items()}
        rec = dict(overall=f_all["ratio"] if f_all else None,
                   levels={f"{k[0]}={k[1]}": (v["ratio"] if v else None) for k, v in point.items() if k != "overall"},
                   cells={cell: (v["ratio"] if v else None) for cell, v in cellpoint.items()})
        if reg == "decision_baseline" and draws > 0:
            boot_lv = defaultdict(list); boot_cell = defaultdict(list); boot_all = []
            scen_idx = {s: np.where(sc == s)[0] for s in scenarios}
            for d in range(draws):
                pick = rng.choice(scenarios, size=len(scenarios), replace=True)
                mult = Counter(pick); w = np.zeros(len(cs))
                for s, k in mult.items():
                    w[scen_idx[s]] = k
                fa = fit_cost(p, a, w); boot_all.append(fa["ratio"] if fa else None)
                for key, m in lv.items():
                    fw = fit_cost(p[m], a[m], w[m]); boot_lv[key].append(fw["ratio"] if fw else None)
                for cell, m in cell_mask.items():
                    fw = fit_cost(p[m], a[m], w[m]); boot_cell[cell].append(fw["ratio"] if fw else None)
            rec["overall_ci"] = pct(boot_all)
            rec["level_ci"] = {f"{k[0]}={k[1]}": pct(v) for k, v in boot_lv.items()}
            rec["cell_ci"] = {cell: pct(v) for cell, v in boot_cell.items()}
            diffs = {}
            for fac, l1, l2 in FACTORS:
                d_pt = (math.log10(point[(fac, l1)]["ratio"]) - math.log10(point[(fac, l2)]["ratio"])) if point[(fac, l1)] and point[(fac, l2)] else None
                dd = [math.log10(x) - math.log10(y) for x, y in zip(boot_lv[(fac, l1)], boot_lv[(fac, l2)]) if x and y and x > 0 and y > 0]
                diffs[fac] = dict(point=d_pt, ci=pct(dd))
            rec["diff_log10"] = diffs
            rec["cells_excluding_overall"] = sum(1 for cell in CELLS if rec["cells"][cell] and rec["overall"] and not (rec["cell_ci"][cell][0] <= rec["overall"] <= rec["cell_ci"][cell][1]))
        inv[reg] = rec
    out["invariance"] = inv

    # --- repeated belief elicitations (5 runs)
    runs = load_repetitions(cfg)
    if runs:
        common = [c for c in ctx if all(c in r for r in runs)]
        M = np.array([[r[c] for c in common] for r in runs])
        sd = M.std(axis=0)
        pair_abs = [np.abs(M[i] - M[j]).mean() for i in range(5) for j in range(i + 1, 5)]
        cs, p, a = pairs("decision_baseline", ctx)
        ratios, aucs_e, aucs_p = [], [], []
        for r in runs:
            cs_r = [c for c in cs if c in r]
            f = fit_cost(np.array([r[c] for c in cs_r]), np.array([decisions["decision_baseline"][c] for c in cs_r]))
            ratios.append(f["ratio"] if f else None)
            ce = [c for c in exp_pos + exp_neg if c in r]
            aucs_e.append(auc([r[c] for c in ce], [meta[c]["er"] for c in ce]))
            cp = [c for c in prim_pos + prim_neg if c in r]
            aucs_p.append(auc([r[c] for c in cp], [1 if meta[c]["gold"] == "D" else 0 for c in cp]))
        mean_b = {c: float(M[:, i].mean()) for i, c in enumerate(common)}
        cs_m = [c for c in cs if c in mean_b]
        fm = fit_cost(np.array([mean_b[c] for c in cs_m]), np.array([decisions["decision_baseline"][c] for c in cs_m]))
        out["repetitions"] = dict(n_contexts=len(common), mean_sd=float(sd.mean()), median_sd=float(np.median(sd)),
                                  frac_sd_zero=float((sd == 0).mean()), mean_pair_abs=float(np.mean(pair_abs)),
                                  ratios=ratios, ratio_mean_belief=fm["ratio"] if fm else None, auc_expanded=aucs_e, auc_primary=aucs_p,
                                  main_run_auc_expanded=auc([beliefs[c] for c in exp_pos + exp_neg if c in beliefs], [meta[c]["er"] for c in exp_pos + exp_neg if c in beliefs]))
    else:
        out["repetitions"] = None

    # --- misses among the 64, per-vignette beliefs, recoveries, over-triage
    d0 = decisions["decision_baseline"]
    miss = [c for c in prim_pos if d0.get(c) == 0]
    out["misses"] = dict(n=sum(1 for c in prim_pos if c in d0), missed=len(miss),
                         by_vig={v: sum(1 for c in miss if meta[c]["vig"] == v) for v in D_ORIG},
                         n_by_vig={v: sum(1 for c in prim_pos if meta[c]["vig"] == v and c in d0) for v in D_ORIG},
                         belief_by_vig={v: [float(np.mean([beliefs[c] for c in prim_pos if meta[c]["vig"] == v and c in beliefs])),
                                            float(np.min([beliefs[c] for c in prim_pos if meta[c]["vig"] == v and c in beliefs])),
                                            float(np.max([beliefs[c] for c in prim_pos if meta[c]["vig"] == v and c in beliefs]))] for v in D_ORIG},
                         recovered5=sum(1 for c in miss if decisions["decision_u_fp1_fn5"].get(c) == 1),
                         recovered10=sum(1 for c in miss if decisions["decision_u_fp1_fn10"].get(c) == 1),
                         fp=sum(1 for c in prim_neg if d0.get(c) == 1), n_neg=sum(1 for c in prim_neg if c in d0),
                         fp5=sum(1 for c in prim_neg if decisions["decision_u_fp1_fn5"].get(c) == 1), n_neg5=sum(1 for c in prim_neg if c in decisions["decision_u_fp1_fn5"]),
                         fp10=sum(1 for c in prim_neg if decisions["decision_u_fp1_fn10"].get(c) == 1), n_neg10=sum(1 for c in prim_neg if c in decisions["decision_u_fp1_fn10"]))
    # per-vignette baseline referrals (all 78) and per regime
    pv = {}
    for reg in REGIMES:
        dd = decisions[reg]; agg = defaultdict(lambda: [0, 0])
        for c in ctx:
            if c in dd:
                agg[meta[c]["vig"]][0] += dd[c]; agg[meta[c]["vig"]][1] += 1
        pv[reg] = {v: tuple(x) for v, x in agg.items()}
    out["per_vignette"] = pv

    # --- three-state referral rates with scenario-cluster CIs
    three = {}
    groups = {"D": D_all, "CD": CD_all, "nonD": exp_neg}
    for reg in REGIMES:
        dd = decisions[reg]; rec = {}
        for g, cs in groups.items():
            cs = [c for c in cs if c in dd]
            if not cs:
                rec[g] = (float("nan"), (float("nan"), float("nan"))); continue
            sc = np.array([scen_of[c] for c in cs]); a = np.array([dd[c] for c in cs])
            gs = sorted(set(sc)); sums = np.array([a[sc == s].sum() for s in gs]); cnts = np.array([(sc == s).sum() for s in gs])
            point = float(a.mean()); bs = []
            for d in range(max(draws, 200)):
                pick = rng.integers(0, len(gs), size=len(gs)); bs.append(sums[pick].sum() / cnts[pick].sum())
            rec[g] = (point, pct(bs))
        three[reg] = rec
    out["three_state"] = three

    # --- subgroup counts (for pooled tables in the parent)
    sub = {}
    for reg in REGIMES:
        dd = decisions[reg]; rec = {}
        for fac, l1, l2 in FACTORS:
            for lv_ in (l1, l2):
                tp = sum(dd[c] for c in prim_pos if meta[c][fac] == lv_ and c in dd); np_ = sum(1 for c in prim_pos if meta[c][fac] == lv_ and c in dd)
                fp = sum(dd[c] for c in prim_neg if meta[c][fac] == lv_ and c in dd); nn = sum(1 for c in prim_neg if meta[c][fac] == lv_ and c in dd)
                tpe = sum(dd[c] for c in D_all if meta[c][fac] == lv_ and c in dd); npe = sum(1 for c in D_all if meta[c][fac] == lv_ and c in dd)
                fpe = sum(dd[c] for c in exp_neg if meta[c][fac] == lv_ and c in dd); nne = sum(1 for c in exp_neg if meta[c][fac] == lv_ and c in dd)
                rec[f"{fac}={lv_}"] = dict(tp=tp, npos=np_, fp=fp, nneg=nn, tpe=tpe, npe=npe, fpe=fpe, nne=nne)
        sub[reg] = rec
    out["subgroups"] = sub

    # --- recalibration (expanded labels), cross-fitted isotonic by scenario folds
    cs = [c for c in ctx if c in beliefs]; p = np.array([beliefs[c] for c in cs]); y = np.array([meta[c]["er"] for c in cs]); sc = np.array([scen_of[c] for c in cs])
    f0 = per_regime["decision_baseline"]
    qs = []
    if "pstar" in f0:
        folds = np.array_split(np.random.default_rng(0).permutation(scenarios), 5)
        for fold in folds:
            train = ~np.isin(sc, fold)
            ux, fit = pav(p[train], y[train])
            qs.append(float(np.interp(f0["pstar"], ux, fit)))
        ux, fit = pav(p, y); q_in = float(np.interp(f0["pstar"], ux, fit))
    out["recalibration"] = dict(pstar=f0.get("pstar"), ratio=f0.get("ratio"), ece=ece(p, y), q_folds=qs,
                                q_mean=float(np.mean(qs)) if qs else None, q_in_sample=q_in if qs else None,
                                effective_ratio=((1 - np.mean(qs)) / np.mean(qs)) if qs and 0 < np.mean(qs) < 1 else None,
                                auroc_expanded=auc(p, y))

    # --- AUROC by scenario
    def A(P, N):
        S = [beliefs[c] for c in P + N if c in beliefs]; L = [1] * sum(1 for c in P if c in beliefs) + [0] * sum(1 for c in N if c in beliefs)
        return auc(S, L)
    pos_scen = sorted(set(scen_of[c] for c in exp_pos))
    loo = [A([c for c in exp_pos if scen_of[c] != s], exp_neg) for s in pos_scen]
    out["auroc"] = dict(primary=A(prim_pos, prim_neg), dka=A([c for c in prim_pos if meta[c]["vig"] in ("E13", "F13")], prim_neg),
                        asthma=A([c for c in prim_pos if meta[c]["vig"] in ("E9", "F9")], prim_neg), expanded=A(exp_pos, exp_neg),
                        loo_min=min(loo), loo_max=max(loo))
    return out


# ----------------------------------------------------------------------------- table helpers
def esc(s):
    return str(s).replace("%", "\\%").replace("_", "\\_").replace("&", "\\&")


def fmt(x, d=2):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def ci(x, d=2):
    return f"[{fmt(x[0], d)}, {fmt(x[1], d)}]"


def _plain(text):
    """Strip the light typesetting markup used when composing cells so the tables read as plain Markdown."""
    text = re.sub(r"\\multicolumn\{1\}\{c\}\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\rotatebox\{90\}\{([^}]*)\}", r"\1", text)
    text = text.replace("\\bar p", "mean p").replace("|\\Delta|", "|Δ|").replace("\\times", "×")
    text = text.replace("\\midrule ", "").replace("$\\cdot$", "·").replace("\\%", "%").replace("\\_", "_")
    text = re.sub(r"\\(textbf|emph|texttt)\{([^}]*)\}", r"\2", text)
    text = re.sub(r"~?\\ref\{[^}]*\}", "", text).replace("Table~", "Table ").replace("Fig.~", "Fig. ")
    text = text.replace("$p^*$", "p*").replace("$p \\ge p^*$", "p >= p*").replace("$n = 78$", "n = 78").replace("$\\approx 1$", "about 1")
    text = text.replace("``", "\"").replace("''", "\"").replace("--", "–").replace("\\ ", " ").replace("~", " ").replace("$", "")
    return text.strip()


def table(name, colspec, header, rows, caption, label, size="", notes="", longtable=False):
    """Write one table as Markdown (tables/<name>.md): a caption paragraph, then a pipe table."""
    cols = [_plain(c) for c in header.split(" & ")]
    lines = [f"**{_plain(caption)}**", "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_plain(c) for c in r.split(" & ")) + " |")
    if notes:
        lines += ["", _plain(notes)]
    (TABLES / f"{name}.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=500); ap.add_argument("--splits", type=int, default=50); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--from-cache", action="store_true")
    args = ap.parse_args()
    TABLES.mkdir(exist_ok=True)
    scen, diag, deployed = load_original()
    import pickle
    cache = HERE / "results_cache.pkl"
    if args.from_cache and cache.exists():
        results = pickle.load(open(cache, "rb"))
    else:
        with Pool(args.workers) as pool:
            results = pool.map(analyze, [(cfg, args.draws, args.splits, scen, deployed) for cfg in CONFIGS])
        pickle.dump(results, open(cache, "wb"))
    R = {cfg: r for cfg, r in zip(CONFIGS, results)}
    meta, _, _ = load_config(CONFIGS[0])
    ctx = sorted(meta)
    orig = {c for c in ctx if scen[meta[c]["vig"]] <= 30}
    prim_pos = [c for c in ctx if c in orig and meta[c]["gold"] == "D"]; prim_neg = [c for c in ctx if c in orig and meta[c]["gold"] not in ("D", "C/D")]
    D_all = [c for c in ctx if meta[c]["gold"] == "D"]; exp_neg = [c for c in ctx if meta[c]["er"] == 0]

    # deployed tool numbers
    dep = dict(sens=sum(deployed[c] for c in prim_pos) / len(prim_pos), fpr=sum(deployed[c] for c in prim_neg) / len(prim_neg),
               missed_by_vig={v: sum(1 for c in prim_pos if meta[c]["vig"] == v and deployed[c] == 0) for v in D_ORIG},
               sens_expanded_D=sum(deployed[c] for c in D_all) / len(D_all), fpr_expanded=sum(deployed[c] for c in exp_neg) / len(exp_neg))
    dep_sub = {}
    for fac, l1, l2 in FACTORS:
        for lv_ in (l1, l2):
            P = [c for c in prim_pos if meta[c][fac] == lv_]; N = [c for c in prim_neg if meta[c][fac] == lv_]
            dep_sub[f"{fac}={lv_}"] = (sum(deployed[c] for c in P) / len(P), sum(deployed[c] for c in N) / len(N))
    dep["subgroups"] = dep_sub

    # ---- Table: hold-out
    hdr = "Configuration & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{RLAB[r]}}}" for r in REGIMES)
    rows = []
    for cfg in CONFIGS:
        h = R[cfg]["holdout"]
        rows.append(cname(cfg) + " & " + " & ".join(f"{fmt(h[r]['held_out'])} ({fmt(h[r]['majority'])})" for r in REGIMES))
    table("tab_holdout", "l" + "r" * 8, hdr, rows,
          "Out-of-sample agreement of the recovered decision rule. For each configuration and prompted cost ratio (columns; FN:FP), the break-even probability was fitted on a random half of the 39 base scenarios and the rule ``refer if $p \\ge p^*$'' was scored on the other half (%d random splits). Cells give mean held-out agreement with the model's actual decisions, with the held-out majority-class rate in parentheses." % args.splits,
          "tab:holdout", size="\\scriptsize")
    hdr = "Configuration & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{RLAB[r]}}}" for r in REGIMES)
    rows = [cname(cfg) + " & " + " & ".join(fmt(R[cfg]["per_regime"][r].get("auc")) for r in REGIMES) for cfg in CONFIGS]
    table("tab_auc", "l" + "r" * 8, hdr, rows,
          "Threshold-free consistency: AUC of the elicited probability for predicting the model's own referral decision, by configuration and prompted cost ratio (FN:FP). Values near 1 indicate that decisions are a monotone function of the stated probability; 0.5 would indicate no relation.",
          "tab:auc", size="\\scriptsize")

    # ---- Table: invariance (default regime)
    hdr = "Configuration & Overall ratio [95\\% CI] & Black/White & Woman/Man & Anchor/None & Barrier/None & Labs/None & 16-cell range & Cells excl."
    rows = []
    for cfg in CONFIGS:
        v = R[cfg]["invariance"]["decision_baseline"]
        cells = [x for x in v["cells"].values() if x]
        def rr(fac):
            d = v["diff_log10"][fac]
            return f"{fmt(10 ** d['point'] if d['point'] is not None else None)} [{fmt(10 ** d['ci'][0])}, {fmt(10 ** d['ci'][1])}]"
        rows.append(f"{cname(cfg)} & {fmt(v['overall'],1)} {ci(v['overall_ci'],1)} & {rr('race')} & {rr('gender')} & {rr('has_anchor')} & {rr('has_barrier')} & {rr('prompt_type')} & {fmt(min(cells),1)}--{fmt(max(cells),1)} & {v['cells_excluding_overall']}/16")
    table("tab_invariance", "l" + "r" * 8, hdr, rows,
          "Invariance of the default (no stated priority) recovered cost ratio across the factorial design, all 1,248 case variants. ``Overall'' is the FN:FP ratio recovered from all variants. The next five columns give the ratio of recovered ratios between the two levels of each factor (e.g.\\ Black/White $= 1$ means identical recovered priorities), with 95\\%% intervals from a cluster bootstrap over the 39 base scenarios (%d draws; each draw keeps all 32 presentations of a selected scenario, so the interval accounts for the correlation among variants). ``16-cell range'' is the range of ratios recovered separately within each of the 16 factorial cells ($n = 78$ each); ``Cells excl.'' counts cells whose bootstrap interval excludes the configuration's overall ratio (expected $\\approx 1$ of 16 by chance)." % args.draws,
          "tab:invariance", size="\\scriptsize")
    # cells table (appendix)
    hdr = "Configuration & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{c.replace('-', '')}}}" for c in CELLS)
    rows = [cname(cfg) + " & " + " & ".join(fmt(R[cfg]["invariance"]["decision_baseline"]["cells"][c], 1) for c in CELLS) for cfg in CONFIGS]
    table("tab_cells", "l" + "r" * 16, hdr, rows,
          "Default recovered FN:FP cost ratio fitted separately within each of the 16 factorial cells ($n = 78$ variants each). Cell codes: W/B = White/Black, M/W = man/woman, A = anchoring statement present, X = access barrier present. Bootstrap intervals per cell are provided in \\texttt{cells\\_ci.csv}.",
          "tab:cells", size="\\tiny")
    with open(HERE / "cells_ci.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "reasoning", "cell", "ratio", "ci_lo", "ci_hi"])
        for cfg in CONFIGS:
            v = R[cfg]["invariance"]["decision_baseline"]
            for c in CELLS:
                w.writerow([cfg[0], cfg[1], c, v["cells"][c], v["cell_ci"][c][0], v["cell_ci"][c][1]])
    # invariance across regimes (appendix): max ratio-of-ratios across the five factors, per regime
    hdr = "Configuration & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{RLAB[r]}}}" for r in REGIMES)
    rows = []
    for cfg in CONFIGS:
        cells = []
        for r in REGIMES:
            v = R[cfg]["invariance"][r]; mx = None
            for fac, l1, l2 in FACTORS:
                a1 = v["levels"].get(f"{fac}={l1}"); a2 = v["levels"].get(f"{fac}={l2}")
                if a1 and a2:
                    q = max(a1 / a2, a2 / a1); mx = q if mx is None or q > mx else mx
            cells.append(fmt(mx))
        rows.append(cname(cfg) + " & " + " & ".join(cells))
    table("tab_invariance_regimes", "l" + "r" * 8, hdr, rows,
          "Largest ratio of recovered cost ratios between the two levels of any single factor (race, sex, anchoring, barrier, presentation), by configuration and prompted cost ratio (FN:FP). A value of 1.3 means the recovered priority differs by at most 30\\% between any two levels of any factor.",
          "tab:invariance_regimes", size="\\scriptsize")

    # ---- Table: repetitions
    hdr = "Configuration & Mean within-context SD & Mean $|\\Delta|$ between runs & Default ratio across 5 belief runs & Ratio, mean belief & AUROC (1,248) across runs & AUROC (576) across runs"
    rows = []
    for cfg in CONFIGS:
        rp = R[cfg]["repetitions"]
        if not rp:
            rows.append(cname(cfg) + " & -- & -- & -- & -- & -- & --"); continue
        rs = [x for x in rp["ratios"] if x]
        rows.append(f"{cname(cfg)} & {fmt(rp['mean_sd'],3)} & {fmt(rp['mean_pair_abs'],3)} & {fmt(min(rs),1)}--{fmt(max(rs),1)} & {fmt(rp['ratio_mean_belief'],1)} & {fmt(min(rp['auc_expanded']),3)}--{fmt(max(rp['auc_expanded']),3)} & {fmt(min(rp['auc_primary']),3)}--{fmt(max(rp['auc_primary']),3)}")
    table("tab_repetitions", "l" + "r" * 6, hdr, rows,
          "Stability under repeated sampling. Each configuration's belief prompt was issued five independent times for every one of the 1,248 case variants. Columns: mean across variants of the standard deviation of the five stated probabilities; mean absolute difference between pairs of runs; range of the default recovered FN:FP ratio when the decisions are re-analysed against each run's probabilities; the ratio recovered against the five-run mean probability; and the range of AUROC across runs for the expanded (1,248-case) and primary (576-case) endpoints.",
          "tab:repetitions", size="\\scriptsize")

    # ---- Table: misses among the 64
    hdr = "Configuration & Missed/64 & E9 & F9 & E13 & F13 & $\\bar p$ E9 & $\\bar p$ F9 & $\\bar p$ E13 & $\\bar p$ F13 & Recovered 5:1 & Recovered 10:1 & Over-triage default & Over-triage 5:1 & Over-triage 10:1"
    rows = []
    tot = 0; tot_mini = 0; s5 = 0; s5m = 0; s10 = 0; s10m = 0
    for cfg in CONFIGS:
        m = R[cfg]["misses"]; b = m["belief_by_vig"]
        rows.append(f"{cname(cfg)} & {m['missed']} & " + " & ".join(str(m["by_vig"][v]) for v in D_ORIG) + " & " + " & ".join(fmt(b[v][0]) for v in D_ORIG) +
                    f" & {m['recovered5']} & {m['recovered10']} & {100*m['fp']/m['n_neg']:.0f}\\% & {100*m['fp5']/m['n_neg5']:.0f}\\% & {100*m['fp10']/m['n_neg10']:.0f}\\%")
        tot += m["missed"]; s5 += m["recovered5"]; s10 += m["recovered10"]
        if cfg[0] == "gpt-5-mini":
            tot_mini += m["missed"]; s5m += m["recovered5"]; s10m += m["recovered10"]
    rows.append("\\midrule Deployed tool (Ramaswamy et al.) & " + str(sum(dep["missed_by_vig"].values())) + " & " + " & ".join(str(dep["missed_by_vig"][v]) for v in D_ORIG) + " & -- & -- & -- & -- & -- & -- & " + f"{100*dep['fpr']:.0f}\\% & -- & --")
    table("tab_misses", "l" + "r" * 14, hdr, rows,
          "Default-regime misses among the 64 definitive-emergency variants of the primary endpoint, by configuration and vignette. E9/F9 = acute asthma exacerbation with/without objective data; E13/F13 = diabetic ketoacidosis with/without objective data (16 demographic/context variants each). $\\bar p$ is the mean elicited probability of needing emergency care for the 16 variants. ``Recovered'' counts missed variants that are referred under the 5:1 and 10:1 safety-weighted prompts; ``Over-triage'' is the referral rate among the 512 non-emergency variants under the default, 5:1 and 10:1 regimes. Totals across the 18 configurations: %d misses (%d from GPT-5-mini), %d recovered at 5:1 (%d GPT-5-mini), %d at 10:1 (%d GPT-5-mini)." % (tot, tot_mini, s5, s5m, s10, s10m),
          "tab:misses", size="\\tiny")

    # ---- Table: per-vignette baseline referrals (appendix, landscape longtable)
    vigs = sorted(set(meta[c]["vig"] for c in ctx), key=lambda v: (scen[v], v))
    short = {cfg: f"{DISPLAY[cfg[0]].replace('DeepSeek-V4-', 'DS-').replace('Claude ', '')[:9]}/{cfg[1][:3]}" for cfg in CONFIGS}
    hdr = "Vignette & Scenario & Gold & " + " & ".join(f"\\rotatebox{{90}}{{{esc(short[cfg])}}}" for cfg in CONFIGS) + " & \\rotatebox{90}{Deployed}"
    rows = []
    for v in vigs:
        cs = [c for c in ctx if meta[c]["vig"] == v]
        dep_n = sum(deployed[c] for c in cs)
        rows.append(f"{v} & {esc(diag[v][:38])} & {meta[cs[0]]['gold']} & " + " & ".join(str(R[cfg]["per_vignette"]["decision_baseline"].get(v, (0, 0))[0]) for cfg in CONFIGS) + f" & {dep_n}")
    table("tab_vignettes", "llc" + "r" * 19, hdr, rows,
          "Default-regime referrals to emergency care out of 16 variants, for every vignette (78 = 39 base scenarios $\\times$ with/without objective data) and every configuration, with the deployed tool's published decisions in the last column. Gold: clinician-adjudicated triage level (D = emergency department now; C/D = edge case; A--C = non-emergency).",
          "tab:vignettes", size="\\tiny", longtable=True)
    with open(HERE / "per_vignette_per_regime.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["vignette", "scenario", "gold", "model", "reasoning", "regime_fn_fp", "referred", "n"])
        for v in vigs:
            g = meta[[c for c in ctx if meta[c]["vig"] == v][0]]["gold"]
            for cfg in CONFIGS:
                for r in REGIMES:
                    a, n = R[cfg]["per_vignette"][r].get(v, (0, 0)); w.writerow([v, diag[v], g, cfg[0], cfg[1], RLAB[r], a, n])

    # ---- Table: three-state (main: 4 regimes, point) and full (appendix, CIs)
    show = ["decision_baseline", "decision_u_fp5_fn1", "decision_u_fp1_fn5", "decision_u_fp1_fn10"]
    hdr = "Configuration & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{RLAB[r]}}}" for r in show)
    rows = [cname(cfg) + " & " + " & ".join(" / ".join(fmt(R[cfg]["three_state"][r][g][0]) for g in ("D", "CD", "nonD")) for r in show) for cfg in CONFIGS]
    table("tab_threestate", "l" + "r" * 4, hdr, rows,
          "Referral rate by clinical state (all 1,248 variants), under the default regime and three prompted cost ratios (FN:FP). Each cell gives the share referred among definitive emergencies (192; gold D) / edge cases (448; gold C/D, urgent but ED attendance not required by the adjudicators) / non-emergencies (608; gold A--C). Scenario-cluster intervals for every regime are in Table~\\ref{tab:threestate_full}.",
          "tab:threestate", size="\\small")
    hdr = "Configuration & FN:FP & Definitive emergencies [95\\% CI] & Edge cases [95\\% CI] & Non-emergencies [95\\% CI]"
    rows = []
    for cfg in CONFIGS:
        for r in REGIMES:
            t = R[cfg]["three_state"][r]
            rows.append(f"{cname(cfg)} & {RLAB[r]} & {fmt(t['D'][0])} {ci(t['D'][1])} & {fmt(t['CD'][0])} {ci(t['CD'][1])} & {fmt(t['nonD'][0])} {ci(t['nonD'][1])}")
    table("tab_threestate_full", "llrrr", hdr, rows,
          "Referral rate by clinical state for every configuration and regime, with 95\\% intervals from a cluster bootstrap over base scenarios (definitive emergencies: 6 scenarios; edge cases: 14; non-emergencies: 19).",
          "tab:threestate_full", size="\\scriptsize", longtable=True)

    # ---- Table: subgroups (pooled over 18 configurations), primary and expanded
    levels = [f"{fac}={l}" for fac, l1, l2 in FACTORS for l in (l1, l2)]
    lab = {f"{fac}={l}": FACTOR_LABEL[fac][i] for fac, l1, l2 in FACTORS for i, l in enumerate((l1, l2))}
    for kind, name, cap in (("prim", "tab_subgroups_primary", "Sensitivity on the 64 definitive-emergency variants / over-triage rate on the 512 non-emergency variants of the primary endpoint"),
                            ("exp", "tab_subgroups_expanded", "Sensitivity on the 192 definitive-emergency variants / over-triage rate on the 608 non-emergency variants of the expanded case set")):
        hdr = "FN:FP & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{lab[l]}}}" for l in levels)
        rows = []
        for r in REGIMES:
            cells = []
            for l in levels:
                tp = sum(R[cfg]["subgroups"][r][l]["tp" if kind == "prim" else "tpe"] for cfg in CONFIGS); np_ = sum(R[cfg]["subgroups"][r][l]["npos" if kind == "prim" else "npe"] for cfg in CONFIGS)
                fp = sum(R[cfg]["subgroups"][r][l]["fp" if kind == "prim" else "fpe"] for cfg in CONFIGS); nn = sum(R[cfg]["subgroups"][r][l]["nneg" if kind == "prim" else "nne"] for cfg in CONFIGS)
                cells.append(f"{tp/np_:.2f} / {fp/nn:.2f}")
            rows.append(RLAB[r] + " & " + " & ".join(cells))
        if kind == "prim":
            rows.append("\\midrule Deployed tool & " + " & ".join(f"{dep_sub[l][0]:.2f} / {dep_sub[l][1]:.2f}" for l in levels))
        table(name, "l" + "r" * 10, hdr, rows,
              cap + ", by patient race, sex, anchoring statement, access barrier and presentation (with/without objective data), pooled over the 18 configurations, for the default regime and each prompted cost ratio (FN:FP). Per-configuration values are provided in \\texttt{subgroups\\_per\\_config.csv}.",
              f"tab:{name[4:]}", size="\\scriptsize")
    with open(HERE / "subgroups_per_config.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "reasoning", "regime_fn_fp", "factor_level", "sens_primary", "n_pos_primary", "overtriage_primary", "n_neg_primary", "sens_expanded", "n_pos_expanded", "overtriage_expanded", "n_neg_expanded"])
        for cfg in CONFIGS:
            for r in REGIMES:
                for l in levels:
                    s = R[cfg]["subgroups"][r][l]
                    w.writerow([cfg[0], cfg[1], RLAB[r], l, s["tp"] / max(1, s["npos"]), s["npos"], s["fp"] / max(1, s["nneg"]), s["nneg"], s["tpe"] / max(1, s["npe"]), s["npe"], s["fpe"] / max(1, s["nne"]), s["nne"]])

    # ---- Table: recalibration
    hdr = "Configuration & Nominal ratio & Nominal $p^*$ & ECE & Calibrated $p^*$ (cross-fitted) & Effective ratio & AUROC"
    rows = []
    for cfg in CONFIGS:
        rc = R[cfg]["recalibration"]
        rows.append(f"{cname(cfg)} & {fmt(rc['ratio'],1)} & {fmt(rc['pstar'])} & {fmt(rc['ece'],3)} & {fmt(rc['q_mean'])} ({fmt(min(rc['q_folds']))}--{fmt(max(rc['q_folds']))}) & {fmt(rc['effective_ratio'],1)} & {fmt(rc['auroc_expanded'],3)}")
    table("tab_recalibration", "l" + "r" * 6, hdr, rows,
          "Nominal versus effective default priorities on the expanded case set. Nominal ratio and $p^*$ are the default FN:FP cost ratio and break-even probability recovered on the model's own probability scale. ECE is the 10-bin expected calibration error against the clinician labels. Calibrated $p^*$ is the observed emergency frequency at the nominal threshold, obtained by isotonic regression fitted on four fifths of the base scenarios and evaluated on the held-out fifth (mean and range over five folds). The effective ratio, $(1-q^*)/q^*$, is the cost ratio a calibrated decision-maker would need in order to place the threshold at the same observed risk.",
          "tab:recalibration", size="\\scriptsize")

    # ---- Table: AUROC by scenario
    hdr = "Configuration & Primary (64 vs 512) & DKA only & Asthma only & Expanded (640 vs 608) & Leave-one-scenario-out range"
    rows = [f"{cname(cfg)} & {fmt(R[cfg]['auroc']['primary'],3)} & {fmt(R[cfg]['auroc']['dka'],3)} & {fmt(R[cfg]['auroc']['asthma'],3)} & {fmt(R[cfg]['auroc']['expanded'],3)} & {fmt(R[cfg]['auroc']['loo_min'],3)}--{fmt(R[cfg]['auroc']['loo_max'],3)}" for cfg in CONFIGS]
    table("tab_auroc", "l" + "r" * 5, hdr, rows,
          "Dependence of discrimination on the positive scenarios. AUROC of the elicited probabilities on the primary endpoint using all 64 emergency variants, only the 32 DKA variants, or only the 32 asthma variants against the same 512 non-emergency variants; on the expanded set; and the range obtained when each of the 20 positive scenarios of the expanded set is left out in turn.",
          "tab:auroc", size="\\scriptsize")

    # ---- results.json
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    summary = dict(settings=dict(draws=args.draws, splits=args.splits), deployed=dep,
                   totals=dict(misses=tot, misses_gpt5mini=tot_mini, recovered5=s5, recovered5_gpt5mini=s5m, recovered10=s10, recovered10_gpt5mini=s10m),
                   configs={f"{cfg[0]}|{cfg[1]}": {k: v for k, v in R[cfg].items() if k not in ("per_vignette", "subgroups")} for cfg in CONFIGS})
    (HERE / "results.json").write_text(json.dumps(clean(summary), indent=1))
    print("tables written to", TABLES)
    print(f"totals: misses={tot} (gpt-5-mini {tot_mini}); recovered@5={s5} ({s5m}); recovered@10={s10} ({s10m})")
    print(f"deployed: sens={dep['sens']:.3f} fpr={dep['fpr']:.3f} missed_by_vig={dep['missed_by_vig']}")


if __name__ == "__main__":
    main()
