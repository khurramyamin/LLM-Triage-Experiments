#!/usr/bin/env python
"""Belief-level invariance across the 16 factorial variants and decision noise at identical stated probabilities.
Existing elicitations only. Writes tables/tab_belief_invariance.md and addendum.json."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
csv.field_size_limit(sys.maxsize)
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from supplemental_analysis import CONFIGS, DISPLAY, cname, load_config, load_original, table, fmt

scen, diag, deployed = load_original()
rows, J = [], {}
for cfg in CONFIGS:
    meta, beliefs, decisions = load_config(cfg)
    byvig = defaultdict(list)
    for c, b in beliefs.items():
        byvig[meta[c]["vig"]].append(c)
    sds, identical, flips_num, flips_den, vig_with_flip = [], 0, 0, 0, 0
    d0 = decisions["decision_baseline"]
    for v, cs in byvig.items():
        b = np.array([beliefs[c] for c in cs]); sds.append(b.std())
        if b.std() == 0: identical += 1
        anyflip = False
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if beliefs[cs[i]] == beliefs[cs[j]] and cs[i] in d0 and cs[j] in d0:
                    flips_den += 1
                    if d0[cs[i]] != d0[cs[j]]:
                        flips_num += 1; anyflip = True
        vig_with_flip += anyflip
    # mean belief difference between factor levels, paired within vignette
    diffs = {}
    for fac, l1, l2 in [("race", "Black", "White"), ("gender", "woman", "man"), ("has_anchor", "yes", "no"), ("has_barrier", "yes", "no")]:
        per = []
        for v, cs in byvig.items():
            a = [beliefs[c] for c in cs if meta[c][fac] == l1]; b = [beliefs[c] for c in cs if meta[c][fac] == l2]
            if a and b: per.append(np.mean(a) - np.mean(b))
        diffs[fac] = float(np.mean(per))
    # labs vs symptoms-only: paired by scenario (E vs F)
    lab = []
    for s in set(scen.values()):
        e = [beliefs[c] for c in beliefs if scen[meta[c]["vig"]] == s and meta[c]["prompt_type"] == "1"]
        f = [beliefs[c] for c in beliefs if scen[meta[c]["vig"]] == s and meta[c]["prompt_type"] == "2"]
        if e and f: lab.append(np.mean(e) - np.mean(f))
    rec = dict(mean_within_vignette_sd=float(np.mean(sds)), vignettes_identical=identical, n_vignettes=len(byvig),
               diff_black_white=diffs["race"], diff_woman_man=diffs["gender"], diff_anchor=diffs["has_anchor"], diff_barrier=diffs["has_barrier"],
               diff_labs=float(np.mean(lab)), same_belief_pairs=flips_den, flip_pairs=flips_num, flip_rate=flips_num / flips_den if flips_den else None, vignettes_with_flip=vig_with_flip)
    J[f"{cfg[0]}|{cfg[1]}"] = rec
    rows.append(f"{cname(cfg)} & {fmt(rec['mean_within_vignette_sd'],3)} & {identical}/{len(byvig)} & {fmt(diffs['race'],3)} & {fmt(diffs['gender'],3)} & {fmt(diffs['has_anchor'],3)} & {fmt(diffs['has_barrier'],3)} & {fmt(rec['diff_labs'],3)} & {fmt(rec['flip_rate'],3)} & {vig_with_flip}/{len(byvig)}")
hdr = "Configuration & Within-vignette SD & Vignettes with identical $p$ & Black$-$White & Woman$-$Man & Anchor$-$None & Barrier$-$None & Labs$-$None & Decision flips at identical $p$ & Vignettes with a flip"
table("tab_belief_invariance", "l" + "r" * 9, hdr, rows,
      "Stated probabilities across informationally equivalent variants. Within-vignette SD is the standard deviation of the elicited probability across the 16 demographic/context variants of a vignette, averaged over the 78 vignettes; the next column counts vignettes whose 16 variants received exactly the same probability. The difference columns give the mean paired difference in elicited probability between factor levels (within vignette; for objective data, within base scenario). ``Decision flips'' is the share of within-vignette variant pairs with identical stated probability whose default decisions nevertheless differ, i.e.\\ decision noise near the threshold that no demographic or context factor explains.",
      "tab:belief_invariance", size="\\scriptsize")
(HERE / "addendum.json").write_text(json.dumps(J, indent=1))
print("\n".join(rows))
