#!/usr/bin/env python
"""Small tables (<=12 rows, <=6 columns) for the supplemental analyses, from the cached analysis."""
import pickle, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from supplemental_analysis import CONFIGS, FACTORS, FACTOR_LABEL, table
R = {tuple(r["cfg"]): r for r in pickle.load(open(HERE / "results_cache.pkl", "rb"))}
show = [("decision_baseline", "none"), ("decision_u_fp5_fn1", "0.2"), ("decision_u_fp1_fn1", "1"), ("decision_u_fp1_fn5", "5"), ("decision_u_fp1_fn10", "10")]
rows = []
for fac, l1, l2 in [x for x in FACTORS if x[0] != "prompt_type"]:
    for i, lv in enumerate((l1, l2)):
        cells = []
        for reg, _ in show:
            tp = sum(R[c]["subgroups"][reg][f"{fac}={lv}"]["tp"] for c in CONFIGS); npos = sum(R[c]["subgroups"][reg][f"{fac}={lv}"]["npos"] for c in CONFIGS)
            fp = sum(R[c]["subgroups"][reg][f"{fac}={lv}"]["fp"] for c in CONFIGS); nneg = sum(R[c]["subgroups"][reg][f"{fac}={lv}"]["nneg"] for c in CONFIGS)
            cells.append(f"{tp/npos:.2f} / {fp/nneg:.2f}")
        rows.append(f"{FACTOR_LABEL[fac][i].capitalize()} & " + " & ".join(cells))
hdr = "Patient variant & \\multicolumn{1}{c}{Default} & \\multicolumn{1}{c}{0.2} & \\multicolumn{1}{c}{1} & \\multicolumn{1}{c}{5} & \\multicolumn{1}{c}{10}"
table("tab_subgroups", "lrrrrr", hdr, rows,
      "Sensitivity on the 64 definitive-emergency variants / over-triage rate on the 512 non-emergency variants of the primary endpoint, by patient race, sex, anchoring statement and access barrier, pooled over the 18 configurations, under the default regime and four prompted FN:FP cost ratios. Per-configuration values for all seven prompted ratios and for the expanded case set are provided as a supplementary CSV file.",
      "tab:subgroups", size="\\small")
print("wrote tables/tab_subgroups.md")
