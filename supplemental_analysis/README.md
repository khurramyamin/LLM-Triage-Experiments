# Supplemental analyses

Supplemental analyses of the manuscript's elicitations. Every result is computed from
the data already shipped in `data/`; no model was queried.
The scripts run from the repository root with the same dependencies as
`run_analysis.py`.

```bash
python supplemental_analysis/supplemental_analysis.py     # statistics, Markdown tables, CSV supplements, results.json (about 2 min with 6 workers)
python supplemental_analysis/supplemental_addendum.py     # belief spread across the 16 variants and decision noise (addendum.json)
python supplemental_analysis/supplemental_figures.py      # figures (PDF and PNG) from the cached results
python supplemental_analysis/supplemental_figures2.py     # condensed variants of three figures used in the supplemental analyses
python supplemental_analysis/supplemental_tables.py       # the subgroup table used in the supplemental analyses
```

`supplemental_analysis.py --from-cache` reuses `results_cache.pkl` (the per-configuration results, including every bootstrap interval) instead of recomputing; the figure and table scripts always read the cache.

## What is computed

| Analysis | Output | Question addressed |
|---|---|---|
| Recovered decision rule fitted on a random half of the 39 base scenarios and scored on the other half (50 splits); AUC of the elicited probability for predicting the model's own decision | `figures/fig_holdout.*`, `tables/tab_holdout.md`, `tables/tab_auc.md` | Does the recovered ratio describe decisions beyond the cases it was fitted to? |
| Recovered ratio fitted within each level of race, sex, anchoring and access barrier, with the ratio of ratios and scenario-cluster bootstrap intervals; per-cell fits | `figures/fig_invariance.*`, `tables/tab_invariance*.md`, `tables/tab_cells.md`, `cells_ci.csv` | Is the recovered ratio invariant to informationally equivalent presentations? |
| Recovered ratio re-fitted against each of the five belief repetitions; within-vignette spread of beliefs; decisions that differ at identical stated probability | `figures/fig_stability.*`, `tables/tab_repetitions.md`, `addendum.json` | Is it stable under repeated sampling? |
| Default-regime misses among the 64 definitive-emergency variants by configuration and vignette, mean beliefs, recoveries under safety-weighted prompts, over-triage | `figures/fig_misses.*`, `tables/tab_misses.md` | Where do the misses come from? |
| Referral rate among definitive emergencies, edge cases and non-emergencies under every regime, with scenario-cluster intervals; referrals per vignette | `figures/fig_threestate.*`, `figures/fig_vignettes.*`, `tables/tab_threestate*.md`, `per_vignette_per_regime.csv` | Under-triage, over-triage and edge cases reported separately |
| Sensitivity and over-triage by race, sex, anchoring, barrier under every regime | `figures/fig_subgroups.*`, `tables/tab_subgroups*.md`, `subgroups_per_config.csv` | Error rates by patient subgroup |
| AUROC when only one of the two primary-endpoint emergency scenarios is used; leave-one-scenario-out on the expanded set | `tables/tab_auroc.md` | Sensitivity of discrimination to the positive scenarios |

`results.json` collects the per-configuration numbers behind all of the above. Bootstrap intervals resample the 39 base clinical scenarios (keeping all 32 presentations of a scenario together) with 500 draws; the hold-out analysis uses 50 random half-splits of the scenarios. The logistic cost fit is the one used in the manuscript (`analysis/metrics.py`, `fit_cost_function`).
