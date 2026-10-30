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
python supplemental_analysis/supplemental_subgroup_ci.py   # subgroup error-rate differences per configuration, with bootstrap intervals (subgroup_ci.json)
python supplemental_analysis/supplemental_invariance_p.py  # p-values and a Benjamini-Hochberg correction for the invariance contrasts (invariance_p.json)
python supplemental_analysis/supplemental_unseen_auc.py    # belief-to-decision AUC restricted to the held-out scenarios (unseen_auc.json)
python supplemental_analysis/supplemental_figures3.py      # the invariance figure, coloured by significance
python supplemental_analysis/supplemental_figures4.py      # the per-model subgroup figure
python supplemental_analysis/supplemental_consistency.py  # majority baseline, slope, split vignettes, misfit, fixed-threshold repeats (consistency.json, CSVs)
python supplemental_analysis/supplemental_pairs.py        # matched-pair probability changes and referral switches (matched_pairs.json, matched_pairs.csv)
python supplemental_analysis/supplemental_figures5.py     # figures for the two scripts above
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
| Threshold rule against a majority-decision baseline, both chosen on the same fitting half and scored on the held-out half; absolute and balanced accuracy | `figures/fig_majority_baseline.*`, `holdout_majority_baseline.csv` | How much does the threshold add beyond predicting the most common decision? |
| Matched pairs of variants (same vignette, other three factors fixed): change in elicited probability and referral switches in each direction, with scenario-cluster intervals, under no stated priority and the 5:1 and 10:1 prompts | `figures/fig_matched_pairs.*`, `matched_pairs.csv`, `matched_pairs.json` | What changes for the same clinical facts? |
| Recovered ratio in each of the 16 factorial conditions relative to the configuration's overall ratio | `figures/fig_cells.*`, `cells_ci.csv` | Which conditions depart from the overall ratio? |
| Ordering within vignettes whose variants received split decisions; departures from the fitted rule by vignette and by factor; fitted slope next to the break-even probability | `figures/fig_misfit.*`, `split_vignettes.csv`, `misfit_by_vignette.csv`, `misfit_by_factor.csv`, `slope_threshold.csv` | How well does a single ratio describe the decisions? |
| Clinician-label-selected threshold (5:1 target) held fixed across the five probability elicitations | `fixed_threshold_repeats.csv` | How often does a fixed external threshold change the recommended action? |

`results.json` collects the per-configuration numbers behind all of the above. Bootstrap intervals resample the 39 base clinical scenarios (keeping all 32 presentations of a scenario together) with 500 draws; the hold-out analysis uses 50 random half-splits of the scenarios. The logistic cost fit is the one used in the manuscript (`analysis/metrics.py`, `fit_cost_function`).

`consistency.json` and `matched_pairs.json` hold the per-configuration numbers behind these five rows. The majority baseline uses the same random half-splits as the hold-out analysis, and the matched-pair intervals resample the 39 base scenarios with 500 draws.
