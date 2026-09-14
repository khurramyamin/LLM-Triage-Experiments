# Manuscript figures

These are the exact image files that appear in the main manuscript, listed by
figure number. The eight ROC figures (Figures 1 and 2 and Extended Data Figures 1 and 2)
are the five-run versions with bootstrap confidence bands; `python run_analysis.py`
regenerates them into `regenerated/five_run/` under the same file names. Every other
figure was produced from a single belief elicitation per case; `python run_analysis.py
--single-run` regenerates those into `regenerated/single_run/`. Both folders are created
on demand and are not tracked in the repository. The seven PDF figures come from the
scripts in `supplemental_analysis/`.

| Manuscript figure | File | Content |
|---|---|---|
| Figure 1 | `fig1_emergency_triage_performance.png` | ROC curves, deployed-tool point, prompted operating points and best fixed utility, primary endpoint (576 cases) |
| Figure 2 | `fig1_original_emergency_triage_performance.png` | Same for the expanded case set (1,248 cases) |
| Figure 3 | `fig1b_default_utilities.png` | Default recovered FN:FP ratios by model family and reasoning level |
| Figure 4 | `fig2_recovered_capability.png` | Recovered versus prompted cost ratio (decision-analytic steering) |
| Extended Data Figure 1a-c | `fig3_roc_grid_gpt.png`, `fig3_roc_grid_deepseek.png`, `fig3_roc_grid_claude.png` | ROC grids for all 18 configurations, primary endpoint |
| Extended Data Figure 2a-c | `fig3_original_roc_grid_gpt.png`, `fig3_original_roc_grid_deepseek.png`, `fig3_original_roc_grid_claude.png` | ROC grids, expanded case set |
| Extended Data Figure 3a-c | `decision_belief_consistency.png`, `nature_decision_consistency.png`, `fig5_rule_generalization.pdf` | Decisions consistent with elicited probability and recovered utility; agreement with the published ChatGPT Health decisions; the rule scored on scenarios it was not fitted to |
| Extended Data Figure 4 | `fig4_recovered_threshold.png` | Recovered versus prompted probability threshold |
| Extended Data Figure 5a-b | `fig_belief_distributions.png`, `fig_original_belief_distributions.png` | Elicited probability distributions with default and best fixed thresholds |
| Extended Data Figure 6a-b | `fig_calibration_curves.png`, `fig_original_calibration_curves.png` | Calibration curves and expected calibration error |
| Extended Data Figure 7 | `fig_prompt_context.png`, `fig_prompt_belief.png` (a), `fig_prompt_baseline.png` (b), `fig_prompt_utility.png` (c), `fig_prompt_threshold.png` (d) | The four prompts and the shared patient context |
| Extended Data Figure 8a-b | `extfig8a_invariance.pdf`, `extfig8b_subgroups.pdf` | Recovered priority across the factorial variants; error-rate differences between patient variants by model |
| Extended Data Figure 9a-b | `extfig9a_threestate.pdf`, `extfig9b_vignettes.pdf` | Referral rate by clinical state; referrals for every vignette and configuration |
| Extended Data Figure 10a-b | `extfig10a_stability.pdf`, `extfig10b_misses.pdf` | Stability under repeated elicitation; where the default-regime misses are |
