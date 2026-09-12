# Manuscript figures

These are the exact image files that appear in the main manuscript, listed by
figure number. `python run_analysis.py --single-run` regenerates every one of them into
`regenerated/single_run/` under the same file names, and `python run_analysis.py` writes the
five-run versions with bootstrap bands into `regenerated/five_run/`; those folders are created
on demand and are not tracked in the repository.

| Manuscript figure | File | Content |
|---|---|---|
| Figure 1 | `fig1_emergency_triage_performance.png` | ROC curves, deployed-tool point, prompted operating points and best fixed utility, primary endpoint (576 cases) |
| Figure 2 | `fig1_original_emergency_triage_performance.png` | Same for the expanded case set (1,248 cases) |
| Figure 3 | `fig1b_default_utilities.png` | Default recovered FN:FP ratios by model family and reasoning level |
| Figure 4 | `fig2_recovered_capability.png` | Recovered versus prompted cost ratio (decision-analytic steering) |
| Extended Data Figure 1a-c | `fig3_roc_grid_gpt.png`, `fig3_roc_grid_deepseek.png`, `fig3_roc_grid_claude.png` | ROC grids for all 18 configurations, primary endpoint |
| Extended Data Figure 2a-c | `fig3_original_roc_grid_gpt.png`, `fig3_original_roc_grid_deepseek.png`, `fig3_original_roc_grid_claude.png` | ROC grids, expanded case set |
| Extended Data Figure 3 | `decision_belief_consistency.png` | Decisions consistent with elicited probability and recovered utility |
| Extended Data Figure 4 | `nature_decision_consistency.png` | Agreement with the published ChatGPT Health decisions |
| Extended Data Figure 5 | `fig4_recovered_threshold.png` | Recovered versus prompted probability threshold |
| Extended Data Figure 6a-b | `fig_belief_distributions.png`, `fig_original_belief_distributions.png` | Elicited probability distributions with default and best fixed thresholds |
| Extended Data Figure 7a-b | `fig_calibration_curves.png`, `fig_original_calibration_curves.png` | Calibration curves and expected calibration error |
| Extended Data Figure 8 | `fig_prompt_context.png`, `fig_prompt_belief.png` (a), `fig_prompt_baseline.png` (b), `fig_prompt_utility.png` (c), `fig_prompt_threshold.png` (d) | The four prompts and the shared patient context |
