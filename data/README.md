# Analysis-ready data

The seven compressed files in `models/` contain the successful parsed model
responses needed by the paper analysis. They include all reasoning levels and
both utility- and threshold-prompting experiments. Raw responses, prompts,
retry logs, API configuration, and credentials are intentionally excluded.

Model columns:

- `model`, `reasoning_effort`, `experiment`
- `context_id`, `case_number`, `gold_standard`
- `regime`, `c_fp`, `c_fn`, `p_star`
- `parsed_probability`, `parsed_decision`
- `their_triage` (the published ChatGPT Health decision for the matched case)

`original_paper/DataExpanded_FINAL.csv.gz` retains the compact columns needed
to validate and reproduce the matched ChatGPT Health comparison. All files are
read directly with Python's standard `csv` and `gzip` modules.

## Belief repetitions

`belief_repetitions/repetition_{1..5}/<model>_re-<reasoning>/results.csv.gz`
holds five independent runs of the belief prompt for every one of the 1,248
contexts and each of the 18 model configurations (90 files). Each row records
`context_id`, the case metadata, the raw model reply and `parsed_probability`.
`run_analysis.py` uses the mean of the five probabilities as the displayed
belief and resamples both the runs and the base clinical cases in its
hierarchical bootstrap; `run_analysis.py --single-run` ignores this folder and
reproduces the single-elicitation figures of the manuscript.
