# Triage as a Decision Problem: Beliefs vs. Utilities in LLM Emergency Triage

Code and data for the *Nature Medicine* commentary re-analysing the ChatGPT Health
emergency-triage stress test. We argue that emergency triage is a **decision
problem**, not merely a prediction problem, and separate two things that a raw
accuracy score conflates:

* **Beliefs** — how well a model's elicited probability of "needs emergency care"
  *ranks* patients (a capability question), and
* **Utility** — *where the model draws the line* between referring and not
  referring (a value choice that can be steered by prompting).

Using a revealed-preference procedure, we elicit each model's beliefs and
decisions separately, recover the implicit false-negative / false-positive cost
ratio each model behaves as if it holds, and show that stating a priority (as a
cost ratio *or* as an equivalent probability threshold) moves capable models
along their ROC curve as intended.

## Repository layout

```
.
├── run_analysis.py                # one-command offline analysis and figure rebuild
├── analysis/                      # data loading, statistics, plots, and prompt figures
├── analysis_summary.json          # generated machine-readable analysis results
├── count_lines.py                 # verifies the public code stays below 5,000 lines
│
├── generic_RoC/                   # portable ROC + utility tool for any labeled dataset
├── data/                          # analysis-ready model data, comparison data, belief repetitions
├── nature_medicine_paper/
│   └── figures/                   # the exact figure files used in the manuscript (+ regenerated copies)
├── supplemental_analysis/         # supplemental analyses, figures and tables
└── tests/                         # focused statistical and command-line tests
```

## Data

**[`data/`](data/README.md)** has everything the paper's figures consume, in one
place: one compressed CSV per model containing every successful parsed belief
and decision needed for the analysis across all reasoning efforts and both
experiments, plus the original study's published decisions used for the
deployed-tool comparison, and the five independent repetitions of the belief
prompt for every configuration (`data/belief_repetitions/`) that the confidence
bands are computed from. Raw responses, failed API attempts, credentials, and
collection logs are intentionally excluded.

## Generic ROC / utility tool

`generic_RoC/` packages the belief-vs-utility analysis as a **standalone,
dataset-agnostic** command-line tool. Given any CSV of elicited **beliefs**
(probabilities), ground-truth **labels**, and optional observed **decisions**,
it builds the belief ROC curve, backs out the FN/FP cost ratio the data behaves
*as if* it holds, lets you name a target cost ratio (FN:FP) to evaluate by, and
finds the **best fixed utility ratio** — the operating point on the ROC that
minimises the target-weighted cost, expressed as the FN/FP ratio you would
prompt an LLM with on unseen data. It depends only on `numpy`, `scipy`, and
`matplotlib` and runs fully offline; see
[`generic_RoC/README.md`](generic_RoC/README.md) for the input schema and usage.

Example:

```bash
cd generic_RoC
python generic_roc.py --input example_dataset.csv \
    --decision-col decision --cost-ratio 5:1
```

The command writes `roc.png` and `summary.json`.

## Method in brief

For each clinical vignette (expanded over a race × gender × anchoring ×
access-barrier factorial design), the study issued, in separate queries:

1. a **belief** prompt — the model's probability that the patient needs emergency care;
2. a **baseline decision** prompt — refer or not, with no stated priorities;
3. **utility-prompted decisions** — refer-or-not under an explicit cost ratio
   (FN\:FP ∈ {.01, .1, .2, 1, 5, 10, 100}); and
4. **threshold-prompted decisions** — the concordant probability threshold
   `p* = c_FP / (c_FP + c_FN)` implied by each of those cost ratios.

From matched (belief, decision) pairs we fit a discrete-choice (logistic) cost
function and read off the recovered FN/FP ratio — the priority the model behaved
*as if* it held. Beliefs are scored against the gold labels with a tie-aware ROC
analysis, so any recovered ratio maps to an operating point on the belief ROC.
When the local five-run belief collection is available, the displayed score for
each context is its mean probability over the five prompt repetitions. Pointwise
95% bands use a hierarchical nonparametric bootstrap: each draw resamples the
five prompt runs and the 78 clinical `case_id` clusters, retaining all 16
correlated factorial variants of every selected case.

## Reproducing the figures

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate

pip install -r requirements.txt
python run_analysis.py --single-run   # the 22 figures as they appear in the manuscript
python run_analysis.py                # the same figures using the five-run mean belief, with bootstrap bands
```

The analysis runs entirely offline from the included data.
`nature_medicine_paper/figures/` holds the exact figure files included in the
manuscript; [`nature_medicine_paper/figures/README.md`](nature_medicine_paper/figures/README.md)
maps each file to its figure number in the paper. `run_analysis.py` never
overwrites them: `--single-run` (one belief elicitation per case, as in the
manuscript) writes to `nature_medicine_paper/figures/regenerated/single_run/`
and `analysis_summary_single_run.json`; the default run uses the five
repetitions in `data/belief_repetitions/` and writes to
`nature_medicine_paper/figures/regenerated/five_run/` and `analysis_summary.json`.
The `regenerated/` folders are created by the command and are not tracked in
the repository, so the only figure files committed are the manuscript's.
Both summaries record the numerical results, sample coverage, bootstrap
settings, every confidence interval, and the output manifest.

## Analysis conventions

- Primary endpoint: 576 intended cases, excluding `C/D` cases; positives are `D`
- Expanded endpoint: all 1,248 intended cases; positives contain `D`
- Utility ratios: `.01`, `.1`, `.2`, `1`, `5`, `10`, and `100`
- Recovered utilities: discrete-choice logit fit
- ROC confidence bands: 1,000 hierarchical bootstrap draws with seed `0`
- Other confidence intervals: 500 bootstrap draws with seed `0`
- Calibration: 10-bin expected calibration error
- Missing or unparseable responses: a present repetition collection must have
  all five parsed beliefs for every paper context; partial collections stop the
  canonical figure build. If no repetition collection is present, the original
  one-run analysis remains available for backward compatibility.
- ROC operating points: restricted to cases with both a belief and decision

## Supplemental analyses

`supplemental_analysis/` contains supplemental analyses computed from the same
elicitations with no new model queries:
held-out validation of the recovered decision rule (fitted on half of the base
scenarios, scored on the other half), invariance of the recovered ratio across
the 16 factorial cells, stability under the five repeated belief elicitations,
misses by scenario and presentation, referral rates by clinical state, and
error rates by patient subgroup under every prompting regime. See
[`supplemental_analysis/README.md`](supplemental_analysis/README.md) for the scripts, the
figures and tables they produce, and the cached results.

## Validation

```bash
python -m unittest discover -s tests -p "test_*.py"
python count_lines.py
```

The test suite covers endpoint definitions, utility recovery, tied-score AUROC,
calibration, best-fixed-utility edge cases, strict JSON output, and the
standalone `generic_RoC` command. `count_lines.py` counts all source and test
code and fails if the repository reaches the 5,000-line publication limit.

The study used clinician-authored synthetic vignettes and publicly reported
model outputs; it involved no human subjects and no identifiable patient data.
