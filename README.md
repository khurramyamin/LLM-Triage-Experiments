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
├── data/                          # compact analysis-ready model and comparison data
├── nature_medicine_paper/
│   └── figures/                   # all main and extended-data figures
└── tests/                         # focused statistical and command-line tests
```

## Data

**[`data/`](data/README.md)** has everything the paper's figures consume, in one
place: one compressed CSV per model containing every successful parsed belief
and decision needed for the analysis across all reasoning efforts and both
experiments, plus the original study's published decisions used for the
deployed-tool comparison. Raw responses, failed API attempts, credentials, and
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

## Reproducing the figures

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate

pip install -r requirements.txt
python run_analysis.py
```

The analysis runs entirely offline from the included data. It writes all 22
manuscript figures to `nature_medicine_paper/figures/` and writes the numerical
results, sample coverage, bootstrap settings, and output manifest to
`analysis_summary.json`.

## Analysis conventions

- Primary endpoint: 576 intended cases, excluding `C/D` cases; positives are `D`
- Expanded endpoint: all 1,248 intended cases; positives contain `D`
- Utility ratios: `.01`, `.1`, `.2`, `1`, `5`, `10`, and `100`
- Recovered utilities: discrete-choice logit fit
- Confidence intervals: 500 bootstrap draws with seed `0`
- Calibration: 10-bin expected calibration error
- Missing or unparseable responses: excluded with denominators reported
- ROC operating points: restricted to cases with both a belief and decision

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
