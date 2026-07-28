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
├── revealed_preferences.py        # core engine: prompts, LLM calls, cost-function fitting
├── run_analysis.py                # base infrastructure (vignette loading, call_llm) reused by the engine
├── run_factorial_sweep.py         # elicit belief + baseline + utility-prompted decisions
├── run_threshold_sweep.py         # elicit probability-threshold-prompted decisions
├── run_extra_utilities.py         # append the extreme cost ratios (.01 and 100) to the sweeps
├── run_deepseek_full.py           # full sweep for the DeepSeek V4 models (chat.completions API)
├── analyze_sweep.py               # per-config analysis: recovered ratios, ROC/AUROC, consistency
├── original_paper_comparison.py   # compare against the deployed tool's published operating point
│
├── make_paper_figures.py          # Figures 1–3 (deployed ROC, recovered-vs-prompted, ROC grid)
├── make_full_utility_figures.py   # full seven-utility versions of the utility/threshold figures
├── make_threshold_figure.py       # Figure 4 companions (default thresholds, belief distributions)
├── make_decomposition_figure.py   # exploratory loss decomposition (not in the paper)
│
├── make_data_folder.py            # collect all figure data into data/ (one CSV per model)
│
├── vignettes.json                 # 39 clinician-authored clinical vignettes
├── anchoring_statements.json      # per-case anchoring-bias statements (factorial design)
├── access_barriers.json           # access-barrier statements (factorial design)
│
├── generic_RoC/                   # portable ROC + utility tool for any labeled dataset
│
├── data/                          # ⇒ all data behind the figures, one CSV per model
├── original_paper_data/           # extracted data from the original ChatGPT Health study
├── revealed_preferences/          # raw sweep outputs (incl. failed attempts and run metadata)
└── nature_medicine_paper/         # LaTeX manuscript, figures, and references
```

## Data

**[`data/`](data/README.md)** has everything the paper's figures consume, in one
place: one CSV per model (click the model name in
[`data/README.md`](data/README.md)) containing every elicited belief and decision
for that model across all reasoning efforts and both experiments, plus the
original study's published decisions that Figure 1 compares against. Failed API
calls are excluded there; the raw append-only run logs remain in
`revealed_preferences/`.

## Generic ROC / utility tool

`generic_RoC/` packages the belief-vs-utility analysis as a **standalone,
dataset-agnostic** command-line tool. Given any CSV of elicited **beliefs**
(probabilities), ground-truth **labels**, and optional observed **decisions**,
it builds the belief ROC curve, backs out the FN/FP cost ratio the data behaves
*as if* it holds, lets you name a target cost ratio (FN:FP) to evaluate by, and
finds the **best fixed utility ratio** — the operating point on the ROC that
minimises the target-weighted cost, expressed as the FN/FP ratio you would
prompt an LLM with on unseen data. The 1:1 case reproduces the "Best Fixed
Utility" markers in the family-specific
`nature_medicine_paper/figures/fig3_roc_grid_*.png` panels. It depends only on
`numpy`, `scipy`, and `matplotlib` and runs fully offline; see
[`generic_RoC/README.md`](generic_RoC/README.md) for the input schema and usage.

## Method in brief

For each clinical vignette (expanded over a race × gender × anchoring ×
access-barrier factorial design) we issue, in separate queries:

1. a **belief** prompt — the model's probability that the patient needs emergency care;
2. a **baseline decision** prompt — refer or not, with no stated priorities;
3. **utility-prompted decisions** — refer-or-not under an explicit cost ratio
   (FN\:FP ∈ {.01, .1, .2, 1, 5, 10, 100}); and
4. **threshold-prompted decisions** — the concordant probability threshold
   `p* = c_FP / (c_FP + c_FN)` implied by each of those cost ratios.

From matched (belief, decision) pairs we fit a discrete-choice (logistic) cost
function and read off the recovered FN/FP ratio — the priority the model behaved
*as if* it held. Beliefs are scored against the gold labels with a standard ROC
analysis, so any recovered ratio maps to an operating point on the belief ROC.

## Reproducing the figures

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# Regenerate the paper figures from the existing sweep outputs
python make_paper_figures.py        # also emits the full-utility versions
python make_threshold_figure.py     # default-threshold + belief-distribution panels
```

Figures are written to `nature_medicine_paper/figures/`. The canonical figures
show the full set of seven prompted cost ratios (including the .01 and 100
extremes); `make_paper_figures.py` finishes by calling `make_full_utility_figures`
so those versions always win.

## Re-running the elicitation sweeps

The sweeps call models through an OpenAI-compatible endpoint (Azure AI Foundry).
Set an API key via `--api-key` or the `AZURE_KEY` environment variable.

```bash
# GPT family: belief + baseline + utility-prompted decisions (factorial design)
python run_factorial_sweep.py --configs gpt-5-mini:medium --parallel 100 \
    --output-dir revealed_preferences/<run>_factorial

# Concordant probability-threshold decisions
python run_threshold_sweep.py --configs gpt-5-mini:medium --parallel 100 \
    --output-dir revealed_preferences/<run>_threshold

# DeepSeek V4 (uses chat.completions reasoning_effort: none / high / max)
python run_deepseek_full.py --model DeepSeek-V4-Pro --parallel 100 --run-tag pro
```

All sweep runners are **resumable**: re-running a config re-fires only the
(context, regime) pairs that don't yet have a parsed result, so an interrupted
run can be continued safely.

## Data

- `vignettes.json`, `anchoring_statements.json`, `access_barriers.json` — the
  clinician-authored synthetic vignettes and factorial-design modifiers.
- `original_paper_data/` — data extracted from the original ChatGPT Health
  emergency-triage evaluation, used for the deployed-tool comparison.
- `revealed_preferences/` — the elicited beliefs/decisions (`results.csv`) and
  recovered cost-function fits (`fit.json`) for each model × configuration.

The study used clinician-authored synthetic vignettes and publicly reported model
outputs; it involved no human subjects and no identifiable patient data.
