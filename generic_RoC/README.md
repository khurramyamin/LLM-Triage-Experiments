# generic_RoC

A small, **portable, self-contained** tool that runs the belief-vs-utility
analysis from this repository on *any* labeled dataset — not just the triage
vignettes. Give it a CSV of elicited **beliefs** (probabilities), ground-truth
**labels**, and (optionally) observed **decisions**, and it will:

1. **Build the belief ROC curve** and report the AUROC — how well the beliefs
   *rank* the positive class (a capability question).
2. **Back out the FN/FP cost ratio implied by the dataset** — the false-negative
   vs false-positive priority the data behaves *as if* it holds — using the same
   revealed-preference discrete-choice (logit) fit as
   [`revealed_preferences.py`](../revealed_preferences.py).
3. Let you **specify a target cost ratio** (FN:FP) to evaluate by.
4. **Find the best fixed utility ratio** for that target: the single operating
   point on the belief ROC that minimises the target-weighted expected cost,
   expressed as the FN/FP cost ratio you would **prompt an LLM with** to
   reproduce that operating point on unseen data.

This generalises the "Best Fixed Utility" markers in the family-specific
[`nature_medicine_paper/figures/fig3_roc_grid_gpt.png`](../nature_medicine_paper/figures/fig3_roc_grid_gpt.png),
DeepSeek, and Claude ROC grids: the gold square is exactly the `--cost-ratio 1`
(1:1) case of this tool — the belief threshold that maximises accuracy.

## Why beliefs and utilities are separate

The target cost ratio you *care about* and the utility ratio you should
*prompt with* are **not** generally the same number. If a model's beliefs are
mis-calibrated, the threshold that best serves your objective sits at a
different point on the ROC than a naive reading of the cost ratio would suggest.
This tool measures that gap for you: it reports both the target cost ratio and
the best fixed **utility** ratio that actually achieves it on your data, so you
can prompt the model with the ratio that works rather than the one you assumed.

## Input format

A CSV with one row per case. Column names are configurable via CLI flags; the
defaults are:

| column     | required | meaning                                                    |
|------------|----------|------------------------------------------------------------|
| `belief`   | yes      | elicited probability of the positive class, a float in `[0, 1]` |
| `label`    | yes      | ground-truth binary outcome (`1` = positive / needs action) |
| `decision` | no       | observed binary action taken (`1` = act / refer)           |

Binary columns accept `0/1`, `yes/no`, `true/false`, `positive/negative`, etc.
Rows with a missing belief or label are skipped.

- If a `decision` column is provided, the implied FN/FP ratio is recovered from
  the **observed decisions** relative to the beliefs.
- If it is omitted, the implied FN/FP ratio is recovered from the
  **ground-truth labels** instead.

See [`example_dataset.csv`](example_dataset.csv) for a worked example.

## Cost-ratio convention

A cost ratio is always **FN/FP** (false-negative cost ÷ false-positive cost),
matching the rest of this repository. On the command line you can pass either a
plain number or an `FN:FP` string:

| `--cost-ratio` | meaning                | leaning              |
|----------------|------------------------|----------------------|
| `1` or `1:1`   | FN/FP = 1              | accuracy (balanced)  |
| `10:1`         | FN/FP = 10             | safety (miss-averse) |
| `1:5`          | FN/FP = 0.2            | resource (over-referral-averse) |

## Usage

```bash
# From the repo root, with the project venv active.
cd generic_RoC

# 1) (Re)generate the deterministic demo dataset (optional; already committed).
python make_example_dataset.py

# 2) Run the analysis for a 1:1 cost objective.
python generic_roc.py --input example_dataset.csv --decision-col decision --cost-ratio 1

# 3) Evaluate a safety-leaning objective instead.
python generic_roc.py --input example_dataset.csv --decision-col decision --cost-ratio 10:1

# 4) Your own data with custom column names.
python generic_roc.py --input mydata.csv \
    --belief-col p_positive --label-col y_true --decision-col action \
    --cost-ratio 5:1 --output-dir out --title "My model — belief ROC"
```

### Outputs

For an `--input` file (or `--output-dir`), the tool writes:

- `roc.png` — the belief ROC with the chance diagonal, the observed decisions'
  operating point (if provided), and the gold **Best Fixed Utility** square for
  the target cost ratio.
- `summary.json` — machine-readable results: AUROC, the implied FN/FP fit, the
  target cost ratio, and the best fixed utility (threshold, operating point, and
  the FN/FP ratio to prompt with).

It also prints a human-readable report to stdout. On the committed demo data
(`--cost-ratio 1`) the beliefs are intentionally inflated, so the best fixed
utility to prompt with (`FN/FP ≈ 1:3.31`) differs from the 1:1 objective — the
gap the beliefs-vs-utilities decomposition is designed to expose.

## Files

| file                     | purpose                                                   |
|--------------------------|-----------------------------------------------------------|
| `generic_roc.py`         | core library + CLI (importable functions + `main`)        |
| `make_example_dataset.py`| deterministic synthetic demo-data generator               |
| `example_dataset.csv`    | committed demo dataset (`belief`, `decision`, `label`)     |
| `roc.png`, `summary.json`| example outputs from running the tool on the demo data    |

## Dependencies

Only the repository's existing dependencies: `numpy`, `scipy`, and `matplotlib`
(see [`../requirements.txt`](../requirements.txt)). No sklearn, no network, no
API keys — the whole tool runs offline.
