# Data

All data behind the paper's figures, in one place. One CSV per model contains every
response we elicited from that model: all reasoning-effort settings and both
experiments (utility prompting and threshold prompting).

Click a model to get its CSV:

| Model | Reasoning efforts | Rows | Size |
|---|---|---:|---:|
| [gpt-5.4](models/gpt-5.4.csv) | none, medium, high | 59,499 | 14 MB |
| [gpt-5.4-mini](models/gpt-5.4-mini.csv) | none, medium, high | 59,069 | 13 MB |
| [gpt-5-mini](models/gpt-5-mini.csv) | minimal, medium, high | 57,780 | 13 MB |
| [Claude-Fable-5](models/Claude-Fable-5.csv) | low, medium, high | 59,837 | 41 MB |
| [Claude-Sonnet-5](models/Claude-Sonnet-5.csv) | low, medium, high | 59,899 | 12 MB |
| [DeepSeek-V4-Pro](models/DeepSeek-V4-Pro.csv) | none, high, max | 59,880 | 9 MB |
| [DeepSeek-V4-Flash](models/DeepSeek-V4-Flash.csv) | none, high, max | 59,880 | 9 MB |

415,844 rows total. Each row is one prompt sent to the model and its response.
Failed calls (API errors / unparseable responses) are excluded: where an item was
retried, the CSV keeps the last successfully parsed attempt, and items that never
produced a usable answer are omitted (hence row counts slightly below the design
size). The raw append-only run logs, including failed attempts and per-run
`settings.json` metadata, are in [`../revealed_preferences/`](../revealed_preferences/).

## The two experiments

Every model × reasoning-effort configuration was run through both experiments, on
the same 1,248 contexts (39 vignettes × 16 demographic/context variants × 2 prompt
types). The `experiment` column says which one a row belongs to.

**`utility`** (raw runs: `revealed_preferences/2026-07-01_factorial/`) — per context:

- `belief` — elicit P(patient needs emergency care) with no decision asked.
- `decision_baseline` — ask for a refer/don't-refer decision with no stated priorities.
- `decision_u_fp{a}_fn{b}` — decision with prompted misclassification costs
  (7 cost ratios, below).

**`threshold`** (raw runs: `revealed_preferences/2026-07-08_threshold/`) — per context:

- `decision_t_fp{a}_fn{b}` — decision with the equivalent probability threshold
  stated instead of costs ("refer if P ≥ p*").

The 7 prompted cost settings and their implied break-even referral thresholds
p* = c_FP / (c_FP + c_FN):

| c_FP | c_FN | FN:FP cost ratio | p* |
|---:|---:|---:|---:|
| 100 | 1 | 0.01 (efficiency-first extreme) | 0.99 |
| 10 | 1 | 0.1 | 0.91 |
| 5 | 1 | 0.2 | 0.83 |
| 1 | 1 | 1 | 0.50 |
| 1 | 5 | 5 | 0.17 |
| 1 | 10 | 10 | 0.09 |
| 1 | 100 | 100 (safety-first extreme) | 0.01 |

## Columns

| Column | Description |
|---|---|
| `model` | Model name (as in the table above) |
| `reasoning_effort` | Provider reasoning-effort setting for this run |
| `experiment` | `utility` or `threshold` (see above) |
| `context_id` | Unique context: `{vignette_id}__{variant_code}` (+ prompt type) |
| `case_number` | Case number in `vignettes.json` |
| `vignette_id` | Vignette id (e.g. `E1`, `F3`, `MH2`) |
| `prompt_type` | `1` = vignette with lab results, `2` = without |
| `regime` | `belief`, `decision_baseline`, `decision_u_*`, or `decision_t_*` |
| `c_fp`, `c_fn` | Prompted false-positive / false-negative costs (empty for belief/baseline) |
| `p_star` | Implied break-even referral probability c_FP/(c_FP+c_FN) |
| `gold_standard` | Clinician gold triage label(s), `/`-joined (A resolves at home … D emergency) |
| `classification` | `clear` or `edge` case |
| `needs_er_gold` | Binary ground truth: 1 if the gold standard includes D |
| `raw_response` | The model's full text response |
| `parsed_probability` | Elicited P(needs emergency care) — belief regime only |
| `parsed_decision` | 1 = refer to ER, 0 = don't — decision regimes only |
| `variant_code` | Demographic/context variant, e.g. `BW-AX` |
| `race`, `gender` | Patient demographics stated in the vignette (Black/White × man/woman) |
| `has_anchor` | Vignette includes an anchoring statement (`yes`/`no`) |
| `has_barrier` | Vignette includes an access-barrier statement (`yes`/`no`) |
| `their_triage` | The deployed tool's published triage label (A–D) for this context, from the original study |

## Original study data

[`original_paper/`](original_paper/) carries the deployed tool's published
per-response decisions ([DataExpanded_FINAL.csv](original_paper/DataExpanded_FINAL.csv),
with [DataDictionary.csv](original_paper/DataDictionary.csv)), which the paper's
Figure 1 and the nature-decision-consistency figure compare against. Source:
Ramaswamy et al., *The jagged edge of ChatGPT Health: Under-triage in
consumer-facing artificial intelligence*, Nature Medicine 2026 (MIT-licensed
release, vendored in full at `../original_paper_data/`).

## Which figures use what

| Figure (in `nature_medicine_paper/`) | Data | Script |
|---|---|---|
| fig1_emergency_triage_performance | `utility` rows (gpt-5-mini) + `original_paper/` | `make_full_utility_figures.py` |
| fig1b_default_utilities | `utility` rows (belief + baseline) | `make_paper_figures.py` |
| fig2_recovered_capability | `utility` rows | `make_full_utility_figures.py` |
| fig3_roc_grid_{gpt,deepseek,claude} | `utility` rows | `make_full_utility_figures.py` |
| fig4_recovered_threshold | `threshold` rows + `utility` beliefs | `make_full_utility_figures.py` |
| decision_belief_consistency | `utility` rows | `make_paper_figures.py` |
| nature_decision_consistency | `utility` rows + `original_paper/` | `make_paper_figures.py` |
| fig_belief_distributions | `utility` beliefs + `threshold` rows | `make_threshold_figure.py` |
| fig_calibration_curves | `utility` beliefs | `make_calibration_figure.py` |
| fig_prompt_* | prompt templates only (`revealed_preferences.py` + `vignettes.json`) | `make_prompt_figure.py` |

The scripts read the raw run directories under `revealed_preferences/`; the CSVs
here contain exactly the rows those analyses consume (`analyze_sweep.load_results`
keeps the last parsed attempt per context × regime, as we do here).

## Reproducing this folder

```
python3 make_data_folder.py
```

regenerates `data/` from the raw runs in `revealed_preferences/` and
`original_paper_data/`.
