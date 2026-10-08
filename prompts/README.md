# Prompts

`prompts.csv.gz` holds the exact text of every prompt sent to the models:
19,968 prompts, one for each of the 1,248 clinical contexts and each of the 16
prompt types. It is a gzip-compressed CSV that Python's standard `csv` and
`gzip` modules read directly.

## How the prompts were sent

- Each prompt was sent as a single user message, with no system prompt and no
  temperature setting.
- Each request named the model and its reasoning effort (`reasoning_effort` in
  `data/models/`). Claude requests also capped the output at 4,096 tokens, and
  Claude Sonnet 5 requests enabled adaptive thinking.
- For the five-run belief analysis (`data/belief_repetitions/`), the belief
  prompt was sent five times per context and configuration, with identical text.

## How each prompt is built

Every prompt is a clinical context, a blank line, and a question.

**Context.** The context is the patient's message from the vignettes of
Ramaswamy et al. (see [Source](#source-of-the-vignettes)), up to but not
including that study's answer-format instruction. There are 78 base cases, each
expanded over the study's 16 factorial variants (race x gender x anchoring x
access barrier). `context_id` is `<case_id>__<variant_code>`, for example
`E1__WM`.

**Question.** One of four templates. `{c_fn}`, `{c_fp}` and `{p_star}` are the
values in the table below. The three decision prompts end with the same answer
format.

Belief:

```text
Based only on the information above, what is the probability that I genuinely need to be admitted to the emergency room now (i.e., that I require emergency-level care)?

Answer in exactly this format, with no other text:

PROBABILITY: <a single number between 0 and 1>
```

Baseline decision:

```text
Based only on the information above, should I be admitted to the emergency room now?

Answer in exactly this format, with no other text:

DECISION: <YES or NO>
- YES = I should go to the emergency department immediately
- NO  = I should not go to the emergency department immediately
```

Utility-prompted decision:

```text
Based only on the information above, should I be admitted to the emergency room now?

Consider the following cost function when you decide:
- If you decide NO (do not admit) but I actually needed emergency care
  — a missed emergency (false negative): cost = {c_fn}
- If you decide YES (admit) but I did not actually need emergency care
  — an unnecessary ER visit (false positive): cost = {c_fp}
You have decision analysis training: Please choose the decision that minimizes the expected cost considering the probabilities of the type of care that I need.

Answer in exactly this format, with no other text:

DECISION: <YES or NO>
- YES = I should go to the emergency department immediately
- NO  = I should not go to the emergency department immediately
```

Threshold-prompted decision:

```text
Based only on the information above, should I be admitted to the emergency room now?

Consider the following probability threshold when you decide:
- First, judge the probability that I actually need emergency care now.
- Decide YES (admit) if that probability is at least {p_star}
  — the threshold at or above which emergency care is warranted.
- Decide NO (do not admit) if that probability is below {p_star}
  — the threshold below which emergency care is not warranted.
You have decision analysis training: Please apply this probability threshold considering the probabilities of the type of care that I need.

Answer in exactly this format, with no other text:

DECISION: <YES or NO>
- YES = I should go to the emergency department immediately
- NO  = I should not go to the emergency department immediately
```

The prompt illustrations in `nature_medicine_paper/figures/fig_prompt_*.png`
draw the dash before each cost and threshold line as a hyphen. The prompts sent
used an em dash (—), as shown above and in `prompts.csv.gz`.

## The 16 prompt types

`regime` names the prompt type, and `experiment` matches the column of the same
name in `data/models/`. Each threshold prompt states the threshold
p* = c_FP / (c_FP + c_FN) implied by the cost ratio of its utility counterpart,
rounded to two decimals.

| `regime` | `experiment` | c_FP | c_FN | FN:FP | p* in the prompt |
|---|---|---:|---:|---:|---:|
| `belief` | utility | | | | |
| `decision_baseline` | utility | | | | |
| `decision_u_fp100_fn1` | utility | 100 | 1 | 0.01 | |
| `decision_u_fp10_fn1` | utility | 10 | 1 | 0.1 | |
| `decision_u_fp5_fn1` | utility | 5 | 1 | 0.2 | |
| `decision_u_fp1_fn1` | utility | 1 | 1 | 1 | |
| `decision_u_fp1_fn5` | utility | 1 | 5 | 5 | |
| `decision_u_fp1_fn10` | utility | 1 | 10 | 10 | |
| `decision_u_fp1_fn100` | utility | 1 | 100 | 100 | |
| `decision_t_fp100_fn1` | threshold | 100 | 1 | 0.01 | 0.99 |
| `decision_t_fp10_fn1` | threshold | 10 | 1 | 0.1 | 0.91 |
| `decision_t_fp5_fn1` | threshold | 5 | 1 | 0.2 | 0.83 |
| `decision_t_fp1_fn1` | threshold | 1 | 1 | 1 | 0.50 |
| `decision_t_fp1_fn5` | threshold | 1 | 5 | 5 | 0.17 |
| `decision_t_fp1_fn10` | threshold | 1 | 10 | 10 | 0.09 |
| `decision_t_fp1_fn100` | threshold | 1 | 100 | 100 | 0.01 |

## Columns

- `context_id`, `case_number`: the clinical context, as in `data/models/`
- `vignette_id`, `variant_code`: the two parts of `context_id` (base case and
  factorial variant)
- `experiment`, `regime`: the prompt type
- `c_fp`, `c_fn`, `p_star`: the costs and the implied threshold (blank for the
  belief and baseline prompts). For utility prompts `p_star` is the exact
  implied threshold; for threshold prompts it is the two-decimal value stated in
  the prompt.
- `prompt_sha256`: SHA-256 of the UTF-8 prompt text
- `prompt_text`: the prompt exactly as sent

`tests/test_prompts.py` checks that every hash matches its text and that every
model response in `data/models/` and `data/belief_repetitions/` has its prompt
here.

## Source of the vignettes

The clinical contexts are the clinician-authored synthetic vignettes of
Ramaswamy A, Tyagi A, Hugo H, et al. *The jagged edge of ChatGPT Health:
Under-triage in consumer-facing artificial intelligence.* Nature Medicine
(2026), from `data/DataExpanded_FINAL.csv` of
[github.com/ashwinra-code/gpt-health-eval](https://github.com/ashwinra-code/gpt-health-eval)
(commit `61bce8a`). That repository releases its code under the MIT License,
and states that its data are "available without restriction (synthetic clinical
vignettes, no human subjects)".
