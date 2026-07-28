# Revealed-preference sweep — analysis

## 1. Default (baseline) implied preference

With **no** cost function in the prompt, the model still makes YES/NO admit decisions. Fitting those against the independently elicited beliefs recovers the model's *default* implied cost ratio c_FN/c_FP. A ratio > 1 means the model behaves as if **missing an emergency is worse** than an unnecessary ED visit.

| config | baseline c_FN/c_FP | baseline admit % |
|---|---|---|
| gpt-5-mini_re-high | 5.63 | 32% |
| gpt-5-mini_re-medium | 3.87 | 30% |
| gpt-5-mini_re-minimal | 6.36 | 38% |
| gpt-5.4-mini_re-high | 4.57 | 56% |
| gpt-5.4-mini_re-medium | 4.95 | 56% |
| gpt-5.4-mini_re-none | 4.59 | 66% |
| gpt-5.4_re-high | 3.27 | 60% |
| gpt-5.4_re-medium | 3.40 | 59% |
| gpt-5.4_re-none | 2.71 | 70% |

## 2. Does the model follow the prompted cost function?

`rho_admit` = Spearman rank corr. between the prompted FN-aversion (ordered 10:1 -> 1:10) and the realised admit rate. `rho_recovered` = same, but on the *recovered* log cost ratio. ~1.0 = the model steers exactly as instructed; ~0 = it ignores the cost function.

| config | rho_admit | rho_recovered | mean self-consistency |
|---|---|---|---|
| gpt-5-mini_re-high | 1.00 | 1.00 | 92% |
| gpt-5-mini_re-medium | 1.00 | 1.00 | 91% |
| gpt-5-mini_re-minimal | 1.00 | 1.00 | 67% |
| gpt-5.4-mini_re-high | 1.00 | 1.00 | 91% |
| gpt-5.4-mini_re-medium | 1.00 | 1.00 | 88% |
| gpt-5.4-mini_re-none | 1.00 | 1.00 | 75% |
| gpt-5.4_re-high | 1.00 | 1.00 | 95% |
| gpt-5.4_re-medium | 1.00 | 1.00 | 95% |
| gpt-5.4_re-none | 1.00 | 1.00 | 91% |

## 3. Self-consistency by regime (actual decision vs. own-belief threshold rule)

For each prompted cost function the expected-cost-minimising rule is `admit <=> belief >= p*`, with p* = c_FP/(c_FP+c_FN). The cells show the fraction of cases where the model's actual decision matches that rule applied to its **own** elicited belief.

| config | u_fp10_fn1 (p*=0.91) | u_fp5_fn1 (p*=0.83) | u_fp1_fn1 (p*=0.50) | u_fp1_fn5 (p*=0.17) | u_fp1_fn10 (p*=0.09) |
|---|---|---|---|---|---|
| gpt-5-mini_re-high | 99% | 100% | 95% | 84% | 82% |
| gpt-5-mini_re-medium | 98% | 100% | 92% | 84% | 84% |
| gpt-5-mini_re-minimal | 70% | 67% | 63% | 56% | 77% |
| gpt-5.4-mini_re-high | 93% | 91% | 84% | 90% | 95% |
| gpt-5.4-mini_re-medium | 88% | 88% | 78% | 90% | 95% |
| gpt-5.4-mini_re-none | 51% | 60% | 75% | 93% | 95% |
| gpt-5.4_re-high | 98% | 95% | 90% | 96% | 97% |
| gpt-5.4_re-medium | 99% | 95% | 90% | 95% | 96% |
| gpt-5.4_re-none | 92% | 78% | 93% | 97% | 98% |

## 4. Recovered cost ratio by regime

Target ratios: u_fp10_fn1=0.1, u_fp5_fn1=0.2, u_fp1_fn1=1, u_fp1_fn5=5, u_fp1_fn10=10

| config | u_fp10_fn1 | u_fp5_fn1 | u_fp1_fn1 | u_fp1_fn5 | u_fp1_fn10 |
|---|---|---|---|---|---|
| gpt-5-mini_re-high | 0.13 | 0.15 | 1.94 | 7.84 | 9.77 |
| gpt-5-mini_re-medium | 0.15 | 0.49 | 2.59 | 6.98 | 9.90 |
| gpt-5-mini_re-minimal | 6.48 | 8.23 | 10.03 | 17.76 | 21.07 |
| gpt-5.4-mini_re-high | 0.21 | 0.37 | 2.96 | 7.34 | 9.37 |
| gpt-5.4-mini_re-medium | 0.39 | 0.60 | 4.06 | 8.74 | 10.88 |
| gpt-5.4-mini_re-none | 4.12 | 4.59 | 6.36 | 8.45 | 8.98 |
| gpt-5.4_re-high | 0.08 | 0.12 | 0.78 | 5.02 | 7.36 |
| gpt-5.4_re-medium | 0.07 | 0.11 | 0.82 | 6.24 | 10.28 |
| gpt-5.4_re-none | 0.03 | 0.04 | 2.75 | 9.61 | 11.71 |

## 5. Decisions vs (elicited belief + ESTIMATED utility)

Here the threshold comes from the utility we **fit** to each regime (the *recovered* ratio), not the prompted one: `admit <=> belief >= p*_est`, p*_est = c_FP/(c_FP+c_FN) from the fit. The cells show how often that rule reproduces the model's actual YES/NO — i.e. how well 'belief + one number' explains the decisions.

| config | baseline | u_fp10_fn1 | u_fp5_fn1 | u_fp1_fn1 | u_fp1_fn5 | u_fp1_fn10 | mean(util) |
|---|---|---|---|---|---|---|---|
| gpt-5-mini_re-high | 84% | 99% | 100% | 95% | 87% | 82% | 92% |
| gpt-5-mini_re-medium | 84% | 98% | 99% | 93% | 85% | 84% | 92% |
| gpt-5-mini_re-minimal | 82% | 79% | 80% | 85% | 80% | 89% | 83% |
| gpt-5.4-mini_re-high | 86% | 93% | 91% | 89% | 93% | 95% | 92% |
| gpt-5.4-mini_re-medium | 88% | 92% | 91% | 89% | 93% | 95% | 92% |
| gpt-5.4-mini_re-none | 92% | 93% | 94% | 95% | 95% | 95% | 94% |
| gpt-5.4_re-high | 86% | 98% | 97% | 89% | 96% | 96% | 95% |
| gpt-5.4_re-medium | 84% | 99% | 98% | 90% | 96% | 95% | 96% |
| gpt-5.4_re-none | 95% | 98% | 96% | 96% | 98% | 99% | 97% |

## 6. Oracle: elicited beliefs vs the true need-ER label

The elicited probabilities are scored against the gold `needs_er_gold` label. The ROC **AUC** is how well the beliefs alone rank who truly needs the ER. *Any* utility function is just a threshold on the belief — i.e. a point on this ROC. `best fixed utility` is the single threshold that best matches the gold labels (the most desired behaviour reachable by fixing one utility value); its implied c_FN/c_FP is shown.

| config | belief AUC | best fixed-utility acc | best threshold | implied c_FN/c_FP | baseline-decision acc |
|---|---|---|---|---|---|
| gpt-5-mini_re-high | 0.88 | 79% | 0.05 | 19.00 | 74% |
| gpt-5-mini_re-medium | 0.88 | 80% | 0.04 | 24.00 | 73% |
| gpt-5-mini_re-minimal | 0.86 | 76% | 0.05 | 19.00 | 76% |
| gpt-5.4-mini_re-high | 0.91 | 81% | 0.15 | 5.67 | 77% |
| gpt-5.4-mini_re-medium | 0.90 | 81% | 0.16 | 5.25 | 78% |
| gpt-5.4-mini_re-none | 0.86 | 78% | 0.20 | 4.00 | 79% |
| gpt-5.4_re-high | 0.90 | 80% | 0.28 | 2.57 | 76% |
| gpt-5.4_re-medium | 0.90 | 81% | 0.20 | 4.00 | 77% |
| gpt-5.4_re-none | 0.87 | 80% | 0.26 | 2.85 | 77% |

## 7. Figures

- `admit_rate.png` — admit rate vs prompted cost function (steering curves).
- `recovered_vs_target.png` — recovered vs prompted ratio, log-log, with identity line.
- `decision_belief_consistency.png` — % of decisions matching belief+estimated utility (stars = baseline).
- `roc_oracle.png` — ROC of elicited beliefs vs true need-ER label, with decision regimes as operating points.
