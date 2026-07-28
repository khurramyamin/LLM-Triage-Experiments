# Revealed-preference sweep — analysis

## 1. Default (baseline) implied preference

With **no** cost function in the prompt, the model still makes YES/NO admit decisions. Fitting those against the independently elicited beliefs recovers the model's *default* implied cost ratio c_FN/c_FP. A ratio > 1 means the model behaves as if **missing an emergency is worse** than an unnecessary ED visit.

| config | baseline c_FN/c_FP | baseline admit % |
|---|---|---|
| gpt-5-mini_re-high | 6.46 | 37% |
| gpt-5-mini_re-medium | 4.92 | 34% |
| gpt-5-mini_re-minimal | 6.90 | 44% |
| gpt-5.4-mini_re-high | 7.49 | 59% |
| gpt-5.4-mini_re-medium | 7.13 | 60% |
| gpt-5.4-mini_re-none | 4.14 | 64% |
| gpt-5.4_re-high | 5.00 | 63% |
| gpt-5.4_re-medium | 4.05 | 60% |
| gpt-5.4_re-none | 2.75 | 65% |

## 2. Does the model follow the prompted cost function?

`rho_admit` = Spearman rank corr. between the prompted FN-aversion (ordered 10:1 -> 1:10) and the realised admit rate. `rho_recovered` = same, but on the *recovered* log cost ratio. ~1.0 = the model steers exactly as instructed; ~0 = it ignores the cost function.

| config | rho_admit | rho_recovered | mean self-consistency |
|---|---|---|---|
| gpt-5-mini_re-high | 0.90 | 0.90 | 92% |
| gpt-5-mini_re-medium | 1.00 | 1.00 | 92% |
| gpt-5-mini_re-minimal | 1.00 | 1.00 | 68% |
| gpt-5.4-mini_re-high | 1.00 | 1.00 | 92% |
| gpt-5.4-mini_re-medium | 1.00 | 1.00 | 87% |
| gpt-5.4-mini_re-none | 1.00 | 0.90 | 72% |
| gpt-5.4_re-high | 1.00 | 1.00 | 95% |
| gpt-5.4_re-medium | 1.00 | 1.00 | 95% |
| gpt-5.4_re-none | 1.00 | 1.00 | 92% |

## 3. Self-consistency by regime (actual decision vs. own-belief threshold rule)

For each prompted cost function the expected-cost-minimising rule is `admit <=> belief >= p*`, with p* = c_FP/(c_FP+c_FN). The cells show the fraction of cases where the model's actual decision matches that rule applied to its **own** elicited belief.

| config | u_fp10_fn1 (p*=0.91) | u_fp5_fn1 (p*=0.83) | u_fp1_fn1 (p*=0.50) | u_fp1_fn5 (p*=0.17) | u_fp1_fn10 (p*=0.09) |
|---|---|---|---|---|---|
| gpt-5-mini_re-high | 99% | 99% | 91% | 90% | 79% |
| gpt-5-mini_re-medium | 97% | 100% | 94% | 83% | 88% |
| gpt-5-mini_re-minimal | 74% | 67% | 59% | 62% | 79% |
| gpt-5.4-mini_re-high | 92% | 92% | 90% | 94% | 92% |
| gpt-5.4-mini_re-medium | 88% | 85% | 77% | 91% | 95% |
| gpt-5.4-mini_re-none | 46% | 54% | 73% | 90% | 95% |
| gpt-5.4_re-high | 99% | 97% | 88% | 99% | 94% |
| gpt-5.4_re-medium | 100% | 96% | 88% | 97% | 95% |
| gpt-5.4_re-none | 96% | 78% | 96% | 96% | 95% |

## 4. Recovered cost ratio by regime

Target ratios: u_fp10_fn1=0.1, u_fp5_fn1=0.2, u_fp1_fn1=1, u_fp1_fn5=5, u_fp1_fn10=10

| config | u_fp10_fn1 | u_fp5_fn1 | u_fp1_fn1 | u_fp1_fn5 | u_fp1_fn10 |
|---|---|---|---|---|---|
| gpt-5-mini_re-high | 0.48 | 0.11 | 2.31 | 6.07 | 7.29 |
| gpt-5-mini_re-medium | 0.04 | 0.22 | 3.11 | 6.15 | 10.57 |
| gpt-5-mini_re-minimal | 4.82 | 6.87 | 10.43 | 17.69 | 22.22 |
| gpt-5.4-mini_re-high | 0.20 | 0.32 | 2.93 | 9.37 | 12.13 |
| gpt-5.4-mini_re-medium | 0.29 | 0.59 | 5.86 | 5.98 | 11.34 |
| gpt-5.4-mini_re-none | 5.19 | 5.04 | 6.14 | 6.29 | 9.16 |
| gpt-5.4_re-high | 0.06 | 0.14 | 0.63 | 5.91 | 7.10 |
| gpt-5.4_re-medium | 0.08 | 0.13 | 0.85 | 5.34 | 9.79 |
| gpt-5.4_re-none | 0.03 | 0.03 | 1.92 | 6.40 | 7.32 |

## 5. Decisions vs (elicited belief + ESTIMATED utility)

Here the threshold comes from the utility we **fit** to each regime (the *recovered* ratio), not the prompted one: `admit <=> belief >= p*_est`, p*_est = c_FP/(c_FP+c_FN) from the fit. The cells show how often that rule reproduces the model's actual YES/NO — i.e. how well 'belief + one number' explains the decisions.

| config | baseline | u_fp10_fn1 | u_fp5_fn1 | u_fp1_fn1 | u_fp1_fn5 | u_fp1_fn10 | mean(util) |
|---|---|---|---|---|---|---|---|
| gpt-5-mini_re-high | 85% | 100% | 100% | 95% | 90% | 82% | 93% |
| gpt-5-mini_re-medium | 84% | 100% | 100% | 94% | 87% | 88% | 94% |
| gpt-5-mini_re-minimal | 82% | 81% | 82% | 81% | 77% | 92% | 83% |
| gpt-5.4-mini_re-high | 92% | 90% | 87% | 88% | 96% | 95% | 91% |
| gpt-5.4-mini_re-medium | 90% | 88% | 88% | 91% | 88% | 95% | 90% |
| gpt-5.4-mini_re-none | 87% | 92% | 90% | 91% | 94% | 95% | 92% |
| gpt-5.4_re-high | 95% | 100% | 99% | 90% | 99% | 95% | 96% |
| gpt-5.4_re-medium | 91% | 100% | 99% | 88% | 97% | 95% | 96% |
| gpt-5.4_re-none | 99% | 100% | 100% | 97% | 97% | 96% | 98% |

## 6. Oracle: elicited beliefs vs the true need-ER label

The elicited probabilities are scored against the gold `needs_er_gold` label. The ROC **AUC** is how well the beliefs alone rank who truly needs the ER. *Any* utility function is just a threshold on the belief — i.e. a point on this ROC. `best fixed utility` is the single threshold that best matches the gold labels (the most desired behaviour reachable by fixing one utility value); its implied c_FN/c_FP is shown.

| config | belief AUC | best fixed-utility acc | best threshold | implied c_FN/c_FP | baseline-decision acc |
|---|---|---|---|---|---|
| gpt-5-mini_re-high | 0.89 | 79% | 0.12 | 7.33 | 81% |
| gpt-5-mini_re-medium | 0.89 | 78% | 0.08 | 11.50 | 82% |
| gpt-5-mini_re-minimal | 0.86 | 76% | 0.05 | 19.00 | 91% |
| gpt-5.4-mini_re-high | 0.90 | 86% | 0.22 | 3.55 | 82% |
| gpt-5.4-mini_re-medium | 0.90 | 82% | 0.18 | 4.56 | 81% |
| gpt-5.4-mini_re-none | 0.87 | 79% | 0.20 | 4.00 | 82% |
| gpt-5.4_re-high | 0.91 | 85% | 0.32 | 2.12 | 78% |
| gpt-5.4_re-medium | 0.89 | 81% | 0.22 | 3.55 | 81% |
| gpt-5.4_re-none | 0.90 | 81% | 0.69 | 0.45 | 78% |

## 7. Figures

- `admit_rate.png` — admit rate vs prompted cost function (steering curves).
- `recovered_vs_target.png` — recovered vs prompted ratio, log-log, with identity line.
- `decision_belief_consistency.png` — % of decisions matching belief+estimated utility (stars = baseline).
- `roc_oracle.png` — ROC of elicited beliefs vs true need-ER label, with decision regimes as operating points.
