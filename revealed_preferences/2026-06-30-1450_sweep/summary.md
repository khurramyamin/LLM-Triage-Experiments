# Revealed-preference sweep summary

Generated: 2026-07-01T13:36:06.430289

Implied cost ratio **c_FN / c_FP** recovered from a binary logit (beta = 1) fit of each decision regime against the independently elicited beliefs. Higher ratio = the model behaves as if missing an emergency is more costly than an unnecessary ED visit.

## Implied c_FN/c_FP ratio by config

| config | baseline | u_fp10_fn1 | u_fp5_fn1 | u_fp1_fn1 | u_fp1_fn5 | u_fp1_fn10 |
|---|---|---|---|---|---|---|
| gpt-5-mini_re-high | 6.46 | 0.48 | 0.11 | 2.31 | 6.07 | 7.29 |
| gpt-5-mini_re-medium | 4.92 | 0.04 | 0.22 | 3.11 | 6.15 | 10.57 |
| gpt-5-mini_re-minimal | 6.90 | 4.82 | 6.87 | 10.43 | 17.69 | 22.22 |
| gpt-5.4-mini_re-high | 7.49 | 0.20 | 0.32 | 2.93 | 9.37 | 12.13 |
| gpt-5.4-mini_re-medium | 7.13 | 0.29 | 0.59 | 5.86 | 5.98 | 11.34 |
| gpt-5.4-mini_re-none | 4.14 | 5.19 | 5.04 | 6.14 | 6.29 | 9.16 |
| gpt-5.4_re-high | 5.00 | 0.06 | 0.14 | 0.63 | 5.91 | 7.10 |
| gpt-5.4_re-medium | 4.05 | 0.08 | 0.13 | 0.85 | 5.34 | 9.79 |
| gpt-5.4_re-none | 2.75 | 0.03 | 0.03 | 1.92 | 6.40 | 7.32 |

**Target ratios (the prompted cost functions):** u_fp10_fn1=0.1, u_fp5_fn1=0.2, u_fp1_fn1=1, u_fp1_fn5=5, u_fp1_fn10=10

## Admit rate (% YES) by config

| config | baseline | u_fp10_fn1 | u_fp5_fn1 | u_fp1_fn1 | u_fp1_fn5 | u_fp1_fn10 |
|---|---|---|---|---|---|---|
| gpt-5-mini_re-high | 37% | 12% | 10% | 19% | 32% | 40% |
| gpt-5-mini_re-medium | 34% | 9% | 12% | 23% | 38% | 49% |
| gpt-5-mini_re-minimal | 44% | 35% | 44% | 54% | 64% | 68% |
| gpt-5.4-mini_re-high | 59% | 19% | 24% | 50% | 59% | 63% |
| gpt-5.4-mini_re-medium | 60% | 23% | 31% | 58% | 58% | 64% |
| gpt-5.4-mini_re-none | 64% | 65% | 65% | 68% | 68% | 72% |
| gpt-5.4_re-high | 63% | 10% | 13% | 37% | 62% | 64% |
| gpt-5.4_re-medium | 60% | 10% | 13% | 38% | 63% | 68% |
| gpt-5.4_re-none | 65% | 10% | 10% | 63% | 72% | 73% |

## Parse coverage (parsed / total)

| config | belief | decisions (mean) | empty responses |
|---|---|---|---|
| gpt-5-mini_re-high | 78/78 | 467/468 | 0 |
| gpt-5-mini_re-medium | 78/78 | 462/468 | 6 |
| gpt-5-mini_re-minimal | 78/78 | 467/468 | 0 |
| gpt-5.4-mini_re-high | 78/78 | 468/468 | 0 |
| gpt-5.4-mini_re-medium | 78/78 | 466/468 | 0 |
| gpt-5.4-mini_re-none | 78/78 | 468/468 | 0 |
| gpt-5.4_re-high | 78/78 | 468/468 | 0 |
| gpt-5.4_re-medium | 78/78 | 467/468 | 0 |
| gpt-5.4_re-none | 78/78 | 468/468 | 0 |