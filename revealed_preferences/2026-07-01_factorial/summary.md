# Revealed-preference sweep summary

Generated: 2026-07-02T14:45:16.175199

Implied cost ratio **c_FN / c_FP** recovered from a binary logit (beta = 1) fit of each decision regime against the independently elicited beliefs. Higher ratio = the model behaves as if missing an emergency is more costly than an unnecessary ED visit.

## Implied c_FN/c_FP ratio by config

| config | baseline | u_fp10_fn1 | u_fp5_fn1 | u_fp1_fn1 | u_fp1_fn5 | u_fp1_fn10 |
|---|---|---|---|---|---|---|
| gpt-5-mini_re-high | 5.63 | 0.13 | 0.15 | 1.94 | 7.84 | 9.77 |
| gpt-5-mini_re-medium | 3.87 | 0.15 | 0.49 | 2.59 | 6.98 | 9.90 |
| gpt-5-mini_re-minimal | 6.36 | 6.48 | 8.23 | 10.03 | 17.76 | 21.07 |
| gpt-5.4-mini_re-high | 4.57 | 0.21 | 0.37 | 2.96 | 7.34 | 9.37 |
| gpt-5.4-mini_re-medium | 4.95 | 0.39 | 0.60 | 4.06 | 8.74 | 10.88 |
| gpt-5.4-mini_re-none | 4.59 | 4.12 | 4.59 | 6.36 | 8.45 | 8.98 |
| gpt-5.4_re-high | 3.27 | 0.08 | 0.12 | 0.78 | 5.02 | 7.36 |
| gpt-5.4_re-medium | 3.40 | 0.07 | 0.11 | 0.82 | 6.24 | 10.28 |
| gpt-5.4_re-none | 2.71 | 0.03 | 0.04 | 2.75 | 9.61 | 11.71 |

**Target ratios (the prompted cost functions):** u_fp10_fn1=0.1, u_fp5_fn1=0.2, u_fp1_fn1=1, u_fp1_fn5=5, u_fp1_fn10=10

## Admit rate (% YES) by config

| config | baseline | u_fp10_fn1 | u_fp5_fn1 | u_fp1_fn1 | u_fp1_fn5 | u_fp1_fn10 |
|---|---|---|---|---|---|---|
| gpt-5-mini_re-high | 32% | 10% | 11% | 17% | 33% | 40% |
| gpt-5-mini_re-medium | 30% | 9% | 12% | 20% | 35% | 42% |
| gpt-5-mini_re-minimal | 38% | 39% | 45% | 49% | 63% | 66% |
| gpt-5.4-mini_re-high | 56% | 18% | 24% | 48% | 59% | 62% |
| gpt-5.4-mini_re-medium | 56% | 24% | 28% | 53% | 63% | 64% |
| gpt-5.4-mini_re-none | 66% | 64% | 65% | 69% | 71% | 71% |
| gpt-5.4_re-high | 60% | 12% | 14% | 38% | 62% | 65% |
| gpt-5.4_re-medium | 59% | 12% | 13% | 37% | 64% | 67% |
| gpt-5.4_re-none | 70% | 9% | 10% | 70% | 75% | 76% |

## Parse coverage (parsed / total)

| config | belief | decisions (mean) | empty responses |
|---|---|---|---|
| gpt-5-mini_re-high | 1238/1248 | 7477/7488 | 0 |
| gpt-5-mini_re-medium | 1203/1248 | 7189/7488 | 328 |
| gpt-5-mini_re-minimal | 1138/1248 | 6762/7488 | 833 |
| gpt-5.4-mini_re-high | 1246/1248 | 7460/7488 | 19 |
| gpt-5.4-mini_re-medium | 1241/1248 | 7385/7488 | 69 |
| gpt-5.4-mini_re-none | 1211/1248 | 7257/7488 | 268 |
| gpt-5.4_re-high | 1247/1248 | 7480/7488 | 0 |
| gpt-5.4_re-medium | 1247/1248 | 7484/7488 | 0 |
| gpt-5.4_re-none | 1218/1248 | 7297/7488 | 221 |