**Referral rate by clinical state (all 1,248 variants), under the default regime and three prompted cost ratios (FN:FP). Each cell gives the share referred among definitive emergencies (192; gold D) / edge cases (448; gold C/D, urgent but ED attendance not required by the adjudicators) / non-emergencies (608; gold A–C). Scenario-cluster intervals for every regime are in Table.**

| Configuration | default | 0.2 | 5 | 10 |
|---|---|---|---|---|
| GPT-5-mini · minimal | 0.95 / 0.49 / 0.10 | 0.98 / 0.60 / 0.17 | 1.00 / 0.87 / 0.33 | 1.00 / 0.90 / 0.39 |
| GPT-5-mini · medium | 0.83 / 0.40 / 0.06 | 0.75 / 0.01 / 0.00 | 0.94 / 0.48 / 0.06 | 0.97 / 0.58 / 0.13 |
| GPT-5-mini · high | 0.80 / 0.46 / 0.07 | 0.73 / 0.00 / 0.00 | 0.94 / 0.46 / 0.05 | 0.98 / 0.55 / 0.10 |
| GPT-5.4 · none | 1.00 / 0.94 / 0.43 | 0.58 / 0.04 / 0.00 | 1.00 / 0.99 / 0.50 | 1.00 / 1.00 / 0.50 |
| GPT-5.4 · medium | 1.00 / 0.79 / 0.32 | 0.75 / 0.04 / 0.00 | 1.00 / 0.89 / 0.35 | 1.00 / 0.93 / 0.38 |
| GPT-5.4 · high | 1.00 / 0.78 / 0.33 | 0.75 / 0.06 / 0.00 | 1.00 / 0.87 / 0.33 | 1.00 / 0.91 / 0.36 |
| DeepSeek-V4-Flash · none | 0.97 / 0.78 / 0.27 | 0.98 / 0.83 / 0.35 | 0.99 / 0.91 / 0.42 | 1.00 / 0.93 / 0.44 |
| DeepSeek-V4-Flash · high | 0.97 / 0.72 / 0.12 | 0.80 / 0.24 / 0.01 | 0.99 / 0.74 / 0.21 | 1.00 / 0.79 / 0.28 |
| DeepSeek-V4-Flash · max | 0.97 / 0.70 / 0.11 | 0.79 / 0.07 / 0.00 | 0.97 / 0.59 / 0.07 | 0.97 / 0.67 / 0.11 |
| DeepSeek-V4-Pro · none | 1.00 / 0.77 / 0.31 | 0.99 / 0.75 / 0.26 | 1.00 / 0.79 / 0.34 | 1.00 / 0.81 / 0.36 |
| DeepSeek-V4-Pro · high | 0.98 / 0.65 / 0.10 | 0.75 / 0.07 / 0.00 | 1.00 / 0.54 / 0.11 | 0.99 / 0.61 / 0.12 |
| DeepSeek-V4-Pro · max | 0.97 / 0.65 / 0.12 | 0.70 / 0.03 / 0.00 | 0.99 / 0.50 / 0.08 | 1.00 / 0.56 / 0.10 |
| Claude Fable 5 · low | 1.00 / 0.69 / 0.13 | 0.97 / 0.38 / 0.01 | 1.00 / 0.92 / 0.33 | 1.00 / 0.98 / 0.41 |
| Claude Fable 5 · medium | 1.00 / 0.72 / 0.15 | 0.97 / 0.41 / 0.01 | 1.00 / 0.92 / 0.34 | 1.00 / 0.98 / 0.41 |
| Claude Fable 5 · high | 1.00 / 0.74 / 0.18 | 0.98 / 0.44 / 0.03 | 1.00 / 0.92 / 0.35 | 1.00 / 0.98 / 0.40 |
| Claude Sonnet 5 · low | 1.00 / 0.74 / 0.23 | 0.87 / 0.58 / 0.07 | 1.00 / 0.84 / 0.30 | 1.00 / 0.88 / 0.32 |
| Claude Sonnet 5 · medium | 0.99 / 0.76 / 0.24 | 0.87 / 0.46 / 0.05 | 1.00 / 0.86 / 0.32 | 1.00 / 0.87 / 0.34 |
| Claude Sonnet 5 · high | 1.00 / 0.77 / 0.25 | 0.87 / 0.36 / 0.04 | 1.00 / 0.85 / 0.32 | 1.00 / 0.89 / 0.35 |
