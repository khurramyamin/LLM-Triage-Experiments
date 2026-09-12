**Invariance of the default (no stated priority) recovered cost ratio across the factorial design, all 1,248 case variants. "Overall" is the FN:FP ratio recovered from all variants. The next five columns give the ratio of recovered ratios between the two levels of each factor (e.g. Black/White = 1 means identical recovered priorities), with 95% intervals from a cluster bootstrap over the 39 base scenarios (500 draws; each draw keeps all 32 presentations of a selected scenario, so the interval accounts for the correlation among variants). "16-cell range" is the range of ratios recovered separately within each of the 16 factorial cells (n = 78 each); "Cells excl." counts cells whose bootstrap interval excludes the configuration's overall ratio (expected about 1 of 16 by chance).**

| Configuration | Overall ratio [95% CI] | Black/White | Woman/Man | Anchor/None | Barrier/None | Labs/None | 16-cell range | Cells excl. |
|---|---|---|---|---|---|---|---|---|
| GPT-5-mini · minimal | 6.4 [4.9, 8.0] | 1.00 [0.86, 1.12] | 0.98 [0.87, 1.10] | 1.00 [0.80, 1.20] | 0.98 [0.81, 1.18] | 0.98 [0.73, 1.29] | 4.9–7.3 | 0/16 |
| GPT-5-mini · medium | 3.9 [2.2, 5.9] | 1.26 [1.01, 1.80] | 0.83 [0.59, 1.21] | 0.83 [0.47, 1.20] | 1.33 [0.92, 2.06] | 0.65 [0.44, 0.95] | 2.0–6.0 | 0/16 |
| GPT-5-mini · high | 5.6 [3.7, 8.6] | 0.87 [0.73, 0.99] | 0.99 [0.80, 1.12] | 1.09 [0.83, 1.45] | 1.09 [0.85, 1.35] | 1.13 [0.77, 2.23] | 4.6–8.0 | 0/16 |
| GPT-5.4 · none | 2.7 [1.7, 5.2] | 1.14 [0.91, 1.42] | 0.98 [0.64, 1.39] | 1.19 [0.77, 1.96] | 0.92 [0.70, 1.22] | 1.65 [0.87, 3.12] | 1.8–5.5 | 3/16 |
| GPT-5.4 · medium | 3.4 [2.1, 6.0] | 1.12 [0.93, 1.43] | 1.03 [0.80, 1.33] | 1.11 [0.90, 1.41] | 1.07 [0.89, 1.28] | 0.70 [0.41, 0.97] | 2.2–4.5 | 0/16 |
| GPT-5.4 · high | 3.3 [2.1, 6.2] | 1.03 [0.76, 1.34] | 1.00 [0.72, 1.23] | 1.02 [0.77, 1.46] | 0.87 [0.74, 1.10] | 0.76 [0.47, 1.01] | 2.6–5.3 | 0/16 |
| DeepSeek-V4-Flash · none | 1.3 [1.0, 1.9] | 0.86 [0.68, 1.12] | 0.92 [0.75, 1.13] | 0.87 [0.65, 1.13] | 1.30 [0.96, 1.94] | 1.03 [0.80, 1.42] | 0.9–2.9 | 1/16 |
| DeepSeek-V4-Flash · high | 2.1 [1.6, 3.0] | 1.45 [1.01, 2.11] | 0.80 [0.59, 1.04] | 0.92 [0.65, 1.43] | 1.71 [1.11, 2.60] | 1.34 [0.93, 2.21] | 1.2–4.6 | 5/16 |
| DeepSeek-V4-Flash · max | 2.1 [1.4, 3.4] | 1.47 [1.08, 2.06] | 1.31 [0.94, 1.99] | 1.04 [0.68, 1.89] | 1.16 [0.69, 2.18] | 1.01 [0.58, 1.69] | 1.3–5.0 | 3/16 |
| DeepSeek-V4-Pro · none | 1.5 [1.1, 2.1] | 1.32 [0.97, 1.86] | 0.99 [0.82, 1.19] | 1.25 [0.94, 1.63] | 1.04 [0.81, 1.36] | 1.28 [0.86, 2.11] | 0.7–3.1 | 3/16 |
| DeepSeek-V4-Pro · high | 1.9 [1.3, 2.8] | 1.32 [0.88, 2.74] | 1.12 [0.86, 1.61] | 0.87 [0.60, 1.18] | 1.75 [1.29, 2.90] | 0.89 [0.50, 1.38] | 1.1–6.2 | 5/16 |
| DeepSeek-V4-Pro · max | 2.9 [2.0, 5.1] | 0.85 [0.48, 1.26] | 0.55 [0.34, 0.82] | 1.50 [1.05, 2.37] | 1.55 [0.91, 2.46] | 0.64 [0.38, 1.08] | 1.5–8.3 | 6/16 |
| Claude Fable 5 · low | 4.1 [3.6, 4.7] | 1.05 [0.95, 1.18] | 1.07 [1.02, 1.13] | 1.14 [0.93, 1.33] | 1.11 [0.93, 1.36] | 0.96 [0.80, 1.12] | 3.3–5.1 | 0/16 |
| Claude Fable 5 · medium | 4.0 [3.5, 4.9] | 1.05 [0.93, 1.18] | 1.07 [0.98, 1.18] | 1.00 [0.83, 1.12] | 1.06 [0.90, 1.25] | 1.09 [0.96, 1.28] | 3.2–4.7 | 0/16 |
| Claude Fable 5 · high | 4.4 [3.8, 5.5] | 1.02 [0.92, 1.13] | 1.05 [0.94, 1.14] | 1.09 [0.90, 1.29] | 1.18 [0.97, 1.48] | 1.15 [0.89, 1.46] | 3.6–5.5 | 0/16 |
| Claude Sonnet 5 · low | 4.3 [3.5, 5.6] | 0.78 [0.67, 0.88] | 1.11 [0.98, 1.27] | 1.36 [1.03, 1.75] | 1.45 [1.13, 1.82] | 0.92 [0.68, 1.28] | 2.4–7.1 | 5/16 |
| Claude Sonnet 5 · medium | 4.2 [3.2, 5.6] | 1.26 [1.00, 1.72] | 1.02 [0.84, 1.26] | 1.03 [0.83, 1.28] | 1.31 [1.05, 1.65] | 0.83 [0.59, 1.16] | 2.6–6.2 | 3/16 |
| Claude Sonnet 5 · high | 3.6 [2.5, 5.6] | 1.05 [0.86, 1.26] | 0.97 [0.81, 1.22] | 1.51 [0.97, 2.18] | 1.16 [0.85, 1.68] | 1.37 [1.00, 2.08] | 2.1–5.8 | 5/16 |
