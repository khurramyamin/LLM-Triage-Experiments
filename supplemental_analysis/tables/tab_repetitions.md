**Stability under repeated sampling. Each configuration's belief prompt was issued five independent times for every one of the 1,248 case variants. Columns: mean across variants of the standard deviation of the five stated probabilities; mean absolute difference between pairs of runs; range of the default recovered FN:FP ratio when the decisions are re-analysed against each run's probabilities; the ratio recovered against the five-run mean probability; and the range of AUROC across runs for the expanded (1,248-case) and primary (576-case) endpoints.**

| Configuration | Mean within-context SD | Mean |Δ| between runs | Default ratio across 5 belief runs | Ratio, mean belief | AUROC (1,248) across runs | AUROC (576) across runs |
|---|---|---|---|---|---|---|
| GPT-5-mini · minimal | 0.019 | 0.022 | 6.1–6.5 | 6.6 | 0.843–0.853 | 0.976–0.980 |
| GPT-5-mini · medium | 0.022 | 0.028 | 3.4–4.5 | 4.3 | 0.882–0.887 | 0.993–0.994 |
| GPT-5-mini · high | 0.021 | 0.026 | 5.2–5.6 | 5.7 | 0.881–0.890 | 0.986–0.993 |
| GPT-5.4 · none | 0.044 | 0.056 | 2.5–2.9 | 2.4 | 0.870–0.877 | 0.933–0.946 |
| GPT-5.4 · medium | 0.051 | 0.066 | 3.1–3.4 | 3.1 | 0.899–0.904 | 0.962–0.970 |
| GPT-5.4 · high | 0.053 | 0.068 | 3.3–3.4 | 3.1 | 0.895–0.902 | 0.951–0.953 |
| DeepSeek-V4-Flash · none | 0.082 | 0.104 | 1.2–1.3 | 1.2 | 0.869–0.882 | 0.921–0.940 |
| DeepSeek-V4-Flash · high | 0.061 | 0.075 | 2.0–2.2 | 2.2 | 0.927–0.937 | 0.988–0.993 |
| DeepSeek-V4-Flash · max | 0.079 | 0.097 | 1.8–2.3 | 2.2 | 0.897–0.909 | 0.981–0.989 |
| DeepSeek-V4-Pro · none | 0.088 | 0.111 | 1.4–1.6 | 1.4 | 0.857–0.869 | 0.906–0.930 |
| DeepSeek-V4-Pro · high | 0.075 | 0.091 | 1.8–2.2 | 2.2 | 0.900–0.911 | 0.967–0.979 |
| DeepSeek-V4-Pro · max | 0.073 | 0.088 | 2.6–2.9 | 2.9 | 0.902–0.912 | 0.969–0.983 |
| Claude Fable 5 · low | 0.016 | 0.018 | 4.1–4.1 | 4.1 | 0.939–0.943 | 0.993–0.996 |
| Claude Fable 5 · medium | 0.017 | 0.020 | 4.0–4.1 | 4.0 | 0.939–0.944 | 0.992–0.995 |
| Claude Fable 5 · high | 0.019 | 0.023 | 4.2–4.4 | 4.3 | 0.938–0.941 | 0.992–0.993 |
| Claude Sonnet 5 · low | 0.012 | 0.014 | 4.2–4.4 | 4.3 | 0.921–0.924 | 0.973–0.977 |
| Claude Sonnet 5 · medium | 0.017 | 0.020 | 4.1–4.4 | 4.2 | 0.919–0.922 | 0.968–0.974 |
| Claude Sonnet 5 · high | 0.033 | 0.039 | 3.6–4.1 | 3.7 | 0.907–0.917 | 0.960–0.966 |
