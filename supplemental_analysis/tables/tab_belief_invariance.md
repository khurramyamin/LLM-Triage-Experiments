**Stated probabilities across informationally equivalent variants. Within-vignette SD is the standard deviation of the elicited probability across the 16 demographic/context variants of a vignette, averaged over the 78 vignettes; the next column counts vignettes whose 16 variants received exactly the same probability. The difference columns give the mean paired difference in elicited probability between factor levels (within vignette; for objective data, within base scenario). "Decision flips" is the share of within-vignette variant pairs with identical stated probability whose default decisions nevertheless differ, i.e. decision noise near the threshold that no demographic or context factor explains.**

| Configuration | Within-vignette SD | Vignettes with identical p | Black-White | Woman-Man | Anchor-None | Barrier-None | Labs-None | Decision flips at identical p | Vignettes with a flip |
|---|---|---|---|---|---|---|---|---|---|
| GPT-5-mini · minimal | 0.029 | 3/78 | 0.002 | -0.001 | -0.019 | -0.000 | 0.014 | 0.095 | 22/78 |
| GPT-5-mini · medium | 0.029 | 5/78 | 0.005 | -0.005 | -0.016 | -0.004 | 0.027 | 0.063 | 22/78 |
| GPT-5-mini · high | 0.030 | 6/78 | 0.004 | -0.004 | -0.016 | -0.012 | 0.022 | 0.078 | 25/78 |
| GPT-5.4 · none | 0.061 | 4/78 | 0.004 | -0.000 | -0.013 | 0.006 | -0.007 | 0.004 | 5/78 |
| GPT-5.4 · medium | 0.076 | 3/78 | -0.006 | -0.015 | -0.007 | -0.001 | 0.040 | 0.033 | 14/78 |
| GPT-5.4 · high | 0.078 | 3/78 | -0.006 | -0.024 | -0.015 | 0.011 | 0.040 | 0.029 | 13/78 |
| DeepSeek-V4-Flash · none | 0.121 | 1/78 | 0.051 | -0.000 | -0.047 | 0.017 | 0.056 | 0.044 | 27/78 |
| DeepSeek-V4-Flash · high | 0.083 | 2/78 | 0.004 | 0.000 | -0.040 | 0.006 | 0.037 | 0.046 | 21/78 |
| DeepSeek-V4-Flash · max | 0.105 | 11/78 | 0.003 | -0.014 | -0.025 | -0.003 | 0.029 | 0.054 | 23/78 |
| DeepSeek-V4-Pro · none | 0.124 | 0/78 | -0.004 | -0.008 | -0.039 | 0.034 | 0.005 | 0.037 | 19/78 |
| DeepSeek-V4-Pro · high | 0.098 | 6/78 | -0.004 | -0.010 | -0.020 | -0.012 | -0.003 | 0.035 | 19/78 |
| DeepSeek-V4-Pro · max | 0.092 | 2/78 | -0.003 | -0.009 | -0.032 | -0.010 | 0.029 | 0.055 | 22/78 |
| Claude Fable 5 · low | 0.035 | 12/78 | 0.007 | -0.002 | 0.021 | 0.012 | 0.011 | 0.057 | 16/78 |
| Claude Fable 5 · medium | 0.037 | 10/78 | 0.008 | -0.003 | 0.027 | 0.015 | 0.009 | 0.047 | 14/78 |
| Claude Fable 5 · high | 0.038 | 10/78 | 0.007 | -0.003 | 0.025 | 0.012 | 0.012 | 0.065 | 20/78 |
| Claude Sonnet 5 · low | 0.046 | 5/78 | 0.008 | -0.002 | -0.024 | -0.008 | 0.075 | 0.035 | 16/78 |
| Claude Sonnet 5 · medium | 0.057 | 6/78 | 0.004 | -0.006 | -0.023 | -0.005 | 0.081 | 0.038 | 17/78 |
| Claude Sonnet 5 · high | 0.076 | 7/78 | 0.006 | 0.000 | -0.040 | 0.002 | 0.058 | 0.045 | 17/78 |
