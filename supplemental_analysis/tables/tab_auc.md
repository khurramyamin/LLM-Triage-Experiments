**Threshold-free consistency: AUC of the elicited probability for predicting the model's own referral decision, by configuration and prompted cost ratio (FN:FP). Values near 1 indicate that decisions are a monotone function of the stated probability; 0.5 would indicate no relation.**

| Configuration | default | 0.01 | 0.1 | 0.2 | 1 | 5 | 10 | 100 |
|---|---|---|---|---|---|---|---|---|
| GPT-5-mini · minimal | 0.91 | 0.90 | 0.89 | 0.90 | 0.92 | 0.92 | 0.94 | 0.92 |
| GPT-5-mini · medium | 0.85 | 0.97 | 0.99 | 1.00 | 0.94 | 0.91 | 0.92 | 0.94 |
| GPT-5-mini · high | 0.87 | 0.97 | 1.00 | 1.00 | 0.96 | 0.92 | 0.91 | 0.94 |
| GPT-5.4 · none | 0.99 | 0.96 | 0.99 | 0.97 | 0.99 | 1.00 | 1.00 | 0.99 |
| GPT-5.4 · medium | 0.94 | 0.98 | 0.99 | 0.99 | 0.96 | 0.99 | 0.99 | 1.00 |
| GPT-5.4 · high | 0.94 | 0.98 | 0.99 | 0.99 | 0.96 | 0.98 | 0.99 | 1.00 |
| DeepSeek-V4-Flash · none | 0.92 | 0.89 | 0.92 | 0.93 | 0.95 | 0.95 | 0.96 | 0.96 |
| DeepSeek-V4-Flash · high | 0.96 | 0.97 | 0.96 | 0.95 | 0.95 | 0.95 | 0.95 | 0.96 |
| DeepSeek-V4-Flash · max | 0.93 | 0.96 | 0.97 | 0.96 | 0.94 | 0.95 | 0.95 | 0.94 |
| DeepSeek-V4-Pro · none | 0.95 | 0.91 | 0.92 | 0.94 | 0.96 | 0.95 | 0.95 | 0.96 |
| DeepSeek-V4-Pro · high | 0.96 | 0.98 | 0.98 | 0.97 | 0.95 | 0.96 | 0.96 | 0.96 |
| DeepSeek-V4-Pro · max | 0.93 | 0.98 | 0.98 | 0.99 | 0.96 | 0.96 | 0.96 | 0.95 |
| Claude Fable 5 · low | 0.98 | 0.97 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 |
| Claude Fable 5 · medium | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 1.00 |
| Claude Fable 5 · high | 0.97 | 0.98 | 0.98 | 0.99 | 0.98 | 0.99 | 0.99 | 1.00 |
| Claude Sonnet 5 · low | 0.98 | 0.97 | 0.97 | 0.97 | 0.98 | 0.98 | 0.98 | 0.98 |
| Claude Sonnet 5 · medium | 0.98 | 0.96 | 0.95 | 0.95 | 0.98 | 0.98 | 0.98 | 0.98 |
| Claude Sonnet 5 · high | 0.96 | 0.95 | 0.94 | 0.92 | 0.95 | 0.96 | 0.97 | 0.97 |
