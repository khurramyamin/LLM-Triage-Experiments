**Default-regime misses among the 64 definitive-emergency variants of the primary endpoint, by configuration and vignette. E9/F9 = acute asthma exacerbation with/without objective data; E13/F13 = diabetic ketoacidosis with/without objective data (16 demographic/context variants each). mean p is the mean elicited probability of needing emergency care for the 16 variants. "Recovered" counts missed variants that are referred under the 5:1 and 10:1 safety-weighted prompts; "Over-triage" is the referral rate among the 512 non-emergency variants under the default, 5:1 and 10:1 regimes. Totals across the 18 configurations: 104 misses (78 from GPT-5-mini), 78 recovered at 5:1 (55 GPT-5-mini), 95 at 10:1 (69 GPT-5-mini).**

| Configuration | Missed/64 | E9 | F9 | E13 | F13 | mean p E9 | mean p F9 | mean p E13 | mean p F13 | Recovered 5:1 | Recovered 10:1 | Over-triage default | Over-triage 5:1 | Over-triage 10:1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-5-mini · minimal | 8 | 0 | 7 | 0 | 1 | 0.28 | 0.16 | 0.85 | 0.17 | 6 | 8 | 6% | 24% | 29% |
| GPT-5-mini · medium | 31 | 11 | 11 | 0 | 9 | 0.27 | 0.16 | 0.90 | 0.18 | 21 | 26 | 7% | 4% | 11% |
| GPT-5-mini · high | 39 | 13 | 13 | 0 | 13 | 0.17 | 0.12 | 0.90 | 0.16 | 28 | 35 | 8% | 4% | 9% |
| GPT-5.4 · none | 0 | 0 | 0 | 0 | 0 | 0.84 | 0.75 | 0.96 | 0.87 | 0 | 0 | 32% | 41% | 41% |
| GPT-5.4 · medium | 0 | 0 | 0 | 0 | 0 | 0.77 | 0.44 | 0.95 | 0.61 | 0 | 0 | 28% | 23% | 27% |
| GPT-5.4 · high | 0 | 0 | 0 | 0 | 0 | 0.72 | 0.45 | 0.95 | 0.67 | 0 | 0 | 28% | 20% | 24% |
| DeepSeek-V4-Flash · none | 5 | 0 | 1 | 0 | 4 | 0.86 | 0.68 | 0.97 | 0.73 | 4 | 5 | 21% | 31% | 34% |
| DeepSeek-V4-Flash · high | 5 | 0 | 1 | 0 | 4 | 0.87 | 0.67 | 0.99 | 0.76 | 4 | 5 | 11% | 15% | 19% |
| DeepSeek-V4-Flash · max | 6 | 0 | 1 | 0 | 5 | 0.91 | 0.39 | 1.00 | 0.83 | 5 | 6 | 11% | 5% | 8% |
| DeepSeek-V4-Pro · none | 0 | 0 | 0 | 0 | 0 | 0.83 | 0.77 | 0.96 | 0.84 | 0 | 0 | 26% | 27% | 29% |
| DeepSeek-V4-Pro · high | 4 | 0 | 1 | 0 | 3 | 0.86 | 0.68 | 0.98 | 0.64 | 4 | 4 | 12% | 12% | 13% |
| DeepSeek-V4-Pro · max | 5 | 3 | 1 | 0 | 1 | 0.72 | 0.43 | 0.99 | 0.67 | 5 | 5 | 14% | 9% | 11% |
| Claude Fable 5 · low | 0 | 0 | 0 | 0 | 0 | 0.65 | 0.36 | 0.88 | 0.61 | 0 | 0 | 14% | 27% | 33% |
| Claude Fable 5 · medium | 0 | 0 | 0 | 0 | 0 | 0.67 | 0.37 | 0.86 | 0.60 | 0 | 0 | 16% | 27% | 32% |
| Claude Fable 5 · high | 0 | 0 | 0 | 0 | 0 | 0.71 | 0.40 | 0.86 | 0.60 | 0 | 0 | 18% | 28% | 31% |
| Claude Sonnet 5 · low | 0 | 0 | 0 | 0 | 0 | 0.85 | 0.55 | 0.92 | 0.40 | 0 | 0 | 23% | 29% | 30% |
| Claude Sonnet 5 · medium | 1 | 0 | 0 | 0 | 1 | 0.85 | 0.54 | 0.92 | 0.46 | 1 | 1 | 24% | 29% | 30% |
| Claude Sonnet 5 · high | 0 | 0 | 0 | 0 | 0 | 0.84 | 0.46 | 0.93 | 0.47 | 0 | 0 | 24% | 28% | 29% |
| Deployed tool (Ramaswamy et al.) | 33 | 13 | 15 | 5 | 0 | – | – | – | – | – | – | 10% | – | – |
