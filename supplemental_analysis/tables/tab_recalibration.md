**Nominal versus effective default priorities on the expanded case set. Nominal ratio and p* are the default FN:FP cost ratio and break-even probability recovered on the model's own probability scale. ECE is the 10-bin expected calibration error against the clinician labels. Calibrated p* is the observed emergency frequency at the nominal threshold, obtained by isotonic regression fitted on four fifths of the base scenarios and evaluated on the held-out fifth (mean and range over five folds). The effective ratio, (1-q^*)/q^*, is the cost ratio a calibrated decision-maker would need in order to place the threshold at the same observed risk.**

| Configuration | Nominal ratio | Nominal p* | ECE | Calibrated p* (cross-fitted) | Effective ratio | AUROC |
|---|---|---|---|---|---|---|
| GPT-5-mini · minimal | 6.4 | 0.14 | 0.327 | 0.69 (0.67–0.75) | 0.4 | 0.849 |
| GPT-5-mini · medium | 3.9 | 0.21 | 0.331 | 0.86 (0.82–0.95) | 0.2 | 0.885 |
| GPT-5-mini · high | 5.6 | 0.15 | 0.337 | 0.83 (0.80–0.89) | 0.2 | 0.881 |
| GPT-5.4 · none | 2.7 | 0.27 | 0.104 | 0.51 (0.25–0.65) | 1.0 | 0.869 |
| GPT-5.4 · medium | 3.4 | 0.23 | 0.115 | 0.50 (0.34–0.63) | 1.0 | 0.899 |
| GPT-5.4 · high | 3.3 | 0.23 | 0.106 | 0.50 (0.36–0.61) | 1.0 | 0.896 |
| DeepSeek-V4-Flash · none | 1.3 | 0.43 | 0.073 | 0.41 (0.30–0.48) | 1.4 | 0.872 |
| DeepSeek-V4-Flash · high | 2.1 | 0.33 | 0.111 | 0.62 (0.43–0.71) | 0.6 | 0.937 |
| DeepSeek-V4-Flash · max | 2.1 | 0.33 | 0.158 | 0.82 (0.74–0.89) | 0.2 | 0.905 |
| DeepSeek-V4-Pro · none | 1.5 | 0.40 | 0.110 | 0.52 (0.32–0.61) | 0.9 | 0.865 |
| DeepSeek-V4-Pro · high | 1.9 | 0.35 | 0.166 | 0.79 (0.64–0.89) | 0.3 | 0.907 |
| DeepSeek-V4-Pro · max | 2.9 | 0.26 | 0.191 | 0.73 (0.64–0.85) | 0.4 | 0.912 |
| Claude Fable 5 · low | 4.1 | 0.20 | 0.206 | 0.51 (0.39–0.58) | 0.9 | 0.942 |
| Claude Fable 5 · medium | 4.0 | 0.20 | 0.207 | 0.53 (0.42–0.61) | 0.9 | 0.944 |
| Claude Fable 5 · high | 4.4 | 0.19 | 0.194 | 0.52 (0.41–0.62) | 0.9 | 0.940 |
| Claude Sonnet 5 · low | 4.3 | 0.19 | 0.170 | 0.51 (0.37–0.59) | 1.0 | 0.922 |
| Claude Sonnet 5 · medium | 4.2 | 0.19 | 0.161 | 0.48 (0.37–0.59) | 1.1 | 0.919 |
| Claude Sonnet 5 · high | 3.6 | 0.22 | 0.162 | 0.48 (0.39–0.57) | 1.1 | 0.911 |
