**Dependence of discrimination on the positive scenarios. AUROC of the elicited probabilities on the primary endpoint using all 64 emergency variants, only the 32 DKA variants, or only the 32 asthma variants against the same 512 non-emergency variants; on the expanded set; and the range obtained when each of the 20 positive scenarios of the expanded set is left out in turn.**

| Configuration | Primary (64 vs 512) | DKA only | Asthma only | Expanded (640 vs 608) | Leave-one-scenario-out range |
|---|---|---|---|---|---|
| GPT-5-mini · minimal | 0.977 | 0.981 | 0.972 | 0.849 | 0.841–0.857 |
| GPT-5-mini · medium | 0.993 | 0.994 | 0.992 | 0.885 | 0.879–0.898 |
| GPT-5-mini · high | 0.993 | 0.998 | 0.988 | 0.881 | 0.875–0.891 |
| GPT-5.4 · none | 0.944 | 0.983 | 0.906 | 0.869 | 0.862–0.881 |
| GPT-5.4 · medium | 0.966 | 0.976 | 0.956 | 0.899 | 0.894–0.913 |
| GPT-5.4 · high | 0.953 | 0.974 | 0.932 | 0.896 | 0.891–0.910 |
| DeepSeek-V4-Flash · none | 0.940 | 0.953 | 0.928 | 0.872 | 0.866–0.890 |
| DeepSeek-V4-Flash · high | 0.993 | 0.994 | 0.992 | 0.937 | 0.934–0.949 |
| DeepSeek-V4-Flash · max | 0.989 | 0.995 | 0.984 | 0.905 | 0.900–0.918 |
| DeepSeek-V4-Pro · none | 0.910 | 0.938 | 0.882 | 0.865 | 0.858–0.876 |
| DeepSeek-V4-Pro · high | 0.969 | 0.974 | 0.964 | 0.907 | 0.902–0.918 |
| DeepSeek-V4-Pro · max | 0.983 | 0.991 | 0.975 | 0.912 | 0.908–0.923 |
| Claude Fable 5 · low | 0.995 | 0.999 | 0.992 | 0.942 | 0.938–0.953 |
| Claude Fable 5 · medium | 0.995 | 1.000 | 0.990 | 0.944 | 0.941–0.956 |
| Claude Fable 5 · high | 0.993 | 0.997 | 0.989 | 0.940 | 0.936–0.953 |
| Claude Sonnet 5 · low | 0.975 | 0.968 | 0.982 | 0.922 | 0.918–0.934 |
| Claude Sonnet 5 · medium | 0.970 | 0.968 | 0.971 | 0.919 | 0.915–0.929 |
| Claude Sonnet 5 · high | 0.961 | 0.968 | 0.953 | 0.911 | 0.907–0.920 |
