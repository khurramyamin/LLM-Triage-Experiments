# Was ChatGPT Health under-triage a utility problem or an off-curve problem?

**Their decisions**: ChatGPT Health (gpt-5-mini thinking) 4-level triage A/B/C/D, reference factorial condition (White man, no anchor, no barrier) — one decision per case, matching the plain vignette our beliefs used. Binary admit = triage is **D** ("go to the ED now").

**Their operating point** (78 cases, 40 truly need ER): sensitivity (TPR) = **50%**, false-positive rate = **8%** (specificity 92%), admit rate = 29%, accuracy 71%.

This is the paper's under-triage in ROC terms: a **very FP-averse (bottom-left) operating point** — it rarely over-triages, but catches only half of true emergencies.

## On-curve vs off-curve

For each belief frontier we read off the sensitivity the frontier *could* achieve at ChatGPT Health's own false-positive rate. The **TPR deficit** is how far their point sits **below** the curve. ~0 => on the curve (an operating-point / utility problem you could fix by prompting a more FN-averse cost function). Large => below the curve (a discrimination problem no single utility fixes).

| belief frontier | AUC | frontier sens @ their FPR | their sens | TPR deficit (below curve) | frontier FPR @ their sens | FPR excess |
|---|---|---|---|---|---|---|
| gpt-5-mini_re-high | 0.89 | 62% | 50% | **+12%** | 0% | +8% |
| gpt-5-mini_re-medium | 0.89 | 65% | 50% | **+15%** | 5% | +3% |
| gpt-5-mini_re-minimal | 0.86 | 60% | 50% | **+10%** | 5% | +3% |
| gpt-5.4-mini_re-high | 0.90 | 70% | 50% | **+20%** | 8% | +0% |
| gpt-5.4-mini_re-medium | 0.90 | 68% | 50% | **+18%** | 3% | +5% |
| gpt-5.4-mini_re-none | 0.87 | 52% | 50% | **+3%** | 8% | +0% |
| gpt-5.4_re-high | 0.91 | 78% | 50% | **+28%** | 5% | +3% |
| gpt-5.4_re-medium | 0.89 | 70% | 50% | **+20%** | 5% | +3% |
| gpt-5.4_re-none | 0.90 | 57% | 50% | **+7%** | 5% | +3% |

Mean TPR deficit across frontiers: **+15%** (against the stronger reasoning frontiers: **+19%**).

## Interpretation — a two-part answer

**1. Yes, it is partly an operating-point / utility problem.** ChatGPT Health sits in the **bottom-left (FP-averse) corner** of the ROC (FPR 8%, sensitivity 50%): it almost never over-triages but misses half of real emergencies. That is exactly what an over-conservative implicit cost function (heavily penalising false alarms) looks like — the kind of thing a more FN-averse prompt could, in principle, push up and to the right.

**2. But it is also partly an off-curve discrimination failure.** At that same 8% false-positive rate, thresholding well-ranked beliefs would catch **62%–78%** of emergencies rather than 50%. Their point lies **+19%** below the stronger belief frontiers, so a real sensitivity gap remains *even after* accounting for their conservative FPR. That portion cannot be fixed by any single cost ratio — the decisions simply don't track a good probability ranking as tightly as the frontier does.

**Frontier-dependence.** The gap is largest against strong reasoning-model beliefs (gpt-5.4 high: +28%) and nearly closes against the weakest belief ranking (mini/none: +3%). So *how much* is 'off-curve' depends on how good a belief model you compare against; against a capable one, a substantial chunk of the under-triage is a genuine discrimination gap, not just a threshold choice.

Pooling all 16 factorial conditions gives essentially the same point (sens 42%, FPR 8%) — slightly *more* under-triage, since anchoring/barrier perturbations mostly de-escalate.

**Same-family check (strongest evidence).** ChatGPT Health's documented backbone is **gpt-5-mini**. Using our beliefs elicited from that same model (gpt-5-mini, high reasoning; AUC 0.89) as the frontier, its product decisions still sit **+12%** below the curve at their own FPR (frontier 62% vs their 50% sensitivity). So the off-curve gap is not an artefact of comparing against a different, stronger model — even the same model family, asked for a probability, ranks emergencies well enough to beat the deployed triage decisions.

*Caveat*: their decisions come from the **ChatGPT Health web product** (system prompt, product guardrails, possible retrieval), while our beliefs come from the raw gpt-5-mini API. The residual gap therefore reflects the deployed product's decision policy, not necessarily the bare model's — but that is exactly the system the paper evaluated.

## Figure
- `original_vs_belief_roc.png`
