# Was ChatGPT Health under-triage a utility problem or an off-curve problem?

**Their decisions**: ChatGPT Health (gpt-5-mini thinking) 4-level triage A/B/C/D, across **all 16 factorial conditions** (race x gender x anchoring x access-barrier) of every case. We elicited our beliefs on the *identical* per-variant context, so this is a **fully matched** belief-vs-decision comparison (no reference-condition proxy). Binary admit = triage is **D** ("go to the ED now").

**Their operating point** (1248 variants, 640 truly need ER): sensitivity (TPR) = **42%**, false-positive rate = **8%** (specificity 92%), admit rate = 26%, accuracy 66%.

This is the paper's under-triage in ROC terms: a **very FP-averse (bottom-left) operating point** — it rarely over-triages, but catches only about half of true emergencies.

## On-curve vs off-curve

For each belief frontier we read off the sensitivity the frontier *could* achieve at ChatGPT Health's own false-positive rate. The **TPR deficit** is how far their point sits **below** the curve. ~0 => on the curve (an operating-point / utility problem you could fix by prompting a more FN-averse cost function). Large => below the curve (a discrimination problem no single utility fixes).

| belief frontier | AUC | frontier sens @ their FPR | their sens | TPR deficit (below curve) | frontier FPR @ their sens | FPR excess |
|---|---|---|---|---|---|---|
| gpt-5-mini_re-high | 0.88 | 50% | 42% | **+8%** | 2% | +6% |
| gpt-5-mini_re-medium | 0.88 | 52% | 42% | **+10%** | 3% | +6% |
| gpt-5-mini_re-minimal | 0.86 | 42% | 42% | **+0%** | 8% | +1% |
| gpt-5.4-mini_re-high | 0.91 | 65% | 42% | **+23%** | 1% | +8% |
| gpt-5.4-mini_re-medium | 0.90 | 61% | 42% | **+19%** | 0% | +8% |
| gpt-5.4-mini_re-none | 0.86 | 51% | 42% | **+9%** | 3% | +6% |
| gpt-5.4_re-high | 0.90 | 62% | 42% | **+19%** | 2% | +6% |
| gpt-5.4_re-medium | 0.90 | 60% | 42% | **+18%** | 3% | +6% |
| gpt-5.4_re-none | 0.87 | 46% | 42% | **+4%** | 6% | +3% |

Mean TPR deficit across frontiers: **+12%** (against the stronger reasoning frontiers: **+16%**).

## Interpretation — a two-part answer

**1. Yes, it is partly an operating-point / utility problem.** ChatGPT Health sits in the **bottom-left (FP-averse) corner** of the ROC (FPR 8%, sensitivity 42%): it almost never over-triages but misses half of real emergencies. That is exactly what an over-conservative implicit cost function (heavily penalising false alarms) looks like — the kind of thing a more FN-averse prompt could, in principle, push up and to the right.

**2. But it is also partly an off-curve discrimination failure.** At that same 8% false-positive rate, thresholding well-ranked beliefs would catch **50%–65%** of emergencies rather than 42%. Their point lies **+16%** below the stronger belief frontiers, so a real sensitivity gap remains *even after* accounting for their conservative FPR. That portion cannot be fixed by any single cost ratio — the decisions simply don't track a good probability ranking as tightly as the frontier does.

**Frontier-dependence.** The gap is largest against strong reasoning-model beliefs (gpt-5.4 high: +23%) and nearly closes against the weakest belief ranking (mini/none: +0%). So *how much* is 'off-curve' depends on how good a belief model you compare against; against a capable one, a substantial chunk of the under-triage is a genuine discrimination gap, not just a threshold choice.

This point uses **all 1248 factorial variants** with our beliefs elicited on the identical per-variant context — a fully matched comparison, not a reference-condition proxy.

**Same-family check (strongest evidence).** ChatGPT Health's documented backbone is **gpt-5-mini**. Using our beliefs elicited from that same model (gpt-5-mini, high reasoning; AUC 0.88) as the frontier, its product decisions still sit **+8%** below the curve at their own FPR (frontier 50% vs their 42% sensitivity). So the off-curve gap is not an artefact of comparing against a different, stronger model — even the same model family, asked for a probability, ranks emergencies well enough to beat the deployed triage decisions.

*Caveat*: their decisions come from the **ChatGPT Health web product** (system prompt, product guardrails, possible retrieval), while our beliefs come from the raw gpt-5-mini API. The residual gap therefore reflects the deployed product's decision policy, not necessarily the bare model's — but that is exactly the system the paper evaluated.

## Figure
- `original_vs_belief_roc.png`
