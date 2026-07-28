#!/usr/bin/env python3
"""
make_example_dataset.py — generate a deterministic synthetic dataset for the demo.

Writes ``example_dataset.csv`` with three columns:

    belief    : the model's elicited probability of the positive class in [0, 1]
    decision  : the observed binary action (1 = act / refer), from a
                safety-leaning decision rule (acts when belief >= 0.30)
    label     : the ground-truth binary outcome (1 = positive)

The beliefs are deliberately **mis-calibrated** (biased upward via a monotone
squashing of the true event probability) so that the accuracy-optimal (1:1)
belief threshold is *not* 0.5. This makes the demo interesting: the best fixed
*utility* ratio you should prompt with to hit a 1:1 cost objective differs from
1:1, which is the whole point of separating beliefs from utilities.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

N = 600
SEED = 7
OUT = Path(__file__).with_name("example_dataset.csv")


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Latent risk -> true event probability (well-separated but noisy).
    latent = rng.normal(0.0, 1.0, size=N)
    true_p = 1.0 / (1.0 + np.exp(-1.6 * latent))
    labels = (rng.uniform(size=N) < true_p).astype(int)

    # The model's beliefs: a noisy, upward-biased read of the true probability
    # (systematically over-estimates risk -> optimal accuracy threshold > 0.5).
    logit = np.log(true_p / (1.0 - true_p))
    biased_logit = 0.9 * logit + 1.1 + rng.normal(0.0, 0.6, size=N)
    beliefs = 1.0 / (1.0 + np.exp(-biased_logit))
    beliefs = np.clip(beliefs, 0.001, 0.999)

    # Observed decisions: a safety-leaning agent that refers whenever its belief
    # clears a low bar (0.30) -> revealed FN/FP ratio well above 1.
    decisions = (beliefs >= 0.30).astype(int)

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["belief", "decision", "label"])
        for b, d, y in zip(beliefs, decisions, labels):
            w.writerow([f"{b:.4f}", int(d), int(y)])

    print(f"wrote {OUT}  ({N} rows, {int(labels.sum())} positive)")


if __name__ == "__main__":
    main()
