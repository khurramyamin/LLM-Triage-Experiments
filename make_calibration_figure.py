r"""Generate reliability diagrams for elicited emergency-care probabilities.

Each panel compares the mean elicited belief with the observed emergency-care
frequency in 10 equal-width probability bins. The expected calibration error
(ECE) is the sample-weighted mean absolute gap between those quantities.

Run:
    .venv\Scripts\python.exe make_calibration_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_sweep as A
from make_paper_figures import (
    FAMILY_CMAP,
    FAMILY_LABEL,
    FAMILY_ORDER,
    FAMILY_REASONS,
    OUT,
    REASON_LABEL,
)


BELIEF_SWEEP = Path("revealed_preferences/2026-07-01_factorial")
N_BINS = 10


def calibration_curve(
    belief: dict[str, float],
    gold: dict[str, int],
    n_bins: int = N_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return nonempty-bin mean beliefs, event rates, counts, and ECE."""
    contexts = sorted(set(belief) & set(gold))
    if not contexts:
        raise ValueError("No observations have both an elicited belief and gold label")

    probabilities = np.asarray([belief[context] for context in contexts], dtype=float)
    labels = np.asarray([gold[context] for context in contexts], dtype=int)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Elicited beliefs must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Elicited beliefs must lie in [0, 1]")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Gold labels must be binary")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.searchsorted(edges, probabilities, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mean_belief, event_rate, counts = [], [], []
    for bin_index in range(n_bins):
        in_bin = bin_indices == bin_index
        count = int(in_bin.sum())
        if count == 0:
            continue
        mean_belief.append(float(probabilities[in_bin].mean()))
        event_rate.append(float(labels[in_bin].mean()))
        counts.append(count)

    mean_belief_array = np.asarray(mean_belief)
    event_rate_array = np.asarray(event_rate)
    count_array = np.asarray(counts)
    ece = float(
        np.sum(count_array * np.abs(event_rate_array - mean_belief_array))
        / len(probabilities)
    )
    return mean_belief_array, event_rate_array, count_array, ece


def build() -> None:
    """Build the model-by-reasoning calibration grid."""
    row_count = len(FAMILY_ORDER)
    fig, axes = plt.subplots(
        row_count,
        3,
        figsize=(9.4, 3.0 * row_count),
        sharex=True,
        sharey=True,
    )

    for row, family in enumerate(FAMILY_ORDER):
        for column, rank in enumerate((0, 1, 2)):
            reason = FAMILY_REASONS[family][rank]
            name = f"{family}_re-{reason}"
            results_path = BELIEF_SWEEP / name / "results.csv"
            if not results_path.exists():
                raise FileNotFoundError(f"Missing calibration input: {results_path}")

            belief, _decisions, gold = A.load_results(results_path)
            mean_belief, event_rate, counts, ece = calibration_curve(belief, gold)
            sample_size = int(counts.sum())
            color = FAMILY_CMAP[family]((0.55, 0.75, 0.98)[rank])
            marker_sizes = 28.0 + 105.0 * counts / counts.max()

            ax = axes[row, column]
            ax.plot([0, 1], [0, 1], "--", color="0.45", lw=1.2, zorder=1)
            ax.vlines(
                mean_belief,
                np.minimum(mean_belief, event_rate),
                np.maximum(mean_belief, event_rate),
                color=color,
                alpha=0.3,
                lw=2.0,
                zorder=2,
            )
            ax.plot(mean_belief, event_rate, color=color, lw=2.0, zorder=3)
            ax.scatter(
                mean_belief,
                event_rate,
                s=marker_sizes,
                color=color,
                edgecolor="black",
                linewidth=0.55,
                zorder=4,
            )
            ax.text(
                0.96,
                0.05,
                f"ECE = {ece:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10.5,
                bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
            )
            ax.set_title(
                f"{FAMILY_LABEL[family]}\nReasoning: {REASON_LABEL[reason]}",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.tick_params(axis="both", labelsize=14)
            ax.grid(True, alpha=0.22)
            ax.set_aspect("equal", adjustable="box")

            print(f"{name:34s} n={sample_size:4d} ECE={ece:.4f}")

    fig.supxlabel("Mean Elicited Probability Within Bin", fontsize=20, y=0.008)
    fig.supylabel("Reference Emergency Care Frequency", fontsize=20, x=0.04)
    fig.tight_layout(rect=(0.065, 0.04, 1.0, 0.995), h_pad=1.5, w_pad=1.0)

    OUT.mkdir(parents=True, exist_ok=True)
    output_path = OUT / "fig_calibration_curves.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    build()
