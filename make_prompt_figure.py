r"""Generate separate images for the shared context and four prompt conditions.

The displayed text is imported from revealed_preferences.py so the images stay
synchronized with the prompt templates used in the experiments.

Run:
    .venv\Scripts\python.exe make_prompt_figure.py
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import revealed_preferences as rp


OUT = Path("nature_medicine_paper/figures")
EXAMPLE_C_FP = 1
EXAMPLE_C_FN = 10
EXAMPLE_CASE_NUMBER = 2
EXAMPLE_VIGNETTE = "without_labs"


def wrap_prompt(prompt: str, width: int) -> str:
    """Wrap prompt text for display while preserving paragraphs and bullets."""
    wrapped_lines: list[str] = []
    for line in prompt.splitlines():
        if not line:
            wrapped_lines.append("")
            continue
        subsequent_indent = "  " if line.startswith("- ") else ""
        wrapped_lines.extend(
            textwrap.wrap(
                line,
                width=width,
                subsequent_indent=subsequent_indent,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(wrapped_lines)


def example_context() -> str:
    """Return one actual clinical context used across all four prompt types."""
    cases = rp.load_vignettes()["cases"]
    case = next(
        (item for item in cases if item["case_number"] == EXAMPLE_CASE_NUMBER),
        None,
    )
    if case is None:
        raise ValueError(f"Example case {EXAMPLE_CASE_NUMBER} was not found")
    return rp.build_context_block(case, EXAMPLE_VIGNETTE)


def render_text_image(
    filename: str,
    title: str,
    text: str,
    facecolor: str,
    edgecolor: str,
    *,
    letter: str | None = None,
    height: float,
    wrap_width: int = 112,
) -> None:
    """Render one context or prompt as an independent image."""
    fig, ax = plt.subplots(figsize=(12.5, height))
    ax.set_axis_off()
    panel = FancyBboxPatch(
        (0.008, 0.02),
        0.984,
        0.96,
        boxstyle="round,pad=0.016,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.8,
        clip_on=False,
    )
    ax.add_patch(panel)

    title_x = 0.09 if letter else 0.035
    if letter:
        ax.text(
            0.035,
            0.91,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
            color=edgecolor,
        )
    ax.text(
        title_x,
        0.91,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.5,
        fontweight="bold",
        color="#202020",
    )
    ax.plot(
        [0.035, 0.965],
        [0.78, 0.78],
        transform=ax.transAxes,
        color=edgecolor,
        alpha=0.42,
        linewidth=1.1,
    )
    ax.text(
        0.04,
        0.71,
        wrap_prompt(text, wrap_width),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.6,
        family="monospace",
        linespacing=1.18,
        color="#181818",
    )

    output_path = OUT / filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {output_path}")


def build() -> None:
    """Render the shared context and each prompt condition separately."""
    OUT.mkdir(parents=True, exist_ok=True)
    p_star = rp.admit_threshold(EXAMPLE_C_FP, EXAMPLE_C_FN)

    render_text_image(
        "fig_prompt_context.png",
        "Example patient context shared across all four prompts",
        example_context(),
        "#fbfbfb",
        "#444444",
        height=3.2,
        wrap_width=112,
    )

    prompts = [
        (
            "fig_prompt_belief.png",
            "a",
            "Belief elicitation",
            rp.BELIEF_QUESTION,
            "#f2f7fc",
            "#2b6da8",
            2.35,
        ),
        (
            "fig_prompt_baseline.png",
            "b",
            "Decision: priorities unspecified",
            rp.DECISION_BASELINE_QUESTION,
            "#f6f6f6",
            "#666666",
            2.35,
        ),
        (
            "fig_prompt_utility.png",
            "c",
            "Decision: utility specified",
            rp.build_decision_utility_question(EXAMPLE_C_FP, EXAMPLE_C_FN),
            "#fff5e9",
            "#c56616",
            5.0,
        ),
        (
            "fig_prompt_threshold.png",
            "d",
            "Decision: probability threshold specified",
            rp.build_decision_threshold_question(p_star),
            "#eef8f1",
            "#31824c",
            5.4,
        ),
    ]
    for filename, letter, title, prompt, facecolor, edgecolor, height in prompts:
        render_text_image(
            filename,
            title,
            prompt,
            facecolor,
            edgecolor,
            letter=letter,
            height=height,
            wrap_width=112,
        )


if __name__ == "__main__":
    build()
