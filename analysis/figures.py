from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from . import constants as C
from .data_io import StudyData, endpoint_gold, paper_config_names


def _cmap(family: str, shade: float):
    return plt.get_cmap(C.FAMILY_CMAP[family])(shade)


def style_for_config(name: str) -> dict[str, object]:
    family, reasoning = C.parse_config(name)
    rank = C.reason_rank(family, reasoning)
    return {
        "color": _cmap(family, C.SHADE_BY_RANK[rank]),
        "marker": C.FAMILY_MARKER[family],
        "linestyle": C.LINESTYLE_BY_RANK[rank],
        "label": f"{C.FAMILY_LABEL[family]} · {C.REASON_LABEL[reasoning]}",
    }


def pretty_config(name: str) -> str:
    family, reasoning = C.parse_config(name)
    return f"{C.FAMILY_LABEL[family]} · {C.REASON_LABEL[reasoning]}"


def build_main_roc(
    study: StudyData,
    summary: dict,
    endpoint: str,
    output_path: Path,
) -> None:
    panel_specs = (
        (
            "gpt-5-mini_re-high",
            "GPT-5 Mini · High",
            "#3976B9",
            "#F3F7FC",
            (0.60, 0.72),
        ),
        (
            "gpt-5.4_re-high",
            "GPT-5.4 · High",
            "#D65A2A",
            "#FCF5F1",
            (0.50, 0.66),
        ),
        (
            "DeepSeek-V4-Pro_re-high",
            "DeepSeek-V4-Pro · High",
            "#2F8A57",
            "#F2F8F4",
            (0.50, 0.84),
        ),
        (
            "Claude-Fable-5_re-high",
            "Claude Fable 5 · High",
            "#A50F15",
            "#FCF2F3",
            (0.44, 0.82),
        ),
    )

    fig, axes = plt.subplots(2, 2, figsize=(18.5, 16.8))
    fig.patch.set_facecolor("#F5F4F0")
    fig.subplots_adjust(
        left=0.12,
        right=0.975,
        top=0.865,
        bottom=0.20,
        wspace=0.33,
        hspace=0.48,
    )

    deployed = summary["chatgpt_health"][endpoint]
    has_band = False
    for ax, (name, panel_name, line_color, panel_bg, best_text) in zip(
        axes.flat, panel_specs
    ):
        cfg = summary["configs"][name]["endpoints"][endpoint]
        roc = cfg["roc"]
        ax.set_facecolor("white")
        ax.fill_between(roc["fpr"], 0, roc["tpr"], color=panel_bg, zorder=0)
        band = cfg["roc"].get("confidence_band")
        if band:
            has_band = True
            ax.fill_between(
                band["fpr"],
                band["lower_tpr"],
                band["upper_tpr"],
                color=line_color,
                alpha=0.20,
                linewidth=0,
                zorder=2,
            )
        ax.plot(
            roc["fpr"],
            roc["tpr"],
            color=line_color,
            lw=2.8,
            zorder=3,
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color="#B9B3A8",
            ls=(0, (1.5, 3.2)),
            lw=1.4,
            zorder=1,
        )

        for regime in C.UTILITY_REGIMES:
            point = cfg["operating_points"][regime]
            if not np.isfinite(point["fpr"]):
                continue
            ratio = C.TARGET_RATIOS[regime]
            ax.plot(
                point["fpr"],
                point["tpr"],
                "o",
                markersize=12,
                markeredgecolor="white",
                markeredgewidth=1.1,
                markerfacecolor=C.RATIO_COLORS[ratio],
                zorder=6,
            )

        baseline = cfg["operating_points"]["decision_baseline"]
        if np.isfinite(baseline["fpr"]):
            ax.plot(
                baseline["fpr"],
                baseline["tpr"],
                "o",
                markersize=12,
                color="#20242A",
                markeredgecolor="white",
                markeredgewidth=1.1,
                zorder=6,
            )

        fixed = cfg["best_fixed"][C.ratio_key(5.0)]
        if np.isfinite(fixed["fpr"]):
            ax.plot(
                fixed["fpr"],
                fixed["tpr"],
                "s",
                color="#F4C542",
                markersize=15,
                markeredgecolor="#20242A",
                markeredgewidth=1.3,
                zorder=8,
            )
            ax.annotate(
                (
                    "Best fixed utility (FN/FP=5)\n"
                    f"{fixed['tpr']:.0%} sensitivity\n"
                    f"{fixed['fpr']:.0%} over-triage"
                ),
                xy=(fixed["fpr"], fixed["tpr"]),
                xytext=best_text,
                fontsize=13,
                color="#25282D",
                fontweight="bold",
                ha="left",
                va="center",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#777169",
                    "lw": 1.3,
                    "shrinkA": 5,
                    "shrinkB": 5,
                },
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "fc": "white",
                    "ec": "#D7D2C9",
                    "lw": 0.8,
                    "alpha": 0.97,
                },
                zorder=9,
            )

        ax.plot(
            deployed["fpr"],
            deployed["tpr"],
            "*",
            color="#D83456",
            markersize=24,
            markeredgecolor="#7E1830",
            markeredgewidth=1.0,
            zorder=8,
        )
        ax.annotate(
            (
                "Deployed tool\n"
                f"{deployed['tpr']:.0%} sens · "
                f"{deployed['fpr']:.0%} over-triage"
            ),
            xy=(deployed["fpr"], deployed["tpr"]),
            xytext=(
                min(deployed["fpr"] + 0.09, 0.67),
                deployed["tpr"] - 0.03,
            ),
            fontsize=12,
            color="#25282D",
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-",
                "color": "#777169",
                "lw": 1.3,
                "shrinkA": 5,
                "shrinkB": 5,
            },
            bbox={
                "boxstyle": "round,pad=0.35",
                "fc": "white",
                "ec": "#D7D2C9",
                "lw": 0.8,
                "alpha": 0.97,
            },
            zorder=9,
        )

        ax.set_xlim(-0.04, 1)
        ax.set_ylim(-0.04, 1.06)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel("False Positive Rate (Over-Triage)", fontsize=18, labelpad=12)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=18, labelpad=14)
        ax.set_title(
            f"{panel_name} (AUC={roc['auc']:.2f})",
            loc="left",
            fontsize=19,
            fontweight="bold",
            pad=15,
        )
        ax.tick_params(labelsize=15, colors="#615D56")
        ax.grid(True, color="#DDD9D1", alpha=0.65, lw=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#DEDAD2")

    for ax in axes.flat:
        pos = ax.get_position()
        card = FancyBboxPatch(
            (pos.x0 - 0.065, pos.y0 - 0.055),
            pos.width + 0.100,
            pos.height + 0.105,
            boxstyle="round,pad=0.008,rounding_size=0.010",
            transform=fig.transFigure,
            facecolor="white",
            edgecolor="#DDD9D1",
            linewidth=0.9,
            zorder=-10,
        )
        fig.add_artist(card)

    ratio_handles = []
    for regime in C.UTILITY_REGIMES:
        ratio = C.TARGET_RATIOS[regime]
        label = C.ratio_label(ratio)
        ratio_handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=13,
                markerfacecolor=C.RATIO_COLORS[ratio],
                markeredgecolor="white",
                label=label[1:] if label.startswith("0.") else label,
            )
        )
    legend_handles = [
        Line2D([], [], linestyle="none", marker=None, label="Prompted utility, FN/FP ="),
        *ratio_handles,
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=13,
            markerfacecolor="#20242A",
            markeredgecolor="white",
            label="No utility (default)",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="s",
            markersize=13,
            markerfacecolor="#F4C542",
            markeredgecolor="#20242A",
            label="Best fixed utility",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="*",
            markersize=17,
            markerfacecolor="#D83456",
            markeredgecolor="#7E1830",
            label="Deployed tool",
        ),
    ]
    if has_band:
        legend_handles.append(
            Patch(facecolor="#7F8790", edgecolor="none", alpha=0.25, label="Pointwise 95% CI")
        )
    left_card = axes[0, 0].get_position()
    right_card = axes[0, 1].get_position()
    legend_left = left_card.x0 - 0.065
    legend_right = right_card.x1 + 0.035
    legend_box = FancyBboxPatch(
        (legend_left, 0.068),
        legend_right - legend_left,
        0.040,
        boxstyle="round,pad=0.008,rounding_size=0.006",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="#DDD9D1",
        linewidth=0.9,
        zorder=10,
    )
    fig.add_artist(legend_box)
    legend = fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(legend_left - 0.010, 0.064),
        ncol=len(legend_handles),
        fontsize=14.0,
        frameon=False,
        columnspacing=0.58,
        handletextpad=0.28,
        handlelength=0.80,
    )
    legend.get_texts()[0].set_fontweight("bold")
    legend.set_zorder(12)
    fig.add_artist(legend)

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.14,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def build_default_utilities(summary: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    names = [name for name in C.PAPER_CONFIG_ORDER if name in summary["configs"]]
    ratios: list[float] = []
    hi_edges: list[float] = []
    for name in names:
        cfg = summary["configs"][name]
        ratio = cfg["baseline_fit"]["ratio_fn_fp"]
        if ratio is None:
            continue
        family, reasoning = C.parse_config(name)
        y = C.reason_rank(family, reasoning)
        ratios.append(ratio)
        ci = cfg["baseline_ratio_ci"]
        if ci is not None:
            lo, hi = ci
            hi_edges.append(hi)
            ax.errorbar(
                ratio,
                y,
                xerr=[[max(ratio - lo, 0.0)], [max(hi - ratio, 0.0)]],
                fmt="none",
                ecolor=_cmap(family, 0.55),
                elinewidth=1.4,
                capsize=3.0,
                capthick=1.4,
                zorder=3,
            )
        ax.plot(
            ratio,
            y,
            marker=C.FAMILY_MARKER[family],
            markersize=13,
            color=_cmap(family, 0.85),
            markeredgecolor="black",
            markeredgewidth=0.7,
            zorder=4,
        )
    upper = max(ratios + hi_edges) * 1.25
    lower = min(ratios) * 0.8
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    tick_labels = [C.ratio_label(value) for value in ticks]
    ax.axvline(1.0, color="0.6", linestyle="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(-0.6, 2.9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(C.TIER_LABELS, fontsize=12)
    ax.set_xticks([tick for tick in ticks if lower <= tick <= upper], labels=[tick_labels[i] for i, tick in enumerate(ticks) if lower <= tick <= upper])
    ax.set_xlabel("Default recovered FN/FP", fontsize=14)
    ax.set_ylabel("Reasoning level", fontsize=14)
    ax.grid(True, axis="x", alpha=0.25)
    ax.text(lower * 1.03, 2.72, "Resource priority", fontsize=11.5, color="0.4", ha="left")
    ax.text(upper / 1.03, 2.72, "Safety priority", fontsize=11.5, color="0.4", ha="right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        Line2D([], [], marker=C.FAMILY_MARKER[family], linestyle="none", markerfacecolor=_cmap(family, 0.85), markeredgecolor="black", markersize=10, label=C.FAMILY_LABEL[family])
        for family in C.FAMILY_ORDER
    ]
    ax.legend(handles=handles, fontsize=10.5, loc="center left", bbox_to_anchor=(1.02, 0.5), title="Model family", title_fontsize=11.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_recovered_capability(summary: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 6.0), sharex=True, sharey=True)
    x_values = [C.TARGET_RATIOS[regime] for regime in C.UTILITY_REGIMES]
    y_values = [
        summary["configs"][name]["recovered_by_regime"].get(regime)
        for name in C.PAPER_CONFIG_ORDER
        if name in summary["configs"]
        for regime in C.UTILITY_REGIMES
        if summary["configs"][name]["recovered_by_regime"].get(regime) is not None
    ]
    y_min = min([0.008] + [value for value in y_values if value and value > 0]) * 0.8
    y_max = max([120.0] + y_values) * 1.15

    for ax, (_slug, title, families) in zip(axes, C.PROVIDER_GROUPS):
        ax.plot([min(x_values), max(x_values)], [min(x_values), max(x_values)], "k--", lw=1.3, alpha=0.55)
        for name in C.PAPER_CONFIG_ORDER:
            if name not in summary["configs"]:
                continue
            family, _reason = C.parse_config(name)
            if family not in families:
                continue
            style = style_for_config(name)
            xs: list[float] = []
            ys: list[float] = []
            for regime in C.UTILITY_REGIMES:
                value = summary["configs"][name]["recovered_by_regime"].get(regime)
                if value is None:
                    continue
                xs.append(C.TARGET_RATIOS[regime])
                ys.append(value)
            ax.plot(
                xs,
                ys,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=2.0,
                markersize=7.5,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=style["label"],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min(x_values), max(x_values))
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x_values, labels=[C.ratio_label(value) for value in x_values])
        ax.set_yticks(x_values, labels=[C.ratio_label(value) for value in x_values])
        ax.grid(True, alpha=0.25, which="both")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.legend(fontsize=8.6, loc="lower right", framealpha=0.92)
    fig.supxlabel("Prompted costs (FN/FP)", fontsize=18, y=0.02)
    fig.supylabel("Recovered costs (FN/FP)", fontsize=18, x=0.03)
    fig.tight_layout(rect=(0.03, 0.04, 1, 1))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_roc_grid(
    study: StudyData,
    summary: dict,
    endpoint: str,
    provider_slug: str,
    output_path: Path,
) -> None:
    provider = next(group for group in C.PROVIDER_GROUPS if group[0] == provider_slug)
    _slug, provider_label, families = provider
    names = [
        name
        for name in paper_config_names(study)
        if C.parse_config(name)[0] in families
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 12.0), squeeze=False)
    legend_handles: list[object] = []
    legend_labels: list[str] = []
    for index, name in enumerate(names):
        ax = axes[index // 3][index % 3]
        cfg = summary["configs"][name]["endpoints"][endpoint]
        band = cfg["roc"].get("confidence_band")
        if band:
            ax.fill_between(
                band["fpr"],
                band["lower_tpr"],
                band["upper_tpr"],
                color="0.55",
                alpha=0.22,
                linewidth=0,
                label="Pointwise 95% CI" if index == 0 else None,
            )
        ax.plot(
            cfg["roc"]["fpr"],
            cfg["roc"]["tpr"],
            color="0.45",
            lw=2.5,
            label=f"Belief ROC (AUROC={cfg['roc']['auc']:.2f})",
        )
        ax.plot([0, 1], [0, 1], "k:", alpha=0.45)
        baseline = cfg["operating_points"]["decision_baseline"]
        if np.isfinite(baseline["fpr"]):
            ax.plot(
                baseline["fpr"],
                baseline["tpr"],
                "o",
                color="black",
                markeredgecolor="black",
                markersize=10,
                label="Default" if index == 0 else None,
            )
        for regime in C.UTILITY_REGIMES:
            point = cfg["operating_points"][regime]
            if np.isfinite(point["fpr"]):
                ratio = C.TARGET_RATIOS[regime]
                ax.plot(
                    point["fpr"],
                    point["tpr"],
                    "o",
                    color=C.RATIO_COLORS[ratio],
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    markersize=9,
                    label=f"Prompt FN/FP={C.ratio_label(ratio)}" if index == 0 else None,
                )
        for ratio in C.FIXED_UTILITY_RATIOS:
            fixed = cfg["best_fixed"][C.ratio_key(ratio)]
            ax.plot(
                fixed["fpr"],
                fixed["tpr"],
                "s",
                color=C.FIXED_RATIO_COLORS[ratio],
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=10,
                label=f"Best fixed FN/FP={C.ratio_label(ratio)}" if index == 0 else None,
            )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=11.5)
        ax.set_ylabel("True Positive Rate", fontsize=11.5)
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"{pretty_config(name)}\nn={cfg['roc']['n']}",
            fontsize=12.5,
            fontweight="bold",
        )
        if index == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    for index in range(len(names), 6):
        axes[index // 3][index % 3].axis("off")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=4,
            fontsize=10,
            framealpha=0.94,
            bbox_to_anchor=(0.5, 0.02),
        )
    subtitle = (
        "Primary endpoint (up to 576 available responses)"
        if endpoint == "primary"
        else "Expanded endpoint (up to 1,248 available responses)"
    )
    fig.suptitle(f"{provider_label} — {subtitle}", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_consistency_family(
    summary: dict,
    metric_key: str,
    ci_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x = list(range(len(C.UTILITY_REGIMES)))
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for name in C.PAPER_CONFIG_ORDER:
        if name not in summary["configs"]:
            continue
        style = style_for_config(name)
        series = summary["configs"][name][metric_key]
        cis = summary["configs"][name][ci_key]
        y: list[float] = []
        err_lo: list[float] = []
        err_hi: list[float] = []
        for regime in C.UTILITY_REGIMES:
            value = series.get(regime)
            ci = cis.get(regime)
            y.append(100.0 * value if value is not None else float("nan"))
            if value is None or ci is None:
                err_lo.append(0.0)
                err_hi.append(0.0)
            else:
                err_lo.append(max(100.0 * value - 100.0 * ci[0], 0.0))
                err_hi.append(max(100.0 * ci[1] - 100.0 * value, 0.0))
        ax.errorbar(
            x,
            y,
            yerr=[err_lo, err_hi],
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=2.0,
            markersize=6.8,
            markeredgecolor="black",
            markeredgewidth=0.5,
            ecolor=style["color"],
            elinewidth=1.0,
            capsize=2.5,
            label=style["label"],
        )
        baseline = series.get("decision_baseline")
        baseline_ci = cis.get("decision_baseline")
        if baseline is not None:
            yerr = None
            if baseline_ci is not None:
                yerr = [
                    [max(100.0 * baseline - 100.0 * baseline_ci[0], 0.0)],
                    [max(100.0 * baseline_ci[1] - 100.0 * baseline, 0.0)],
                ]
            ax.errorbar(
                -0.6,
                100.0 * baseline,
                yerr=yerr,
                fmt=style["marker"],
                color=style["color"],
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=8.0,
                ecolor=style["color"],
                elinewidth=1.0,
                capsize=2.5,
                clip_on=False,
            )
    ax.set_xticks(x, labels=[C.ratio_label(C.TARGET_RATIOS[regime]) for regime in C.UTILITY_REGIMES])
    ax.set_xlim(-0.9, len(C.UTILITY_REGIMES) - 0.45)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Prompted costs (FN/FP)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=12.5)
    ax.set_title(title, fontsize=13.5, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.02,
        0.03,
        "Marker left of x-axis = no-utility baseline",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )
    ax.legend(fontsize=8.4, loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.93)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_decision_consistency(summary: dict, output_path: Path) -> None:
    _plot_consistency_family(
        summary,
        metric_key="est_consistency",
        ci_key="est_consistency_ci",
        title="Decisions are consistent with elicited probability + recovered utility",
        ylabel="Matching decisions (%)",
        output_path=output_path,
    )


def build_nature_consistency(summary: dict, output_path: Path) -> None:
    _plot_consistency_family(
        summary,
        metric_key="nature_agreement",
        ci_key="nature_agreement_ci",
        title="Agreement with published ChatGPT Health decisions",
        ylabel="Decision agreement (%)",
        output_path=output_path,
    )


def build_recovered_threshold(summary: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), sharex=True, sharey=True)
    x_ticks = sorted(C.PROMPTED_THRESHOLDS.values())
    for ax, (_slug, title, families) in zip(axes, C.PROVIDER_GROUPS):
        ax.plot([0, 1], [0, 1], "k--", lw=1.3, alpha=0.55)
        for name in C.PAPER_CONFIG_ORDER:
            if name not in summary["configs"]:
                continue
            family, _reason = C.parse_config(name)
            if family not in families:
                continue
            style = style_for_config(name)
            xs: list[float] = []
            ys: list[float] = []
            for regime in C.THRESHOLD_REGIMES:
                result = summary["configs"][name]["threshold_following"].get(regime)
                if result is None or result["recovered_threshold"] is None:
                    continue
                xs.append(result["prompted_threshold"])
                ys.append(result["recovered_threshold"])
            order = np.argsort(xs)
            ax.plot(
                np.asarray(xs)[order],
                np.asarray(ys)[order],
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=2.0,
                markersize=7.0,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=style["label"],
            )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(x_ticks, labels=[f"{tick:.2f}" for tick in x_ticks], rotation=0)
        ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.legend(fontsize=8.4, loc="upper left", framealpha=0.93)
    fig.supxlabel("Prompted probability threshold", fontsize=18, y=0.03)
    fig.supylabel("Recovered probability threshold", fontsize=18, x=0.03)
    fig.tight_layout(rect=(0.03, 0.04, 1, 1))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_belief_distributions(
    study: StudyData,
    summary: dict,
    endpoint: str,
    output_path: Path,
) -> None:
    gold = endpoint_gold(study.meta, endpoint)
    fig, ax = plt.subplots(figsize=(11.0, 10.0))
    positions: list[float] = []
    labels: list[str] = []
    box_data: list[list[float]] = []
    box_colors: list[tuple] = []
    default_x: list[float] = []
    default_y: list[float] = []
    fixed_x: list[float] = []
    fixed_y: list[float] = []
    pos = 0.0
    for family in C.FAMILY_ORDER:
        for rank in (2, 1, 0):
            reason = C.FAMILY_REASONS[family][rank]
            name = f"{family}_re-{reason}"
            if name not in study.configs or name not in summary["configs"]:
                continue
            config = study.configs[name]
            values = [config.belief[context] for context in gold if context in config.belief]
            positions.append(pos)
            labels.append(
                f"{C.FAMILY_LABEL[family]}  {C.REASON_LABEL[reason]}  (n={len(values)})"
            )
            box_data.append(values)
            box_colors.append(_cmap(family, 0.75))
            default_threshold = summary["configs"][name]["baseline_fit"]["threshold"]
            if default_threshold is not None:
                default_x.append(default_threshold)
                default_y.append(pos)
            fixed = summary["configs"][name]["endpoints"][endpoint]["best_fixed"][C.ratio_key(5.0)]
            fixed_x.append(fixed["threshold"])
            fixed_y.append(pos)
            pos += 1.0
        pos += 0.8
    boxplot = ax.boxplot(
        box_data,
        positions=positions,
        vert=False,
        widths=0.7,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=2.5, markerfacecolor="0.35", markeredgecolor="none", alpha=0.35),
        medianprops=dict(color="black", lw=1.5),
    )
    for patch, color in zip(boxplot["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")
    ax.plot(default_x, default_y, "s", color="#9C27B0", markersize=11, markeredgecolor="black", markeredgewidth=0.8, linestyle="none", zorder=5)
    ax.plot(fixed_x, fixed_y, "s", color="gold", markersize=11, markeredgecolor="black", markeredgewidth=0.8, linestyle="none", zorder=5)
    ax.set_yticks(positions, labels=labels, fontsize=10.5)
    ax.set_xlim(-0.02, 1.02)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    endpoint_title = (
        "primary endpoint (up to 576 responses)"
        if endpoint == "primary"
        else "expanded endpoint (up to 1,248 responses)"
    )
    ax.set_xlabel("Elicited probability of needing emergency care", fontsize=13.5)
    ax.set_title(f"Belief distributions — {endpoint_title}", fontsize=14.5, fontweight="bold")
    handles = [
        Line2D([], [], marker="s", linestyle="none", color="#9C27B0", markeredgecolor="black", label="Default recovered threshold"),
        Line2D([], [], marker="s", linestyle="none", color="gold", markeredgecolor="black", label="Best fixed threshold (FN/FP=5)"),
    ]
    ax.legend(handles=handles, fontsize=10.5, loc="upper right", framealpha=0.93)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_calibration(summary: dict, endpoint: str, output_path: Path) -> None:
    fig, axes = plt.subplots(len(C.FAMILY_ORDER), 3, figsize=(10.2, 3.1 * len(C.FAMILY_ORDER)), sharex=True, sharey=True)
    for row, family in enumerate(C.FAMILY_ORDER):
        for col, rank in enumerate((0, 1, 2)):
            reason = C.FAMILY_REASONS[family][rank]
            name = f"{family}_re-{reason}"
            ax = axes[row, col]
            calibration = summary["configs"][name]["endpoints"][endpoint]["calibration"]
            mean_belief = np.asarray(calibration["mean_belief"], dtype=float)
            event_rate = np.asarray(calibration["event_rate"], dtype=float)
            counts = np.asarray(calibration["counts"], dtype=float)
            color = _cmap(family, C.SHADE_BY_RANK[rank])
            ax.plot([0, 1], [0, 1], "--", color="0.45", lw=1.2)
            ax.vlines(mean_belief, np.minimum(mean_belief, event_rate), np.maximum(mean_belief, event_rate), color=color, alpha=0.28, lw=1.8)
            ax.plot(mean_belief, event_rate, color=color, lw=1.8)
            if len(counts):
                ax.scatter(mean_belief, event_rate, s=28 + 110 * counts / counts.max(), color=color, edgecolor="black", linewidth=0.5)
            ax.text(
                0.96,
                0.05,
                f"ECE={calibration['ece']:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9.5,
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.92),
            )
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.2)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(
                f"{C.FAMILY_LABEL[family]}\n{C.REASON_LABEL[reason]} (n={int(counts.sum())})",
                fontsize=10.8,
                fontweight="bold",
            )
    fig.supxlabel("Mean elicited probability within bin", fontsize=16, y=0.01)
    fig.supylabel("Observed emergency-care frequency", fontsize=16, x=0.03)
    fig.tight_layout(rect=(0.05, 0.03, 1, 0.995), h_pad=1.2, w_pad=0.8)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
