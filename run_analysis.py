from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis import FIGURES_DIR, SUMMARY_PATH, load_study_data
from analysis import constants as C
from analysis.data_io import (
    endpoint_counts,
    endpoint_gold,
    original_paper_decisions,
    original_paper_labels,
    validate_repetition_collection,
)
from analysis.figures import (
    build_belief_distributions,
    build_calibration,
    build_decision_consistency,
    build_default_utilities,
    build_main_roc,
    build_nature_consistency,
    build_recovered_capability,
    build_recovered_threshold,
    build_roc_grid,
)
from analysis.metrics import (
    best_fixed_threshold,
    bootstrap_prop_ci,
    bootstrap_ratio_ci,
    calibration_curve,
    consistency_indicators,
    fit_cost_function_from_maps,
    hierarchical_roc_ci,
    mean_repetition_scores,
    operating_point,
    roc_curve,
    scores_and_labels,
    spearman,
)
from analysis.prompts import build_prompt_figures


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_summary(
    repetitions_root: Path | None = None,
    roc_bootstrap_draws: int = C.ROC_BOOTSTRAP_DRAWS,
) -> dict:
    study = load_study_data(repetitions_root=repetitions_root)
    validate_repetition_collection(study)
    original_decisions = original_paper_decisions(study.original_paper)
    summary = {
        "metadata": {
            "bootstrap_draws": C.BOOTSTRAP_DRAWS,
            "bootstrap_seed": C.BOOTSTRAP_SEED,
            "roc_bootstrap_draws": roc_bootstrap_draws,
            "roc_bootstrap_unit": "case_id cluster (all 16 factorial variants retained)",
            "belief_repetitions": C.BELIEF_REPETITION_COUNT,
            "calibration_bins": C.CALIBRATION_BINS,
            "response_handling": "Unanswered or unparseable model responses are excluded per analysis; denominators are reported below.",
            "utility_costs": [{"c_fp": fp, "c_fn": fn, "ratio_fn_fp": fn / fp} for fp, fn in C.UTILITY_COSTS],
            "source_rows": study.source_rows,
            "repetition_source_rows": study.repetition_source_rows,
            "repetition_data_loaded": bool(study.belief_repetitions),
        },
        "endpoints": {
            "primary": endpoint_counts(study.meta, "primary"),
            "expanded": endpoint_counts(study.meta, "expanded"),
        },
        "configs": {},
        "chatgpt_health": {},
    }

    gold_by_endpoint = {
        endpoint: endpoint_gold(study.meta, endpoint) for endpoint in ("primary", "expanded")
    }
    for endpoint in ("primary", "expanded"):
        gold = original_paper_labels(study.original_paper, endpoint)
        decisions = {context: original_decisions[context] for context in gold}
        summary["chatgpt_health"][endpoint] = operating_point(decisions, gold)

    for name, config in study.configs.items():
        baseline_decisions = config.decisions.get("decision_baseline", {})
        baseline_fit = fit_cost_function_from_maps(config.belief, baseline_decisions)
        recovered_by_regime: dict[str, float | None] = {}
        threshold_following: dict[str, dict[str, float | None]] = {}
        est_consistency: dict[str, float | None] = {}
        est_consistency_n: dict[str, int] = {}
        est_consistency_ci: dict[str, list[float] | None] = {}
        nature_agreement: dict[str, float | None] = {}
        nature_agreement_n: dict[str, int] = {}
        nature_agreement_ci: dict[str, list[float] | None] = {}

        target_logs: list[float] = []
        recovered_logs: list[float] = []

        all_fit_records = {"decision_baseline": baseline_fit}
        for regime in C.UTILITY_REGIMES:
            fit = fit_cost_function_from_maps(config.belief, config.decisions.get(regime, {}))
            all_fit_records[regime] = fit
            recovered_by_regime[regime] = fit["ratio_fn_fp"]
            if fit["ratio_fn_fp"] is not None:
                target_logs.append(np.log(C.TARGET_RATIOS[regime]))
                recovered_logs.append(np.log(fit["ratio_fn_fp"]))

        for regime in ["decision_baseline", *C.UTILITY_REGIMES]:
            fit = all_fit_records[regime]
            indicators = consistency_indicators(config.belief, config.decisions.get(regime, {}), fit["threshold"])
            est_consistency[regime] = float(np.mean(indicators)) if indicators else None
            est_consistency_n[regime] = len(indicators)
            ci = bootstrap_prop_ci(indicators) if indicators else None
            est_consistency_ci[regime] = list(ci) if ci is not None else None

            decision_map = config.decisions.get(regime, {})
            contexts = [context for context in decision_map if context in original_decisions]
            if contexts:
                matches = [int(decision_map[context] == original_decisions[context]) for context in contexts]
                nature_agreement[regime] = float(np.mean(matches))
                nature_agreement_n[regime] = len(matches)
                ci = bootstrap_prop_ci(matches)
                nature_agreement_ci[regime] = list(ci) if ci is not None else None
            else:
                nature_agreement[regime] = None
                nature_agreement_n[regime] = 0
                nature_agreement_ci[regime] = None

        for regime in C.THRESHOLD_REGIMES:
            fit = fit_cost_function_from_maps(config.belief, config.decisions.get(regime, {}))
            threshold_following[regime] = {
                "prompted_threshold": C.PROMPTED_THRESHOLDS[regime],
                "recovered_threshold": fit["threshold"],
                "recovered_ratio_fn_fp": fit["ratio_fn_fp"],
                "n": fit["n"],
            }

        endpoint_data = {}
        for endpoint, gold in gold_by_endpoint.items():
            repetitions = study.belief_repetitions.get(name)
            repeated_coverage = None
            roc_ci = None
            roc_belief = config.belief
            if repetitions and any(repetitions):
                per_repetition = [
                    sum(context in repetition for context in gold)
                    for repetition in repetitions
                ]
                complete_collection = (
                    len(repetitions) == C.BELIEF_REPETITION_COUNT
                    and all(
                        set(study.meta).issubset(repetition)
                        for repetition in repetitions
                    )
                )
                all_repetitions = sum(
                    all(context in repetition for repetition in repetitions)
                    for context in gold
                )
                any_repetition = sum(
                    any(context in repetition for repetition in repetitions)
                    for context in gold
                )
                roc_belief = mean_repetition_scores(
                    repetitions, gold, require_all=complete_collection
                )
                repeated_coverage = {
                    "parsed_per_repetition": per_repetition,
                    "contexts_with_any_repetition": any_repetition,
                    "contexts_with_all_five": all_repetitions,
                    "display_requires_all_five": complete_collection,
                    "collection_complete": complete_collection,
                }
                if roc_belief:
                    try:
                        roc_ci = hierarchical_roc_ci(
                            repetitions,
                            gold,
                            n_boot=roc_bootstrap_draws,
                            seed=C.BOOTSTRAP_SEED,
                        )
                    except ValueError:
                        roc_ci = None

            scores, labels = scores_and_labels(roc_belief, gold)
            fpr, tpr, auc = roc_curve(scores, labels)
            operating_points = {}
            for regime in ["decision_baseline", *C.UTILITY_REGIMES]:
                decision_map = config.decisions.get(regime, {})
                matched_gold = {
                    context: gold[context]
                    for context in decision_map
                    if context in roc_belief and context in gold
                }
                operating_points[regime] = operating_point(decision_map, matched_gold)
            best_fixed = {
                C.ratio_key(ratio): best_fixed_threshold(roc_belief, gold, ratio)
                for ratio in C.FIXED_UTILITY_RATIOS
            }
            endpoint_data[endpoint] = {
                "coverage": {
                    "expected_n": len(gold),
                    "observed_n": len(labels),
                    "missing_n": len(gold) - len(labels),
                    "observed_positive": int(labels.sum()),
                    "observed_negative": int(len(labels) - labels.sum()),
                    "repeated_beliefs": repeated_coverage,
                },
                "roc": {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": auc,
                    "n": len(labels),
                    "score_source": (
                        "mean_five_repetition_probability"
                        if repeated_coverage
                        and repeated_coverage["collection_complete"]
                        else "mean_available_repetition_probability"
                        if repeated_coverage
                        else "single_run_probability"
                    ),
                    "confidence_band": roc_ci,
                },
                "operating_points": operating_points,
                "best_fixed": best_fixed,
                "calibration": calibration_curve(config.belief, gold),
            }

        summary["configs"][name] = {
            "family": config.family,
            "reasoning": config.reasoning,
            "paper_config": name in C.PAPER_CONFIG_ORDER,
            "baseline_fit": baseline_fit,
            "baseline_ratio_ci": None,
            "recovered_by_regime": recovered_by_regime,
            "spearman_recovered_utilities": spearman(target_logs, recovered_logs) if len(recovered_logs) >= 3 else None,
            "est_consistency": est_consistency,
            "est_consistency_n": est_consistency_n,
            "est_consistency_ci": est_consistency_ci,
            "nature_agreement": nature_agreement,
            "nature_agreement_n": nature_agreement_n,
            "nature_agreement_ci": nature_agreement_ci,
            "threshold_following": threshold_following,
            "endpoints": endpoint_data,
        }
        baseline_ci = bootstrap_ratio_ci(config.belief, baseline_decisions) if baseline_decisions else None
        if baseline_ci is not None:
            summary["configs"][name]["baseline_ratio_ci"] = list(baseline_ci)
    return {"study": study, "summary": summary}


def write_outputs(study, summary: dict, output_dir: Path = FIGURES_DIR) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    build_main_roc(study, summary, "primary", output_dir / "fig1_emergency_triage_performance.png")
    generated.append("fig1_emergency_triage_performance.png")
    build_main_roc(study, summary, "expanded", output_dir / "fig1_original_emergency_triage_performance.png")
    generated.append("fig1_original_emergency_triage_performance.png")
    build_default_utilities(summary, output_dir / "fig1b_default_utilities.png")
    generated.append("fig1b_default_utilities.png")
    build_recovered_capability(summary, output_dir / "fig2_recovered_capability.png")
    generated.append("fig2_recovered_capability.png")
    for provider_slug in ("gpt", "deepseek", "claude"):
        build_roc_grid(study, summary, "primary", provider_slug, output_dir / f"fig3_roc_grid_{provider_slug}.png")
        generated.append(f"fig3_roc_grid_{provider_slug}.png")
        build_roc_grid(study, summary, "expanded", provider_slug, output_dir / f"fig3_original_roc_grid_{provider_slug}.png")
        generated.append(f"fig3_original_roc_grid_{provider_slug}.png")
    build_decision_consistency(summary, output_dir / "decision_belief_consistency.png")
    generated.append("decision_belief_consistency.png")
    build_nature_consistency(summary, output_dir / "nature_decision_consistency.png")
    generated.append("nature_decision_consistency.png")
    build_recovered_threshold(summary, output_dir / "fig4_recovered_threshold.png")
    generated.append("fig4_recovered_threshold.png")
    build_belief_distributions(study, summary, "primary", output_dir / "fig_belief_distributions.png")
    generated.append("fig_belief_distributions.png")
    build_belief_distributions(study, summary, "expanded", output_dir / "fig_original_belief_distributions.png")
    generated.append("fig_original_belief_distributions.png")
    build_calibration(summary, "primary", output_dir / "fig_calibration_curves.png")
    generated.append("fig_calibration_curves.png")
    build_calibration(summary, "expanded", output_dir / "fig_original_calibration_curves.png")
    generated.append("fig_original_calibration_curves.png")
    generated.extend(build_prompt_figures(output_dir))
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repetitions-root",
        type=Path,
        default=None,
        help=(
            "Belief repetition root (default: "
            f"{C.BELIEF_REPETITIONS_DIR}); absent data preserves one-run analysis."
        ),
    )
    parser.add_argument(
        "--roc-bootstrap-draws",
        type=int,
        default=C.ROC_BOOTSTRAP_DRAWS,
        help="Hierarchical ROC bootstrap draws (default: 1000).",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help=(
            "Ignore the five-run belief repetitions and use the single belief elicitation, "
            "which reproduces the figures in the manuscript."
        ),
    )
    args = parser.parse_args()
    if args.roc_bootstrap_draws < 1:
        parser.error("--roc-bootstrap-draws must be positive")
    if args.single_run:
        args.repetitions_root = Path("/nonexistent-belief-repetitions")
    output_dir = FIGURES_DIR / "regenerated" / ("single_run" if args.single_run else "five_run")

    built = build_summary(
        repetitions_root=args.repetitions_root,
        roc_bootstrap_draws=args.roc_bootstrap_draws,
    )
    study = built["study"]
    summary = built["summary"]
    generated = write_outputs(study, summary, output_dir)
    summary["metadata"]["generated_figures"] = generated
    summary["metadata"]["figure_output_dir"] = str(output_dir.relative_to(FIGURES_DIR.parent.parent))
    summary["metadata"]["belief_mode"] = "single_run" if args.single_run else "five_run_mean"
    summary["metadata"]["figure_sizes_bytes"] = {
        filename: (output_dir / filename).stat().st_size for filename in generated
    }
    summary_path = SUMMARY_PATH if not args.single_run else SUMMARY_PATH.with_name("analysis_summary_single_run.json")
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {len(generated)} figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
