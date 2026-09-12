from __future__ import annotations

import csv
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    import run_belief_repetitions as collector
finally:
    sys.path.remove(str(ROOT))
from analysis import constants as C
from analysis import figures as figure_module
from analysis.figures import build_main_roc
from analysis.metrics import hierarchical_roc_ci, mean_repetition_scores, roc_curve
from analysis.data_io import load_belief_repetitions, validate_repetition_collection


class RepetitionPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = Path(__file__).parent / "artifacts" / "belief_repetition_tests"
        shutil.rmtree(self.artifacts, ignore_errors=True)
        self.artifacts.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.artifacts, ignore_errors=True)

    def _write(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=collector.fs.FACTORIAL_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _row(context: str, probability: str, raw: str = "") -> dict[str, str]:
        return {
            "context_id": context,
            "regime": "belief",
            "parsed_probability": probability,
            "raw_response": raw,
        }

    def test_seed_uses_last_success_and_preserves_resumed_success(self) -> None:
        source = self.artifacts / "source.csv"
        destination = self.artifacts / "destination.csv"
        self._write(
            source,
            [
                self._row("a", "0.1"),
                self._row("a", "0.7"),
                self._row("a", ""),
                self._row("b", "0.2"),
            ],
        )
        self._write(destination, [self._row("b", "0.9")])

        collector.seed_repetition_one(source, destination)
        seeded = collector.last_successful_beliefs(destination)

        self.assertEqual(seeded["a"]["parsed_probability"], "0.7")
        self.assertEqual(seeded["b"]["parsed_probability"], "0.9")
        with destination.open(encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_missing_plan_retries_unparseable_contexts_only(self) -> None:
        path = self.artifacts / "results.csv"
        self._write(
            path,
            [self._row("a", "0.4"), self._row("b", ""), self._row("c", "invalid")],
        )
        items = [
            collector.fs._Item({"context_id": context, "regime": "belief"})
            for context in ("a", "b", "c")
        ]
        self.assertEqual(collector.missing_contexts(items, path), ["b", "c"])

    def test_analysis_loader_ignores_invalid_and_keeps_last_parseable(self) -> None:
        path = (
            self.artifacts
            / "repetition_1"
            / "gpt-5-mini_re-high"
            / "results.csv"
        )
        self._write(
            path,
            [
                self._row("a", "0.2"),
                self._row("a", "invalid"),
                self._row("b", "nan"),
                self._row("c", "1.2"),
            ],
        )
        repetitions, rows = load_belief_repetitions(
            self.artifacts, ["gpt-5-mini_re-high"]
        )
        self.assertEqual(repetitions["gpt-5-mini_re-high"][0], {"a": 0.2})
        self.assertEqual(rows[str(Path("repetition_1") / "gpt-5-mini_re-high" / "results.csv")], 4)

    def test_invalid_results_schema_fails_closed(self) -> None:
        path = self.artifacts / "invalid.csv"
        path.write_text("wrong,columns\n1,2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid results schema"):
            collector.last_successful_beliefs(path)

    def test_partial_collection_is_rejected_for_canonical_analysis(self) -> None:
        study = SimpleNamespace(
            meta={"a": object(), "b": object()},
            belief_repetitions={
                "config-a": [
                    {"a": 0.1, "b": 0.2},
                    {"a": 0.1, "b": 0.2},
                    {"a": 0.1, "b": 0.2},
                    {"a": 0.1, "b": 0.2},
                    {"a": 0.1},
                ]
            },
        )
        with self.assertRaisesRegex(ValueError, "collection is incomplete"):
            validate_repetition_collection(study, required_configs=("config-a",))


class HierarchicalRocTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gold = {
            "case_a__1": 0,
            "case_a__2": 0,
            "case_b__1": 1,
            "case_b__2": 1,
            "case_c__1": 0,
            "case_c__2": 0,
            "case_d__1": 1,
            "case_d__2": 1,
        }
        self.repetitions = [
            {context: (0.8 if label else 0.2) for context, label in self.gold.items()},
            {context: (0.6 if label else 0.4) for context, label in self.gold.items()},
            {context: (0.9 if label else 0.1) for context, label in self.gold.items()},
            {context: (0.7 if label else 0.3) for context, label in self.gold.items()},
            {context: (1.0 if label else 0.0) for context, label in self.gold.items()},
        ]

    def test_center_curve_uses_repetition_means(self) -> None:
        means = mean_repetition_scores(self.repetitions, self.gold, require_all=True)
        self.assertAlmostEqual(means["case_b__1"], 0.8)
        scores = [means[context] for context in self.gold]
        labels = list(self.gold.values())
        auc = roc_curve(scores, labels)[2]
        self.assertEqual(auc, 1.0)

    def test_bootstrap_is_deterministic_bounded_and_clustered(self) -> None:
        first = hierarchical_roc_ci(
            self.repetitions, self.gold, n_boot=40, seed=17, grid_points=21
        )
        second = hierarchical_roc_ci(
            self.repetitions, self.gold, n_boot=40, seed=17, grid_points=21
        )
        self.assertEqual(first, second)
        self.assertEqual(first["case_clusters"], 4)
        self.assertEqual(first["draws_valid"], 40)
        self.assertTrue(np.all(np.asarray(first["lower_tpr"]) >= 0))
        self.assertTrue(np.all(np.asarray(first["upper_tpr"]) <= 1))
        self.assertTrue(
            np.all(np.asarray(first["lower_tpr"]) <= np.asarray(first["upper_tpr"]))
        )

    def test_require_all_excludes_incomplete_context(self) -> None:
        repetitions = [dict(repetition) for repetition in self.repetitions]
        del repetitions[-1]["case_b__1"]
        available = mean_repetition_scores(repetitions, self.gold, require_all=False)
        complete = mean_repetition_scores(repetitions, self.gold, require_all=True)
        self.assertIn("case_b__1", available)
        self.assertNotIn("case_b__1", complete)


class RepetitionFigureLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = Path(__file__).parent / "artifacts" / "figure_layout_tests"
        shutil.rmtree(self.artifacts, ignore_errors=True)
        self.artifacts.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.artifacts, ignore_errors=True)

    @staticmethod
    def _summary() -> dict:
        configs = {}
        for index, name in enumerate(C.FIG1_CONFIGS):
            operating_points = {
                "decision_baseline": {"fpr": 0.08, "tpr": 0.50}
            }
            for regime_index, regime in enumerate(C.UTILITY_REGIMES):
                operating_points[regime] = {
                    "fpr": 0.02 + 0.02 * regime_index,
                    "tpr": min(0.75 + 0.04 * regime_index, 1.0),
                }
            configs[name] = {
                "endpoints": {
                    "primary": {
                        "roc": {
                            "fpr": [0.0, 0.05 + 0.01 * index, 1.0],
                            "tpr": [0.0, 0.90, 1.0],
                            "auc": 0.95,
                            "n": 576,
                            "confidence_band": {
                                "fpr": [0.0, 0.5, 1.0],
                                "lower_tpr": [0.0, 0.75, 1.0],
                                "upper_tpr": [0.1, 0.98, 1.0],
                            },
                        },
                        "operating_points": operating_points,
                        "best_fixed": {
                            C.ratio_key(5.0): {"fpr": 0.08, "tpr": 0.95}
                        },
                    }
                }
            }
        return {
            "configs": configs,
            "chatgpt_health": {"primary": {"fpr": 0.10, "tpr": 0.48}},
        }

    def test_main_roc_matches_four_panel_paper_layout_with_ci(self) -> None:
        captured = {}
        real_subplots = figure_module.plt.subplots

        def capture_subplots(*args, **kwargs):
            fig, axes = real_subplots(*args, **kwargs)
            captured["fig"] = fig
            captured["axes"] = axes
            return fig, axes

        output = self.artifacts / "fig1.png"
        with patch.object(
            figure_module.plt, "subplots", side_effect=capture_subplots
        ) as subplots:
            build_main_roc(SimpleNamespace(), self._summary(), "primary", output)

        subplots.assert_called_once_with(2, 2, figsize=(18.5, 16.8))
        self.assertTrue(output.exists())
        self.assertGreater(output.stat().st_size, 0)
        axes = captured["axes"]
        self.assertEqual(axes.shape, (2, 2))
        self.assertEqual(
            [ax.get_title(loc="left") for ax in axes.flat],
            [
                "GPT-5 Mini · High (AUC=0.95)",
                "GPT-5.4 · High (AUC=0.95)",
                "DeepSeek-V4-Pro · High (AUC=0.95)",
                "Claude Fable 5 · High (AUC=0.95)",
            ],
        )
        self.assertTrue(all(len(ax.collections) >= 2 for ax in axes.flat))
        legend_labels = [
            text.get_text()
            for legend in captured["fig"].legends
            for text in legend.get_texts()
        ]
        self.assertIn("Pointwise 95% CI", legend_labels)


if __name__ == "__main__":
    unittest.main()
