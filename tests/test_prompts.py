from __future__ import annotations

import csv
import gzip
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prompts = _rows(ROOT / "prompts" / "prompts.csv.gz")
        cls.keys = {(row["context_id"], row["regime"]) for row in cls.prompts}

    def test_every_context_has_all_sixteen_prompts(self) -> None:
        self.assertEqual(len({row["regime"] for row in self.prompts}), 16)
        self.assertEqual(len({row["context_id"] for row in self.prompts}), 1248)
        self.assertEqual(len(self.keys), len(self.prompts))
        self.assertEqual(len(self.prompts), 1248 * 16)

    def test_hashes_match_the_prompt_text(self) -> None:
        for row in self.prompts:
            digest = hashlib.sha256(row["prompt_text"].encode("utf-8")).hexdigest()
            self.assertEqual(digest, row["prompt_sha256"])

    def test_every_model_response_has_its_prompt(self) -> None:
        paths = [
            *sorted((ROOT / "data" / "models").glob("*.csv.gz")),
            *sorted((ROOT / "data" / "belief_repetitions").glob("*/*/results.csv.gz")),
        ]
        self.assertTrue(paths)
        for path in paths:
            missing = {(row["context_id"], row["regime"]) for row in _rows(path)} - self.keys
            self.assertFalse(missing, f"{path}: {sorted(missing)[:3]}")


if __name__ == "__main__":
    unittest.main()
