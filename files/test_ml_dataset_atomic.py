from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ml_dataset


class TestMLDatasetAtomicRewrite(unittest.TestCase):
    def test_label_rewrite_streams_and_preserves_other_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset = Path(td) / "ml_dataset.jsonl"
            rows = [
                {"id": "a", "labels": {"ret_5": None}},
                {"id": "b", "labels": {"ret_5": None}},
            ]
            dataset.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            original = ml_dataset.ML_FILE
            try:
                ml_dataset.ML_FILE = dataset
                with patch.object(Path, "read_text", side_effect=AssertionError("whole-file read")):
                    ml_dataset.fill_learning_labels("a", {"ret_5": 1.25, "label_5": True})
            finally:
                ml_dataset.ML_FILE = original

            with dataset.open(encoding="utf-8") as source:
                persisted = [json.loads(line) for line in source if line.strip()]
            temp_files = list(dataset.parent.glob("ml_dataset.jsonl.*.tmp"))

        self.assertEqual(persisted[0]["labels"]["ret_5"], 1.25)
        self.assertTrue(persisted[0]["labels"]["label_5"])
        self.assertIsNone(persisted[1]["labels"]["ret_5"])
        self.assertFalse(temp_files)

    def test_append_and_rewrite_share_the_dataset_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset = Path(td) / "ml_dataset.jsonl"
            original = ml_dataset.ML_FILE
            try:
                ml_dataset.ML_FILE = dataset
                ml_dataset._w({"id": "a", "labels": {}})
                ml_dataset.fill_forward_label("a", 3, 0.5)
            finally:
                ml_dataset.ML_FILE = original

            persisted = json.loads(dataset.read_text(encoding="utf-8"))

        self.assertEqual(persisted["labels"]["ret_3"], 0.5)
        self.assertTrue(persisted["labels"]["label_3"])


if __name__ == "__main__":
    unittest.main()
