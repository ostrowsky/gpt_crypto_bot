from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


SCRIPT = Path(__file__).resolve().parent.parent / "skills" / "signal-quality-evaluator" / "scripts" / "evaluate_signals.py"
SPEC = importlib.util.spec_from_file_location("signal_quality_evaluator_script", SCRIPT)
assert SPEC and SPEC.loader
EVALUATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVALUATOR
SPEC.loader.exec_module(EVALUATOR)


class SignalQualityEvaluatorCacheTest(unittest.TestCase):
    def test_empty_cache_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            cache = EVALUATOR._cache_path(cache_dir, "AAAUSDT", "15m", 0, 900_000)
            cache.write_text("[]", encoding="utf-8")
            response = [[0, "1", "2", "0.5", "1.5", "10"]]
            fake = MagicMock()
            fake.__enter__.return_value.read.return_value = json.dumps(response).encode("utf-8")

            with patch.object(EVALUATOR.urllib.request, "urlopen", return_value=fake) as urlopen:
                rows = EVALUATOR._fetch_klines("AAAUSDT", "15m", 0, 900_000, cache_dir)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["c"], 1.5)
            urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
