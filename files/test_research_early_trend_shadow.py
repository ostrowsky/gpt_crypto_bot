from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path


class _FakeModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def predict_proba(self, _features):
        return [[1.0 - self.score, self.score]]


class TestResearchEarlyTrendShadow(unittest.TestCase):
    def _scorer(self, root: Path, score: float = 0.7):
        import research_early_trend_shadow as mod

        model_file = root / "model.cbm"
        metadata_file = root / "model.json"
        model_file.write_bytes(b"fake")
        metadata = {
            "profile": "test_profile",
            "threshold": 0.6,
            "feature_names": list(mod.FEATURE_NAMES),
            "created_at_utc": "2026-08-03T00:00:00Z",
            "holdout_end": "2026-08-02",
            "model_sha256": hashlib.sha256(b"fake").hexdigest(),
        }
        metadata_file.write_text(json.dumps(metadata), encoding="utf-8")
        scorer = mod.ShadowScorer(model_file, metadata_file)
        scorer._model = _FakeModel(score)
        scorer._metadata = metadata
        scorer._model_mtime_ns = model_file.stat().st_mtime_ns
        return scorer

    def test_eligible_row_is_annotated_without_trading_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scorer = self._scorer(Path(tmp))
            record = {
                "tf": "15m",
                "in_trade_watchlist": True,
                "rule_signal": "none",
                "rank_24h": 10,
                "price_change_pct_24h": 1.0,
                "quote_volume_24h": 2_000_000,
                "f": {"rsi": 55, "adx": 20},
            }
            annotation = scorer.annotate(record)

        self.assertTrue(annotation["candidate"])
        self.assertEqual(record["early_trend_shadow"]["profile"], "test_profile")
        self.assertNotIn("action", record["early_trend_shadow"])

    def test_v1_signal_and_outside_watchlist_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scorer = self._scorer(Path(tmp))
            v1 = {"tf": "15m", "in_trade_watchlist": True, "rule_signal": "trend"}
            outside = {"tf": "15m", "in_trade_watchlist": False, "rule_signal": "none"}
            self.assertIsNone(scorer.annotate(v1))
            self.assertIsNone(scorer.annotate(outside))
            self.assertNotIn("early_trend_shadow", v1)
            self.assertNotIn("early_trend_shadow", outside)

    def test_missing_model_does_not_annotate(self) -> None:
        import research_early_trend_shadow as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scorer = mod.ShadowScorer(root / "missing.cbm", root / "missing.json")
            record = {"tf": "15m", "in_trade_watchlist": True, "rule_signal": "none"}
            self.assertIsNone(scorer.annotate(record))
            self.assertEqual(scorer.last_error, "model_missing")

    def test_hash_mismatch_fails_closed(self) -> None:
        import research_early_trend_shadow as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.cbm"
            metadata = root / "model.json"
            model.write_bytes(b"changed")
            metadata.write_text(
                json.dumps(
                    {
                        "threshold": 0.5,
                        "feature_names": list(mod.FEATURE_NAMES),
                        "model_sha256": "wrong",
                    }
                ),
                encoding="utf-8",
            )
            scorer = mod.ShadowScorer(model, metadata)
            record = {"tf": "15m", "in_trade_watchlist": True, "rule_signal": "none"}
            self.assertIsNone(scorer.annotate(record))
            self.assertIn("hash mismatch", scorer.last_error)


if __name__ == "__main__":
    unittest.main()
