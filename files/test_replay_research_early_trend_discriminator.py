from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestResearchEarlyTrendDiscriminator(unittest.TestCase):
    def test_load_rows_filters_and_uses_budapest_local_day(self) -> None:
        import replay_research_early_trend_discriminator as mod

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            good = {
                "sym": "AAAUSDT",
                "tf": "15m",
                "bar_ts": 1780353000000,
                "in_trade_watchlist": True,
                "rule_signal": "trend",
                "rank_24h": 5,
                "quote_volume_24h": 1_000_000,
                "f": {"rsi": 55, "adx": 20},
                "labels": {"ret_3": 0.2, "ret_5": 0.6, "ret_10": 1.2},
            }
            immature = {**good, "sym": "BBBUSDT", "labels": {"ret_3": None}}
            path.write_text(
                "bad-json\n" + json.dumps(good) + "\n" + json.dumps(immature),
                encoding="utf-8",
            )
            rows, stats = mod.load_rows(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].local_day, "2026-06-02")
        self.assertEqual(len(rows[0].features), len(mod.FEATURE_NAMES))
        self.assertEqual(stats["malformed"], 1)
        self.assertEqual(stats["immature"], 1)

    def test_chronological_split_has_four_embargo_days(self) -> None:
        import replay_research_early_trend_discriminator as mod

        days = [f"2026-06-{day:02d}" for day in range(1, 21)]
        split = mod.chronological_day_splits(days)
        used = set(split["train"]) | set(split["validation"]) | set(split["holdout"])
        self.assertEqual(len(split["embargo"]), 4)
        self.assertFalse(used & set(split["embargo"]))
        self.assertEqual(len(used) + 4, len(days))

    def test_first_signal_counts_only_first_symbol_day_event(self) -> None:
        import replay_research_early_trend_discriminator as mod

        mask = np.asarray([False, True, True, True, False, True])
        groups = np.asarray([0, 0, 0, 1, 1, 1])
        first = mod.first_signal_indices(mask, groups, 2)
        self.assertEqual(first.tolist(), [1, 3])

    def test_top_movers_use_canonical_exchange_denominator(self) -> None:
        import replay_research_early_trend_discriminator as mod

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            payload = {
                "exchange_top_gainers": [
                    {"symbol": "AAAUSDT", "in_watchlist": True},
                    {"symbol": "OUTUSDT", "in_watchlist": False},
                ],
                "watchlist_top_gainers": [
                    {"symbol": "WRONGUSDT", "in_watchlist": True}
                ],
            }
            (reports / "top_gainer_critic_2026-06-01_final.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            keys, loaded = mod.load_canonical_top_movers(reports, ["2026-06-01"])

        self.assertEqual(keys, {("2026-06-01", "AAAUSDT")})
        self.assertEqual(loaded, ["2026-06-01"])

    def test_holdout_gate_requires_precision_and_strict_stability(self) -> None:
        import replay_research_early_trend_discriminator as mod

        baseline = {
            "primary": {
                "selected": 100,
                "precision_pct": 10.0,
                "recall_pct": 10.0,
                "avg_ret_10_pct": 0.1,
            },
            "strict": {"precision_pct": 5.0, "recall_pct": 8.0},
        }
        passing = {
            "primary": {
                "selected": 120,
                "precision_pct": 11.0,
                "recall_pct": 13.0,
                "avg_ret_10_pct": 0.2,
            },
            "strict": {"precision_pct": 5.1, "recall_pct": 9.0},
        }
        failing = {
            **passing,
            "strict": {"precision_pct": 4.9, "recall_pct": 9.0},
        }
        self.assertTrue(mod._passes_holdout(passing, baseline))
        self.assertFalse(mod._passes_holdout(failing, baseline))


if __name__ == "__main__":
    unittest.main()
