from __future__ import annotations

import unittest
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo
from unittest.mock import patch

import numpy as np

import replay_backtest as rb


class ObjectiveCandidateRankingReplayTest(unittest.TestCase):
    def _candidate(self, **overrides) -> rb.ReplayCandidate:
        values = {
            "sym": "AAAUSDT", "tf": "15m", "mode": "strong_trend",
            "ts_ms": 1_000_000, "i": 0, "price": 100.0, "trail_k": 2.0,
            "max_hold_bars": 10, "score": 100.0,
            "ranker_final_score": 0.0, "ranker_top_gainer_prob": 0.0,
            "top_gainer_score": 50.0, "rsi": 65.0, "daily_range": 5.0,
            "vol_x": 1.5, "adx": 25.0, "intraday_change_pct": 2.0,
        }
        values.update(overrides)
        return rb.ReplayCandidate(**values)

    def test_structural_rank_does_not_use_unverified_ranker_values(self) -> None:
        plain = self._candidate()
        legacy_boosted = self._candidate(ranker_final_score=5.0, ranker_top_gainer_prob=1.0, top_gainer_score=80.0)
        self.assertEqual(
            rb._candidate_allocation_score(plain, "objective_rank_structural"),
            rb._candidate_allocation_score(legacy_boosted, "objective_rank_structural"),
        )

    def test_extension_efficiency_prefers_strength_before_extension(self) -> None:
        early = self._candidate(adx=35.0, vol_x=2.5, rsi=66.0, daily_range=5.0)
        late = self._candidate(adx=25.0, vol_x=1.3, rsi=76.0, daily_range=12.0)
        self.assertGreater(
            rb._candidate_allocation_score(early, "objective_rank_extension_efficiency"),
            rb._candidate_allocation_score(late, "objective_rank_extension_efficiency"),
        )

    def test_capacity_regret_reports_mature_denominators_and_objective_miss(self) -> None:
        tz = ZoneInfo(rb.OBJECTIVE_TZ)
        local_day = date(2026, 8, 18)
        start = datetime.combine(local_day, time(10, 0), tzinfo=tz).astimezone(timezone.utc)
        ts_ms = int(start.timestamp() * 1000)

        def series(start_price: float, end_price: float) -> np.ndarray:
            rows = np.zeros(7, dtype=rb._KLINE_DTYPE)
            rows["t"] = [ts_ms - rb.BAR_MS["15m"] + i * rb.BAR_MS["15m"] for i in range(7)]
            rows["o"] = rows["h"] = rows["l"] = rows["c"] = np.linspace(start_price, end_price, 7)
            rows["v"] = 1.0
            return rows

        events = [{
            "ts_ms": ts_ms,
            "candidate_sym": "AAAUSDT",
            "incumbent_sym": "BBBUSDT",
            "candidate_score": 60.0,
            "incumbent_score": 50.0,
        }, {
            "ts_ms": ts_ms,
            "candidate_sym": "AAAUSDT",
            "incumbent_sym": "BBBUSDT",
            "candidate_score": 60.0,
            "incumbent_score": 50.0,
        }]
        report = rb._capacity_regret_report(
            events,
            cache_15m={
                "AAAUSDT": (series(100.0, 110.0), {}),
                "BBBUSDT": (series(100.0, 101.0), {}),
            },
            daily_objective={
                "eligible_days": [local_day.isoformat()],
                "label_pairs": {(local_day.isoformat(), "AAAUSDT")},
            },
        )
        self.assertEqual(report["event_count"], 2)
        self.assertEqual(report["mature_forward_5_count"], 2)
        self.assertEqual(report["candidate_win_count"], 2)
        self.assertEqual(report["missed_objective_event_count"], 2)
        self.assertEqual(report["missed_objective_pair_count"], 1)
        self.assertGreater(report["candidate_minus_incumbent_ret5"]["avg"], 0.0)

    def test_cli_accepts_research_only_rank_variants(self) -> None:
        for variant in sorted(rb.OBJECTIVE_RANK_VARIANTS):
            with self.subTest(variant=variant), patch(
                "sys.argv", ["replay_backtest.py", "--variant", variant]
            ):
                self.assertEqual(rb.parse_args().variant, variant)


if __name__ == "__main__":
    unittest.main()
