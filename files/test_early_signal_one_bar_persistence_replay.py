from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit_early_signal_one_bar_persistence import (
    _align_research_pair,
    _load_candidates_and_entries,
    _load_final_top_labels,
    _split_by_day_with_purge,
    _variant_matches,
)


def _row(ts: int, *, slope: float = 0.2, macd: float = 0.1, rank: int = 10,
         vol_x: float = 1.0, body: float = 0.3, wick: float = 0.1,
         rsi: float = 60.0) -> dict:
    return {
        "bar_ts": ts,
        "rank_24h": rank,
        "f": {
            "slope": slope,
            "macd_hist_norm": macd,
            "vol_x": vol_x,
            "body_pct": body,
            "upper_wick_pct": wick,
            "rsi": rsi,
        },
    }


class EarlySignalOneBarPersistenceReplayTest(unittest.TestCase):
    def test_candidate_join_deduplicates_delivery_retries_and_finds_later_buy(self) -> None:
        events = [
            {
                "event": "blocked_learning_label",
                "sym": "AUSDT",
                "tf": "15m",
                "mode": "trend",
                "reason_code": "top_gainer_score_gate",
                "live_score": 33.0,
                "ts": "2026-08-01T12:00:00Z",
            },
            {
                "event": "telegram_delivery",
                "delivery_stage": "ok",
                "text_preview": "entry blocked by score gate",
                "sym": "AUSDT",
                "tf": "15m",
                "ts": "2026-08-01T12:00:05Z",
            },
            {
                "event": "telegram_delivery",
                "delivery_stage": "ok",
                "text_preview": "entry blocked by score gate retry",
                "sym": "AUSDT",
                "tf": "15m",
                "ts": "2026-08-01T12:00:10Z",
            },
            {
                "event": "entry",
                "sym": "AUSDT",
                "tf": "15m",
                "mode": "trend",
                "ts": "2026-08-01T12:30:00Z",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            candidates, coverage = _load_candidates_and_entries(path, {"2026-08-01"})
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["later_buy_after_alert"])
        self.assertEqual(coverage["eligible_deliveries"], 2)
        self.assertEqual(coverage["duplicate_deliveries_removed"], 1)

    def test_alignment_uses_latest_available_row_and_exact_next_bar(self) -> None:
        rows = [_row(1_000_000), _row(1_900_000), _row(2_800_000)]
        base, confirm = _align_research_pair(rows, 2_000_000)
        self.assertEqual(base["bar_ts"], 1_000_000)
        self.assertEqual(confirm["bar_ts"], 1_900_000)

    def test_alignment_rejects_stale_or_missing_next_bar(self) -> None:
        rows = [_row(1_000_000), _row(3_000_000)]
        self.assertIsNone(_align_research_pair(rows, 2_000_000))
        exact_pair = [_row(1_000_000), _row(1_900_000)]
        self.assertIsNone(_align_research_pair(exact_pair, 2_000_000, max_age_ms=50_000))

    def test_frozen_variants_apply_incremental_guards(self) -> None:
        base = _row(1_000_000, slope=0.20, macd=0.10, rank=10)
        confirm = _row(1_900_000, slope=0.17, macd=0.09, rank=9, vol_x=0.9, body=0.3, wick=0.2, rsi=70)
        self.assertTrue(_variant_matches("persistence_structure", base, confirm))
        self.assertTrue(_variant_matches("persistence_rank", base, confirm))
        self.assertTrue(_variant_matches("persistence_quality", base, confirm))

        worse_rank = _row(1_900_000, slope=0.17, macd=0.09, rank=11)
        self.assertTrue(_variant_matches("persistence_structure", base, worse_rank))
        self.assertFalse(_variant_matches("persistence_rank", base, worse_rank))

        weak_quality = _row(1_900_000, slope=0.17, macd=0.09, rank=9, vol_x=0.7)
        self.assertTrue(_variant_matches("persistence_rank", base, weak_quality))
        self.assertFalse(_variant_matches("persistence_quality", base, weak_quality))

    def test_temporal_split_purges_boundary_days(self) -> None:
        all_days = [f"2026-08-{day:02d}" for day in range(1, 16)]
        rows = [
            {"local_day": f"2026-08-{day:02d}", "id": day}
            for day in range(1, 16)
            if day != 5
        ]
        train, validation, holdout, meta = _split_by_day_with_purge(
            rows, all_days=all_days
        )
        self.assertEqual([row["id"] for row in train], [1, 2, 3, 4, 6, 7, 8])
        self.assertEqual([row["id"] for row in validation], [11])
        self.assertEqual([row["id"] for row in holdout], [14, 15])
        self.assertEqual(
            meta["purged_days"],
            ["2026-08-09", "2026-08-10", "2026-08-12", "2026-08-13"],
        )

    def test_final_top_labels_use_exchange_rows_filtered_to_watchlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "top_gainer_critic_2026-08-01_final.json"
            path.write_text(
                json.dumps(
                    {
                        "target_day_local": "2026-08-01",
                        "exchange_top_gainers": [
                            {"symbol": "AUSDT", "in_watchlist": True},
                            {"symbol": "BUSDT", "in_watchlist": False},
                        ],
                        "watchlist_top_gainers": [{"symbol": "WRONGUSDT"}],
                    }
                ),
                encoding="utf-8",
            )
            days, positives = _load_final_top_labels(Path(tmp))
        self.assertEqual(days, {"2026-08-01"})
        self.assertEqual(positives, {("2026-08-01", "AUSDT")})


if __name__ == "__main__":
    unittest.main()
