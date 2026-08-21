from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import validate_external_top50_screen as screen


HOUR_MS = 3_600_000


def _bar_ending(close_dt: datetime, close: float, quote_volume: float = 100.0, buy_ratio: float = 0.5) -> screen.Bar:
    close_ms = int(close_dt.timestamp() * 1000)
    return screen.Bar(
        open_ts_ms=close_ms - HOUR_MS,
        close=close,
        quote_volume=quote_volume,
        taker_buy_quote=quote_volume * buy_ratio,
    )


def _history(
    *,
    observation_price: float,
    target_price: float,
    observation_base: float = 100.0,
    target_base: float = 100.0,
) -> tuple[screen.Bar, ...]:
    rows: list[screen.Bar] = []
    start = datetime(2026, 8, 19, 6, tzinfo=timezone.utc)
    end = datetime(2026, 8, 20, 23, tzinfo=timezone.utc)
    cursor = start
    while cursor <= end:
        rows.append(_bar_ending(cursor, 100.0))
        cursor += timedelta(hours=1)

    values = {row.open_ts_ms: row for row in rows}

    def replace(close_dt: datetime, close: float, quote_volume: float = 100.0) -> None:
        row = _bar_ending(close_dt, close, quote_volume)
        values[row.open_ts_ms] = row

    replace(datetime(2026, 8, 19, 12, tzinfo=timezone.utc), observation_base)
    replace(datetime(2026, 8, 19, 23, tzinfo=timezone.utc), target_base)
    replace(datetime(2026, 8, 20, 9, tzinfo=timezone.utc), observation_price * 0.98, 80.0)
    replace(datetime(2026, 8, 20, 10, tzinfo=timezone.utc), observation_price * 0.99, 90.0)
    replace(datetime(2026, 8, 20, 11, tzinfo=timezone.utc), observation_price * 0.995, 100.0)
    replace(datetime(2026, 8, 20, 12, tzinfo=timezone.utc), observation_price, 200.0)
    replace(datetime(2026, 8, 20, 23, tzinfo=timezone.utc), target_price)
    return tuple(sorted(values.values(), key=lambda row: row.open_ts_ms))


class ExternalTop50ScreenValidationTests(unittest.TestCase):
    def test_last_visible_bar_uses_only_closed_data(self) -> None:
        decision = datetime(2026, 8, 20, 12, 15, tzinfo=timezone.utc)
        history = (
            _bar_ending(datetime(2026, 8, 20, 12, tzinfo=timezone.utc), 101.0),
            _bar_ending(datetime(2026, 8, 20, 13, tzinfo=timezone.utc), 999.0),
        )
        visible = screen.last_bar_at_or_before(history, int(decision.timestamp() * 1000))
        self.assertIsNotNone(visible)
        self.assertEqual(visible.close, 101.0)

    def test_screen_formula_is_frozen(self) -> None:
        self.assertAlmostEqual(
            screen.score_screen_v1(
                static_target_return=10.0,
                ret_1h=2.0,
                ret_3h=4.0,
                volume_accel=2.0,
            ),
            10.95,
        )
        self.assertAlmostEqual(
            screen.score_screen_v1(0.0, 0.0, 0.0, 100.0),
            1.0,
        )

    def test_forward_target_changes_label_but_not_candidate_score(self) -> None:
        histories = {
            "MKTUSDT": _history(observation_price=120.0, target_price=121.0),
            "AAAUSDT": _history(observation_price=110.0, target_price=130.0),
            "BBBUSDT": _history(observation_price=105.0, target_price=106.0),
        }
        first = screen.build_day_snapshot(
            histories,
            watchlist={"AAAUSDT", "BBBUSDT"},
            local_day=date(2026, 8, 20),
            timezone_name="UTC",
            top_n=1,
            min_market_symbols=3,
            min_watchlist_symbols=2,
        )
        self.assertIsNotNone(first)
        self.assertEqual([candidate.symbol for candidate in first.candidates], ["AAAUSDT", "BBBUSDT"])
        self.assertIn("AAAUSDT", first.target_top_symbols)

        changed = dict(histories)
        changed["AAAUSDT"] = _history(observation_price=110.0, target_price=90.0)
        second = screen.build_day_snapshot(
            changed,
            watchlist={"AAAUSDT", "BBBUSDT"},
            local_day=date(2026, 8, 20),
            timezone_name="UTC",
            top_n=1,
            min_market_symbols=3,
            min_watchlist_symbols=2,
        )
        self.assertIsNotNone(second)
        self.assertEqual(
            [(item.symbol, item.score_screen_v1) for item in first.candidates],
            [(item.symbol, item.score_screen_v1) for item in second.candidates],
        )
        self.assertNotIn("AAAUSDT", second.target_top_symbols)

    def test_metrics_disclose_denominators_base_rate_and_lift(self) -> None:
        histories = {
            "MKTUSDT": _history(observation_price=120.0, target_price=121.0),
            "AAAUSDT": _history(observation_price=110.0, target_price=130.0),
            "BBBUSDT": _history(observation_price=105.0, target_price=106.0),
        }
        snapshot = screen.build_day_snapshot(
            histories,
            watchlist={"AAAUSDT", "BBBUSDT"},
            local_day=date(2026, 8, 20),
            timezone_name="UTC",
            top_n=1,
            min_market_symbols=3,
            min_watchlist_symbols=2,
        )
        report = screen.evaluate_snapshots([snapshot], selection_size=1, bootstrap_samples=200)
        metrics = report["screen_v1"]
        self.assertEqual(metrics["top1"], {"hits": 1, "days": 1, "rate": 1.0, "wilson95": metrics["top1"]["wilson95"]})
        self.assertEqual(metrics["topk"]["hits"], 1)
        self.assertEqual(metrics["topk"]["selections"], 1)
        self.assertEqual(metrics["target_entrant_recall"]["hits"], 1)
        self.assertEqual(metrics["target_entrant_recall"]["entrants"], 1)
        self.assertEqual(metrics["candidate_base_rate"]["hits"], 1)
        self.assertEqual(metrics["candidate_base_rate"]["candidates"], 2)
        self.assertEqual(metrics["precision_lift_over_base"], 2.0)

    def test_merge_binance_rows_deduplicates_open_time(self) -> None:
        first = [[1, "1", "1", "1", "10", "2", 2, "20", 1, "1", "10", "0"]]
        second = [
            [1, "1", "1", "1", "11", "3", 2, "30", 1, "1", "15", "0"],
            [HOUR_MS + 1, "1", "1", "1", "12", "4", HOUR_MS + 2, "40", 1, "1", "20", "0"],
        ]
        bars = screen.merge_binance_rows([first, second])
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0].close, 11.0)
        self.assertEqual(bars[1].quote_volume, 40.0)

    def test_cache_discovery_keeps_one_widest_snapshot_per_source(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy"
            tail = root / "tail"
            legacy.mkdir()
            tail.mkdir()
            for name in (
                "AAAUSDT_1h_0_100.json",
                "AAAUSDT_1h_10_110.json",
                "AAAUSDT_1h_20_50.json",
            ):
                (legacy / name).write_text("[]", encoding="utf-8")
            (tail / "AAAUSDT_1h_100_200.json").write_text("[]", encoding="utf-8")
            selected = screen.discover_cache_files((legacy, tail))
            self.assertEqual(
                [item.path.name for item in selected["AAAUSDT"]],
                ["AAAUSDT_1h_10_110.json", "AAAUSDT_1h_100_200.json"],
            )


if __name__ == "__main__":
    unittest.main()
