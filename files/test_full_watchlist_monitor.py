from __future__ import annotations

import unittest

from monitor import _build_full_watchlist_reports, _build_shortlist_reports, _select_poll_coins
from strategy import CoinReport


def _report(symbol: str, *, note: str = "") -> CoinReport:
    return CoinReport(
        symbol=symbol,
        tf="15m",
        today_signals=0,
        today_accuracy={},
        today_confirmed=False,
        best_horizon=0,
        best_accuracy=0.0,
        in_play=False,
        note=note,
    )


class FullWatchlistMonitorTest(unittest.TestCase):
    def test_rollback_shortlist_keeps_confirmed_and_active_only(self) -> None:
        confirmed = _report("BTCUSDT")
        active = _report("POLUSDT")
        active.signal_now = True
        quiet = _report("ETHUSDT")

        rows = _build_shortlist_reports([confirmed], [active, quiet])

        self.assertEqual([row.symbol for row in rows], ["BTCUSDT", "POLUSDT"])

    def test_builder_contains_each_watchlist_symbol_once(self) -> None:
        newest = _report("BTCUSDT", note="new")
        old_eth = _report("ETHUSDT", note="old")

        rows = _build_full_watchlist_reports(
            ["BTCUSDT", "ETHUSDT", "POLUSDT", "BTCUSDT"],
            [newest],
            [],
            [old_eth],
        )

        self.assertEqual([row.symbol for row in rows], ["BTCUSDT", "ETHUSDT", "POLUSDT"])
        self.assertIs(rows[0], newest)
        self.assertIs(rows[1], old_eth)
        self.assertIn("awaiting analysis", rows[2].note)

    def test_rotation_covers_all_symbols_without_exceeding_cap(self) -> None:
        coins = [_report(f"S{i}USDT") for i in range(11)]
        cursor = 0
        seen: set[str] = set()

        for _ in range(3):
            selected, cursor = _select_poll_coins(coins, set(), 4, cursor)
            self.assertLessEqual(len(selected), 4)
            seen.update(row.symbol for row in selected)

        self.assertEqual(seen, {row.symbol for row in coins})

    def test_open_positions_are_polled_every_cycle(self) -> None:
        coins = [_report(f"S{i}USDT") for i in range(10)]
        held = {"S1USDT", "S8USDT"}
        cursor = 0

        for _ in range(5):
            selected, cursor = _select_poll_coins(coins, held, 4, cursor)
            symbols = {row.symbol for row in selected}
            self.assertTrue(held.issubset(symbols))
            self.assertLessEqual(len(selected), 4)

    def test_duplicate_reports_do_not_consume_poll_budget(self) -> None:
        coins = [_report("BTCUSDT"), _report("BTCUSDT"), _report("ETHUSDT")]

        selected, _ = _select_poll_coins(coins, set(), 10, 0)

        self.assertEqual([row.symbol for row in selected], ["BTCUSDT", "ETHUSDT"])

    def test_positions_over_cap_are_never_dropped(self) -> None:
        coins = [_report(f"S{i}USDT") for i in range(7)]
        held = {row.symbol for row in coins[:5]}

        selected, _ = _select_poll_coins(coins, held, 3, 0)

        self.assertEqual({row.symbol for row in selected}, held)


if __name__ == "__main__":
    unittest.main()
