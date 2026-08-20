from __future__ import annotations

import unittest
import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from portfolio_alpha import BENCHMARK_NAME, evaluate_portfolio_alpha
from replay_backtest import (
    ReplayRunStats,
    ReplayTrade,
    _make_report,
    _load_cached_klines,
    _parse_end_at,
    parse_args,
    render_text,
    run_replay,
)


DAY_MS = 86_400_000


def _series(prices: list[float], *, step_ms: int = DAY_MS) -> list[tuple[int, float]]:
    return [((idx + 1) * step_ms, price) for idx, price in enumerate(prices)]


def _trade(
    symbol: str,
    *,
    entry_ts: int,
    exit_ts: int,
    entry_price: float,
    exit_price: float,
) -> dict:
    return {
        "sym": symbol,
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }


def _manifest() -> dict:
    return {
        "policy_epoch": "pe-test",
        "policy_hash": "policy-hash",
        "config_hash": "config-hash",
        "watchlist_hash": "watchlist-hash",
    }


class PortfolioAlphaTest(unittest.TestCase):
    def test_replay_legacy_total_is_explicitly_diagnostic(self) -> None:
        trade = ReplayTrade(
            sym="ETHUSDT",
            tf="15m",
            mode="trend",
            entry_ts=DAY_MS,
            entry_price=100.0,
            entry_i=1,
            trail_k=2.0,
            max_hold_bars=10,
            trail_stop=90.0,
            exit_ts=2 * DAY_MS,
            exit_price=110.0,
        )
        report = _make_report(
            start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 1, 2, tzinfo=timezone.utc),
            symbols=["ETHUSDT"],
            timeframes=["15m"],
            trades=[trade],
            run_stats=ReplayRunStats(),
            label="portfolio",
        )

        self.assertEqual(
            report["totals"]["pnl_total_contract"],
            "non_capital_weighted_diagnostic_sum_not_portfolio_return",
        )

    def test_replay_cli_accepts_canonical_alpha_output_and_costs(self) -> None:
        with patch(
            "sys.argv",
            [
                "replay_backtest.py",
                "--portfolio-alpha-output",
                ".runtime/reports/canonical_portfolio_alpha.json",
                "--portfolio-initial-capital",
                "5000",
                "--portfolio-fee-bps",
                "8",
                "--portfolio-slippage-bps",
                "6",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.portfolio_initial_capital, 5000.0)
        self.assertEqual(args.portfolio_fee_bps, 8.0)
        self.assertEqual(args.portfolio_slippage_bps, 6.0)
        self.assertTrue(str(args.portfolio_alpha_output).endswith("canonical_portfolio_alpha.json"))

    def test_replay_accepts_explicit_timezone_aware_end_cutoff(self) -> None:
        parsed = _parse_end_at("2026-08-13T18:22:00Z")
        self.assertEqual(parsed, datetime(2026, 8, 13, 18, 22, tzinfo=timezone.utc))
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            _parse_end_at("2026-08-13T18:22:00")

    def test_incomplete_portfolio_alpha_renders_without_false_numeric_claim(self) -> None:
        report = {
            "window_start": "x",
            "window_end": "y",
            "symbols": [],
            "timeframes": ["15m"],
            "max_open_positions": 10,
            "replace_min_delta": 8.0,
            "reports": {"portfolio": {
                "trades_total": 0,
                "totals": {"pnl_total": 0.0, "pnl_avg": 0.0, "win_rate": 0.0},
                "run_stats": {
                    "candidates_total": 0,
                    "skipped_portfolio_full": 0,
                    "replacements_total": 0,
                },
                "objective": {
                    "symbol_precision": 0.0,
                    "trade_precision": 0.0,
                    "capture_rate": 0.0,
                    "captured_top_symbols": [],
                    "final_top_n": 0,
                },
                "summary": {},
                "suggestions": [],
                "portfolio_alpha": {
                    "portfolio": {"net_return_after_costs_pct": 0.0},
                    "benchmark": {"status": "insufficient_data"},
                    "net_alpha_after_costs": None,
                    "evidence_grade": "diagnostic",
                },
            }},
        }
        text = render_text(report)
        self.assertIn("alpha=n/a", text)

    def test_local_cache_loader_merges_deduplicates_and_clips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = [
                {"t": 0, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
                {"t": 900_000, "o": 1.5, "h": 2, "l": 1, "c": 1.7, "v": 11},
            ]
            second = [
                {"t": 900_000, "o": 1.5, "h": 2.1, "l": 1, "c": 1.8, "v": 12},
                {"t": 1_800_000, "o": 1.8, "h": 2.2, "l": 1.7, "c": 2, "v": 13},
            ]
            (root / "AAAUSDT_15m_0_1800000.json").write_text(json.dumps(first), encoding="utf-8")
            (root / "AAAUSDT_15m_900000_2700000.json").write_text(json.dumps(second), encoding="utf-8")
            rows = _load_cached_klines(root, "AAAUSDT", "15m", 900_000, 2_700_000)

        self.assertEqual(rows["t"].tolist(), [900_000, 1_800_000])
        self.assertEqual(rows["c"].tolist(), [1.8, 2.0])

    def test_paired_replay_comparison_uses_capital_weighted_alpha(self) -> None:
        import numpy as np

        data = np.zeros(
            61,
            dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")],
        )
        data["o"] = data["h"] = data["l"] = data["c"] = 100.0
        data["v"] = 1.0

        async def fake_fetch(_session, _sym, _tf, start_ms, end_ms):
            row = data.copy()
            step = max(1, (end_ms - start_ms) // len(row))
            row["t"] = start_ms + np.arange(len(row), dtype=np.int64) * step
            return row

        async def fake_simulate(_symbols, _timeframes, cache, *_args, **kwargs):
            btc = cache[("BTCUSDT", "15m")][0]
            row = ReplayTrade(
                sym="BTCUSDT", tf="15m", mode="trend",
                entry_ts=int(btc["t"][0]) + 900_000,
                entry_price=100.0, entry_i=0, trail_k=2.0, max_hold_bars=10,
                trail_stop=90.0,
                exit_ts=int(btc["t"][-1]) + 900_000,
                exit_price=110.0 if kwargs.get("variant") == "score_replace" else 100.0,
            )
            return [row], ReplayRunStats()

        with patch("replay_backtest.fetch_klines", side_effect=fake_fetch), patch(
            "replay_backtest.compute_features", return_value={}
        ), patch("replay_backtest.simulate_portfolio", side_effect=fake_simulate), patch(
            "replay_backtest._build_bull_day_context", return_value=None
        ), patch("replay_backtest._final_top_symbols", return_value=set()):
            report = asyncio.run(
                run_replay(
                    ["BTCUSDT"], 30, ["15m"], max_open_positions=10,
                    compare_baseline=False, replace_min_delta=8.0,
                    variant="score_replace", compare_variant="agent_allowed",
                    portfolio_fee_bps=7.5, portfolio_slippage_bps=5.0,
                )
            )

        self.assertIn("portfolio_net_return_after_costs_delta", report["comparison"])
        self.assertGreater(report["comparison"]["portfolio_net_return_after_costs_delta"], 0.0)
        self.assertEqual(
            report["comparison"]["canonical_profitability_contract"],
            "capital_weighted_ten_slot_after_costs_same_btc_benchmark",
        )

    def test_single_trade_is_capital_weighted_to_one_of_ten_slots(self) -> None:
        benchmark = _series([100.0] * 31)
        trade = _trade(
            "ETHUSDT",
            entry_ts=DAY_MS,
            exit_ts=2 * DAY_MS,
            entry_price=100.0,
            exit_price=110.0,
        )
        payload = evaluate_portfolio_alpha(
            [trade],
            price_series_by_symbol={"ETHUSDT": _series([100.0, 110.0] + [110.0] * 29)},
            benchmark_series=benchmark,
            window_start_ms=0,
            window_end_ms=31 * DAY_MS,
            requested_days=30,
            universe=["ETHUSDT"],
            variant="test",
            fee_bps=0.0,
            slippage_bps=0.0,
            policy_manifest=_manifest(),
        )

        self.assertAlmostEqual(payload["portfolio"]["net_return_after_costs_pct"], 1.0, places=6)
        self.assertAlmostEqual(payload["net_alpha_after_costs"], 1.0, places=6)
        self.assertFalse(payload["decision_grade"], "zero costs must keep the result diagnostic")

    def test_costs_apply_on_both_sides_to_portfolio_and_benchmark(self) -> None:
        benchmark = _series([100.0] * 31)
        trade = _trade(
            "BTCUSDT",
            entry_ts=DAY_MS,
            exit_ts=31 * DAY_MS,
            entry_price=100.0,
            exit_price=100.0,
        )
        payload = evaluate_portfolio_alpha(
            [trade],
            price_series_by_symbol={"BTCUSDT": benchmark},
            benchmark_series=benchmark,
            window_start_ms=0,
            window_end_ms=31 * DAY_MS,
            requested_days=30,
            universe=["BTCUSDT"],
            variant="test",
            fee_bps=10.0,
            slippage_bps=10.0,
            policy_manifest=_manifest(),
        )

        portfolio_return = payload["portfolio"]["net_return_after_costs_pct"]
        benchmark_return = payload["benchmark"]["net_return_after_costs_pct"]
        self.assertLess(portfolio_return, 0.0)
        self.assertLess(benchmark_return, 0.0)
        self.assertGreater(payload["portfolio"]["fees_quote"], 0.0)
        self.assertGreater(payload["portfolio"]["slippage_quote"], 0.0)
        self.assertEqual(payload["benchmark"]["name"], BENCHMARK_NAME)

    def test_complete_max_period_ten_slot_contract_is_decision_grade(self) -> None:
        benchmark = _series([100.0 + idx for idx in range(31)])
        trade = _trade(
            "ETHUSDT",
            entry_ts=DAY_MS,
            exit_ts=31 * DAY_MS,
            entry_price=50.0,
            exit_price=60.0,
        )
        payload = evaluate_portfolio_alpha(
            [trade],
            price_series_by_symbol={"ETHUSDT": _series([50.0 + idx / 3.0 for idx in range(31)])},
            benchmark_series=benchmark,
            window_start_ms=0,
            window_end_ms=31 * DAY_MS,
            requested_days=30,
            universe=["BTCUSDT", "ETHUSDT"],
            variant="test",
            capacity=10,
            fee_bps=7.5,
            slippage_bps=5.0,
            policy_manifest=_manifest(),
        )

        self.assertTrue(payload["decision_grade"])
        self.assertEqual(payload["evidence_grade"], "decision_grade")
        expected = (
            payload["portfolio"]["net_return_after_costs_pct"]
            - payload["benchmark"]["net_return_after_costs_pct"]
        )
        self.assertAlmostEqual(payload["net_alpha_after_costs"], expected, places=6)

    def test_duplicate_symbol_overlap_fails_closed(self) -> None:
        benchmark = _series([100.0] * 31)
        trades = [
            _trade("ETHUSDT", entry_ts=DAY_MS, exit_ts=3 * DAY_MS, entry_price=10.0, exit_price=11.0),
            _trade("ETHUSDT", entry_ts=2 * DAY_MS, exit_ts=4 * DAY_MS, entry_price=10.0, exit_price=12.0),
        ]
        payload = evaluate_portfolio_alpha(
            trades,
            price_series_by_symbol={"ETHUSDT": _series([10.0] * 31)},
            benchmark_series=benchmark,
            window_start_ms=0,
            window_end_ms=31 * DAY_MS,
            requested_days=30,
            universe=["ETHUSDT"],
            variant="test",
            policy_manifest=_manifest(),
        )

        self.assertFalse(payload["decision_grade"])
        self.assertTrue(any(row.startswith("duplicate_symbol:ETHUSDT") for row in payload["coverage"]["contract_violations"]))

    def test_partial_exit_is_booked_once_before_final_exit(self) -> None:
        benchmark = _series([100.0] * 31)
        trade = {
            **_trade("ETHUSDT", entry_ts=DAY_MS, exit_ts=3 * DAY_MS, entry_price=100.0, exit_price=100.0),
            "partial_exit_taken": True,
            "partial_exit_fraction": 0.5,
            "partial_exit_ts": 2 * DAY_MS,
            "partial_exit_price": 120.0,
        }
        payload = evaluate_portfolio_alpha(
            [trade],
            price_series_by_symbol={"ETHUSDT": _series([100.0, 120.0, 100.0] + [100.0] * 28)},
            benchmark_series=benchmark,
            window_start_ms=0,
            window_end_ms=31 * DAY_MS,
            requested_days=30,
            universe=["ETHUSDT"],
            variant="test",
            fee_bps=0.0,
            slippage_bps=0.0,
            policy_manifest=_manifest(),
        )

        # One 10% slot realizes +20% on half its size: +1% portfolio return.
        self.assertAlmostEqual(payload["portfolio"]["net_return_after_costs_pct"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
