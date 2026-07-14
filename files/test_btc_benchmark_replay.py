from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from audit_btc_benchmark_replay import _decision, _period_metrics, _policy_admissions
from replay_backtest import (
    ReplayCandidate,
    ReplayTrade,
    _btc_1h_leader_admission_replay_ok,
    _btc_cluster_exempt_replay,
    _build_bull_day_context,
    _find_last_closed_candle_index,
    _discovery_catchup_replay_candidate,
    _top_gainer_replay_score,
    _btc_benchmark_rotation_replay_ok,
)
from monitor import _top_gainer_live_score


def _candidate(**overrides) -> ReplayCandidate:
    values = {
        "sym": "BTCUSDT",
        "tf": "1h",
        "mode": "breakout",
        "ts_ms": 1_000,
        "i": 50,
        "price": 63_840.01,
        "trail_k": 1.5,
        "max_hold_bars": 6,
        "score": 142.7988,
        "top_gainer_score": 33.0882,
        "rsi": 68.43,
        "daily_range": 3.2593,
        "vol_x": 2.2914,
        "adx": 20.58,
        "intraday_change_pct": 3.0532,
    }
    values.update(overrides)
    return ReplayCandidate(**values)


class BtcBenchmarkReplayTest(unittest.TestCase):
    def test_last_closed_candle_does_not_expose_current_open_bar(self) -> None:
        timestamps = np.array([0, 900_000, 1_800_000], dtype=np.int64)
        self.assertIsNone(_find_last_closed_candle_index(timestamps, 899_999, 900_000))
        self.assertEqual(_find_last_closed_candle_index(timestamps, 900_000, 900_000), 0)
        self.assertEqual(_find_last_closed_candle_index(timestamps, 1_799_999, 900_000), 0)
        self.assertEqual(_find_last_closed_candle_index(timestamps, 1_800_000, 900_000), 1)

    def test_bull_context_timestamps_are_one_hour_candle_closes(self) -> None:
        data = np.zeros(
            60,
            dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")],
        )
        data["t"] = np.arange(60, dtype=np.int64) * 3_600_000
        data["o"] = data["h"] = data["l"] = data["c"] = 100.0
        data["v"] = 1.0
        context = _build_bull_day_context(data)
        self.assertIsNotNone(context)
        self.assertEqual(int(context[0][0]), 3_600_000)

    def test_discovery_catchup_reuses_prior_mode_with_bounded_slippage(self) -> None:
        prior = ("breakout", 1.5, 6, False)
        with patch("replay_backtest._entry_candidate", return_value=prior):
            result = _discovery_catchup_replay_candidate(
                {},
                2,
                np.array([99.0, 100.0, 100.4]),
                "1h",
            )
        self.assertIsNotNone(result)
        self.assertEqual(result[0], prior)
        self.assertEqual(result[1], 1)
        self.assertAlmostEqual(result[2], 0.4)

    def test_discovery_catchup_rejects_excess_positive_slippage(self) -> None:
        prior = ("breakout", 1.5, 6, False)
        with patch("replay_backtest._entry_candidate", return_value=prior):
            result = _discovery_catchup_replay_candidate(
                {},
                2,
                np.array([99.0, 100.0, 100.46]),
                "1h",
            )
        self.assertIsNone(result)

    def test_replay_top_gainer_score_has_exact_live_formula_parity(self) -> None:
        kwargs = {
            "mode": "trend",
            "intraday_change_pct": 2.5,
            "daily_range": 3.0,
            "vol_x": 1.8,
            "adx": 24.0,
            "rsi": 65.0,
        }
        ranker_info = {"final_score": 0.2, "top_gainer_prob": 0.1}
        live = _top_gainer_live_score(**kwargs, ranker_info=ranker_info)
        replay = _top_gainer_replay_score(
            tf="15m",
            **kwargs,
            ranker_final_score=ranker_info["final_score"],
            ranker_top_gainer_prob=ranker_info["top_gainer_prob"],
        )
        self.assertEqual(replay, live)

    def test_observed_pre_overbought_btc_breakout_matches_frozen_profile(self) -> None:
        self.assertTrue(
            _btc_1h_leader_admission_replay_ok(
                _candidate(),
                normal_score_min=34.0,
            )
        )

    def test_profile_rejects_overbought_or_non_btc_candidates(self) -> None:
        self.assertFalse(
            _btc_1h_leader_admission_replay_ok(
                _candidate(rsi=70.01),
                normal_score_min=34.0,
            )
        )
        self.assertFalse(
            _btc_1h_leader_admission_replay_ok(
                _candidate(sym="ETHUSDT"),
                normal_score_min=34.0,
            )
        )

    def test_profile_is_not_used_when_normal_gate_already_passes(self) -> None:
        self.assertFalse(
            _btc_1h_leader_admission_replay_ok(
                _candidate(top_gainer_score=34.0),
                normal_score_min=34.0,
            )
        )

    def test_rotation_profile_covers_earlier_btc_breakout_but_not_overbought(self) -> None:
        self.assertTrue(
            _btc_benchmark_rotation_replay_ok(
                _candidate(
                    top_gainer_score=37.9997,
                    score=136.6428,
                    rsi=71.46,
                    adx=18.55,
                    vol_x=4.81,
                    intraday_change_pct=3.05,
                    daily_range=3.45,
                )
            )
        )
        self.assertFalse(_btc_benchmark_rotation_replay_ok(_candidate(rsi=72.01)))

    def test_cluster_exemption_is_btc_and_variant_specific(self) -> None:
        self.assertTrue(_btc_cluster_exempt_replay("BTCUSDT", "btc_cluster_exempt"))
        self.assertTrue(_btc_cluster_exempt_replay("btcusdt", "btc_benchmark_combined"))
        self.assertFalse(_btc_cluster_exempt_replay("ETHUSDT", "btc_cluster_exempt"))
        self.assertFalse(_btc_cluster_exempt_replay("BTCUSDT", "score_replace_cluster"))

    def test_period_metrics_deduct_round_trip_cost_per_trade(self) -> None:
        trades = [
            ReplayTrade(
                sym="BTCUSDT",
                tf="1h",
                mode="breakout",
                entry_ts=1_000,
                entry_price=100.0,
                entry_i=1,
                trail_k=1.5,
                max_hold_bars=6,
                trail_stop=98.0,
                exit_price=101.0,
            ),
            ReplayTrade(
                sym="ETHUSDT",
                tf="15m",
                mode="trend",
                entry_ts=2_000,
                entry_price=100.0,
                entry_i=1,
                trail_k=2.0,
                max_hold_bars=12,
                trail_stop=98.0,
                exit_price=99.0,
            ),
        ]
        metrics = _period_metrics(trades, start_ms=0, end_ms=3_000, cost_bps=20.0)
        self.assertEqual(metrics["trades"], 2)
        self.assertAlmostEqual(metrics["net_pnl_pct"], -0.4)
        self.assertAlmostEqual(metrics["btc_net_pnl_pct"], 0.8)

    def test_decision_requires_sample_before_quality(self) -> None:
        result = _decision(
            variant="btc_cluster_exempt",
            run_stats={"btc_cluster_exemption_admitted": 4},
            all_metrics={"btc_net_pnl_pct": 1.0},
            holdout_metrics={"trades": 10},
            holdout_delta={"net_pnl_pct": 1.0, "net_avg_pct": 0.1, "net_win_rate": 0.0},
        )
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_combined_admissions_are_unique_not_sum_of_bypass_counters(self) -> None:
        self.assertEqual(
            _policy_admissions(
                "btc_benchmark_combined",
                {
                    "btc_cluster_exemption_admitted": 2,
                    "btc_1h_leader_admitted": 1,
                    "btc_benchmark_policy_admitted": 2,
                },
            ),
            2,
        )

    def test_decision_advances_only_when_all_frozen_gates_pass(self) -> None:
        result = _decision(
            variant="btc_1h_leader_admission",
            run_stats={"btc_1h_leader_admitted": 5},
            all_metrics={"btc_net_pnl_pct": 1.0},
            holdout_metrics={"trades": 10},
            holdout_delta={"net_pnl_pct": 1.0, "net_avg_pct": 0.1, "net_win_rate": -0.01},
        )
        self.assertEqual(result["status"], "advance_to_shadow_review")


if __name__ == "__main__":
    unittest.main()
