from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

import replay_early_rsi_weak_exit as base
import replay_impulse_expansion_tail as replay


def _strong_path() -> base.CasePath:
    n = 40
    candles = np.zeros(
        n,
        dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")],
    )
    close = np.linspace(100.0, 120.0, n)
    candles["t"] = np.arange(n) * 900_000
    candles["c"] = close
    candles["o"] = close - 0.2
    candles["h"] = close + 0.4
    candles["l"] = close - 0.4
    candles["v"] = np.linspace(100.0, 200.0, n)
    feat = {
        "atr": np.full(n, 1.0),
        "adx": np.linspace(35.0, 60.0, n),
        "slope": np.linspace(0.2, 1.5, n),
        "macd_hist": np.full(n, 0.1),
        "ema_fast": close - 2.0,
        "ema_slow": close - 4.0,
        "rsi": np.full(n, 70.0),
        "rsi_divergence": np.ones(n),
        "vol_exhaustion": np.zeros(n),
        "ema_fan_spread": np.zeros(n),
    }
    return base.CasePath(
        event={
            "event": "exit", "sym": "AAAUSDT", "tf": "15m", "mode": "impulse_speed",
            "entry_price": 110.0, "exit_price": close[29], "pnl_pct": (close[29] / 110.0 - 1.0) * 100.0,
            "bars_held": 4, "trail_k": 2.5, "reason": "WEAK: RSI divergence",
            "ts": "2026-01-01T00:00:00Z",
        },
        candles=candles,
        feat=feat,
        decision_idx=29,
    )


class ImpulseExpansionTailReplayTest(unittest.TestCase):
    def test_loader_keeps_unique_15m_impulse_weak_exits(self) -> None:
        row = {
            "event": "exit", "sym": "AAAUSDT", "tf": "15m", "mode": "impulse_speed",
            "entry_price": 100, "exit_price": 102, "pnl_pct": 2, "bars_held": 4,
            "reason": "WEAK: volume exhaustion", "ts": "2026-01-01T00:00:00Z",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text("\n".join((json.dumps(row), json.dumps(row), json.dumps({**row, "mode": "retest"}))), encoding="utf-8")
            self.assertEqual(len(replay.load_events(path)), 1)

    def test_strong_trend_scores_above_primary_floor(self) -> None:
        features = replay.expansion_features(_strong_path(), 29)
        self.assertGreaterEqual(features["score"], 8)
        self.assertTrue(features["components"]["adx_high"])
        self.assertTrue(features["components"]["efficient_trend"])

    def test_protected_tail_keeps_most_profit_when_following_bar_falls(self) -> None:
        path = _strong_path()
        path.candles["c"][30] = path.candles["c"][29] * 0.99
        path.candles["l"][30] = path.candles["c"][30] - 0.2
        features = replay.expansion_features(path, 29)
        policy = replay.TailPolicy("primary", min_score=8)
        with patch.object(replay, "check_exit_conditions", return_value="WEAK: RSI divergence"):
            outcome = replay.simulate_tail(path, features, policy, replay.ReplayConfig(horizon_bars=2))
        self.assertTrue(outcome["applicable"])
        self.assertEqual(outcome["exit_reason"], "protected_trail")
        self.assertGreater(outcome["profit_retention_pct"], 90.0)

    def test_soft_weak_is_ignored_while_expansion_remains_active(self) -> None:
        path = _strong_path()
        features = replay.expansion_features(path, 29)
        policy = replay.TailPolicy("primary", min_score=8, decay_score=None)
        with patch.object(replay, "check_exit_conditions", return_value="WEAK: RSI divergence"):
            outcome = replay.simulate_tail(path, features, policy, replay.ReplayConfig(horizon_bars=2))
        self.assertEqual(outcome["exit_reason"], "horizon")
        self.assertGreater(outcome["net_delta_pct"], 0.0)

    def test_cmf_handles_zero_range_candles_without_runtime_warning(self) -> None:
        path = _strong_path()
        path.candles[10:30]["h"] = path.candles[10:30]["l"]
        path.candles[10:30]["c"] = path.candles[10:30]["l"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            value = replay._cmf(path.candles, 29, 20)
        self.assertEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
