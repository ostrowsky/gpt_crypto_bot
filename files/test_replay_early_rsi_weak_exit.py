from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import replay_early_rsi_weak_exit as replay


def _path(*, structure: bool = True, bars_held: int = 2) -> replay.CasePath:
    candles = np.zeros(
        6,
        dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")],
    )
    candles["t"] = np.arange(6) * 900_000
    candles["c"] = [99.0, 100.0, 101.0, 102.0, 103.0, 104.0]
    candles["o"] = candles["c"]
    candles["h"] = candles["c"] + 0.2
    candles["l"] = candles["c"] - 0.2
    candles["v"] = 1.0
    ema20 = np.array([98.0, 99.0, 100.0, 101.0, 102.0, 103.0])
    if not structure:
        ema20[2] = 102.0
    feat = {
        "atr": np.full(6, 10.0),
        "ema_fast": ema20,
        "ema_slow": np.array([97.0, 98.0, 99.0, 100.0, 101.0, 102.0]),
        "adx": np.full(6, 30.0),
        "slope": np.full(6, 0.2),
        "macd_hist": np.full(6, 0.1),
    }
    return replay.CasePath(
        event={
            "event": "exit",
            "sym": "AAAUSDT",
            "tf": "15m",
            "mode": "retest",
            "entry_price": 100.0,
            "exit_price": 101.0,
            "pnl_pct": 1.0,
            "bars_held": bars_held,
            "trail_k": 1.8,
            "reason": "WEAK: RSI divergence",
            "ts": "2026-01-01T00:00:00Z",
        },
        candles=candles,
        feat=feat,
        decision_idx=2,
        context_1h={"intact": True},
    )


class EarlyRsiWeakExitReplayTest(unittest.TestCase):
    def test_event_loader_keeps_only_unique_rsi_weak_exits(self) -> None:
        row = {
            "event": "exit", "sym": "AAAUSDT", "tf": "15m", "mode": "retest",
            "entry_price": 100, "exit_price": 101, "bars_held": 2,
            "reason": "WEAK: RSI divergence", "ts": "2026-01-01T00:00:00Z",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text(
                "\n".join((json.dumps(row), json.dumps(row), json.dumps({**row, "reason": "ATR trail"}))),
                encoding="utf-8",
            )
            self.assertEqual(len(replay.load_exit_events(path)), 1)

    def test_grace_four_waits_until_replayed_weak_at_bar_four(self) -> None:
        reasons = iter(("WEAK: RSI divergence", "WEAK: RSI divergence"))
        outcome = replay.simulate_policy(
            _path(),
            replay.PolicySpec("grace", "grace", "retest15", 4),
            replay.ReplayConfig(horizon_bars=2),
            reason_fn=lambda *args, **kwargs: next(reasons),
        )
        self.assertTrue(outcome["applicable"])
        self.assertEqual(outcome["exit_step"], 2)
        self.assertEqual(outcome["exit_reason"], "weak_after_grace")
        self.assertGreater(outcome["net_delta_pct"], 0.0)

    def test_two_observation_confirmation_exits_on_next_weak_bar(self) -> None:
        outcome = replay.simulate_policy(
            _path(),
            replay.PolicySpec("confirm", "confirm2", "retest15", 2),
            replay.ReplayConfig(horizon_bars=2),
            reason_fn=lambda *args, **kwargs: "WEAK: RSI divergence",
        )
        self.assertEqual(outcome["exit_step"], 1)
        self.assertEqual(outcome["exit_reason"], "weak_confirmed")

    def test_structure_policy_does_not_change_broken_structure(self) -> None:
        outcome = replay.simulate_policy(
            _path(structure=False),
            replay.PolicySpec("structure", "structure", "retest15", 1.2),
            replay.ReplayConfig(horizon_bars=2),
            reason_fn=lambda *args, **kwargs: None,
        )
        self.assertFalse(outcome["applicable"])
        self.assertEqual(outcome["exit_reason"], "structure_not_intact")
        self.assertEqual(outcome["delta_pct"], 0.0)

    def test_partial_mtf_tail_scales_full_tail_delta(self) -> None:
        full = replay.simulate_policy(
            _path(),
            replay.PolicySpec("mtf", "mtf", "retest15", 1.4),
            replay.ReplayConfig(horizon_bars=2),
            reason_fn=lambda *args, **kwargs: None,
        )
        partial = replay.simulate_policy(
            _path(),
            replay.PolicySpec("partial", "partial_mtf", "retest15", 1.4, 0.5),
            replay.ReplayConfig(horizon_bars=2),
            reason_fn=lambda *args, **kwargs: None,
        )
        self.assertAlmostEqual(partial["delta_pct"], full["delta_pct"] * 0.5, places=3)
        self.assertAlmostEqual(partial["net_delta_pct"], full["delta_pct"] * 0.5 - 0.025, places=3)

    def test_day_split_keeps_each_day_in_one_partition(self) -> None:
        result = replay.chronological_day_splits([f"2026-01-{day:02d}" for day in range(1, 11)])
        self.assertEqual(list(result.values()).count("train"), 6)
        self.assertEqual(list(result.values()).count("validation"), 2)
        self.assertEqual(list(result.values()).count("holdout"), 2)


if __name__ == "__main__":
    unittest.main()
