from __future__ import annotations

import unittest

from v2.belief import BeliefState
from v2.history import CanonicalBar
from v2.lifecycle_labeling import LifecycleLabel
from v2.offline_env import DecisionFrame, OfflineDecisionEnvironment
from v2.state import Action, SymbolState


def _frame(i: int, state: SymbolState, close: float, high: float) -> DecisionFrame:
    return DecisionFrame(
        bar=CanonicalBar(
            symbol="TESTUSDT",
            timeframe="15m",
            open_ts_ms=1_700_000_000_000 + (i * 900_000),
            open=100.0,
            high=high,
            low=99.0,
            close=close,
            volume=1_000.0,
        ),
        label=LifecycleLabel(
            symbol="TESTUSDT",
            timeframe="15m",
            open_ts_ms=1_700_000_000_000 + (i * 900_000),
            local_day="2026-05-18",
            state=state,
            label_version="test",
            day_open=100.0,
            day_mfe_pct=5.0,
            peak_index=2,
            confirmation_index=1,
        ),
        belief=BeliefState.uniform(list(SymbolState)),
        prediction=state,
    )


class OfflineDecisionEnvironmentTest(unittest.TestCase):
    def test_open_hold_sell_path_is_deterministic(self) -> None:
        frames = [
            _frame(0, SymbolState.EMERGING_MOVE, 101.0, 101.2),
            _frame(1, SymbolState.CONFIRMED_TREND, 103.0, 103.5),
            _frame(2, SymbolState.EXHAUSTION, 102.0, 105.0),
        ]
        env = OfflineDecisionEnvironment(frames, max_open_positions=1)
        self.assertIn(Action.OPEN_FULL, env.legal_actions())

        opened = env.step(Action.OPEN_FULL)
        self.assertGreater(opened.reward.early_capture_reward, 0.0)
        self.assertEqual(env.legal_actions(), {Action.HOLD, Action.TIGHTEN_EXIT, Action.REDUCE, Action.SELL})

        held = env.step(Action.HOLD)
        self.assertEqual(held.reward.trend_hold_reward, 0.25)

        sold = env.step(Action.SELL)
        self.assertTrue(sold.done)
        self.assertGreater(sold.reward.realized_pnl_reward, 0.0)
        self.assertEqual(env.reset().bar.open_ts_ms, frames[0].bar.open_ts_ms)

    def test_illegal_action_is_rejected(self) -> None:
        env = OfflineDecisionEnvironment([_frame(0, SymbolState.NOISE, 100.0, 100.5)])
        with self.assertRaises(ValueError):
            env.step(Action.SELL)


if __name__ == "__main__":
    unittest.main()
