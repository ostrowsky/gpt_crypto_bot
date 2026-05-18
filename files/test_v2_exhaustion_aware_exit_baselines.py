from __future__ import annotations

import unittest

from audit_v2_exhaustion_aware_exit_baselines import _policy
from test_v2_offline_env import _frame
from v2.offline_env import DecisionFrame, OfflineDecisionEnvironment
from v2.state import Action, SymbolState


def _replace_belief(frame: DecisionFrame, probs) -> DecisionFrame:
    return DecisionFrame(frame.bar, frame.label, frame.belief.__class__(probs), frame.prediction)


class ExhaustionAwareExitBaselinesTest(unittest.TestCase):
    def test_consensus_profile_requires_all_conditions(self) -> None:
        open_frame = _replace_belief(
            _frame(0, SymbolState.EMERGING_MOVE, 101.0, 101.2),
            {
                SymbolState.NOISE: 0.05,
                SymbolState.EMERGING_MOVE: 0.60,
                SymbolState.CONFIRMED_TREND: 0.15,
                SymbolState.MATURE_TREND: 0.05,
                SymbolState.EXHAUSTION: 0.10,
                SymbolState.REVERSAL: 0.05,
            },
        )
        exit_frame = _replace_belief(
            _frame(1, SymbolState.EXHAUSTION, 102.0, 103.0),
            {
                SymbolState.NOISE: 0.05,
                SymbolState.EMERGING_MOVE: 0.05,
                SymbolState.CONFIRMED_TREND: 0.10,
                SymbolState.MATURE_TREND: 0.20,
                SymbolState.EXHAUSTION: 0.35,
                SymbolState.REVERSAL: 0.25,
            },
        )
        rows = {
            (open_frame.bar.symbol, open_frame.bar.open_ts_ms): {
                "belief": {"emerging_move": 0.60, "confirmed_trend": 0.15},
                "v1_projected_structural": {"projected_leader_score_trend": 4.0},
            },
            (exit_frame.bar.symbol, exit_frame.bar.open_ts_ms): {
                "v1_projected_structural": {"rsi": 55.0, "price_vs_ema20_pct": 0.2},
            },
        }
        env = OfflineDecisionEnvironment([open_frame, exit_frame])
        self.assertEqual(_policy(rows, profile="consensus_exhaustion")(env), Action.OPEN_FULL)
        env.step(Action.OPEN_FULL)
        self.assertEqual(_policy(rows, profile="consensus_exhaustion")(env), Action.SELL)


if __name__ == "__main__":
    unittest.main()
