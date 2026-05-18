from __future__ import annotations

import unittest

from audit_v2_exit_quality_baselines import _policy
from test_v2_offline_env import _frame
from v2.offline_env import DecisionFrame, OfflineDecisionEnvironment
from v2.state import Action, SymbolState


def _replace_belief(frame: DecisionFrame, probs) -> DecisionFrame:
    return DecisionFrame(frame.bar, frame.label, frame.belief.__class__(probs), frame.prediction)


class ExitQualityBaselinesTest(unittest.TestCase):
    def test_early_sell_profile_uses_lower_late_mass(self) -> None:
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
        sell_frame = _replace_belief(
            _frame(1, SymbolState.EXHAUSTION, 102.0, 103.0),
            {
                SymbolState.NOISE: 0.05,
                SymbolState.EMERGING_MOVE: 0.05,
                SymbolState.CONFIRMED_TREND: 0.10,
                SymbolState.MATURE_TREND: 0.15,
                SymbolState.EXHAUSTION: 0.35,
                SymbolState.REVERSAL: 0.30,
            },
        )
        rows = {
            (open_frame.bar.symbol, open_frame.bar.open_ts_ms): {
                "belief": {"emerging_move": 0.60, "confirmed_trend": 0.15},
                "v1_projected_structural": {"projected_leader_score_trend": 4.0},
            }
        }
        env = OfflineDecisionEnvironment([open_frame, sell_frame])
        self.assertEqual(_policy(rows, profile="early_sell_0_60")(env), Action.OPEN_FULL)
        env.step(Action.OPEN_FULL)
        self.assertEqual(_policy(rows, profile="early_sell_0_60")(env), Action.SELL)


if __name__ == "__main__":
    unittest.main()
