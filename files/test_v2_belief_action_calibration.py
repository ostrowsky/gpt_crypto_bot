from __future__ import annotations

import unittest

from audit_v2_belief_action_calibration import _threshold_policy
from test_v2_offline_env import _frame
from v2.belief import BeliefState
from v2.offline_env import DecisionFrame, OfflineDecisionEnvironment
from v2.state import Action, SymbolState


def _replace_belief(frame: DecisionFrame, probs: dict[SymbolState, float]) -> DecisionFrame:
    return DecisionFrame(frame.bar, frame.label, BeliefState(probs), frame.prediction)


class BeliefActionCalibrationTest(unittest.TestCase):
    def test_threshold_policy_requires_sufficient_open_mass(self) -> None:
        weak = _replace_belief(
            _frame(0, SymbolState.EMERGING_MOVE, 101.0, 101.2),
            {
                SymbolState.NOISE: 0.50,
                SymbolState.EMERGING_MOVE: 0.20,
                SymbolState.CONFIRMED_TREND: 0.10,
                SymbolState.MATURE_TREND: 0.10,
                SymbolState.EXHAUSTION: 0.05,
                SymbolState.REVERSAL: 0.05,
            },
        )
        strong = _replace_belief(
            _frame(0, SymbolState.EMERGING_MOVE, 101.0, 101.2),
            {
                SymbolState.NOISE: 0.10,
                SymbolState.EMERGING_MOVE: 0.35,
                SymbolState.CONFIRMED_TREND: 0.30,
                SymbolState.MATURE_TREND: 0.10,
                SymbolState.EXHAUSTION: 0.10,
                SymbolState.REVERSAL: 0.05,
            },
        )
        self.assertEqual(_threshold_policy(0.50, 0.50)(OfflineDecisionEnvironment([weak])), Action.IGNORE)
        self.assertEqual(_threshold_policy(0.50, 0.50)(OfflineDecisionEnvironment([strong])), Action.OPEN_FULL)


if __name__ == "__main__":
    unittest.main()
