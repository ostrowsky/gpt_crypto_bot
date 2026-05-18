from __future__ import annotations

import unittest

from audit_v2_entry_admission_reward_replay import _threshold_policy
from test_v2_offline_env import _frame
from v2.offline_env import DecisionFrame, OfflineDecisionEnvironment
from v2.state import Action, SymbolState


def _belief_frame() -> DecisionFrame:
    frame = _frame(0, SymbolState.EMERGING_MOVE, 101.0, 101.2)
    probs = frame.belief.probabilities.copy()
    probs[SymbolState.EMERGING_MOVE] = 0.60
    probs[SymbolState.CONFIRMED_TREND] = 0.15
    probs[SymbolState.NOISE] = 0.05
    probs[SymbolState.MATURE_TREND] = 0.05
    probs[SymbolState.EXHAUSTION] = 0.10
    probs[SymbolState.REVERSAL] = 0.05
    return DecisionFrame(frame.bar, frame.label, frame.belief.__class__(probs), frame.prediction)


class EntryAdmissionRewardReplayTest(unittest.TestCase):
    def test_combined_admission_can_reject_open(self) -> None:
        frame = _belief_frame()
        key = (frame.bar.symbol, frame.bar.open_ts_ms)
        rows = {
            key: {
                "belief": {"emerging_move": 0.60, "confirmed_trend": 0.15},
                "v1_projected_structural": {"projected_leader_score_trend": 2.0},
            }
        }
        env = OfflineDecisionEnvironment([frame])
        self.assertEqual(_threshold_policy(rows, admission="combined")(env), Action.IGNORE)
        rows[key]["v1_projected_structural"]["projected_leader_score_trend"] = 4.0
        self.assertEqual(_threshold_policy(rows, admission="combined")(env), Action.OPEN_FULL)


if __name__ == "__main__":
    unittest.main()
