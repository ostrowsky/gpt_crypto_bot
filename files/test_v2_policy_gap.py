from __future__ import annotations

import unittest

from audit_v2_policy_gap import _summarize_trades
from test_v2_offline_env import _frame
from v2.offline_env import OfflineDecisionEnvironment
from v2.policy_baselines import lifecycle_oracle_policy, rollout
from v2.state import SymbolState


class PolicyGapAuditTest(unittest.TestCase):
    def test_trade_summary_tracks_entry_and_exit_states(self) -> None:
        env = OfflineDecisionEnvironment(
            [
                _frame(0, SymbolState.EMERGING_MOVE, 101.0, 101.2),
                _frame(1, SymbolState.CONFIRMED_TREND, 103.0, 103.5),
                _frame(2, SymbolState.EXHAUSTION, 102.0, 105.0),
            ]
        )
        summary = _summarize_trades([rollout(env, lifecycle_oracle_policy)])
        self.assertEqual(summary["trade_count"], 1)
        self.assertEqual(summary["entries_by_true_state"], {"emerging_move": 1})
        self.assertEqual(summary["exits_by_true_state"], {"exhaustion": 1})


if __name__ == "__main__":
    unittest.main()
