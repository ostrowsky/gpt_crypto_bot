from __future__ import annotations

import unittest

from test_v2_offline_env import _frame
from v2.offline_env import OfflineDecisionEnvironment
from v2.policy_baselines import belief_policy_v1, lifecycle_oracle_policy, rollout, summarize_policy
from v2.state import Action, SymbolState


class PolicyBaselinesTest(unittest.TestCase):
    def test_oracle_opens_and_sells_on_lifecycle(self) -> None:
        env = OfflineDecisionEnvironment(
            [
                _frame(0, SymbolState.EMERGING_MOVE, 101.0, 101.2),
                _frame(1, SymbolState.CONFIRMED_TREND, 103.0, 103.5),
                _frame(2, SymbolState.EXHAUSTION, 102.0, 105.0),
            ]
        )
        actions = [step.action for step in rollout(env, lifecycle_oracle_policy)]
        self.assertEqual(actions, [Action.OPEN_FULL, Action.HOLD, Action.SELL])

    def test_summary_counts_trades(self) -> None:
        env = OfflineDecisionEnvironment(
            [
                _frame(0, SymbolState.EMERGING_MOVE, 101.0, 101.2),
                _frame(1, SymbolState.REVERSAL, 100.0, 101.2),
            ]
        )
        summary = summarize_policy("belief_policy_v1", [rollout(env, belief_policy_v1)])
        self.assertEqual(summary["trade_count"], 1)
        self.assertEqual(summary["episodes"], 1)


if __name__ == "__main__":
    unittest.main()
