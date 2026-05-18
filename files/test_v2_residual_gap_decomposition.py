from __future__ import annotations

import unittest

from audit_v2_residual_gap_decomposition import _admission_mix
from test_v2_offline_env import _frame
from v2.offline_env import OfflineDecisionEnvironment
from v2.policy_baselines import lifecycle_oracle_policy, rollout
from v2.state import SymbolState


class ResidualGapDecompositionTest(unittest.TestCase):
    def test_admission_mix_reports_productive_share(self) -> None:
        env = OfflineDecisionEnvironment(
            [
                _frame(0, SymbolState.EMERGING_MOVE, 101.0, 101.2),
                _frame(1, SymbolState.CONFIRMED_TREND, 103.0, 103.5),
                _frame(2, SymbolState.EXHAUSTION, 102.0, 105.0),
            ]
        )
        summary = _admission_mix([rollout(env, lifecycle_oracle_policy)])
        self.assertEqual(summary["productive_entry_share"], 1.0)
        self.assertEqual(summary["noise_entry_share"], 0.0)


if __name__ == "__main__":
    unittest.main()
