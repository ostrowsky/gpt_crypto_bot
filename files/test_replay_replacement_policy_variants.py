from __future__ import annotations

import unittest

from replay_backtest import _replacement_policy_block_reason


class ReplacementPolicyVariantTest(unittest.TestCase):
    def test_blocks_non_losing_position_for_non_losing_variant(self) -> None:
        reason = _replacement_policy_block_reason(
            variant="replacement_block_non_losing",
            replaced_pnl_pct=0.01,
            leader_delta=25.0,
        )
        self.assertIn("non-losing", reason)

    def test_allows_losing_position_for_non_losing_variant(self) -> None:
        reason = _replacement_policy_block_reason(
            variant="replacement_block_non_losing",
            replaced_pnl_pct=-0.01,
            leader_delta=1.0,
        )
        self.assertEqual(reason, "")

    def test_blocks_low_leader_delta(self) -> None:
        reason = _replacement_policy_block_reason(
            variant="replacement_block_leader_delta_lt_10",
            replaced_pnl_pct=-1.0,
            leader_delta=9.99,
        )
        self.assertIn("low leader delta", reason)

    def test_non_losing_unless_delta20_requires_both_conditions(self) -> None:
        self.assertIn(
            "leader_delta",
            _replacement_policy_block_reason(
                variant="replacement_block_non_losing_unless_delta20",
                replaced_pnl_pct=0.2,
                leader_delta=19.99,
            ),
        )
        self.assertEqual(
            _replacement_policy_block_reason(
                variant="replacement_block_non_losing_unless_delta20",
                replaced_pnl_pct=0.2,
                leader_delta=20.0,
            ),
            "",
        )
        self.assertEqual(
            _replacement_policy_block_reason(
                variant="replacement_block_non_losing_unless_delta20",
                replaced_pnl_pct=-0.2,
                leader_delta=1.0,
            ),
            "",
        )


if __name__ == "__main__":
    unittest.main()
