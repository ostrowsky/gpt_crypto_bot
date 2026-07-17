from __future__ import annotations

import unittest

import config
from market_signal_agent import AgentPosition, _replacement_block_non_losing_context


def _pos(*, mark_price: float, leader_score: float = 20.0) -> AgentPosition:
    return AgentPosition(
        symbol="OLDUSDT",
        tf="15m",
        entry_price=100.0,
        entry_bar=1,
        entry_ts=1,
        entry_ema20=99.0,
        entry_slope=0.1,
        entry_adx=20.0,
        entry_rsi=55.0,
        entry_vol_x=1.2,
        leader_score=leader_score,
        signal_mode="trend",
        mark_price=mark_price,
    )


class MarketSignalAgentReplacementPolicyTest(unittest.TestCase):
    def test_replay_validated_guard_is_enabled_by_default(self) -> None:
        self.assertTrue(config.AGENT_REPLACEMENT_BLOCK_NON_LOSING_ENABLED)
        self.assertTrue(config.AGENT_REPLACEMENT_BLOCK_NON_LOSING_SHADOW)

    def test_block_non_losing_context_blocks_non_losing_position(self) -> None:
        ctx = _replacement_block_non_losing_context(
            {"symbol": "NEWUSDT", "tf": "15m", "mode": "trend", "leader_score": 35.0},
            _pos(mark_price=100.0),
        )
        self.assertTrue(ctx["would_block"])
        self.assertEqual(ctx["replaced_pnl_pct"], 0.0)
        self.assertEqual(ctx["leader_delta"], 15.0)
        self.assertIn("block_non_losing", ctx["reason"])

    def test_block_non_losing_context_allows_losing_position(self) -> None:
        ctx = _replacement_block_non_losing_context(
            {"symbol": "NEWUSDT", "tf": "15m", "mode": "trend", "leader_score": 35.0},
            _pos(mark_price=99.0),
        )
        self.assertFalse(ctx["would_block"])
        self.assertAlmostEqual(ctx["replaced_pnl_pct"], -1.0)
        self.assertEqual(ctx["reason"], "")


if __name__ == "__main__":
    unittest.main()
