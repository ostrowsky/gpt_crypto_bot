from __future__ import annotations

import unittest


class TestV2StateGraph(unittest.TestCase):
    def test_symbol_graph_allows_plausible_progression(self) -> None:
        from v2.state import SymbolState, symbol_transition_allowed

        self.assertTrue(symbol_transition_allowed(SymbolState.NOISE, SymbolState.EMERGING_MOVE))
        self.assertTrue(symbol_transition_allowed(SymbolState.CONFIRMED_TREND, SymbolState.MATURE_TREND))
        self.assertFalse(symbol_transition_allowed(SymbolState.NOISE, SymbolState.EXHAUSTION))


class TestV2BeliefState(unittest.TestCase):
    def test_belief_normalizes_and_updates(self) -> None:
        from v2.belief import BeliefState
        from v2.state import SymbolState

        belief = BeliefState({SymbolState.NOISE: 2.0, SymbolState.EMERGING_MOVE: 1.0})
        self.assertAlmostEqual(belief.probability(SymbolState.NOISE), 2 / 3)
        updated = belief.update(
            {
                SymbolState.NOISE: 0.1,
                SymbolState.EMERGING_MOVE: 0.9,
            }
        )
        self.assertEqual(updated.most_likely(), SymbolState.EMERGING_MOVE)


class TestV2Reward(unittest.TestCase):
    def test_reward_breakdown_is_named_and_objective_aligned(self) -> None:
        from v2.reward import RewardInputs, compute_reward

        reward = compute_reward(
            RewardInputs(
                realized_pnl_pct=2.0,
                capture_ratio_at_entry=0.2,
                exit_efficiency=0.7,
                giveback_pct=0.1,
                held_during_confirmed_trend=True,
            )
        )
        self.assertAlmostEqual(reward.early_capture_reward, 0.8)
        self.assertAlmostEqual(reward.trend_hold_reward, 0.25)
        self.assertAlmostEqual(reward.total, 3.65)


if __name__ == "__main__":
    unittest.main()
