from __future__ import annotations

import unittest

from audit_v2_temporal_exit_baselines import _build_temporal_rows, _policy
from test_v2_offline_env import _frame
from v2.offline_env import DecisionFrame, OfflineDecisionEnvironment
from v2.state import Action, SymbolState


def _replace_belief(frame: DecisionFrame, probs) -> DecisionFrame:
    return DecisionFrame(frame.bar, frame.label, frame.belief.__class__(probs), frame.prediction)


class TemporalExitBaselinesTest(unittest.TestCase):
    def test_temporal_rows_compute_three_bar_deltas(self) -> None:
        rows = {
            ("AAA", 1): {
                "belief": {"exhaustion": 0.10, "reversal": 0.05, "mature_trend": 0.40},
                "v1_projected_structural": {"rsi": 62.0, "price_vs_ema20_pct": 1.20},
            },
            ("AAA", 2): {
                "belief": {"exhaustion": 0.12, "reversal": 0.05, "mature_trend": 0.35},
                "v1_projected_structural": {"rsi": 60.0, "price_vs_ema20_pct": 1.00},
            },
            ("AAA", 3): {
                "belief": {"exhaustion": 0.14, "reversal": 0.06, "mature_trend": 0.30},
                "v1_projected_structural": {"rsi": 58.0, "price_vs_ema20_pct": 0.80},
            },
            ("AAA", 4): {
                "belief": {"exhaustion": 0.30, "reversal": 0.10, "mature_trend": 0.20},
                "v1_projected_structural": {"rsi": 55.0, "price_vs_ema20_pct": 0.40},
            },
        }
        temporal = _build_temporal_rows(rows)
        self.assertFalse(temporal[("AAA", 3)]["has_3bar_history"])
        self.assertTrue(temporal[("AAA", 4)]["has_3bar_history"])
        self.assertAlmostEqual(temporal[("AAA", 4)]["late_mass_delta_3"], 0.25)
        self.assertAlmostEqual(temporal[("AAA", 4)]["mature_delta_3"], -0.20)
        self.assertAlmostEqual(temporal[("AAA", 4)]["rsi_delta_3"], -7.0)
        self.assertAlmostEqual(temporal[("AAA", 4)]["price_vs_ema20_delta_3"], -0.8)

    def test_consensus_temporal_requires_three_bar_confirmation(self) -> None:
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
                SymbolState.REVERSAL: 0.20,
            },
        )
        rows = {
            (open_frame.bar.symbol, open_frame.bar.open_ts_ms): {
                "belief": {"emerging_move": 0.60, "confirmed_trend": 0.15},
                "v1_projected_structural": {"projected_leader_score_trend": 4.0},
            }
        }
        temporal = {
            (exit_frame.bar.symbol, exit_frame.bar.open_ts_ms): {
                "has_3bar_history": True,
                "late_mass_delta_3": 0.15,
                "mature_delta_3": -0.20,
                "rsi_delta_3": -4.0,
                "price_vs_ema20_delta_3": -0.30,
            }
        }
        env = OfflineDecisionEnvironment([open_frame, exit_frame])
        self.assertEqual(_policy(rows, temporal, profile="consensus_temporal")(env), Action.OPEN_FULL)
        env.step(Action.OPEN_FULL)
        self.assertEqual(_policy(rows, temporal, profile="consensus_temporal")(env), Action.SELL)


if __name__ == "__main__":
    unittest.main()
