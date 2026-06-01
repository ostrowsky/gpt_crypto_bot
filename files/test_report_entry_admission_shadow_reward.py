from __future__ import annotations

import unittest

import report_entry_admission_shadow_reward as rep


class EntryAdmissionShadowRewardTests(unittest.TestCase):
    def test_candidate_reward_for_missed_top_uses_opportunity(self) -> None:
        reward = rep._candidate_reward(
            {"is_top15": True, "status": "missed", "opportunity_from_first_block_pct": 8.5},
            False,
            rep.RewardConfig(),
        )
        self.assertEqual(reward, 8.5)

    def test_candidate_reward_zero_when_already_bought_before_rescue(self) -> None:
        reward = rep._candidate_reward(
            {"is_top15": True, "status": "bought", "capture_ratio_at_entry": 0.1, "opportunity_from_first_block_pct": 8.5},
            True,
            rep.RewardConfig(),
        )
        self.assertEqual(reward, 0.0)

    def test_evaluate_penalizes_false_candidates(self) -> None:
        events = [
            {"day": "2026-05-30", "symbol": "AAAUSDT", "hour": 3, "ts": "a", "reason_code": "agent_mode_disabled"},
            {"day": "2026-05-30", "symbol": "AAAUSDT", "hour": 3, "ts": "b", "reason_code": "agent_mode_disabled"},
            {"day": "2026-05-30", "symbol": "BBBUSDT", "hour": 3, "ts": "c", "reason_code": "agent_mode_disabled"},
            {"day": "2026-05-30", "symbol": "BBBUSDT", "hour": 3, "ts": "d", "reason_code": "agent_mode_disabled"},
        ]
        labels = {
            ("2026-05-30", "AAAUSDT"): {"is_top15": True, "status": "missed", "opportunity_from_first_block_pct": 10.0},
        }
        item = rep._evaluate(events, {}, labels, "agent_only", {"agent_mode_disabled"}, 4, 2, rep.RewardConfig(false_candidate_penalty_pct=1.0))
        self.assertEqual(item["top_candidates"], 1)
        self.assertEqual(item["false_candidates"], 1)
        self.assertEqual(item["net_reward_pct"], 9.0)

    def test_decision_advances_only_when_gate_passes(self) -> None:
        cfg = rep.RewardConfig(min_net_reward_pct=5.0, min_top_precision=0.2, min_rescued_top=1)
        item = {"net_reward_pct": 9.0, "top_precision": 0.5, "rewarded_top_candidates": 1, "candidate_count": 2}
        self.assertEqual(rep._decision(item, cfg), "advance_to_entry_admission_behavior_replay")


if __name__ == "__main__":
    unittest.main()
