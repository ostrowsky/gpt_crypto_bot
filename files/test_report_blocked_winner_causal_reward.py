from __future__ import annotations

import unittest

import report_blocked_winner_causal_reward as rep


class BlockedWinnerCausalRewardTests(unittest.TestCase):
    def test_harm_reward_for_missed_top(self) -> None:
        reward = rep._harm_reward({"is_top15": True, "status": "missed", "opportunity_from_first_block_pct": 7.0}, False, rep.BlockerRewardConfig())
        self.assertEqual(reward, 7.0)

    def test_harm_reward_zero_when_already_bought_before_block(self) -> None:
        reward = rep._harm_reward({"is_top15": True, "status": "missed", "opportunity_from_first_block_pct": 7.0}, True, rep.BlockerRewardConfig())
        self.assertEqual(reward, 0.0)

    def test_reason_table_advances_harmful_reason(self) -> None:
        rows = [
            {"reason_code": "score", "is_top15": True, "harm_pct": 7.0, "protection_credit_pct": 0.0, "block_count": 5},
            {"reason_code": "score", "is_top15": True, "harm_pct": 6.0, "protection_credit_pct": 0.0, "block_count": 4},
            {"reason_code": "score", "is_top15": False, "harm_pct": 0.0, "protection_credit_pct": 1.0, "block_count": 3},
        ]
        table = rep._reason_table(rows, rep.BlockerRewardConfig(min_harm_pct=10.0, min_harmful_cases=2))
        self.assertEqual(table[0]["reason_code"], "score")
        self.assertEqual(table[0]["decision"], "advance_to_behavior_replay")
        self.assertEqual(table[0]["net_harm_pct"], 12.0)

    def test_case_rows_credit_false_candidate_protection(self) -> None:
        events = [{"day": "2026-05-30", "symbol": "AAAUSDT", "reason_code": "score", "ts": "t", "hour": 1, "block_count": 12}]
        rows = rep._case_rows(events, {}, {}, rep.BlockerRewardConfig(false_candidate_credit_pct=1.5))
        self.assertEqual(rows[0]["block_count"], 12)
        self.assertEqual(rows[0]["protection_credit_pct"], 1.5)
        self.assertEqual(rows[0]["net_harm_pct"], -1.5)


if __name__ == "__main__":
    unittest.main()
