from __future__ import annotations

import unittest

from audit_post_block_causal_discriminator import _base_rates, _passes_gate, _split_by_day


class PostBlockCausalDiscriminatorAuditTest(unittest.TestCase):
    def test_split_by_day_is_chronological(self) -> None:
        rows = [{"local_day": f"2026-05-0{i}", "label_useful_missed_winner": False, "label_top15": False, "label_bad_candidate": True} for i in range(1, 6)]
        train, holdout = _split_by_day(rows, 0.6)
        self.assertEqual([r["local_day"] for r in train], ["2026-05-01", "2026-05-02", "2026-05-03"])
        self.assertEqual([r["local_day"] for r in holdout], ["2026-05-04", "2026-05-05"])

    def test_base_rates(self) -> None:
        rows = [
            {"label_useful_missed_winner": True, "label_top15": True, "label_bad_candidate": False},
            {"label_useful_missed_winner": False, "label_top15": False, "label_bad_candidate": True},
        ]
        self.assertEqual(_base_rates(rows)["useful_rate"], 0.5)

    def test_gate_requires_precision_and_bounded_count(self) -> None:
        item = {"candidate_count": 20, "useful_missed_winners": 5, "useful_precision": 0.25, "top15_precision": 0.4, "bad_ratio": 0.6}
        self.assertTrue(_passes_gate(item))
        item["bad_ratio"] = 0.9
        self.assertFalse(_passes_gate(item))


if __name__ == "__main__":
    unittest.main()
