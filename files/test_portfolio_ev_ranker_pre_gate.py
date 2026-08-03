from __future__ import annotations

import unittest

import audit_portfolio_ev_ranker_pre_gate as audit


def _payload(*, return_delta: float, top_delta: float, capture_delta: float) -> dict:
    rows = []
    for top_n in (1, 3, 5):
        rows.append(
            {
                "top_n": top_n,
                "eligible_groups": 150,
                "baseline": {},
                "ranker": {},
                "delta": {
                    "avg_target_return": return_delta,
                    "win_rate": 0.02,
                    "avg_drawdown": -0.10,
                    "teacher_top_gainer_rate": top_delta,
                    "teacher_capture_ratio": capture_delta,
                },
                "overlap_ratio": 0.5,
            }
        )
    return {
        "rows_total": 1000,
        "train_rows": 600,
        "val_rows": 200,
        "test_rows": 200,
        "chosen_model": "catboost",
        "test_group_ranking": {"grouped_competitions": 150, "top_n": rows},
    }


class PortfolioEvRankerPreGateTests(unittest.TestCase):
    def test_passes_only_when_return_and_north_star_guardrails_pass(self) -> None:
        report = audit.build_report(_payload(return_delta=0.08, top_delta=0.01, capture_delta=0.01))

        self.assertEqual(report["decision"], "advance_to_full_ten_slot_portfolio_replay")
        self.assertTrue(all(item["passed"] for item in report["slices"]))

    def test_rejects_return_uplift_that_loses_top_mover_capture(self) -> None:
        report = audit.build_report(_payload(return_delta=0.08, top_delta=-0.01, capture_delta=-0.02))

        self.assertEqual(report["decision"], "reject_current_ranker_for_capacity_ranking")
        self.assertIn("top1:teacher_top_gainer_rate", report["failed_checks"])
        self.assertIn("top1:teacher_capture_ratio", report["failed_checks"])

    def test_missing_required_slice_is_partial(self) -> None:
        payload = _payload(return_delta=0.08, top_delta=0.01, capture_delta=0.01)
        payload["test_group_ranking"]["top_n"] = payload["test_group_ranking"]["top_n"][:2]

        report = audit.build_report(payload)

        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["decision"], "insufficient_chronological_test_coverage")


if __name__ == "__main__":
    unittest.main()
