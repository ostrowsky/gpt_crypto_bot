from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import report_portfolio_replacement_shadow_reward as rpr


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class PortfolioReplacementShadowRewardTests(unittest.TestCase):
    def test_matches_replacement_entry_exit_and_computes_delta(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = root / "files"
            reports = root / ".runtime" / "reports"
            files.mkdir(parents=True)
            reports.mkdir(parents=True)
            _write_jsonl(
                files / "agent_events.jsonl",
                [
                    {
                        "event": "exit",
                        "ts": "2026-05-31T10:00:00Z",
                        "sym": "OLDUSDT",
                        "pnl_pct": -0.4,
                        "reason": "portfolio replacement: NEWUSDT leader 45.5 > OLDUSDT leader 22.0",
                    },
                    {"event": "entry", "ts": "2026-05-31T10:05:00Z", "sym": "NEWUSDT", "price": 1.0},
                    {"event": "exit", "ts": "2026-05-31T11:00:00Z", "sym": "NEWUSDT", "pnl_pct": 1.2, "reason": "regular exit"},
                ],
            )
            (reports / "top_gainer_critic_2026-05-31_final.json").write_text(
                json.dumps(
                    {
                        "target_day_local": "2026-05-31",
                        "watchlist_top_gainers": [
                            {"symbol": "NEWUSDT", "status": "bought", "capture_ratio_at_entry": 0.35}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = rpr.build_report(
                files_dir=files,
                reports_dir=reports,
                cfg=rpr.ReplacementConfig(min_closed_cases=1, min_avg_delta_pct=0.1),
                save=False,
            )

        self.assertEqual(report["coverage"]["replacement_events"], 1)
        self.assertEqual(report["coverage"]["closed_incoming"], 1)
        self.assertEqual(report["summary"]["avg_replacement_delta_pct"], 1.6)
        self.assertEqual(report["summary"]["incoming_watchlist_top_count"], 1)
        self.assertEqual(report["segments"][0]["segment"], "all")
        self.assertEqual(report["segments"][0]["avg_delta_pct"], 1.6)
        self.assertIn("policy_simulations", report)
        self.assertEqual(report["decision"], "advance_replacement_policy_to_counterfactual_replay")

    def test_collects_more_when_closed_cases_are_insufficient(self) -> None:
        summary = {
            "closed_incoming_count": 1,
            "avg_replacement_delta_pct": 5.0,
            "median_replacement_delta_pct": 5.0,
        }
        decision = rpr._decision(summary, rpr.ReplacementConfig(min_closed_cases=10))
        self.assertEqual(decision, "collect_more_replacement_outcomes")

    def test_marks_policy_hurting_when_enough_cases_have_negative_average(self) -> None:
        summary = {
            "closed_incoming_count": 10,
            "avg_replacement_delta_pct": -0.25,
            "median_replacement_delta_pct": 0.0,
        }
        decision = rpr._decision(summary, rpr.ReplacementConfig(min_closed_cases=10))
        self.assertEqual(decision, "replacement_policy_hurting_in_shadow_monitor")

    def test_render_text_contains_key_metrics(self) -> None:
        text = rpr.render_text(
            {
                "coverage": {"events_loaded": 3, "replacement_events": 1, "closed_incoming": 1},
                "decision": "collect_more_replacement_outcomes",
                "summary": {
                    "avg_replacement_delta_pct": 0.1,
                    "median_replacement_delta_pct": 0.0,
                    "positive_delta_rate_pct": 50.0,
                    "avg_replaced_exit_pnl_pct": -0.1,
                    "avg_incoming_exit_pnl_pct": 0.0,
                    "incoming_watchlist_top_count": 1,
                },
                "segments": [
                    {
                        "segment": "all",
                        "closed_count": 1,
                        "avg_delta_pct": 0.1,
                        "median_delta_pct": 0.0,
                        "positive_delta_rate_pct": 100.0,
                    }
                ],
                "policy_simulations": [
                    {
                        "policy": "block_replaced_non_losing",
                        "blocked_count": 1,
                        "net_saved_delta_pct": 0.1,
                        "regret_rate_pct": 0.0,
                        "decision": "collect_more_cases",
                    }
                ],
            }
        )
        self.assertIn("Portfolio replacement shadow reward", text)
        self.assertIn("avg_delta=0.1%", text)
        self.assertIn("segments:", text)
        self.assertIn("policy simulations:", text)

    def test_segment_table_surfaces_harmful_non_losing_replacements(self) -> None:
        rows = [
            {
                "replacement_delta_pct": -2.0,
                "replaced_exit_pnl_pct": 0.5,
                "incoming_exit_pnl_pct": -1.5,
                "leader_delta": 8.0,
                "incoming_watchlist_top": False,
            },
            {
                "replacement_delta_pct": 1.0,
                "replaced_exit_pnl_pct": -0.5,
                "incoming_exit_pnl_pct": 0.5,
                "leader_delta": 22.0,
                "incoming_watchlist_top": True,
            },
        ]
        segments = {row["segment"]: row for row in rpr._segment_table(rows)}
        self.assertEqual(segments["replaced_non_losing_at_rotation"]["avg_delta_pct"], -2.0)
        self.assertEqual(segments["incoming_watchlist_top"]["avg_delta_pct"], 1.0)

    def test_policy_simulation_promotes_causal_non_losing_block_when_saved_delta_is_high(self) -> None:
        rows = []
        for idx in range(5):
            rows.append(
                {
                    "replacement_delta_pct": -0.5,
                    "replaced_exit_pnl_pct": 0.1,
                    "incoming_exit_pnl_pct": -0.4,
                    "leader_delta": 8.0 + idx,
                    "incoming_watchlist_top": False,
                }
            )
        policies = {row["policy"]: row for row in rpr._policy_simulations(rows)}
        policy = policies["block_replaced_non_losing"]
        self.assertEqual(policy["net_saved_delta_pct"], 2.5)
        self.assertEqual(policy["decision"], "advance_to_behavior_replay")

    def test_policy_simulation_marks_future_label_rule_diagnostic_only(self) -> None:
        rows = [
            {
                "replacement_delta_pct": -1.0,
                "replaced_exit_pnl_pct": -0.1,
                "incoming_exit_pnl_pct": -1.1,
                "leader_delta": 25.0,
                "incoming_watchlist_top": False,
            }
        ]
        policies = {row["policy"]: row for row in rpr._policy_simulations(rows)}
        self.assertEqual(policies["block_incoming_not_watchlist_top"]["decision"], "diagnostic_only_future_label")


if __name__ == "__main__":
    unittest.main()
