from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import learning_progress_report as lpr


class LearningProgressReportTest(unittest.TestCase):
    def test_safe_partial_coverage_does_not_override_developing_verdict(self) -> None:
        latest = lpr.DayMetrics(
            day="2026-05-30",
            watchlist_top_count=15,
            early_pct=100.0,
            coverage_status="partial",
            coverage_reasons=("partial candle coverage 186/206",),
            coverage_assessment="partial_safe_inactive_symbols_only",
            missing_series_count=20,
            missing_symbol_status_counts={"BREAK": 20},
        )
        previous = [
            lpr.DayMetrics(day=f"2026-05-2{i}", watchlist_top_count=15, early_pct=40.0, coverage_status="complete")
            for i in range(3, 10)
        ]
        older = [
            lpr.DayMetrics(day=f"2026-05-1{i}", watchlist_top_count=15, early_pct=20.0, coverage_status="complete")
            for i in range(6, 23)
        ]

        verdict = lpr._verdict(latest, previous, older, {"training": {"last_finished_at": "2026-05-31T06:00:00Z"}})
        alerts = lpr._alerts(latest, {}, {}, [], {})

        self.assertEqual(verdict["label"], "РАЗВИВАЕТСЯ ПО ЦЕЛЕВОЙ МЕТРИКЕ")
        self.assertEqual(alerts[0]["severity"], "warn")
        self.assertIn("partial_safe_inactive_symbols_only", alerts[0]["text"])


    def test_zero_denominator_day_does_not_get_green_developing_verdict(self) -> None:
        latest = lpr.DayMetrics(
            day="2026-06-04",
            watchlist_top_count=0,
            early_pct=0.0,
            coverage_status="partial",
            coverage_assessment="partial_safe_inactive_symbols_only",
        )
        previous = [
            lpr.DayMetrics(day=f"2026-06-0{i}", watchlist_top_count=15, early_pct=60.0, coverage_status="complete")
            for i in range(1, 7)
        ]
        older = [
            lpr.DayMetrics(day=f"2026-05-2{i}", watchlist_top_count=15, early_pct=20.0, coverage_status="complete")
            for i in range(1, 8)
        ]

        verdict = lpr._verdict(latest, previous, older, {"training": {"last_finished_at": "2026-06-05T06:00:00Z"}})
        rolling = lpr._rolling_summary([*older, *previous, latest])

        self.assertEqual(verdict["label"], 'ROLLING УЛУЧШАЕТСЯ, ДЕНЬ НЕИНФОРМАТИВЕН')
        self.assertNotEqual(verdict["label"], "??????????? ?? ??????? ???????")
        self.assertEqual(rolling["early_last7_pct"], 60.0)
        self.assertEqual(rolling["n_last7_top_days"], 6)

    def test_build_report_flags_stale_training_and_low_early_capture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / "reports"
            reports.mkdir()
            (reports / "top_gainer_critic_2026-05-19_final.json").write_text(json.dumps({
                "target_day_local": "2026-05-19",
                "summary": {
                    "watchlist_top_count": 15,
                    "watchlist_top_bought": 9,
                    "watchlist_top_missed": 6,
                    "watchlist_top_early_captured": 2,
                    "watchlist_top_capture_rate_pct": 60.0,
                    "watchlist_top_early_capture_rate_pct": 13.33,
                    "bot_false_positive_buys": 6,
                    "blocked_winner_count": 6,
                    "blocked_winner_symbols": ["ARUSDT"],
                    "blocked_winner_reason_counts": {"agent_mode_disabled": 3},
                },
                "watchlist_top_gainers": [{"symbol": "ARUSDT", "status": "blocked_rule", "reason": "agent mode disabled", "entries_count": 0, "blocked_count": 10}],
            }), encoding="utf-8")
            (reports / "signal_quality_2026-05-19_final.json").write_text(json.dumps({
                "summary": {
                    "miss_rate": 0.9,
                    "false_positive_rate": 0.05,
                    "capture_ratio_at_entry": {"median": 0.1},
                    "exit_efficiency": {"median": -2.0},
                    "giveback_pct": {"median": 1.5},
                },
                "coverage": {"status": "partial", "reasons": ["partial candle coverage"]},
            }), encoding="utf-8")
            status = root / "status.json"
            status.write_text(json.dumps({
                "training": {"last_finished_at": "2026-05-08T00:00:00Z", "last_rows_total": 100},
                "top_gainer_critic": {"last_target_day_local": "2026-05-19"},
                "signal_quality_evaluator": {"last_target_day_local": "2026-05-19"},
            }), encoding="utf-8")
            feedback = root / "feedback.json"
            feedback.write_text(json.dumps({"apply_cooldown_relaxation": True, "reason": "miss pressure"}), encoding="utf-8")
            report = lpr.build_report(reports, status, feedback, ["AR/USDT"], root / "out.json", root / "out.txt")
            text = lpr.render_text(report)
            self.assertIn("Бот — 2026-05-19", text)
            self.assertIn("shadow re-entry", text)
            self.assertIn("shadow tail selector", text)
            self.assertIn("shadow entry admission", text)
            self.assertIn("blocker reward", text)
            self.assertIn("shadow_tail_selector", report)
            self.assertIn("shadow_entry_admission", report)
            self.assertIn("blocker_reward", report)
            self.assertIn("portfolio_replacement", report)
            self.assertTrue(any(a["severity"] == "serious" for a in report["alerts"]))
            self.assertEqual(report["learning_components"]["ranker_training"]["status"], "stale")
            self.assertEqual(report["focus_symbols"][0]["symbol"], "ARUSDT")

    def test_shadow_reentry_summary_is_rendered_and_actionable(self) -> None:
        scorecard = {
            "status": "complete",
            "summary": {
                "alerts_total": 12,
                "labeled_ret5": 10,
                "pending": 2,
                "avg_ret5": 0.42,
                "ret5_positive_rate": 0.6,
            },
        }
        summary = lpr._shadow_reentry_summary(scorecard)
        self.assertEqual(summary["status"], "complete")
        self.assertIn("avg_ret5=+0.42%", summary["detail"])

        actions = lpr._next_actions(
            lpr.DayMetrics(day="2026-05-31", early_pct=30.0, coverage_status="complete"),
            {"training": {"last_finished_at": "2026-06-01T00:00:00Z"}},
            {},
            [],
            scorecard,
        )
        self.assertTrue(any("Shadow re-entry выглядит promising" in x for x in actions))

    def test_shadow_tail_selector_summary_and_actions(self) -> None:
        class FakeReplay:
            @staticmethod
            def build_replay(**_kwargs):
                return {
                    "decision": "advance_selector_to_shadow_observable_tail_selector",
                    "ranked_selectors": [
                        {
                            "name": "selector",
                            "test": {
                                "avg_delta_pct": 0.25,
                                "median_delta_pct": 0.0,
                                "worse_rate_pct": 10.0,
                                "allowed_rate_pct": 20.0,
                                "false_positive_allowed_rate_pct": 0.0,
                            },
                        }
                    ],
                    "files": {"json": "x.json"},
                }

        original = lpr.replay_observable_tail_selector
        try:
            lpr.replay_observable_tail_selector = FakeReplay
            summary = lpr._build_shadow_tail_selector_summary(Path("."))
        finally:
            lpr.replay_observable_tail_selector = original

        self.assertEqual(summary["status"], "passed_shadow_gate")
        self.assertIn("selector", summary["detail"])
        actions = lpr._next_actions(
            lpr.DayMetrics(day="2026-05-31", coverage_status="complete"),
            {"training": {"last_finished_at": "2026-06-01T00:00:00Z"}},
            {},
            [],
            {"summary": {"alerts_total": 10, "labeled_ret5": 10, "avg_ret5": 0.0}},
            summary,
        )
        self.assertTrue(any("Shadow tail selector прошёл replay-gate" in x for x in actions))

    def test_shadow_tail_selector_uses_fresh_cache_without_recomputing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            (reports / "observable_tail_selector_replay_latest.json").write_text(json.dumps({
                "decision": "no_selector_passed_observable_shadow_gate",
                "ranked_selectors": [
                    {
                        "name": "cached_selector",
                        "test": {
                            "avg_delta_pct": 0.09,
                            "median_delta_pct": 0.0,
                            "worse_rate_pct": 26.5,
                            "allowed_rate_pct": 31.0,
                            "false_positive_allowed_rate_pct": 0.0,
                        },
                    }
                ],
            }), encoding="utf-8")

            class FailingReplay:
                @staticmethod
                def build_replay(**_kwargs):
                    raise AssertionError("daily report should use fresh cache")

            original = lpr.replay_observable_tail_selector
            try:
                lpr.replay_observable_tail_selector = FailingReplay
                summary = lpr._build_shadow_tail_selector_summary(reports)
            finally:
                lpr.replay_observable_tail_selector = original

        self.assertEqual(summary["status"], "failed_gate")
        self.assertIn("cached_selector", summary["detail"])

    def test_shadow_entry_admission_summary_and_actions(self) -> None:
        class FakeAdmission:
            @staticmethod
            def build_report(**_kwargs):
                return {
                    "decision": "advance_to_entry_admission_behavior_replay",
                    "best_variant": {
                        "reason_set": "agent_score",
                        "net_reward_pct": 12.5,
                        "top_precision": 0.4,
                        "candidate_count": 20,
                        "top_candidates": 8,
                        "false_candidates": 12,
                    },
                    "files": {"json": "x.json"},
                }

        original = lpr.report_entry_admission_shadow_reward
        try:
            lpr.report_entry_admission_shadow_reward = FakeAdmission
            summary = lpr._build_shadow_entry_admission_summary(Path("."))
        finally:
            lpr.report_entry_admission_shadow_reward = original

        self.assertEqual(summary["status"], "passed_shadow_gate")
        self.assertIn("agent_score", summary["detail"])
        actions = lpr._next_actions(
            lpr.DayMetrics(day="2026-05-31", coverage_status="complete"),
            {"training": {"last_finished_at": "2026-06-01T00:00:00Z"}},
            {},
            [],
            {"summary": {"alerts_total": 10, "labeled_ret5": 10, "avg_ret5": 0.0}},
            {"status": "failed_gate"},
            summary,
        )
        self.assertTrue(any("Entry admission shadow reward положительный" in x for x in actions))

    def test_blocker_reward_summary_and_actions(self) -> None:
        class FakeBlocker:
            @staticmethod
            def build_report(**_kwargs):
                return {
                    "decision": "advance_top_harmful_blockers_to_behavior_replay",
                    "reason_table": [
                        {
                            "reason_code": "score_gate",
                            "net_harm_pct": 12.0,
                            "harm_pct": 14.0,
                            "protection_credit_pct": 2.0,
                            "decision": "advance_to_behavior_replay",
                        }
                    ],
                    "files": {"json": "x.json"},
                }

        original = lpr.report_blocked_winner_causal_reward
        try:
            lpr.report_blocked_winner_causal_reward = FakeBlocker
            summary = lpr._build_blocker_reward_summary(Path("."))
        finally:
            lpr.report_blocked_winner_causal_reward = original

        self.assertEqual(summary["status"], "passed_harm_gate")
        self.assertEqual(summary["evidence_strength"], "strong")
        self.assertIn("score_gate", summary["detail"])
        actions = lpr._next_actions(
            lpr.DayMetrics(day="2026-05-31", coverage_status="complete"),
            {"training": {"last_finished_at": "2026-06-01T00:00:00Z"}},
            {},
            [],
            {"summary": {"alerts_total": 10, "labeled_ret5": 10, "avg_ret5": 0.0}},
            {"status": "failed_gate"},
            {"status": "no_positive_reward"},
            summary,
        )
        self.assertTrue(any("Blocker reward нашёл сильный blocker-кандидат" in x for x in actions))

    def test_blocker_reward_weak_net_harm_action_is_cautious(self) -> None:
        summary = lpr._blocker_reward_summary_from_report(
            {
                "decision": "advance_top_harmful_blockers_to_behavior_replay",
                "reason_table": [
                    {
                        "reason_code": "chase_guard",
                        "net_harm_pct": 0.75,
                        "harm_pct": 125.75,
                        "protection_credit_pct": 125.0,
                        "decision": "advance_to_behavior_replay",
                    }
                ],
            }
        )

        self.assertEqual(summary["status"], "passed_harm_gate")
        self.assertEqual(summary["evidence_strength"], "weak")
        actions = lpr._next_actions(
            lpr.DayMetrics(day="2026-06-11", coverage_status="complete"),
            {"training": {"last_finished_at": "2026-06-12T00:00:00Z"}},
            {},
            [],
            {"summary": {"alerts_total": 0, "labeled_ret5": 0}},
            {"status": "failed_gate"},
            {"status": "no_positive_reward"},
            summary,
        )
        self.assertTrue(any("слабый blocker-кандидат" in x for x in actions))
        self.assertFalse(any("вредный blocker" in x for x in actions))

    def test_portfolio_replacement_summary_and_actions(self) -> None:
        class FakeReplacement:
            @staticmethod
            def build_report(**_kwargs):
                return {
                    "decision": "advance_replacement_policy_to_counterfactual_replay",
                    "summary": {
                        "replacement_count": 12,
                        "closed_incoming_count": 10,
                        "avg_replacement_delta_pct": 0.42,
                        "median_replacement_delta_pct": 0.0,
                        "positive_delta_rate_pct": 60.0,
                    },
                    "policy_simulations": [],
                    "files": {"json": "x.json"},
                }

        original = lpr.report_portfolio_replacement_shadow_reward
        try:
            lpr.report_portfolio_replacement_shadow_reward = FakeReplacement
            summary = lpr._build_portfolio_replacement_summary(Path("."))
        finally:
            lpr.report_portfolio_replacement_shadow_reward = original

        self.assertEqual(summary["status"], "passed_shadow_gate")
        self.assertIn("avg_delta=0.42%", summary["detail"])
        actions = lpr._next_actions(
            lpr.DayMetrics(day="2026-05-31", coverage_status="complete"),
            {"training": {"last_finished_at": "2026-06-01T00:00:00Z"}},
            {},
            [],
            {"summary": {"alerts_total": 10, "labeled_ret5": 10, "avg_ret5": 0.0}},
            {"status": "failed_gate"},
            {"status": "no_positive_reward"},
            {"status": "monitor"},
            summary,
        )
        self.assertTrue(any("Portfolio replacement shadow reward положительный" in x for x in actions))

    def test_portfolio_replacement_policy_candidate_action(self) -> None:
        class FakeReplacement:
            @staticmethod
            def build_report(**_kwargs):
                return {
                    "decision": "replacement_policy_hurting_in_shadow_monitor",
                    "summary": {
                        "replacement_count": 28,
                        "closed_incoming_count": 24,
                        "avg_replacement_delta_pct": -0.44,
                        "median_replacement_delta_pct": -0.53,
                        "positive_delta_rate_pct": 20.8,
                    },
                    "policy_simulations": [
                        {
                            "policy": "block_replaced_non_losing",
                            "kind": "causal",
                            "net_saved_delta_pct": 9.66,
                            "regret_rate_pct": 20.0,
                            "decision": "advance_to_behavior_replay",
                        }
                    ],
                    "files": {"json": "x.json"},
                }

        original = lpr.report_portfolio_replacement_shadow_reward
        try:
            lpr.report_portfolio_replacement_shadow_reward = FakeReplacement
            summary = lpr._build_portfolio_replacement_summary(Path("."))
        finally:
            lpr.report_portfolio_replacement_shadow_reward = original

        self.assertEqual(summary["status"], "policy_candidate")
        self.assertEqual(summary["advanced_policy"]["policy"], "block_replaced_non_losing")
        self.assertIn("candidate=block_replaced_non_losing", summary["detail"])
        actions = lpr._next_actions(
            lpr.DayMetrics(day="2026-05-31", coverage_status="complete"),
            {"training": {"last_finished_at": "2026-06-01T00:00:00Z"}},
            {},
            [],
            {"summary": {"alerts_total": 10, "labeled_ret5": 10, "avg_ret5": 0.0}},
            {"status": "failed_gate"},
            {"status": "no_positive_reward"},
            {"status": "monitor"},
            summary,
        )
        self.assertTrue(any("есть policy-кандидат" in x for x in actions))

    def test_measurement_is_ok_when_critic_is_newer_than_report_day(self) -> None:
        components = lpr._learning_components(
            {
                "training": {"last_finished_at": "2026-05-31T06:00:00Z"},
                "top_gainer_critic": {"last_target_day_local": "2026-05-31"},
                "signal_quality_evaluator": {"last_target_day_local": "2026-05-30"},
            },
            {},
            "2026-05-30",
        )
        self.assertEqual(components["measurement"]["status"], "ok")

    def test_verdict_downgrade_uses_low_confidence_on_small_sparse_denominator(self) -> None:
        latest = lpr.DayMetrics(
            day="2026-06-02",
            watchlist_top_count=3,
            early_pct=0.0,
            coverage_status="partial",
            coverage_assessment="partial_safe_inactive_symbols_only",
        )
        previous = [
            lpr.DayMetrics(day="2026-05-26", watchlist_top_count=5, early_pct=60.0, coverage_status="complete"),
            lpr.DayMetrics(day="2026-05-27", watchlist_top_count=3, early_pct=0.0, coverage_status="complete"),
        ]
        older = [
            lpr.DayMetrics(day="2026-05-18", watchlist_top_count=5, early_pct=60.0, coverage_status="complete"),
            lpr.DayMetrics(day="2026-05-19", watchlist_top_count=5, early_pct=50.0, coverage_status="complete"),
        ]

        verdict = lpr._verdict(latest, previous, older, {"training": {"last_finished_at": "2026-06-03T06:00:00Z"}})

        self.assertEqual(verdict["label"], "УХУДШИЛСЯ ПО EARLY-CAPTURE")
        self.assertEqual(verdict["confidence"], "low")
        self.assertIn("малый denominator", verdict["confidence_reason"])

    def test_verdict_can_degrade_only_with_high_confidence_window(self) -> None:
        latest = lpr.DayMetrics(day="2026-06-02", watchlist_top_count=10, early_pct=10.0, coverage_status="complete")
        previous = [
            lpr.DayMetrics(day=f"2026-05-{day:02d}", watchlist_top_count=10, early_pct=10.0, coverage_status="complete")
            for day in range(26, 32)
        ]
        older = [
            lpr.DayMetrics(day=f"2026-05-{day:02d}", watchlist_top_count=10, early_pct=45.0, coverage_status="complete")
            for day in range(19, 26)
        ]

        verdict = lpr._verdict(latest, previous, older, {"training": {"last_finished_at": "2026-06-03T06:00:00Z"}})

        self.assertEqual(verdict["label"], "ДЕГРАДИРУЕТ")
        self.assertEqual(verdict["confidence"], "high")

    def test_zero_watchlist_denominator_is_not_serious_early_failure(self) -> None:
        latest = lpr.DayMetrics(
            day="2026-05-31",
            watchlist_top_count=0,
            early_pct=0.0,
            coverage_status="complete",
        )
        alerts = lpr._alerts(latest, {}, {}, [], {})
        self.assertFalse(any("early capture only" in item["text"] for item in alerts))
        report = {
            "latest_day": latest.day,
            "latest": latest.__dict__,
            "rolling": {"early_last7_pct": 44.8, "early_prev7_pct": 17.1},
            "verdict": {"emoji": "📈", "label": "РАЗВИВАЕТСЯ", "operator_hint": "наблюдать"},
            "previous_decisions": [],
            "alerts": alerts,
            "learning_components": {},
            "shadow_reentry": {"status": "complete", "detail": "alerts=0"},
            "shadow_tail_selector": {"status": "missing", "detail": "нет отчёта"},
            "shadow_entry_admission": {"status": "missing", "detail": "нет отчёта"},
            "blocker_reward": {"status": "missing", "detail": "нет отчёта"},
            "portfolio_replacement": {"status": "missing", "detail": "нет отчёта"},
            "next_actions": [],
        }
        text = lpr.render_text(report)
        self.assertIn("confidence=unknown", text)
        self.assertIn("метрика дня не применима", text)

    def test_measurement_freshness_falls_back_to_report_files_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            (reports / "top_gainer_critic_2026-05-31_final.json").write_text("{}", encoding="utf-8")
            (reports / "signal_quality_2026-05-31_final.json").write_text("{}", encoding="utf-8")
            components = lpr._learning_components(
                {"training": {"last_finished_at": "2026-06-01T06:00:00Z"}},
                {},
                "2026-05-31",
                reports,
            )
            self.assertEqual(components["measurement"]["status"], "ok")
            self.assertIn("critic=2026-05-31", components["measurement"]["detail"])


class LearningProgressWorkerIntegrationTest(unittest.TestCase):
    def test_rl_worker_has_0900_learning_progress_task(self) -> None:
        src = Path("rl_headless_worker.py").read_text(encoding="utf-8")
        self.assertIn("def _scheduled_learning_progress_slot", src)
        self.assertIn("LEARNING_PROGRESS_DAILY_REPORT_HOUR_LOCAL", src)
        self.assertIn("asyncio.create_task(_learning_progress_loop(state)", src)
        cfg = Path("config.py").read_text(encoding="utf-8")
        self.assertIn("LEARNING_PROGRESS_DAILY_REPORT_HOUR_LOCAL: int = 9", cfg)
        self.assertIn("LEARNING_PROGRESS_DAILY_REPORT_TELEGRAM_ENABLED: bool = True", cfg)

    def test_worker_state_has_learning_progress_fields(self) -> None:
        import rl_headless_worker as worker
        state = worker.WorkerState(train_interval_sec=60, status_interval_sec=60, min_rows=1, min_new_rows=1, collector_enabled=False)
        self.assertTrue(hasattr(state, "learning_progress_enabled"))
        self.assertTrue(hasattr(state, "learning_progress_last_verdict"))


if __name__ == "__main__":
    unittest.main()
