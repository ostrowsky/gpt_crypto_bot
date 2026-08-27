from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import learning_progress_report as lpr


class LearningProgressReportTest(unittest.TestCase):
    def test_rolling_top_rates_are_weighted_by_denominator(self) -> None:
        days = [
            lpr.DayMetrics(day="2026-06-01", watchlist_top_count=1, bought=1, early=1, early_pct=100.0, capture_pct=100.0),
            lpr.DayMetrics(day="2026-06-02", watchlist_top_count=9, bought=3, early=0, early_pct=0.0, capture_pct=33.33),
        ]

        rolling = lpr._rolling_summary(days)

        self.assertEqual(rolling["early_last7_pct"], 10.0)
        self.assertEqual(rolling["capture_last7_pct"], 40.0)

    def test_rolling_comparison_pairs_valid_objective_days_and_exposes_counts(self) -> None:
        older = [
            lpr.DayMetrics(day="2026-08-12", watchlist_top_count=10, early=0, bought=5),
            lpr.DayMetrics(day="2026-08-13", watchlist_top_count=2, early=2, bought=2),
            lpr.DayMetrics(day="2026-08-14", watchlist_top_count=2, early=2, bought=2),
            lpr.DayMetrics(day="2026-08-15", watchlist_top_count=2, early=1, bought=2),
            lpr.DayMetrics(day="2026-08-16", watchlist_top_count=2, early=1, bought=2),
            lpr.DayMetrics(day="2026-08-17", watchlist_top_count=1, early=1, bought=1),
            lpr.DayMetrics(day="2026-08-18", watchlist_top_count=1, early=1, bought=1),
        ]
        recent = [
            lpr.DayMetrics(day="2026-08-19", watchlist_top_count=5, early=4, bought=5),
            lpr.DayMetrics(day="2026-08-20", watchlist_top_count=5, early=4, bought=5),
            lpr.DayMetrics(
                day="2026-08-21",
                watchlist_top_count=5,
                early=0,
                bought=0,
                goal_present=False,
                goal_denominator_consistent=False,
            ),
            lpr.DayMetrics(day="2026-08-22", watchlist_top_count=5, early=4, bought=5),
            lpr.DayMetrics(day="2026-08-23", watchlist_top_count=5, early=4, bought=5),
            lpr.DayMetrics(day="2026-08-24", watchlist_top_count=5, early=4, bought=5),
            lpr.DayMetrics(day="2026-08-25", watchlist_top_count=3, early=2, bought=3),
        ]

        rolling = lpr._rolling_summary([*older, *recent])

        self.assertEqual(rolling["comparison_days_per_window"], 6)
        self.assertEqual(rolling["early_last7_numerator"], 22)
        self.assertEqual(rolling["early_last7_denominator"], 28)
        self.assertEqual(rolling["early_last7_pct"], 78.57)
        self.assertEqual(rolling["early_prev7_numerator"], 8)
        self.assertEqual(rolling["early_prev7_denominator"], 10)
        self.assertEqual(rolling["early_prev7_pct"], 80.0)

    def test_small_observed_drop_without_separated_interval_is_not_directional(self) -> None:
        older = [
            lpr.DayMetrics(day=f"2026-08-{day:02d}", watchlist_top_count=count, early=early, coverage_status="complete")
            for day, count, early in (
                (13, 1, 1), (14, 2, 1), (15, 1, 1),
                (16, 1, 1), (17, 2, 2), (18, 2, 2),
            )
        ]
        recent = [
            lpr.DayMetrics(day=f"2026-08-{day:02d}", watchlist_top_count=count, early=early, coverage_status="complete")
            for day, count, early in (
                (19, 7, 6), (20, 4, 2), (22, 3, 2),
                (23, 5, 5), (24, 6, 5), (25, 3, 2),
            )
        ]
        latest = recent[-1]

        verdict = lpr._verdict(latest, [*older, *recent[:-1]], [], {})
        rolling = lpr._rolling_summary([*older, *recent])

        self.assertEqual(rolling["early_delta_pp"], -10.32)
        self.assertLess(rolling["early_delta_ci95_low_pp"], 0.0)
        self.assertGreater(rolling["early_delta_ci95_high_pp"], 0.0)
        self.assertEqual(verdict["label"], "НЕТ ДОКАЗАННОГО ИЗМЕНЕНИЯ")

    def test_missing_goal_excludes_day_and_makes_latest_objective_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            (reports / "top_gainer_critic_2026-08-25_final.json").write_text(json.dumps({
                "target_day_local": "2026-08-25",
                "summary": {
                    "watchlist_top_count": 3,
                    "watchlist_top_bought": 3,
                    "watchlist_top_early_captured": 2,
                    "watchlist_top_capture_rate_pct": 100.0,
                    "watchlist_top_early_capture_rate_pct": 66.67,
                },
                "watchlist_top_gainers": [],
            }), encoding="utf-8")

            days = lpr._load_day_metrics(reports)
            rolling = lpr._rolling_summary(days)
            verdict = lpr._verdict(days[-1], [], [], {})

        self.assertFalse(days[-1].goal_present)
        self.assertFalse(days[-1].goal_denominator_consistent)
        self.assertIsNone(rolling["early_last7_pct"])
        self.assertEqual(verdict["label"], "СТАТУС НЕПОЛНЫЙ")
        self.assertIn("goal denominator", verdict["operator_hint"])

    def test_latest_entry_and_exit_metrics_use_watchlist_top_cohort_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            critic = {
                "target_day_local": "2026-08-25",
                "summary": {
                    "watchlist_top_count": 2,
                    "watchlist_top_bought": 2,
                    "watchlist_top_early_captured": 1,
                    "watchlist_top_capture_rate_pct": 100.0,
                    "watchlist_top_early_capture_rate_pct": 50.0,
                },
                "watchlist_top_gainers": [
                    {"symbol": "AAAUSDT", "capture_ratio": 0.25, "exit_efficiency": 0.5, "giveback_pct": 1.0},
                    {"symbol": "BBBUSDT", "capture_ratio": 0.75, "exit_efficiency": -0.1, "giveback_pct": 3.0},
                ],
            }
            goal = {
                "target_day_local": "2026-08-25",
                "summary": {"watchlist_top_count": 2},
                # The immutable 22h cohort may differ from the later final critic.
                "watchlist_top_gainers": [{"symbol": "AAAUSDT"}, {"symbol": "CCCUSDT"}],
            }
            signal_quality = {
                "summary": {
                    "capture_ratio_at_entry": {"median": 0.1},
                    "exit_efficiency": {"median": -2.0},
                    "giveback_pct": {"median": 9.0},
                },
                "coverage": {"status": "complete"},
            }
            (reports / "top_gainer_critic_2026-08-25_final.json").write_text(json.dumps(critic), encoding="utf-8")
            (reports / "watchlist_top_gainer_goal_2026-08-25_22h.json").write_text(json.dumps(goal), encoding="utf-8")
            (reports / "signal_quality_2026-08-25_final.json").write_text(json.dumps(signal_quality), encoding="utf-8")

            day = lpr._load_day_metrics(reports)[0]

        self.assertTrue(day.goal_denominator_consistent)
        self.assertEqual(day.top_median_capture_ratio, 0.5)
        self.assertEqual(day.top_capture_sample_count, 2)
        self.assertEqual(day.top_median_exit_efficiency, 0.2)
        self.assertEqual(day.top_exit_sample_count, 2)
        self.assertEqual(day.top_median_giveback_pct, 2.0)
        self.assertEqual(day.median_capture_ratio, 0.1)
        self.assertEqual(day.median_exit_efficiency, -2.0)

    def test_safe_partial_coverage_does_not_override_developing_verdict(self) -> None:
        latest = lpr.DayMetrics(
            day="2026-05-30",
            watchlist_top_count=15,
            early=15,
            early_pct=100.0,
            coverage_status="partial",
            coverage_reasons=("partial candle coverage 186/206",),
            coverage_assessment="partial_safe_inactive_symbols_only",
            missing_series_count=20,
            missing_symbol_status_counts={"BREAK": 20},
        )
        previous = [
            lpr.DayMetrics(day=f"2026-05-2{i}", watchlist_top_count=15, early=6, early_pct=40.0, coverage_status="complete")
            for i in range(3, 10)
        ]
        older = [
            lpr.DayMetrics(day=f"2026-05-1{i}", watchlist_top_count=15, early=3, early_pct=20.0, coverage_status="complete")
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
            lpr.DayMetrics(day=f"2026-06-0{i}", watchlist_top_count=15, early=9, early_pct=60.0, coverage_status="complete")
            for i in range(1, 7)
        ]
        older = [
            lpr.DayMetrics(day=f"2026-05-2{i}", watchlist_top_count=15, early=3, early_pct=20.0, coverage_status="complete")
            for i in range(1, 8)
        ]

        verdict = lpr._verdict(latest, previous, older, {"training": {"last_finished_at": "2026-06-05T06:00:00Z"}})
        rolling = lpr._rolling_summary([*older, *previous, latest])

        self.assertEqual(verdict["label"], 'ROLLING УЛУЧШАЕТСЯ, ДЕНЬ НЕИНФОРМАТИВЕН')
        self.assertNotEqual(verdict["label"], "??????????? ?? ??????? ???????")
        self.assertEqual(rolling["early_last7_pct"], 60.0)
        self.assertEqual(rolling["n_last7_top_days"], 4)

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

    def test_unchanged_ranker_dataset_waits_for_labels_without_false_stale_alert(self) -> None:
        latest = lpr.DayMetrics(day="2026-08-04", watchlist_top_count=1, early=1, early_pct=100.0, coverage_status="complete")
        previous = [
            lpr.DayMetrics(day=f"2026-08-0{day}", watchlist_top_count=4, early=2, early_pct=50.0, coverage_status="complete")
            for day in range(1, 4)
        ]
        older = [
            lpr.DayMetrics(day=f"2026-07-{day:02d}", watchlist_top_count=4, early=3, early_pct=75.0, coverage_status="complete")
            for day in range(25, 32)
        ]
        status = {
            "training": {
                "last_finished_at": "2026-08-03T22:35:22Z",
                "last_rows_total": 22163,
                "last_dataset_mtime": 123.0,
                "min_new_rows": 50,
                "last_error": "",
            },
            "datasets": {"critic_dataset": {"last_write_time": 123.0}},
        }

        state = lpr._ranker_training_state(status, latest.day)
        components = lpr._learning_components(status, {}, latest.day)
        alerts = lpr._alerts(latest, status, {}, [], {})
        actions = lpr._next_actions(latest, status, {}, [], {})
        verdict = lpr._verdict(latest, previous, older, status)
        decisions = lpr._previous_decisions({"apply_cooldown_relaxation": True}, status, latest.day)

        self.assertEqual(state["status"], "waiting_for_new_labels")
        self.assertEqual(components["ranker_training"]["status"], "waiting_for_new_labels")
        self.assertFalse(any("ML ranker training stale" in item["text"] for item in alerts))
        self.assertFalse(any("Починить ML/ranker" in item for item in actions))
        self.assertTrue(any("ждать новых ranker-eligible labels" in item for item in actions))
        self.assertEqual(verdict["label"], "НЕТ ДОКАЗАННОГО ИЗМЕНЕНИЯ")
        self.assertTrue(any(item["status"] == "ожидает новые labels" for item in decisions))
        self.assertIn("аудит score 32–33 завершён", decisions[1]["impact"])

    def test_ranker_reports_verified_cohort_accumulation_instead_of_false_stale(self) -> None:
        latest = lpr.DayMetrics(day="2026-08-13", watchlist_top_count=1, early_pct=20.0, coverage_status="complete")
        status = {
            "training": {
                "last_finished_at": "2026-08-03T22:35:22Z",
                "last_rows_total": 22163,
                "last_dataset_mtime": 123.0,
                "min_rows": 500,
                "provenance": {
                    "labeled_rows": 22163,
                    "verified_rows": 12,
                    "legacy_unknown_rows": 22151,
                },
            },
            "datasets": {"critic_dataset": {"last_write_time": 456.0}},
        }

        state = lpr._ranker_training_state(status, latest.day)
        actions = lpr._next_actions(latest, status, {}, [], {})
        decisions = lpr._previous_decisions({}, status, latest.day)
        alerts = lpr._alerts(latest, status, {}, [], {})

        self.assertEqual(state["status"], "accumulating_verified_cohort")
        self.assertFalse(any("Проверить ML/ranker worker" in item for item in actions))
        self.assertTrue(any("provenance-verified cohort" in item for item in actions))
        self.assertTrue(any(item["status"] == "копит доказательную когорту" for item in decisions))
        self.assertTrue(any("not decision-grade" in item["text"] for item in alerts))

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

    def test_shadow_reentry_uses_registered_counterfactual_labels_when_alerts_are_zero(self) -> None:
        scorecard = {
            "status": "complete",
            "summary": {
                "alerts_total": 0,
                "labeled_ret5": 0,
                "watch_registered": 6,
                "registered_labeled_ret5": 6,
                "registered_avg_ret5": -0.31,
                "registered_ret5_positive_rate": 0.3333,
            },
        }

        summary = lpr._shadow_reentry_summary(scorecard)
        self.assertIn("registered=6, labeled=6", summary["detail"])
        self.assertIn("avg_ret5=-0.31%", summary["detail"])
        self.assertNotIn("данных для оценки re-entry пока нет", summary["detail"])

        latest = lpr.DayMetrics(day="2026-08-02", coverage_status="complete")
        alerts = lpr._alerts(latest, {}, {}, [], scorecard)
        self.assertTrue(any("shadow re-entry noisy" in item["text"] for item in alerts))
        actions = lpr._next_actions(
            latest,
            {"training": {"last_finished_at": "2026-08-03T00:00:00Z"}},
            {},
            [],
            scorecard,
        )
        self.assertTrue(any("registered-watch labels" in item for item in actions))

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
                "provenance": lpr.artifact_provenance.build_provenance(
                    builder="observable_tail_selector_replay_v1",
                    research_config=lpr.replay_observable_tail_selector.ObservableSelectorConfig(),
                ),
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

    def test_shadow_tail_selector_uses_stale_cache_without_recomputing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            cache = reports / "observable_tail_selector_replay_latest.json"
            cache.write_text(json.dumps({
                "decision": "no_selector_passed_observable_shadow_gate",
                "ranked_selectors": [
                    {
                        "name": "stale_selector",
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
            old_time = 1_700_000_000
            import os
            os.utime(cache, (old_time, old_time))

            class FailingReplay:
                @staticmethod
                def build_replay(**_kwargs):
                    raise AssertionError("daily report must not recompute stale heavy replay inline")

            original = lpr.replay_observable_tail_selector
            try:
                lpr.replay_observable_tail_selector = FailingReplay
                summary = lpr._build_shadow_tail_selector_summary(reports)
            finally:
                lpr.replay_observable_tail_selector = original

        self.assertEqual(summary["status"], "stale")
        self.assertEqual(summary["cached_status"], "failed_gate")
        self.assertTrue(summary["stale"])
        self.assertIn("stale cache", summary["detail"])
        self.assertIn("stale_selector", summary["detail"])

    def test_current_mtime_does_not_override_policy_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            provenance = lpr.artifact_provenance.build_provenance(
                builder="observable_tail_selector_replay_v1",
                research_config=lpr.replay_observable_tail_selector.ObservableSelectorConfig(),
            )
            provenance["source_policy_hash"] = "old-policy"
            (reports / "observable_tail_selector_replay_latest.json").write_text(json.dumps({
                "provenance": provenance,
                "decision": "advance_selector_to_shadow_observable_tail_selector",
                "ranked_selectors": [{"name": "selector", "test": {"n": 20, "avg_delta_pct": 1.0}}],
            }), encoding="utf-8")

            summary = lpr._build_shadow_tail_selector_summary(reports)

        self.assertEqual(summary["status"], "stale")
        self.assertEqual(summary["cached_status"], "passed_shadow_gate")
        self.assertIn("source_policy_hash_mismatch", summary["stale_reasons"])

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

    def test_feedback_and_alerts_do_not_claim_broad_episode_metrics_are_daily_top_movers(self) -> None:
        components = lpr._learning_components(
            {},
            {"reason": "top-mover miss pressure: miss_rate=0.85, top_mover_missed=71"},
            "2026-08-26",
        )
        alerts = lpr._alerts(
            lpr.DayMetrics(day="2026-08-26", miss_rate=0.85),
            {},
            {},
            [],
            {},
            {},
            {},
            {},
            {},
        )

        self.assertIn("signal-quality episode pressure", components["feedback"]["detail"])
        self.assertNotIn("top-mover miss pressure", components["feedback"]["detail"])
        self.assertTrue(any("broad signal-quality trend miss-rate" in item["text"] for item in alerts))

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

        self.assertEqual(verdict["label"], "НЕДОСТАТОЧНО ДАННЫХ")
        self.assertEqual(verdict["confidence"], "low")
        self.assertIn("малый denominator", verdict["confidence_reason"])

    def test_verdict_can_degrade_only_with_high_confidence_window(self) -> None:
        latest = lpr.DayMetrics(day="2026-06-02", watchlist_top_count=10, early=1, early_pct=10.0, coverage_status="complete")
        previous = [
            lpr.DayMetrics(day=f"2026-05-{day:02d}", watchlist_top_count=10, early=1, early_pct=10.0, coverage_status="complete")
            for day in range(26, 32)
        ]
        older = [
            lpr.DayMetrics(day=f"2026-05-{day:02d}", watchlist_top_count=10, early=5, early_pct=50.0, coverage_status="complete")
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

    def test_missing_critic_marks_denominator_unknown_not_zero_top_day(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / "reports"
            reports.mkdir()
            (reports / "signal_quality_2026-06-14_final.json").write_text(json.dumps({
                "summary": {
                    "miss_rate": 0.9,
                    "false_positive_rate": 0.2,
                    "capture_ratio_at_entry": {"median": 0.54},
                    "exit_efficiency": {"median": 0.61},
                    "giveback_pct": {"median": 1.0},
                },
                "coverage": {"status": "partial", "reasons": ["partial candle coverage"]},
            }), encoding="utf-8")
            status = root / "status.json"
            status.write_text(json.dumps({"training": {"last_finished_at": "2026-06-15T05:00:00Z"}}), encoding="utf-8")
            feedback = root / "feedback.json"
            feedback.write_text(json.dumps({"reason": "feedback"}), encoding="utf-8")

            report = lpr.build_report(reports, status, feedback, [], root / "out.json", root / "out.txt")
            text = lpr.render_text(report)

            self.assertFalse(report["latest"]["critic_present"])
            self.assertEqual(report["verdict"]["label"], "СТАТУС НЕПОЛНЫЙ")
            self.assertEqual(report["data_confidence"]["status"], "diagnostic_only")
            self.assertEqual(report["data_confidence"]["items"][2]["status"], "unknown")
            self.assertTrue(any("top-gainer critic final missing" in a["text"] for a in report["alerts"]))
            self.assertIn("top-mover denominator недоступен", text)
            self.assertNotIn("watchlist top movers: 0 — метрика дня не применима", text)


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
