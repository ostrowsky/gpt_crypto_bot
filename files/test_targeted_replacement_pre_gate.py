from __future__ import annotations

import unittest

import audit_targeted_replacement_pre_gate as audit


def _row(day: str, pnl: float, leader_delta: float, replacement_delta: float) -> dict:
    return {
        "day": day,
        "ts": day + "T12:00:00Z",
        "replaced_exit_pnl_pct": pnl,
        "leader_delta": leader_delta,
        "replacement_delta_pct": replacement_delta,
    }


class TargetedReplacementPreGateTests(unittest.TestCase):
    def test_policy_uses_only_losing_incumbents_above_threshold(self) -> None:
        cfg = audit.TargetedReplacementConfig(min_train_allowed=1, min_holdout_allowed=1, min_recent_allowed=1)
        rows = [
            _row("2026-05-01", -1.0, 10.0, 1.0),
            _row("2026-05-01", 0.5, 20.0, 2.0),
            _row("2026-05-01", -1.0, 2.0, -1.0),
        ]

        stats = audit._evaluate(rows, 5.0, cfg, min_allowed=1)

        self.assertEqual(stats["allowed_count"], 1)
        self.assertEqual(stats["avg_delta_pct"], 1.0)

    def test_regret_gate_rejects_policy_that_blocks_too_many_winners(self) -> None:
        cfg = audit.TargetedReplacementConfig(min_train_allowed=1, min_holdout_allowed=1, min_recent_allowed=1)
        rows = [
            _row("2026-05-01", -1.0, 10.0, 1.0),
            _row("2026-05-01", 0.5, 20.0, 2.0),
            _row("2026-05-01", 0.5, 20.0, 2.0),
        ]

        stats = audit._evaluate(rows, 5.0, cfg, min_allowed=1)

        self.assertFalse(stats["passed"])
        self.assertFalse(stats["checks"]["regret_rate"])

    def test_threshold_selection_does_not_use_holdout(self) -> None:
        cfg = audit.TargetedReplacementConfig(
            leader_delta_thresholds=(0.0, 10.0),
            train_fraction=0.60,
            purge_days=0,
            recent_days=2,
            min_train_allowed=1,
            min_holdout_allowed=1,
            min_recent_allowed=1,
            max_regret_rate_pct=100.0,
        )
        rows = []
        for day in range(1, 11):
            value = 2.0 if day <= 6 else -2.0
            rows.append(_row(f"2026-05-{day:02d}", -1.0, 15.0, value))
            rows.append(_row(f"2026-05-{day:02d}", -1.0, 5.0, -1.0 if day <= 6 else 3.0))

        report = audit.build_from_rows(rows, cfg=cfg)

        self.assertEqual(report["selected"]["leader_delta_min"], 10.0)
        self.assertTrue(report["selected"]["train"]["passed"])
        self.assertFalse(report["selected"]["holdout"]["passed"])


if __name__ == "__main__":
    unittest.main()
