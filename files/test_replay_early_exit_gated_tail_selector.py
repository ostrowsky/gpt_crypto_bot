from __future__ import annotations

import unittest

import replay_early_exit_gated_tail_selector as gated


def _row(bucket: str, reason: str, baseline: float, tail: float) -> dict:
    return {
        "bucket": bucket,
        "exit_reason_bucket": reason,
        "pnl_pct": baseline,
        "tail50_h10_ema20_cap150_pnl_pct": tail,
    }


class EarlyExitGatedTailSelectorReplayTests(unittest.TestCase):
    def test_oracle_early_exit_allows_early_and_blocks_false_positive(self) -> None:
        rows = [
            _row("early_exits", "weak_signal", 1.0, 3.0),
            _row("false_positive_buys", "weak_signal", -1.0, -3.0),
        ]
        gated._apply_selector(rows, "gate_oracle_early_exit", "tail50_h10_ema20_cap150", gated._selector_fn("gate_oracle_early_exit"))
        summary = gated._gated_summary(rows, "gate_oracle_early_exit")
        self.assertEqual(summary["allowed_total"], 1)
        self.assertEqual(summary["false_positive_allowed_rate_pct"], 0.0)
        self.assertGreater(summary["avg_delta_pct"], 0)

    def test_weak_signal_only_is_cautionary_and_allows_false_positive(self) -> None:
        rows = [
            _row("early_exits", "weak_signal", 1.0, 3.0),
            _row("false_positive_buys", "weak_signal", -1.0, -3.0),
        ]
        gated._apply_selector(rows, "gate_weak_signal_only", "tail50_h10_ema20_cap150", gated._selector_fn("gate_weak_signal_only"))
        summary = gated._gated_summary(rows, "gate_weak_signal_only")
        self.assertEqual(summary["allowed_total"], 2)
        self.assertEqual(summary["false_positive_allowed_rate_pct"], 100.0)
        self.assertLess(summary["avg_delta_pct"], 0.1)

    def test_early_non_ema_break_blocks_ema_cleanup(self) -> None:
        rows = [
            _row("early_exits", "ema_break", -1.0, 2.0),
            _row("early_exits", "weak_signal", 1.0, 2.0),
        ]
        gated._apply_selector(rows, "gate_early_non_ema_break", "tail50_h10_ema20_cap150", gated._selector_fn("gate_early_non_ema_break"))
        self.assertFalse(rows[0]["gate_early_non_ema_break_allowed"])
        self.assertTrue(rows[1]["gate_early_non_ema_break_allowed"])


if __name__ == "__main__":
    unittest.main()
