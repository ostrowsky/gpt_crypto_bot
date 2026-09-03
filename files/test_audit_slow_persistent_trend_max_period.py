from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_slow_persistent_trend_max_period as audit


def bars(count: int, *, growth: float = 1.0002) -> np.ndarray:
    data = np.zeros(count, dtype=audit.DTYPE)
    close = 100.0 * np.power(growth, np.arange(count))
    data["t"] = np.arange(count, dtype=np.int64) * audit.HOUR_MS
    data["o"] = close
    data["h"] = close * 1.002
    data["l"] = close * 0.998
    data["c"] = close
    data["v"] = 1000.0
    return data


def signal(index: int = 200) -> audit.Signal:
    return audit.Signal(
        symbol="AAAUSDT",
        profile="balanced_v1",
        bar_index=index,
        bar_open_ts_ms=index * audit.HOUR_MS,
        decision_ts_ms=(index + 1) * audit.HOUR_MS,
        close=100.0,
        features={},
    )


class SlowPersistentTrendCausalityTests(unittest.TestCase):
    def test_incomplete_current_hour_is_excluded(self) -> None:
        rows = [
            [0, "100", "101", "99", "100", "10"],
            [audit.HOUR_MS, "100", "101", "99", "100", "10"],
        ]
        result = audit._rows_to_array(rows, end_ms=2 * audit.HOUR_MS - 1)
        self.assertEqual(list(result["t"]), [0])

    def test_features_do_not_change_when_future_bars_are_appended(self) -> None:
        prefix = bars(260)
        extended = bars(320)
        prefix_features = audit.build_features(prefix)
        extended_features = audit.build_features(extended)
        for name in prefix_features:
            np.testing.assert_allclose(
                prefix_features[name],
                extended_features[name][: len(prefix)],
                equal_nan=True,
                err_msg=name,
            )

    def test_label_enters_on_next_open_and_charges_round_trip_cost(self) -> None:
        data = bars(260, growth=1.0)
        data["o"][201] = 100.0
        data["c"][212] = 101.0
        data["c"][224] = 102.0
        data["c"][236] = 103.0
        data["h"][201:225] = 102.5
        data["l"][201:225] = 99.5
        labeled = audit.label_signal(signal(), data, cost_bps=20.0)
        self.assertIsNotNone(labeled)
        assert labeled is not None
        self.assertEqual(labeled.entry_ts_ms, 201 * audit.HOUR_MS)
        expected_entry = 100.0 * 1.001
        expected_ret24 = ((102.0 * 0.999 / expected_entry) - 1.0) * 100.0
        self.assertAlmostEqual(labeled.entry_price_gross, expected_entry)
        self.assertAlmostEqual(labeled.ret_24h_net_pct, expected_ret24)
        self.assertTrue(labeled.useful)


class SlowPersistentTrendEventTests(unittest.TestCase):
    def test_detector_emits_only_false_to_true_transitions(self) -> None:
        data = bars(260)
        features = {name: np.full(len(data), 1.0) for name in (
            "swing_low_12h_pct",
            "return_12h_pct",
            "return_24h_pct",
            "ema7_slope_3h_pct",
            "ema25_slope_6h_pct",
            "rsi",
            "adx",
            "adx_delta_3h",
            "volume_ratio",
            "ema25_edge_pct",
            "macd_hist",
            "macd_delta_3h",
            "nonnegative_returns_12h",
            "atr_pct",
        )}
        mask = np.zeros(len(data), dtype=bool)
        mask[205:210] = True
        mask[211:214] = True
        mask[240:] = True
        detected = audit.detect_signals("AAAUSDT", data, features, mask, profile_name="test", cooldown_hours=24)
        self.assertEqual([row.bar_index for row in detected], [205, 240])

    def test_embargo_removes_rows_near_both_boundaries(self) -> None:
        bounds = {
            "train_validation_cut_ms": 100 * audit.HOUR_MS,
            "validation_holdout_cut_ms": 200 * audit.HOUR_MS,
            "embargo_ms": 10 * audit.HOUR_MS,
        }
        self.assertEqual(audit.split_name(89 * audit.HOUR_MS, bounds), "train")
        self.assertIsNone(audit.split_name(95 * audit.HOUR_MS, bounds))
        self.assertEqual(audit.split_name(110 * audit.HOUR_MS, bounds), "validation")
        self.assertIsNone(audit.split_name(195 * audit.HOUR_MS, bounds))
        self.assertEqual(audit.split_name(210 * audit.HOUR_MS, bounds), "holdout")


class SlowPersistentTrendEvidenceTests(unittest.TestCase):
    def test_zero_denominator_is_unknown_not_zero_percent(self) -> None:
        result = audit.metrics([], period_start_ms=0, period_end_ms=audit.DAY_MS, base_rows=[])
        self.assertEqual(result["useful_numerator"], 0)
        self.assertEqual(result["labeled_denominator"], 0)
        self.assertIsNone(result["useful_precision_pct"])
        self.assertIsNone(result["base_useful_precision_pct"])
        self.assertIsNone(result["lift_x"])

    def test_profile_selection_uses_eligible_validation_metrics(self) -> None:
        selected = audit.choose_profile(
            {
                "too_small": {"labeled_denominator": 99, "useful_precision_pct": 99.0, "mean_ret_24h_net_pct": 9.0},
                "lower": {"labeled_denominator": 200, "useful_precision_pct": 35.0, "mean_ret_24h_net_pct": 1.0},
                "winner": {"labeled_denominator": 150, "useful_precision_pct": 36.0, "mean_ret_24h_net_pct": 0.5},
            }
        )
        self.assertEqual(selected, "winner")

    def test_acceptance_never_approves_production(self) -> None:
        validation = {
            "labeled_denominator": 150,
            "mean_ret_24h_net_pct": 0.5,
            "median_ret_24h_net_pct": 0.2,
            "useful_precision_pct": 35.0,
        }
        holdout = {
            "labeled_denominator": 250,
            "calendar_days": 200.0,
            "mean_ret_24h_net_pct": 0.4,
            "median_ret_24h_net_pct": 0.1,
            "useful_precision_pct": 34.0,
            "lift_x": 1.4,
            "lift_pp": 6.0,
            "p10_ret_24h_net_pct": -1.5,
            "signals_per_calendar_day": 2.0,
        }
        result = audit.evaluate_shadow_acceptance(
            coverage_ratio=1.0,
            validation=validation,
            holdout=holdout,
            incident_detected_by_deadline=True,
        )
        self.assertEqual(result["decision"], "eligible_for_shadow")
        self.assertFalse(result["production_buy_approved"])
        self.assertFalse(result["production_watch_approved"])
        self.assertFalse(result["actual_bot_candidate_population_validated"])

    def test_adequate_but_failed_evidence_is_rejected(self) -> None:
        validation = {
            "labeled_denominator": 150,
            "mean_ret_24h_net_pct": 0.5,
            "median_ret_24h_net_pct": 0.2,
            "useful_precision_pct": 35.0,
        }
        holdout = {
            "labeled_denominator": 250,
            "calendar_days": 200.0,
            "mean_ret_24h_net_pct": -0.1,
            "median_ret_24h_net_pct": -0.2,
            "useful_precision_pct": 20.0,
            "lift_x": 0.8,
            "lift_pp": -4.0,
            "p10_ret_24h_net_pct": -3.0,
            "signals_per_calendar_day": 2.0,
        }
        result = audit.evaluate_shadow_acceptance(
            coverage_ratio=1.0,
            validation=validation,
            holdout=holdout,
            incident_detected_by_deadline=False,
        )
        self.assertEqual(result["decision"], "rejected")


if __name__ == "__main__":
    unittest.main()
