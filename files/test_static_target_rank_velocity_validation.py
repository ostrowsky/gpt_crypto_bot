from __future__ import annotations

import copy
import inspect
import unittest
from datetime import date, datetime, timedelta, timezone

import validate_static_target_rank_velocity as rank_velocity
import verify_static_target_rank_velocity as verifier
from validate_external_top50_screen import Bar


HOUR_MS = 3_600_000


def _bar_ending(close_dt: datetime, close: float) -> Bar:
    return Bar(
        open_ts_ms=int(close_dt.timestamp() * 1000) - HOUR_MS,
        close=close,
        quote_volume=100.0,
        taker_buy_quote=50.0,
    )


def _history(
    *,
    observation_price: float,
    prior_price: float,
    target_price: float,
    observation_base: float = 100.0,
    prior_base: float = 100.0,
    target_base: float = 100.0,
) -> tuple[Bar, ...]:
    rows: list[Bar] = []
    cursor = datetime(2026, 8, 19, 7, tzinfo=timezone.utc)
    end = datetime(2026, 8, 20, 23, tzinfo=timezone.utc)
    while cursor <= end:
        rows.append(_bar_ending(cursor, 100.0))
        cursor += timedelta(hours=1)
    by_open = {row.open_ts_ms: row for row in rows}

    def replace(close_dt: datetime, close: float) -> None:
        row = _bar_ending(close_dt, close)
        by_open[row.open_ts_ms] = row

    replace(datetime(2026, 8, 19, 9, tzinfo=timezone.utc), prior_base)
    replace(datetime(2026, 8, 19, 12, tzinfo=timezone.utc), observation_base)
    replace(datetime(2026, 8, 19, 23, tzinfo=timezone.utc), target_base)
    replace(datetime(2026, 8, 20, 9, tzinfo=timezone.utc), prior_price)
    replace(datetime(2026, 8, 20, 12, tzinfo=timezone.utc), observation_price)
    replace(datetime(2026, 8, 20, 23, tzinfo=timezone.utc), target_price)
    return tuple(sorted(by_open.values(), key=lambda row: row.open_ts_ms))


def _histories(*, aaa_target: float = 140.0) -> dict[str, tuple[Bar, ...]]:
    return {
        "MKTUSDT": _history(
            observation_price=130.0,
            prior_price=100.0,
            target_price=131.0,
        ),
        "AAAUSDT": _history(
            observation_price=110.0,
            prior_price=120.0,
            target_price=aaa_target,
        ),
        "BBBUSDT": _history(
            observation_price=115.0,
            prior_price=105.0,
            target_price=110.0,
        ),
    }


def _snapshot(*, aaa_target: float = 140.0) -> rank_velocity.DaySnapshot:
    snapshot = rank_velocity.build_day_snapshot(
        _histories(aaa_target=aaa_target),
        watchlist={"AAAUSDT", "BBBUSDT"},
        local_day=date(2026, 8, 20),
        timezone_name="UTC",
        top_n=1,
        min_market_symbols=3,
        min_watchlist_symbols=2,
        min_candidates=2,
    )
    assert snapshot is not None
    return snapshot


class StaticTargetRankVelocityValidationTests(unittest.TestCase):
    def test_future_target_changes_label_but_not_features(self) -> None:
        first = _snapshot(aaa_target=140.0)
        changed = _snapshot(aaa_target=90.0)

        first_features = [
            (
                row.symbol,
                row.current_return,
                row.static_target_return,
                row.prior_current_return,
                row.current_market_rank,
                row.prior_market_rank,
                row.rank_velocity_3h,
            )
            for row in first.candidates
        ]
        changed_features = [
            (
                row.symbol,
                row.current_return,
                row.static_target_return,
                row.prior_current_return,
                row.current_market_rank,
                row.prior_market_rank,
                row.rank_velocity_3h,
            )
            for row in changed.candidates
        ]
        self.assertEqual(first_features, changed_features)
        self.assertIn("AAAUSDT", first.target_entrant_symbols)
        self.assertNotIn("AAAUSDT", changed.target_entrant_symbols)

    def test_rank_velocity_uses_prior_and_current_market_rank(self) -> None:
        by_symbol = {row.symbol: row for row in _snapshot().candidates}
        self.assertEqual(by_symbol["AAAUSDT"].prior_market_rank, 1)
        self.assertEqual(by_symbol["AAAUSDT"].current_market_rank, 3)
        self.assertEqual(by_symbol["AAAUSDT"].rank_velocity_3h, -2)
        self.assertEqual(by_symbol["BBBUSDT"].prior_market_rank, 2)
        self.assertEqual(by_symbol["BBBUSDT"].current_market_rank, 2)
        self.assertEqual(by_symbol["BBBUSDT"].rank_velocity_3h, 0)

    def test_frozen_percentile_score_is_deterministic_and_bounded(self) -> None:
        candidates = _snapshot().candidates
        first = rank_velocity.rank_candidates(candidates, "static_target_rank_velocity_v1")
        second = rank_velocity.rank_candidates(tuple(reversed(candidates)), "static_target_rank_velocity_v1")
        self.assertEqual(
            [(row.symbol, row.composite_score) for row in first],
            [(row.symbol, row.composite_score) for row in second],
        )
        for row in first:
            self.assertGreaterEqual(row.static_target_percentile, 0.0)
            self.assertLessEqual(row.static_target_percentile, 1.0)
            self.assertGreaterEqual(row.rank_velocity_percentile, 0.0)
            self.assertLessEqual(row.rank_velocity_percentile, 1.0)
            self.assertAlmostEqual(
                row.composite_score,
                row.static_target_percentile + 0.25 * row.rank_velocity_percentile,
            )

    def test_missing_prior_feature_bar_is_coverage_loss(self) -> None:
        histories = _histories()
        missing_close = int(datetime(2026, 8, 19, 9, tzinfo=timezone.utc).timestamp() * 1000)
        histories["AAAUSDT"] = tuple(
            row for row in histories["AAAUSDT"] if row.close_ts_ms != missing_close
        )
        snapshot = rank_velocity.build_day_snapshot(
            histories,
            watchlist={"AAAUSDT", "BBBUSDT"},
            local_day=date(2026, 8, 20),
            timezone_name="UTC",
            top_n=1,
            min_market_symbols=3,
            min_watchlist_symbols=2,
            min_candidates=2,
        )
        self.assertIsNone(snapshot)

    def test_metrics_and_holdout_expose_reconstructable_denominators(self) -> None:
        snapshots = [_snapshot(), _snapshot(), _snapshot()]
        snapshots = [
            rank_velocity.replace_snapshot_day(snapshot, f"2026-08-{20 + index:02d}")
            for index, snapshot in enumerate(snapshots)
        ]
        payload = rank_velocity.normalized_snapshot_payload(
            snapshots,
            provenance={"requested_days": 3, "used_content_hash": "a" * 64},
            watchlist_sha256="b" * 64,
            source_contract={"selection_size": 1},
        )
        contract = rank_velocity.build_contract(
            payload,
            selection_size=1,
            holdout_days=1,
            bootstrap_samples=200,
            minimum_practical_effect=0.02,
        )
        result = rank_velocity.validate_payload(payload, contract)
        full = result["metrics"]["full_period"]
        candidate = full["policies"]["static_target_rank_velocity_v1"]
        self.assertEqual(candidate["topk_selections"], 3)
        self.assertEqual(candidate["entrant_total"], 3)
        self.assertEqual(full["candidate_count"], 6)
        self.assertEqual(full["candidate_base_rate"], 0.5)
        self.assertIsNotNone(candidate["precision_lift_over_base"])
        self.assertEqual(result["metrics"]["holdout_period"]["eligible_days"], 1)

    def test_independent_verifier_rejects_tampered_metrics(self) -> None:
        payload = rank_velocity.normalized_snapshot_payload(
            [_snapshot()],
            provenance={"requested_days": 1, "used_content_hash": "a" * 64},
            watchlist_sha256="b" * 64,
            source_contract={"selection_size": 1},
        )
        contract = rank_velocity.build_contract(
            payload,
            selection_size=1,
            holdout_days=1,
            bootstrap_samples=100,
            minimum_practical_effect=0.02,
        )
        result = rank_velocity.validate_payload(payload, contract)
        self.assertTrue(verifier.verify_payload(payload, contract, result)["valid"])

        tampered = copy.deepcopy(result)
        tampered["metrics"]["full_period"]["paired_daily_precision_delta"] = 0.99
        verification = verifier.verify_payload(payload, contract, tampered)
        self.assertFalse(verification["valid"])
        self.assertIn("metrics", verification["errors"])

    def test_non_decision_grade_snapshot_cannot_be_supported(self) -> None:
        payload = rank_velocity.normalized_snapshot_payload(
            [_snapshot()],
            provenance={"requested_days": 2, "used_content_hash": "a" * 64},
            watchlist_sha256="b" * 64,
            source_contract={"selection_size": 1},
        )
        self.assertFalse(payload["decision_grade_input"])
        contract = rank_velocity.build_contract(
            payload,
            selection_size=1,
            holdout_days=1,
            bootstrap_samples=100,
            minimum_practical_effect=0.02,
        )
        result = rank_velocity.validate_payload(payload, contract)
        self.assertIn("snapshot_not_decision_grade", result["decision"]["reasons"])
        self.assertNotEqual(result["decision"]["verdict"], "SUPPORTED_FOR_FORWARD_SHADOW_ONLY")
        self.assertTrue(verifier.verify_payload(payload, contract, result)["valid"])

    def test_verifier_does_not_import_validator_module(self) -> None:
        source = inspect.getsource(verifier)
        self.assertNotIn("import validate_static_target_rank_velocity", source)
        self.assertNotIn("from validate_static_target_rank_velocity", source)


if __name__ == "__main__":
    unittest.main()
