from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, time, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from validate_external_top50_screen import Bar, HOUR_MS

import static_target_top50_shadow as shadow


TZ = ZoneInfo("Europe/Budapest")


def _bar(closes_at: datetime, close: float) -> Bar:
    close_ms = int(closes_at.astimezone(timezone.utc).timestamp() * 1000)
    return Bar(
        open_ts_ms=close_ms - HOUR_MS,
        close=close,
        quote_volume=1_000.0,
        taker_buy_quote=500.0,
    )


def _history(local_day: date, *, observation_base: float, target_base: float, observation: float) -> tuple[Bar, ...]:
    return (
        _bar(datetime.combine(local_day, time(12, 0), TZ), observation_base),
        _bar(datetime.combine(local_day, time(23, 0), TZ), target_base),
        _bar(datetime.combine(local_day.replace(day=local_day.day + 1), time(12, 0), TZ), observation),
    )


class TestObservationContract(unittest.TestCase):
    def test_static_target_prediction_uses_only_closed_observation_bars(self) -> None:
        day = date(2026, 8, 22)
        histories = {
            "TOP1USDT": _history(date(2026, 8, 21), observation_base=100, target_base=100, observation=130),
            "TOP2USDT": _history(date(2026, 8, 21), observation_base=100, target_base=100, observation=120),
            # Current rank is outside Top-2, but the target-aligned denominator
            # makes C the strongest watchlist candidate.
            "CUSDT": _history(date(2026, 8, 21), observation_base=100, target_base=80, observation=105),
            "DUSDT": _history(date(2026, 8, 21), observation_base=100, target_base=95, observation=104),
        }
        observed_at = datetime(2026, 8, 22, 12, 16, tzinfo=TZ)

        payload = shadow.build_observation_payload(
            histories,
            active_symbols=set(histories),
            watchlist={"CUSDT", "DUSDT"},
            watchlist_sha256="watch-hash",
            exchange_info_sha256="exchange-hash",
            local_day=day,
            observed_at=observed_at,
            top_n=2,
            selection_size=1,
            min_market_symbols=4,
            min_watchlist_symbols=2,
        )

        self.assertEqual(payload["status"], "observation_complete")
        self.assertEqual(payload["selections"]["static_target"][0]["symbol"], "CUSDT")
        self.assertEqual(payload["selections"]["current_rank"][0]["symbol"], "CUSDT")
        self.assertEqual(payload["coverage"]["market_valid"], 4)
        self.assertTrue(all("target_return_pct" not in row for row in payload["market_reference"]))
        feature_cutoff = datetime.fromisoformat(payload["timing"]["feature_cutoff_utc"])
        self.assertLessEqual(feature_cutoff, observed_at.astimezone(timezone.utc))
        self.assertEqual(payload["contract"]["formula"], "observation_close / target_minus_24h_close - 1")

    def test_incomplete_observation_is_partial_not_zero_performance(self) -> None:
        day = date(2026, 8, 22)
        histories = {
            "CUSDT": _history(date(2026, 8, 21), observation_base=100, target_base=80, observation=105),
        }
        payload = shadow.build_observation_payload(
            histories,
            active_symbols={"CUSDT", "MISSINGUSDT"},
            watchlist={"CUSDT", "MISSINGUSDT"},
            watchlist_sha256="watch-hash",
            exchange_info_sha256="exchange-hash",
            local_day=day,
            observed_at=datetime(2026, 8, 22, 12, 16, tzinfo=TZ),
            top_n=1,
            selection_size=1,
            min_market_symbols=2,
            min_watchlist_symbols=2,
        )

        self.assertEqual(payload["status"], "partial")
        self.assertFalse(payload["eligible"])
        self.assertNotIn("metrics", payload)


class TestSchedulingAndImmutability(unittest.TestCase):
    def test_runtime_contract_drift_fails_closed(self) -> None:
        with patch.object(shadow.config, "STATIC_TARGET_TOP50_SHADOW_TOP_N", 49):
            with self.assertRaises(shadow.EvidenceContractError):
                shadow.validate_runtime_contract()

    def test_late_worker_records_missed_slot_and_never_backfills_observation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            action = shadow.scheduled_action(
                datetime(2026, 8, 22, 13, 0, tzinfo=TZ),
                reports_dir=Path(td),
                observation_time=time(12, 15),
                observation_grace_minutes=30,
                target_time=time(23, 0),
                label_delay_minutes=5,
            )

        self.assertEqual(action, ("record_missed", date(2026, 8, 22)))

    def test_predeployment_day_is_not_counted_as_a_missed_slot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            action = shadow.scheduled_action(
                datetime(2026, 8, 21, 23, 30, tzinfo=TZ),
                reports_dir=Path(td),
                start_day=date(2026, 8, 22),
            )

        self.assertIsNone(action)

    def test_existing_observation_can_be_finalized_after_target(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            observation = reports / "static_target_top50_shadow_2026-08-22_observation.json"
            observation.write_text("{}", encoding="utf-8")
            action = shadow.scheduled_action(
                datetime(2026, 8, 22, 23, 6, tzinfo=TZ),
                reports_dir=reports,
                observation_time=time(12, 15),
                observation_grace_minutes=30,
                target_time=time(23, 0),
                label_delay_minutes=5,
            )

        self.assertEqual(action, ("finalize", date(2026, 8, 22)))

    def test_irrecoverable_current_observation_preempts_old_label_catchup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            (reports / "static_target_top50_shadow_2026-08-21_observation.json").write_text(
                "{}", encoding="utf-8"
            )
            action = shadow.scheduled_action(
                datetime(2026, 8, 22, 12, 16, tzinfo=TZ),
                reports_dir=reports,
                observation_time=time(12, 15),
                observation_grace_minutes=30,
                target_time=time(23, 0),
                label_delay_minutes=5,
            )

        self.assertEqual(action, ("observe", date(2026, 8, 22)))

    def test_observation_artifact_is_create_once(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "observation.json"
            shadow.write_json_once(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                shadow.write_json_once(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})


class TestForwardLabelsAndScorecard(unittest.TestCase):
    def _observation(self) -> dict:
        return {
            "schema_version": shadow.SCHEMA_VERSION,
            "contract_hash": shadow.CONTRACT_HASH,
            "status": "observation_complete",
            "eligible": True,
            "local_day": "2026-08-22",
            "contract": {"top_n": 2, "selection_size": 1},
            "timing": {
                "observation_utc": "2026-08-22T10:15:00+00:00",
                "feature_cutoff_utc": "2026-08-22T10:00:00+00:00",
                "target_utc": "2026-08-22T21:00:00+00:00",
            },
            "watchlist": {"symbols": ["CUSDT", "DUSDT"]},
            "coverage": {"market_valid": 4, "watchlist_valid": 2},
            "market_reference": [
                {"symbol": "TOP1USDT", "current_rank": 1, "target_base_price": 100.0},
                {"symbol": "TOP2USDT", "current_rank": 2, "target_base_price": 100.0},
                {"symbol": "CUSDT", "current_rank": 3, "target_base_price": 80.0},
                {"symbol": "DUSDT", "current_rank": 4, "target_base_price": 95.0},
            ],
            "candidate_population": {"count": 2},
            "selections": {
                "static_target": [{"symbol": "CUSDT"}],
                "current_rank": [{"symbol": "DUSDT"}],
            },
        }

    def test_labels_are_attached_only_after_target_and_keep_denominators(self) -> None:
        observation = self._observation()
        target = datetime(2026, 8, 22, 23, 0, tzinfo=TZ)
        histories = {
            "TOP1USDT": (_bar(target, 130),),
            "TOP2USDT": (_bar(target, 105),),
            "CUSDT": (_bar(target, 110),),
            "DUSDT": (_bar(target, 100),),
        }
        final = shadow.build_final_payload(
            observation,
            histories,
            labeled_at=datetime(2026, 8, 22, 23, 6, tzinfo=TZ),
            min_market_symbols=4,
        )

        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["target"]["top_symbols"], ["CUSDT", "TOP1USDT"])
        self.assertEqual(final["target"]["entrant_symbols"], ["CUSDT"])
        self.assertEqual(final["metrics"]["static_target"]["topk"], {"hits": 1, "selections": 1})
        self.assertEqual(final["metrics"]["current_rank"]["topk"], {"hits": 0, "selections": 1})
        self.assertEqual(final["metrics"]["static_target"]["entrant_recall"], {"hits": 1, "entrants": 1})

    def test_early_label_attempt_fails_closed(self) -> None:
        with self.assertRaises(shadow.LabelNotMatureError):
            shadow.build_final_payload(
                self._observation(),
                {},
                labeled_at=datetime(2026, 8, 22, 22, 59, tzinfo=TZ),
                min_market_symbols=1,
            )

    def test_missing_selected_target_label_makes_day_partial_not_a_false_hit(self) -> None:
        observation = self._observation()
        target = datetime(2026, 8, 22, 23, 0, tzinfo=TZ)
        histories = {
            "TOP1USDT": (_bar(target, 130),),
            "TOP2USDT": (_bar(target, 105),),
            # CUSDT is the selected static-target symbol and is deliberately
            # missing. It must not become an invented miss.
            "DUSDT": (_bar(target, 100),),
        }
        final = shadow.build_final_payload(
            observation,
            histories,
            labeled_at=datetime(2026, 8, 22, 23, 6, tzinfo=TZ),
            min_market_symbols=3,
        )

        self.assertEqual(final["status"], "partial")
        self.assertFalse(final["eligible"])
        self.assertFalse(final["coverage"]["selected_labels_complete"])
        self.assertNotIn("metrics", final)

    def test_scorecard_never_auto_promotes_and_exposes_calendar_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            for offset, static_hit in enumerate((1, 1), start=22):
                day = f"2026-08-{offset}"
                observation = {
                    "schema_version": shadow.SCHEMA_VERSION,
                    "contract_hash": shadow.CONTRACT_HASH,
                    "local_day": day,
                    "status": "observation_complete",
                    "eligible": True,
                }
                (reports / f"static_target_top50_shadow_{day}_observation.json").write_text(
                    json.dumps(observation), encoding="utf-8"
                )
                final = {
                    "schema_version": shadow.SCHEMA_VERSION,
                    "contract_hash": shadow.CONTRACT_HASH,
                    "local_day": day,
                    "status": "complete",
                    "eligible": True,
                    "timing": {
                        "observation_utc": f"{day}T10:15:00+00:00",
                        "feature_cutoff_utc": f"{day}T10:00:00+00:00",
                        "target_utc": f"{day}T21:00:00+00:00",
                        "labeled_at_utc": f"{day}T21:06:00+00:00",
                        "labels_mature_at_utc": f"{day}T21:05:00+00:00",
                    },
                    "candidate_population": {"hits": 1, "candidates": 2},
                    "metrics": {
                        "static_target": {
                            "top1": {"hits": static_hit, "days": 1},
                            "topk": {"hits": static_hit, "selections": 1},
                            "entrant_recall": {"hits": static_hit, "entrants": 1},
                        },
                        "current_rank": {
                            "top1": {"hits": 0, "days": 1},
                            "topk": {"hits": 0, "selections": 1},
                            "entrant_recall": {"hits": 0, "entrants": 1},
                        },
                    },
                }
                (reports / f"static_target_top50_shadow_{day}_final.json").write_text(
                    json.dumps(final), encoding="utf-8"
                )

            scorecard = shadow.build_scorecard(
                reports,
                as_of_day=date(2026, 8, 23),
                min_eligible_days=2,
                min_observation_coverage=0.9,
                min_final_coverage=0.85,
                bootstrap_samples=200,
            )

        self.assertEqual(scorecard["coverage"]["scheduled_slots"], 2)
        self.assertEqual(scorecard["coverage"]["observed_slots"], 2)
        self.assertEqual(scorecard["coverage"]["eligible_finals"], 2)
        self.assertEqual(scorecard["decision"]["verdict"], "ELIGIBLE_FOR_SEPARATE_PRODUCTION_REVIEW")
        self.assertEqual(scorecard["production_effect"], "none_shadow_only")

    def test_scorecard_fails_on_tampered_observation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            day = "2026-08-22"
            (reports / f"static_target_top50_shadow_{day}_observation.json").write_text(
                json.dumps({
                    "schema_version": shadow.SCHEMA_VERSION,
                    "contract_hash": "tampered",
                    "local_day": day,
                    "status": "observation_complete",
                    "eligible": True,
                }),
                encoding="utf-8",
            )
            scorecard = shadow.build_scorecard(reports, as_of_day=date(2026, 8, 22))

        self.assertEqual(scorecard["decision"]["verdict"], "FAIL")
        self.assertTrue(scorecard["contract_violations"])


if __name__ == "__main__":
    unittest.main()
