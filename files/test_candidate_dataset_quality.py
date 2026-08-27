from __future__ import annotations

import json
import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np

import critic_dataset
import config
import data_collector
import ml_candidate_ranker
import policy_provenance


CONTRACT = "candidate-outcome-v2"


def _row(index: int, *, action: str, teacher_top: bool = False) -> dict:
    feature = datetime(2026, 8, 1, tzinfo=timezone.utc) + timedelta(hours=(index // 3) * 6)
    label_time = feature + timedelta(hours=2)
    ts = policy_provenance.utc_iso(feature - timedelta(minutes=15))
    ret5 = 1.0 if index % 2 else -0.5
    row = {
        "id": f"R-{index}",
        "sym": f"X{index % 20}USDT",
        "tf": "15m",
        "ts_signal": ts,
        "bar_ts": int((feature - timedelta(minutes=15)).timestamp() * 1000),
        "signal_type": "trend" if index % 2 else "retest",
        "f": {"slope": 0.1 + index / 1000, "rsi": 50 + index % 10, "adx": 20 + index % 20, "vol_x": 1.0 + index % 5},
        "seq": [],
        "decision": {"action": action, "candidate_score": float(index % 50)},
        "labels": {
            "ret_3": ret5 * 0.8,
            "ret_5": ret5,
            "ret_10": ret5 * 1.2,
            "label_3": ret5 > 0,
            "label_5": ret5 > 0,
            "label_10": ret5 > 0,
            "trade_taken": action == "take",
        },
        "provenance": {
            "policy_epoch": "pe-quality-v2",
            "policy_hash": "hash",
            "feature_time": policy_provenance.utc_iso(feature),
            "feature_contract": "closed-bar features only",
            "dataset_contract": CONTRACT,
        },
        "decision_provenance": {
            "policy_epoch": "pe-quality-v2",
            "policy_hash": "hash",
            "decision_time": policy_provenance.utc_iso(feature),
        },
        "label_provenance": {
            "ret_3": {
                "definition": "T+3 return",
                "label_time": policy_provenance.utc_iso(label_time),
                "recorded_at": policy_provenance.utc_iso(label_time),
                "source": "test",
            },
            "ret_5": {
                "definition": "T+5 return",
                "label_time": policy_provenance.utc_iso(label_time),
                "recorded_at": policy_provenance.utc_iso(label_time),
                "source": "test",
            },
            "ret_10": {
                "definition": "T+10 return",
                "label_time": policy_provenance.utc_iso(label_time),
                "recorded_at": policy_provenance.utc_iso(label_time),
                "source": "test",
            },
        },
    }
    if index < 30:
        row["teacher"] = {
            "final": {
                "phase": "final",
                "watchlist_top_gainer": teacher_top,
                "capture_ratio": 0.7 if teacher_top else 0.0,
            }
        }
        row["label_provenance"]["teacher.final"] = {
            "definition": "final teacher",
            "label_time": policy_provenance.utc_iso(label_time),
            "recorded_at": policy_provenance.utc_iso(label_time),
            "source": "test",
        }
    return row


def _minimal_market_fixture() -> tuple[np.ndarray, dict]:
    data = np.zeros(
        2,
        dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")],
    )
    for name in ("o", "h", "l", "c"):
        data[name] = 1.0
    data["v"] = 100.0
    feat = {
        "ema_fast": np.ones(2), "ema_slow": np.ones(2), "ema200": np.ones(2),
        "atr": np.ones(2) * 0.01, "slope": np.ones(2) * 0.1,
        "rsi": np.ones(2) * 50.0, "adx": np.ones(2) * 25.0,
        "vol_x": np.ones(2), "macd_hist": np.zeros(2),
        "daily_range_pct": np.ones(2),
    }
    return data, feat


class CandidateDatasetQualityTests(unittest.TestCase):
    def test_legacy_ml_dataset_collection_is_disabled_by_default(self) -> None:
        self.assertFalse(config.LEGACY_ML_DATASET_COLLECTION_ENABLED)

    def test_collector_does_not_mutate_legacy_ml_stream_by_default(self) -> None:
        size = 35
        data = np.zeros(
            size,
            dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")],
        )
        data["t"] = np.arange(size, dtype=np.int64) * 900_000 + 1_786_000_000_000
        for name in ("o", "h", "l", "c"):
            data[name] = np.linspace(1.0, 1.1, size)
        data["v"] = 100.0
        _, feat = _minimal_market_fixture()
        feat = {name: np.resize(values, size) for name, values in feat.items()}
        batches: list[dict] = []
        with patch.object(
            data_collector, "fetch_klines", new=AsyncMock(return_value=data)
        ), patch.object(
            data_collector, "compute_features", return_value=feat
        ), patch.object(
            data_collector, "_detect_rule_signal", return_value="none"
        ), patch.object(
            data_collector.ml_dataset, "log_bar_snapshot"
        ) as log_snapshot, patch.object(
            data_collector.ml_dataset, "fill_pending_from_data"
        ) as fill_legacy:
            ok = asyncio.run(
                data_collector._process_coin(
                    None, "TESTUSDT", "15m", False, 0.0,
                    critic_label_batches=batches,
                )
            )
        self.assertTrue(ok)
        log_snapshot.assert_not_called()
        fill_legacy.assert_not_called()
        self.assertEqual(len(batches), 1)

    def test_current_contract_uses_dedicated_v2_stream(self) -> None:
        self.assertEqual(critic_dataset.CRITIC_FILE.name, "critic_dataset_v2.jsonl")
        self.assertEqual(critic_dataset.LEGACY_CRITIC_FILE.name, "critic_dataset.jsonl")

    def test_strict_collector_append_raises_instead_of_counting_lost_row(self) -> None:
        data, feat = _minimal_market_fixture()
        with patch.object(critic_dataset, "_append", side_effect=TimeoutError("locked")):
            with self.assertRaises(critic_dataset.DatasetIntegrityError):
                critic_dataset.log_candidate(
                    sym="FAILUSDT", tf="15m", bar_ts=1_786_000_000_000,
                    signal_type="trend", is_bull_day=False, feat=feat, i=0,
                    data=data, action="candidate", stage="collector", strict=True,
                )

    def test_best_effort_monitor_append_returns_empty_id_on_write_loss(self) -> None:
        data, feat = _minimal_market_fixture()
        with patch.object(critic_dataset, "_append", side_effect=TimeoutError("locked")):
            record_id = critic_dataset.log_candidate(
                sym="BESTUSDT", tf="15m", bar_ts=1_786_000_000_000,
                signal_type="trend", is_bull_day=False, feat=feat, i=0,
                data=data, action="candidate", stage="monitor",
            )
        self.assertEqual(record_id, "")

    def test_candidate_wide_maturation_labels_blocked_and_shadow_rows(self) -> None:
        size = 12
        bar_ms = 900_000
        first = 1_786_000_000_000
        t_arr = np.arange(size, dtype=np.int64) * bar_ms + first
        c_arr = np.linspace(1.0, 1.11, size)
        data = np.zeros(size, dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")])
        data["t"], data["o"], data["h"], data["l"], data["c"], data["v"] = t_arr, c_arr, c_arr, c_arr, c_arr, 100
        feat = {"ema_fast": c_arr, "ema_slow": c_arr, "ema200": c_arr, "atr": c_arr * 0 + 0.01, "slope": c_arr * 0 + 0.2, "rsi": c_arr * 0 + 60, "adx": c_arr * 0 + 25, "vol_x": c_arr * 0 + 1.5, "macd_hist": c_arr * 0 + 0.01, "daily_range_pct": c_arr * 0 + 2}
        with tempfile.TemporaryDirectory() as td, patch.object(critic_dataset, "CRITIC_FILE", Path(td) / "critic.jsonl"):
            critic_dataset._logged_candidates.clear()
            ids = []
            for action in ("blocked", "shadow"):
                ids.append(critic_dataset.log_candidate(sym=f"{action}USDT", tf="15m", bar_ts=int(t_arr[0]), signal_type="trend", is_bull_day=False, feat=feat, i=0, data=data, action=action, stage="test"))
            for action in ("blocked", "shadow"):
                critic_dataset.fill_pending_from_data(f"{action}USDT", "15m", t_arr, c_arr, bar_ms)
            rows = critic_dataset.get_records(set(ids))
        self.assertTrue(all(rows[row_id]["labels"]["ret_5"] is not None for row_id in ids))

    def test_batch_maturation_uses_one_dataset_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.object(
            critic_dataset, "CRITIC_FILE", Path(td) / "critic.jsonl"
        ), patch.object(critic_dataset, "_rewrite_records") as rewrite:
            critic_dataset.CRITIC_FILE.write_text("{}\n", encoding="utf-8")
            critic_dataset.fill_pending_batch(
                [
                    {
                        "sym": "AUSDT",
                        "tf": "15m",
                        "t_arr": np.array([1, 2]),
                        "c_arr": np.array([1.0, 1.1]),
                        "bar_ms": 900_000,
                    },
                    {
                        "sym": "BUSDT",
                        "tf": "1h",
                        "t_arr": np.array([1, 2]),
                        "c_arr": np.array([2.0, 2.1]),
                        "bar_ms": 3_600_000,
                    },
                ]
            )
        rewrite.assert_called_once()

    def test_old_contract_is_not_training_eligible(self) -> None:
        row = _row(1, action="take")
        row["provenance"].pop("dataset_contract")
        self.assertFalse(ml_candidate_ranker.training_row_provenance_valid(row))

    def test_current_contract_pending_row_is_not_reported_as_legacy(self) -> None:
        row = _row(1, action="blocked")
        row["labels"]["ret_10"] = None
        row["label_provenance"].pop("ret_10")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "critic.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            coverage = ml_candidate_ranker.training_provenance_coverage(path)
        self.assertEqual(coverage["legacy_unknown_rows"], 0)
        self.assertEqual(coverage["excluded_contract_rows"], 0)
        self.assertEqual(coverage["incomplete_current_contract_rows"], 1)

    def test_quality_preflight_rejects_take_only_selection_bias(self) -> None:
        rows = [_row(i, action="take", teacher_top=i < 5) for i in range(120)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "critic.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = ml_candidate_ranker.candidate_dataset_quality_report(path, min_rows=120)
        self.assertFalse(report["quality_passed"])
        self.assertIn("action_diversity", report["blocking_checks"])

    def test_readiness_never_approves_a_count_only_take_cohort(self) -> None:
        rows = [_row(i, action="take", teacher_top=i < 5) for i in range(120)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "critic.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = ml_candidate_ranker.build_training_readiness_report(
                path, min_verified_rows=120
            )
        self.assertEqual(report["evidence_status"], "blocked_dataset_quality")
        self.assertEqual(
            report["evaluation_provenance"]["evaluation_scope"],
            "not_evaluated_dataset_quality",
        )
        self.assertFalse(report["training_eligible"])

    def test_quality_preflight_accepts_mature_multi_action_micro_cohort(self) -> None:
        actions = ("take", "blocked", "shadow")
        rows = [_row(i, action=actions[i % 3], teacher_top=i < 10) for i in range(120)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "critic.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = ml_candidate_ranker.candidate_dataset_quality_report(path, min_rows=120)
        self.assertTrue(report["quality_passed"], report)
        self.assertEqual(report["mature_rows"], 120)
        self.assertGreaterEqual(report["decision_groups"]["multi_candidate"], 10)


if __name__ == "__main__":
    unittest.main()
