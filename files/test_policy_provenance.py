from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

import critic_dataset
import config
import ml_candidate_ranker
import monitor
import policy_provenance
import truth_harness


def _market_fixture(bar_ts: int = 1_786_000_000_000):
    size = 30
    data = np.zeros(size, dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")])
    data["t"] = np.arange(size) * 900_000 + bar_ts - (size - 1) * 900_000
    data["o"] = np.linspace(1.0, 1.1, size)
    data["c"] = data["o"] + 0.001
    data["h"] = data["c"] + 0.002
    data["l"] = data["o"] - 0.002
    data["v"] = 100.0
    feat = {
        "ema_fast": data["c"] * 0.99,
        "ema_slow": data["c"] * 0.98,
        "ema200": data["c"] * 0.97,
        "atr": np.full(size, 0.01),
        "slope": np.full(size, 0.2),
        "rsi": np.full(size, 60.0),
        "adx": np.full(size, 25.0),
        "vol_x": np.full(size, 1.5),
        "macd_hist": np.full(size, 0.001),
        "daily_range_pct": np.full(size, 2.0),
    }
    return data, feat, size - 1


def _verified_row(ts: str, epoch: str = "pe-test") -> dict:
    feature = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    label = feature.replace(tzinfo=timezone.utc) + timedelta(hours=2)
    return {
        "id": f"X-{ts}-{epoch}",
        "sym": "XUSDT",
        "tf": "15m",
        "ts_signal": ts,
        "bar_ts": int(feature.timestamp() * 1000),
        "signal_type": "trend",
        "f": {},
        "seq": [],
        "decision": {},
        "labels": {"ret_3": 0.8, "ret_5": 1.0, "ret_10": 1.2},
        "provenance": {
            "policy_epoch": epoch,
            "policy_hash": "hash",
            "dataset_contract": "candidate-outcome-v2",
            "feature_time": policy_provenance.utc_iso(feature),
            "feature_contract": "closed-bar features only",
        },
        "decision_provenance": {
            "policy_epoch": epoch,
            "policy_hash": "hash",
            "decision_time": policy_provenance.utc_iso(feature),
        },
        "label_provenance": {
            "ret_3": {
                "definition": "T+3 return",
                "label_time": policy_provenance.utc_iso(label),
                "recorded_at": policy_provenance.utc_iso(label),
                "source": "test",
            },
            "ret_5": {
                "definition": "T+5 return",
                "label_time": policy_provenance.utc_iso(label),
                "recorded_at": policy_provenance.utc_iso(label),
                "source": "test",
            },
            "ret_10": {
                "definition": "T+10 return",
                "label_time": policy_provenance.utc_iso(label),
                "recorded_at": policy_provenance.utc_iso(label),
                "source": "test",
            },
        },
    }


class PolicyProvenanceTests(unittest.TestCase):
    def test_live_critic_candidate_collection_is_explicitly_enabled(self) -> None:
        self.assertIs(config.CRITIC_DATASET_ENABLED, True)

    def test_duplicate_legacy_candidate_cannot_gain_current_decision_provenance(self) -> None:
        legacy = {
            "id": "legacy",
            "provenance": {},
            "decision_provenance": {},
            "decision": {"action": "candidate", "stage": "collector"},
        }
        changed = policy_provenance.update_decision_provenance(
            legacy,
            {
                "policy_epoch": "pe-current",
                "policy_hash": "current",
                "decision_time": "2026-08-20T10:00:00Z",
                "source": "main_monitor",
            },
        )
        self.assertFalse(changed)
        self.assertEqual(legacy["decision_provenance"], {})

    def test_new_mature_candidate_is_counted_as_verified(self) -> None:
        data, feat, i = _market_fixture()
        with tempfile.TemporaryDirectory() as td, patch.object(
            critic_dataset, "CRITIC_FILE", Path(td) / "critic.jsonl"
        ):
            critic_dataset._logged_candidates.clear()
            rec_id = critic_dataset.log_candidate(
                sym="XUSDT", tf="15m", bar_ts=int(data["t"][i]),
                signal_type="trend", is_bull_day=False, feat=feat, i=i,
                data=data, action="take", stage="entry",
            )
            label_time = policy_provenance.forward_label_time(
                bar_ts=int(data["t"][i]), tf="15m", horizon=10
            )
            with patch(
                "policy_provenance.datetime"
            ) as mocked_datetime:
                mocked_datetime.now.return_value = label_time + timedelta(minutes=1)
                mocked_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
                for horizon in (3, 5, 10):
                    critic_dataset.fill_forward_label(rec_id, horizon, 1.2)
            coverage = ml_candidate_ranker.training_provenance_coverage(
                critic_dataset.CRITIC_FILE
            )

        self.assertEqual(coverage["labeled_rows"], 1)
        self.assertEqual(coverage["verified_rows"], 1)
        self.assertEqual(coverage["legacy_unknown_rows"], 0)

    def test_candidate_keeps_observation_epoch_and_records_decision_epoch_history(self) -> None:
        data, feat, i = _market_fixture()
        first = policy_provenance.build_observation_provenance(
            bar_ts=int(data["t"][i]), tf="15m", source="data_collector"
        )
        second = dict(first, policy_epoch="pe-new", policy_hash="new-hash", source="main_monitor")
        with tempfile.TemporaryDirectory() as td, patch.object(critic_dataset, "CRITIC_FILE", Path(td) / "critic.jsonl"), patch.object(
            policy_provenance, "build_observation_provenance", side_effect=[first, second]
        ):
            critic_dataset._logged_candidates.clear()
            rec_id = critic_dataset.log_candidate(
                sym="XUSDT", tf="15m", bar_ts=int(data["t"][i]), signal_type="trend",
                is_bull_day=False, feat=feat, i=i, data=data, action="candidate", stage="collector",
            )
            critic_dataset.log_candidate(
                sym="XUSDT", tf="15m", bar_ts=int(data["t"][i]), signal_type="trend",
                is_bull_day=False, feat=feat, i=i, data=data, action="take", stage="entry",
            )
            row = critic_dataset.get_record(rec_id)

        self.assertEqual(row["provenance"]["policy_epoch"], first["policy_epoch"])
        self.assertEqual(row["decision_provenance"]["policy_epoch"], "pe-new")
        self.assertEqual(row["decision_provenance_history"][0]["policy_epoch"], first["policy_epoch"])

    def test_forward_label_has_causal_definition_and_immutable_time(self) -> None:
        data, feat, i = _market_fixture()
        with tempfile.TemporaryDirectory() as td, patch.object(critic_dataset, "CRITIC_FILE", Path(td) / "critic.jsonl"):
            critic_dataset._logged_candidates.clear()
            rec_id = critic_dataset.log_candidate(
                sym="XUSDT", tf="15m", bar_ts=int(data["t"][i]), signal_type="trend",
                is_bull_day=False, feat=feat, i=i, data=data, action="take", stage="entry",
            )
            critic_dataset.fill_forward_label(rec_id, 5, 1.2)
            first = critic_dataset.get_record(rec_id)["label_provenance"]["ret_5"]
            critic_dataset.fill_forward_label(rec_id, 5, 1.2)
            second = critic_dataset.get_record(rec_id)["label_provenance"]["ret_5"]

        self.assertIn("T+5", first["definition"])
        self.assertGreater(
            policy_provenance.parse_utc(first["label_time"]),
            policy_provenance.feature_cutoff(int(data["t"][i]), "15m"),
        )
        self.assertEqual(first, second)

    def test_pending_label_waits_for_exact_target_bar_close(self) -> None:
        data, feat, i = _market_fixture()
        bar_ts = int(data["t"][i])
        bar_ms = 900_000
        target_open = bar_ts + 3 * bar_ms
        with tempfile.TemporaryDirectory() as td, patch.object(critic_dataset, "CRITIC_FILE", Path(td) / "critic.jsonl"):
            critic_dataset._logged_candidates.clear()
            rec_id = critic_dataset.log_candidate(
                sym="XUSDT", tf="15m", bar_ts=bar_ts, signal_type="trend",
                is_bull_day=False, feat=feat, i=i, data=data, action="take", stage="entry",
            )
            opens_with_forming_target = np.array([bar_ts, bar_ts + bar_ms, bar_ts + 2 * bar_ms, target_open])
            closes_with_forming_target = np.array([1.0, 1.01, 1.02, 1.03])
            critic_dataset.fill_pending_from_data(
                "XUSDT", "15m", opens_with_forming_target, closes_with_forming_target, bar_ms
            )
            self.assertIsNone(critic_dataset.get_record(rec_id)["labels"]["ret_3"])

            critic_dataset.fill_pending_from_data(
                "XUSDT",
                "15m",
                np.append(opens_with_forming_target, target_open + bar_ms),
                np.append(closes_with_forming_target, 1.04),
                bar_ms,
            )
            row = critic_dataset.get_record(rec_id)

        self.assertAlmostEqual(row["labels"]["ret_3"], 3.0, places=4)
        self.assertIn("ret_3", row["label_provenance"])

    def test_legacy_rows_are_excluded_and_coverage_is_explicit(self) -> None:
        verified = _verified_row("2026-08-13T10:00:00Z")
        legacy = dict(verified, id="legacy", provenance={}, decision_provenance={}, label_provenance={})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "critic.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in (legacy, verified)) + "\n", encoding="utf-8")
            with patch.object(policy_provenance.config, "POLICY_PROVENANCE_REQUIRED_FOR_RANKER", True, create=True):
                rows = ml_candidate_ranker.load_training_rows(path)
                coverage = ml_candidate_ranker.training_provenance_coverage(path)

        self.assertEqual([row["id"] for row in rows], [verified["id"]])
        self.assertEqual(coverage["labeled_rows"], 2)
        self.assertEqual(coverage["verified_rows"], 1)
        self.assertEqual(coverage["legacy_unknown_rows"], 1)

    def test_chronological_split_never_splits_equal_timestamp_group(self) -> None:
        rows = [
            _verified_row("2026-08-13T10:00:00Z", f"pe-{idx}") for idx in range(4)
        ] + [
            _verified_row("2026-08-13T13:00:00Z"),
            _verified_row("2026-08-13T16:00:00Z"),
            _verified_row("2026-08-13T19:00:00Z"),
        ]
        train_end, val_end = ml_candidate_ranker._chronological_group_boundaries(rows)
        groups = [set(row["ts_signal"] for row in part) for part in (rows[:train_end], rows[train_end:val_end], rows[val_end:])]

        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

    def test_purged_split_labels_precede_next_feature_window(self) -> None:
        rows = [
            _verified_row(f"2026-08-{day:02d}T10:00:00Z")
            for day in range(1, 11)
        ]
        bundle = ml_candidate_ranker.build_dataset(rows)

        train_labels = [ml_candidate_ranker._row_max_label_time(row) for row in bundle.meta_train]
        val_features = [ml_candidate_ranker._row_feature_time(row) for row in bundle.meta_val]
        val_labels = [ml_candidate_ranker._row_max_label_time(row) for row in bundle.meta_val]
        test_features = [ml_candidate_ranker._row_feature_time(row) for row in bundle.meta_test]

        self.assertLess(max(train_labels), min(val_features))
        self.assertLess(max(val_labels), min(test_features))
        evaluation = ml_candidate_ranker._evaluation_provenance(
            bundle,
            {"verified_rows": len(rows)},
        )
        self.assertEqual(evaluation["evaluation_scope"], "out_of_sample_time_holdout")
        self.assertEqual(evaluation["cross_split_group_overlap_count"], 0)
        self.assertIn("purged chronological", evaluation["split_method"])

    def test_shadow_legacy_loader_is_explicitly_diagnostic_only(self) -> None:
        import report_candidate_ranker_shadow

        legacy = _verified_row("2026-08-13T10:00:00Z")
        legacy["provenance"] = {}
        legacy["decision_provenance"] = {}
        legacy["label_provenance"] = {}
        peer = dict(legacy, id="peer", sym="YUSDT")
        feature_names = ml_candidate_ranker.safe_feature_names()
        payload = {
            "feature_names": feature_names,
            "scaler_mean": [0.0] * len(feature_names),
            "scaler_scale": [1.0] * len(feature_names),
            "model": {"type": "logistic", "weights": [0.0] * len(feature_names), "bias": 0.0},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "critic.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in (legacy, peer)) + "\n", encoding="utf-8")
            report = report_candidate_ranker_shadow.build_shadow_report(path, payload, top_ns=(1,))

        self.assertEqual(report["rows_total"], 2)
        self.assertEqual(report["evidence_status"], "diagnostic_only")
        self.assertFalse(report["runtime_eligible"])
        self.assertEqual(report["data_provenance"]["verified_rows"], 0)

    def test_runtime_rejects_unproven_model_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.object(monitor, "_RANKER_MODEL_FILE", Path(td) / "ranker.json"), patch.object(
            monitor.config, "POLICY_PROVENANCE_REQUIRED_FOR_RANKER", True, create=True
        ):
            monitor._RANKER_MODEL_FILE.write_text(json.dumps({"payload_version": 3}), encoding="utf-8")
            monitor._RANKER_MODEL_CACHE = None
            self.assertIsNone(monitor._load_ranker_payload())

    def test_harness_accepts_complete_temporal_provenance(self) -> None:
        latest = {
            "evaluation_provenance": {
                "feature_time": "closed bars",
                "label_time": "future bars",
                "label_definition": {"ret_5": "T+5"},
                "evaluation_scope": "out_of_sample_time_holdout",
                "cross_split_group_overlap_count": 0,
                "data_provenance": {"verified_rows": 100},
            }
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = root / ".runtime" / "reports"
            report.mkdir(parents=True)
            (report / "rl_train_latest.json").write_text(json.dumps(latest), encoding="utf-8")
            audit = truth_harness.Audit("full")
            truth_harness.audit_model_provenance(audit, root)

        self.assertFalse(any(f.severity == "error" for f in audit.findings))

    def test_training_readiness_is_fresh_but_does_not_claim_training(self) -> None:
        legacy = _verified_row("2026-08-13T10:00:00Z")
        legacy["provenance"] = {}
        legacy["decision_provenance"] = {}
        legacy["label_provenance"] = {}
        with tempfile.TemporaryDirectory() as td:
            dataset = Path(td) / "critic.jsonl"
            dataset.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
            report = ml_candidate_ranker.build_training_readiness_report(
                dataset,
                min_verified_rows=500,
                generated_at="2026-08-14T12:00:00+00:00",
            )
            dataset_size = dataset.stat().st_size

        self.assertEqual(report["evidence_status"], "blocked_insufficient_provenance")
        self.assertFalse(report["runtime_eligible"])
        self.assertFalse(report["achievement_claimed"])
        self.assertEqual(report["data_provenance"]["labeled_rows"], 1)
        self.assertEqual(report["data_provenance"]["verified_rows"], 0)
        self.assertEqual(
            report["evaluation_provenance"]["evaluation_scope"],
            "not_evaluated_insufficient_provenance",
        )
        self.assertEqual(report["dataset_watermark"]["byte_count"], dataset_size)

    def test_readiness_cli_does_not_write_model_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = root / "critic.jsonl"
            dataset.write_text("", encoding="utf-8")
            model = root / "model.json"
            report = root / "readiness.json"
            argv = [
                "ml_candidate_ranker.py",
                "--dataset",
                str(dataset),
                "--model-out",
                str(model),
                "--report-out",
                str(report),
                "--readiness-only",
                "--min-verified-rows",
                "500",
                "--json",
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                ml_candidate_ranker.main()

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(model.exists())

        self.assertEqual(payload["evidence_status"], "blocked_insufficient_provenance")


if __name__ == "__main__":
    unittest.main()
