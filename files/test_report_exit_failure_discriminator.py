from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import report_exit_failure_discriminator as discriminator


class ExitFailureDiscriminatorTest(unittest.TestCase):
    def _write_report(self, root: Path, day: str, rows: list[dict]) -> None:
        (root / f"signal_quality_{day}_final.json").write_text(
            json.dumps({
                "coverage": {"status": "complete", "paired_trades": len(rows), "reasons": []},
                "summary": {"closed_trades": len(rows)},
                "trades": rows,
            }),
            encoding="utf-8",
        )

    def test_build_scores_wrong_exit_continuation_segments(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            weak_bad = {
                "sym": "AAAUSDT",
                "tf": "15m",
                "source": "agent",
                "mode": "trend",
                "entry_ts": "2026-05-20T10:00:00Z",
                "exit_ts": "2026-05-20T11:00:00Z",
                "exit_reason": "WEAK: RSI divergence",
                "pnl_pct": 0.4,
                "max_favorable_pct": 1.0,
                "future_favorable_pct": 3.0,
                "exit_efficiency": 0.2,
                "giveback_pct": 0.8,
                "entry_timing": "ok",
                "exit_timing": "early",
                "trend": {"top_mover_rank": 4},
            }
            weak_good = dict(weak_bad, sym="BBBUSDT", future_favorable_pct=1.1, exit_timing="ok")
            atr_good = dict(weak_bad, sym="CCCUSDT", exit_reason="ATR trail stop", future_favorable_pct=1.0, exit_timing="ok")
            for i, day in enumerate(["2026-05-20", "2026-05-21", "2026-05-22", "2026-05-23"]):
                rows = [dict(weak_bad, sym=f"AAA{i}USDT"), dict(atr_good, sym=f"CCC{i}USDT")]
                if i % 2:
                    rows.append(dict(weak_good, sym=f"BBB{i}USDT"))
                self._write_report(root, day, rows)

            report = discriminator.build(days=0, reports_dir=root, continuation_margin_pct=0.75, min_train_days=2)

            self.assertEqual(report["status"], "ok")
            self.assertGreater(report["summary"]["cases_labeled"], 0)
            self.assertIsNotNone(report["summary"]["test_wrong_exit_rate"])
            self.assertTrue(report["top_train_segments"])
            self.assertIn(report["decision"], {"promising_shadow_segments_only", "inconclusive_or_weak", "research_only"})
            self.assertEqual(report["model_payload"]["feature_contract"], "causal_at_exit_only")
            self.assertFalse(report["model_payload"]["runtime_eligible"])
            self.assertEqual(report["model_payload"]["production_effect"], "none_shadow_only")

    def test_features_exclude_retrospective_day_labels(self) -> None:
        features = discriminator._case_features({
            "exit_reason_bucket": "weak",
            "source": "bot",
            "mode": "trend",
            "tf": "15m",
            "pnl_pct": 0.5,
            "max_favorable_pct": 1.0,
            "giveback_pct": 0.5,
            "exit_efficiency": 0.5,
            "entry_timing": "early",
            "exit_timing": "early",
            "capture_ratio_at_entry": 0.9,
            "top_mover_rank": 1,
        })

        self.assertEqual(set(features), set(discriminator.CAUSAL_FEATURE_NAMES))
        self.assertNotIn("entry_timing", features)
        self.assertNotIn("exit_timing", features)
        self.assertNotIn("capture_bucket", features)
        self.assertNotIn("top_rank_bucket", features)

    def test_online_training_persists_separate_shadow_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / "reports"
            reports.mkdir()
            template = {
                "sym": "ETCUSDT",
                "tf": "15m",
                "source": "bot",
                "mode": "trend",
                "entry_ts": "2026-05-20T10:00:00Z",
                "exit_ts": "2026-05-20T11:00:00Z",
                "exit_reason": "WEAK",
                "pnl_pct": 0.5,
                "max_favorable_pct": 1.0,
                "future_favorable_pct": 3.0,
                "exit_efficiency": 0.5,
                "giveback_pct": 0.5,
            }
            for i, day in enumerate(("2026-05-20", "2026-05-21", "2026-05-22", "2026-05-23")):
                self._write_report(reports, day, [dict(template, sym=f"ETC{i}USDT")])
            report_path = root / "exit_report.json"
            model_path = root / "exit_model.json"

            report = discriminator.train_online_shadow(
                reports_dir=reports, report_path=report_path, model_path=model_path
            )

            self.assertTrue(report_path.exists())
            self.assertTrue(model_path.exists())
            saved_model = json.loads(model_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_model["feature_contract"], "causal_at_exit_only")
            self.assertEqual(saved_model["production_effect"], "none_shadow_only")
            self.assertEqual(report["summary"]["data_through_day"], "2026-05-23")

    def test_empty_reports_are_marked_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = discriminator.build(days=0, reports_dir=Path(td))
            self.assertEqual(report["status"], "empty")
            self.assertEqual(report["summary"]["cases_labeled"], 0)


if __name__ == "__main__":
    unittest.main()
