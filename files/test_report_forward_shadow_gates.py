from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import report_forward_shadow_gates as gates


def _research_row(symbol: str, ts: str, *, signal: str = "none", candidate: bool = False, ret5: float | None = 1.0, ret10: float | None = 2.0) -> dict:
    row = {
        "sym": symbol,
        "tf": "15m",
        "ts_utc": ts,
        "in_trade_watchlist": True,
        "rule_signal": signal,
        "labels": {"ret_5": ret5, "ret_10": ret10},
    }
    if candidate:
        row["early_trend_shadow"] = {"profile": "frozen_v1", "candidate": True}
    return row


def _event(event: str, ts: str, *, horizon: int | None = None, delta: float = 0.5) -> dict:
    row = {
        "event": event,
        "ts": ts,
        "sym": "AAAUSDT",
        "tf": "15m",
        "selector": "exclude_ema_and_false_cleanup",
        "exit_price": 10.0,
    }
    if horizon is not None:
        row.update({"horizon": horizon, "partial_delta_pct": delta, "tail_return_pct": delta * 2})
    return row


class ForwardShadowGatesTest(unittest.TestCase):
    def test_early_gate_excludes_replay_rows_and_uses_canonical_critic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = root / "research.jsonl"
            metadata = root / "model.json"
            critics = root / "reports"
            critics.mkdir()
            metadata.write_text(json.dumps({"profile": "frozen_v1", "created_at_utc": "2026-08-03T12:00:00Z"}), encoding="utf-8")
            rows = [
                _research_row("OLDUSDT", "2026-08-03T11:00:00Z", candidate=True),
                _research_row("AAAUSDT", "2026-08-03T13:00:00Z", signal="trend"),
                _research_row("AAAUSDT", "2026-08-03T13:15:00Z", candidate=True),
                _research_row("AAAUSDT", "2026-08-03T13:30:00Z", candidate=True, ret5=-5, ret10=-5),
                _research_row("BBBUSDT", "2026-08-03T13:10:00Z", candidate=True),
            ]
            dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n{bad\n", encoding="utf-8")
            critic = {
                "target_day_local": "2026-08-03",
                "phase": "final",
                "exchange_top_gainers": [
                    {"symbol": "AAAUSDT", "in_watchlist": True},
                    {"symbol": "OUTUSDT", "in_watchlist": False},
                ],
            }
            (critics / "top_gainer_critic_2026-08-03_final.json").write_text(json.dumps(critic), encoding="utf-8")

            report = gates._build_early_gate(dataset, metadata, critics, gates.ForwardGateConfig())

        self.assertEqual(report["coverage"]["first_symbol_day_candidates"], 2)
        self.assertEqual(report["coverage"]["mature_both"], 2)
        self.assertEqual(report["coverage"]["malformed_rows_skipped"], 1)
        self.assertEqual(report["north_star"]["canonical_top_movers"], 1)
        self.assertEqual(report["north_star"]["baseline_recall_pct"], 100.0)
        self.assertEqual(report["north_star"]["adjunct_recall_pct"], 100.0)
        self.assertEqual(report["decision"], "collect_forward_labels")

    def test_tail_gate_matches_each_label_to_latest_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.jsonl"
            rows = [
                _event("observable_tail_shadow_candidate", "2026-08-03T10:00:00Z"),
                _event("observable_tail_shadow_label", "2026-08-03T10:30:00Z", horizon=2, delta=0.1),
                _event("observable_tail_shadow_label", "2026-08-03T11:00:00Z", horizon=10, delta=0.5),
                _event("observable_tail_shadow_candidate", "2026-08-04T10:00:00Z"),
                _event("observable_tail_shadow_label", "2026-08-04T11:00:00Z", horizon=10, delta=-0.2),
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            report = gates._build_tail_gate(path, gates.ForwardGateConfig())

        self.assertEqual(report["coverage"]["candidates"], 2)
        self.assertEqual(report["coverage"]["mature_t10"], 2)
        self.assertEqual(report["coverage"]["matched_labels"], 3)
        self.assertEqual(report["metrics"]["avg_partial_delta_10_pct"], 0.15)
        self.assertEqual(report["metrics"]["worse_rate_pct"], 50.0)
        self.assertEqual(report["decision"], "collect_forward_labels")

    def test_tail_gate_can_only_advance_to_full_replay(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.jsonl"
            rows = []
            for day in range(1, 6):
                for idx in range(6):
                    symbol = f"A{day}{idx}USDT"
                    candidate = _event("observable_tail_shadow_candidate", f"2026-08-0{day}T10:{idx:02d}:00Z")
                    candidate["sym"] = symbol
                    label = _event("observable_tail_shadow_label", f"2026-08-0{day}T13:{idx:02d}:00Z", horizon=10, delta=0.4)
                    label["sym"] = symbol
                    rows.extend([candidate, label])
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            report = gates._build_tail_gate(path, gates.ForwardGateConfig())

        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["decision"], "ready_for_full_portfolio_replay_not_production")
        self.assertFalse(report["production_eligible"])

    def test_missing_metadata_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = gates.build_report(
                research_dataset=root / "research.jsonl",
                model_metadata=root / "missing.json",
                bot_events=root / "events.jsonl",
                critics_dir=root,
            )

        self.assertEqual(report["decision"], "measurement_error_fail_closed")
        self.assertFalse(report["production_eligible"])


if __name__ == "__main__":
    unittest.main()
