from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from phase0_canonical_audit import build_maximum_critic_audit


def _row(
    symbol: str,
    change_pct: float,
    *,
    in_watchlist: bool,
    status: str = "no_signal",
    capture_ratio: float | None = None,
) -> dict:
    close = 100.0 * (1.0 + change_pct / 100.0)
    return {
        "symbol": symbol,
        "day_change_pct": change_pct,
        "day_open": 100.0,
        "day_close": close,
        "quote_volume_24h": 1_000_000.0,
        "in_watchlist": in_watchlist,
        "status": status,
        "capture_ratio": capture_ratio,
    }


def _report(day: str, rows: list[dict], *, summary_override: dict | None = None) -> dict:
    filtered = [row for row in rows if row["in_watchlist"]]
    bought = [row for row in filtered if row["status"] == "bought"]
    early = [row for row in bought if (row.get("capture_ratio") or 0.0) >= 0.35]
    summary = {
        "exchange_top_count": len(rows),
        "exchange_top_in_watchlist": len(filtered),
        "watchlist_top_count": len(filtered),
        "watchlist_top_bought": len(bought),
        "watchlist_top_early_captured": len(early),
        "watchlist_top_missed": len(filtered) - len(bought),
        "watchlist_top_capture_rate_pct": round(len(bought) / len(filtered) * 100.0, 2)
        if filtered
        else 0.0,
        "watchlist_top_early_capture_rate_pct": round(
            len(early) / len(filtered) * 100.0, 2
        )
        if filtered
        else 0.0,
        "watchlist_top_denominator": "exchange_top_filtered_to_watchlist",
    }
    summary.update(summary_override or {})
    return {
        "target_day_local": day,
        "phase": "final",
        "settings": {
            "timezone": "UTC",
            "top_n": len(rows),
            "early_capture_ratio_min": 0.35,
        },
        "summary": summary,
        "exchange_top_gainers": rows,
    }


class MaximumCanonicalCriticAuditTest(unittest.TestCase):
    def _write(self, root: Path, day: str, report: dict, *, prefix: str = "") -> Path:
        path = root / f"top_gainer_critic_{prefix}{day}_final.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_recomputes_real_objective_and_ignores_manual_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(
                root,
                "2026-08-01",
                _report(
                    "2026-08-01",
                    [
                        _row(
                            "AAAUSDT",
                            10.0,
                            in_watchlist=True,
                            status="bought",
                            capture_ratio=0.60,
                        ),
                        _row("BBBUSDT", 8.0, in_watchlist=False),
                    ],
                ),
            )
            self._write(
                root,
                "2026-08-02",
                _report(
                    "2026-08-02",
                    [
                        _row("CCCUSDT", 9.0, in_watchlist=True, status="blocked_rule"),
                        _row("DDDUSDT", 7.0, in_watchlist=False),
                    ],
                ),
            )
            (root / "top_gainer_critic_manual_2026-08-02_final.json").write_text(
                "not json and must be ignored", encoding="utf-8"
            )

            audit = build_maximum_critic_audit(
                root, as_of="2026-08-14T00:00:00+00:00", sesoi=0.05
            )

            self.assertEqual(audit["snapshot"]["discovered_exact_final_files"], 2)
            self.assertEqual(audit["snapshot"]["valid_final_days"], 2)
            self.assertEqual(audit["snapshot"]["calendar_coverage_numerator"], 2)
            self.assertEqual(audit["snapshot"]["calendar_coverage_denominator"], 2)
            self.assertEqual(audit["objective_report"]["numerator"], 1)
            self.assertEqual(audit["objective_report"]["denominator"], 2)
            self.assertEqual(audit["objective_report"]["estimate"], 0.5)
            self.assertEqual(audit["objective_report"]["evidence_status"], "INSUFFICIENT_EVIDENCE")
            self.assertEqual(
                audit["objective_report"]["verdict_rule"],
                "baseline_only_no_directional_verdict",
            )
            self.assertFalse(audit["objective_report"]["verdict_rule_passed"])
            self.assertEqual(audit["objective_contract_errors"], [])
            self.assertEqual(len(audit["labels"]), 2)
            self.assertEqual({label["symbol"] for label in audit["labels"]}, {"AAAUSDT", "CCCUSDT"})
            self.assertEqual(audit["power_report"]["raw_event_count"], 2)
            self.assertFalse(audit["promotion_grade"])

    def test_missing_calendar_days_are_named_partial_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for day in ("2026-08-01", "2026-08-03"):
                self._write(
                    root,
                    day,
                    _report(
                        day,
                        [
                            _row(
                                f"{day[-2:]}AUSDT",
                                10.0,
                                in_watchlist=True,
                                status="bought",
                                capture_ratio=0.5,
                            )
                        ],
                    ),
                )
            audit = build_maximum_critic_audit(
                root, as_of="2026-08-14T00:00:00+00:00", sesoi=0.05
            )
            self.assertEqual(audit["snapshot"]["calendar_coverage_numerator"], 2)
            self.assertEqual(audit["snapshot"]["calendar_coverage_denominator"], 3)
            self.assertEqual(audit["snapshot"]["missing_calendar_days"], ["2026-08-02"])
            self.assertEqual(audit["objective_report"]["coverage_status"], "partial")
            self.assertFalse(audit["measurement_grade"])

    def test_summary_detail_mismatch_excludes_whole_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(
                root,
                "2026-08-01",
                _report(
                    "2026-08-01",
                    [
                        _row(
                            "AAAUSDT",
                            10.0,
                            in_watchlist=True,
                            status="bought",
                            capture_ratio=0.6,
                        )
                    ],
                    summary_override={"watchlist_top_bought": 0},
                ),
            )
            audit = build_maximum_critic_audit(
                root, as_of="2026-08-14T00:00:00+00:00", sesoi=0.05
            )
            self.assertEqual(audit["snapshot"]["valid_final_days"], 0)
            self.assertEqual(len(audit["snapshot"]["invalid_reports"]), 1)
            self.assertIn(
                "summary_mismatch:watchlist_top_bought",
                audit["snapshot"]["invalid_reports"][0]["reasons"],
            )
            self.assertEqual(audit["labels"], [])
            self.assertIsNone(audit["objective_report"]["estimate"])

    def test_zero_watchlist_event_day_is_coverage_not_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(
                root,
                "2026-08-01",
                _report(
                    "2026-08-01", [_row("AAAUSDT", 10.0, in_watchlist=False)]
                ),
            )
            audit = build_maximum_critic_audit(
                root, as_of="2026-08-14T00:00:00+00:00", sesoi=0.05
            )
            self.assertEqual(audit["snapshot"]["valid_final_days"], 1)
            self.assertEqual(audit["objective_report"]["denominator"], 0)
            self.assertIsNone(audit["objective_report"]["estimate"])
            self.assertEqual(audit["power_report"]["status"], "NO_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
