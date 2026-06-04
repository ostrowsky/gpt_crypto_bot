from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import report_research_universe_shadow_scorecard as scorecard


def _row(symbol: str, ts: str, ret5: float | None, *, signal: str = "trend", inside: bool = False, adx: float = 30, vol_x: float = 2.5, slope: float = 0.4) -> dict:
    return {
        "id": f"{symbol}_15m_{ts}",
        "source": "research_universe_shadow",
        "sym": symbol,
        "tf": "15m",
        "ts_utc": ts,
        "in_trade_watchlist": inside,
        "rule_signal": signal,
        "f": {"adx": adx, "vol_x": vol_x, "slope": slope},
        "labels": {"ret_3": None, "ret_5": ret5, "ret_10": None},
    }


class ResearchUniverseShadowScorecardTest(unittest.TestCase):
    def test_scorecard_finds_research_only_promotion_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = root / "research.jsonl"
            rows = [
                _row("AAAUSDT", "2026-06-01T00:00:00Z", 1.2, signal="trend"),
                _row("AAAUSDT", "2026-06-01T00:15:00Z", 0.4, signal="trend"),
                _row("AAAUSDT", "2026-06-01T00:30:00Z", -0.1, signal="trend"),
                _row("BBBUSDT", "2026-06-01T00:00:00Z", -0.5, signal="breakout"),
                _row("BBBUSDT", "2026-06-01T00:15:00Z", -0.2, signal="breakout"),
                _row("WATCHUSDT", "2026-06-01T00:30:00Z", 0.8, signal="trend", inside=True),
                _row("CCCUSDT", "2026-06-01T00:45:00Z", None, signal="alignment"),
            ]
            dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            report = scorecard.build_scorecard(
                dataset_file=dataset,
                output_json=root / "out.json",
                output_txt=root / "out.txt",
                days=14,
                min_mature=3,
                min_positive_rate_pct=55.0,
                min_avg_ret_pct=0.05,
            )
            self.assertTrue((Path(report["files"]["json"])).exists())

        self.assertEqual(report["coverage"]["rows_in_window"], 7)
        self.assertEqual(report["coverage"]["mature_rows"], 6)
        self.assertEqual(report["coverage"]["immature_rows"], 1)
        self.assertEqual(report["outside_watchlist"]["count"], 5)
        self.assertEqual([row["symbol"] for row in report["promotion_candidates"]], ["AAAUSDT"])
        self.assertEqual(report["recommendation"]["decision"], "advance_candidates_to_replay_gate")
        self.assertIn("Promotion candidates", scorecard.render_text(report))

    def test_scorecard_marks_empty_dataset_as_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = scorecard.build_scorecard(
                dataset_file=root / "missing.jsonl",
                output_json=root / "out.json",
                output_txt=root / "out.txt",
            )

        self.assertEqual(report["recommendation"]["decision"], "insufficient_data")
        self.assertEqual(report["coverage"]["rows_in_window"], 0)


if __name__ == "__main__":
    unittest.main()
