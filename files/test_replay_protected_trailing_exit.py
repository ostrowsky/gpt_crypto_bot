from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from replay_protected_trailing_exit import ReplayConfig, build_replay


class ProtectedTrailingExitReplayTest(unittest.TestCase):
    def test_estimates_uplift_only_for_eligible_continuation_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "reports"
            reports.mkdir()
            (reports / "signal_quality_2026-05-28_final.json").write_text(
                json.dumps(
                    {
                        "trades": [
                            {
                                "sym": "AAAUSDT",
                                "tf": "15m",
                                "source": "bot",
                                "mode": "strong_trend",
                                "entry_ts": "a",
                                "exit_ts": "b",
                                "pnl_pct": -1.0,
                                "max_favorable_pct": 0.5,
                                "future_favorable_pct": 3.0,
                                "giveback_pct": 1.5,
                                "exit_timing": "early",
                                "exit_reason": "Цена ниже EMA20",
                            },
                            {
                                "sym": "BBBUSDT",
                                "tf": "15m",
                                "source": "bot",
                                "mode": "trend",
                                "entry_ts": "c",
                                "exit_ts": "d",
                                "pnl_pct": 1.0,
                                "max_favorable_pct": 1.2,
                                "future_favorable_pct": 1.1,
                                "giveback_pct": 0.2,
                                "exit_timing": "on_time",
                                "exit_reason": "time",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = build_replay(
                reports_dir=reports,
                cfg=ReplayConfig(days=0, min_continuation_edge_pct=0.75, fractions=(0.5,)),
                output=root / "out.json",
                text_output=root / "out.txt",
            )

            self.assertEqual(report["coverage"]["case_rows"], 2)
            self.assertEqual(report["coverage"]["eligible_rows"], 1)
            self.assertEqual(report["policies"]["baseline"]["total_pnl_pct"], 0.0)
            # Eligible trade moves from -1.0 to +1.0 under protected_50.
            self.assertEqual(report["policies"]["protected_50"]["total_pnl_pct"], 2.0)
            self.assertEqual(report["policies"]["protected_50"]["estimated_total_uplift_pct"], 2.0)
            self.assertEqual(report["decision"], "insufficient_independent_cases_keep_research_only")


if __name__ == "__main__":
    unittest.main()

