from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import replay_partial_exit_after_weak_sell as partial


DAY = "2026-05-31"
BASE_TS = 1_780_000_000_000
BAR = 15 * 60 * 1000


def _row(**overrides) -> dict:
    row = {
        "sym": "AAAUSDT",
        "tf": "15m",
        "source": "bot",
        "mode": "trend",
        "entry_ts": "2026-05-31T08:00:00Z",
        "exit_ts": BASE_TS,
        "entry_price": 100.0,
        "exit_price": 101.0,
        "exit_reason": "WEAK: RSI divergence",
        "pnl_pct": 1.0,
        "max_favorable_pct": 3.0,
        "future_favorable_pct": 5.0,
        "exit_efficiency": 0.33,
        "giveback_pct": 2.0,
        "exit_timing": "early",
    }
    row.update(overrides)
    return row


def _write_report(root: Path, row: dict) -> None:
    (root / f"signal_quality_{DAY}_final.json").write_text(json.dumps({"early_exits": [row]}), encoding="utf-8")


def _write_cache(root: Path, closes: list[float]) -> None:
    rows = []
    for i, close in enumerate(closes):
        t = BASE_TS + (i - 1) * BAR
        rows.append({"t": t, "o": close, "h": close * 1.01, "l": close * 0.99, "c": close, "v": 1000})
    (root / f"AAAUSDT_15m_{BASE_TS - BAR}_{BASE_TS + len(closes) * BAR}.json").write_text(json.dumps(rows), encoding="utf-8")


class PartialExitAfterWeakSellReplayTests(unittest.TestCase):
    def test_partial_policy_is_between_baseline_and_full_hold(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / "reports"; reports.mkdir()
            cache = root / "cache"; cache.mkdir()
            _write_report(reports, _row())
            _write_cache(cache, [101, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111])

            payload = partial.build_replay(reports_dir=reports, cache_dir=cache, cfg=partial.PartialExitConfig(days=14), save=False)
            p50 = payload["policies"]["partial_50_hold_5"]
            p70 = payload["policies"]["partial_70_hold_5"]

            self.assertGreater(p50["avg_delta_pct"], 0)
            self.assertGreater(p50["avg_delta_pct"], p70["avg_delta_pct"])
            self.assertAlmostEqual(p50["avg_pnl_pct"], 3.5, places=4)

    def test_partial_policy_reduces_harm_vs_full_tail_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / "reports"; reports.mkdir()
            cache = root / "cache"; cache.mkdir()
            _write_report(reports, _row())
            _write_cache(cache, [101, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91])

            payload = partial.build_replay(reports_dir=reports, cache_dir=cache, cfg=partial.PartialExitConfig(days=14), save=False)
            p50 = payload["policies"]["partial_50_hold_5"]
            p70 = payload["policies"]["partial_70_hold_5"]

            self.assertLess(p50["avg_delta_pct"], 0)
            self.assertLess(abs(p70["avg_delta_pct"]), abs(p50["avg_delta_pct"]))
            self.assertGreater(p50["worse_rate_pct"], 0)

    def test_non_target_case_keeps_coverage_but_not_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / "reports"; reports.mkdir()
            cache = root / "cache"; cache.mkdir()
            _write_report(reports, _row(exit_reason="time exit", max_favorable_pct=0.1, giveback_pct=0.0, exit_timing="on_time"))
            _write_cache(cache, [101] * 12)

            payload = partial.build_replay(reports_dir=reports, cache_dir=cache, cfg=partial.PartialExitConfig(days=14), save=False)
            self.assertEqual(payload["coverage"]["eligible_total"], 0)
            self.assertEqual(payload["coverage"]["labeled_total"], 0)


if __name__ == "__main__":
    unittest.main()

