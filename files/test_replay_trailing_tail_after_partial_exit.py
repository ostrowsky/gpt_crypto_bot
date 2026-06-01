from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import replay_trailing_tail_after_partial_exit as tail


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
    start_offset = -25
    for i, close in enumerate(closes):
        t = BASE_TS + (start_offset + i) * BAR
        rows.append({"t": t, "o": close, "h": close * 1.01, "l": close * 0.999, "c": close, "v": 1000})
    end = BASE_TS + (start_offset + len(closes)) * BAR
    (root / f"AAAUSDT_15m_{BASE_TS + start_offset * BAR}_{end}.json").write_text(json.dumps(rows), encoding="utf-8")


def _run(row: dict, closes: list[float]) -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        reports = root / "reports"; reports.mkdir()
        cache = root / "cache"; cache.mkdir()
        _write_report(reports, row)
        _write_cache(cache, closes)
        return tail.build_replay(reports_dir=reports, cache_dir=cache, cfg=tail.TrailingTailConfig(days=14), save=False)


class TrailingTailAfterPartialExitReplayTests(unittest.TestCase):
    def test_tail_keeps_continuation_until_horizon(self) -> None:
        closes = [100.0] * 25 + [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
        payload = _run(_row(), closes)
        m = payload["policies"]["tail50_h10_ema20_cap100"]
        reasons = payload["exit_reasons"]["tail50_h10_ema20_cap100"]
        self.assertEqual(payload["coverage"]["labeled_total"], 1)
        self.assertGreater(m["avg_delta_pct"], 0)
        self.assertEqual(reasons.get("max_horizon"), 1)

    def test_tail_exits_on_ema_loss_before_horizon(self) -> None:
        closes = [100.0] * 25 + [99, 98.8, 98.6, 98.4, 98.2, 98.0, 97.8, 97.6, 97.4, 97.2, 97.0]
        payload = _run(_row(exit_price=99.0, pnl_pct=-1.0), closes)
        m = payload["policies"]["tail50_h10_ema20_cap100"]
        reasons = payload["exit_reasons"]["tail50_h10_ema20_cap100"]
        self.assertLessEqual(m["median_tail_bars"], 3)
        self.assertEqual(reasons.get("ema20_loss"), 1)

    def test_tail_exits_on_adverse_cap(self) -> None:
        closes = [100.0] * 25 + [101, 101.2, 101.1, 101.0, 100.9, 100.8, 100.7, 100.6, 100.5, 100.4, 100.3]
        # Force the first future candle low below the adverse cap while close remains above EMA.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / "reports"; reports.mkdir()
            cache = root / "cache"; cache.mkdir()
            _write_report(reports, _row())
            rows = []
            start_offset = -25
            for i, close in enumerate(closes):
                t = BASE_TS + (start_offset + i) * BAR
                low = 98.5 if start_offset + i == 1 else close * 0.995
                rows.append({"t": t, "o": close, "h": close * 1.01, "l": low, "c": close, "v": 1000})
            (cache / f"AAAUSDT_15m_{BASE_TS + start_offset * BAR}_{BASE_TS + (start_offset + len(closes)) * BAR}.json").write_text(json.dumps(rows), encoding="utf-8")
            payload = tail.build_replay(reports_dir=reports, cache_dir=cache, cfg=tail.TrailingTailConfig(days=14), save=False)
        reasons = payload["exit_reasons"]["tail50_h10_ema20_cap100"]
        self.assertEqual(reasons.get("adverse_cap"), 1)


if __name__ == "__main__":
    unittest.main()

