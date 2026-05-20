from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import watchlist_rescue_admission_replay as replay


def _row(day: str, sym: str, signal_type: str, flags: dict, ret5: float, *, slope: float = 0.8, vol_x: float = 1.5, rsi: float = 65.0, adx: float = 30.0) -> dict:
    return {
        "sym": sym,
        "tf": "15m",
        "signal_type": signal_type,
        "ts_signal": f"{day}T10:00:00Z",
        "f": {
            "slope": slope,
            "vol_x": vol_x,
            "rsi": rsi,
            "adx": adx,
            "daily_range": 8.0,
            "macd_hist_norm": 0.2,
            "close_vs_ema20": 1.0,
        },
        "decision": {"signal_flags": flags},
        "labels": {"ret_3": ret5 / 2.0, "ret_5": ret5, "ret_10": ret5},
        "teacher": {"final": {"watchlist_top_gainer": ret5 > 0, "status": "blocked_rule" if ret5 > 0 else "not_top15"}},
    }


class WatchlistRescueAdmissionReplayTest(unittest.TestCase):
    def test_build_selects_structural_profile_and_focus_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset = Path(td) / "critic_dataset.jsonl"
            rows = []
            for i in range(20):
                rows.append(_row("2026-05-01", f"GOOD{i}USDT", "trend", {"entry_ok": True}, 1.2))
                rows.append(_row("2026-05-02", f"GOODB{i}USDT", "alignment", {"alignment_ok": True}, 0.9))
            for i in range(15):
                rows.append(_row("2026-05-03", f"BAD{i}USDT", "trend", {"entry_ok": True}, -0.5, slope=0.1))
            dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            result = replay.build(dataset, Path(td) / "out.json", focus_symbols=["GOOD1USDT"])
            self.assertEqual(result["rows"], len(rows))
            self.assertTrue(result["profiles_all"]["entry_ok_trend"]["selected_count"] >= 20)
            self.assertEqual(result["focus_symbols"][0]["symbol"], "GOOD1USDT")
            self.assertIn(result["decision"], {"advance_selected_profile_to_fee_slippage_behavior_replay", "research_only_no_profile_passed_gate"})


if __name__ == "__main__":
    unittest.main()
