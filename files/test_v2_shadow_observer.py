from __future__ import annotations

import unittest


class TestV2ShadowObserver(unittest.TestCase):
    def test_emerging_move(self) -> None:
        from v2.shadow_observer import FeatureSnapshot, estimate_shadow_state
        from v2.state import SymbolState

        d = estimate_shadow_state(
            FeatureSnapshot(10.1, 10.0, 0.10, 20.0, 58.0, 1.0, 3.0, 0.01)
        )
        self.assertEqual(d.state, SymbolState.EMERGING_MOVE)
        self.assertEqual(d.action, "elevate_priority")

    def test_confirmed_trend(self) -> None:
        from v2.shadow_observer import FeatureSnapshot, estimate_shadow_state
        from v2.state import SymbolState

        d = estimate_shadow_state(
            FeatureSnapshot(10.2, 10.0, 0.25, 30.0, 60.0, 1.5, 4.0, 0.02)
        )
        self.assertEqual(d.state, SymbolState.CONFIRMED_TREND)

    def test_transition_dedupe(self) -> None:
        from v2.shadow_observer import ShadowDecision, material_transition
        from v2.state import SymbolState

        previous = {"state": "emerging_move", "action": "elevate_priority"}
        current = ShadowDecision(SymbolState.EMERGING_MOVE, "elevate_priority", 0.6, "same")
        self.assertFalse(material_transition(previous, current))

    def test_telegram_only_for_upside_discovery(self) -> None:
        from v2.shadow_observer import ShadowDecision, telegram_eligible
        from v2.state import SymbolState

        previous = {"state": "noise", "action": "watch"}
        emerging = ShadowDecision(SymbolState.EMERGING_MOVE, "elevate_priority", 0.64, "early")
        reversal = ShadowDecision(SymbolState.REVERSAL, "sell_candidate", 0.80, "down")
        noise = ShadowDecision(SymbolState.NOISE, "watch", 0.55, "flat")

        self.assertTrue(telegram_eligible(previous, emerging))
        self.assertFalse(telegram_eligible(previous, reversal))
        self.assertFalse(telegram_eligible({"state": "emerging_move", "action": "elevate_priority"}, noise))

    def test_decision_trace_dedupes_by_symbol_timeframe_bar(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        from v2.shadow_observer import append_decision_trace

        event = {"sym": "AAAUSDT", "tf": "15m", "bar_ts": 1, "state": "noise"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            append_decision_trace(path, event)
            append_decision_trace(path, event)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)

    def test_worker_cycle_snapshot_exposes_in_progress_heartbeat_fields(self) -> None:
        import v2_shadow_worker as worker

        cycle = worker._cycle_snapshot(
            started_at="2026-06-05T08:00:00Z",
            scanned=12,
            emitted=2,
            stale=1,
            errors=3,
            in_progress=True,
            current_symbol="BTCUSDT",
            current_tf="15m",
        )

        self.assertTrue(cycle["in_progress"])
        self.assertEqual(cycle["finished_at"], None)
        self.assertEqual(cycle["scanned"], 12)
        self.assertEqual(cycle["current"], {"symbol": "BTCUSDT", "tf": "15m"})

    def test_worker_cycle_snapshot_marks_finished_cycle(self) -> None:
        import v2_shadow_worker as worker

        cycle = worker._cycle_snapshot(
            started_at="2026-06-05T08:00:00Z",
            scanned=20,
            emitted=0,
            stale=2,
            errors=0,
            in_progress=False,
        )

        self.assertFalse(cycle["in_progress"])
        self.assertIsNotNone(cycle["finished_at"])
        self.assertNotIn("current", cycle)


if __name__ == "__main__":
    unittest.main()
