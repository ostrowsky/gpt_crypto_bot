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


if __name__ == "__main__":
    unittest.main()
