from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestExplainV2ShadowSignal(unittest.TestCase):
    def test_reports_why_no_signal_from_latest_trace(self) -> None:
        import explain_v2_shadow_signal as expl

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            rows = [
                {
                    "sym": "AAAUSDT",
                    "tf": "15m",
                    "bar_ts": 1_779_019_200_000,
                    "state": "noise",
                    "action": "watch",
                    "reason": "insufficient evidence",
                    "material_transition": False,
                }
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with patch.object(expl, "TRACE", path):
                payload = expl.explain("AAAUSDT", "15m", "2026-05-17")
        self.assertEqual(payload["why_no_signal"], "insufficient evidence")

    def test_bootstrap_transition_is_not_counted_as_material_signal(self) -> None:
        import explain_v2_shadow_signal as expl

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            rows = [
                {
                    "sym": "AAAUSDT",
                    "tf": "15m",
                    "bar_ts": 1_779_019_200_000,
                    "state": "emerging_move",
                    "action": "elevate_priority",
                    "reason": "early positive structure",
                    "material_transition": True,
                    "bootstrap": True,
                }
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with patch.object(expl, "TRACE", path):
                payload = expl.explain("AAAUSDT", "15m", "2026-05-17")
        self.assertEqual(payload["material_signals"], 0)
        self.assertEqual(payload["bootstrap_rows"], 1)


if __name__ == "__main__":
    unittest.main()
