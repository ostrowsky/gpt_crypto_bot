from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TestV2LifecycleSensitivity(unittest.TestCase):
    def test_builds_full_grid(self) -> None:
        from v2.history import CanonicalBar
        from v2.history_store import LocalHistoryStore
        import audit_v2_lifecycle_sensitivity as audit

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            store = LocalHistoryStore(root)
            bars = [
                CanonicalBar("AAAUSDT", "15m", 1_779_000_000_000 + i * 900_000, 100, 105, 99, 100 + i * 0.1, 1)
                for i in range(10)
            ]
            store.upsert("AAAUSDT", "15m", bars, source="test")
            payload = audit.build(root, Path(tmp) / "audit.json")
        self.assertEqual(len(payload["variants"]), 27)
