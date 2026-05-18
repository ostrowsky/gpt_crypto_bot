from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestMarketAgentSoftBlockState(unittest.TestCase):
    def test_soft_block_watch_dedupe_persists(self) -> None:
        import market_signal_agent as agent

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            with patch.object(agent, "STATE_FILE", state_path):
                agent._save_state({}, {}, {"CHZUSDT|15m|impulse_mode_watch": "2026-05-18"})
                last_exit, cooldowns, soft_blocks = agent._load_state()
        self.assertEqual(last_exit, {})
        self.assertEqual(cooldowns, {})
        self.assertEqual(soft_blocks, {"CHZUSDT|15m|impulse_mode_watch": "2026-05-18"})
