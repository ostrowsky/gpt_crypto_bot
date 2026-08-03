from __future__ import annotations

import unittest
from pathlib import Path


class TestRLLauncherConfig(unittest.TestCase):
    def test_detached_launcher_does_not_duplicate_bot_collector_by_default(self) -> None:
        launcher = (Path(__file__).resolve().parent.parent / "start_rl_worker_bg.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn("[switch]$EnableCollector = $false", launcher)
        self.assertIn("if ($EnableCollector)", launcher)
        self.assertIn('$wrapperArgs += "--enable-collector"', launcher)
        self.assertIn('$wrapperArgs += "--disable-collector"', launcher)
        self.assertNotIn('$loopScript, "--enable-collector"', launcher)


if __name__ == "__main__":
    unittest.main()
