"""Own-environment startup contract; no keys, sockets or background workers."""

import json
import os
import subprocess
import unittest
from pathlib import Path


class OfflineStartupTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def run_app(self, pythonpath=None):
        interpreter = self.ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        if pythonpath is not None:
            environment["PYTHONPATH"] = pythonpath
        return subprocess.run([str(interpreter), "-I", "-m", "binance_demo_bot"],
                              cwd=self.ROOT.parent.parent, env=environment,
                              capture_output=True, text=True, timeout=15)

    def test_ISO_01_independent_installed_package_starts_with_trading_disabled(self):
        result = self.run_app()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["application_id"], "binance_demo_bot")
        self.assertEqual(Path(payload["app_root"]), self.ROOT)
        self.assertIs(payload["order_send_enabled"], False)
        self.assertIs(payload["financial_core_only"], True)

    def test_ISO_02_startup_rejects_inherited_legacy_pythonpath(self):
        result = self.run_app(str(self.ROOT.parent.parent / "files"))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("PYTHONPATH", result.stderr)
        self.assertEqual(result.stdout, "")
