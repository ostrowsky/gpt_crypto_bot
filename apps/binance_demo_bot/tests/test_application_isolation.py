"""Test-first isolation kernel; full process/account isolation is a later gate."""

import importlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from _support import FinancialContractCase


class ApplicationIsolationTests(FinancialContractCase):
    def setUp(self):
        self.api = importlib.import_module("binance_demo_bot.isolation")
        fixture_root = Path(__file__).resolve().parents[1] / ".runtime" / "unit-fixtures"
        fixture_root.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=fixture_root)
        self.base = Path(directory.name).resolve()
        self.assertTrue(self.base.is_relative_to(fixture_root.resolve()))
        self.addCleanup(directory.cleanup)
        self.root = self.base / "app"
        self.root.mkdir()
        self.paths = self.api.AppPaths(self.root)

    def test_ISO_02_paths_do_not_depend_on_current_directory(self):
        before = Path.cwd()
        try:
            os.chdir(self.base)
            self.assertEqual(self.paths.runtime_path("ledger.sqlite"),
                             self.root / ".runtime" / "ledger.sqlite")
            self.assertEqual(self.paths.env_file, self.root / ".env")
        finally:
            os.chdir(before)

    def test_ISO_02_injected_pythonpath_is_rejected(self):
        self.api.assert_clean_pythonpath(None)
        self.api.assert_clean_pythonpath("")
        for value in ("../files", str(self.base / "legacy"), "."):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.api.assert_clean_pythonpath(value)

    def test_ISO_03_absolute_and_parent_traversal_paths_are_rejected(self):
        for value in ("../positions.json", "sub/../../legacy.db", self.base / "foreign.sqlite"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.paths.runtime_path(value)

    def test_ISO_03_resolved_link_escape_is_rejected(self):
        target = self.base / "foreign"
        target.mkdir()
        runtime = self.root / ".runtime"
        runtime.mkdir()
        link = runtime / "escape"
        if os.name == "nt":
            created = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(target)],
                                     capture_output=True, text=True)
            self.assertEqual(created.returncode, 0, created.stderr)
        else:
            link.symlink_to(target, target_is_directory=True)
        try:
            with self.assertRaises(ValueError):
                self.paths.runtime_path("escape/ledger.sqlite")
            self.assertEqual(list(target.iterdir()), [])
        finally:
            # Remove only the link, before bounded fixture-directory cleanup.
            if os.name == "nt":
                link.rmdir()
            else:
                link.unlink()

    def test_ISO_02_parent_env_is_never_a_fallback(self):
        (self.base / ".env").write_text(
            "BINANCE_DEMO_API_KEY=parent-fixture\nBINANCE_DEMO_API_SECRET=parent-secret\n",
            encoding="utf-8")
        with self.assertRaises(ValueError):
            self.api.load_demo_credentials(self.paths)

    def test_ISO_10_only_own_demo_pair_is_loaded_and_repr_is_redacted(self):
        self.paths.env_file.write_text(
            '# offline fixture\nBINANCE_DEMO_API_KEY="fixture-key"\n'
            "BINANCE_DEMO_API_SECRET='fixture-secret'\n", encoding="utf-8")
        value = self.api.load_demo_credentials(self.paths)
        self.assertEqual(value.api_key, "fixture-key")
        self.assertEqual(value.api_secret, "fixture-secret")
        self.assertNotIn("fixture-key", repr(value))
        self.assertNotIn("fixture-secret", str(value))

    def test_ISO_10_explicit_pair_does_not_mix_with_file(self):
        self.paths.env_file.write_text(
            "BINANCE_DEMO_API_KEY=file-key\nBINANCE_DEMO_API_SECRET=file-secret\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.api.load_demo_credentials(self.paths, explicit={"BINANCE_DEMO_API_KEY": "explicit-key"})
        value = self.api.load_demo_credentials(self.paths, explicit={
            "BINANCE_DEMO_API_KEY": "explicit-key", "BINANCE_DEMO_API_SECRET": "explicit-secret"})
        self.assertEqual((value.api_key, value.api_secret), ("explicit-key", "explicit-secret"))

    def test_ISO_10_legacy_names_do_not_supply_demo_credentials(self):
        self.paths.env_file.write_text("BINANCE_API_KEY=legacy-key\nBINANCE_API_SECRET=legacy-secret\n",
                                       encoding="utf-8")
        with self.assertRaises(ValueError):
            self.api.load_demo_credentials(self.paths)

    def test_ISO_10_duplicate_or_empty_key_is_rejected_without_secret_echo(self):
        for content in (
            "BINANCE_DEMO_API_KEY=fixture-key\nBINANCE_DEMO_API_KEY=other-key\nBINANCE_DEMO_API_SECRET=fixture-secret\n",
            "BINANCE_DEMO_API_KEY=\nBINANCE_DEMO_API_SECRET=fixture-secret\n",
        ):
            with self.subTest(content_type="synthetic malformed env"):
                self.paths.env_file.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError) as caught:
                    self.api.load_demo_credentials(self.paths)
                self.assertNotIn("fixture-secret", str(caught.exception))
                self.assertNotIn("fixture-key", str(caught.exception))

    def test_ISO_03_root_must_be_explicit_and_absolute(self):
        with self.assertRaises(ValueError):
            self.api.AppPaths(Path("relative-app"))
