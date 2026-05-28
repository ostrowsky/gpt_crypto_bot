import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_info


class BuildInfoTests(unittest.TestCase):
    def test_git_build_info_clean(self):
        calls = {
            ("rev-parse", "--short", "HEAD"): "abc1234",
            ("show", "-s", "--format=%cI", "HEAD"): "2026-05-28T14:30:00+00:00",
        }

        with patch.object(build_info, "_run_git", side_effect=lambda _root, *args: calls[args]), \
             patch.object(build_info, "_git_status_porcelain", return_value=""):
            info = build_info.get_build_info(repo_root=Path("C:/repo"))

        self.assertEqual(info.version, "abc1234")
        self.assertIn("2026-05-28", info.built_at)
        self.assertEqual(info.source, "git")

    def test_git_build_info_marks_dirty(self):
        calls = {
            ("rev-parse", "--short", "HEAD"): "abc1234",
            ("show", "-s", "--format=%cI", "HEAD"): "2026-05-28T14:30:00+00:00",
        }

        with patch.object(build_info, "_run_git", side_effect=lambda _root, *args: calls[args]), \
             patch.object(build_info, "_git_status_porcelain", return_value=" M files/bot.py"):
            info = build_info.get_build_info(repo_root=Path("C:/repo"))

        self.assertEqual(info.version, "abc1234+dirty")

    def test_runtime_artifacts_do_not_mark_dirty(self):
        calls = {
            ("rev-parse", "--short", "HEAD"): "abc1234",
            ("show", "-s", "--format=%cI", "HEAD"): "2026-05-28T14:30:00+00:00",
        }
        status = (
            " M .runtime/reports/top_gainer_critic_history.jsonl\n"
            " M files/.runtime/market_agent_status.json\n"
            " M files/ml_candidate_ranker.json\n"
            " M files/positions.json"
        )

        with patch.object(build_info, "_run_git", side_effect=lambda _root, *args: calls[args]), \
             patch.object(build_info, "_git_status_porcelain", return_value=status):
            info = build_info.get_build_info(repo_root=Path("C:/repo"))

        self.assertEqual(info.version, "abc1234")

    def test_fallback_when_git_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            fallback = Path(tmp) / "bot.py"
            fallback.write_text("x", encoding="utf-8")
            with patch.object(build_info, "_run_git", side_effect=RuntimeError("no git")):
                info = build_info.get_build_info(repo_root=Path(tmp), fallback_file=fallback)

        self.assertEqual(info.version, "unknown")
        self.assertEqual(info.source, "fallback")


if __name__ == "__main__":
    unittest.main()
