# -*- coding: utf-8 -*-
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class BuildInfo:
    version: str
    built_at: str
    source: str


def _local_tz():
    try:
        return ZoneInfo("Europe/Budapest")
    except ZoneInfoNotFoundError:
        return timezone.utc


def _run_git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(repo_root),
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=2.0,
    ).strip()


def _git_status_porcelain(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=2.0,
    )


def _is_material_dirty(status_porcelain: str) -> bool:
    ignored_prefixes = (
        ".runtime/",
        "files/.runtime/",
    )
    ignored_exact = {
        "files/.chat_ids",
        "files/.chat_ids.v2_temp_backup",
        "files/agent_positions.json",
        "files/positions.json",
        "files/ml_candidate_ranker.json",
        "files/ml_candidate_ranker_report.json",
        "files/ml_candidate_ranker_shadow_report.json",
    }
    for raw_line in status_porcelain.splitlines():
        path = raw_line[3:].replace("\\", "/") if len(raw_line) >= 4 else raw_line.strip()
        if not path:
            continue
        if path in ignored_exact:
            continue
        if any(path.startswith(prefix) for prefix in ignored_prefixes):
            continue
        return True
    return False


def _format_commit_date(value: str) -> str:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_local_tz()).strftime("%Y-%m-%d %H:%M:%S %z")


def get_build_info(repo_root: Path | None = None, fallback_file: Path | None = None) -> BuildInfo:
    """Return runtime build metadata for Telegram/operator display."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    repo_root = Path(repo_root).resolve()

    try:
        commit = _run_git(repo_root, "rev-parse", "--short", "HEAD")
        commit_date = _run_git(repo_root, "show", "-s", "--format=%cI", "HEAD")
        dirty = _is_material_dirty(_git_status_porcelain(repo_root))
        version = f"{commit}{'+dirty' if dirty else ''}"
        return BuildInfo(
            version=version,
            built_at=_format_commit_date(commit_date),
            source="git",
        )
    except Exception:
        pass

    if fallback_file is None:
        fallback_file = Path(__file__).resolve()
    fallback_file = Path(fallback_file)
    try:
        mtime = datetime.fromtimestamp(fallback_file.stat().st_mtime, tz=timezone.utc)
    except Exception:
        mtime = datetime.now(timezone.utc)
    return BuildInfo(
        version="unknown",
        built_at=mtime.astimezone(_local_tz()).strftime("%Y-%m-%d %H:%M:%S %z"),
        source="fallback",
    )


def build_badge(repo_root: Path | None = None, fallback_file: Path | None = None) -> str:
    info = get_build_info(repo_root=repo_root, fallback_file=fallback_file)
    return f"`v:{info.version}`  `build:{info.built_at}`"
