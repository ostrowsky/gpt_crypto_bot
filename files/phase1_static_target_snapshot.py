"""Build the orchestrator-owned normalized snapshot for the Phase 1 experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

from policy_provenance import current_policy_manifest
from validate_external_top50_screen import (
    DEFAULT_LEGACY_CACHE,
    DEFAULT_TAIL_CACHE,
    DEFAULT_WATCHLIST,
    build_snapshots_from_cache,
    discover_cache_files,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_KIND = "static_target_top50_normalized_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_real_snapshot(
    *,
    start_day: date,
    end_day: date,
    phase0_completion_path: Path,
    legacy_cache: Path = DEFAULT_LEGACY_CACHE,
    tail_cache: Path = DEFAULT_TAIL_CACHE,
    watchlist_path: Path = DEFAULT_WATCHLIST,
    timezone_name: str = "Europe/Budapest",
    top_n: int = 50,
    selection_size: int = 10,
    min_market_symbols: int = 200,
    min_watchlist_symbols: int = 50,
) -> dict[str, Any]:
    phase0_payload = json.loads(phase0_completion_path.read_text(encoding="utf-8"))
    if phase0_payload.get("phase") != "PHASE_0" or phase0_payload.get("status") != "COMPLETE":
        raise ValueError("Phase 0 completion artifact is not COMPLETE")
    if phase0_payload.get("trading_behavior_changed") is not False:
        raise ValueError("Phase 0 completion artifact reports a behavior change")
    watchlist_raw = watchlist_path.read_bytes()
    watchlist = set(json.loads(watchlist_raw.decode("utf-8-sig")))
    selected = discover_cache_files((legacy_cache, tail_cache))
    snapshots, provenance = build_snapshots_from_cache(
        selected,
        watchlist=watchlist,
        start_day=start_day,
        end_day=end_day,
        timezone_name=timezone_name,
        top_n=top_n,
        min_market_symbols=min_market_symbols,
        min_watchlist_symbols=min_watchlist_symbols,
    )
    requested_days = (end_day - start_day).days + 1
    serialized_days = []
    for snapshot in snapshots:
        serialized_days.append({
            "local_day": snapshot.local_day,
            "market_symbol_count": snapshot.market_symbol_count,
            "watchlist_symbol_count": snapshot.watchlist_symbol_count,
            "target_entrant_symbols": sorted(snapshot.target_entrant_symbols),
            "candidates": [
                {
                    "symbol": candidate.symbol,
                    "current_return": candidate.current_return,
                    "static_target_return": candidate.static_target_return,
                    "target_rank": candidate.target_rank,
                    "is_target_top": candidate.is_target_top,
                }
                for candidate in snapshot.candidates
            ],
        })
    eligible_days = len(serialized_days)
    coverage = eligible_days / requested_days if requested_days else 0.0
    malformed_count = int(provenance.get("malformed_file_count") or 0)
    policy = current_policy_manifest()
    decision_grade = bool(
        eligible_days >= 30
        and coverage >= 0.95
        and malformed_count == 0
        and provenance.get("used_content_hash")
        and policy.get("policy_epoch")
        and _git_commit()
    )
    return {
        "schema_version": 1,
        "snapshot_kind": SNAPSHOT_KIND,
        "decision_grade_input": decision_grade,
        "source_contract": {
            "source": "local Binance 1h cache normalized before validation",
            "timezone": timezone_name,
            "observation_time_local": "12:15",
            "target_time_local": "23:00",
            "top_n": int(top_n),
            "selection_size": int(selection_size),
            "min_market_symbols": int(min_market_symbols),
            "min_watchlist_symbols": int(min_watchlist_symbols),
            "start_day": start_day.isoformat(),
            "end_day": end_day.isoformat(),
            "requested_day_count": requested_days,
        },
        "provenance": {
            **provenance,
            "calendar_coverage": round(coverage, 6),
            "watchlist_count": len(watchlist),
            "watchlist_sha256": hashlib.sha256(watchlist_raw).hexdigest(),
            "policy_epoch": policy.get("policy_epoch"),
            "policy_hash": policy.get("policy_hash"),
            "phase0_completion_sha256": _sha256(phase0_completion_path),
            "phase0_completion_path": str(phase0_completion_path.resolve()),
            "snapshot_builder_sha256": _sha256(Path(__file__)),
            "git_commit": _git_commit(),
        },
        "days": serialized_days,
    }
