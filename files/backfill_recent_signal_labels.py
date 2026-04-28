from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np


ROOT = Path(__file__).resolve().parent
ML_FILE = ROOT / "ml_dataset.jsonl"
BINANCE_BASE = "https://api.binance.com/api/v3/klines"
BAR_MS = {"15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000}


def _parse_ts(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad = 0
    if not path.exists():
        return rows, bad
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows, bad


def _needs_label(rec: dict[str, Any], min_ts: datetime) -> bool:
    if str(rec.get("signal_type", "none")) == "none":
        return False
    dt = _parse_ts(rec.get("ts_signal"))
    if dt is None or dt < min_ts:
        return False
    labels = rec.get("labels") or {}
    return any(labels.get(f"ret_{h}") is None for h in (3, 5, 10))


async def _fetch_window(
    session: aiohttp.ClientSession,
    sym: str,
    tf: str,
    start_ms: int,
    end_ms: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    params = {
        "symbol": sym,
        "interval": tf,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }
    try:
        async with session.get(BINANCE_BASE, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                return None
            raw = await resp.json()
    except Exception:
        return None
    if not isinstance(raw, list) or not raw:
        return None
    t_arr = np.asarray([int(item[0]) for item in raw], dtype=np.int64)
    c_arr = np.asarray([float(item[4]) for item in raw], dtype=np.float64)
    return t_arr, c_arr


def _fill_from_arrays(rec: dict[str, Any], t_arr: np.ndarray, c_arr: np.ndarray) -> bool:
    tf = str(rec.get("tf", ""))
    bar_ms = BAR_MS.get(tf)
    if not bar_ms:
        return False
    labels = rec.setdefault("labels", {})
    bar_ts = int(rec.get("bar_ts") or 0)
    if bar_ts <= 0:
        return False
    idx = np.where(t_arr >= bar_ts)[0]
    if len(idx) == 0:
        return False
    entry_close = float(c_arr[int(idx[0])])
    if entry_close <= 0:
        return False
    changed = False
    for horizon in (3, 5, 10):
        ret_key = f"ret_{horizon}"
        label_key = f"label_{horizon}"
        if labels.get(ret_key) is not None:
            continue
        fut = np.where(t_arr >= bar_ts + horizon * bar_ms)[0]
        if len(fut) == 0:
            continue
        ret_pct = (float(c_arr[int(fut[0])]) / entry_close - 1.0) * 100.0
        labels[ret_key] = round(ret_pct, 4)
        labels[label_key] = ret_pct > 0.0
        changed = True
    return changed


async def run(min_date: str, dry_run: bool) -> dict[str, Any]:
    min_ts = datetime.fromisoformat(min_date).replace(tzinfo=timezone.utc)
    rows, bad_lines = _load_rows(ML_FILE)
    pending_idx = [idx for idx, rec in enumerate(rows) if _needs_label(rec, min_ts)]
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx in pending_idx:
        rec = rows[idx]
        groups[(str(rec.get("sym", "")), str(rec.get("tf", "")))].append(idx)

    updated = 0
    fetched = 0
    failed_groups: list[str] = []
    async with aiohttp.ClientSession() as session:
        for (sym, tf), indices in sorted(groups.items()):
            if not sym or tf not in BAR_MS:
                continue
            ts_values = [int(rows[idx].get("bar_ts") or 0) for idx in indices]
            ts_values = [ts for ts in ts_values if ts > 0]
            if not ts_values:
                continue
            start_ms = min(ts_values)
            end_ms = max(ts_values) + 12 * BAR_MS[tf]
            data = await _fetch_window(session, sym, tf, start_ms, end_ms)
            if data is None:
                failed_groups.append(f"{sym}:{tf}")
                continue
            fetched += 1
            t_arr, c_arr = data
            for idx in indices:
                if _fill_from_arrays(rows[idx], t_arr, c_arr):
                    updated += 1
            await asyncio.sleep(0.05)

    if updated and not dry_run:
        tmp = ML_FILE.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(tmp, ML_FILE)

    return {
        "file": str(ML_FILE),
        "min_date": min_date,
        "dry_run": dry_run,
        "rows_loaded": len(rows),
        "bad_lines_skipped": bad_lines,
        "pending_signal_rows": len(pending_idx),
        "groups": len(groups),
        "groups_fetched": fetched,
        "groups_failed": failed_groups[:20],
        "rows_updated": updated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill recent signal forward labels in ml_dataset.jsonl")
    parser.add_argument("--min-date", default="2026-04-25")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(run(args.min_date, args.dry_run))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
