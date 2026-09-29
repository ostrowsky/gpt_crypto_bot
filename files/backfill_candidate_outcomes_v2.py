from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp
import numpy as np

import config
import critic_dataset
import policy_provenance
from ml_signal_model import save_json


BINANCE_KLINES_URL = f"{config.BINANCE_REST}/api/v3/klines"
BAR_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
PAGE_LIMIT = 1_000


def pending_requirements(
    dataset_path: Path,
    *,
    observed_at: datetime | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Find mature, still-missing forward targets without inventing outcomes."""
    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    if not dataset_path.exists():
        return grouped

    with dataset_path.open("r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(rec, dict):
                continue
            provenance = rec.get("provenance") or {}
            if (
                str(provenance.get("dataset_contract") or "")
                != str(config.RANKER_DATASET_CONTRACT)
            ):
                continue
            if not policy_provenance.observation_provenance_valid(rec):
                continue
            sym = str(rec.get("sym") or "").strip().upper()
            tf = str(rec.get("tf") or "").strip()
            bar_ts = rec.get("bar_ts")
            if not sym or tf not in BAR_MS:
                continue
            try:
                bar_ts = int(bar_ts)
            except (TypeError, ValueError):
                continue
            labels = rec.get("labels") or {}
            missing_mature: list[int] = []
            for horizon in (3, 5, 10):
                if labels.get(f"ret_{horizon}") is not None:
                    continue
                available = policy_provenance.forward_label_time(
                    bar_ts=bar_ts, tf=tf, horizon=horizon
                )
                if now >= available:
                    missing_mature.append(horizon)
            if not missing_mature:
                continue
            key = (sym, tf)
            item = grouped.setdefault(
                key,
                {
                    "sym": sym,
                    "tf": tf,
                    "bar_ms": BAR_MS[tf],
                    "earliest_bar_ts": bar_ts,
                    "latest_target_ts": bar_ts,
                    "rows": 0,
                    "targets": 0,
                },
            )
            item["earliest_bar_ts"] = min(int(item["earliest_bar_ts"]), bar_ts)
            item["latest_target_ts"] = max(
                int(item["latest_target_ts"]),
                bar_ts + max(missing_mature) * BAR_MS[tf] + BAR_MS[tf],
            )
            item["rows"] += 1
            item["targets"] += len(missing_mature)
    return grouped


async def fetch_kline_range(
    session: aiohttp.ClientSession,
    *,
    sym: str,
    tf: str,
    start_ms: int,
    end_ms: int,
) -> np.ndarray | None:
    """Fetch an exact paginated Binance range, including a proof bar after T+N."""
    bar_ms = BAR_MS[tf]
    cursor = int(start_ms)
    rows: dict[int, list[Any]] = {}
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    while cursor <= int(end_ms):
        params = {
            "symbol": sym,
            "interval": tf,
            "startTime": cursor,
            "endTime": int(end_ms),
            "limit": PAGE_LIMIT,
        }
        try:
            async with session.get(
                BINANCE_KLINES_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response.raise_for_status()
                payload = await response.json()
        except Exception:
            return None
        if not isinstance(payload, list) or not payload:
            break
        try:
            for raw in payload:
                if not isinstance(raw, list) or len(raw) < 7:
                    return None
                ts = int(raw[0])
                o, h, l, c, v = [float(x) for x in raw[1:6]]
                if (ts % bar_ms or not cursor <= ts <= end_ms
                        or int(raw[6]) != ts + bar_ms - 1
                        or not all(math.isfinite(x) for x in (o, h, l, c, v))
                        or min(o, h, l, c) <= 0 or v < 0
                        or h < max(o, l, c) or l > min(o, h, c)):
                    return None
                if ts + bar_ms > now_ms:
                    continue
                if ts in rows and rows[ts] != raw:
                    return None
                rows[ts] = raw
            last_open = int(payload[-1][0])
        except (TypeError, ValueError, OverflowError):
            return None
        next_cursor = last_open + bar_ms
        if len(payload) < PAGE_LIMIT or next_cursor <= cursor:
            break
        cursor = next_cursor

    if len(rows) < 2:
        return None
    ordered = [rows[key] for key in sorted(rows)]
    data = np.zeros(
        len(ordered),
        dtype=[
            ("t", "i8"),
            ("o", "f8"),
            ("h", "f8"),
            ("l", "f8"),
            ("c", "f8"),
            ("v", "f8"),
        ],
    )
    data["t"] = [int(row[0]) for row in ordered]
    data["o"] = [float(row[1]) for row in ordered]
    data["h"] = [float(row[2]) for row in ordered]
    data["l"] = [float(row[3]) for row in ordered]
    data["c"] = [float(row[4]) for row in ordered]
    data["v"] = [float(row[5]) for row in ordered]
    return data


async def backfill(
    dataset_path: Path,
    *,
    concurrency: int = 8,
    fetcher: Callable[..., Awaitable[np.ndarray | None]] = fetch_kline_range,
) -> dict[str, Any]:
    requirements = pending_requirements(dataset_path)
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    series: list[dict[str, Any]] = []
    failures: list[str] = []

    async with aiohttp.ClientSession() as session:
        async def _one(item: dict[str, Any]) -> None:
            async with semaphore:
                data = await fetcher(
                    session,
                    sym=str(item["sym"]),
                    tf=str(item["tf"]),
                    start_ms=int(item["earliest_bar_ts"]),
                    end_ms=int(item["latest_target_ts"]),
                )
            if data is None:
                failures.append(f"{item['sym']}:{item['tf']}")
                return
            series.append(
                {
                    "sym": item["sym"],
                    "tf": item["tf"],
                    "t_arr": data["t"].astype(int),
                    "c_arr": data["c"].astype(float),
                    "bar_ms": int(item["bar_ms"]),
                    "source": "historical_candidate_label_recovery",
                    "market_evidence": {
                        "endpoint": BINANCE_KLINES_URL,
                        "sym": item["sym"], "tf": item["tf"],
                        "start_ms": int(item["earliest_bar_ts"]),
                        "end_ms": int(item["latest_target_ts"]),
                        "retrieved_at_utc": policy_provenance.utc_iso(),
                        "closed_series_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                    },
                }
            )

        await asyncio.gather(*(_one(item) for item in requirements.values()))

    old_path = critic_dataset.CRITIC_FILE
    try:
        critic_dataset.CRITIC_FILE = dataset_path
        critic_dataset.fill_pending_batch(series, strict=True)
    finally:
        critic_dataset.CRITIC_FILE = old_path

    remaining = pending_requirements(dataset_path)
    return {
        "schema_version": 1,
        "generated_at_utc": policy_provenance.utc_iso(),
        "dataset": str(dataset_path.resolve()),
        "pairs_requested": len(requirements),
        "rows_requested": sum(int(item["rows"]) for item in requirements.values()),
        "targets_requested": sum(int(item["targets"]) for item in requirements.values()),
        "pairs_fetched": len(series),
        "pairs_failed": sorted(failures),
        "remaining_pairs": len(remaining),
        "remaining_rows": sum(int(item["rows"]) for item in remaining.values()),
        "remaining_targets": sum(int(item["targets"]) for item in remaining.values()),
        "missing_market_data_remains_unknown": True,
        "evidence_status": "complete" if not remaining else "blocked_missing_market_data",
        "training_eligible": False,
        "achievement_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill mature candidate-outcome-v2 forward labels from Binance"
    )
    parser.add_argument("--dataset", type=Path, default=critic_dataset.CRITIC_FILE)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = asyncio.run(backfill(args.dataset, concurrency=args.concurrency))
    if args.report:
        save_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["remaining_rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
