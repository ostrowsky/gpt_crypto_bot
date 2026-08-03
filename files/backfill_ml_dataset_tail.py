from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

import config
import ml_dataset
from backfill_history import BAR_MS, BINANCE_URL, _detect_signal, _load_bull_day_cache
from indicators import compute_features


ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH_SIZE = 8
WARMUP_BARS = 300
FETCH_LIMIT = 1000


def _scan_latest_bar_ts(path: Path) -> tuple[dict[tuple[str, str], int], int]:
    latest: dict[tuple[str, str], int] = {}
    bad_rows = 0
    with path.open("r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                key = (str(rec["sym"]), str(rec["tf"]))
                bar_ts = int(rec["bar_ts"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                bad_rows += 1
                continue
            latest[key] = max(latest.get(key, 0), bar_ts)
    return latest, bad_rows


async def _fetch_tail(
    session: aiohttp.ClientSession,
    symbol: str,
    tf: str,
    latest_bar_ts: int,
) -> np.ndarray | None:
    bar_ms = BAR_MS[tf]
    params = {
        "symbol": symbol,
        "interval": tf,
        "startTime": max(0, latest_bar_ts - WARMUP_BARS * bar_ms),
        "limit": FETCH_LIMIT,
    }
    try:
        async with session.get(
            BINANCE_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            response.raise_for_status()
            rows = await response.json()
    except Exception:
        return None
    if not isinstance(rows, list) or len(rows) < WARMUP_BARS + 2:
        return None
    data = np.zeros(
        len(rows),
        dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")],
    )
    data["t"] = [int(row[0]) for row in rows]
    data["o"] = [float(row[1]) for row in rows]
    data["h"] = [float(row[2]) for row in rows]
    data["l"] = [float(row[3]) for row in rows]
    data["c"] = [float(row[4]) for row in rows]
    data["v"] = [float(row[5]) for row in rows]
    return data


def _append_pair_tail(
    symbol: str,
    tf: str,
    latest_bar_ts: int,
    data: np.ndarray,
    bull_days: dict[str, bool],
) -> int:
    feature_map = compute_features(data["o"], data["h"], data["l"], data["c"], data["v"])
    written = 0
    for index in range(WARMUP_BARS, len(data) - 1):
        bar_ts = int(data["t"][index])
        if bar_ts <= latest_bar_ts:
            continue
        day = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        ml_dataset.log_bar_snapshot(
            sym=symbol,
            tf=tf,
            bar_ts=bar_ts,
            rule_signal=_detect_signal(feature_map, index, data),
            is_bull_day=bull_days.get(day, False),
            feat=feature_map,
            i=index,
            data=data,
            btc_vs_ema50=0.0,
            btc_momentum_4h=0.0,
            market_vol_24h=0.0,
        )
        written += 1
    return written


def _fill_available_labels(market_data: dict[tuple[str, str], np.ndarray]) -> tuple[int, int]:
    close_maps = {
        key: {int(ts): float(close) for ts, close in zip(data["t"], data["c"])}
        for key, data in market_data.items()
    }

    def _mutate(rec: dict[str, Any]) -> bool:
        key = (str(rec.get("sym") or ""), str(rec.get("tf") or ""))
        prices = close_maps.get(key)
        if not prices:
            return False
        bar_ts = int(rec.get("bar_ts") or 0)
        entry = prices.get(bar_ts)
        if not entry or entry <= 0:
            return False
        labels = rec.setdefault("labels", {})
        changed = False
        for horizon in (3, 5, 10):
            if labels.get(f"ret_{horizon}") is not None:
                continue
            future = prices.get(bar_ts + horizon * BAR_MS[key[1]])
            if future is None:
                continue
            ret_pct = (future / entry - 1.0) * 100.0
            labels[f"ret_{horizon}"] = round(ret_pct, 4)
            labels[f"label_{horizon}"] = ret_pct > 0
            changed = True
        return changed

    return ml_dataset._rewrite_records(_mutate)


async def run_tail_backfill(
    *,
    dataset_file: Path = ml_dataset.ML_FILE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, int]:
    latest, bad_rows = _scan_latest_bar_ts(dataset_file)
    bull_days = _load_bull_day_cache(ROOT / "bot_events.jsonl")
    pairs = [
        (symbol, tf, latest.get((symbol, tf), 0))
        for symbol in config.load_watchlist()
        for tf in config.TIMEFRAMES
        if latest.get((symbol, tf), 0) > 0
    ]
    rows_written = 0
    pairs_completed = 0
    market_data: dict[tuple[str, str], np.ndarray] = {}
    headers = {"User-Agent": "Mozilla/5.0"}
    connector = aiohttp.TCPConnector(limit=max(4, batch_size * 2))
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            results = await asyncio.gather(
                *[_fetch_tail(session, symbol, tf, latest_ts) for symbol, tf, latest_ts in batch]
            )
            for (symbol, tf, latest_ts), data in zip(batch, results):
                if data is None:
                    continue
                market_data[(symbol, tf)] = data
                rows_written += _append_pair_tail(symbol, tf, latest_ts, data, bull_days)
                pairs_completed += 1
    labels_updated, malformed_removed = _fill_available_labels(market_data)
    return {
        "pairs_requested": len(pairs),
        "pairs_completed": pairs_completed,
        "rows_written": rows_written,
        "labels_updated": labels_updated,
        "bad_rows_seen": bad_rows,
        "malformed_removed": malformed_removed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill only the ML dataset tail after each pair's latest bar")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    result = asyncio.run(run_tail_backfill(batch_size=max(1, args.batch_size)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
