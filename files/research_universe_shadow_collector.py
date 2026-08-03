from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import aiohttp
import numpy as np

import config
import data_collector
import research_early_trend_shadow
from indicators import compute_features
from runtime_executors import run_cpu
from strategy import fetch_klines


ROOT = Path(__file__).resolve().parent
DATASET_FILE = ROOT / "research_universe_shadow.jsonl"
STATUS_FILE = ROOT / ".runtime" / "research_universe_shadow_status.json"
BINANCE_API = "https://api.binance.com"
STABLE_OR_NON_TARGET = {
    "USDCUSDT",
    "BUSDUSDT",
    "FDUSDUSDT",
    "TUSDUSDT",
    "DAIUSDT",
    "USDPUSDT",
    "USDSUSDT",
    "EURUSDT",
    "GBPUSDT",
    "TRYUSDT",
    "BRLUSDT",
}
LEVERAGED_TOKENS = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
DEFAULT_FETCH_LIMIT = 120
DEFAULT_HORIZONS = (3, 5, 10)
TIMEFRAME_UNITS_MS = {
    "m": 60_000,
    "h": 60 * 60_000,
    "d": 24 * 60 * 60_000,
    "w": 7 * 24 * 60 * 60_000,
}

log = logging.getLogger("research_universe_shadow_collector")


@dataclass(frozen=True)
class ResearchSymbol:
    symbol: str
    rank_24h: int
    price_change_pct_24h: float
    quote_volume_24h: float
    in_trade_watchlist: bool


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def run_once(
    *,
    dataset_file: Path = DATASET_FILE,
    status_file: Path = STATUS_FILE,
    max_symbols: int | None = None,
    timeframes: Iterable[str] | None = None,
    batch_size: int | None = None,
    min_quote_volume: float | None = None,
    fetch_limit: int = DEFAULT_FETCH_LIMIT,
) -> dict[str, Any]:
    max_symbols = int(max_symbols if max_symbols is not None else getattr(config, "RESEARCH_UNIVERSE_SHADOW_MAX_SYMBOLS", 80))
    batch_size = int(batch_size if batch_size is not None else getattr(config, "RESEARCH_UNIVERSE_SHADOW_BATCH_SIZE", 8))
    symbol_timeout_sec = int(getattr(config, "RESEARCH_UNIVERSE_SHADOW_SYMBOL_TIMEOUT_SEC", 40))
    min_quote_volume = float(min_quote_volume if min_quote_volume is not None else getattr(config, "RESEARCH_UNIVERSE_SHADOW_MIN_QUOTE_VOLUME", 1_000_000.0))
    tf_list = tuple(str(x) for x in (timeframes or getattr(config, "RESEARCH_UNIVERSE_SHADOW_TIMEFRAMES", ("15m",))))
    status = {
        "started_at": _utc_now_iso(),
        "finished_at": None,
        "enabled": True,
        "running": True,
        "last_error": "",
        "symbols_total": 0,
        "symbols_scanned": 0,
        "pairs_scanned": 0,
        "rows_written": 0,
        "labels_updated": 0,
        "malformed_rows_quarantined": 0,
        "early_trend_scored": 0,
        "early_trend_candidates": 0,
        "early_trend_model_error": "",
        "dataset_file": str(dataset_file),
    }
    _write_status(status_file, status)
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        connector = aiohttp.TCPConnector(limit=max(4, batch_size * 2))
        timeout = aiohttp.ClientTimeout(total=max(30, symbol_timeout_sec + 10))
        async with aiohttp.ClientSession(headers=headers, connector=connector, timeout=timeout) as session:
            universe = await build_research_universe(
                session,
                max_symbols=max_symbols,
                min_quote_volume=min_quote_volume,
            )
            status["symbols_total"] = len(universe)
            status["symbols_scanned"] = len(universe)
            _write_status(status_file, status)
            result = await collect_symbols(
                session,
                universe,
                dataset_file=dataset_file,
                timeframes=tf_list,
                batch_size=batch_size,
                fetch_limit=fetch_limit,
                symbol_timeout_sec=symbol_timeout_sec,
                status_file=status_file,
                status=status,
            )
        status.update(result)
        status["finished_at"] = _utc_now_iso()
        status["running"] = False
        _write_status(status_file, status)
        return status
    except Exception as exc:
        status["finished_at"] = _utc_now_iso()
        status["running"] = False
        status["last_error"] = f"{type(exc).__name__}: {exc}"
        _write_status(status_file, status)
        raise


async def build_research_universe(
    session: aiohttp.ClientSession,
    *,
    max_symbols: int,
    min_quote_volume: float,
) -> list[ResearchSymbol]:
    trade_watchlist = set(config.load_watchlist())
    exchange_symbols = await _fetch_exchange_symbols(session)
    tickers = await _fetch_all_tickers(session)
    rows: list[tuple[str, float, float]] = []
    for symbol in exchange_symbols:
        if not _is_research_symbol(symbol):
            continue
        ticker = tickers.get(symbol)
        if not ticker:
            continue
        quote_volume = _safe(ticker.get("quoteVolume"))
        if quote_volume < min_quote_volume:
            continue
        change = _safe(ticker.get("priceChangePercent"))
        rows.append((symbol, change, quote_volume))
    rows.sort(key=lambda item: (item[2], item[1]), reverse=True)
    if max_symbols > 0:
        rows = rows[:max_symbols]
    return [
        ResearchSymbol(
            symbol=symbol,
            rank_24h=idx,
            price_change_pct_24h=change,
            quote_volume_24h=quote_volume,
            in_trade_watchlist=symbol in trade_watchlist,
        )
        for idx, (symbol, change, quote_volume) in enumerate(rows, start=1)
    ]


async def collect_symbols(
    session: aiohttp.ClientSession,
    universe: list[ResearchSymbol],
    *,
    dataset_file: Path,
    timeframes: Iterable[str],
    batch_size: int,
    fetch_limit: int,
    symbol_timeout_sec: int,
    status_file: Path | None = None,
    status: dict[str, Any] | None = None,
) -> dict[str, int]:
    existing_ids = _existing_ids(dataset_file)
    rows_written = 0
    early_trend_scored = 0
    early_trend_candidates = 0
    market_data: dict[tuple[str, str], Any] = {}
    pairs = [(item, tf) for item in universe for tf in timeframes]
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        results = await asyncio.gather(
            *[
                asyncio.wait_for(
                    _collect_one(
                        session,
                        item,
                        tf,
                        dataset_file=dataset_file,
                        existing_ids=existing_ids,
                        fetch_limit=fetch_limit,
                    ),
                    timeout=symbol_timeout_sec,
                )
                for item, tf in batch
            ],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                log.debug("research universe collect error: %s", result)
                continue
            rows_written += int(result.get("rows_written", 0))
            early_trend_scored += int(result.get("early_trend_scored", 0))
            early_trend_candidates += int(result.get("early_trend_candidates", 0))
            key = result.get("market_data_key")
            data = result.get("market_data")
            if isinstance(key, tuple) and len(key) == 2 and data is not None:
                market_data[(str(key[0]), str(key[1]))] = data
        if status_file is not None and status is not None:
            status.update(
                {
                    "pairs_scanned": min(start + len(batch), len(pairs)),
                    "rows_written": rows_written,
                    "early_trend_scored": early_trend_scored,
                    "early_trend_candidates": early_trend_candidates,
                    "early_trend_model_error": research_early_trend_shadow.SCORER.last_error,
                }
            )
            _write_status(status_file, status)
    label_result = await run_cpu(_fill_mature_labels_batch, dataset_file, market_data)
    labels_updated = int(label_result["labels_updated"])
    malformed_rows_quarantined = int(label_result["malformed_rows_quarantined"])
    return {
        "symbols_scanned": len(universe),
        "pairs_scanned": len(pairs),
        "rows_written": rows_written,
        "labels_updated": labels_updated,
        "malformed_rows_quarantined": malformed_rows_quarantined,
        "early_trend_scored": early_trend_scored,
        "early_trend_candidates": early_trend_candidates,
        "early_trend_model_error": research_early_trend_shadow.SCORER.last_error,
    }


async def _collect_one(
    session: aiohttp.ClientSession,
    item: ResearchSymbol,
    tf: str,
    *,
    dataset_file: Path,
    existing_ids: set[str],
    fetch_limit: int,
) -> dict[str, Any]:
    data = await fetch_klines(session, item.symbol, tf, limit=fetch_limit)
    if data is None or len(data) < 30:
        return {"rows_written": 0}
    result: dict[str, Any] = {
        "rows_written": 0,
        "market_data_key": (item.symbol, tf),
        "market_data": data,
    }
    i = len(data["c"]) - 2
    if i < 20:
        return result
    bar_ts = int(data["t"][i])
    record_id = _record_id(item.symbol, tf, bar_ts)
    if record_id in existing_ids:
        return result
    feat = await run_cpu(compute_features, data["o"], data["h"], data["l"], data["c"].astype(float), data["v"])
    rule_signal = await run_cpu(data_collector._detect_rule_signal, feat, i, data)
    record = _build_record(item, tf, bar_ts, rule_signal, feat, i, data)
    annotation = research_early_trend_shadow.annotate_record(record)
    _append_jsonl(dataset_file, record)
    existing_ids.add(record_id)
    result["rows_written"] = 1
    result["early_trend_scored"] = int(annotation is not None)
    result["early_trend_candidates"] = int(bool(annotation and annotation.get("candidate")))
    return result


def _build_record(
    item: ResearchSymbol,
    tf: str,
    bar_ts: int,
    rule_signal: str,
    feat: dict[str, Any],
    i: int,
    data: Any,
) -> dict[str, Any]:
    c = data["c"].astype(float)
    o = data["o"].astype(float)
    h = data["h"].astype(float)
    l = data["l"].astype(float)
    close = _safe(c[i])
    body_pct = abs(_safe(c[i]) - _safe(o[i])) / close * 100 if close > 0 else 0.0
    upper_wick_pct = (_safe(h[i]) - max(_safe(o[i]), _safe(c[i]))) / close * 100 if close > 0 else 0.0
    lower_wick_pct = (min(_safe(o[i]), _safe(c[i])) - _safe(l[i])) / close * 100 if close > 0 else 0.0
    return {
        "id": _record_id(item.symbol, tf, bar_ts),
        "source": "research_universe_shadow",
        "sym": item.symbol,
        "tf": tf,
        "bar_ts": int(bar_ts),
        "ts_utc": datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "in_trade_watchlist": bool(item.in_trade_watchlist),
        "rank_24h": int(item.rank_24h),
        "price_change_pct_24h": round(float(item.price_change_pct_24h), 6),
        "quote_volume_24h": round(float(item.quote_volume_24h), 2),
        "rule_signal": str(rule_signal),
        "f": {
            "rsi": round(_feature(feat, "rsi", i), 4),
            "adx": round(_feature(feat, "adx", i), 4),
            "slope": round(_feature(feat, "slope", i), 4),
            "vol_x": round(_feature(feat, "vol_x", i), 4),
            "macd_hist_norm": round((_feature(feat, "macd_hist", i) / close * 100) if close > 0 else 0.0, 6),
            "atr_pct": round((_feature(feat, "atr", i) / close * 100) if close > 0 else 0.0, 6),
            "body_pct": round(body_pct, 6),
            "upper_wick_pct": round(upper_wick_pct, 6),
            "lower_wick_pct": round(lower_wick_pct, 6),
        },
        "labels": {f"ret_{horizon}": None for horizon in DEFAULT_HORIZONS},
    }


def _fill_mature_labels_for_symbol(dataset_file: Path, symbol: str, tf: str, data: Any) -> int:
    result = _fill_mature_labels_batch(dataset_file, {(symbol, tf): data})
    return int(result["labels_updated"])


def _fill_mature_labels_batch(
    dataset_file: Path,
    market_data: dict[tuple[str, str], Any],
    *,
    quarantine_file: Path | None = None,
) -> dict[str, int]:
    """Fill all labels in one streaming, atomic dataset pass.

    Malformed JSONL rows are removed from the active dataset and copied to a
    quarantine file. One bad observation therefore cannot stop future labels.
    """
    if not dataset_file.exists():
        return {
            "labels_updated": 0,
            "malformed_rows_quarantined": 0,
            "rows_scanned": 0,
        }

    indexed_data: dict[tuple[str, str], tuple[dict[int, int], np.ndarray]] = {}
    for key, data in market_data.items():
        try:
            indexed_data[(str(key[0]), str(key[1]))] = (
                {int(ts): idx for idx, ts in enumerate(data["t"])},
                data["c"].astype(float),
            )
        except Exception:
            log.debug("invalid market data for research label key=%s", key, exc_info=True)

    quarantine_file = quarantine_file or dataset_file.with_name(f"{dataset_file.stem}_quarantine.jsonl")
    temp_file = dataset_file.with_name(f"{dataset_file.name}.labels.tmp")
    labels_updated = 0
    malformed: list[dict[str, Any]] = []
    rows_scanned = 0
    source_stat = dataset_file.stat()
    try:
        with dataset_file.open("r", encoding="utf-8", errors="replace") as source, temp_file.open(
            "w", encoding="utf-8"
        ) as target:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                    if not isinstance(row, dict):
                        raise ValueError("JSONL observation is not an object")
                except Exception as exc:
                    malformed.append(
                        {
                            "quarantined_at": _utc_now_iso(),
                            "line_number": line_number,
                            "error": f"{type(exc).__name__}: {exc}",
                            "raw": raw_line.rstrip("\r\n"),
                        }
                    )
                    continue

                rows_scanned += 1
                series = indexed_data.get((str(row.get("sym") or ""), str(row.get("tf") or "")))
                if series is not None:
                    ts_to_idx, closes = series
                    labels = row.setdefault("labels", {})
                    idx = ts_to_idx.get(int(row.get("bar_ts") or -1))
                    entry = _safe(closes[idx]) if idx is not None else 0.0
                    if idx is not None and entry > 0:
                        for horizon in DEFAULT_HORIZONS:
                            key = f"ret_{horizon}"
                            if labels.get(key) is not None or idx + horizon >= len(closes):
                                continue
                            labels[key] = round((_safe(closes[idx + horizon]) / entry - 1.0) * 100.0, 6)
                            labels[f"label_{horizon}"] = labels[key] > 0
                            labels_updated += 1
                target.write(json.dumps(row, ensure_ascii=False, cls=_JsonEncoder) + "\n")

        if labels_updated or malformed:
            current_stat = dataset_file.stat()
            if (current_stat.st_size, current_stat.st_mtime_ns) != (
                source_stat.st_size,
                source_stat.st_mtime_ns,
            ):
                raise RuntimeError("research dataset changed during label pass")
            _replace_dataset_with_retry(temp_file, dataset_file, source_stat)
            if malformed:
                quarantine_file.parent.mkdir(parents=True, exist_ok=True)
                with quarantine_file.open("a", encoding="utf-8") as target:
                    for row in malformed:
                        target.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            temp_file.unlink(missing_ok=True)
    except Exception:
        temp_file.unlink(missing_ok=True)
        raise

    return {
        "labels_updated": labels_updated,
        "malformed_rows_quarantined": len(malformed),
        "rows_scanned": rows_scanned,
    }


def _replace_dataset_with_retry(
    temp_file: Path,
    dataset_file: Path,
    source_stat: Any,
    *,
    delays: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
) -> None:
    for attempt in range(len(delays) + 1):
        current_stat = dataset_file.stat()
        if (current_stat.st_size, current_stat.st_mtime_ns) != (
            source_stat.st_size,
            source_stat.st_mtime_ns,
        ):
            raise RuntimeError("research dataset changed before atomic replacement")
        try:
            temp_file.replace(dataset_file)
            return
        except PermissionError:
            if attempt >= len(delays):
                raise
            time.sleep(delays[attempt])


def _incomplete_label_ranges(dataset_file: Path) -> dict[tuple[str, str], dict[str, int]]:
    ranges: dict[tuple[str, str], dict[str, int]] = {}
    if not dataset_file.exists():
        return ranges
    with dataset_file.open("r", encoding="utf-8", errors="replace") as source:
        for raw_line in source:
            try:
                row = json.loads(raw_line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            labels = row.get("labels") or {}
            if all(labels.get(f"ret_{horizon}") is not None for horizon in DEFAULT_HORIZONS):
                continue
            symbol = str(row.get("sym") or "")
            tf = str(row.get("tf") or "")
            bar_ts = int(row.get("bar_ts") or 0)
            if not symbol or not tf or bar_ts <= 0:
                continue
            current = ranges.setdefault(
                (symbol, tf),
                {"min_bar_ts": bar_ts, "max_bar_ts": bar_ts, "rows": 0},
            )
            current["min_bar_ts"] = min(current["min_bar_ts"], bar_ts)
            current["max_bar_ts"] = max(current["max_bar_ts"], bar_ts)
            current["rows"] += 1
    return ranges


def _timeframe_ms(tf: str) -> int:
    value = str(tf or "").strip().lower()
    if len(value) < 2 or value[-1] not in TIMEFRAME_UNITS_MS:
        raise ValueError(f"unsupported timeframe: {tf}")
    amount = int(value[:-1])
    if amount <= 0:
        raise ValueError(f"unsupported timeframe: {tf}")
    return amount * TIMEFRAME_UNITS_MS[value[-1]]


async def backfill_mature_labels(
    *,
    dataset_file: Path = DATASET_FILE,
    concurrency: int = 4,
) -> dict[str, Any]:
    """Fetch the maximum required Binance history and label every mature row."""
    ranges = await run_cpu(_incomplete_label_ranges, dataset_file)
    result: dict[str, Any] = {
        "started_at": _utc_now_iso(),
        "finished_at": None,
        "pairs_requested": len(ranges),
        "pairs_fetched": 0,
        "pairs_failed": 0,
        "bars_fetched": 0,
        "labels_updated": 0,
        "malformed_rows_quarantined": 0,
        "rows_scanned": 0,
    }
    if not ranges:
        result["finished_at"] = _utc_now_iso()
        return result

    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    market_data: dict[tuple[str, str], Any] = {}

    async def fetch_pair(
        session: aiohttp.ClientSession,
        key: tuple[str, str],
        span: dict[str, int],
    ) -> tuple[tuple[str, str], Any | None]:
        symbol, tf = key
        bar_ms = _timeframe_ms(tf)
        start_ms = int(span["min_bar_ts"])
        end_ms = int(span["max_bar_ts"]) + (max(DEFAULT_HORIZONS) + 2) * bar_ms
        async with semaphore:
            data = await _fetch_range_klines(session, symbol, tf, start_ms, end_ms)
        return key, data

    headers = {"User-Agent": "Mozilla/5.0"}
    connector = aiohttp.TCPConnector(limit=max(4, int(concurrency) * 2))
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(headers=headers, connector=connector, timeout=timeout) as session:
        fetched = await asyncio.gather(
            *[fetch_pair(session, key, span) for key, span in ranges.items()]
        )
    for key, data in fetched:
        if data is None or len(data) == 0:
            result["pairs_failed"] += 1
            continue
        market_data[key] = data
        result["pairs_fetched"] += 1
        result["bars_fetched"] += len(data)

    label_result = await run_cpu(_fill_mature_labels_batch, dataset_file, market_data)
    result.update(label_result)
    result["finished_at"] = _utc_now_iso()
    return result


async def _fetch_range_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    tf: str,
    start_ms: int,
    end_ms: int,
) -> np.ndarray | None:
    rows: list[list[Any]] = []
    cursor = int(start_ms)
    bar_ms = _timeframe_ms(tf)
    while cursor <= end_ms:
        batch: Any = None
        for attempt in range(3):
            try:
                batch = await _fetch_json(
                    session,
                    f"{BINANCE_API}/api/v3/klines",
                    {
                        "symbol": symbol,
                        "interval": tf,
                        "startTime": cursor,
                        "endTime": int(end_ms),
                        "limit": 1000,
                    },
                )
                break
            except Exception:
                if attempt == 2:
                    log.warning("research label backfill fetch failed for %s %s", symbol, tf)
                    return None
                await asyncio.sleep(0.5 * (attempt + 1))
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + bar_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
    if not rows:
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


async def _fetch_json(session: aiohttp.ClientSession, url: str, params: dict[str, Any] | None = None) -> Any:
    timeout = aiohttp.ClientTimeout(total=30)
    async with session.get(url, params=params, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.json()


async def _fetch_exchange_symbols(session: aiohttp.ClientSession) -> list[str]:
    payload = await _fetch_json(session, f"{BINANCE_API}/api/v3/exchangeInfo")
    out: list[str] = []
    for item in payload.get("symbols", []):
        symbol = str(item.get("symbol") or "").upper()
        if item.get("status") != "TRADING":
            continue
        if not _is_research_symbol(symbol):
            continue
        out.append(symbol)
    return out


async def _fetch_all_tickers(session: aiohttp.ClientSession) -> dict[str, dict[str, Any]]:
    payload = await _fetch_json(session, f"{BINANCE_API}/api/v3/ticker/24hr", {"type": "FULL"})
    return {str(item.get("symbol") or "").upper(): item for item in payload if isinstance(item, dict)}


def _is_research_symbol(symbol: str) -> bool:
    symbol = str(symbol or "").strip().upper()
    if not symbol.endswith("USDT"):
        return False
    if not symbol.isascii() or not symbol.isalnum():
        return False
    if symbol in STABLE_OR_NON_TARGET:
        return False
    if any(symbol.endswith(suffix) for suffix in LEVERAGED_TOKENS):
        return False
    return True


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict) and rec.get("id"):
                out.add(str(rec["id"]))
    return out


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, cls=_JsonEncoder) + "\n")


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _record_id(symbol: str, tf: str, bar_ts: int) -> str:
    return f"{symbol}_{tf}_{int(bar_ts)}"


def _feature(feat: dict[str, Any], key: str, i: int) -> float:
    try:
        return _safe(feat[key][i])
    except Exception:
        return 0.0


def _safe(value: Any) -> float:
    try:
        out = float(value)
        return out if out == out else 0.0
    except Exception:
        return 0.0


class _JsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


async def run_forever() -> None:
    interval_sec = int(getattr(config, "RESEARCH_UNIVERSE_SHADOW_INTERVAL_SEC", 15 * 60))
    while True:
        try:
            await run_once()
        except Exception as exc:
            log.exception("research universe shadow cycle failed: %s", exc)
        await asyncio.sleep(max(60, interval_sec))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run research-universe shadow collection once.")
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--timeframes", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--min-quote-volume", type=float, default=None)
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument("--status", type=Path, default=STATUS_FILE)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    tfs = tuple(x.strip() for x in args.timeframes.split(",") if x.strip()) or None
    result = asyncio.run(
        run_once(
            dataset_file=args.dataset,
            status_file=args.status,
            max_symbols=args.max_symbols,
            timeframes=tfs,
            batch_size=args.batch_size,
            min_quote_volume=args.min_quote_volume,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.as_json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
