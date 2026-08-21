from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import aiohttp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_CACHE = ROOT / ".runtime" / "price_cluster_cache"
DEFAULT_TAIL_CACHE = ROOT / "files" / ".runtime" / "external_top50_history"
DEFAULT_WATCHLIST = ROOT / "files" / "watchlist.json"
DEFAULT_REPORT = ROOT / "files" / ".runtime" / "reports" / "external_top50_screen_validation_latest.json"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
CACHE_RE = re.compile(r"^(?P<symbol>.+)_1h_(?P<start>\d+)_(?P<end>\d+)\.json$")
VARIANTS = ("current_rank", "static_target", "screen_v1")


@dataclass(frozen=True, slots=True)
class Bar:
    open_ts_ms: int
    close: float
    quote_volume: float
    taker_buy_quote: float

    @property
    def close_ts_ms(self) -> int:
        return self.open_ts_ms + HOUR_MS


@dataclass(frozen=True, slots=True)
class MarketRow:
    symbol: str
    current_return: float
    static_target_return: float
    ret_1h: float
    ret_3h: float
    volume_accel: float
    taker_buy_ratio: float
    target_return: float


@dataclass(frozen=True, slots=True)
class Candidate:
    symbol: str
    current_rank: int
    current_return: float
    static_target_return: float
    ret_1h: float
    ret_3h: float
    volume_accel: float
    taker_buy_ratio: float
    score_screen_v1: float
    target_rank: int
    is_target_top: bool


@dataclass(frozen=True, slots=True)
class DaySnapshot:
    local_day: str
    market_symbol_count: int
    watchlist_symbol_count: int
    current_cutoff_return: float
    target_cutoff_return: float
    candidates: tuple[Candidate, ...]
    target_top_symbols: frozenset[str]
    target_entrant_symbols: frozenset[str]


@dataclass(frozen=True, slots=True)
class CacheFile:
    path: Path
    symbol: str
    start_ms: int
    end_ms: int


def score_screen_v1(
    static_target_return: float,
    ret_1h: float,
    ret_3h: float,
    volume_accel: float,
) -> float:
    volume_term = min(1.0, max(-0.5, (volume_accel - 1.0) * 0.35))
    return static_target_return + 0.20 * ret_1h + 0.05 * ret_3h + volume_term


def merge_binance_rows(batches: Iterable[Iterable[Sequence[object]]]) -> tuple[Bar, ...]:
    by_open: dict[int, Bar] = {}
    for batch in batches:
        for row in batch:
            if len(row) < 11:
                continue
            try:
                bar = Bar(
                    open_ts_ms=int(row[0]),
                    close=float(row[4]),
                    quote_volume=float(row[7]),
                    taker_buy_quote=float(row[10]),
                )
            except (TypeError, ValueError):
                continue
            if bar.close > 0:
                by_open[bar.open_ts_ms] = bar
    return tuple(by_open[key] for key in sorted(by_open))


def last_bar_at_or_before(history: Sequence[Bar], cutoff_ms: int) -> Bar | None:
    if not history:
        return None
    closes = [bar.close_ts_ms for bar in history]
    index = bisect_right(closes, cutoff_ms) - 1
    return history[index] if index >= 0 else None


def _bar_index_at_or_before(
    history: Sequence[Bar],
    cutoff_ms: int,
    close_times: Sequence[int] | None = None,
) -> int:
    closes = close_times if close_times is not None else [bar.close_ts_ms for bar in history]
    return bisect_right(closes, cutoff_ms) - 1


def _fresh_bar(
    history: Sequence[Bar],
    cutoff_ms: int,
    *,
    close_times: Sequence[int] | None = None,
    max_age_ms: int = HOUR_MS,
) -> tuple[int, Bar] | None:
    index = _bar_index_at_or_before(history, cutoff_ms, close_times)
    if index < 0:
        return None
    bar = history[index]
    if cutoff_ms - bar.close_ts_ms < 0 or cutoff_ms - bar.close_ts_ms >= max_age_ms:
        return None
    return index, bar


def _safe_return(current: float, previous: float) -> float:
    return 100.0 * (current / previous - 1.0) if current > 0 and previous > 0 else math.nan


def _extract_market_row(
    symbol: str,
    history: Sequence[Bar],
    *,
    observation_ms: int,
    target_ms: int,
    close_times: Sequence[int] | None = None,
) -> MarketRow | None:
    observation = _fresh_bar(history, observation_ms, close_times=close_times)
    observation_base = _fresh_bar(history, observation_ms - DAY_MS, close_times=close_times)
    target_base = _fresh_bar(history, target_ms - DAY_MS, close_times=close_times)
    target = _fresh_bar(history, target_ms, close_times=close_times)
    if not all((observation, observation_base, target_base, target)):
        return None
    obs_index, obs_bar = observation
    if obs_index < 4:
        return None
    one_back = history[obs_index - 1]
    three_back = history[obs_index - 3]
    if obs_bar.open_ts_ms - one_back.open_ts_ms != HOUR_MS:
        return None
    if obs_bar.open_ts_ms - three_back.open_ts_ms != 3 * HOUR_MS:
        return None
    previous_volume = [history[index].quote_volume for index in range(obs_index - 4, obs_index)]
    if any(value <= 0 for value in previous_volume) or obs_bar.quote_volume <= 0:
        return None
    volume_accel = obs_bar.quote_volume / mean(previous_volume)
    return MarketRow(
        symbol=symbol,
        current_return=_safe_return(obs_bar.close, observation_base[1].close),
        static_target_return=_safe_return(obs_bar.close, target_base[1].close),
        ret_1h=_safe_return(obs_bar.close, one_back.close),
        ret_3h=_safe_return(obs_bar.close, three_back.close),
        volume_accel=volume_accel,
        taker_buy_ratio=obs_bar.taker_buy_quote / obs_bar.quote_volume,
        target_return=_safe_return(target[1].close, target_base[1].close),
    )


def _snapshot_from_market_rows(
    rows: Sequence[MarketRow],
    *,
    watchlist: set[str],
    local_day: date,
    top_n: int,
    min_market_symbols: int,
    min_watchlist_symbols: int,
) -> DaySnapshot | None:
    finite_rows = [
        row for row in rows
        if all(math.isfinite(value) for value in (
            row.current_return, row.static_target_return, row.ret_1h,
            row.ret_3h, row.volume_accel, row.target_return,
        ))
    ]
    valid_watchlist = {row.symbol for row in finite_rows if row.symbol in watchlist}
    if len(finite_rows) < min_market_symbols or len(valid_watchlist) < min_watchlist_symbols:
        return None
    current_order = sorted(finite_rows, key=lambda row: (-row.current_return, row.symbol))
    target_order = sorted(finite_rows, key=lambda row: (-row.target_return, row.symbol))
    current_ranks = {row.symbol: index + 1 for index, row in enumerate(current_order)}
    target_ranks = {row.symbol: index + 1 for index, row in enumerate(target_order)}
    target_top = frozenset(row.symbol for row in target_order[:top_n])
    candidates = []
    for row in finite_rows:
        current_rank = current_ranks[row.symbol]
        if row.symbol not in watchlist or current_rank <= top_n:
            continue
        candidates.append(Candidate(
            symbol=row.symbol,
            current_rank=current_rank,
            current_return=row.current_return,
            static_target_return=row.static_target_return,
            ret_1h=row.ret_1h,
            ret_3h=row.ret_3h,
            volume_accel=row.volume_accel,
            taker_buy_ratio=row.taker_buy_ratio,
            score_screen_v1=score_screen_v1(
                row.static_target_return, row.ret_1h, row.ret_3h, row.volume_accel,
            ),
            target_rank=target_ranks[row.symbol],
            is_target_top=row.symbol in target_top,
        ))
    entrant_symbols = frozenset(
        symbol for symbol in target_top
        if symbol in watchlist and current_ranks.get(symbol, 0) > top_n
    )
    return DaySnapshot(
        local_day=local_day.isoformat(),
        market_symbol_count=len(finite_rows),
        watchlist_symbol_count=len(valid_watchlist),
        current_cutoff_return=current_order[top_n - 1].current_return,
        target_cutoff_return=target_order[top_n - 1].target_return,
        candidates=tuple(sorted(candidates, key=lambda item: item.symbol)),
        target_top_symbols=target_top,
        target_entrant_symbols=entrant_symbols,
    )


def build_day_snapshot(
    histories: Mapping[str, Sequence[Bar]],
    *,
    watchlist: set[str],
    local_day: date,
    timezone_name: str = "Europe/Budapest",
    observation_time: time = time(12, 15),
    target_time: time = time(23, 0),
    top_n: int = 50,
    min_market_symbols: int = 200,
    min_watchlist_symbols: int = 50,
) -> DaySnapshot | None:
    tz = ZoneInfo(timezone_name)
    observation_dt = datetime.combine(local_day, observation_time, tzinfo=tz)
    target_dt = datetime.combine(local_day, target_time, tzinfo=tz)
    observation_ms = int(observation_dt.astimezone(timezone.utc).timestamp() * 1000)
    target_ms = int(target_dt.astimezone(timezone.utc).timestamp() * 1000)
    rows = []
    for symbol, history in histories.items():
        close_times = tuple(bar.close_ts_ms for bar in history)
        row = _extract_market_row(
            symbol,
            history,
            observation_ms=observation_ms,
            target_ms=target_ms,
            close_times=close_times,
        )
        if row is not None:
            rows.append(row)
    return _snapshot_from_market_rows(
        rows,
        watchlist=watchlist,
        local_day=local_day,
        top_n=top_n,
        min_market_symbols=min_market_symbols,
        min_watchlist_symbols=min_watchlist_symbols,
    )


def wilson_interval(hits: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = hits / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _variant_order(candidates: Sequence[Candidate], variant: str) -> list[Candidate]:
    if variant == "current_rank":
        key = lambda item: (-item.current_return, item.symbol)
    elif variant == "static_target":
        key = lambda item: (-item.static_target_return, item.symbol)
    elif variant == "screen_v1":
        key = lambda item: (-item.score_screen_v1, item.symbol)
    else:
        raise ValueError(f"unsupported variant: {variant}")
    return sorted(candidates, key=key)


def _bootstrap_delta(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, object] | None:
    if len(left) != len(right) or not left:
        return None
    differences = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    estimates = []
    for _ in range(max(1, samples)):
        estimates.append(mean(differences[rng.randrange(len(differences))] for _ in differences))
    estimates.sort()
    lo = estimates[int(0.025 * (len(estimates) - 1))]
    hi = estimates[int(0.975 * (len(estimates) - 1))]
    return {"mean": round(mean(differences), 6), "bootstrap95": [round(lo, 6), round(hi, 6)]}


def evaluate_snapshots(
    snapshots: Sequence[DaySnapshot | None],
    *,
    selection_size: int = 10,
    bootstrap_samples: int = 5000,
) -> dict[str, dict[str, object]]:
    valid = [snapshot for snapshot in snapshots if snapshot is not None and snapshot.candidates]
    candidate_count = sum(len(snapshot.candidates) for snapshot in valid)
    candidate_hits = sum(len(snapshot.target_entrant_symbols) for snapshot in valid)
    base_rate = candidate_hits / candidate_count if candidate_count else None
    daily_precision: dict[str, list[float]] = {variant: [] for variant in VARIANTS}
    result: dict[str, dict[str, object]] = {}
    for variant in VARIANTS:
        top1_hits = 0
        top1_days = 0
        topk_hits = 0
        selections = 0
        entrant_hits = 0
        entrant_total = 0
        target_ranks: list[int] = []
        for snapshot in valid:
            ordered = _variant_order(snapshot.candidates, variant)
            selected = ordered[:selection_size]
            if not selected:
                continue
            hits = sum(item.is_target_top for item in selected)
            top1_days += 1
            top1_hits += int(selected[0].is_target_top)
            topk_hits += hits
            selections += len(selected)
            entrant_hits += hits
            entrant_total += len(snapshot.target_entrant_symbols)
            target_ranks.extend(item.target_rank for item in selected)
            daily_precision[variant].append(hits / len(selected))
        precision = topk_hits / selections if selections else None
        recall = entrant_hits / entrant_total if entrant_total else None
        result[variant] = {
            "eligible_days": top1_days,
            "top1": {
                "hits": top1_hits,
                "days": top1_days,
                "rate": round(top1_hits / top1_days, 6) if top1_days else None,
                "wilson95": wilson_interval(top1_hits, top1_days),
            },
            "topk": {
                "k": selection_size,
                "hits": topk_hits,
                "selections": selections,
                "precision": round(precision, 6) if precision is not None else None,
                "wilson95": wilson_interval(topk_hits, selections),
            },
            "target_entrant_recall": {
                "hits": entrant_hits,
                "entrants": entrant_total,
                "recall": round(recall, 6) if recall is not None else None,
            },
            "candidate_base_rate": {
                "hits": candidate_hits,
                "candidates": candidate_count,
                "rate": round(base_rate, 6) if base_rate is not None else None,
            },
            "precision_lift_over_base": (
                round(precision / base_rate, 6)
                if precision is not None and base_rate not in (None, 0.0) else None
            ),
            "selected_target_rank_mean": round(mean(target_ranks), 3) if target_ranks else None,
        }
    control = daily_precision["current_rank"]
    static = daily_precision["static_target"]
    for index, variant in enumerate(VARIANTS):
        result[variant]["paired_precision_delta_vs_current_rank"] = _bootstrap_delta(
            daily_precision[variant], control,
            samples=bootstrap_samples,
            seed=1103 + index,
        )
        result[variant]["paired_precision_delta_vs_static_target"] = _bootstrap_delta(
            daily_precision[variant], static,
            samples=bootstrap_samples,
            seed=2103 + index,
        )
    return result


def _parse_cache_file(path: Path) -> CacheFile | None:
    match = CACHE_RE.match(path.name)
    if not match:
        return None
    return CacheFile(
        path=path,
        symbol=match.group("symbol"),
        start_ms=int(match.group("start")),
        end_ms=int(match.group("end")),
    )


def discover_cache_files(cache_dirs: Iterable[Path]) -> dict[str, tuple[CacheFile, ...]]:
    per_source: dict[tuple[str, Path], list[CacheFile]] = defaultdict(list)
    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue
        for path in cache_dir.glob("*_1h_*.json"):
            parsed = _parse_cache_file(path)
            if parsed is not None:
                per_source[(parsed.symbol, cache_dir.resolve())].append(parsed)
    grouped: dict[str, list[CacheFile]] = defaultdict(list)
    for (symbol, _source), files in per_source.items():
        # Snapshot caches often contain several almost-identical year files
        # shifted by a few hours. Parsing all of them multiplies I/O without
        # adding eligible local days, so retain the widest/latest snapshot per
        # source directory. A separately collected tail remains a second file.
        grouped[symbol].append(max(
            files,
            key=lambda item: (item.end_ms - item.start_ms, item.end_ms, -item.start_ms),
        ))
    selected: dict[str, tuple[CacheFile, ...]] = {}
    for symbol, files in grouped.items():
        ordered = sorted(files, key=lambda item: (item.start_ms, -item.end_ms, str(item.path)))
        keep: list[CacheFile] = []
        covered_until = -1
        for item in ordered:
            if item.end_ms <= covered_until:
                continue
            keep.append(item)
            covered_until = item.end_ms
        selected[symbol] = tuple(keep)
    return selected


def _load_json_rows(path: Path) -> list[list[object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"cache is not a list: {path}")
    return payload


def _date_range(start_day: date, end_day: date) -> Iterable[date]:
    cursor = start_day
    while cursor <= end_day:
        yield cursor
        cursor += timedelta(days=1)


def build_snapshots_from_cache(
    selected_files: Mapping[str, Sequence[CacheFile]],
    *,
    watchlist: set[str],
    start_day: date,
    end_day: date,
    timezone_name: str,
    top_n: int,
    min_market_symbols: int,
    min_watchlist_symbols: int,
) -> tuple[list[DaySnapshot], dict[str, object]]:
    tz = ZoneInfo(timezone_name)
    days = tuple(_date_range(start_day, end_day))
    clocks = {}
    for local_day in days:
        observation_dt = datetime.combine(local_day, time(12, 15), tzinfo=tz).astimezone(timezone.utc)
        target_dt = datetime.combine(local_day, time(23, 0), tzinfo=tz).astimezone(timezone.utc)
        clocks[local_day] = (
            int(observation_dt.timestamp() * 1000),
            int(target_dt.timestamp() * 1000),
        )
    by_day: dict[date, list[MarketRow]] = defaultdict(list)
    used_files: list[Path] = []
    malformed_files: list[str] = []
    symbols_with_rows = 0
    for symbol in sorted(selected_files):
        batches = []
        for cache_file in selected_files[symbol]:
            try:
                batches.append(_load_json_rows(cache_file.path))
                used_files.append(cache_file.path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                malformed_files.append(f"{cache_file.path}: {type(exc).__name__}: {exc}")
        history = merge_binance_rows(batches)
        if not history:
            continue
        symbols_with_rows += 1
        close_times = tuple(bar.close_ts_ms for bar in history)
        for local_day, (observation_ms, target_ms) in clocks.items():
            row = _extract_market_row(
                symbol,
                history,
                observation_ms=observation_ms,
                target_ms=target_ms,
                close_times=close_times,
            )
            if row is not None:
                by_day[local_day].append(row)
    snapshots = []
    rejected_days = []
    for local_day in days:
        snapshot = _snapshot_from_market_rows(
            by_day.get(local_day, ()),
            watchlist=watchlist,
            local_day=local_day,
            top_n=top_n,
            min_market_symbols=min_market_symbols,
            min_watchlist_symbols=min_watchlist_symbols,
        )
        if snapshot is None:
            rejected_days.append(local_day.isoformat())
        else:
            snapshots.append(snapshot)
    digest = hashlib.sha256()
    for path in sorted(set(used_files), key=str):
        digest.update(str(path.resolve()).encode("utf-8"))
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    provenance = {
        "cache_symbol_count": len(selected_files),
        "symbols_with_rows": symbols_with_rows,
        "used_file_count": len(set(used_files)),
        "used_content_hash": digest.hexdigest(),
        "malformed_file_count": len(malformed_files),
        "malformed_files": malformed_files[:20],
        "requested_days": len(days),
        "eligible_days": len(snapshots),
        "rejected_days": rejected_days,
    }
    return snapshots, provenance


async def _fetch_json(session: aiohttp.ClientSession, url: str, params: dict[str, object] | None = None) -> object:
    delay = 1.0
    for attempt in range(6):
        async with session.get(url, params=params) as response:
            if response.status in (418, 429):
                await asyncio.sleep(float(response.headers.get("Retry-After", delay)))
                delay *= 2.0
                continue
            response.raise_for_status()
            return await response.json()
    raise RuntimeError(f"rate limited after retries: {url}")


async def _fetch_symbol_tail(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> tuple[str, list[list[object]], str | None]:
    rows: list[list[object]] = []
    cursor = start_ms
    try:
        async with semaphore:
            while cursor < end_ms:
                payload = await _fetch_json(session, BINANCE_KLINES_URL, {
                    "symbol": symbol,
                    "interval": "1h",
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                })
                if not isinstance(payload, list) or not payload:
                    break
                rows.extend(payload)
                next_cursor = int(payload[-1][0]) + HOUR_MS
                if next_cursor <= cursor:
                    break
                cursor = next_cursor
                if len(payload) < 1000:
                    break
        return symbol, rows, None
    except Exception as exc:
        return symbol, rows, f"{type(exc).__name__}: {exc}"


async def refresh_tail_cache(
    output_dir: Path,
    *,
    start_ms: int,
    end_ms: int,
    concurrency: int = 8,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=max(2, concurrency))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        exchange_info = await _fetch_json(session, BINANCE_EXCHANGE_INFO_URL)
        symbols = sorted(
            item["symbol"] for item in exchange_info.get("symbols", [])
            if item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and bool(item.get("isSpotTradingAllowed"))
        )
        semaphore = asyncio.Semaphore(max(1, concurrency))
        tasks = [
            _fetch_symbol_tail(
                session, semaphore, symbol=symbol, start_ms=start_ms, end_ms=end_ms,
            )
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks)
    success = 0
    failures = []
    row_count = 0
    for symbol, rows, error in results:
        if error:
            failures.append({"symbol": symbol, "error": error, "partial_rows": len(rows)})
            continue
        target = output_dir / f"{symbol}_1h_{start_ms}_{end_ms}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
        temporary.replace(target)
        success += 1
        row_count += len(rows)
    return {
        "requested_symbols": len(symbols),
        "successful_symbols": success,
        "failed_symbols": len(failures),
        "rows": row_count,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "failures": failures[:30],
    }


def _parse_day(value: str) -> date:
    return date.fromisoformat(value)


def _default_period(selected_files: Mapping[str, Sequence[CacheFile]], timezone_name: str) -> tuple[date, date]:
    starts = [item.start_ms for files in selected_files.values() for item in files]
    ends = [item.end_ms for files in selected_files.values() for item in files]
    if not starts or not ends:
        raise ValueError("no cache files found")
    tz = ZoneInfo(timezone_name)
    start_day = datetime.fromtimestamp(min(starts) / 1000, tz=timezone.utc).astimezone(tz).date() + timedelta(days=2)
    end_day = datetime.fromtimestamp(max(ends) / 1000, tz=timezone.utc).astimezone(tz).date() - timedelta(days=1)
    return start_day, end_day


def _slice_report(snapshots: Sequence[DaySnapshot], *, selection_size: int, bootstrap_samples: int) -> dict[str, object]:
    metrics = evaluate_snapshots(
        snapshots, selection_size=selection_size, bootstrap_samples=bootstrap_samples,
    )
    screen = metrics["screen_v1"]
    controls = [metrics["current_rank"], metrics["static_target"]]
    screen_precision = screen["topk"]["precision"]
    screen_recall = screen["target_entrant_recall"]["recall"]
    improves_both = all(
        (screen_precision or 0.0) > (control["topk"]["precision"] or 0.0)
        or (screen_recall or 0.0) > (control["target_entrant_recall"]["recall"] or 0.0)
        for control in controls
    )
    intervals = [
        screen["paired_precision_delta_vs_current_rank"],
        screen["paired_precision_delta_vs_static_target"],
    ]
    positive_intervals = all(
        item is not None and item["bootstrap95"][0] > 0.0 for item in intervals
    )
    eligible_days = screen["eligible_days"]
    if eligible_days < 30:
        verdict = "INCONCLUSIVE"
        reason = "fewer_than_30_eligible_days"
    elif improves_both and positive_intervals:
        verdict = "SUPPORTED_FOR_FORWARD_SHADOW_ONLY"
        reason = "paired_precision_and_recall_gate_passed"
    elif not improves_both:
        verdict = "REJECTED"
        reason = "does_not_improve_both_baselines"
    else:
        verdict = "INCONCLUSIVE"
        reason = "paired_interval_includes_zero"
    return {"metrics": metrics, "verdict": verdict, "reason": reason}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal external Binance Top-50 screen validation")
    parser.add_argument("--legacy-cache", type=Path, default=DEFAULT_LEGACY_CACHE)
    parser.add_argument("--tail-cache", type=Path, default=DEFAULT_TAIL_CACHE)
    parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--timezone", default="Europe/Budapest")
    parser.add_argument("--start-day", type=_parse_day)
    parser.add_argument("--end-day", type=_parse_day)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--selection-size", type=int, default=10)
    parser.add_argument("--min-market-symbols", type=int, default=200)
    parser.add_argument("--min-watchlist-symbols", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--recent-days", type=int, default=60)
    parser.add_argument("--refresh-start-day", type=_parse_day)
    parser.add_argument("--refresh-end-day", type=_parse_day)
    parser.add_argument("--concurrency", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refresh_report = None
    if args.refresh_start_day:
        refresh_end = args.refresh_end_day or datetime.now(timezone.utc).date()
        refresh_start_ms = int(datetime.combine(args.refresh_start_day, time(0), tzinfo=timezone.utc).timestamp() * 1000)
        refresh_end_ms = int(datetime.combine(refresh_end + timedelta(days=1), time(0), tzinfo=timezone.utc).timestamp() * 1000)
        refresh_report = asyncio.run(refresh_tail_cache(
            args.tail_cache,
            start_ms=refresh_start_ms,
            end_ms=refresh_end_ms,
            concurrency=args.concurrency,
        ))
    selected = discover_cache_files((args.legacy_cache, args.tail_cache))
    default_start, default_end = _default_period(selected, args.timezone)
    start_day = args.start_day or default_start
    end_day = args.end_day or default_end
    watchlist_payload = args.watchlist.read_bytes()
    watchlist = set(json.loads(watchlist_payload.decode("utf-8-sig")))
    snapshots, provenance = build_snapshots_from_cache(
        selected,
        watchlist=watchlist,
        start_day=start_day,
        end_day=end_day,
        timezone_name=args.timezone,
        top_n=args.top_n,
        min_market_symbols=args.min_market_symbols,
        min_watchlist_symbols=args.min_watchlist_symbols,
    )
    recent = snapshots[-max(1, args.recent_days):]
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if snapshots else "insufficient_coverage",
        "contract": {
            "timezone": args.timezone,
            "observation_time_local": "12:15",
            "source_timeframe": "1h",
            "target_time_local": "23:00",
            "top_n": args.top_n,
            "selection_size": args.selection_size,
            "start_day": start_day.isoformat(),
            "end_day": end_day.isoformat(),
            "watchlist_count": len(watchlist),
            "watchlist_sha256": hashlib.sha256(watchlist_payload).hexdigest(),
            "feature_formula": "static_target + 0.20*ret_1h + 0.05*ret_3h + clamp(0.35*(vol_accel-1),-0.5,1.0)",
        },
        "provenance": provenance,
        "refresh": refresh_report,
        "coverage": {
            "eligible_day_count": len(snapshots),
            "first_eligible_day": snapshots[0].local_day if snapshots else None,
            "last_eligible_day": snapshots[-1].local_day if snapshots else None,
            "market_symbols_min": min((item.market_symbol_count for item in snapshots), default=0),
            "market_symbols_max": max((item.market_symbol_count for item in snapshots), default=0),
            "watchlist_symbols_min": min((item.watchlist_symbol_count for item in snapshots), default=0),
            "watchlist_symbols_max": max((item.watchlist_symbol_count for item in snapshots), default=0),
        },
        "full_period": _slice_report(
            snapshots, selection_size=args.selection_size, bootstrap_samples=args.bootstrap_samples,
        ),
        "recent_period": {
            "requested_days": args.recent_days,
            "first_day": recent[0].local_day if recent else None,
            "last_day": recent[-1].local_day if recent else None,
            **_slice_report(
                recent, selection_size=args.selection_size, bootstrap_samples=args.bootstrap_samples,
            ),
        },
        "limitations": [
            "Historical exchangeInfo snapshots are unavailable; historical universe membership is inferred from bars.",
            "The 1h replay observes the last fully closed bar before 12:15 and cannot reproduce 5m order-flow features.",
            "Historical depth trajectories are unavailable and are not used.",
            "Candidate outcomes are correlated within day; paired day bootstrap is primary, per-selection Wilson intervals are descriptive.",
        ],
        "production_effect": "none_research_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"status={report['status']} period={start_day}..{end_day} eligible_days={len(snapshots)}")
    print(f"coverage market={report['coverage']['market_symbols_min']}..{report['coverage']['market_symbols_max']} watchlist={report['coverage']['watchlist_symbols_min']}..{report['coverage']['watchlist_symbols_max']}")
    for period in ("full_period", "recent_period"):
        section = report[period]
        metrics = section["metrics"]["screen_v1"]
        print(
            f"{period}: verdict={section['verdict']} top1={metrics['top1']['hits']}/{metrics['top1']['days']} "
            f"top10={metrics['topk']['hits']}/{metrics['topk']['selections']} "
            f"recall={metrics['target_entrant_recall']['hits']}/{metrics['target_entrant_recall']['entrants']}"
        )
    print(f"report={args.output}")
    return 0 if snapshots else 2


if __name__ == "__main__":
    raise SystemExit(main())
