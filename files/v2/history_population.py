from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Iterable

import aiohttp

from .history import BAR_MS, CanonicalBar
from .history_store import LocalHistoryStore


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


@dataclass(frozen=True)
class PopulationRequest:
    symbol: str
    timeframe: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class PopulationRow:
    symbol: str
    timeframe: str
    requested_start_ms: int
    requested_end_ms: int
    fetched_rows: int
    stored_rows: int
    contiguous: bool
    missing_intervals: int
    expected_rows: int
    full_window_covered: bool
    source: str
    error: str | None = None


@dataclass(frozen=True)
class PopulationReport:
    requested_rows: tuple[PopulationRow, ...]
    target_days: int
    valid_symbol_ratio: float
    valid_symbols: tuple[str, ...]
    coverage_passed: bool

    def to_dict(self) -> dict:
        return {
            "requested_rows": [asdict(row) for row in self.requested_rows],
            "target_days": self.target_days,
            "valid_symbol_ratio": self.valid_symbol_ratio,
            "valid_symbols": list(self.valid_symbols),
            "coverage_passed": self.coverage_passed,
        }


async def fetch_binance_klines(
    session: aiohttp.ClientSession,
    request: PopulationRequest,
) -> tuple[CanonicalBar, ...]:
    rows: list[list] = []
    cursor = request.start_ms
    step = BAR_MS[request.timeframe]
    while cursor < request.end_ms:
        params = {
            "symbol": request.symbol,
            "interval": request.timeframe,
            "startTime": cursor,
            "endTime": request.end_ms,
            "limit": 1000,
        }
        async with session.get(
            BINANCE_KLINES_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            response.raise_for_status()
            batch = await response.json()
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + step
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
    return tuple(
        CanonicalBar(
            symbol=request.symbol,
            timeframe=request.timeframe,
            open_ts_ms=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        for row in rows
        if int(row[0]) < request.end_ms
    )


def build_requests(
    symbols: Iterable[str],
    timeframes: Iterable[str],
    *,
    days: int,
    end: datetime | None = None,
) -> tuple[PopulationRequest, ...]:
    end_dt = end or datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    return tuple(
        PopulationRequest(
            symbol=symbol,
            timeframe=timeframe,
            start_ms=int(start_dt.timestamp() * 1000),
            end_ms=int(end_dt.timestamp() * 1000),
        )
        for symbol in symbols
        for timeframe in timeframes
    )


async def populate_requests(
    store: LocalHistoryStore,
    requests: Iterable[PopulationRequest],
    *,
    target_days: int,
    min_valid_symbol_ratio: float = 0.80,
    fetcher: Callable[
        [aiohttp.ClientSession, PopulationRequest],
        Awaitable[tuple[CanonicalBar, ...]],
    ] = fetch_binance_klines,
    source: str = "binance_rest_klines",
) -> PopulationReport:
    request_rows: list[PopulationRow] = []
    request_list = tuple(requests)
    async with aiohttp.ClientSession() as session:
        for request in request_list:
            try:
                bars = await fetcher(session, request)
                slice_ = store.upsert(
                    request.symbol,
                    request.timeframe,
                    bars,
                    source=source,
                )
                request_rows.append(
                    PopulationRow(
                        symbol=request.symbol,
                        timeframe=request.timeframe,
                        requested_start_ms=request.start_ms,
                        requested_end_ms=request.end_ms,
                        fetched_rows=len(bars),
                        stored_rows=len(slice_.bars),
                        contiguous=slice_.is_contiguous,
                        missing_intervals=len(slice_.continuity.missing_intervals),
                        expected_rows=max(0, (request.end_ms - request.start_ms) // BAR_MS[request.timeframe]),
                        full_window_covered=len(bars)
                        >= max(0, (request.end_ms - request.start_ms) // BAR_MS[request.timeframe]),
                        source=source,
                    )
                )
            except Exception as exc:
                request_rows.append(
                    PopulationRow(
                        symbol=request.symbol,
                        timeframe=request.timeframe,
                        requested_start_ms=request.start_ms,
                        requested_end_ms=request.end_ms,
                        fetched_rows=0,
                        stored_rows=0,
                        contiguous=False,
                        missing_intervals=0,
                        expected_rows=max(0, (request.end_ms - request.start_ms) // BAR_MS[request.timeframe]),
                        full_window_covered=False,
                        source=source,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

    required_timeframes = {row.timeframe for row in request_rows}
    by_symbol: dict[str, list[PopulationRow]] = {}
    for row in request_rows:
        by_symbol.setdefault(row.symbol, []).append(row)
    valid_symbols = []
    for symbol, rows in by_symbol.items():
        covered = {
            row.timeframe
            for row in rows
            if row.contiguous and row.full_window_covered and row.error is None
        }
        if covered == required_timeframes:
            valid_symbols.append(symbol)
    ratio = round(len(valid_symbols) / len(by_symbol), 4) if by_symbol else 0.0
    return PopulationReport(
        requested_rows=tuple(request_rows),
        target_days=target_days,
        valid_symbol_ratio=ratio,
        valid_symbols=tuple(sorted(valid_symbols)),
        coverage_passed=ratio >= min_valid_symbol_ratio,
    )
