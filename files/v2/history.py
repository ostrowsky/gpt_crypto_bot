from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


BAR_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}


@dataclass(frozen=True)
class CanonicalBar:
    symbol: str
    timeframe: str
    open_ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class MissingInterval:
    after_ts_ms: int
    before_ts_ms: int
    missing_bars: int


@dataclass(frozen=True)
class ContinuityReport:
    expected_step_ms: int
    missing_intervals: tuple[MissingInterval, ...]

    @property
    def is_contiguous(self) -> bool:
        return not self.missing_intervals


@dataclass(frozen=True)
class HistorySlice:
    symbol: str
    timeframe: str
    bars: tuple[CanonicalBar, ...]
    source: str
    continuity: ContinuityReport

    @property
    def start_ts_ms(self) -> int | None:
        return self.bars[0].open_ts_ms if self.bars else None

    @property
    def end_ts_ms(self) -> int | None:
        return self.bars[-1].open_ts_ms if self.bars else None

    @property
    def is_contiguous(self) -> bool:
        return bool(self.bars) and self.continuity.is_contiguous


def _to_bar(symbol: str, timeframe: str, row: Mapping[str, object]) -> CanonicalBar:
    return CanonicalBar(
        symbol=symbol,
        timeframe=timeframe,
        open_ts_ms=int(row["open_ts_ms"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
    )


def build_history_slice(
    symbol: str,
    timeframe: str,
    rows: Iterable[Mapping[str, object]],
    *,
    source: str,
) -> HistorySlice:
    if timeframe not in BAR_MS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    by_ts = {}
    for row in rows:
        bar = _to_bar(symbol, timeframe, row)
        by_ts[bar.open_ts_ms] = bar
    bars = tuple(by_ts[ts] for ts in sorted(by_ts))
    continuity = validate_continuity(bars, timeframe)
    return HistorySlice(
        symbol=symbol,
        timeframe=timeframe,
        bars=bars,
        source=source,
        continuity=continuity,
    )


def validate_continuity(
    bars: Iterable[CanonicalBar],
    timeframe: str,
) -> ContinuityReport:
    if timeframe not in BAR_MS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    ordered = tuple(sorted(bars, key=lambda bar: bar.open_ts_ms))
    step = BAR_MS[timeframe]
    missing = []
    for prev, current in zip(ordered, ordered[1:]):
        delta = current.open_ts_ms - prev.open_ts_ms
        if delta == step:
            continue
        if delta < step or delta % step != 0:
            raise ValueError("history contains overlapping or misaligned bars")
        missing.append(
            MissingInterval(
                after_ts_ms=prev.open_ts_ms,
                before_ts_ms=current.open_ts_ms,
                missing_bars=(delta // step) - 1,
            )
        )
    return ContinuityReport(expected_step_ms=step, missing_intervals=tuple(missing))
