from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .history import CanonicalBar, HistorySlice, build_history_slice


@dataclass(frozen=True)
class StoredHistoryMetadata:
    symbol: str
    timeframe: str
    source: str
    rows: int
    start_ts_ms: int | None
    end_ts_ms: int | None
    updated_at_utc: str


class LocalHistoryStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _data_path(self, symbol: str, timeframe: str) -> Path:
        return self.root / symbol / f"{timeframe}.jsonl"

    def _meta_path(self, symbol: str, timeframe: str) -> Path:
        return self.root / symbol / f"{timeframe}.meta.json"

    def upsert(
        self,
        symbol: str,
        timeframe: str,
        bars: Iterable[CanonicalBar],
        *,
        source: str,
    ) -> HistorySlice:
        merged = {bar.open_ts_ms: bar for bar in self._read_bars(symbol, timeframe)}
        for bar in bars:
            if bar.symbol != symbol or bar.timeframe != timeframe:
                raise ValueError("bar key does not match target slice")
            merged[bar.open_ts_ms] = bar
        rows = [
            {
                "open_ts_ms": bar.open_ts_ms,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for _, bar in sorted(merged.items())
        ]
        slice_ = build_history_slice(symbol, timeframe, rows, source=source)
        data_path = self._data_path(symbol, timeframe)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        metadata = StoredHistoryMetadata(
            symbol=symbol,
            timeframe=timeframe,
            source=source,
            rows=len(slice_.bars),
            start_ts_ms=slice_.start_ts_ms,
            end_ts_ms=slice_.end_ts_ms,
            updated_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        self._meta_path(symbol, timeframe).write_text(
            json.dumps(asdict(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return slice_

    def load(self, symbol: str, timeframe: str) -> HistorySlice:
        bars = self._read_bars(symbol, timeframe)
        metadata = self.metadata(symbol, timeframe)
        source = metadata.source if metadata is not None else "local_history_store"
        rows = [
            {
                "open_ts_ms": bar.open_ts_ms,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
        return build_history_slice(symbol, timeframe, rows, source=source)

    def metadata(self, symbol: str, timeframe: str) -> StoredHistoryMetadata | None:
        path = self._meta_path(symbol, timeframe)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return StoredHistoryMetadata(**payload)

    def keys(self) -> tuple[tuple[str, str], ...]:
        out = []
        if not self.root.exists():
            return tuple()
        for symbol_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            for path in sorted(symbol_dir.glob("*.jsonl")):
                out.append((symbol_dir.name, path.stem))
        return tuple(out)

    def _read_bars(self, symbol: str, timeframe: str) -> tuple[CanonicalBar, ...]:
        path = self._data_path(symbol, timeframe)
        if not path.exists():
            return tuple()
        rows = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                CanonicalBar(
                    symbol=symbol,
                    timeframe=timeframe,
                    open_ts_ms=int(payload["open_ts_ms"]),
                    open=float(payload["open"]),
                    high=float(payload["high"]),
                    low=float(payload["low"]),
                    close=float(payload["close"]),
                    volume=float(payload["volume"]),
                )
            )
        return tuple(rows)

