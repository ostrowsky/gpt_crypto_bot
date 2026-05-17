from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping
from zoneinfo import ZoneInfo

from .dataset import Observation, TransitionRecord
from .state import Action


BAR_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
}
DEFAULT_TZ = ZoneInfo("Europe/Budapest")


@dataclass(frozen=True)
class ObservationSequence:
    symbol: str
    timeframe: str
    local_day: str
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class SequenceDatasetBuildResult:
    sequences: tuple[ObservationSequence, ...]
    transitions: tuple[TransitionRecord, ...]
    summary: Mapping[str, object]


def _iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            yield row


def _local_day(ts_ms: int, tz: ZoneInfo = DEFAULT_TZ) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(tz).date().isoformat()


def _to_observation(row: Mapping[str, object], tz: ZoneInfo = DEFAULT_TZ) -> Observation | None:
    symbol = str(row.get("sym") or "")
    timeframe = str(row.get("tf") or "")
    try:
        ts_ms = int(row.get("bar_ts"))
    except Exception:
        return None
    features = row.get("f")
    if not symbol or timeframe not in BAR_MS or not isinstance(features, Mapping):
        return None
    clean_features = {
        str(key): float(value)
        for key, value in features.items()
        if isinstance(value, (int, float))
    }
    teacher = row.get("teacher")
    return Observation(
        symbol=symbol,
        timeframe=timeframe,
        ts_ms=ts_ms,
        features=clean_features,
        local_day=_local_day(ts_ms, tz),
        teacher=teacher if isinstance(teacher, Mapping) else None,
    )


def _split_contiguous(observations: Iterable[Observation]) -> tuple[list[list[Observation]], int]:
    chunks: list[list[Observation]] = []
    current: list[Observation] = []
    gap_breaks = 0
    for obs in observations:
        if not current:
            current = [obs]
            continue
        prev = current[-1]
        expected_gap = BAR_MS[obs.timeframe]
        if obs.ts_ms - prev.ts_ms != expected_gap:
            chunks.append(current)
            current = [obs]
            gap_breaks += 1
            continue
        current.append(obs)
    if current:
        chunks.append(current)
    return chunks, gap_breaks


def build_from_rows(rows: Iterable[Mapping[str, object]], tz: ZoneInfo = DEFAULT_TZ) -> SequenceDatasetBuildResult:
    rows_read = 0
    rows_rejected = 0
    duplicates_removed = 0
    grouped: dict[tuple[str, str, str], dict[int, Observation]] = defaultdict(dict)

    for row in rows:
        rows_read += 1
        obs = _to_observation(row, tz)
        if obs is None:
            rows_rejected += 1
            continue
        key = (obs.symbol, obs.timeframe, str(obs.local_day))
        if obs.ts_ms in grouped[key]:
            duplicates_removed += 1
            continue
        grouped[key][obs.ts_ms] = obs

    sequences: list[ObservationSequence] = []
    transitions: list[TransitionRecord] = []
    gap_breaks = 0
    days = set()
    symbols = set()

    for (symbol, timeframe, local_day), by_ts in sorted(grouped.items()):
        ordered = [by_ts[ts] for ts in sorted(by_ts)]
        chunks, chunk_gap_breaks = _split_contiguous(ordered)
        gap_breaks += chunk_gap_breaks
        for chunk in chunks:
            if not chunk:
                continue
            days.add(local_day)
            symbols.add(symbol)
            sequences.append(
                ObservationSequence(
                    symbol=symbol,
                    timeframe=timeframe,
                    local_day=local_day,
                    observations=tuple(chunk),
                )
            )
            for current, nxt in zip(chunk, chunk[1:]):
                transitions.append(
                    TransitionRecord(
                        observation=current,
                        action=Action.IGNORE,
                        reward=0.0,
                        next_observation=nxt,
                        done=False,
                        label=None,
                    )
                )

    summary = {
        "rows_read": rows_read,
        "rows_accepted": rows_read - rows_rejected - duplicates_removed,
        "rows_rejected": rows_rejected,
        "duplicates_removed": duplicates_removed,
        "days_covered": len(days),
        "symbols_covered": len(symbols),
        "sequences_built": len(sequences),
        "transitions_built": len(transitions),
        "gap_breaks": gap_breaks,
        "coverage_status": "usable_partial" if transitions else "insufficient",
    }
    return SequenceDatasetBuildResult(
        sequences=tuple(sequences),
        transitions=tuple(transitions),
        summary=summary,
    )


def build_from_jsonl(path: Path) -> SequenceDatasetBuildResult:
    return build_from_rows(_iter_jsonl(path))

