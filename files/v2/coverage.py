from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

from .sequence_dataset import BAR_MS, SequenceDatasetBuildResult


@dataclass(frozen=True)
class SequenceCoverageAudit:
    summary: Mapping[str, object]
    by_day: Mapping[str, Mapping[str, object]]
    by_timeframe: Mapping[str, Mapping[str, object]]
    fragmented_slices: tuple[Mapping[str, object], ...]


def build_coverage_audit(result: SequenceDatasetBuildResult) -> SequenceCoverageAudit:
    by_day: dict[str, dict[str, int]] = defaultdict(
        lambda: {"sequences": 0, "observations": 0, "transitions": 0}
    )
    by_timeframe: dict[str, dict[str, int]] = defaultdict(
        lambda: {"sequences": 0, "observations": 0, "transitions": 0}
    )
    by_slice: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"sequences": 0, "observations": 0, "transitions": 0}
    )

    longest_sequence_bars = 0
    longest_sequence_minutes = 0

    for seq in result.sequences:
        bars = len(seq.observations)
        transitions = max(0, bars - 1)
        by_day[seq.local_day]["sequences"] += 1
        by_day[seq.local_day]["observations"] += bars
        by_day[seq.local_day]["transitions"] += transitions
        by_timeframe[seq.timeframe]["sequences"] += 1
        by_timeframe[seq.timeframe]["observations"] += bars
        by_timeframe[seq.timeframe]["transitions"] += transitions
        key = (seq.symbol, seq.timeframe, seq.local_day)
        by_slice[key]["sequences"] += 1
        by_slice[key]["observations"] += bars
        by_slice[key]["transitions"] += transitions
        if bars > longest_sequence_bars:
            longest_sequence_bars = bars
            longest_sequence_minutes = int(bars * BAR_MS[seq.timeframe] / 60_000)

    rows_accepted = int(result.summary.get("rows_accepted", 0))
    transitions_built = int(result.summary.get("transitions_built", 0))
    summary = {
        **dict(result.summary),
        "transition_density": round(transitions_built / rows_accepted, 4) if rows_accepted else 0.0,
        "longest_sequence_bars": longest_sequence_bars,
        "longest_sequence_minutes": longest_sequence_minutes,
    }
    fragmented = []
    for (symbol, timeframe, local_day), stats in by_slice.items():
        if stats["sequences"] <= 1:
            continue
        fragmented.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "local_day": local_day,
                "segments": stats["sequences"],
                "observations": stats["observations"],
                "transitions": stats["transitions"],
            }
        )
    fragmented.sort(key=lambda row: (row["segments"], row["observations"]), reverse=True)

    return SequenceCoverageAudit(
        summary=summary,
        by_day={key: dict(value) for key, value in sorted(by_day.items())},
        by_timeframe={key: dict(value) for key, value in sorted(by_timeframe.items())},
        fragmented_slices=tuple(fragmented[:20]),
    )

