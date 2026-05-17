from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

from .history import CanonicalBar
from .state import SymbolState, symbol_transition_allowed


LABEL_VERSION = "hindsight_lifecycle_v1"


@dataclass(frozen=True)
class LifecycleThresholds:
    min_favorable_move_pct: float = 4.0
    confirmed_move_pct: float = 2.0
    mature_move_pct: float = 3.0
    min_persistence_bars: int = 4
    exhaustion_giveback_ratio: float = 0.35
    reversal_giveback_ratio: float = 0.60


@dataclass(frozen=True)
class LifecycleLabel:
    symbol: str
    timeframe: str
    open_ts_ms: int
    local_day: str
    state: SymbolState
    label_version: str
    day_open: float
    day_mfe_pct: float
    peak_index: int | None
    confirmation_index: int | None


def _local_day(ts_ms: int, timezone_name: str) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=ZoneInfo("UTC")).astimezone(ZoneInfo(timezone_name)).date().isoformat()


def _split_by_day(bars: Iterable[CanonicalBar], timezone_name: str) -> list[list[CanonicalBar]]:
    groups: list[list[CanonicalBar]] = []
    current: list[CanonicalBar] = []
    current_day: str | None = None
    for bar in sorted(bars, key=lambda item: item.open_ts_ms):
        day = _local_day(bar.open_ts_ms, timezone_name)
        if current_day is not None and day != current_day:
            groups.append(current)
            current = []
        current.append(bar)
        current_day = day
    if current:
        groups.append(current)
    return groups


def label_bars(
    bars: Iterable[CanonicalBar],
    *,
    timezone_name: str = "Europe/Budapest",
    thresholds: LifecycleThresholds = LifecycleThresholds(),
) -> list[LifecycleLabel]:
    out: list[LifecycleLabel] = []
    for day_bars in _split_by_day(bars, timezone_name):
        out.extend(_label_day(day_bars, timezone_name=timezone_name, thresholds=thresholds))
    return out


def _label_day(
    bars: list[CanonicalBar],
    *,
    timezone_name: str,
    thresholds: LifecycleThresholds,
) -> list[LifecycleLabel]:
    if not bars:
        return []
    day_open = bars[0].open
    highs_pct = [((bar.high / day_open) - 1.0) * 100.0 for bar in bars]
    closes_pct = [((bar.close / day_open) - 1.0) * 100.0 for bar in bars]
    day_mfe = max(highs_pct)
    peak_index = highs_pct.index(day_mfe)
    local_day = _local_day(bars[0].open_ts_ms, timezone_name)
    confirmation_index = next(
        (i for i, value in enumerate(highs_pct) if value >= thresholds.confirmed_move_pct),
        None,
    )
    qualifies = (
        day_mfe >= thresholds.min_favorable_move_pct
        and confirmation_index is not None
        and len(bars) - confirmation_index >= thresholds.min_persistence_bars
    )
    states = [SymbolState.NOISE for _ in bars]
    if qualifies and confirmation_index is not None:
        for i in range(0, confirmation_index):
            states[i] = SymbolState.EMERGING_MOVE
        current_peak = 0.0
        exhausted = False
        reversed_ = False
        mature_seen = False
        for i in range(confirmation_index, len(bars)):
            current_peak = max(current_peak, highs_pct[i])
            giveback = 0.0 if current_peak <= 0 else max(0.0, (current_peak - closes_pct[i]) / current_peak)
            if i == confirmation_index:
                states[i] = SymbolState.CONFIRMED_TREND
                mature_seen = highs_pct[i] >= thresholds.mature_move_pct
                continue
            if i <= peak_index:
                if mature_seen or highs_pct[i] >= thresholds.mature_move_pct:
                    mature_seen = True
                    states[i] = SymbolState.MATURE_TREND
                else:
                    states[i] = SymbolState.CONFIRMED_TREND
                continue
            reversal_condition = giveback >= thresholds.reversal_giveback_ratio or closes_pct[i] <= 0
            if reversed_:
                reversed_ = True
                states[i] = SymbolState.REVERSAL
            elif reversal_condition and not exhausted:
                exhausted = True
                states[i] = SymbolState.EXHAUSTION
            elif reversal_condition:
                reversed_ = True
                states[i] = SymbolState.REVERSAL
            elif exhausted or giveback >= thresholds.exhaustion_giveback_ratio:
                exhausted = True
                states[i] = SymbolState.EXHAUSTION
            elif mature_seen:
                states[i] = SymbolState.MATURE_TREND
            else:
                states[i] = SymbolState.CONFIRMED_TREND
    return [
        LifecycleLabel(
            symbol=bar.symbol,
            timeframe=bar.timeframe,
            open_ts_ms=bar.open_ts_ms,
            local_day=local_day,
            state=states[i],
            label_version=LABEL_VERSION,
            day_open=day_open,
            day_mfe_pct=round(day_mfe, 6),
            peak_index=peak_index if qualifies else None,
            confirmation_index=confirmation_index if qualifies else None,
        )
        for i, bar in enumerate(bars)
    ]


def summarize_labels(labels: Iterable[LifecycleLabel]) -> dict:
    ordered = list(labels)
    state_counts = Counter(label.state.value for label in ordered)
    transition_counts = Counter()
    invalid_transition_counts = Counter()
    for prev, current in zip(ordered, ordered[1:]):
        if (
            prev.symbol == current.symbol
            and prev.timeframe == current.timeframe
            and prev.local_day == current.local_day
        ):
            transition_counts[f"{prev.state.value}->{current.state.value}"] += 1
            if not symbol_transition_allowed(prev.state, current.state):
                invalid_transition_counts[f"{prev.state.value}->{current.state.value}"] += 1
    qualifying_days = {
        (label.symbol, label.local_day)
        for label in ordered
        if label.confirmation_index is not None
    }
    total_days = {(label.symbol, label.local_day) for label in ordered}
    return {
        "label_version": LABEL_VERSION,
        "rows": len(ordered),
        "symbols": len({label.symbol for label in ordered}),
        "days": len(total_days),
        "qualifying_days": len(qualifying_days),
        "state_counts": dict(state_counts),
        "transition_counts": dict(transition_counts),
        "invalid_transition_counts": dict(invalid_transition_counts),
    }


def thresholds_dict(thresholds: LifecycleThresholds = LifecycleThresholds()) -> dict:
    return asdict(thresholds)
