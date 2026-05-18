from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .belief_filter import FilteredBeliefRow
from .state import SymbolState


@dataclass(frozen=True)
class V1StructuralFeatures:
    candidate_score: float | None = None
    base_score: float | None = None
    score_floor: float | None = None
    forecast_return_pct: float | None = None
    today_change_pct: float | None = None
    ml_proba: float | None = None
    mtf_soft_penalty: float | None = None
    fresh_priority: bool = False
    catchup: bool = False
    continuation_profile: bool = False
    near_miss: bool = False
    signal_flags: Mapping[str, bool] | None = None


@dataclass(frozen=True)
class V1ProjectedStructuralFeatures:
    projected_today_change_pct: float
    projected_forecast_proxy_pct: float
    projected_candidate_score_trend: float
    projected_candidate_score_impulse_speed: float
    projected_leader_score_trend: float
    slope: float
    adx: float
    rsi: float
    vol_x: float
    daily_range_pct: float
    price_vs_ema20_pct: float


@dataclass(frozen=True)
class V1TemporalFeatures:
    prior_structural_scout: bool = False
    prior_wakeup_scout: bool = False
    minutes_since_first_structural_scout: float | None = None
    minutes_since_latest_wakeup_scout: float | None = None


@dataclass(frozen=True)
class EntryAdmissionRow:
    symbol: str
    local_day: str
    ts_ms: int
    true_state: SymbolState
    predicted_state: SymbolState
    belief: Mapping[str, float]
    belief_entropy: float
    belief_max_prob: float
    v1_structural: V1StructuralFeatures | None
    v1_projected_structural: V1ProjectedStructuralFeatures | None
    v1_temporal: V1TemporalFeatures


def build_row(
    item: FilteredBeliefRow,
    *,
    structural: V1StructuralFeatures | None = None,
    projected_structural: V1ProjectedStructuralFeatures | None = None,
    temporal: V1TemporalFeatures | None = None,
) -> EntryAdmissionRow:
    probs = {state.value: float(prob) for state, prob in item.belief.probabilities.items()}
    return EntryAdmissionRow(
        symbol=item.row.symbol,
        local_day=item.row.local_day,
        ts_ms=item.row.ts_ms,
        true_state=item.row.label,
        predicted_state=item.prediction,
        belief=probs,
        belief_entropy=round(_entropy(probs), 6),
        belief_max_prob=round(max(probs.values()), 6),
        v1_structural=structural,
        v1_projected_structural=projected_structural,
        v1_temporal=temporal or V1TemporalFeatures(),
    )


def _entropy(probs: Mapping[str, float]) -> float:
    return -sum(prob * math.log(prob) for prob in probs.values() if prob > 0)
