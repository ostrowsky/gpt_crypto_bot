from __future__ import annotations

from dataclasses import dataclass

from .lifecycle_labeling import LifecycleLabel, LifecycleThresholds
from .state import SymbolState


@dataclass(frozen=True)
class TeacherConfidence:
    value: float
    move_strength_score: float
    confirmation_score: float
    persistence_score: float
    clean_path_score: float


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_label(
    label: LifecycleLabel,
    *,
    bars_in_day: int,
    thresholds: LifecycleThresholds = LifecycleThresholds(),
) -> TeacherConfidence:
    move_strength = _clip((label.day_mfe_pct - 3.0) / 2.0)
    if label.confirmation_index is None:
        confirmation = 0.0
        persistence = 0.0
    else:
        confirmation = _clip(1.0 - (label.confirmation_index / max(1, bars_in_day - 1)))
        persistence = _clip((bars_in_day - label.confirmation_index) / max(1, thresholds.min_persistence_bars * 2))

    if label.state == SymbolState.NOISE:
        clean_path = 0.25
        raw = 0.15 + 0.10 * (1.0 - move_strength)
    elif label.state == SymbolState.EMERGING_MOVE:
        clean_path = 0.65
        raw = 0.10 + 0.25 * move_strength + 0.15 * confirmation + 0.10 * persistence
    elif label.state in {SymbolState.CONFIRMED_TREND, SymbolState.MATURE_TREND}:
        clean_path = 0.85
        raw = 0.20 + 0.35 * move_strength + 0.20 * confirmation + 0.15 * persistence + 0.10 * clean_path
    elif label.state == SymbolState.EXHAUSTION:
        clean_path = 0.70
        raw = 0.20 + 0.30 * move_strength + 0.15 * confirmation + 0.15 * persistence + 0.20 * clean_path
    else:
        clean_path = 0.60
        raw = 0.20 + 0.25 * move_strength + 0.10 * confirmation + 0.10 * persistence + 0.20 * clean_path
    return TeacherConfidence(
        value=round(_clip(raw), 6),
        move_strength_score=round(move_strength, 6),
        confirmation_score=round(confirmation, 6),
        persistence_score=round(persistence, 6),
        clean_path_score=round(clean_path, 6),
    )
