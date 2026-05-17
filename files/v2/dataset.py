from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .state import Action, MarketState, SymbolState


@dataclass(frozen=True)
class Observation:
    symbol: str
    ts_ms: int
    features: Mapping[str, float]


@dataclass(frozen=True)
class LabeledState:
    symbol_state: SymbolState
    market_state: MarketState
    hindsight_confidence: float


@dataclass(frozen=True)
class TransitionRecord:
    observation: Observation
    action: Action
    reward: float
    next_observation: Observation | None
    done: bool
    label: LabeledState | None = None

