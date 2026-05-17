from __future__ import annotations

from enum import StrEnum
from typing import Dict, FrozenSet


class SymbolState(StrEnum):
    NOISE = "noise"
    EMERGING_MOVE = "emerging_move"
    CONFIRMED_TREND = "confirmed_trend"
    MATURE_TREND = "mature_trend"
    EXHAUSTION = "exhaustion"
    REVERSAL = "reversal"


class MarketState(StrEnum):
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"
    RISK_ON = "risk_on"
    HIGH_DISPERSION = "high_dispersion"


class Action(StrEnum):
    IGNORE = "ignore"
    WATCH = "watch"
    ELEVATE_PRIORITY = "elevate_priority"
    RESERVE_SLOT = "reserve_slot"
    OPEN_SMALL = "open_small"
    OPEN_FULL = "open_full"
    HOLD = "hold"
    TIGHTEN_EXIT = "tighten_exit"
    REDUCE = "reduce"
    SELL = "sell"


SYMBOL_TRANSITIONS: Dict[SymbolState, FrozenSet[SymbolState]] = {
    SymbolState.NOISE: frozenset({SymbolState.NOISE, SymbolState.EMERGING_MOVE}),
    SymbolState.EMERGING_MOVE: frozenset(
        {SymbolState.NOISE, SymbolState.EMERGING_MOVE, SymbolState.CONFIRMED_TREND}
    ),
    SymbolState.CONFIRMED_TREND: frozenset(
        {SymbolState.CONFIRMED_TREND, SymbolState.MATURE_TREND, SymbolState.EXHAUSTION}
    ),
    SymbolState.MATURE_TREND: frozenset(
        {SymbolState.MATURE_TREND, SymbolState.EXHAUSTION}
    ),
    SymbolState.EXHAUSTION: frozenset(
        {SymbolState.EXHAUSTION, SymbolState.REVERSAL, SymbolState.NOISE}
    ),
    SymbolState.REVERSAL: frozenset(
        {SymbolState.REVERSAL, SymbolState.NOISE, SymbolState.EMERGING_MOVE}
    ),
}


MARKET_TRANSITIONS: Dict[MarketState, FrozenSet[MarketState]] = {
    MarketState.RISK_OFF: frozenset(
        {MarketState.RISK_OFF, MarketState.NEUTRAL, MarketState.HIGH_DISPERSION}
    ),
    MarketState.NEUTRAL: frozenset(
        {
            MarketState.RISK_OFF,
            MarketState.NEUTRAL,
            MarketState.RISK_ON,
            MarketState.HIGH_DISPERSION,
        }
    ),
    MarketState.RISK_ON: frozenset(
        {MarketState.RISK_ON, MarketState.NEUTRAL, MarketState.HIGH_DISPERSION}
    ),
    MarketState.HIGH_DISPERSION: frozenset(
        {
            MarketState.RISK_OFF,
            MarketState.NEUTRAL,
            MarketState.RISK_ON,
            MarketState.HIGH_DISPERSION,
        }
    ),
}


def symbol_transition_allowed(source: SymbolState, target: SymbolState) -> bool:
    return target in SYMBOL_TRANSITIONS[source]


def market_transition_allowed(source: MarketState, target: MarketState) -> bool:
    return target in MARKET_TRANSITIONS[source]

