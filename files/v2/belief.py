from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, Mapping, TypeVar

from .state import MarketState, SymbolState

StateT = TypeVar("StateT", SymbolState, MarketState)


@dataclass(frozen=True)
class BeliefState(Generic[StateT]):
    probabilities: Dict[StateT, float]

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("belief distribution cannot be empty")
        if any(value < 0 for value in self.probabilities.values()):
            raise ValueError("belief probabilities cannot be negative")
        total = sum(self.probabilities.values())
        if total <= 0:
            raise ValueError("belief probabilities must have positive mass")
        normalized = {state: float(value) / total for state, value in self.probabilities.items()}
        object.__setattr__(self, "probabilities", normalized)

    @classmethod
    def uniform(cls, states: list[StateT]) -> "BeliefState[StateT]":
        if not states:
            raise ValueError("states cannot be empty")
        return cls({state: 1.0 for state in states})

    def probability(self, state: StateT) -> float:
        return float(self.probabilities.get(state, 0.0))

    def most_likely(self) -> StateT:
        return max(self.probabilities, key=self.probabilities.get)

    def update(self, likelihoods: Mapping[StateT, float]) -> "BeliefState[StateT]":
        weighted = {
            state: self.probability(state) * max(0.0, float(likelihoods.get(state, 0.0)))
            for state in self.probabilities
        }
        if sum(weighted.values()) <= 0:
            raise ValueError("likelihood update removed all probability mass")
        return BeliefState(weighted)


def initial_symbol_belief() -> BeliefState[SymbolState]:
    return BeliefState.uniform(list(SymbolState))


def initial_market_belief() -> BeliefState[MarketState]:
    return BeliefState.uniform(list(MarketState))

