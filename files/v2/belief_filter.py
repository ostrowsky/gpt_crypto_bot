from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .belief import BeliefState
from .state import SYMBOL_TRANSITIONS, SymbolState
from .state_reconstruction import ReconstructionRow


@dataclass(frozen=True)
class FilteredBeliefRow:
    row: ReconstructionRow
    belief: BeliefState[SymbolState]
    prediction: SymbolState


def transition_matrix(self_bias: float = 0.70) -> dict[SymbolState, dict[SymbolState, float]]:
    matrix = {}
    for source, targets in SYMBOL_TRANSITIONS.items():
        others = [target for target in targets if target != source]
        if source in targets:
            other_mass = max(0.0, 1.0 - self_bias)
            matrix[source] = {source: self_bias}
            if others:
                for target in others:
                    matrix[source][target] = other_mass / len(others)
        else:
            matrix[source] = {target: 1.0 / len(targets) for target in targets}
    return matrix


def predict_prior(
    belief: BeliefState[SymbolState],
    matrix: dict[SymbolState, dict[SymbolState, float]],
) -> BeliefState[SymbolState]:
    masses = {state: 0.0 for state in SymbolState}
    for source, prob in belief.probabilities.items():
        for target, trans_prob in matrix[source].items():
            masses[target] += prob * trans_prob
    return BeliefState(masses)


def centroid_likelihoods(
    features: tuple[float, ...],
    centroids: dict[SymbolState, tuple[float, ...]],
    *,
    temperature: float = 1.0,
) -> dict[SymbolState, float]:
    distances = {
        state: float(np.linalg.norm(np.array(features) - np.array(centroid)))
        for state, centroid in centroids.items()
    }
    scores = {state: math.exp(-distance / max(temperature, 1e-9)) for state, distance in distances.items()}
    for state in SymbolState:
        scores.setdefault(state, 1e-12)
    return scores


def filter_rows(
    rows: list[ReconstructionRow],
    centroids: dict[SymbolState, tuple[float, ...]],
    *,
    self_bias: float = 0.70,
    temperature: float = 1.0,
) -> list[FilteredBeliefRow]:
    matrix = transition_matrix(self_bias)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.symbol, row.local_day)].append(row)
    out = []
    for _, group in sorted(grouped.items()):
        belief = BeliefState.uniform(list(SymbolState))
        for row in sorted(group, key=lambda item: item.ts_ms):
            prior = predict_prior(belief, matrix)
            likelihoods = centroid_likelihoods(row.features, centroids, temperature=temperature)
            belief = prior.update(likelihoods)
            out.append(FilteredBeliefRow(row=row, belief=belief, prediction=belief.most_likely()))
    return sorted(out, key=lambda item: (item.row.local_day, item.row.symbol, item.row.ts_ms))
