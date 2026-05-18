from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Callable, Iterable

from .offline_env import OfflineDecisionEnvironment, StepResult
from .state import Action, SymbolState


PolicyFn = Callable[[OfflineDecisionEnvironment], Action]


def always_flat_policy(env: OfflineDecisionEnvironment) -> Action:
    return Action.IGNORE if env.position is None else Action.SELL


def lifecycle_oracle_policy(env: OfflineDecisionEnvironment) -> Action:
    state = env.current_frame.label.state
    if env.position is None:
        return Action.OPEN_FULL if state in {SymbolState.EMERGING_MOVE, SymbolState.CONFIRMED_TREND} else Action.IGNORE
    return Action.SELL if state in {SymbolState.EXHAUSTION, SymbolState.REVERSAL} else Action.HOLD


def belief_policy_v1(env: OfflineDecisionEnvironment) -> Action:
    state = env.current_frame.prediction
    if env.position is None:
        return Action.OPEN_FULL if state in {SymbolState.EMERGING_MOVE, SymbolState.CONFIRMED_TREND} else Action.IGNORE
    return Action.SELL if state in {SymbolState.EXHAUSTION, SymbolState.REVERSAL} else Action.HOLD


def rollout(env: OfflineDecisionEnvironment, policy: PolicyFn) -> list[StepResult]:
    env.reset()
    results: list[StepResult] = []
    while True:
        if env.done and env.position is not None:
            action = Action.SELL
        else:
            action = policy(env)
        results.append(env.step(action))
        if results[-1].done:
            return results


def summarize_policy(name: str, rollouts: Iterable[list[StepResult]]) -> dict:
    episodes = list(rollouts)
    action_counts = Counter()
    component_totals = Counter()
    trade_count = 0
    total_reward = 0.0
    for steps in episodes:
        for step in steps:
            action_counts[step.action.value] += 1
            total_reward += step.reward.total
            component_totals.update(asdict(step.reward))
            if step.action in {Action.OPEN_SMALL, Action.OPEN_FULL}:
                trade_count += 1
    return {
        "policy": name,
        "episodes": len(episodes),
        "actions": sum(action_counts.values()),
        "trade_count": trade_count,
        "total_reward": round(total_reward, 6),
        "action_counts": dict(sorted(action_counts.items())),
        "reward_components": {key: round(value, 6) for key, value in sorted(component_totals.items())},
    }
