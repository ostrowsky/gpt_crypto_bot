from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .belief import BeliefState
from .history import CanonicalBar
from .lifecycle_labeling import LifecycleLabel
from .reward import RewardBreakdown, RewardInputs, compute_reward
from .state import Action, SymbolState


@dataclass(frozen=True)
class DecisionFrame:
    bar: CanonicalBar
    label: LifecycleLabel
    belief: BeliefState[SymbolState]
    prediction: SymbolState


@dataclass(frozen=True)
class PositionState:
    entry_price: float
    entry_index: int
    peak_price: float


@dataclass(frozen=True)
class StepResult:
    frame: DecisionFrame
    action: Action
    reward: RewardBreakdown
    position: PositionState | None
    done: bool


class OfflineDecisionEnvironment:
    def __init__(self, frames: Iterable[DecisionFrame], *, max_open_positions: int = 10) -> None:
        ordered = tuple(sorted(frames, key=lambda item: item.bar.open_ts_ms))
        if not ordered:
            raise ValueError("offline decision environment requires at least one frame")
        symbols = {frame.bar.symbol for frame in ordered}
        local_days = {frame.label.local_day for frame in ordered}
        if len(symbols) != 1 or len(local_days) != 1:
            raise ValueError("first offline environment slice expects one symbol-day episode")
        self._frames = ordered
        self.max_open_positions = int(max_open_positions)
        self.reset()

    def reset(self) -> DecisionFrame:
        self.index = 0
        self.position: PositionState | None = None
        self.open_positions = 0
        return self.current_frame

    @property
    def current_frame(self) -> DecisionFrame:
        return self._frames[self.index]

    @property
    def done(self) -> bool:
        return self.index >= len(self._frames) - 1

    def legal_actions(self) -> set[Action]:
        if self.position is None:
            actions = {Action.IGNORE, Action.WATCH, Action.ELEVATE_PRIORITY}
            if self.open_positions < self.max_open_positions:
                actions |= {Action.OPEN_SMALL, Action.OPEN_FULL}
            return actions
        return {Action.HOLD, Action.TIGHTEN_EXIT, Action.REDUCE, Action.SELL}

    def step(self, action: Action) -> StepResult:
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action.value} for current environment state")
        frame = self.current_frame
        reward = self._reward_for_action(frame, action)
        self._apply_action(frame, action)
        done = self.done
        result = StepResult(
            frame=frame,
            action=action,
            reward=reward,
            position=self.position,
            done=done,
        )
        if not done:
            self.index += 1
            if self.position is not None:
                self.position = PositionState(
                    entry_price=self.position.entry_price,
                    entry_index=self.position.entry_index,
                    peak_price=max(self.position.peak_price, self.current_frame.bar.high),
                )
        return result

    def _apply_action(self, frame: DecisionFrame, action: Action) -> None:
        if action in {Action.OPEN_SMALL, Action.OPEN_FULL}:
            self.position = PositionState(
                entry_price=frame.bar.close,
                entry_index=self.index,
                peak_price=frame.bar.high,
            )
            self.open_positions += 1
        elif action == Action.SELL:
            self.position = None
            self.open_positions = max(0, self.open_positions - 1)

    def _reward_for_action(self, frame: DecisionFrame, action: Action) -> RewardBreakdown:
        if action in {Action.OPEN_SMALL, Action.OPEN_FULL}:
            capture_ratio = _capture_ratio(frame)
            return compute_reward(
                RewardInputs(
                    capture_ratio_at_entry=capture_ratio,
                    late_entry=frame.label.state in {SymbolState.MATURE_TREND, SymbolState.EXHAUSTION, SymbolState.REVERSAL},
                    false_buy=frame.label.state == SymbolState.NOISE,
                )
            )
        if action in {Action.HOLD, Action.TIGHTEN_EXIT, Action.REDUCE}:
            return compute_reward(
                RewardInputs(
                    held_during_confirmed_trend=frame.label.state
                    in {SymbolState.CONFIRMED_TREND, SymbolState.MATURE_TREND},
                )
            )
        if action == Action.SELL and self.position is not None:
            realized_pnl_pct = ((frame.bar.close / self.position.entry_price) - 1.0) * 100.0
            mfe_pct = ((self.position.peak_price / self.position.entry_price) - 1.0) * 100.0
            exit_efficiency = realized_pnl_pct / mfe_pct if mfe_pct > 0 else 0.0
            giveback_pct = max(0.0, mfe_pct - realized_pnl_pct)
            return compute_reward(
                RewardInputs(
                    realized_pnl_pct=realized_pnl_pct,
                    exit_efficiency=exit_efficiency,
                    giveback_pct=giveback_pct,
                )
            )
        return compute_reward(RewardInputs())


def _capture_ratio(frame: DecisionFrame) -> float | None:
    if frame.label.day_mfe_pct <= 0:
        return None
    entry_move_pct = ((frame.bar.close / frame.label.day_open) - 1.0) * 100.0
    return max(0.0, entry_move_pct / frame.label.day_mfe_pct)
