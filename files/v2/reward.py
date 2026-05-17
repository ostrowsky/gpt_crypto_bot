from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardInputs:
    realized_pnl_pct: float = 0.0
    capture_ratio_at_entry: float | None = None
    lead_time_to_final_top_min: float | None = None
    exit_efficiency: float | None = None
    giveback_pct: float = 0.0
    false_buy: bool = False
    late_entry: bool = False
    blocked_winner: bool = False
    churn_cost_pct: float = 0.0
    held_during_confirmed_trend: bool = False


@dataclass(frozen=True)
class RewardBreakdown:
    early_capture_reward: float
    trend_hold_reward: float
    realized_pnl_reward: float
    mfe_retention_reward: float
    false_buy_penalty: float
    late_entry_penalty: float
    giveback_penalty: float
    churn_penalty: float
    blocked_winner_penalty: float

    @property
    def total(self) -> float:
        return sum(
            (
                self.early_capture_reward,
                self.trend_hold_reward,
                self.realized_pnl_reward,
                self.mfe_retention_reward,
                self.false_buy_penalty,
                self.late_entry_penalty,
                self.giveback_penalty,
                self.churn_penalty,
                self.blocked_winner_penalty,
            )
        )


def compute_reward(inputs: RewardInputs) -> RewardBreakdown:
    early_capture_reward = 0.0
    if inputs.capture_ratio_at_entry is not None:
        early_capture_reward = max(0.0, 1.0 - float(inputs.capture_ratio_at_entry))

    trend_hold_reward = 0.25 if inputs.held_during_confirmed_trend else 0.0
    realized_pnl_reward = float(inputs.realized_pnl_pct)
    mfe_retention_reward = max(0.0, float(inputs.exit_efficiency or 0.0))
    false_buy_penalty = -1.0 if inputs.false_buy else 0.0
    late_entry_penalty = -0.5 if inputs.late_entry else 0.0
    giveback_penalty = -max(0.0, float(inputs.giveback_pct))
    churn_penalty = -max(0.0, float(inputs.churn_cost_pct))
    blocked_winner_penalty = -1.0 if inputs.blocked_winner else 0.0

    return RewardBreakdown(
        early_capture_reward=early_capture_reward,
        trend_hold_reward=trend_hold_reward,
        realized_pnl_reward=realized_pnl_reward,
        mfe_retention_reward=mfe_retention_reward,
        false_buy_penalty=false_buy_penalty,
        late_entry_penalty=late_entry_penalty,
        giveback_penalty=giveback_penalty,
        churn_penalty=churn_penalty,
        blocked_winner_penalty=blocked_winner_penalty,
    )

