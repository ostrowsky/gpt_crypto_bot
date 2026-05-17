from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .state import Action, SymbolState


@dataclass(frozen=True)
class FeatureSnapshot:
    price: float
    ema20: float
    slope: float
    adx: float
    rsi: float
    vol_x: float
    daily_range: float
    macd_hist: float


@dataclass(frozen=True)
class ShadowDecision:
    state: SymbolState
    action: str
    confidence: float
    reason: str


def estimate_shadow_state(features: FeatureSnapshot) -> ShadowDecision:
    above_ema = features.price >= features.ema20 if features.ema20 > 0 else False
    if features.rsi < 45.0 and not above_ema and features.slope < 0:
        return ShadowDecision(SymbolState.REVERSAL, "sell_candidate", 0.80, "below EMA20 with weak RSI and negative slope")
    if (
        features.rsi >= 72.0
        or (features.daily_range >= 12.0 and features.macd_hist <= 0)
    ):
        return ShadowDecision(SymbolState.EXHAUSTION, "tighten_exit", 0.72, "overextended or MACD fade after wide range")
    if (
        above_ema
        and features.slope >= 0.20
        and features.adx >= 28.0
        and features.vol_x >= 1.20
    ):
        if features.rsi >= 62.0 and features.daily_range >= 5.0:
            return ShadowDecision(SymbolState.MATURE_TREND, "hold", 0.82, "strong trend with extended participation")
        return ShadowDecision(SymbolState.CONFIRMED_TREND, "buy_candidate", 0.78, "trend strength confirmed")
    if (
        above_ema
        and features.slope >= 0.08
        and features.rsi >= 54.0
        and features.vol_x >= 0.80
    ):
        return ShadowDecision(SymbolState.EMERGING_MOVE, "elevate_priority", 0.64, "early positive structure")
    return ShadowDecision(SymbolState.NOISE, "watch", 0.55, "insufficient evidence")


def material_transition(previous: Mapping[str, object] | None, decision: ShadowDecision) -> bool:
    if not previous:
        return decision.state != SymbolState.NOISE
    return (
        str(previous.get("state")) != decision.state.value
        or str(previous.get("action")) != decision.action
    )


def append_shadow_event(path: Path, event: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False) + "\n")


def append_decision_trace(path: Path, event: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dedup_key = f"{event['sym']}|{event['tf']}|{event['bar_ts']}"
    seen = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
                seen.add(f"{row.get('sym')}|{row.get('tf')}|{row.get('bar_ts')}")
            except Exception:
                continue
    if dedup_key in seen:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False) + "\n")
