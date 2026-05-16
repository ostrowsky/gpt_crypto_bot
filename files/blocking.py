from __future__ import annotations

from typing import Any, Dict


def normalize_blocked_reason(signal_type: str = "", reason: str = "") -> str:
    signal = str(signal_type or "").strip().lower()
    text = f"{signal} {str(reason or '').strip().lower()}"
    if "portfolio" in text or "портфель" in text:
        return "portfolio_full"
    if "open_cluster_cap" in text or ("cluster" in text and "cap" in text):
        return "open_cluster_cap"
    if "symbol_cooldown" in text or "cooldown" in text:
        return "symbol_cooldown"
    if "top_gainer_score" in text:
        return "top_gainer_score_gate"
    if "top_gainer_objective" in text or "objective gate" in text:
        return "top_gainer_objective_gate"
    if "mtf" in text or "deep correction" in text:
        return "mtf_correction"
    if "best_accuracy" in text or "best accuracy" in text or "accuracy <" in text:
        return "accuracy_gate"
    if "mode disabled" in text or "agent mode disabled" in text:
        return "agent_mode_disabled"
    if "replacement_filter" in text or "replacement filter" in text:
        return "agent_replacement_filter"
    if "chase_guard" in text or "chase guard" in text:
        return "chase_guard"
    if "agent_leader_filter" in text:
        return "agent_leader_filter"
    if "ranker_hard_veto" in text:
        return "ranker_hard_veto"
    if "ranker_veto" in text:
        return "ranker_veto"
    if "clone_guard" in text:
        return "clone_signal_guard"
    if "late_impulse_rotation" in text:
        return "late_impulse_rotation"
    if "trend_quality" in text:
        return "trend_quality"
    if "impulse_guard" in text:
        return "impulse_guard"
    if "late_continuation" in text:
        return "late_continuation"
    if "time_block" in text:
        return "time_block"
    if "entry_score" in text:
        return "entry_score"
    if "adx" in text:
        return "adx_gate"
    if "volume" in text or "vol_x" in text:
        return "volume_gate"
    if signal:
        return signal
    if reason:
        return "blocked_rule"
    return "blocked_unknown"


def blocked_gate(signal_type: str = "", reason_code: str = "") -> str:
    code = str(reason_code or "").strip().lower()
    if code in {"strategy_cap", "portfolio_full", "open_cluster_cap", "clone_signal_guard", "late_impulse_rotation", "symbol_cooldown"}:
        return "portfolio"
    if code in {"mtf_correction", "time_block"}:
        return "context"
    if code in {"ranker_veto", "ranker_hard_veto", "entry_score", "top_gainer_score_gate", "top_gainer_objective_gate", "trend_quality", "impulse_guard", "late_continuation", "adx_gate", "volume_gate", "chase_guard"}:
        return "quality"
    return str(signal_type or "").strip().lower() or "unknown"


def compact_block_context(**kwargs: Any) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}
