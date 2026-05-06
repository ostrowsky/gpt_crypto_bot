from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import config


log = logging.getLogger("signal_quality_feedback")
ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent

_CACHE: dict[str, Any] = {"loaded_at": 0.0, "policy": None}
_CACHE_TTL_SEC = 60.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    text = str(raw).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        out = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out and out not in (float("inf"), float("-inf")) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _metric_median(summary: dict[str, Any], key: str) -> Optional[float]:
    raw = summary.get(key)
    if isinstance(raw, dict) and raw.get("median") is not None:
        return _safe_float(raw.get("median"))
    return None


def feedback_path() -> Path:
    raw = Path(str(getattr(config, "SIGNAL_QUALITY_FEEDBACK_FILE", "../.runtime/signal_quality_feedback.json")))
    if raw.is_absolute():
        return raw
    return (ROOT / raw).resolve()


def build_feedback(report: dict[str, Any], *, report_path: str = "") -> dict[str, Any]:
    summary = report.get("summary") or {}
    miss_rate = _safe_float(summary.get("miss_rate"))
    false_positive_rate = _safe_float(summary.get("false_positive_rate"))
    top_mover_missed = _safe_int(summary.get("top_mover_missed_trends"))
    top_mover_caught = _safe_int(summary.get("top_mover_caught_trends"))
    missed_trends = _safe_int(summary.get("missed_trends"))
    early_exits = _safe_int(summary.get("early_exits"))
    late_entries = _safe_int(summary.get("late_entries"))
    cooldown_bars_on_pressure = max(
        0,
        int(getattr(config, "SIGNAL_QUALITY_FEEDBACK_COOLDOWN_BARS_ON_PRESSURE", 2)),
    )

    recall_pressure = (
        miss_rate >= float(getattr(config, "SIGNAL_QUALITY_FEEDBACK_MISS_RATE_MIN", 0.65))
        and top_mover_missed >= int(getattr(config, "SIGNAL_QUALITY_FEEDBACK_TOP_MOVER_MISSED_MIN", 20))
        and false_positive_rate <= float(getattr(config, "SIGNAL_QUALITY_FEEDBACK_FALSE_POSITIVE_MAX", 0.22))
    )
    exit_pressure = (
        early_exits > 0
        and (
            (_metric_median(summary, "exit_efficiency") is not None and _metric_median(summary, "exit_efficiency") < 0.35)
            or (_metric_median(summary, "giveback_pct") is not None and _metric_median(summary, "giveback_pct") > 0.70)
        )
    )

    reasons: list[str] = []
    if recall_pressure:
        reasons.append(
            f"top-mover miss pressure: miss_rate={miss_rate:.4f}, "
            f"top_mover_missed={top_mover_missed}, false_positive_rate={false_positive_rate:.4f}"
        )
    if exit_pressure:
        reasons.append(
            f"exit pressure: early_exits={early_exits}, "
            f"exit_eff_median={_metric_median(summary, 'exit_efficiency')}, "
            f"giveback_median={_metric_median(summary, 'giveback_pct')}"
        )
    if not reasons:
        reasons.append("no feedback action: thresholds not met")

    apply_cooldown = bool(
        getattr(config, "SIGNAL_QUALITY_FEEDBACK_ENABLED", True)
        and getattr(config, "SIGNAL_QUALITY_FEEDBACK_AUTO_APPLY_COOLDOWN", True)
        and recall_pressure
    )

    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "target_day_local": report.get("target_day_local", ""),
        "source_report_json": report_path or str((report.get("files") or {}).get("json") or ""),
        "metrics": {
            "miss_rate": miss_rate,
            "missed_trends": missed_trends,
            "top_mover_caught_trends": top_mover_caught,
            "top_mover_missed_trends": top_mover_missed,
            "false_positive_rate": false_positive_rate,
            "late_entries": late_entries,
            "early_exits": early_exits,
            "capture_ratio_at_entry_median": _metric_median(summary, "capture_ratio_at_entry"),
            "exit_efficiency_median": _metric_median(summary, "exit_efficiency"),
            "giveback_pct_median": _metric_median(summary, "giveback_pct"),
        },
        "pressures": {
            "top_mover_recall": recall_pressure,
            "exit_quality": exit_pressure,
            "cooldown_harm": "replay_metric_required",
            "cluster_cap": "observe_only",
            "portfolio_replacement": "observe_only",
        },
        "policy": {
            "apply_cooldown_relaxation": apply_cooldown,
            "cooldown_bars": cooldown_bars_on_pressure if apply_cooldown else None,
            "apply_exit_rule_changes": False,
            "apply_cluster_rule_changes": False,
            "apply_replacement_rule_changes": False,
        },
        "validation": {
            "cooldown_bars": {
                "status": "replay_confirmed",
                "window": "7d ending 2026-05-06",
                "baseline": {
                    "cooldown_bars": 8,
                    "pnl_total": 8.7857,
                    "pnl_avg": 0.0446,
                    "win_rate": 0.4315,
                    "trade_precision": 0.3401,
                    "top15_recall": 1.0,
                    "cooldown_harm_pct": 78.4354,
                },
                "variant": {
                    "cooldown_bars": 2,
                    "pnl_total": 15.8008,
                    "pnl_avg": 0.0728,
                    "win_rate": 0.4378,
                    "trade_precision": 0.3641,
                    "top15_recall": 1.0,
                    "cooldown_harm_pct": 50.6918,
                },
                "report": ".runtime/reports/feedback_policy_hypothesis_sweep_7d_20260506.json",
            },
            "exit_rules": {
                "status": "not_applied_mixed_replay",
                "reason": "weak-exit hold variants improved some PnL metrics but worsened giveback or exit efficiency",
            },
            "cluster_rules": {
                "status": "not_applied_rejected",
                "reason": "current cluster cap outperformed no-cluster replay on 7d PnL",
            },
            "replacement_rules": {
                "status": "not_applied_no_signal",
                "reason": "replacement did not trigger in the 7d replay window",
            },
        },
        "reason": "; ".join(reasons),
    }


def save_feedback(report: dict[str, Any], *, report_path: str = "") -> dict[str, Any]:
    feedback = build_feedback(report, report_path=report_path)
    path = feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(feedback, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    _CACHE["loaded_at"] = time.time()
    _CACHE["policy"] = feedback
    return feedback


def load_feedback(*, force: bool = False) -> Optional[dict[str, Any]]:
    if not bool(getattr(config, "SIGNAL_QUALITY_FEEDBACK_ENABLED", True)):
        return None
    now = time.time()
    if not force and _CACHE.get("policy") is not None and now - float(_CACHE.get("loaded_at", 0.0)) < _CACHE_TTL_SEC:
        return _CACHE["policy"]
    path = feedback_path()
    if not path.exists():
        _CACHE["loaded_at"] = now
        _CACHE["policy"] = None
        return None
    try:
        feedback = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read signal-quality feedback %s: %s", path, exc)
        feedback = None
    _CACHE["loaded_at"] = now
    _CACHE["policy"] = feedback
    return feedback if isinstance(feedback, dict) else None


def is_fresh(feedback: Optional[dict[str, Any]] = None) -> bool:
    feedback = feedback if feedback is not None else load_feedback()
    if not feedback:
        return False
    generated = _parse_utc(feedback.get("generated_at_utc"))
    if generated is None:
        return False
    max_age_hours = max(1.0, float(getattr(config, "SIGNAL_QUALITY_FEEDBACK_MAX_AGE_HOURS", 48)))
    age_hours = (datetime.now(timezone.utc) - generated).total_seconds() / 3600.0
    return age_hours <= max_age_hours


def effective_cooldown_bars(
    base_bars: int,
    *,
    mode: str = "",
    reason: Optional[str] = None,
    tf: Optional[str] = None,
    pnl_pct: Optional[float] = None,
) -> int:
    if base_bars <= 0:
        return max(0, int(base_bars))
    feedback = load_feedback()
    if not is_fresh(feedback):
        return int(base_bars)
    policy = feedback.get("policy") or {}
    if not bool(policy.get("apply_cooldown_relaxation")):
        return int(base_bars)
    target = policy.get("cooldown_bars")
    if target is None:
        return int(base_bars)
    return max(0, min(int(base_bars), int(target)))


def status_snapshot() -> dict[str, Any]:
    feedback = load_feedback()
    if not feedback:
        return {
            "enabled": bool(getattr(config, "SIGNAL_QUALITY_FEEDBACK_ENABLED", True)),
            "file": str(feedback_path()),
            "loaded": False,
        }
    policy = feedback.get("policy") or {}
    return {
        "enabled": bool(getattr(config, "SIGNAL_QUALITY_FEEDBACK_ENABLED", True)),
        "file": str(feedback_path()),
        "loaded": True,
        "fresh": is_fresh(feedback),
        "generated_at_utc": feedback.get("generated_at_utc"),
        "target_day_local": feedback.get("target_day_local"),
        "reason": feedback.get("reason", ""),
        "apply_cooldown_relaxation": bool(policy.get("apply_cooldown_relaxation")),
        "cooldown_bars": policy.get("cooldown_bars"),
        "apply_exit_rule_changes": bool(policy.get("apply_exit_rule_changes")),
        "apply_cluster_rule_changes": bool(policy.get("apply_cluster_rule_changes")),
        "apply_replacement_rule_changes": bool(policy.get("apply_replacement_rule_changes")),
    }
