from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import audit_early_block_rescue_event_replay as event_replay

ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_OUTPUT_JSON = REPORTS / "entry_admission_shadow_reward_latest.json"
DEFAULT_OUTPUT_TXT = REPORTS / "entry_admission_shadow_reward_latest.txt"

REASON_SETS = {
    "agent_only": {"agent_mode_disabled", "agent_leader_filter"},
    "score_only": {"top_gainer_score_gate"},
    "cooldown_only": {"symbol_cooldown"},
    "agent_score": {"agent_mode_disabled", "agent_leader_filter", "top_gainer_score_gate"},
    "agent_score_cooldown": {"agent_mode_disabled", "agent_leader_filter", "top_gainer_score_gate", "symbol_cooldown"},
    "agent_score_chase": {"agent_mode_disabled", "agent_leader_filter", "top_gainer_score_gate", "chase_guard"},
}


@dataclass(frozen=True)
class RewardConfig:
    false_candidate_penalty_pct: float = 1.0
    late_capture_threshold: float = 0.25
    max_candidate_count: int = 250
    min_top_precision: float = 0.20
    min_net_reward_pct: float = 10.0
    min_rescued_top: int = 2


def build_report(
    reports_dir: Path = REPORTS,
    files_dir: Path = FILES,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
    cfg: RewardConfig = RewardConfig(),
    save: bool = True,
) -> dict[str, Any]:
    labels = event_replay._load_labels(reports_dir)
    events = event_replay._load_blocked_events(files_dir, labels)
    entries = event_replay._load_entries(files_dir, labels)
    variants = []
    for reason_name, reasons in REASON_SETS.items():
        for max_hour in (2, 4, 6, 8, 12):
            for min_blocks in (3, 5, 10, 20, 50):
                variants.append(_evaluate(events, entries, labels, reason_name, reasons, max_hour, min_blocks, cfg))
    variants.sort(key=_rank_key, reverse=True)
    best = variants[0] if variants else None
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "research_only",
        "config": cfg.__dict__,
        "coverage": {
            "label_days": len({day for day, _ in labels}),
            "labeled_day_symbols": len(labels),
            "blocked_events_loaded": len(events),
            "entries_loaded": sum(len(v) for v in entries.values()),
        },
        "best_variant": best,
        "top_variants": variants[:20],
        "decision": _decision(best, cfg),
    }
    text = render_text(payload)
    if save:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output_txt.write_text(text, encoding="utf-8")
        payload["files"] = {"json": str(output_json), "txt": str(output_txt)}
    return payload


def render_text(report: dict[str, Any]) -> str:
    c = report.get("coverage") or {}
    best = report.get("best_variant") or {}
    lines = [
        "Entry admission shadow reward (research-only)",
        f"coverage: days={c.get('label_days')} labels={c.get('labeled_day_symbols')} blocked_events={c.get('blocked_events_loaded')}",
        f"decision: {report.get('decision')}",
        "",
        "Best variant:",
    ]
    if best:
        lines.append(
            f"  {best.get('reason_set')} max_hour={best.get('max_first_block_hour')} min_blocks={best.get('min_blocked_count')} "
            f"candidates={best.get('candidate_count')} top={best.get('top_candidates')} false={best.get('false_candidates')} "
            f"precision={_pct(best.get('top_precision'))} net_reward={best.get('net_reward_pct')}% "
            f"top_reward={best.get('top_reward_pct')}% false_penalty={best.get('false_penalty_pct')}%"
        )
    else:
        lines.append("  no candidates")
    lines.extend(["", "Top variants:"])
    for item in (report.get("top_variants") or [])[:8]:
        lines.append(
            f"  {item.get('reason_set')} h<={item.get('max_first_block_hour')} blocks>={item.get('min_blocked_count')}: "
            f"net={item.get('net_reward_pct')} top={item.get('top_candidates')} false={item.get('false_candidates')} precision={_pct(item.get('top_precision'))}"
        )
    return "\n".join(lines) + "\n"


def _evaluate(events: list[dict], entries: dict, labels: dict, reason_name: str, reasons: set[str], max_hour: int, min_blocks: int, cfg: RewardConfig) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for event in events:
        if event.get("hour") is None or event["hour"] > max_hour or event.get("reason_code") not in reasons:
            continue
        grouped.setdefault((event["day"], event["symbol"]), []).append(event)
    candidates = []
    for key, rows in grouped.items():
        if len(rows) < min_blocks:
            continue
        rows = sorted(rows, key=lambda r: str(r.get("ts") or ""))
        label = labels.get(key, {"is_top15": False, "status": "not_top15"})
        entry_rows = sorted(entries.get(key, []), key=lambda r: str(r.get("ts") or ""))
        first = rows[0]
        bought_before = bool(entry_rows and str(entry_rows[0].get("ts") or "") <= str(first.get("ts") or ""))
        reward = _candidate_reward(label, bought_before, cfg)
        candidates.append({
            "day": key[0],
            "symbol": key[1],
            "first_ts": first.get("ts"),
            "first_hour": first.get("hour"),
            "first_reason_code": first.get("reason_code"),
            "block_count": len(rows),
            "is_top15": bool(label.get("is_top15")),
            "critic_status": label.get("status"),
            "capture_ratio_at_entry": label.get("capture_ratio_at_entry"),
            "opportunity_from_first_block_pct": label.get("opportunity_from_first_block_pct"),
            "bought_before_rescue": bought_before,
            "reward_pct": reward,
        })
    top = [c for c in candidates if c["is_top15"]]
    false = [c for c in candidates if not c["is_top15"]]
    rewarded_top = [c for c in top if c["reward_pct"] > 0]
    top_reward = sum(c["reward_pct"] for c in rewarded_top)
    false_penalty = len(false) * cfg.false_candidate_penalty_pct
    net = top_reward - false_penalty
    return {
        "reason_set": reason_name,
        "max_first_block_hour": max_hour,
        "min_blocked_count": min_blocks,
        "candidate_count": len(candidates),
        "top_candidates": len(top),
        "false_candidates": len(false),
        "rewarded_top_candidates": len(rewarded_top),
        "top_precision": round(len(top) / len(candidates), 6) if candidates else 0.0,
        "false_candidate_ratio": round(len(false) / len(candidates), 6) if candidates else 0.0,
        "top_reward_pct": round(top_reward, 6),
        "false_penalty_pct": round(false_penalty, 6),
        "net_reward_pct": round(net, 6),
        "avg_net_reward_per_candidate_pct": round(net / len(candidates), 6) if candidates else 0.0,
        "top_examples": sorted(rewarded_top, key=lambda r: r["reward_pct"], reverse=True)[:12],
        "false_examples": sorted(false, key=lambda r: r["block_count"], reverse=True)[:8],
    }


def _candidate_reward(label: dict, bought_before: bool, cfg: RewardConfig) -> float:
    if not label.get("is_top15"):
        return 0.0
    if bought_before:
        return 0.0
    status = str(label.get("status") or "")
    capture = _num(label.get("capture_ratio_at_entry"), None)
    opportunity = _num(label.get("opportunity_from_first_block_pct"), 0.0)
    if status != "bought":
        return max(0.0, opportunity)
    if capture is None or capture <= cfg.late_capture_threshold:
        return max(0.0, opportunity * (1.0 - max(0.0, min(1.0, capture or 0.0))))
    return 0.0


def _rank_key(item: dict[str, Any]) -> tuple[bool, float, float, float, int]:
    return (_passes_gate(item, RewardConfig()), item.get("net_reward_pct") or 0.0, item.get("top_precision") or 0.0, -(item.get("false_candidates") or 0), item.get("rewarded_top_candidates") or 0)


def _passes_gate(item: dict[str, Any] | None, cfg: RewardConfig) -> bool:
    if not item:
        return False
    return (
        item.get("net_reward_pct", 0.0) >= cfg.min_net_reward_pct
        and item.get("top_precision", 0.0) >= cfg.min_top_precision
        and item.get("rewarded_top_candidates", 0) >= cfg.min_rescued_top
        and item.get("candidate_count", 0) <= cfg.max_candidate_count
    )


def _decision(best: dict[str, Any] | None, cfg: RewardConfig) -> str:
    if not best:
        return "no_candidates"
    if _passes_gate(best, cfg):
        return "advance_to_entry_admission_behavior_replay"
    if (best.get("net_reward_pct") or 0.0) > 0:
        return "collect_more_or_refine_false_candidate_penalty"
    return "no_positive_shadow_reward"


def _num(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        out = float(value)
        return out if out == out else default
    except Exception:
        return default


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "н/д"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research-only entry admission shadow reward report")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--files-dir", type=Path, default=FILES)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--false-candidate-penalty-pct", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    payload = build_report(
        reports_dir=args.reports_dir,
        files_dir=args.files_dir,
        output_json=args.output_json,
        output_txt=args.output_txt,
        cfg=RewardConfig(false_candidate_penalty_pct=args.false_candidate_penalty_pct),
        save=not args.no_save,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
