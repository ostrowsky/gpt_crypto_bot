from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import audit_early_block_rescue_event_replay as event_replay
import research_artifact_provenance as artifact_provenance

ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_OUTPUT_JSON = REPORTS / "blocked_winner_causal_reward_latest.json"
DEFAULT_OUTPUT_TXT = REPORTS / "blocked_winner_causal_reward_latest.txt"


@dataclass(frozen=True)
class BlockerRewardConfig:
    false_candidate_credit_pct: float = 1.0
    late_capture_threshold: float = 0.25
    min_harm_pct: float = 10.0
    min_harmful_cases: int = 2


def build_report(
    reports_dir: Path = REPORTS,
    files_dir: Path = FILES,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
    cfg: BlockerRewardConfig = BlockerRewardConfig(),
    save: bool = True,
) -> dict[str, Any]:
    labels = event_replay._load_labels(reports_dir)
    events = event_replay._load_blocked_events(files_dir, labels)
    entries = event_replay._load_entries(files_dir, labels)
    rows = _case_rows(events, entries, labels, cfg)
    table = _reason_table(rows, cfg)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "research_only",
        "config": cfg.__dict__,
        "coverage": {
            "label_days": len({day for day, _ in labels}),
            "labeled_day_symbols": len(labels),
            "blocked_events_loaded": len(events),
            "case_rows": len(rows),
        },
        "reason_table": table,
        "top_harmful_reasons": [r for r in table if r["decision"] == "advance_to_behavior_replay"][:10],
        "decision": _decision(table),
        "provenance": artifact_provenance.build_provenance(
            builder="blocked_winner_causal_reward_v1",
            research_config=cfg,
            input_paths=[
                files_dir / "bot_events.jsonl",
                files_dir / "agent_events.jsonl",
                *([latest_critic] if (latest_critic := artifact_provenance.latest_path(reports_dir, "top_gainer_critic_*_final.json")) else []),
            ],
        ),
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
    lines = [
        "Blocked-winner causal reward table (research-only)",
        f"coverage: days={c.get('label_days')} labels={c.get('labeled_day_symbols')} blocked_events={c.get('blocked_events_loaded')} cases={c.get('case_rows')}",
        f"decision: {report.get('decision')}",
        "",
        "Top blocker reasons:",
    ]
    for row in (report.get("reason_table") or [])[:12]:
        lines.append(
            f"  {row.get('reason_code')}: net_harm={row.get('net_harm_pct')}% harm={row.get('harm_pct')}% "
            f"protect={row.get('protection_credit_pct')}% top={row.get('top_cases')} false={row.get('false_cases')} decision={row.get('decision')}"
        )
    return "\n".join(lines) + "\n"


def _case_rows(events: list[dict], entries: dict, labels: dict, cfg: BlockerRewardConfig) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for ev in events:
        key = (str(ev.get("day")), str(ev.get("symbol")), str(ev.get("reason_code") or "blocked_unknown"))
        grouped[key].append(ev)
    out = []
    for (day, symbol, reason), evs in grouped.items():
        evs = sorted(evs, key=lambda r: str(r.get("ts") or ""))
        first = evs[0]
        label = labels.get((day, symbol), {"is_top15": False, "status": "not_top15"})
        entry_rows = sorted(entries.get((day, symbol), []), key=lambda r: str(r.get("ts") or ""))
        bought_before = bool(entry_rows and str(entry_rows[0].get("ts") or "") <= str(first.get("ts") or ""))
        harm = _harm_reward(label, bought_before, cfg)
        protection = 0.0 if label.get("is_top15") else cfg.false_candidate_credit_pct
        out.append({
            "day": day,
            "symbol": symbol,
            "reason_code": reason,
            "first_ts": first.get("ts"),
            "first_hour": first.get("hour"),
            "block_count": len(evs),
            "is_top15": bool(label.get("is_top15")),
            "critic_status": label.get("status"),
            "capture_ratio_at_entry": label.get("capture_ratio_at_entry"),
            "opportunity_from_first_block_pct": label.get("opportunity_from_first_block_pct"),
            "bought_before_block": bought_before,
            "harm_pct": round(harm, 6),
            "protection_credit_pct": round(protection, 6),
            "net_harm_pct": round(harm - protection, 6),
        })
    return out


def _harm_reward(label: dict, bought_before: bool, cfg: BlockerRewardConfig) -> float:
    if not label.get("is_top15") or bought_before:
        return 0.0
    status = str(label.get("status") or "")
    opportunity = _num(label.get("opportunity_from_first_block_pct"), 0.0) or 0.0
    capture = _num(label.get("capture_ratio_at_entry"), None)
    if status != "bought":
        return max(0.0, opportunity)
    if capture is None or capture <= cfg.late_capture_threshold:
        return max(0.0, opportunity * (1.0 - max(0.0, min(1.0, capture or 0.0))))
    return 0.0


def _reason_table(rows: list[dict[str, Any]], cfg: BlockerRewardConfig) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["reason_code"]].append(row)
    out = []
    for reason, items in grouped.items():
        harm_cases = [r for r in items if r["harm_pct"] > 0]
        false_cases = [r for r in items if not r["is_top15"]]
        harm = sum(r["harm_pct"] for r in items)
        protection = sum(r["protection_credit_pct"] for r in items)
        net = harm - protection
        decision = "advance_to_behavior_replay" if harm >= cfg.min_harm_pct and len(harm_cases) >= cfg.min_harmful_cases and net > 0 else "keep_or_monitor"
        out.append({
            "reason_code": reason,
            "cases": len(items),
            "top_cases": sum(1 for r in items if r["is_top15"]),
            "false_cases": len(false_cases),
            "harmful_cases": len(harm_cases),
            "harm_pct": round(harm, 6),
            "protection_credit_pct": round(protection, 6),
            "net_harm_pct": round(net, 6),
            "top_precision": round(sum(1 for r in items if r["is_top15"]) / len(items), 6) if items else 0.0,
            "decision": decision,
            "top_harm_examples": sorted(harm_cases, key=lambda r: r["harm_pct"], reverse=True)[:10],
            "protected_examples": sorted(false_cases, key=lambda r: r["block_count"], reverse=True)[:8],
        })
    out.sort(key=lambda r: (r["decision"] == "advance_to_behavior_replay", r["net_harm_pct"], r["harm_pct"]), reverse=True)
    return out


def _decision(table: list[dict[str, Any]]) -> str:
    actionable = [r for r in table if r.get("decision") == "advance_to_behavior_replay"]
    if actionable:
        return "advance_top_harmful_blockers_to_behavior_replay"
    if any((r.get("harm_pct") or 0.0) > 0 for r in table):
        return "monitor_no_blocker_passed_harm_gate"
    return "no_harmful_blocker_detected"


def _num(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        out = float(value)
        return out if out == out else default
    except Exception:
        return default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research-only blocked-winner causal reward table")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--files-dir", type=Path, default=FILES)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--false-candidate-credit-pct", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    payload = build_report(
        reports_dir=args.reports_dir,
        files_dir=args.files_dir,
        output_json=args.output_json,
        output_txt=args.output_txt,
        cfg=BlockerRewardConfig(false_candidate_credit_pct=args.false_candidate_credit_pct),
        save=not args.no_save,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
