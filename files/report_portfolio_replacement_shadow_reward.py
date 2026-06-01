from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_OUTPUT_JSON = REPORTS / "portfolio_replacement_shadow_reward_latest.json"
DEFAULT_OUTPUT_TXT = REPORTS / "portfolio_replacement_shadow_reward_latest.txt"
REPL_RE = re.compile(r"portfolio replacement:\s*([A-Z0-9]+USDT)\s+leader\s+([\d.\-]+)\s*>\s*([A-Z0-9]+USDT)\s+leader\s+([\d.\-]+)", re.I)


@dataclass(frozen=True)
class ReplacementConfig:
    match_window_minutes: int = 20
    min_closed_cases: int = 10
    min_avg_delta_pct: float = 0.10


def build_report(files_dir: Path = FILES, reports_dir: Path = REPORTS, output_json: Path = DEFAULT_OUTPUT_JSON, output_txt: Path = DEFAULT_OUTPUT_TXT, cfg: ReplacementConfig = ReplacementConfig(), save: bool = True) -> dict[str, Any]:
    events = _load_events(files_dir)
    labels = _load_watchlist_labels(reports_dir)
    replacements = _replacement_rows(events, labels, cfg)
    closed = [r for r in replacements if r.get("incoming_exit_pnl_pct") is not None]
    deltas = [r["replacement_delta_pct"] for r in closed if r.get("replacement_delta_pct") is not None]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "research_only",
        "config": cfg.__dict__,
        "coverage": {"events_loaded": len(events), "replacement_events": len(replacements), "closed_incoming": len(closed)},
        "summary": {
            "replacement_count": len(replacements),
            "closed_incoming_count": len(closed),
            "avg_replaced_exit_pnl_pct": _avg([r.get("replaced_exit_pnl_pct") for r in replacements]),
            "avg_incoming_exit_pnl_pct": _avg([r.get("incoming_exit_pnl_pct") for r in closed]),
            "median_incoming_exit_pnl_pct": _median([r.get("incoming_exit_pnl_pct") for r in closed]),
            "avg_replacement_delta_pct": _avg(deltas),
            "median_replacement_delta_pct": _median(deltas),
            "positive_delta_rate_pct": round(sum(1 for d in deltas if d > 0) / len(deltas) * 100, 2) if deltas else 0.0,
            "incoming_watchlist_top_count": sum(1 for r in replacements if r.get("incoming_watchlist_top")),
        },
        "segments": _segment_table(closed),
        "policy_simulations": _policy_simulations(closed),
        "top_positive": sorted(closed, key=lambda r: r.get("replacement_delta_pct") or 0.0, reverse=True)[:12],
        "top_negative": sorted(closed, key=lambda r: r.get("replacement_delta_pct") or 0.0)[:12],
        "decision": "",
    }
    payload["decision"] = _decision(payload["summary"], cfg)
    text = render_text(payload)
    if save:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output_txt.write_text(text, encoding="utf-8")
        payload["files"] = {"json": str(output_json), "txt": str(output_txt)}
    return payload


def render_text(report: dict[str, Any]) -> str:
    c = report.get("coverage") or {}; s = report.get("summary") or {}
    segments = report.get("segments") or []
    policies = report.get("policy_simulations") or []
    lines = [
        "Portfolio replacement shadow reward (research-only)",
        f"coverage: events={c.get('events_loaded')} replacements={c.get('replacement_events')} closed={c.get('closed_incoming')}",
        f"decision: {report.get('decision')}",
        "",
        f"avg_delta={s.get('avg_replacement_delta_pct')}% median_delta={s.get('median_replacement_delta_pct')}% positive={s.get('positive_delta_rate_pct')}%",
        f"avg_replaced_exit={s.get('avg_replaced_exit_pnl_pct')}% avg_incoming_exit={s.get('avg_incoming_exit_pnl_pct')}% watchlist_top_incoming={s.get('incoming_watchlist_top_count')}",
    ]
    if segments:
        lines.extend(["", "segments:"])
        for row in segments[:8]:
            lines.append(
                f"- {row.get('segment')}: n={row.get('closed_count')} "
                f"avg_delta={row.get('avg_delta_pct')}% med={row.get('median_delta_pct')}% "
                f"positive={row.get('positive_delta_rate_pct')}%"
            )
    if policies:
        lines.extend(["", "policy simulations:"])
        for row in policies[:8]:
            lines.append(
                f"- {row.get('policy')}: block={row.get('blocked_count')} "
                f"net_saved={row.get('net_saved_delta_pct')}% regret={row.get('regret_rate_pct')}% "
                f"decision={row.get('decision')}"
            )
    return "\n".join(lines) + "\n"


def _load_events(files_dir: Path) -> list[dict[str, Any]]:
    out=[]
    for name in ("agent_events.jsonl", "bot_events.jsonl"):
        p=files_dir/name
        if not p.exists(): continue
        source="market_agent" if name.startswith("agent") else "bot"
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            try: row=json.loads(line)
            except Exception: continue
            if isinstance(row, dict):
                row.setdefault("source", source); out.append(row)
    out.sort(key=lambda r: str(r.get("ts") or ""))
    return out


def _replacement_rows(events: list[dict[str, Any]], labels: dict[tuple[str,str], dict], cfg: ReplacementConfig) -> list[dict[str, Any]]:
    entries=[e for e in events if e.get("event")=="entry"]
    exits=[e for e in events if e.get("event")=="exit"]
    out=[]
    for ev in exits:
        m=REPL_RE.search(str(ev.get("reason") or ""))
        if not m: continue
        incoming, incoming_leader, replaced, replaced_leader = m.group(1).upper(), float(m.group(2)), m.group(3).upper(), float(m.group(4))
        ts=_parse_ts(ev.get("ts")); day=ts.date().isoformat() if ts else ""
        incoming_entry=_find_near_entry(entries, incoming, ts, cfg.match_window_minutes)
        incoming_exit=_find_next_exit(exits, incoming, incoming_entry.get("ts") if incoming_entry else ev.get("ts"))
        replaced_pnl=_num(ev.get("pnl_pct"))
        incoming_pnl=_num(incoming_exit.get("pnl_pct")) if incoming_exit else None
        delta=None if replaced_pnl is None or incoming_pnl is None else round(incoming_pnl - replaced_pnl, 6)
        label=labels.get((day,incoming),{})
        out.append({
            "ts": ev.get("ts"), "day": day,
            "incoming_symbol": incoming, "replaced_symbol": replaced,
            "incoming_leader": incoming_leader, "replaced_leader": replaced_leader,
            "leader_delta": round(incoming_leader-replaced_leader,6),
            "replaced_exit_pnl_pct": replaced_pnl,
            "incoming_entry_ts": incoming_entry.get("ts") if incoming_entry else None,
            "incoming_entry_price": _num(incoming_entry.get("price")) if incoming_entry else None,
            "incoming_exit_ts": incoming_exit.get("ts") if incoming_exit else None,
            "incoming_exit_pnl_pct": incoming_pnl,
            "replacement_delta_pct": delta,
            "incoming_watchlist_top": bool(label),
            "incoming_watchlist_status": label.get("status"),
            "incoming_capture_ratio_at_entry": label.get("capture_ratio_at_entry"),
        })
    return out


def _find_near_entry(entries: list[dict[str,Any]], sym: str, ts: datetime | None, minutes: int) -> dict[str,Any] | None:
    if ts is None: return None
    end=ts+timedelta(minutes=minutes)
    for e in entries:
        if str(e.get("sym") or e.get("symbol") or "").upper()!=sym: continue
        et=_parse_ts(e.get("ts"))
        if et and ts <= et <= end: return e
    return None


def _find_next_exit(exits: list[dict[str,Any]], sym: str, after_ts: Any) -> dict[str,Any] | None:
    after=_parse_ts(after_ts)
    if after is None: return None
    for e in exits:
        if str(e.get("sym") or e.get("symbol") or "").upper()!=sym: continue
        et=_parse_ts(e.get("ts"))
        if et and et > after and not REPL_RE.search(str(e.get("reason") or "")):
            return e
    return None


def _load_watchlist_labels(reports_dir: Path) -> dict[tuple[str,str],dict]:
    out={}
    for p in sorted(reports_dir.glob("top_gainer_critic_*_final.json")):
        try: d=json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception: continue
        day=str(d.get("target_day_local") or "")
        for r in d.get("watchlist_top_gainers") or []:
            sym=str(r.get("symbol") or "").upper()
            if day and sym: out[(day,sym)]={"status":r.get("status"),"capture_ratio_at_entry":r.get("capture_ratio_at_entry")}
    return out


def _decision(summary: dict[str, Any], cfg: ReplacementConfig) -> str:
    closed=int(summary.get("closed_incoming_count") or 0)
    avg=_num(summary.get("avg_replacement_delta_pct"),0.0) or 0.0
    med=_num(summary.get("median_replacement_delta_pct"),0.0) or 0.0
    if closed < cfg.min_closed_cases: return "collect_more_replacement_outcomes"
    if avg >= cfg.min_avg_delta_pct and med >= 0: return "advance_replacement_policy_to_counterfactual_replay"
    if avg < 0: return "replacement_policy_hurting_in_shadow_monitor"
    return "replacement_policy_neutral_keep_collecting"


def _segment_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("all", lambda r: True),
        ("incoming_watchlist_top", lambda r: bool(r.get("incoming_watchlist_top"))),
        ("incoming_not_watchlist_top", lambda r: not bool(r.get("incoming_watchlist_top"))),
        ("replaced_losing_at_rotation", lambda r: (_num(r.get("replaced_exit_pnl_pct"), 0.0) or 0.0) < 0.0),
        ("replaced_non_losing_at_rotation", lambda r: (_num(r.get("replaced_exit_pnl_pct"), 0.0) or 0.0) >= 0.0),
        ("leader_delta_lt_10", lambda r: (_num(r.get("leader_delta"), 0.0) or 0.0) < 10.0),
        ("leader_delta_10_to_20", lambda r: 10.0 <= (_num(r.get("leader_delta"), 0.0) or 0.0) < 20.0),
        ("leader_delta_ge_20", lambda r: (_num(r.get("leader_delta"), 0.0) or 0.0) >= 20.0),
        ("leader_delta_ge_15", lambda r: (_num(r.get("leader_delta"), 0.0) or 0.0) >= 15.0),
    ]
    out = []
    for name, pred in specs:
        selected = [r for r in rows if pred(r) and r.get("replacement_delta_pct") is not None]
        if not selected:
            continue
        deltas = [r["replacement_delta_pct"] for r in selected]
        out.append({
            "segment": name,
            "closed_count": len(selected),
            "avg_delta_pct": _avg(deltas),
            "median_delta_pct": _median(deltas),
            "positive_delta_rate_pct": round(sum(1 for d in deltas if d > 0) / len(deltas) * 100, 2),
            "avg_replaced_exit_pnl_pct": _avg([r.get("replaced_exit_pnl_pct") for r in selected]),
            "avg_incoming_exit_pnl_pct": _avg([r.get("incoming_exit_pnl_pct") for r in selected]),
        })
    return out


def _policy_simulations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        (
            "block_replaced_non_losing",
            "causal",
            lambda r: (_num(r.get("replaced_exit_pnl_pct"), 0.0) or 0.0) >= 0.0,
        ),
        (
            "block_non_losing_unless_leader_delta_ge_15",
            "causal",
            lambda r: (_num(r.get("replaced_exit_pnl_pct"), 0.0) or 0.0) >= 0.0
            and (_num(r.get("leader_delta"), 0.0) or 0.0) < 15.0,
        ),
        (
            "block_non_losing_unless_leader_delta_ge_20",
            "causal",
            lambda r: (_num(r.get("replaced_exit_pnl_pct"), 0.0) or 0.0) >= 0.0
            and (_num(r.get("leader_delta"), 0.0) or 0.0) < 20.0,
        ),
        (
            "block_leader_delta_lt_10",
            "causal",
            lambda r: (_num(r.get("leader_delta"), 0.0) or 0.0) < 10.0,
        ),
        (
            "block_incoming_not_watchlist_top",
            "diagnostic_future_label",
            lambda r: not bool(r.get("incoming_watchlist_top")),
        ),
    ]
    out = []
    for name, kind, pred in specs:
        blocked = [r for r in rows if pred(r) and r.get("replacement_delta_pct") is not None]
        kept = [r for r in rows if r not in blocked and r.get("replacement_delta_pct") is not None]
        if not blocked:
            continue
        deltas = [float(r["replacement_delta_pct"]) for r in blocked]
        savings = [-d for d in deltas]
        missed_positive = [d for d in deltas if d > 0]
        avoided_negative = [-d for d in deltas if d < 0]
        net_saved = round(sum(savings), 6)
        row = {
            "policy": name,
            "kind": kind,
            "blocked_count": len(blocked),
            "kept_count": len(kept),
            "net_saved_delta_pct": net_saved,
            "avg_saved_per_block_pct": round(mean(savings), 6),
            "median_saved_per_block_pct": round(median(savings), 6),
            "avoided_negative_delta_pct": round(sum(avoided_negative), 6),
            "missed_positive_delta_pct": round(sum(missed_positive), 6),
            "regret_count": len(missed_positive),
            "regret_rate_pct": round(len(missed_positive) / len(blocked) * 100, 2),
            "decision": _policy_decision(name, kind, len(blocked), net_saved, len(missed_positive) / len(blocked) * 100),
        }
        out.append(row)
    out.sort(key=lambda r: (r.get("kind") != "causal", -(r.get("net_saved_delta_pct") or 0.0)))
    return out


def _policy_decision(name: str, kind: str, blocked_count: int, net_saved: float, regret_rate_pct: float) -> str:
    if kind != "causal":
        return "diagnostic_only_future_label"
    if blocked_count < 5:
        return "collect_more_cases"
    if net_saved >= 2.0 and regret_rate_pct <= 25.0:
        return "advance_to_behavior_replay"
    if net_saved > 0:
        return "monitor_positive_but_not_promoted"
    return "reject_or_keep_current"


def _parse_ts(v: Any) -> datetime | None:
    try: return datetime.fromisoformat(str(v).replace("Z","+00:00")).astimezone(timezone.utc)
    except Exception: return None

def _num(v: Any, default: float | None=None) -> float | None:
    try:
        if v is None: return default
        f=float(v); return f if f==f else default
    except Exception: return default

def _avg(vals) -> float | None:
    nums=[float(v) for v in vals if v is not None]
    return round(mean(nums),6) if nums else None

def _median(vals) -> float | None:
    nums=[float(v) for v in vals if v is not None]
    return round(median(nums),6) if nums else None


def main(argv: list[str] | None=None) -> int:
    ap=argparse.ArgumentParser(description="Research-only portfolio replacement shadow reward report")
    ap.add_argument("--files-dir", type=Path, default=FILES); ap.add_argument("--reports-dir", type=Path, default=REPORTS)
    ap.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON); ap.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    ap.add_argument("--json", action="store_true"); ap.add_argument("--no-save", action="store_true")
    args=ap.parse_args(argv)
    payload=build_report(args.files_dir,args.reports_dir,args.output_json,args.output_txt,save=not args.no_save)
    print(json.dumps(payload,ensure_ascii=False,indent=2) if args.json else render_text(payload)); return 0

if __name__=="__main__":
    raise SystemExit(main())
