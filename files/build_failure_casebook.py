from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_JSON = REPORTS / "failure_casebook_latest.json"
DEFAULT_MD = REPORTS / "failure_casebook_latest.md"


def build(reports_dir: Path = REPORTS, limit: int = 30) -> dict:
    signal_reports = _load_signal_quality(reports_dir)
    critic_reports = _load_top_gainer_critics(reports_dir)
    exit_cases = _worst_exit_cases(signal_reports, limit)
    late_entry_cases = _late_entry_cases(critic_reports, signal_reports, limit)
    missed_cases = _missed_or_blocked_cases(critic_reports, limit)
    false_positive_cases = _false_positive_cases(signal_reports, limit)
    blocker_summary = _blocker_summary(critic_reports)
    hypotheses = _hypotheses(exit_cases, late_entry_cases, missed_cases, false_positive_cases, blocker_summary)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_counts": {
            "signal_quality_reports": len(signal_reports),
            "top_gainer_critic_reports": len(critic_reports),
        },
        "worst_exit_cases": exit_cases,
        "late_entry_cases": late_entry_cases,
        "missed_or_blocked_winner_cases": missed_cases,
        "false_positive_cases": false_positive_cases,
        "blocker_summary": blocker_summary,
        "hypotheses": hypotheses[:3],
    }


def _load_signal_quality(reports_dir: Path) -> list[dict]:
    out = []
    for path in sorted(reports_dir.glob("signal_quality_*_final.json")):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        out.append({"path": str(path), "day": _day_from_name(path.name) or data.get("target_day_local"), "data": data})
    return out


def _load_top_gainer_critics(reports_dir: Path) -> list[dict]:
    out = []
    for path in sorted(reports_dir.glob("top_gainer_critic_*_final.json")):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        out.append({"path": str(path), "day": data.get("target_day_local") or _day_from_name(path.name), "data": data})
    return out


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _day_from_name(name: str) -> str | None:
    match = re.search(r"(20\d\d-\d\d-\d\d)", name)
    return match.group(1) if match else None


def _worst_exit_cases(signal_reports: list[dict], limit: int) -> list[dict]:
    rows = []
    for report in signal_reports:
        data = report["data"]
        for bucket in ("late_entries", "early_exits", "false_positive_buys", "trades"):
            for row in data.get(bucket) or []:
                if not isinstance(row, dict):
                    continue
                giveback = _num(row.get("giveback_pct"))
                pnl = _num(row.get("pnl_pct"))
                mfe = _num(row.get("max_favorable_pct"))
                exit_eff = _num(row.get("exit_efficiency"))
                score = giveback + max(0.0, -pnl) + max(0.0, mfe - max(0.0, pnl))
                rows.append(
                    {
                        "day": report["day"],
                        "source_file": report["path"],
                        "bucket": bucket,
                        "symbol": row.get("sym") or row.get("symbol"),
                        "tf": row.get("tf"),
                        "mode": row.get("mode"),
                        "entry_ts": row.get("entry_ts"),
                        "exit_ts": row.get("exit_ts"),
                        "pnl_pct": pnl,
                        "max_favorable_pct": mfe,
                        "giveback_pct": giveback,
                        "exit_efficiency": exit_eff,
                        "entry_timing": row.get("entry_timing"),
                        "exit_timing": row.get("exit_timing"),
                        "exit_reason": row.get("exit_reason"),
                        "score": round(score, 6),
                    }
                )
    return _dedupe_cases(rows, ["day", "bucket", "symbol", "tf", "mode"], limit)


def _late_entry_cases(critic_reports: list[dict], signal_reports: list[dict], limit: int) -> list[dict]:
    rows = []
    for report in critic_reports:
        for row in report["data"].get("watchlist_top_gainers") or []:
            if not isinstance(row, dict) or row.get("status") != "bought":
                continue
            capture = _num(row.get("capture_ratio_at_entry"))
            opportunity = _num(row.get("opportunity_from_entry_pct"))
            score = max(0.0, 1.0 - capture) + max(0.0, -opportunity / 5.0)
            rows.append(
                {
                    "day": report["day"],
                    "source_file": report["path"],
                    "symbol": row.get("symbol"),
                    "day_change_pct": _num(row.get("day_change_pct")),
                    "first_block_time": row.get("first_block_time"),
                    "first_block_reason_code": row.get("first_block_reason_code"),
                    "first_entry_time": row.get("first_entry_time"),
                    "first_entry_mode": row.get("first_entry_mode"),
                    "first_entry_source": row.get("first_entry_source"),
                    "capture_ratio_at_entry": capture,
                    "opportunity_from_entry_pct": opportunity,
                    "latest_exit_pnl_pct": _num(row.get("latest_exit_pnl_pct")),
                    "giveback_pct": _num(row.get("giveback_pct")),
                    "blocked_reason_counts": row.get("blocked_reason_counts") or {},
                    "score": round(score, 6),
                }
            )
    return _dedupe_cases(rows, ["day", "symbol"], limit)


def _missed_or_blocked_cases(critic_reports: list[dict], limit: int) -> list[dict]:
    rows = []
    for report in critic_reports:
        for row in report["data"].get("watchlist_top_gainers") or []:
            if not isinstance(row, dict) or row.get("status") == "bought":
                continue
            opportunity = _num(row.get("opportunity_no_entry_pct"), _num(row.get("opportunity_from_first_block_pct")))
            score = max(0.0, opportunity) + max(0.0, _num(row.get("day_change_pct")) / 10.0)
            rows.append(
                {
                    "day": report["day"],
                    "source_file": report["path"],
                    "symbol": row.get("symbol"),
                    "status": row.get("status"),
                    "missed_reason_code": row.get("missed_reason_code") or row.get("reason"),
                    "day_change_pct": _num(row.get("day_change_pct")),
                    "first_block_time": row.get("first_block_time"),
                    "first_block_reason_code": row.get("first_block_reason_code"),
                    "opportunity_no_entry_pct": _num(row.get("opportunity_no_entry_pct")),
                    "opportunity_from_first_block_pct": _num(row.get("opportunity_from_first_block_pct")),
                    "blocked_count": int(row.get("blocked_count") or 0),
                    "blocked_reason_counts": row.get("blocked_reason_counts") or {},
                    "score": round(score, 6),
                }
            )
    return _dedupe_cases(rows, ["day", "symbol"], limit)


def _false_positive_cases(signal_reports: list[dict], limit: int) -> list[dict]:
    rows = []
    for report in signal_reports:
        for row in report["data"].get("false_positive_buys") or []:
            if not isinstance(row, dict):
                continue
            pnl = _num(row.get("pnl_pct"))
            giveback = _num(row.get("giveback_pct"))
            rows.append(
                {
                    "day": report["day"],
                    "source_file": report["path"],
                    "symbol": row.get("sym") or row.get("symbol"),
                    "tf": row.get("tf"),
                    "source": row.get("source"),
                    "mode": row.get("mode"),
                    "entry_ts": row.get("entry_ts"),
                    "exit_ts": row.get("exit_ts"),
                    "pnl_pct": pnl,
                    "max_favorable_pct": _num(row.get("max_favorable_pct")),
                    "giveback_pct": giveback,
                    "exit_reason": row.get("exit_reason"),
                    "score": round(max(0.0, -pnl) + giveback, 6),
                }
            )
    return _dedupe_cases(rows, ["day", "symbol", "tf", "source", "mode"], limit)


def _dedupe_cases(rows: list[dict], key_fields: list[str], limit: int) -> list[dict]:
    best = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if key not in best or row.get("score", 0.0) > best[key].get("score", 0.0):
            best[key] = row
    return sorted(best.values(), key=lambda item: item.get("score", 0.0), reverse=True)[:limit]


def _blocker_summary(critic_reports: list[dict]) -> list[dict]:
    counts = Counter()
    missed_symbols = defaultdict(set)
    missed_opportunity = defaultdict(float)
    examples = defaultdict(list)
    for report in critic_reports:
        for harm in report["data"].get("blocked_reason_harm") or []:
            if not isinstance(harm, dict):
                continue
            reason = str(harm.get("reason_code") or "unknown")
            counts[reason] += int(harm.get("blocked_events") or 0)
            missed_opportunity[reason] += _num(harm.get("missed_opportunity_pct"))
            for sym in harm.get("missed_symbols") or []:
                missed_symbols[reason].add(str(sym))
            examples[reason].extend(harm.get("examples") or [])
    out = []
    for reason, count in counts.most_common():
        out.append(
            {
                "reason_code": reason,
                "blocked_events": count,
                "missed_symbols_count": len(missed_symbols[reason]),
                "missed_symbols": sorted(missed_symbols[reason])[:20],
                "missed_opportunity_pct_sum": round(missed_opportunity[reason], 6),
                "examples": examples[reason][:8],
            }
        )
    return out


def _hypotheses(exit_cases: list[dict], late_entries: list[dict], missed: list[dict], false_pos: list[dict], blockers: list[dict]) -> list[dict]:
    out = []
    if late_entries:
        top_reasons = Counter()
        for case in late_entries[:20]:
            for reason, count in (case.get("blocked_reason_counts") or {}).items():
                top_reasons[str(reason)] += int(count)
        out.append(
            {
                "name": "early-block-to-entry rescue case review",
                "why": "Many bought top movers had a much earlier block than entry; this is a concrete late-entry failure, not a generic recall problem.",
                "evidence": {
                    "top_late_entry_symbols": [case["symbol"] for case in late_entries[:10]],
                    "dominant_prior_blockers": top_reasons.most_common(5),
                },
                "candidate_change": "Replay a narrow rescue rule only for symbols with repeated early structural blocks that later satisfy existing production BUY quality gates.",
                "replay_gate": "Must improve capture_ratio_at_entry and net PnL without increasing false_positive_buys more than 10%.",
            }
        )
    if exit_cases:
        out.append(
            {
                "name": "MFE giveback case replay",
                "why": "Worst exit cases show realized PnL far below MFE; target one exit failure slice instead of training a broad exit model.",
                "evidence": {
                    "top_exit_symbols": [case["symbol"] for case in exit_cases[:10]],
                    "top_exit_reasons": Counter(str(case.get("exit_reason")) for case in exit_cases[:20]).most_common(5),
                },
                "candidate_change": "Replay a conservative MFE-protection rule only after trade MFE exceeds a minimum threshold and structure deteriorates.",
                "replay_gate": "Must improve total realized PnL and exit_efficiency while not increasing false early exits on top movers.",
            }
        )
    if false_pos:
        out.append(
            {
                "name": "false-positive entry slice veto",
                "why": "False positives are concrete losing buys with no matched trend; a narrow veto may be cheaper than broader admission redesign.",
                "evidence": {
                    "top_false_positive_symbols": [case["symbol"] for case in false_pos[:10]],
                    "top_false_positive_modes": Counter(str(case.get("mode")) for case in false_pos).most_common(5),
                },
                "candidate_change": "Replay a narrow veto for the worst false-positive mode/source slice only, keeping top-mover rescue untouched.",
                "replay_gate": "Must reduce false_positive_buys and improve PnL without reducing watchlist_top_bought.",
            }
        )
    return out


def _num(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if value != value:
            return default
        return value
    except Exception:
        return default


def write_markdown(payload: dict, path: Path) -> None:
    lines = ["# Failure Casebook", "", f"Generated: `{payload['generated_at_utc']}`", ""]
    lines.append("## Source Coverage")
    for key, value in payload["source_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    _md_table(lines, "Worst exit / MFE cases", payload["worst_exit_cases"], ["day", "symbol", "bucket", "pnl_pct", "max_favorable_pct", "giveback_pct", "exit_timing", "score"])
    _md_table(lines, "Late-entry top mover cases", payload["late_entry_cases"], ["day", "symbol", "first_block_time", "first_entry_time", "capture_ratio_at_entry", "opportunity_from_entry_pct", "score"])
    _md_table(lines, "Missed / blocked winner cases", payload["missed_or_blocked_winner_cases"], ["day", "symbol", "missed_reason_code", "first_block_time", "first_block_reason_code", "opportunity_from_first_block_pct", "score"])
    _md_table(lines, "False-positive cases", payload["false_positive_cases"], ["day", "symbol", "mode", "pnl_pct", "max_favorable_pct", "giveback_pct", "score"])
    lines.append("## Hypothesis Shortlist")
    for idx, hyp in enumerate(payload["hypotheses"], 1):
        lines.append(f"### {idx}. {hyp['name']}")
        lines.append(f"- Why: {hyp['why']}")
        lines.append(f"- Candidate change: {hyp['candidate_change']}")
        lines.append(f"- Replay gate: {hyp['replay_gate']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _md_table(lines: list[str], title: str, rows: list[dict], columns: list[str], max_rows: int = 10) -> None:
    lines.append(f"## {title}")
    if not rows:
        lines.append("No cases found.\n")
        return
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows[:max_rows]:
        vals = [str(row.get(col, ""))[:120].replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.reports_dir, args.limit)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, args.md_output)
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print({"source_counts": payload["source_counts"], "hypotheses": [h["name"] for h in payload["hypotheses"]]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
