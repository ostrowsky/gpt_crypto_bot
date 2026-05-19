from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from blocking import normalize_blocked_reason


ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_OUTPUT = REPORTS / "early_block_rescue_event_replay_latest.json"
TZ = ZoneInfo("Europe/Budapest")

REASON_SETS = {
    "agent_plus_score": {"agent_mode_disabled", "agent_leader_filter", "top_gainer_score_gate"},
    "agent_only": {"agent_mode_disabled", "agent_leader_filter"},
    "score_only": {"top_gainer_score_gate"},
}


def build(reports_dir: Path = REPORTS, files_dir: Path = FILES, output: Path = DEFAULT_OUTPUT) -> dict:
    labels = _load_labels(reports_dir)
    events = _load_blocked_events(files_dir, labels)
    entries = _load_entries(files_dir, labels)
    variants = []
    for reason_name, reasons in REASON_SETS.items():
        for max_hour in (2, 4, 6, 8, 12):
            for min_blocks in (3, 5, 10, 20, 50):
                variants.append(_evaluate(events, entries, labels, reason_name, reasons, max_hour, min_blocks))
    variants = sorted(
        variants,
        key=lambda item: (
            _passes_proxy_gate(item),
            item["missed_winner_coverage"],
            item["top15_precision"],
            -item["false_positive_candidates"],
            item["candidate_count"],
        ),
        reverse=True,
    )
    best = variants[0] if variants else None
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "label_days": len({day for day, _ in labels}),
        "labeled_day_symbols": len(labels),
        "blocked_events_loaded": len(events),
        "entries_loaded": len(entries),
        "best_variant": best,
        "top_variants": variants[:20],
        "decision": _decision(best),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _load_labels(reports_dir: Path) -> dict[tuple[str, str], dict]:
    labels = {}
    for path in sorted(reports_dir.glob("top_gainer_critic_*_final.json")):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        day = str(data.get("target_day_local") or _day_from_name(path.name) or "")
        for row in data.get("watchlist_top_gainers") or []:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol") or "")
            if not day or not symbol:
                continue
            labels[(day, symbol)] = {
                "is_top15": True,
                "status": str(row.get("status") or ""),
                "day_change_pct": _num(row.get("day_change_pct")),
                "opportunity_from_first_block_pct": _num(row.get("opportunity_from_first_block_pct"), _num(row.get("opportunity_no_entry_pct"))),
                "capture_ratio_at_entry": _num(row.get("capture_ratio_at_entry"), None),
            }
    return labels


def _load_blocked_events(files_dir: Path, labels: dict) -> list[dict]:
    allowed_days = {day for day, _ in labels}
    out = []
    for path in (files_dir / "bot_events.jsonl", files_dir / "agent_events.jsonl"):
        if not path.exists():
            continue
        for row in _iter_jsonl(path):
            if row.get("event") != "blocked":
                continue
            day, hour = _local_day_hour(row.get("ts"))
            if day not in allowed_days or hour is None:
                continue
            symbol = str(row.get("sym") or row.get("symbol") or "")
            if not symbol:
                continue
            reason = normalize_blocked_reason(str(row.get("signal_type") or ""), str(row.get("reason") or ""))
            out.append(
                {
                    "day": day,
                    "hour": hour,
                    "ts": row.get("ts"),
                    "symbol": symbol,
                    "reason_code": reason,
                    "source": row.get("source") or ("market_agent" if path.name.startswith("agent") else "bot"),
                    "price": _num(row.get("price"), None),
                    "tf": row.get("tf"),
                    "signal_type": row.get("signal_type"),
                }
            )
    return out


def _load_entries(files_dir: Path, labels: dict) -> dict[tuple[str, str], list[dict]]:
    allowed_days = {day for day, _ in labels}
    out = defaultdict(list)
    for path in (files_dir / "bot_events.jsonl", files_dir / "agent_events.jsonl"):
        if not path.exists():
            continue
        for row in _iter_jsonl(path):
            if row.get("event") != "entry":
                continue
            day, hour = _local_day_hour(row.get("ts"))
            if day not in allowed_days or hour is None:
                continue
            symbol = str(row.get("sym") or row.get("symbol") or "")
            if not symbol:
                continue
            out[(day, symbol)].append({"ts": row.get("ts"), "hour": hour, "price": _num(row.get("price"), None), "mode": row.get("mode")})
    return out


def _evaluate(events: list[dict], entries: dict, labels: dict, reason_name: str, reasons: set[str], max_hour: int, min_blocks: int) -> dict:
    grouped = defaultdict(list)
    for event in events:
        if event["hour"] <= max_hour and event["reason_code"] in reasons:
            grouped[(event["day"], event["symbol"])].append(event)
    candidates = []
    for key, rows in grouped.items():
        if len(rows) < min_blocks:
            continue
        rows = sorted(rows, key=lambda r: str(r.get("ts") or ""))
        label = labels.get(key, {"is_top15": False, "status": "not_top15"})
        first = rows[0]
        entry_rows = sorted(entries.get(key, []), key=lambda r: str(r.get("ts") or ""))
        bought_before = bool(entry_rows and str(entry_rows[0].get("ts") or "") <= str(first.get("ts") or ""))
        candidates.append(
            {
                "day": key[0],
                "symbol": key[1],
                "first_rescue_ts": first.get("ts"),
                "first_rescue_hour": first.get("hour"),
                "first_reason_code": first.get("reason_code"),
                "first_source": first.get("source"),
                "first_tf": first.get("tf"),
                "block_count": len(rows),
                "reason_counts": dict(Counter(r["reason_code"] for r in rows).most_common()),
                "is_top15": bool(label.get("is_top15")),
                "critic_status": label.get("status"),
                "day_change_pct": label.get("day_change_pct"),
                "opportunity_from_first_block_pct": label.get("opportunity_from_first_block_pct"),
                "capture_ratio_at_entry": label.get("capture_ratio_at_entry"),
                "had_entry": bool(entry_rows),
                "bought_before_rescue": bought_before,
                "first_entry_ts": entry_rows[0].get("ts") if entry_rows else None,
            }
        )
    top = [c for c in candidates if c["is_top15"]]
    false = [c for c in candidates if not c["is_top15"]]
    missed_top = [c for c in top if c.get("critic_status") != "bought"]
    late_top = [c for c in top if c.get("critic_status") == "bought" and (c.get("capture_ratio_at_entry") is None or c.get("capture_ratio_at_entry") <= 0.25)]
    label_top_missed_total = sum(1 for data in labels.values() if data.get("status") != "bought")
    return {
        "reason_set": reason_name,
        "max_first_block_hour": max_hour,
        "min_blocked_count": min_blocks,
        "candidate_count": len(candidates),
        "top15_candidates": len(top),
        "false_positive_candidates": len(false),
        "top15_precision": round(len(top) / len(candidates), 6) if candidates else 0.0,
        "false_positive_ratio": round(len(false) / len(candidates), 6) if candidates else 0.0,
        "missed_top15_candidates": len(missed_top),
        "late_bought_top15_candidates": len(late_top),
        "missed_winner_coverage": round(len(missed_top) / label_top_missed_total, 6) if label_top_missed_total else 0.0,
        "bought_before_rescue": sum(1 for c in candidates if c["bought_before_rescue"]),
        "proxy_top_opportunity_pct": round(sum(_num(c.get("opportunity_from_first_block_pct")) for c in missed_top), 6),
        "top_examples": sorted(missed_top + late_top, key=lambda c: _num(c.get("opportunity_from_first_block_pct")), reverse=True)[:15],
        "false_positive_examples": sorted(false, key=lambda c: c["block_count"], reverse=True)[:15],
    }


def _passes_proxy_gate(item: dict) -> bool:
    return (
        item["missed_top15_candidates"] >= 10
        and item["proxy_top_opportunity_pct"] >= 50.0
        and item["top15_precision"] >= 0.25
        and item["candidate_count"] <= 250
        and item["false_positive_ratio"] <= 0.75
    )


def _decision(best: dict | None) -> str:
    if not best:
        return "no_candidate"
    if _passes_proxy_gate(best):
        return "advance_to_candle_level_behavior_replay"
    return "diagnostic_only_rejected_event_proxy_gate"


def _iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            yield row


def _local_day_hour(ts) -> tuple[str | None, int | None]:
    if not ts:
        return None, None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(TZ)
        return dt.date().isoformat(), dt.hour
    except Exception:
        return None, None


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _day_from_name(name: str) -> str | None:
    match = re.search(r"(20\d\d-\d\d-\d\d)", name)
    return match.group(1) if match else None


def _num(value, default=0.0):
    try:
        if value is None:
            return default
        value = float(value)
        if value != value:
            return default
        return value
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--files-dir", type=Path, default=FILES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.reports_dir, args.files_dir, args.output)
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print({"best_variant": payload["best_variant"], "decision": payload["decision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
