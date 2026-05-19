from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_OUTPUT = REPORTS / "early_block_rescue_audit_latest.json"

REASON_SETS = {
    "agent_only": {"agent_mode_disabled", "agent_leader_filter"},
    "score_only": {"top_gainer_score_gate"},
    "agent_plus_score": {"agent_mode_disabled", "agent_leader_filter", "top_gainer_score_gate"},
    "agent_score_chase": {"agent_mode_disabled", "agent_leader_filter", "top_gainer_score_gate", "chase_guard"},
}


def build(reports_dir: Path = REPORTS, output: Path = DEFAULT_OUTPUT) -> dict:
    rows = _load_rows(reports_dir)
    variants = []
    for reason_name, reasons in REASON_SETS.items():
        for max_hour in (2, 4, 6, 8, 12):
            for min_blocks in (5, 20, 50):
                for min_opp in (0.0, 1.0, 3.0):
                    variants.append(_evaluate(rows, reason_name, reasons, max_hour, min_blocks, min_opp))
    variants = sorted(
        variants,
        key=lambda item: (
            _is_admissible(item),
            item["proxy_gain_pct"],
            item["rescued_missed_winners"],
            item["rescued_late_bought_winners"],
            -item["non_positive_cases"],
        ),
        reverse=True,
    )
    best = variants[0] if variants else None
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_rows": len(rows),
        "unique_day_symbols": len({(r["day"], r["symbol"]) for r in rows}),
        "reason_sets": {k: sorted(v) for k, v in REASON_SETS.items()},
        "best_variant": best,
        "top_variants": variants[:20],
        "decision": _decision(best),
        "next_replay_gate": {
            "required": True,
            "description": "Implement candle-level replay using only causal early-block features; must improve capture_ratio_at_entry and net PnL without increasing false_positive_buys by more than 10%.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _load_rows(reports_dir: Path) -> list[dict]:
    best: dict[tuple[str, str], dict] = {}
    for path in sorted(reports_dir.glob("top_gainer_critic_*_final.json")):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        day = data.get("target_day_local") or _day_from_name(path.name)
        for row in data.get("watchlist_top_gainers") or []:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol") or "")
            if not symbol or not day:
                continue
            normalized = _normalize_row(row, str(day), str(path))
            key = (normalized["day"], normalized["symbol"])
            if key not in best or normalized["blocked_count"] > best[key]["blocked_count"]:
                best[key] = normalized
    return list(best.values())


def _normalize_row(row: dict, day: str, source_file: str) -> dict:
    first_block_hour = _hour(row.get("first_block_time"))
    opportunity_from_block = _num(row.get("opportunity_from_first_block_pct"), _num(row.get("opportunity_no_entry_pct")))
    opportunity_from_entry = _num(row.get("opportunity_from_entry_pct"))
    status = str(row.get("status") or "")
    if status == "bought":
        potential_gain = max(0.0, opportunity_from_block - opportunity_from_entry)
    else:
        potential_gain = max(0.0, opportunity_from_block)
    return {
        "day": day,
        "source_file": source_file,
        "symbol": str(row.get("symbol") or ""),
        "status": status,
        "day_change_pct": _num(row.get("day_change_pct")),
        "first_block_time": row.get("first_block_time"),
        "first_block_hour": first_block_hour,
        "first_block_reason_code": str(row.get("first_block_reason_code") or row.get("missed_reason_code") or "unknown"),
        "blocked_count": int(row.get("blocked_count") or 0),
        "blocked_reason_counts": row.get("blocked_reason_counts") or {},
        "first_entry_time": row.get("first_entry_time"),
        "capture_ratio_at_entry": _num(row.get("capture_ratio_at_entry"), default=None),
        "opportunity_from_first_block_pct": opportunity_from_block,
        "opportunity_from_entry_pct": opportunity_from_entry,
        "potential_gain_pct": potential_gain,
        "latest_exit_pnl_pct": _num(row.get("latest_exit_pnl_pct"), default=None),
        "giveback_pct": _num(row.get("giveback_pct"), default=None),
    }


def _evaluate(rows: list[dict], reason_name: str, reasons: set[str], max_hour: int, min_blocks: int, min_opp: float) -> dict:
    selected = [
        row
        for row in rows
        if row["first_block_hour"] is not None
        and row["first_block_hour"] <= max_hour
        and row["blocked_count"] >= min_blocks
        and row["first_block_reason_code"] in reasons
        and row["opportunity_from_first_block_pct"] >= min_opp
    ]
    missed = [row for row in selected if row["status"] != "bought"]
    late_bought = [row for row in selected if row["status"] == "bought" and (row.get("capture_ratio_at_entry") is None or row.get("capture_ratio_at_entry", 1.0) <= 0.25)]
    non_positive = [row for row in selected if row["opportunity_from_first_block_pct"] <= 0]
    proxy_gain = sum(row["potential_gain_pct"] for row in missed + late_bought)
    reason_counts = Counter(row["first_block_reason_code"] for row in selected)
    return {
        "reason_set": reason_name,
        "max_first_block_hour": max_hour,
        "min_blocked_count": min_blocks,
        "min_opportunity_from_block_pct": min_opp,
        "selected_cases": len(selected),
        "rescued_missed_winners": len(missed),
        "rescued_late_bought_winners": len(late_bought),
        "non_positive_cases": len(non_positive),
        "proxy_gain_pct": round(proxy_gain, 6),
        "avg_proxy_gain_pct": round(proxy_gain / max(1, len(missed) + len(late_bought)), 6),
        "first_block_reason_counts": dict(reason_counts.most_common()),
        "top_examples": sorted(missed + late_bought, key=lambda row: row["potential_gain_pct"], reverse=True)[:12],
    }


def _is_admissible(item: dict) -> bool:
    return (
        item["proxy_gain_pct"] >= 25.0
        and item["rescued_missed_winners"] >= 3
        and item["non_positive_cases"] <= max(2, item["selected_cases"] * 0.10)
    )


def _decision(best: dict | None) -> str:
    if not best:
        return "no_candidate"
    if best["proxy_gain_pct"] < 25.0 or best["rescued_missed_winners"] < 3:
        return "diagnostic_only_insufficient_proxy_gain"
    if best["non_positive_cases"] > max(2, best["selected_cases"] * 0.10):
        return "diagnostic_only_too_many_non_positive_cases"
    return "advance_to_candle_level_replay"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _day_from_name(name: str) -> str | None:
    match = re.search(r"(20\d\d-\d\d-\d\d)", name)
    return match.group(1) if match else None


def _hour(value) -> int | None:
    if not value:
        return None
    match = re.match(r"^(\d{1,2})", str(value))
    if not match:
        return None
    hour = int(match.group(1))
    return hour if 0 <= hour <= 23 else None


def _num(value, default: float | None = 0.0):
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.reports_dir, args.output)
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print({"best_variant": payload["best_variant"], "decision": payload["decision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
