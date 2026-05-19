from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from audit_early_block_rescue_event_replay import _load_blocked_events, _load_entries, _load_labels


ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_HISTORY = FILES / ".runtime" / "v2_history"
DEFAULT_OUTPUT = REPORTS / "post_block_causal_discriminator_dataset_15m.jsonl"
DEFAULT_AUDIT = REPORTS / "post_block_causal_discriminator_dataset_audit_15m.json"
TZ = ZoneInfo("Europe/Budapest")

ALLOWED_REASONS = {"agent_mode_disabled", "agent_leader_filter", "top_gainer_score_gate"}
HORIZON_BARS = {"15m": 1, "30m": 2, "60m": 4, "120m": 8}


def build(
    reports_dir: Path = REPORTS,
    files_dir: Path = FILES,
    history_root: Path = DEFAULT_HISTORY,
    output: Path = DEFAULT_OUTPUT,
    audit_output: Path = DEFAULT_AUDIT,
    max_hour: int = 12,
    min_blocks: int = 3,
) -> dict:
    labels = _load_labels(reports_dir)
    events = _load_blocked_events(files_dir, labels)
    entries = _load_entries(files_dir, labels)
    candidates = _candidate_events(events, labels, max_hour=max_hour, min_blocks=min_blocks)
    history_cache: dict[str, list[dict]] = {}
    btc_bars = _load_history(history_root, "BTCUSDT", history_cache)
    rows = []
    for candidate in candidates:
        bars = _load_history(history_root, candidate["symbol"], history_cache)
        feature_row = _row_from_candidate(candidate, labels, entries, bars, btc_bars)
        if feature_row:
            rows.append(feature_row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    audit = _audit(rows, labels, output, max_hour, min_blocks)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def _candidate_events(events: list[dict], labels: dict, max_hour: int, min_blocks: int) -> list[dict]:
    counts = defaultdict(int)
    emitted = set()
    out = []
    for event in sorted(events, key=lambda row: str(row.get("ts") or "")):
        if event["hour"] > max_hour or event["reason_code"] not in ALLOWED_REASONS:
            continue
        key = (event["day"], event["symbol"])
        counts[key] += 1
        if counts[key] >= min_blocks and key not in emitted:
            emitted.add(key)
            out.append({**event, "block_count_at_candidate": counts[key], "is_labeled_top15": key in labels})
    return out


def _row_from_candidate(candidate: dict, labels: dict, entries: dict, bars: list[dict], btc_bars: list[dict]) -> dict | None:
    if not bars:
        return None
    ts_ms = _ts_ms(candidate.get("ts"))
    idx = _first_bar_at_or_after(bars, ts_ms)
    if idx is None:
        return None
    key = (candidate["day"], candidate["symbol"])
    label = labels.get(key, {"is_top15": False, "status": "not_top15"})
    base = bars[idx]
    features = _features(bars, idx, btc_bars, ts_ms)
    entry_rows = sorted(entries.get(key, []), key=lambda row: str(row.get("ts") or ""))
    return {
        "local_day": candidate["day"],
        "symbol": candidate["symbol"],
        "candidate_ts": candidate.get("ts"),
        "candidate_bar_ts_ms": base["open_ts_ms"],
        "hour": candidate["hour"],
        "reason_code": candidate["reason_code"],
        "source": candidate.get("source"),
        "tf": candidate.get("tf"),
        "signal_type": candidate.get("signal_type"),
        "block_count_at_candidate": candidate["block_count_at_candidate"],
        "candidate_price": candidate.get("price"),
        "bar_close": base["close"],
        "features": features,
        "label_top15": bool(label.get("is_top15")),
        "label_status": label.get("status"),
        "label_useful_missed_winner": bool(label.get("is_top15") and label.get("status") != "bought" and float(label.get("opportunity_from_first_block_pct") or 0.0) > 0.0),
        "label_bad_candidate": not bool(label.get("is_top15")),
        "day_change_pct": label.get("day_change_pct"),
        "opportunity_from_first_block_pct": label.get("opportunity_from_first_block_pct"),
        "had_entry": bool(entry_rows),
        "first_entry_ts": entry_rows[0].get("ts") if entry_rows else None,
    }


def _features(bars: list[dict], idx: int, btc_bars: list[dict], ts_ms: int | None) -> dict:
    base = bars[idx]
    close = float(base["close"])
    recent = bars[max(0, idx - 8): idx]
    recent_vol = _avg([float(row.get("volume") or 0.0) for row in recent])
    recent_range = _avg([_range_pct(row) for row in recent])
    out = {
        "recent_volume_avg": round(recent_vol, 6),
        "recent_range_pct_avg": round(recent_range, 6),
    }
    btc_idx = _first_bar_at_or_after(btc_bars, ts_ms) if btc_bars and ts_ms is not None else None
    for name, step in HORIZON_BARS.items():
        end = idx + step
        if end < len(bars):
            window = bars[idx : end + 1]
            end_close = float(bars[end]["close"])
            ret = _ret_pct(end_close, close)
            out[f"ret_{name}_pct"] = round(ret, 6)
            out[f"max_high_{name}_pct"] = round(_ret_pct(max(float(row["high"]) for row in window), close), 6)
            out[f"min_low_{name}_pct"] = round(_ret_pct(min(float(row["low"]) for row in window), close), 6)
            out[f"volume_x_{name}"] = round((_avg([float(row.get("volume") or 0.0) for row in window[1:]]) / recent_vol) if recent_vol > 0 else 0.0, 6)
            out[f"range_x_{name}"] = round((_avg([_range_pct(row) for row in window[1:]]) / recent_range) if recent_range > 0 else 0.0, 6)
            if btc_idx is not None and btc_idx + step < len(btc_bars):
                btc_ret = _ret_pct(float(btc_bars[btc_idx + step]["close"]), float(btc_bars[btc_idx]["close"]))
                out[f"btc_ret_{name}_pct"] = round(btc_ret, 6)
                out[f"rel_ret_{name}_pct"] = round(ret - btc_ret, 6)
            else:
                out[f"btc_ret_{name}_pct"] = None
                out[f"rel_ret_{name}_pct"] = None
        else:
            for suffix in ("ret", "max_high", "min_low"):
                out[f"{suffix}_{name}_pct"] = None
            out[f"volume_x_{name}"] = None
            out[f"range_x_{name}"] = None
            out[f"btc_ret_{name}_pct"] = None
            out[f"rel_ret_{name}_pct"] = None
    return out


def _load_history(history_root: Path, symbol: str, cache: dict[str, list[dict]]) -> list[dict]:
    if symbol in cache:
        return cache[symbol]
    path = history_root / symbol / "15m.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and row.get("open_ts_ms") is not None:
                rows.append(row)
    rows.sort(key=lambda row: int(row["open_ts_ms"]))
    cache[symbol] = rows
    return rows


def _first_bar_at_or_after(bars: list[dict], ts_ms: int | None) -> int | None:
    if ts_ms is None or not bars:
        return None
    lo, hi = 0, len(bars)
    while lo < hi:
        mid = (lo + hi) // 2
        if int(bars[mid]["open_ts_ms"]) < ts_ms:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(bars) else None


def _ts_ms(ts) -> int | None:
    if not ts:
        return None
    try:
        return int(datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


def _ret_pct(value: float, base: float) -> float:
    return (value / base - 1.0) * 100.0 if base else 0.0


def _range_pct(row: dict) -> float:
    close = float(row.get("close") or 0.0)
    return ((float(row.get("high") or 0.0) - float(row.get("low") or 0.0)) / close * 100.0) if close else 0.0


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _audit(rows: list[dict], labels: dict, output: Path, max_hour: int, min_blocks: int) -> dict:
    labels_counter = Counter()
    reasons = Counter()
    coverage = Counter()
    for row in rows:
        labels_counter["top15"] += int(row["label_top15"])
        labels_counter["useful_missed_winner"] += int(row["label_useful_missed_winner"])
        labels_counter["bad_candidate"] += int(row["label_bad_candidate"])
        reasons[row["reason_code"]] += 1
        for key, value in row["features"].items():
            if value is not None:
                coverage[key] += 1
    n = len(rows) or 1
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "output": str(output),
        "rows": len(rows),
        "settings": {"max_hour": max_hour, "min_blocks": min_blocks, "allowed_reasons": sorted(ALLOWED_REASONS)},
        "label_counts": dict(labels_counter),
        "label_rates": {key: round(value / n, 6) for key, value in labels_counter.items()},
        "reason_counts": dict(reasons.most_common()),
        "feature_coverage": {key: round(value / n, 6) for key, value in sorted(coverage.items())},
        "top_positive_examples": sorted([row for row in rows if row["label_useful_missed_winner"]], key=lambda row: float(row.get("opportunity_from_first_block_pct") or 0.0), reverse=True)[:15],
        "top_bad_examples": sorted([row for row in rows if row["label_bad_candidate"]], key=lambda row: row["block_count_at_candidate"], reverse=True)[:15],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--files-dir", type=Path, default=FILES)
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--max-hour", type=int, default=12)
    parser.add_argument("--min-blocks", type=int, default=3)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.reports_dir, args.files_dir, args.history_root, args.output, args.audit_output, args.max_hour, args.min_blocks)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
