from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
CRITIC_FILE = ROOT / "critic_dataset.jsonl"
EVENT_FILE = ROOT / "bot_events.jsonl"
MODE_RE = re.compile(r"\bfor\s+(15m|1h)\s+([a-z0-9_]+)\b", re.I)
SCORE_FLOOR_RE = re.compile(r"\bscore\s+-?\d+(?:\.\d+)?\s*<\s*(-?\d+(?:\.\d+)?)", re.I)
TF_SECONDS = {"15m": 15 * 60, "1h": 60 * 60}


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iter_jsonl_reverse(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("rb") as handle:
        position = handle.seek(0, 2)
        carry = b""
        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            carry = handle.read(read_size) + carry
            lines = carry.split(b"\n")
            carry = lines.pop(0)
            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(rec, dict):
                    yield rec
        if carry.strip():
            try:
                rec = json.loads(carry)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return
            if isinstance(rec, dict):
                yield rec


def _iter_window_jsonl(
    path: Path,
    *,
    start: datetime,
    end: datetime,
    ts_fields: tuple[str, ...],
) -> Iterable[dict[str, Any]]:
    """Read an append-only chronological journal backwards and stop at start."""
    for rec in _iter_jsonl_reverse(path):
        ts_raw = next((str(rec.get(field) or "") for field in ts_fields if rec.get(field)), "")
        if not ts_raw:
            continue
        try:
            ts = _parse_ts(ts_raw)
        except ValueError:
            continue
        if ts > end:
            continue
        if ts < start:
            break
        yield rec


def _mode_from_reason(reason: str, fallback: Any) -> str:
    match = MODE_RE.search(reason)
    return match.group(2).lower() if match else str(fallback or "")


def _normalise_critic(rec: dict[str, Any]) -> dict[str, Any] | None:
    decision = rec.get("decision") or {}
    if str(decision.get("action") or "").lower() != "blocked":
        return None
    reason = str(decision.get("reason") or "")
    reason_code = str(decision.get("reason_code") or "blocked_unknown")
    score_floor = decision.get("score_floor")
    if reason_code == "top_gainer_score_gate":
        match = SCORE_FLOOR_RE.search(reason)
        if match:
            score_floor = float(match.group(1))
    return {
        "ts": str(rec.get("ts_signal") or ""),
        "tf": str(rec.get("tf") or ""),
        "signal_type": _mode_from_reason(reason, rec.get("signal_type")),
        "reason_code": reason_code,
        "reason": reason,
        "stage": str(decision.get("stage") or ""),
        "candidate_score": decision.get("candidate_score"),
        "score_floor": score_floor,
        "signal_flags": decision.get("signal_flags") or {},
        "source": "critic_dataset",
    }


def _normalise_event(rec: dict[str, Any]) -> dict[str, Any] | None:
    if str(rec.get("event") or "").lower() != "blocked":
        return None
    reason = str(rec.get("reason") or "")
    raw_signal_type = rec.get("signal_type")
    reason_code = str(rec.get("reason_code") or "blocked_unknown")
    score_floor = rec.get("score_floor")
    if reason_code == "top_gainer_score_gate":
        match = SCORE_FLOOR_RE.search(reason)
        if match:
            score_floor = float(match.group(1))
    return {
        "ts": str(rec.get("ts") or ""),
        "tf": str(rec.get("tf") or ""),
        "signal_type": _mode_from_reason(reason, raw_signal_type),
        "reason_code": reason_code,
        "reason": reason,
        "stage": str(rec.get("stage") or rec.get("gate") or ""),
        "candidate_score": rec.get("candidate_score"),
        "score_floor": score_floor,
        "signal_flags": rec.get("signal_flags") or {},
        "source": "bot_events",
    }


def _signature(row: dict[str, Any]) -> tuple[Any, ...]:
    def number(value: Any) -> float | None:
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return None

    return (
        row.get("tf"),
        row.get("signal_type"),
        row.get("reason_code"),
        row.get("reason"),
        number(row.get("candidate_score")),
        number(row.get("score_floor")),
    )


def _deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    latest_by_signature: dict[tuple[Any, ...], int] = {}
    for row in sorted(rows, key=lambda item: item["_parsed_ts"]):
        signature = _signature(row)
        previous_index = latest_by_signature.get(signature)
        if previous_index is not None:
            previous = unique[previous_index]
            interval = TF_SECONDS.get(str(row.get("tf") or ""), 5 * 60)
            elapsed = (row["_parsed_ts"] - previous["_last_parsed_ts"]).total_seconds()
            if 0 <= elapsed <= interval:
                previous["last_ts"] = row["ts"]
                previous["_last_parsed_ts"] = row["_parsed_ts"]
                previous["repeat_count"] += 1
                previous["sources"] = sorted(set(previous["sources"]) | {row["source"]})
                if row["source"] == "critic_dataset":
                    for field in ("stage", "candidate_score", "score_floor", "signal_flags"):
                        if row.get(field) not in (None, "", {}):
                            previous[field] = row[field]
                continue
        clean = dict(row)
        clean["first_ts"] = clean["ts"]
        clean["last_ts"] = clean["ts"]
        clean["repeat_count"] = 1
        clean["sources"] = [clean.pop("source")]
        clean["_last_parsed_ts"] = clean["_parsed_ts"]
        latest_by_signature[signature] = len(unique)
        unique.append(clean)
    for row in unique:
        row.pop("_parsed_ts", None)
        row.pop("_last_parsed_ts", None)
        row.pop("ts", None)
    return unique


def build_report(
    symbol: str,
    *,
    days: int = 7,
    critic_file: Path = CRITIC_FILE,
    event_file: Path = EVENT_FILE,
    now: datetime | None = None,
) -> dict[str, Any]:
    end = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    start = end - timedelta(days=days)
    rows: list[dict[str, Any]] = []
    sources = (
        (critic_file, _normalise_critic, ("ts_signal", "ts")),
        (event_file, _normalise_event, ("ts", "ts_signal")),
    )
    for path, normalise, ts_fields in sources:
        for rec in _iter_window_jsonl(path, start=start, end=end, ts_fields=ts_fields):
            if str(rec.get("sym") or rec.get("symbol") or "").upper() != symbol.upper():
                continue
            row = normalise(rec)
            if not row or not row.get("ts"):
                continue
            try:
                ts = _parse_ts(str(row["ts"]))
            except ValueError:
                continue
            if start <= ts <= end:
                row["_parsed_ts"] = ts
                rows.append(row)
    unique_rows = _deduplicate(rows)
    counts = Counter(str(x["reason_code"]) for x in unique_rows)
    return {
        "symbol": symbol.upper(),
        "window": {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
        },
        "blocked_events": len(unique_rows),
        "blocked_events_raw": len(rows),
        "reason_counts": dict(counts.most_common()),
        "sources": {
            "critic_dataset": str(critic_file),
            "bot_events": str(event_file),
        },
        "trace": unique_rows[-50:],
    }


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"Why-no-signal trace for {report['symbol']}",
        f"window: {report['window']['start']} .. {report['window']['end']}",
        f"blocked events: {report['blocked_events']} unique / {report.get('blocked_events_raw', report['blocked_events'])} raw",
        "reasons: "
        + (
            ", ".join(f"{k}={v}" for k, v in report["reason_counts"].items())
            if report["reason_counts"]
            else "none"
        ),
    ]
    if report["trace"]:
        lines.extend(["", "latest blocker chain:"])
        for row in report["trace"][-15:]:
            lines.append(
                f"- {row['first_ts']}..{row['last_ts']} {row['tf']} {row['signal_type']} "
                f"{row['reason_code']} stage={row['stage']} "
                f"score={row['candidate_score']} floor={row['score_floor']} "
                f"repeats={row['repeat_count']} sources={','.join(row['sources'])} "
                f"reason={row['reason']}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only blocker-chain report from critic and runtime event journals.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(args.symbol, days=args.days)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
