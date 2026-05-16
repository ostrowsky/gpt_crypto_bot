from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
CRITIC_FILE = ROOT / "critic_dataset.jsonl"


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            yield rec


def build_report(symbol: str, *, days: int = 7) -> dict[str, Any]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    rows: list[dict[str, Any]] = []
    for rec in _iter_jsonl(CRITIC_FILE):
        if str(rec.get("sym") or "").upper() != symbol.upper():
            continue
        ts_raw = str(rec.get("ts_signal") or "")
        if not ts_raw:
            continue
        ts = _parse_ts(ts_raw)
        if not (start <= ts <= end):
            continue
        decision = rec.get("decision") or {}
        if str(decision.get("action") or "") != "blocked":
            continue
        rows.append(
            {
                "ts": ts_raw,
                "tf": rec.get("tf"),
                "signal_type": rec.get("signal_type"),
                "reason_code": decision.get("reason_code") or "blocked_unknown",
                "reason": decision.get("reason") or "",
                "stage": decision.get("stage") or "",
                "candidate_score": decision.get("candidate_score"),
                "score_floor": decision.get("score_floor"),
                "signal_flags": decision.get("signal_flags") or {},
            }
        )
    rows.sort(key=lambda x: x["ts"])
    counts = Counter(str(x["reason_code"]) for x in rows)
    return {
        "symbol": symbol.upper(),
        "window": {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
        },
        "blocked_events": len(rows),
        "reason_counts": dict(counts.most_common()),
        "trace": rows[-50:],
    }


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"Why-no-signal trace for {report['symbol']}",
        f"window: {report['window']['start']} .. {report['window']['end']}",
        f"blocked events: {report['blocked_events']}",
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
                f"- {row['ts']} {row['tf']} {row['signal_type']} "
                f"{row['reason_code']} stage={row['stage']} "
                f"score={row['candidate_score']} floor={row['score_floor']} "
                f"reason={row['reason']}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only blocker-chain report from critic_dataset.jsonl.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(args.symbol, days=args.days)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
