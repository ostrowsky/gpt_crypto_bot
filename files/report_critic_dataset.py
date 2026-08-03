from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable

import critic_dataset


def _iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def build_report(now: datetime | None = None) -> Dict[str, Any]:
    now = now or datetime.utcnow()
    path = critic_dataset.CRITIC_FILE
    last_24h_cutoff = now - timedelta(hours=24)

    def _parse_ts(rec: Dict[str, Any]) -> datetime | None:
        raw = rec.get("ts_signal")
        if not raw:
            return None
        try:
            return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return None

    rows_total = 0
    rows_last_24h = 0
    labeled = 0
    taken = 0
    outcomes = 0
    actions: Counter[str] = Counter()
    signals: Counter[str] = Counter()
    for rec in _iter_rows(path):
        rows_total += 1
        if (_parse_ts(rec) or datetime.min) >= last_24h_cutoff:
            rows_last_24h += 1
        actions[str(rec.get("decision", {}).get("action", ""))] += 1
        signals[str(rec.get("signal_type", ""))] += 1
        labels = rec.get("labels", {})
        if labels.get("ret_3") is not None:
            labeled += 1
        if bool(labels.get("trade_taken")):
            taken += 1
        if labels.get("trade_exit_pnl") is not None:
            outcomes += 1

    return {
        "path": str(path),
        "exists": path.exists(),
        "rows_total": rows_total,
        "rows_last_24h": rows_last_24h,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "last_write_time": path.stat().st_mtime if path.exists() else 0,
        "labeled_rows": labeled,
        "trade_taken_rows": taken,
        "trade_outcome_rows": outcomes,
        "actions": dict(actions),
        "signal_types": dict(signals),
    }


def render_text(report: Dict[str, Any]) -> str:
    if not report["exists"]:
        return "Critic dataset: file not created yet"
    lines = [
        f"Critic dataset: {report['rows_total']} rows",
        f"Last 24h: {report['rows_last_24h']}",
        f"Labeled: {report['labeled_rows']}",
        f"Trade taken: {report['trade_taken_rows']}",
        f"Trade outcomes: {report['trade_outcome_rows']}",
        f"Size: {report['size_bytes']} bytes",
    ]
    if report["actions"]:
        actions = ", ".join(f"{k}={v}" for k, v in sorted(report["actions"].items()))
        lines.append(f"Actions: {actions}")
    if report["signal_types"]:
        sigs = ", ".join(f"{k}={v}" for k, v in sorted(report["signal_types"].items()))
        lines.append(f"Signal types: {sigs}")
    return "\n".join(lines)


def main() -> int:
    print(render_text(build_report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
