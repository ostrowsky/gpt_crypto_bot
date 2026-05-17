from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRACE = ROOT / "v2_shadow_decisions.jsonl"


def _local_day_utc(raw_ms: int) -> str:
    return datetime.fromtimestamp(raw_ms / 1000, tz=timezone.utc).date().isoformat()


def explain(symbol: str, timeframe: str | None = None, day: str | None = None) -> dict:
    rows = []
    if TRACE.exists():
        for line in TRACE.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if str(row.get("sym")) != symbol:
                continue
            if timeframe and str(row.get("tf")) != timeframe:
                continue
            if day and _local_day_utc(int(row.get("bar_ts") or 0)) != day:
                continue
            rows.append(row)
    rows.sort(key=lambda row: (int(row.get("bar_ts") or 0), str(row.get("observed_at") or "")))
    material = [
        row
        for row in rows
        if row.get("material_transition")
        and not row.get("bootstrap", row.get("previous_state") is None)
    ]
    latest = rows[-1] if rows else None
    latest_material = material[-1] if material else None
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "day_utc": day,
        "rows": len(rows),
        "material_signals": len(material),
        "bootstrap_rows": sum(1 for row in rows if row.get("bootstrap", row.get("previous_state") is None)),
        "latest_decision": latest,
        "latest_material_signal": latest_material,
        "why_no_signal": None
        if latest_material is not None
        else (None if latest is None else latest.get("reason")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe")
    parser.add_argument("--date")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = explain(args.symbol.upper(), args.timeframe, args.date)
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
