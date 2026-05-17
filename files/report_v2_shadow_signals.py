from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVENTS = ROOT / "v2_shadow_events.jsonl"


def build(hours: int = 24) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = []
    if EVENTS.exists():
        for line in EVENTS.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
                ts = datetime.fromisoformat(str(row["ts"]).replace("Z", "+00:00"))
            except Exception:
                continue
            if ts >= cutoff:
                rows.append(row)
    bootstrap_rows = [row for row in rows if row.get("bootstrap", row.get("previous_state") is None)]
    material_rows = [row for row in rows if not row.get("bootstrap", row.get("previous_state") is None)]
    return {
        "hours": hours,
        "events": len(material_rows),
        "bootstrap_events": len(bootstrap_rows),
        "states": Counter(row.get("state") for row in material_rows),
        "actions": Counter(row.get("action") for row in material_rows),
        "latest": material_rows[-30:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.hours)
    payload["states"] = dict(payload["states"])
    payload["actions"] = dict(payload["actions"])
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"v2 shadow: events={payload['events']} states={payload['states']} actions={payload['actions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
