from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / ".runtime" / "reports"


def _val(summary: dict, key: str):
    return (summary.get(key) or {}).get("median")


def build(days: int) -> dict:
    rows = []
    for path in sorted(REPORTS.glob("signal_quality_*_final.json"))[-days:]:
        match = re.search(r"(2026-\d\d-\d\d)", path.name)
        if not match:
            continue
        summary = json.loads(path.read_text(encoding="utf-8-sig")).get("summary", {})
        rows.append(
            {
                "day": match.group(1),
                "closed_trades": summary.get("closed_trades"),
                "early_exits": summary.get("early_exits"),
                "late_exits": summary.get("late_exits"),
                "exit_efficiency_median": _val(summary, "exit_efficiency"),
                "giveback_pct_median": _val(summary, "giveback_pct"),
                "realized_capture_ratio_median": _val(summary, "realized_capture_ratio"),
            }
        )
    def med(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(median(vals), 4) if vals else None
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "days_loaded": len(rows),
        "rows": rows,
        "summary": {
            "closed_trades_total": sum(int(r.get("closed_trades") or 0) for r in rows),
            "early_exits_total": sum(int(r.get("early_exits") or 0) for r in rows),
            "late_exits_total": sum(int(r.get("late_exits") or 0) for r in rows),
            "exit_efficiency_median": med("exit_efficiency_median"),
            "giveback_pct_median": med("giveback_pct_median"),
            "realized_capture_ratio_median": med("realized_capture_ratio_median"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = build(args.days)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
