from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "files" / "bot_events.jsonl"
REPORTS = ROOT / ".runtime" / "reports"


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)


def _day(ts: str) -> str:
    return _parse_ts(ts).strftime("%Y-%m-%d")


def _load_events() -> list[dict[str, Any]]:
    if not EVENTS.exists():
        return []
    out = []
    for line in EVENTS.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _top_symbols(day: str) -> set[str]:
    path = REPORTS / f"top_gainer_critic_{day}_final.json"
    if not path.exists():
        return set()
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    return {row["symbol"] for row in report.get("watchlist_top_gainers", [])}


def build(days: int = 14) -> dict[str, Any]:
    events = _load_events()
    scout = [
        e for e in events
        if e.get("event") == "scout_shadow"
        and str(e.get("scout_profile") or "").startswith("wake_up_")
    ]
    entries = [e for e in events if e.get("event") == "entry"]
    seen_days = sorted({_day(e["ts"]) for e in scout if e.get("ts")})[-days:]
    rows = []
    for day in seen_days:
        wakes = [e for e in scout if _day(e["ts"]) == day]
        syms = {row["sym"] for row in wakes}
        day_entries = [e for e in entries if _day(e["ts"]) == day]
        bought = {e["sym"] for e in day_entries if e["sym"] in syms}
        top = _top_symbols(day)
        rows.append({
            "day": day,
            "wakeups": len(syms),
            "admitted": len(syms),
            "bought_after_wakeup": len(bought),
            "buy_conversion_pct": round(len(bought) / len(syms) * 100.0, 2) if syms else 0.0,
            "final_top_movers_woken": len(syms & top),
            "final_top_mover_conversion_pct": round(len(syms & top) / len(syms) * 100.0, 2) if syms else 0.0,
            "final_top_movers_bought_after_wakeup": len(bought & top),
        })
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "days": rows,
    }


def render(report: dict[str, Any]) -> str:
    lines = ["Wake-up scout funnel", f"generated_at_utc: {report['generated_at_utc']}"]
    for row in report["days"]:
        lines.append(
            f"{row['day']}: wakeups {row['wakeups']} | buys {row['bought_after_wakeup']} "
            f"({row['buy_conversion_pct']}%) | top movers woken {row['final_top_movers_woken']} "
            f"({row['final_top_mover_conversion_pct']}%) | top movers bought {row['final_top_movers_bought_after_wakeup']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = build()
    (REPORTS / "wakeup_funnel_latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "wakeup_funnel_latest.txt").write_text(render(report), encoding="utf-8")


if __name__ == "__main__":
    main()
