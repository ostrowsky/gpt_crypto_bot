from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from blocking import normalize_blocked_reason

ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
REPORTS = ROOT / ".runtime" / "reports"
TZ = ZoneInfo("Europe/Budapest")


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            yield row


def _local_day(row: dict) -> str | None:
    raw = row.get("ts")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(TZ).date().isoformat()
    except ValueError:
        return None


def _hhmm(row: dict | None) -> str | None:
    if not row:
        return None
    raw = row.get("ts")
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(TZ).strftime("%H:%M")
    except Exception:
        return None


def _dt(row: dict | None):
    if not row:
        return None
    try:
        return datetime.fromisoformat(str(row.get("ts")).replace("Z", "+00:00"))
    except Exception:
        return None


def _reason_code(row: dict | None) -> str | None:
    if not row:
        return None
    return str(row.get("reason_code") or normalize_blocked_reason(str(row.get("signal_type") or ""), str(row.get("reason") or "")))


def build(day: str) -> dict:
    critic_path = REPORTS / f"top_gainer_critic_{day}_final.json"
    critic = json.loads(critic_path.read_text(encoding="utf-8-sig"))
    winners = critic.get("watchlist_top_gainers") or []
    symbols = {str(row.get("symbol")) for row in winners}
    buckets = {sym: [] for sym in symbols}
    for path in (FILES / "bot_events.jsonl", FILES / "agent_events.jsonl"):
        for row in _iter_jsonl(path):
            sym = str(row.get("sym") or row.get("symbol") or "")
            if sym in buckets and _local_day(row) == day:
                buckets[sym].append(row)
    out = []
    for winner in winners:
        sym = str(winner.get("symbol"))
        rows = sorted(buckets.get(sym, []), key=lambda x: str(x.get("ts") or ""))
        structural = next((r for r in rows if r.get("event") == "scout_shadow" and r.get("scout_profile") != "wake_up_1m_light15_v1"), None)
        wakeup = next((r for r in rows if r.get("event") == "scout_shadow" and r.get("scout_profile") == "wake_up_1m_light15_v1"), None)
        block = next((r for r in rows if r.get("event") == "blocked"), None)
        buy = next((r for r in rows if r.get("event") == "entry"), None)
        wake_dt = _dt(wakeup)
        block_dt = _dt(block)
        if buy:
            first_loss_point = "bought"
        elif block and wakeup and wake_dt and block_dt and block_dt >= wake_dt:
            first_loss_point = "blocked_after_wakeup"
        elif block and wakeup:
            first_loss_point = "blocked_before_wakeup"
        elif block:
            first_loss_point = "blocked_without_wakeup"
        elif wakeup:
            first_loss_point = "wakeup_no_buy"
        else:
            first_loss_point = "no_logged_funnel_event"
        out.append(
            {
                "symbol": sym,
                "final_top15": True,
                "day_change_pct": winner.get("day_change_pct"),
                "first_structural_alert": _hhmm(structural),
                "first_wakeup": _hhmm(wakeup),
                "first_block": _hhmm(block),
                "first_block_reason_code": _reason_code(block),
                "first_buy": _hhmm(buy),
                "final_outcome": "bought" if buy else "missed",
                "first_loss_point": first_loss_point,
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "target_day_local": day,
        "source": str(critic_path),
        "rows": out,
        "summary": {
            "top15_count": len(out),
            "bought": sum(1 for r in out if r["final_outcome"] == "bought"),
            "missed": sum(1 for r in out if r["final_outcome"] == "missed"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = build(args.date)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
