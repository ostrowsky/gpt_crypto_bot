from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp

import top_gainer_critic as critic


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / ".runtime" / "reports" / "top20_backfill"


async def _fetch_history(start_ms: int, end_ms: int) -> dict[str, list[list[Any]]]:
    headers = {"User-Agent": "Mozilla/5.0"}
    connector = aiohttp.TCPConnector(limit=25)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        symbols = await critic._fetch_exchange_symbols(session)
        sem = asyncio.Semaphore(20)

        async def one(symbol: str) -> tuple[str, list[list[Any]]]:
            async with sem:
                try:
                    rows = await critic._fetch_json(
                        session,
                        f"{critic.BINANCE_API}/api/v3/klines",
                        {
                            "symbol": symbol,
                            "interval": "1h",
                            "startTime": start_ms,
                            "endTime": end_ms - 1,
                            "limit": 1000,
                        },
                    )
                    return symbol, rows if isinstance(rows, list) else []
                except Exception:
                    return symbol, []

        return dict(await asyncio.gather(*(one(symbol) for symbol in symbols)))


def _group_events(tz: ZoneInfo, first_day: date, last_day: date) -> dict[tuple[str, str], dict[str, list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for event_file, default_source in critic._event_log_files():
        for raw in critic._iter_jsonl(event_file):
            symbol = str(raw.get("sym") or "")
            ts = critic._parse_utc_ts(raw.get("ts"))
            if not symbol or ts is None:
                continue
            local = ts.astimezone(tz)
            if not first_day <= local.date() <= last_day:
                continue
            bucket = grouped.setdefault(
                (local.date().isoformat(), symbol),
                {"entries": [], "exits": [], "blocked": [], "forwards": []},
            )
            rec = dict(raw)
            rec["_ts_local"] = local.strftime("%H:%M")
            rec["_log_source"] = default_source
            rec["_source"] = str(rec.get("source") or default_source)
            event = str(rec.get("event") or "")
            if event == "entry":
                bucket["entries"].append(rec)
            elif event == "exit":
                bucket["exits"].append(rec)
            elif event == "blocked":
                bucket["blocked"].append(rec)
            elif event == "forward":
                bucket["forwards"].append(rec)
    return grouped


def _day_performance(
    history: dict[str, list[list[Any]]],
    target_day: date,
    tz: ZoneInfo,
    watchlist: set[str],
    min_quote_volume: float,
) -> list[critic.DayPerformance]:
    start = datetime.combine(target_day, time.min, tzinfo=tz).astimezone(timezone.utc)
    end = datetime.combine(target_day + timedelta(days=1), time.min, tzinfo=tz).astimezone(timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    out: list[critic.DayPerformance] = []
    for symbol, rows in history.items():
        selected = [row for row in rows if start_ms <= int(row[0]) < end_ms]
        if not selected:
            continue
        quote_volume = sum(float(row[7]) for row in selected)
        if quote_volume < min_quote_volume:
            continue
        day_open = float(selected[0][1])
        day_close = float(selected[-1][4])
        if day_open <= 0:
            continue
        out.append(
            critic.DayPerformance(
                symbol=symbol,
                day_open=day_open,
                day_close=day_close,
                day_high=max(float(row[2]) for row in selected),
                day_low=min(float(row[3]) for row in selected),
                day_change_pct=(day_close / day_open - 1.0) * 100.0,
                quote_volume_24h=quote_volume,
                in_watchlist=symbol in watchlist,
            )
        )
    out.sort(key=lambda row: row.day_change_pct, reverse=True)
    return out


def build(days: int, top_n: int, min_quote_volume: float, output_dir: Path) -> dict[str, Any]:
    tz = ZoneInfo("Europe/Budapest")
    last_day = datetime.now(tz).date() - timedelta(days=1)
    first_day = last_day - timedelta(days=days - 1)
    start = datetime.combine(first_day, time.min, tzinfo=tz).astimezone(timezone.utc)
    end = datetime.combine(last_day + timedelta(days=1), time.min, tzinfo=tz).astimezone(timezone.utc)
    history = asyncio.run(_fetch_history(int(start.timestamp() * 1000), int(end.timestamp() * 1000)))
    watchlist = critic._load_watchlist()
    events = _group_events(tz, first_day, last_day)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    current = first_day
    while current <= last_day:
        perf = _day_performance(history, current, tz, watchlist, min_quote_volume)
        top = perf[:top_n]
        labels = []
        end_local = datetime.combine(current, time.max, tzinfo=tz)
        for row in top:
            if not row.in_watchlist:
                continue
            labels.append(
                critic.summarize_top_gainer(
                    row,
                    events.get((current.isoformat(), row.symbol), {}),
                    end_local=end_local,
                    tz=tz,
                )
            )
        payload = {
            "target_day_local": current.isoformat(),
            "phase": "final",
            "settings": {"timezone": "Europe/Budapest", "top_n": top_n, "min_quote_volume_24h": min_quote_volume},
            "summary": {"exchange_top_count": len(top), "watchlist_top_count": len(labels)},
            "watchlist_top_gainers": labels,
        }
        path = output_dir / f"top_gainer_critic_{current.isoformat()}_final.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1
        current += timedelta(days=1)
    return {
        "first_day": first_day.isoformat(),
        "last_day": last_day.isoformat(),
        "days": written,
        "symbols_loaded": sum(bool(rows) for rows in history.values()),
        "symbols_requested": len(history),
        "top_n": top_n,
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill daily Binance top-N critic labels in one historical fetch")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-quote-volume", type=float, default=1_000_000.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.days, args.top_n, args.min_quote_volume, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
