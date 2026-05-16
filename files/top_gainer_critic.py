from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import aiohttp

import critic_dataset
from blocking import normalize_blocked_reason


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
RUNTIME_DIR = WORKSPACE_ROOT / ".runtime"
REPORT_DIR = RUNTIME_DIR / "reports"
HISTORY_FILE = REPORT_DIR / "top_gainer_critic_history.jsonl"

BINANCE_API = "https://api.binance.com"
WATCHLIST_FILE = ROOT / "watchlist.json"
BOT_EVENTS_FILE = ROOT / "bot_events.jsonl"
AGENT_EVENTS_FILE = ROOT / "agent_events.jsonl"

DEFAULT_TOP_N = 15
DEFAULT_MIN_QUOTE_VOLUME = 1_000_000.0
DEFAULT_TZ = "Europe/Budapest"

_STABLE_OR_NON_TARGET = {
    "USDCUSDT",
    "BUSDUSDT",
    "FDUSDUSDT",
    "TUSDUSDT",
    "DAIUSDT",
    "USDPUSDT",
    "USDSUSDT",
    "EURUSDT",
    "GBPUSDT",
    "TRYUSDT",
    "BRLUSDT",
}


@dataclass
class DayPerformance:
    symbol: str
    day_open: float
    day_close: float
    day_high: float
    day_low: float
    day_change_pct: float
    quote_volume_24h: float
    in_watchlist: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Critic report: compare bot BUY decisions with top gainers of the day or midday window."
    )
    parser.add_argument(
        "--phase",
        choices=("midday", "final"),
        required=True,
        help="midday = 00:00-12:00 local window, final = 00:00-24:00 completed local day",
    )
    parser.add_argument(
        "--date",
        default="",
        help="Local date YYYY-MM-DD. For final reports this is the audited local day.",
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-quote-volume", type=float, default=DEFAULT_MIN_QUOTE_VOLUME)
    parser.add_argument("--timezone", default=DEFAULT_TZ)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def _parse_utc_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _event_log_files() -> tuple[tuple[Path, str], ...]:
    files = [(BOT_EVENTS_FILE, "bot")]
    if AGENT_EVENTS_FILE != BOT_EVENTS_FILE:
        files.append((AGENT_EVENTS_FILE, "agent"))
    return tuple(files)


def _event_source(rec: dict[str, Any]) -> str:
    source = str(
        rec.get("_source")
        or rec.get("source")
        or rec.get("_log_source")
        or "bot"
    ).strip()
    return source or "bot"


def _blocked_reason_code(rec: dict[str, Any]) -> str:
    explicit = str(rec.get("reason_code") or "").strip()
    if explicit:
        return explicit
    signal_type = str(rec.get("signal_type") or "").strip().lower()
    reason = str(rec.get("reason") or "").strip().lower()
    return normalize_blocked_reason(signal_type, reason)


def _counter_payload(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


def _compact_count_map(counts: Any, limit: int = 5) -> str:
    if not isinstance(counts, dict) or not counts:
        return "none"
    items = sorted(
        ((str(key), int(value or 0)) for key, value in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return ", ".join(f"{key}: {value}" for key, value in items[:limit])


def _load_watchlist() -> set[str]:
    try:
        payload = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return set()
    return {str(x) for x in payload if isinstance(x, str)}


def _local_window(
    target_day: date,
    phase: str,
    tz: ZoneInfo,
    *,
    cutoff_hour: int | None = None,
) -> tuple[datetime, datetime]:
    start_local = datetime.combine(target_day, time.min, tzinfo=tz)
    if cutoff_hour is not None:
        cutoff_hour = int(cutoff_hour)
        if not 0 <= cutoff_hour <= 23:
            raise ValueError(f"cutoff_hour must be in [0, 23], got {cutoff_hour!r}")
        end_local = datetime.combine(target_day, time(hour=cutoff_hour), tzinfo=tz)
    elif phase == "midday":
        end_local = datetime.combine(target_day, time(hour=12), tzinfo=tz)
    else:
        end_local = datetime.combine(target_day, time.max, tzinfo=tz)
    return start_local, end_local


def _as_utc_ms(dt_local: datetime) -> int:
    return int(dt_local.astimezone(timezone.utc).timestamp() * 1000)


def _is_target_symbol(symbol: str) -> bool:
    if not symbol.endswith("USDT"):
        return False
    if symbol in _STABLE_OR_NON_TARGET:
        return False
    for suffix in ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"):
        if symbol.endswith(suffix):
            return False
    return True


async def _fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    timeout = aiohttp.ClientTimeout(total=30)
    async with session.get(url, params=params, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.json()


async def _fetch_exchange_symbols(session: aiohttp.ClientSession) -> list[str]:
    payload = await _fetch_json(session, f"{BINANCE_API}/api/v3/exchangeInfo")
    out: list[str] = []
    for item in payload.get("symbols", []):
        sym = str(item.get("symbol", ""))
        if item.get("status") != "TRADING":
            continue
        if item.get("quoteAsset") != "USDT":
            continue
        if not bool(item.get("isSpotTradingAllowed", True)):
            continue
        if _is_target_symbol(sym):
            out.append(sym)
    return out


async def _fetch_all_tickers(session: aiohttp.ClientSession) -> dict[str, dict[str, Any]]:
    payload = await _fetch_json(session, f"{BINANCE_API}/api/v3/ticker/24hr", {"type": "FULL"})
    return {
        str(item.get("symbol", "")): item
        for item in payload
        if isinstance(item, dict)
    }


async def _fetch_symbol_window(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    in_watchlist: bool,
    quote_volume_24h: float,
) -> DayPerformance | None:
    payload = await _fetch_json(
        session,
        f"{BINANCE_API}/api/v3/klines",
        {
            "symbol": symbol,
            "interval": "1h",
            "startTime": start_ms,
            "endTime": end_ms - 1,
            "limit": 32,
        },
    )
    if not payload:
        return None
    opens = [float(row[1]) for row in payload]
    highs = [float(row[2]) for row in payload]
    lows = [float(row[3]) for row in payload]
    closes = [float(row[4]) for row in payload]
    day_open = float(opens[0])
    day_close = float(closes[-1])
    if day_open <= 0:
        return None
    return DayPerformance(
        symbol=symbol,
        day_open=day_open,
        day_close=day_close,
        day_high=max(highs),
        day_low=min(lows),
        day_change_pct=(day_close / day_open - 1.0) * 100.0,
        quote_volume_24h=float(quote_volume_24h),
        in_watchlist=bool(in_watchlist),
    )


async def fetch_day_performance(
    *,
    target_day: date,
    phase: str,
    timezone_name: str,
    min_quote_volume: float,
    cutoff_hour: int | None = None,
) -> list[DayPerformance]:
    tz = ZoneInfo(timezone_name)
    start_local, end_local = _local_window(target_day, phase, tz, cutoff_hour=cutoff_hour)
    start_ms = _as_utc_ms(start_local)
    end_ms = _as_utc_ms(end_local)
    watchlist = _load_watchlist()

    headers = {"User-Agent": "Mozilla/5.0"}
    connector = aiohttp.TCPConnector(limit=25)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        exchange_symbols = await _fetch_exchange_symbols(session)
        ticker_map = await _fetch_all_tickers(session)

        sem = asyncio.Semaphore(20)

        async def _one(symbol: str) -> DayPerformance | None:
            ticker = ticker_map.get(symbol)
            if not ticker:
                return None
            quote_volume = float(ticker.get("quoteVolume") or 0.0)
            if quote_volume < min_quote_volume:
                return None
            async with sem:
                try:
                    return await _fetch_symbol_window(
                        session,
                        symbol,
                        start_ms,
                        end_ms,
                        in_watchlist=symbol in watchlist,
                        quote_volume_24h=quote_volume,
                    )
                except Exception:
                    return None

        results = await asyncio.gather(*(_one(sym) for sym in exchange_symbols))

    perf = [item for item in results if item is not None]
    perf.sort(key=lambda x: x.day_change_pct, reverse=True)
    return perf


def _load_day_events(start_local: datetime, end_local: datetime, tz: ZoneInfo) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for event_file, default_source in _event_log_files():
        for rec in _iter_jsonl(event_file):
            sym = str(rec.get("sym", ""))
            if not sym:
                continue
            ts = _parse_utc_ts(rec.get("ts"))
            if not ts:
                continue
            ts_local = ts.astimezone(tz)
            if not (start_local <= ts_local <= end_local):
                continue
            bucket = rows.setdefault(
                sym,
                {"entries": [], "exits": [], "blocked": [], "forwards": []},
            )
            rec = dict(rec)
            rec["_ts_local"] = ts_local.strftime("%H:%M")
            rec["_log_source"] = default_source
            rec["_source"] = str(rec.get("source") or default_source)
            event = str(rec.get("event", ""))
            if event == "entry":
                bucket["entries"].append(rec)
            elif event == "exit":
                bucket["exits"].append(rec)
            elif event == "blocked":
                bucket["blocked"].append(rec)
            elif event == "forward":
                bucket["forwards"].append(rec)
    return rows


def _capture_ratio(day_open: float, day_close: float, entry_price: float) -> float | None:
    move = day_close - day_open
    if move <= 0:
        return None
    ratio = (day_close - entry_price) / move
    return max(0.0, min(1.5, ratio))


def _event_local_dt(rec: dict[str, Any] | None, tz: ZoneInfo) -> datetime | None:
    if not rec:
        return None
    ts = _parse_utc_ts(rec.get("ts"))
    if not ts:
        return None
    return ts.astimezone(tz)


def _event_price(rec: dict[str, Any] | None, field: str) -> float | None:
    if not rec:
        return None
    try:
        value = float(rec.get(field, 0.0))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _exit_quality_metrics(
    *,
    entry_price: float | None,
    day_high: float,
    last_exit: dict[str, Any] | None,
) -> tuple[float | None, float | None]:
    if not entry_price or entry_price <= 0 or not last_exit:
        return None, None
    try:
        exit_pnl = float(last_exit.get("pnl_pct"))
    except (TypeError, ValueError):
        exit_price = _event_price(last_exit, "exit_price")
        exit_pnl = ((exit_price / entry_price) - 1.0) * 100.0 if exit_price else 0.0
    max_favorable = (day_high / entry_price - 1.0) * 100.0
    if max_favorable <= 0:
        return None, None
    exit_efficiency = exit_pnl / max_favorable
    giveback_pct = max(0.0, max_favorable - exit_pnl)
    return exit_efficiency, giveback_pct


def _status_for_symbol(in_watchlist: bool, events: dict[str, list[dict[str, Any]]]) -> tuple[str, str | None]:
    if not in_watchlist:
        return "not_in_watchlist", None
    entries = events.get("entries", [])
    blocked = events.get("blocked", [])
    if entries:
        return "bought", None
    if blocked:
        reasons = Counter(str(x.get("reason", "")) for x in blocked)
        top_reason = reasons.most_common(1)[0][0] if reasons else None
        reason_codes = Counter(_blocked_reason_code(x) for x in blocked)
        top_reason_code = reason_codes.most_common(1)[0][0] if reason_codes else None
        if top_reason_code == "portfolio_full":
            return "blocked_portfolio", top_reason
        return "blocked_rule", top_reason
    return "no_signal", None


def summarize_top_gainer(
    perf: DayPerformance,
    events: dict[str, list[dict[str, Any]]],
    *,
    end_local: datetime | None = None,
    tz: ZoneInfo | None = None,
) -> dict[str, Any]:
    tz = tz or ZoneInfo(DEFAULT_TZ)
    entries = sorted(events.get("entries", []), key=lambda x: x.get("ts", ""))
    exits = sorted(events.get("exits", []), key=lambda x: x.get("ts", ""))
    blocked = sorted(events.get("blocked", []), key=lambda x: x.get("ts", ""))
    first_entry = entries[0] if entries else None
    last_exit = exits[-1] if exits else None
    first_block = blocked[0] if blocked else None
    last_block = blocked[-1] if blocked else None
    first_cooldown_block = next((item for item in blocked if _blocked_reason_code(item) == "symbol_cooldown"), None)
    status, reason = _status_for_symbol(perf.in_watchlist, events)

    first_entry_price = float(first_entry.get("price", 0.0)) if first_entry else None
    first_block_price = _event_price(first_block, "price")
    first_cooldown_block_price = _event_price(first_cooldown_block, "price")
    capture_ratio = (
        _capture_ratio(perf.day_open, perf.day_close, first_entry_price)
        if first_entry_price
        else None
    )
    first_entry_dt = _event_local_dt(first_entry, tz)
    lead_time_to_final_top_min = (
        max(0, int((end_local - first_entry_dt).total_seconds() // 60))
        if end_local is not None and first_entry_dt is not None
        else None
    )
    exit_efficiency, giveback_pct = _exit_quality_metrics(
        entry_price=first_entry_price,
        day_high=perf.day_high,
        last_exit=last_exit,
    )
    opportunity_from_entry = (
        (perf.day_close / first_entry_price - 1.0) * 100.0
        if first_entry_price and first_entry_price > 0
        else None
    )
    opportunity_from_first_block = (
        (perf.day_close / first_block_price - 1.0) * 100.0
        if first_block_price and first_block_price > 0 and not first_entry
        else None
    )
    cooldown_harm = None
    if first_cooldown_block_price and first_cooldown_block_price > 0:
        cooldown_move = (perf.day_close / first_cooldown_block_price - 1.0) * 100.0
        cooldown_harm = max(0.0, cooldown_move)
    blocked_reason_counts = Counter(_blocked_reason_code(item) for item in blocked)
    missed_reason_code = None
    if status != "bought":
        missed_reason_code = (
            blocked_reason_counts.most_common(1)[0][0]
            if blocked_reason_counts
            else status
        )

    summary = {
        "symbol": perf.symbol,
        "day_change_pct": round(perf.day_change_pct, 3),
        "day_open": perf.day_open,
        "day_close": perf.day_close,
        "quote_volume_24h": round(perf.quote_volume_24h, 2),
        "in_watchlist": perf.in_watchlist,
        "status": status,
        "reason": reason,
        "entries_count": len(entries),
        "blocked_count": len(blocked),
        "blocked_reason_counts": _counter_payload(blocked_reason_counts),
        "blocked_sources": sorted({_event_source(item) for item in blocked}),
        "first_block_time": first_block.get("_ts_local") if first_block else None,
        "last_block_time": last_block.get("_ts_local") if last_block else None,
        "first_block_reason_code": _blocked_reason_code(first_block) if first_block else None,
        "first_block_price": first_block_price,
        "latest_block_reason_code": _blocked_reason_code(last_block) if last_block else None,
        "latest_block_reason": last_block.get("reason") if last_block else None,
        "latest_block_gate": last_block.get("gate") if last_block else None,
        "missed_reason_code": missed_reason_code,
        "opportunity_no_entry_pct": None if first_entry else round(perf.day_change_pct, 3),
        "opportunity_from_first_block_pct": None if opportunity_from_first_block is None else round(opportunity_from_first_block, 3),
        "entry_sources": sorted({_event_source(item) for item in entries}),
        "first_entry_time": first_entry.get("_ts_local") if first_entry else None,
        "first_entry_mode": (first_entry.get("mode") or first_entry.get("signal_type")) if first_entry else None,
        "first_entry_source": _event_source(first_entry) if first_entry else None,
        "first_entry_price": first_entry_price,
        "capture_ratio": None if capture_ratio is None else round(capture_ratio, 4),
        "capture_ratio_at_entry": None if capture_ratio is None else round(capture_ratio, 4),
        "lead_time_to_final_top_min": lead_time_to_final_top_min,
        "opportunity_from_entry_pct": None if opportunity_from_entry is None else round(opportunity_from_entry, 3),
        "latest_exit_time": last_exit.get("_ts_local") if last_exit else None,
        "latest_exit_pnl_pct": last_exit.get("pnl_pct") if last_exit else None,
        "latest_exit_reason": last_exit.get("reason") if last_exit else None,
        "exit_efficiency": None if exit_efficiency is None else round(exit_efficiency, 4),
        "giveback_pct": None if giveback_pct is None else round(giveback_pct, 4),
        "first_cooldown_block_time": first_cooldown_block.get("_ts_local") if first_cooldown_block else None,
        "first_cooldown_block_price": first_cooldown_block_price,
        "cooldown_harm_pct": None if cooldown_harm is None else round(cooldown_harm, 4),
    }
    return summary


def _blocked_reason_harm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    acc: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "")
        counts = row.get("blocked_reason_counts") or {}
        if not symbol or not isinstance(counts, dict):
            continue
        for code, count in counts.items():
            code_text = str(code)
            payload = acc.setdefault(
                code_text,
                {
                    "reason_code": code_text,
                    "blocked_events": 0,
                    "symbols": set(),
                    "missed_symbols": set(),
                    "missed_opportunity_pct": 0.0,
                    "examples": [],
                },
            )
            payload["blocked_events"] += int(count or 0)
            payload["symbols"].add(symbol)
            if row.get("status") != "bought":
                payload["missed_symbols"].add(symbol)
                opportunity = row.get("opportunity_from_first_block_pct")
                if opportunity is None:
                    opportunity = row.get("opportunity_no_entry_pct")
                try:
                    opportunity_value = float(opportunity or 0.0)
                except (TypeError, ValueError):
                    opportunity_value = 0.0
                payload["missed_opportunity_pct"] += opportunity_value
                if len(payload["examples"]) < 5:
                    payload["examples"].append(
                        {
                            "symbol": symbol,
                            "opportunity_pct": round(opportunity_value, 3),
                            "status": row.get("status"),
                        }
                    )

    out: list[dict[str, Any]] = []
    for payload in acc.values():
        symbols = sorted(payload.pop("symbols"))
        missed_symbols = sorted(payload.pop("missed_symbols"))
        missed_count = len(missed_symbols)
        missed_opportunity = round(float(payload["missed_opportunity_pct"]), 3)
        out.append(
            {
                **payload,
                "symbols": symbols,
                "symbols_count": len(symbols),
                "missed_symbols": missed_symbols,
                "missed_symbols_count": missed_count,
                "missed_opportunity_pct": missed_opportunity,
                "avg_missed_opportunity_pct": round(missed_opportunity / missed_count, 3)
                if missed_count
                else 0.0,
            }
        )
    out.sort(
        key=lambda item: (
            -int(item.get("missed_symbols_count") or 0),
            -float(item.get("missed_opportunity_pct") or 0.0),
            -int(item.get("blocked_events") or 0),
            str(item.get("reason_code") or ""),
        )
    )
    return out


def build_report(
    *,
    target_day: date,
    phase: str,
    timezone_name: str = DEFAULT_TZ,
    top_n: int = DEFAULT_TOP_N,
    min_quote_volume: float = DEFAULT_MIN_QUOTE_VOLUME,
    day_performance: Optional[list[DayPerformance]] = None,
    cutoff_hour: int | None = None,
) -> dict[str, Any]:
    tz = ZoneInfo(timezone_name)
    start_local, end_local = _local_window(target_day, phase, tz, cutoff_hour=cutoff_hour)
    if day_performance is None:
        day_performance = asyncio.run(
            fetch_day_performance(
                target_day=target_day,
                phase=phase,
                timezone_name=timezone_name,
                min_quote_volume=min_quote_volume,
                cutoff_hour=cutoff_hour,
            )
        )

    all_top = day_performance[:top_n]
    watchlist_top = [x for x in day_performance if x.in_watchlist][:top_n]
    event_rows = _load_day_events(start_local, end_local, tz)

    all_top_summary = [
        summarize_top_gainer(item, event_rows.get(item.symbol, {}), end_local=end_local, tz=tz)
        for item in all_top
    ]
    watchlist_top_summary = [
        summarize_top_gainer(item, event_rows.get(item.symbol, {}), end_local=end_local, tz=tz)
        for item in watchlist_top
    ]

    watchlist_top_set = {item["symbol"] for item in watchlist_top_summary}
    bought_symbols: set[str] = set()
    entry_source_counts: Counter = Counter()
    for sym, buckets in event_rows.items():
        for rec in buckets.get("entries", []):
            bought_symbols.add(sym)
            entry_source_counts[_event_source(rec)] += 1
    false_positive_symbols = sorted(sym for sym in bought_symbols if sym not in watchlist_top_set)

    bought_top = [x for x in watchlist_top_summary if x["status"] == "bought"]
    missed_top = [x for x in watchlist_top_summary if x["status"] != "bought"]
    blocked_winners = [x for x in missed_top if int(x.get("blocked_count") or 0) > 0]
    missed_reason_counts = Counter(
        str(x.get("missed_reason_code") or x.get("status") or "unknown")
        for x in missed_top
    )
    missed_reason_symbols: dict[str, list[str]] = {}
    for item in missed_top:
        reason_code = str(item.get("missed_reason_code") or item.get("status") or "unknown")
        missed_reason_symbols.setdefault(reason_code, []).append(str(item.get("symbol") or ""))
    watchlist_blocked_reason_counts: Counter = Counter()
    for item in watchlist_top_summary:
        for reason_code, count in (item.get("blocked_reason_counts") or {}).items():
            watchlist_blocked_reason_counts[str(reason_code)] += int(count or 0)
    blocked_winner_reason_counts: Counter = Counter()
    for item in blocked_winners:
        for reason_code, count in (item.get("blocked_reason_counts") or {}).items():
            blocked_winner_reason_counts[str(reason_code)] += int(count or 0)
    early_captured = [
        x for x in bought_top
        if (x.get("capture_ratio") or 0.0) >= 0.35
    ]
    coverage = [x for x in all_top_summary if x["in_watchlist"]]

    report = {
        "target_day_local": target_day.isoformat(),
        "phase": phase,
        "window_local": {
            "start": start_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "end": end_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        },
        "settings": {
            "timezone": timezone_name,
            "top_n": top_n,
            "min_quote_volume_24h": min_quote_volume,
            "early_capture_ratio_min": 0.35,
            "cutoff_hour_local": cutoff_hour,
        },
        "summary": {
            "exchange_top_count": len(all_top_summary),
            "exchange_top_in_watchlist": len(coverage),
            "watchlist_top_count": len(watchlist_top_summary),
            "watchlist_top_bought": len(bought_top),
            "watchlist_top_early_captured": len(early_captured),
            "watchlist_top_missed": len(missed_top),
            "watchlist_top_capture_rate_pct": round((len(bought_top) / len(watchlist_top_summary) * 100.0), 2) if watchlist_top_summary else 0.0,
            "watchlist_top_early_capture_rate_pct": round((len(early_captured) / len(watchlist_top_summary) * 100.0), 2) if watchlist_top_summary else 0.0,
            "bot_unique_buys": len(bought_symbols),
            "bot_false_positive_buys": len(false_positive_symbols),
            "entry_source_counts": _counter_payload(entry_source_counts),
            "blocked_winner_count": len(blocked_winners),
            "blocked_winner_symbols": [str(x.get("symbol")) for x in blocked_winners],
            "missed_reason_counts": _counter_payload(missed_reason_counts),
            "watchlist_blocked_reason_counts": _counter_payload(watchlist_blocked_reason_counts),
            "blocked_winner_reason_counts": _counter_payload(blocked_winner_reason_counts),
        },
        "exchange_top_gainers": all_top_summary,
        "watchlist_top_gainers": watchlist_top_summary,
        "missed_reason_symbols": missed_reason_symbols,
        "blocked_reason_harm": _blocked_reason_harm(watchlist_top_summary),
        "why_no_signal_examples": [
            {
                "symbol": item.get("symbol"),
                "missed_reason_code": item.get("missed_reason_code"),
                "blocked_reason_counts": item.get("blocked_reason_counts"),
                "first_block_time": item.get("first_block_time"),
                "last_block_time": item.get("last_block_time"),
                "latest_block_reason_code": item.get("latest_block_reason_code"),
                "latest_block_reason": item.get("latest_block_reason"),
                "latest_block_gate": item.get("latest_block_gate"),
            }
            for item in blocked_winners[:10]
        ],
        "bot_false_positive_symbols": false_positive_symbols[:top_n],
    }
    return report


def render_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    cutoff_hour = (report.get("settings") or {}).get("cutoff_hour_local")
    cutoff_text = "" if cutoff_hour is None else f", cutoff={int(cutoff_hour):02d}:00"
    lines = [
        f"Top-gainer critic for {report['target_day_local']} ({report['phase']})",
        f"Window: {report['window_local']['start']} -> {report['window_local']['end']}{cutoff_text}",
        "",
        "Summary:",
        f"  exchange top in watchlist: {summary['exchange_top_in_watchlist']}/{summary['exchange_top_count']}",
        f"  watchlist top bought: {summary['watchlist_top_bought']}/{summary['watchlist_top_count']} ({summary['watchlist_top_capture_rate_pct']:.1f}%)",
        f"  early captures: {summary['watchlist_top_early_captured']}/{summary['watchlist_top_count']} ({summary['watchlist_top_early_capture_rate_pct']:.1f}%)",
        f"  bot false-positive buys: {summary['bot_false_positive_buys']}/{summary['bot_unique_buys']}",
        f"  blocked winners: {summary.get('blocked_winner_count', 0)}",
        f"  missed reasons: {_compact_count_map(summary.get('missed_reason_counts'))}",
        f"  blocked winner filters: {_compact_count_map(summary.get('blocked_winner_reason_counts'))}",
        f"  blocked filters: {_compact_count_map(summary.get('watchlist_blocked_reason_counts'))}",
        "",
    ]
    if report.get("blocked_reason_harm"):
        lines.append("Filter harm by missed top-gainers:")
        for item in report["blocked_reason_harm"][:5]:
            lines.append(
                "  "
                f"{item['reason_code']}: missed={item['missed_symbols_count']} "
                f"events={item['blocked_events']} "
                f"opportunity={float(item['missed_opportunity_pct']):+.2f}%"
            )
        lines.append("")
    if report.get("why_no_signal_examples"):
        lines.append("Why-no-signal examples:")
        for item in report["why_no_signal_examples"][:5]:
            lines.append(
                f"  {item['symbol']}: dominant={item.get('missed_reason_code')} "
                f"latest={item.get('latest_block_reason_code')} "
                f"gate={item.get('latest_block_gate')} "
                f"last={item.get('last_block_time')} "
                f"reason={item.get('latest_block_reason')}"
            )
        lines.append("")
    lines.append("Watchlist top gainers and bot reaction:")
    for idx, item in enumerate(report["watchlist_top_gainers"], start=1):
        lines.append(
            f"{idx}. {item['symbol']} {item['day_change_pct']:+.2f}% status={item['status']}"
        )
        if item["first_entry_time"]:
            capture = "n/a" if item["capture_ratio"] is None else f"{float(item['capture_ratio']) * 100:.1f}%"
            lead = item.get("lead_time_to_final_top_min")
            lead_text = "n/a" if lead is None else f"{int(lead)}m"
            lines.append(
                f"   BUY {item['first_entry_time']} mode={item['first_entry_mode']} source={item.get('first_entry_source')} entry={item['first_entry_price']} capture={capture}"
                f" lead={lead_text}"
            )
        if item.get("first_block_time"):
            opp = item.get("opportunity_from_first_block_pct")
            opp_text = "n/a" if opp is None else f"{float(opp):+.2f}%"
            lines.append(
                f"   BLOCKED {item['first_block_time']}..{item['last_block_time']} "
                f"codes={_compact_count_map(item.get('blocked_reason_counts'))} "
                f"sources={','.join(item.get('blocked_sources') or [])} "
                f"opp_from_first_block={opp_text}"
            )
        if item["latest_exit_pnl_pct"] is not None:
            eff = item.get("exit_efficiency")
            giveback = item.get("giveback_pct")
            eff_text = "n/a" if eff is None else f"{float(eff) * 100:.1f}%"
            giveback_text = "n/a" if giveback is None else f"{float(giveback):+.2f}%"
            lines.append(
                f"   EXIT {item['latest_exit_time']} pnl={float(item['latest_exit_pnl_pct']):+.2f}% "
                f"eff={eff_text} giveback={giveback_text}"
            )
        if item.get("cooldown_harm_pct") is not None:
            lines.append(
                f"   COOLDOWN harm={float(item['cooldown_harm_pct']):+.2f}% "
                f"from={item.get('first_cooldown_block_time')}"
            )
        if item["reason"]:
            code = item.get("missed_reason_code") or item.get("first_block_reason_code") or "rule"
            lines.append(f"   WHY {code}: {item['reason']}")

    if report["bot_false_positive_symbols"]:
        lines.append("")
        lines.append("Bot bought, but symbol did not finish in watchlist top gainers:")
        lines.append("  " + ", ".join(report["bot_false_positive_symbols"]))
    return "\n".join(lines)


def save_report(report: dict[str, Any]) -> dict[str, str]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = f"top_gainer_critic_{report['target_day_local']}_{report['phase']}"
    json_path = REPORT_DIR / f"{base}.json"
    txt_path = REPORT_DIR / f"{base}.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(render_text(report), encoding="utf-8")
    with HISTORY_FILE.open("a", encoding="utf-8") as fh:
        history_row = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target_day_local": report["target_day_local"],
            "phase": report["phase"],
            "summary": report["summary"],
            "files": {"json": str(json_path), "txt": str(txt_path)},
        }
        fh.write(json.dumps(history_row, ensure_ascii=False) + "\n")
    return {"json": str(json_path), "txt": str(txt_path)}


def run_report(
    *,
    target_day: date,
    phase: str,
    timezone_name: str = DEFAULT_TZ,
    top_n: int = DEFAULT_TOP_N,
    min_quote_volume: float = DEFAULT_MIN_QUOTE_VOLUME,
) -> dict[str, Any]:
    report = build_report(
        target_day=target_day,
        phase=phase,
        timezone_name=timezone_name,
        top_n=top_n,
        min_quote_volume=min_quote_volume,
    )
    report["teacher_annotation"] = critic_dataset.annotate_top_gainer_teacher(report)
    files = save_report(report)
    report["files"] = files
    return report


def main() -> int:
    args = parse_args()
    tz = ZoneInfo(args.timezone)
    if args.date:
        target_day = date.fromisoformat(args.date)
    else:
        today = datetime.now(tz).date()
        target_day = today if args.phase == "midday" else date.fromordinal(today.toordinal() - 1)
    report = run_report(
        target_day=target_day,
        phase=args.phase,
        timezone_name=args.timezone,
        top_n=args.top,
        min_quote_volume=args.min_quote_volume,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_text(report))
        print("")
        print(f"JSON report saved to: {report['files']['json']}")
        print(f"Text report saved to: {report['files']['txt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
