#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable
from zoneinfo import ZoneInfo


BINANCE_URL = "https://api.binance.com/api/v3/klines"
BAR_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}
DEFAULT_HORIZON_BARS = {
    "15m": 32,
    "1h": 24,
    "4h": 12,
}


@dataclass
class Event:
    event: str
    sym: str
    tf: str
    ts_ms: int
    ts: str
    source: str
    price: float
    mode: str = ""
    reason: str = ""
    pnl_pct: float | None = None
    raw: dict[str, Any] | None = None


@dataclass
class Trade:
    sym: str
    tf: str
    source: str
    mode: str
    entry_ts_ms: int
    entry_ts: str
    entry_price: float
    exit_ts_ms: int | None = None
    exit_ts: str | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    pnl_pct: float | None = None


@dataclass
class TrendEpisode:
    sym: str
    tf: str
    start_i: int
    peak_i: int
    end_i: int
    start_ts: str
    peak_ts: str
    end_ts: str
    start_price: float
    peak_price: float
    end_price: float
    move_pct: float
    duration_bars: int
    top_mover_rank: int | None = None
    top_mover_change_pct: float | None = None


def _repo_root_from_script() -> Path:
    # .../skills/signal-quality-evaluator/scripts/evaluate_signals.py
    return Path(__file__).resolve().parents[3]


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        out = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, digits)


def _pct(a: float, b: float) -> float:
    if b <= 0:
        return 0.0
    return (a / b - 1.0) * 100.0


def _norm_source(raw: str, default: str) -> str:
    source = str(raw or default or "").strip().lower()
    if "agent" in source or source == "market_agent":
        return "agent"
    if source == "bot" or not source:
        return "bot"
    return source


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def _load_events(
    *,
    files_root: Path,
    start_ms: int,
    end_ms: int,
    symbols: set[str] | None,
    tfs: set[str],
    source_filter: str,
) -> list[Event]:
    paths = [
        (files_root / "bot_events.jsonl", "bot"),
        (files_root / "agent_events.jsonl", "agent"),
    ]
    rows: list[Event] = []
    for path, default_source in paths:
        for rec in _iter_jsonl(path):
            kind = str(rec.get("event") or "").strip()
            if kind not in {"entry", "exit"}:
                continue
            sym = str(rec.get("sym") or rec.get("symbol") or "").strip().upper()
            tf = str(rec.get("tf") or "").strip()
            if not sym or not tf:
                continue
            if symbols and sym not in symbols:
                continue
            if tf not in tfs:
                continue
            ts_dt = _parse_ts(rec.get("ts"))
            if not ts_dt:
                continue
            ts_ms = _ms(ts_dt)
            if ts_ms < start_ms or ts_ms > end_ms:
                continue
            source = _norm_source(str(rec.get("source") or ""), default_source)
            if source_filter != "all" and source != source_filter:
                continue
            price_key = "price" if kind == "entry" else "exit_price"
            price = _safe_float(rec.get(price_key), 0.0)
            if price <= 0:
                price = _safe_float(rec.get("price"), 0.0)
            if price <= 0:
                continue
            rows.append(
                Event(
                    event=kind,
                    sym=sym,
                    tf=tf,
                    ts_ms=ts_ms,
                    ts=_iso(ts_ms),
                    source=source,
                    price=price,
                    mode=str(rec.get("mode") or ""),
                    reason=str(rec.get("reason") or ""),
                    pnl_pct=None if rec.get("pnl_pct") is None else _safe_float(rec.get("pnl_pct")),
                    raw=rec,
                )
            )
    rows.sort(key=lambda x: (x.ts_ms, x.source, x.sym, x.tf, x.event))
    return rows


def _pair_trades(events: list[Event]) -> list[Trade]:
    open_by_key: dict[tuple[str, str, str], deque[Trade]] = defaultdict(deque)
    trades: list[Trade] = []
    for ev in events:
        key = (ev.source, ev.sym, ev.tf)
        if ev.event == "entry":
            trade = Trade(
                sym=ev.sym,
                tf=ev.tf,
                source=ev.source,
                mode=ev.mode,
                entry_ts_ms=ev.ts_ms,
                entry_ts=ev.ts,
                entry_price=ev.price,
            )
            open_by_key[key].append(trade)
            trades.append(trade)
        elif ev.event == "exit":
            queue = open_by_key.get(key)
            if not queue:
                continue
            trade = queue.popleft()
            trade.exit_ts_ms = ev.ts_ms
            trade.exit_ts = ev.ts
            trade.exit_price = ev.price
            trade.exit_reason = ev.reason
            trade.pnl_pct = ev.pnl_pct
    trades.sort(key=lambda x: x.entry_ts_ms)
    return trades


def _load_watchlist(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [str(x).strip().upper() for x in payload if str(x).strip()]


def _cache_path(cache_dir: Path, symbol: str, tf: str, start_ms: int, end_ms: int) -> Path:
    return cache_dir / f"{symbol}_{tf}_{start_ms}_{end_ms}.json"


def _fetch_klines(symbol: str, tf: str, start_ms: int, end_ms: int, cache_dir: Path) -> list[dict[str, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(cache_dir, symbol, tf, start_ms, end_ms)
    if cache.exists():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                return cached
        except Exception:
            pass

    rows: list[list[Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        params = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": tf,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            }
        )
        req = urllib.request.Request(f"{BINANCE_URL}?{params}", headers={"User-Agent": "signal-quality-evaluator/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                batch = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return []
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + BAR_MS[tf]
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
        time.sleep(0.05)

    out: list[dict[str, float]] = []
    for item in rows:
        try:
            out.append(
                {
                    "t": int(item[0]),
                    "o": float(item[1]),
                    "h": float(item[2]),
                    "l": float(item[3]),
                    "c": float(item[4]),
                    "v": float(item[5]),
                }
            )
        except (TypeError, ValueError, IndexError):
            continue
    try:
        cache.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    except Exception:
        pass
    return out


def _find_bar_index(candles: list[dict[str, float]], ts_ms: int) -> int | None:
    times = [int(x["t"]) for x in candles]
    idx = bisect.bisect_right(times, ts_ms) - 1
    return idx if 0 <= idx < len(candles) else None


def _detect_trends(
    *,
    symbol: str,
    tf: str,
    candles: list[dict[str, float]],
    min_move_pct: float,
    min_bars: int,
    reversal_pct: float,
) -> list[TrendEpisode]:
    if len(candles) < max(8, min_bars + 2):
        return []
    episodes: list[TrendEpisode] = []
    trough_i = 0
    active = False
    start_i = 0
    peak_i = 0
    peak_price = float(candles[0]["h"])
    reversal_factor = 1.0 - max(0.0, reversal_pct) / 100.0

    for i in range(1, len(candles)):
        low = float(candles[i]["l"])
        high = float(candles[i]["h"])
        close = float(candles[i]["c"])
        if not active:
            if low < float(candles[trough_i]["l"]):
                trough_i = i
            start_price = float(candles[trough_i]["l"])
            move_pct = _pct(high, start_price)
            if move_pct >= min_move_pct and i - trough_i >= min_bars:
                active = True
                start_i = trough_i
                peak_i = i
                peak_price = high
            continue

        if high > peak_price:
            peak_i = i
            peak_price = high
        should_close = close <= peak_price * reversal_factor
        is_last = i == len(candles) - 1
        if should_close or is_last:
            if peak_i - start_i >= min_bars:
                start_price = float(candles[start_i]["l"])
                end_i = i
                end_price = float(candles[end_i]["c"])
                episodes.append(
                    TrendEpisode(
                        sym=symbol,
                        tf=tf,
                        start_i=start_i,
                        peak_i=peak_i,
                        end_i=end_i,
                        start_ts=_iso(int(candles[start_i]["t"])),
                        peak_ts=_iso(int(candles[peak_i]["t"])),
                        end_ts=_iso(int(candles[end_i]["t"])),
                        start_price=start_price,
                        peak_price=peak_price,
                        end_price=end_price,
                        move_pct=_pct(peak_price, start_price),
                        duration_bars=end_i - start_i,
                    )
                )
            active = False
            trough_i = i
            peak_i = i
            peak_price = high
    return episodes


def _local_day(ms: int, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(tz).date().isoformat()


def _annotate_top_movers(
    episodes_by_key: dict[tuple[str, str], list[TrendEpisode]],
    candles_by_key: dict[tuple[str, str], list[dict[str, float]]],
    *,
    tz: ZoneInfo,
    top_n: int,
) -> dict[str, Any]:
    by_day: dict[str, list[tuple[str, float]]] = defaultdict(list)
    preferred: dict[str, list[dict[str, float]]] = {}
    for (sym, tf), candles in candles_by_key.items():
        if not candles:
            continue
        if sym not in preferred or tf == "15m":
            preferred[sym] = candles
    for sym, candles in preferred.items():
        grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
        for row in candles:
            grouped[_local_day(int(row["t"]), tz)].append(row)
        for day, rows in grouped.items():
            rows.sort(key=lambda x: int(x["t"]))
            first = rows[0]
            last = rows[-1]
            open_price = float(first["o"])
            close_price = float(last["c"])
            if open_price > 0:
                by_day[day].append((sym, _pct(close_price, open_price)))

    ranks: dict[tuple[str, str], tuple[int, float]] = {}
    for day, rows in by_day.items():
        rows.sort(key=lambda x: x[1], reverse=True)
        for rank, (sym, change_pct) in enumerate(rows[:top_n], start=1):
            ranks[(day, sym)] = (rank, change_pct)

    for (sym, _tf), episodes in episodes_by_key.items():
        for ep in episodes:
            day = _local_day(int(candles_by_key[(sym, ep.tf)][ep.start_i]["t"]), tz)
            rank = ranks.get((day, sym))
            if rank:
                ep.top_mover_rank = rank[0]
                ep.top_mover_change_pct = rank[1]
    return {
        "top_n": top_n,
        "days_ranked": len(by_day),
        "partial_universe": False,
    }


def _match_trend(entry_i: int, episodes: list[TrendEpisode]) -> TrendEpisode | None:
    best: TrendEpisode | None = None
    for ep in episodes:
        if ep.start_i <= entry_i <= ep.end_i:
            if best is None or ep.move_pct > best.move_pct:
                best = ep
    return best


def _window_high(candles: list[dict[str, float]], start_i: int, end_i: int) -> float:
    if not candles:
        return 0.0
    lo = max(0, start_i)
    hi = min(len(candles) - 1, end_i)
    if hi < lo:
        return 0.0
    return max(float(candles[i]["h"]) for i in range(lo, hi + 1))


def _evaluate_trade(
    trade: Trade,
    candles: list[dict[str, float]],
    episodes: list[TrendEpisode],
    *,
    horizon_bars: int,
    late_entry_capture_max: float,
    early_exit_after_pct: float,
    late_exit_giveback_pct: float,
    false_positive_max_fav_pct: float,
) -> dict[str, Any] | None:
    entry_i = _find_bar_index(candles, trade.entry_ts_ms)
    if entry_i is None:
        return None
    exit_i = _find_bar_index(candles, trade.exit_ts_ms) if trade.exit_ts_ms else None
    analysis_end_i = exit_i if exit_i is not None else len(candles) - 1
    entry_price = trade.entry_price
    exit_price = trade.exit_price if trade.exit_price and trade.exit_price > 0 else float(candles[analysis_end_i]["c"])
    pnl_pct = trade.pnl_pct if trade.pnl_pct is not None else _pct(exit_price, entry_price)
    high_until_exit = _window_high(candles, entry_i, analysis_end_i)
    high_future = _window_high(candles, entry_i, min(len(candles) - 1, entry_i + horizon_bars))
    max_favorable_pct = _pct(high_until_exit, entry_price)
    future_favorable_pct = _pct(high_future, entry_price)
    exit_efficiency = pnl_pct / max_favorable_pct if max_favorable_pct > 0 else None
    giveback_pct = max(0.0, max_favorable_pct - pnl_pct)

    ep = _match_trend(entry_i, episodes)
    trend_payload = None
    capture_ratio_at_entry = None
    realized_capture_ratio = None
    mfe_capture_ratio = None
    entry_timing = "no_trend"
    exit_timing = "open" if trade.exit_ts_ms is None else "unmatched"
    false_positive = ep is None and future_favorable_pct < false_positive_max_fav_pct

    if ep is not None:
        full_move = ep.peak_price - ep.start_price
        if full_move > 0:
            capture_ratio_at_entry = max(0.0, min(1.5, (ep.peak_price - entry_price) / full_move))
            realized_capture_ratio = (exit_price - entry_price) / full_move
            mfe_capture_ratio = (min(high_until_exit, ep.peak_price) - entry_price) / full_move
        delay_bars = entry_i - ep.start_i
        if capture_ratio_at_entry is not None and capture_ratio_at_entry < late_entry_capture_max:
            entry_timing = "late"
        elif delay_bars <= 1:
            entry_timing = "early"
        else:
            entry_timing = "on_time"
        trend_payload = asdict(ep)
        if trade.exit_ts_ms is not None:
            if exit_i is not None and exit_i < ep.peak_i:
                post_exit_high = _window_high(candles, exit_i + 1, ep.end_i)
                post_exit_runup_pct = _pct(post_exit_high, exit_price)
                exit_timing = "early" if post_exit_runup_pct >= early_exit_after_pct else "timely"
            elif giveback_pct >= late_exit_giveback_pct:
                exit_timing = "late"
            else:
                exit_timing = "timely"

    return {
        "sym": trade.sym,
        "tf": trade.tf,
        "source": trade.source,
        "mode": trade.mode,
        "entry_ts": trade.entry_ts,
        "entry_price": trade.entry_price,
        "exit_ts": trade.exit_ts,
        "exit_price": _round_or_none(exit_price, 10),
        "exit_reason": trade.exit_reason,
        "pnl_pct": _round_or_none(pnl_pct),
        "max_favorable_pct": _round_or_none(max_favorable_pct),
        "future_favorable_pct": _round_or_none(future_favorable_pct),
        "exit_efficiency": _round_or_none(exit_efficiency),
        "giveback_pct": _round_or_none(giveback_pct),
        "entry_timing": entry_timing,
        "exit_timing": exit_timing,
        "false_positive": bool(false_positive),
        "capture_ratio_at_entry": _round_or_none(capture_ratio_at_entry),
        "realized_capture_ratio": _round_or_none(realized_capture_ratio),
        "mfe_capture_ratio": _round_or_none(mfe_capture_ratio),
        "trend": trend_payload,
    }


def _stats(values: list[float | None]) -> dict[str, Any]:
    rows = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not rows:
        return {"n": 0, "avg": None, "median": None, "min": None, "max": None}
    return {
        "n": len(rows),
        "avg": round(mean(rows), 4),
        "median": round(median(rows), 4),
        "min": round(min(rows), 4),
        "max": round(max(rows), 4),
    }


def _summarize(evaluated: list[dict[str, Any]], missed: list[TrendEpisode]) -> dict[str, Any]:
    buys = len(evaluated)
    matched = [x for x in evaluated if x.get("trend")]
    false_positive = [x for x in evaluated if x.get("false_positive")]
    late_entries = [x for x in evaluated if x.get("entry_timing") == "late"]
    early_entries = [x for x in evaluated if x.get("entry_timing") == "early"]
    early_exits = [x for x in evaluated if x.get("exit_timing") == "early"]
    late_exits = [x for x in evaluated if x.get("exit_timing") == "late"]
    closed = [x for x in evaluated if x.get("exit_ts")]
    top_mover_missed = [x for x in missed if x.top_mover_rank is not None]
    top_mover_caught = [x for x in matched if x.get("trend", {}).get("top_mover_rank") is not None]
    trend_count = len(missed) + len({(x["trend"]["sym"], x["trend"]["tf"], x["trend"]["start_ts"]) for x in matched if x.get("trend")})
    return {
        "buys_total": buys,
        "closed_trades": len(closed),
        "matched_trend_buys": len(matched),
        "trend_episodes_total": trend_count,
        "missed_trends": len(missed),
        "miss_rate": round(len(missed) / trend_count, 4) if trend_count else None,
        "false_positive_buys": len(false_positive),
        "false_positive_rate": round(len(false_positive) / buys, 4) if buys else None,
        "late_entries": len(late_entries),
        "early_entries": len(early_entries),
        "early_exits": len(early_exits),
        "late_exits": len(late_exits),
        "top_mover_caught_trends": len(top_mover_caught),
        "top_mover_missed_trends": len(top_mover_missed),
        "capture_ratio_at_entry": _stats([x.get("capture_ratio_at_entry") for x in evaluated]),
        "realized_capture_ratio": _stats([x.get("realized_capture_ratio") for x in evaluated]),
        "exit_efficiency": _stats([x.get("exit_efficiency") for x in evaluated]),
        "giveback_pct": _stats([x.get("giveback_pct") for x in evaluated]),
        "pnl_pct": _stats([x.get("pnl_pct") for x in evaluated]),
    }


def _render_text(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "Signal quality evaluator",
        f"window: {report['window']['start']} .. {report['window']['end']}",
        f"scope: source={report['scope']['source']} symbols={report['scope']['symbols_count']} tfs={','.join(report['scope']['timeframes'])}",
        (
            f"buys: {s['buys_total']}  matched trends: {s['matched_trend_buys']}  "
            f"missed trends: {s['missed_trends']}  false-positive buys: {s['false_positive_buys']}"
        ),
        (
            f"miss_rate: {s['miss_rate']}  false_positive_rate: {s['false_positive_rate']}  "
            f"late_entries: {s['late_entries']}  early_exits: {s['early_exits']}  late_exits: {s['late_exits']}"
        ),
        (
            f"capture_ratio median: {s['capture_ratio_at_entry']['median']}  "
            f"exit_eff median: {s['exit_efficiency']['median']}  giveback median: {s['giveback_pct']['median']}"
        ),
        (
            f"top-mover trends caught/missed: {s['top_mover_caught_trends']}/{s['top_mover_missed_trends']}"
        ),
    ]
    if report.get("top_movers", {}).get("partial_universe"):
        lines.append("note: top-mover ranks are partial because the run used a symbol/max-symbols filter")
    if report["late_entries"]:
        lines.append("")
        lines.append("Worst late entries:")
        for row in report["late_entries"][:10]:
            lines.append(
                f"- {row['sym']} {row['tf']} {row['source']} {row['mode']} "
                f"entry={row['entry_ts']} capture={row.get('capture_ratio_at_entry')} pnl={row.get('pnl_pct')}"
            )
    if report["early_exits"]:
        lines.append("")
        lines.append("Early exits:")
        for row in report["early_exits"][:10]:
            lines.append(
                f"- {row['sym']} {row['tf']} exit={row.get('exit_ts')} "
                f"eff={row.get('exit_efficiency')} giveback={row.get('giveback_pct')} reason={row.get('exit_reason')}"
            )
    if report["missed_trends"]:
        lines.append("")
        lines.append("Missed trends:")
        for ep in report["missed_trends"][:15]:
            rank = "" if ep.get("top_mover_rank") is None else f" top#{ep.get('top_mover_rank')}"
            lines.append(
                f"- {ep['sym']} {ep['tf']}{rank} start={ep['start_ts']} peak={ep['peak_ts']} move={ep['move_pct']:+.2f}%"
            )
    if report["false_positive_buys"]:
        lines.append("")
        lines.append("False-positive buys:")
        for row in report["false_positive_buys"][:15]:
            lines.append(
                f"- {row['sym']} {row['tf']} {row['source']} entry={row['entry_ts']} "
                f"future_mfe={row.get('future_favorable_pct')} pnl={row.get('pnl_pct')}"
            )
    return "\n".join(lines)


def _parse_symbols(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    rows = {x.strip().upper() for x in raw.split(",") if x.strip()}
    return rows or None


def _parse_tfs(raw: str) -> set[str]:
    rows = {x.strip() for x in raw.split(",") if x.strip()}
    bad = rows - set(BAR_MS)
    if bad:
        raise SystemExit(f"Unsupported timeframe(s): {', '.join(sorted(bad))}")
    return rows or {"15m", "1h"}


def _window_from_args(args: argparse.Namespace, tz: ZoneInfo) -> tuple[int, int]:
    if args.date:
        day = datetime.strptime(args.date, "%Y-%m-%d").date()
        start_local = datetime.combine(day, datetime.min.time(), tzinfo=tz)
        end_local = start_local + timedelta(days=1)
        return _ms(start_local.astimezone(timezone.utc)), _ms(end_local.astimezone(timezone.utc))
    if args.start or args.end:
        start_dt = _parse_ts(args.start) if args.start else datetime.now(timezone.utc) - timedelta(days=args.days)
        end_dt = _parse_ts(args.end) if args.end else datetime.now(timezone.utc)
        if not start_dt or not end_dt:
            raise SystemExit("Could not parse --start/--end")
        return _ms(start_dt), _ms(end_dt)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)
    return _ms(start_dt), _ms(end_dt)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root_from_script()
    files_root = repo_root / "files"
    tz = ZoneInfo(args.timezone)
    tfs = _parse_tfs(args.tf)
    requested_symbols = _parse_symbols(args.symbol)
    watchlist = _load_watchlist(Path(args.watchlist) if args.watchlist else files_root / "watchlist.json")
    if requested_symbols:
        symbols = sorted(requested_symbols)
    else:
        symbols = watchlist
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]
    symbol_set = set(symbols)
    start_ms, end_ms = _window_from_args(args, tz)
    fetch_start_ms = start_ms - max(BAR_MS[tf] for tf in tfs) * 4
    fetch_end_ms = end_ms + max(BAR_MS[tf] for tf in tfs) * max(DEFAULT_HORIZON_BARS.values())
    cache_dir = Path(args.cache_dir) if args.cache_dir else repo_root / ".runtime" / "signal_quality_cache"

    events = _load_events(
        files_root=files_root,
        start_ms=start_ms,
        end_ms=end_ms,
        symbols=symbol_set,
        tfs=tfs,
        source_filter=args.source,
    )
    trades = _pair_trades(events)

    candles_by_key: dict[tuple[str, str], list[dict[str, float]]] = {}
    episodes_by_key: dict[tuple[str, str], list[TrendEpisode]] = {}
    for sym in symbols:
        for tf in sorted(tfs):
            candles = _fetch_klines(sym, tf, fetch_start_ms, fetch_end_ms, cache_dir)
            if not candles:
                continue
            key = (sym, tf)
            candles_by_key[key] = candles
            episodes_by_key[key] = _detect_trends(
                symbol=sym,
                tf=tf,
                candles=candles,
                min_move_pct=args.trend_min_pct,
                min_bars=args.trend_min_bars,
                reversal_pct=args.reversal_pct,
            )

    top_mover_meta = _annotate_top_movers(episodes_by_key, candles_by_key, tz=tz, top_n=args.top_movers_n)
    top_mover_meta["partial_universe"] = bool(requested_symbols or args.max_symbols)

    evaluated: list[dict[str, Any]] = []
    caught_keys: set[tuple[str, str, str]] = set()
    for trade in trades:
        key = (trade.sym, trade.tf)
        candles = candles_by_key.get(key)
        if not candles:
            continue
        row = _evaluate_trade(
            trade,
            candles,
            episodes_by_key.get(key, []),
            horizon_bars=args.horizon_bars or DEFAULT_HORIZON_BARS.get(trade.tf, 24),
            late_entry_capture_max=args.late_entry_capture_max,
            early_exit_after_pct=args.early_exit_after_pct,
            late_exit_giveback_pct=args.late_exit_giveback_pct,
            false_positive_max_fav_pct=args.false_positive_max_fav_pct,
        )
        if not row:
            continue
        evaluated.append(row)
        trend = row.get("trend")
        if trend:
            caught_keys.add((trend["sym"], trend["tf"], trend["start_ts"]))

    missed: list[TrendEpisode] = []
    for (sym, tf), episodes in episodes_by_key.items():
        for ep in episodes:
            ep_start_ms = int(candles_by_key[(sym, tf)][ep.start_i]["t"])
            ep_end_ms = int(candles_by_key[(sym, tf)][ep.end_i]["t"])
            if ep_end_ms < start_ms or ep_start_ms > end_ms:
                continue
            if (ep.sym, ep.tf, ep.start_ts) not in caught_keys:
                missed.append(ep)
    missed.sort(key=lambda x: (x.top_mover_rank is None, x.top_mover_rank or 999, -x.move_pct))
    evaluated.sort(key=lambda x: (x["entry_ts"], x["sym"], x["tf"]))

    false_positive_rows = [x for x in evaluated if x.get("false_positive")]
    late_entry_rows = sorted(
        [x for x in evaluated if x.get("entry_timing") == "late"],
        key=lambda x: (x.get("capture_ratio_at_entry") is None, x.get("capture_ratio_at_entry") or 99),
    )
    early_exit_rows = sorted(
        [x for x in evaluated if x.get("exit_timing") == "early"],
        key=lambda x: -(x.get("giveback_pct") or 0.0),
    )

    report = {
        "window": {"start": _iso(start_ms), "end": _iso(end_ms), "timezone": args.timezone},
        "scope": {
            "source": args.source,
            "symbols_count": len(symbols),
            "timeframes": sorted(tfs),
            "requested_symbol_filter": sorted(requested_symbols) if requested_symbols else None,
        },
        "params": {
            "trend_min_pct": args.trend_min_pct,
            "trend_min_bars": args.trend_min_bars,
            "reversal_pct": args.reversal_pct,
            "top_movers_n": args.top_movers_n,
            "late_entry_capture_max": args.late_entry_capture_max,
            "early_exit_after_pct": args.early_exit_after_pct,
            "late_exit_giveback_pct": args.late_exit_giveback_pct,
            "false_positive_max_fav_pct": args.false_positive_max_fav_pct,
        },
        "top_movers": top_mover_meta,
        "summary": _summarize(evaluated, missed),
        "late_entries": late_entry_rows[:50],
        "early_exits": early_exit_rows[:50],
        "missed_trends": [asdict(x) for x in missed[:100]],
        "false_positive_buys": false_positive_rows[:100],
        "trades": evaluated if args.include_trades else [],
    }
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-factum evaluator for crypto bot BUY/SELL signal quality."
    )
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--date", help="Local date YYYY-MM-DD. Overrides --days.")
    parser.add_argument("--start", help="UTC/ISO start time.")
    parser.add_argument("--end", help="UTC/ISO end time.")
    parser.add_argument("--timezone", default="Europe/Budapest")
    parser.add_argument("--symbol", help="Comma-separated symbols, e.g. ICPUSDT,ACHUSDT.")
    parser.add_argument("--tf", default="15m,1h", help="Comma-separated timeframes: 15m,1h,4h.")
    parser.add_argument("--source", choices=["all", "bot", "agent"], default="all")
    parser.add_argument("--watchlist", default="")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--trend-min-pct", type=float, default=3.0)
    parser.add_argument("--trend-min-bars", type=int, default=4)
    parser.add_argument("--reversal-pct", type=float, default=1.2)
    parser.add_argument("--horizon-bars", type=int, default=0)
    parser.add_argument("--top-movers-n", type=int, default=15)
    parser.add_argument("--late-entry-capture-max", type=float, default=0.35)
    parser.add_argument("--early-exit-after-pct", type=float, default=1.0)
    parser.add_argument("--late-exit-giveback-pct", type=float, default=1.0)
    parser.add_argument("--false-positive-max-fav-pct", type=float, default=1.0)
    parser.add_argument("--include-trades", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", help="Optional output file. Writes JSON when --json is set, otherwise text.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args(argv or sys.argv[1:])
    report = build_report(args)
    text = json.dumps(report, ensure_ascii=False, indent=2) if args.json else _render_text(report)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
