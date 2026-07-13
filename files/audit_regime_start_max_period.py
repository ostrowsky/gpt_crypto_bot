from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import aiohttp
import numpy as np

import config
from regime_start import DAY_MS, FOUR_H_MS, RegimeStartProfile, RegimeStartSignal, detect_regime_starts, profile_from_config


ROOT = Path(__file__).resolve().parent.parent
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
DEFAULT_START = datetime(2017, 8, 17, tzinfo=timezone.utc)
DEFAULT_CACHE = ROOT / ".runtime" / "regime_start_history"
DEFAULT_OUTPUT = ROOT / ".runtime" / "reports" / "regime_start_max_period_latest.json"
BAR_MS = {"4h": FOUR_H_MS, "1d": DAY_MS}
DTYPE = [("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")]


@dataclass(frozen=True)
class LabeledSignal:
    symbol: str
    decision_ts_ms: int
    price: float
    ret_3d_pct: float
    ret_5d_pct: float
    ret_10d_pct: float
    mfe_5d_pct: float
    mae_5d_pct: float
    useful: bool
    signal: dict[str, Any]


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).isoformat().replace("+00:00", "Z")


def _rows_to_array(rows: Sequence[Sequence[Any]], *, end_ms: int, timeframe: str) -> np.ndarray:
    step = BAR_MS[timeframe]
    usable = [row for row in rows if int(row[0]) + step <= end_ms]
    arr = np.zeros(len(usable), dtype=DTYPE)
    if not usable:
        return arr
    arr["t"] = [int(row[0]) for row in usable]
    arr["o"] = [float(row[1]) for row in usable]
    arr["h"] = [float(row[2]) for row in usable]
    arr["l"] = [float(row[3]) for row in usable]
    arr["c"] = [float(row[4]) for row in usable]
    arr["v"] = [float(row[5]) for row in usable]
    return arr


async def _fetch_history(
    session: aiohttp.ClientSession,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
) -> np.ndarray:
    rows: list[list[Any]] = []
    cursor = start_ms
    step = BAR_MS[timeframe]
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "startTime": cursor,
            "endTime": end_ms - 1,
            "limit": 1000,
        }
        async with session.get(BINANCE_KLINES_URL, params=params, timeout=aiohttp.ClientTimeout(total=45)) as response:
            response.raise_for_status()
            batch = await response.json()
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + step
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
        await asyncio.sleep(0.025)
    return _rows_to_array(rows, end_ms=end_ms, timeframe=timeframe)


def _cache_path(cache_root: Path, symbol: str, timeframe: str) -> Path:
    return cache_root / symbol / f"{timeframe}.npz"


def _load_cache(path: Path, *, start_ms: int, end_ms: int, timeframe: str) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as payload:
            requested_start = int(payload["requested_start_ms"])
            requested_end = int(payload["requested_end_ms"])
            if requested_start > start_ms or requested_end < end_ms - BAR_MS[timeframe]:
                return None
            arr = np.zeros(len(payload["t"]), dtype=DTYPE)
            for field in ("t", "o", "h", "l", "c", "v"):
                arr[field] = payload[field]
            return arr[arr["t"] + BAR_MS[timeframe] <= end_ms]
    except Exception:
        return None


def _save_cache(path: Path, data: np.ndarray, *, start_ms: int, end_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        requested_start_ms=np.int64(start_ms),
        requested_end_ms=np.int64(end_ms),
        **{field: data[field] for field in ("t", "o", "h", "l", "c", "v")},
    )


async def _load_symbol_history(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol: str,
    *,
    start_ms: int,
    end_ms: int,
    cache_root: Path,
    refresh: bool,
) -> tuple[str, dict[str, np.ndarray], str | None]:
    out: dict[str, np.ndarray] = {}
    try:
        async with semaphore:
            for timeframe in ("4h", "1d"):
                path = _cache_path(cache_root, symbol, timeframe)
                data = None if refresh else _load_cache(path, start_ms=start_ms, end_ms=end_ms, timeframe=timeframe)
                if data is None:
                    data = await _fetch_history(session, symbol, timeframe, start_ms, end_ms)
                    _save_cache(path, data, start_ms=start_ms, end_ms=end_ms)
                out[timeframe] = data
        return symbol, out, None
    except Exception as exc:
        return symbol, out, f"{type(exc).__name__}: {exc}"


async def load_histories(
    symbols: Sequence[str],
    *,
    start_ms: int,
    end_ms: int,
    cache_root: Path,
    refresh: bool,
    max_concurrency: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    connector = aiohttp.TCPConnector(limit=max(2, max_concurrency * 2))
    semaphore = asyncio.Semaphore(max_concurrency)
    histories: dict[str, dict[str, np.ndarray]] = {}
    errors: dict[str, str] = {}
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _load_symbol_history(
                session,
                semaphore,
                symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                cache_root=cache_root,
                refresh=refresh,
            )
            for symbol in symbols
        ]
        for task in asyncio.as_completed(tasks):
            symbol, data, error = await task
            if error:
                errors[symbol] = error
            elif len(data.get("4h", ())) >= 80 and len(data.get("1d", ())) >= 40:
                histories[symbol] = data
            else:
                errors[symbol] = "insufficient closed 4h/1d history"
    return histories, errors


def label_signal(symbol: str, signal: RegimeStartSignal, four_h: np.ndarray) -> LabeledSignal | None:
    i = signal.bar_index_4h
    horizons = {"3d": 18, "5d": 30, "10d": 60}
    if i + horizons["10d"] >= len(four_h):
        return None
    entry = float(signal.price)
    ret3 = (float(four_h["c"][i + horizons["3d"]]) / entry - 1.0) * 100.0
    ret5 = (float(four_h["c"][i + horizons["5d"]]) / entry - 1.0) * 100.0
    ret10 = (float(four_h["c"][i + horizons["10d"]]) / entry - 1.0) * 100.0
    forward5 = four_h[i + 1:i + horizons["5d"] + 1]
    mfe5 = (float(np.max(forward5["h"])) / entry - 1.0) * 100.0
    mae5 = (float(np.min(forward5["l"])) / entry - 1.0) * 100.0
    return LabeledSignal(
        symbol=symbol,
        decision_ts_ms=signal.decision_ts_ms,
        price=entry,
        ret_3d_pct=ret3,
        ret_5d_pct=ret5,
        ret_10d_pct=ret10,
        mfe_5d_pct=mfe5,
        mae_5d_pct=mae5,
        useful=bool(mfe5 >= 8.0 and mae5 > -8.0),
        signal=signal.to_dict(),
    )


def _metrics(rows: Sequence[LabeledSignal], *, period_start_ms: int, period_end_ms: int) -> dict[str, Any]:
    period_days = max(1.0, (period_end_ms - period_start_ms) / DAY_MS)
    by_day = Counter(datetime.fromtimestamp(row.decision_ts_ms / 1000.0, timezone.utc).date().isoformat() for row in rows)
    if not rows:
        return {
            "signals": 0,
            "useful": 0,
            "useful_precision_pct": 0.0,
            "median_ret_3d_pct": None,
            "median_ret_5d_pct": None,
            "median_ret_10d_pct": None,
            "median_mfe_5d_pct": None,
            "median_mae_5d_pct": None,
            "calendar_days": round(period_days, 2),
            "active_signal_days": 0,
            "signals_per_calendar_day": 0.0,
            "max_signals_single_day": 0,
        }
    useful = sum(1 for row in rows if row.useful)
    return {
        "signals": len(rows),
        "useful": useful,
        "useful_precision_pct": round(100.0 * useful / len(rows), 4),
        "median_ret_3d_pct": round(float(median(row.ret_3d_pct for row in rows)), 4),
        "median_ret_5d_pct": round(float(median(row.ret_5d_pct for row in rows)), 4),
        "median_ret_10d_pct": round(float(median(row.ret_10d_pct for row in rows)), 4),
        "median_mfe_5d_pct": round(float(median(row.mfe_5d_pct for row in rows)), 4),
        "median_mae_5d_pct": round(float(median(row.mae_5d_pct for row in rows)), 4),
        "calendar_days": round(period_days, 2),
        "active_signal_days": len(by_day),
        "signals_per_calendar_day": round(len(rows) / period_days, 4),
        "max_signals_single_day": max(by_day.values(), default=0),
    }


def evaluate_promotion(
    train: dict[str, Any],
    holdout: dict[str, Any],
    *,
    pol_detected_by_deadline: bool,
) -> dict[str, Any]:
    checks = {
        "holdout_signals_at_least_100": int(holdout.get("signals", 0)) >= 100,
        "holdout_calendar_days_at_least_30": float(holdout.get("calendar_days", 0.0)) >= 30.0,
        "holdout_useful_precision_at_least_35pct": float(holdout.get("useful_precision_pct", 0.0)) >= 35.0,
        "holdout_median_ret5_positive": float(holdout.get("median_ret_5d_pct") or 0.0) > 0.0,
        "holdout_median_mfe5_at_least_6pct": float(holdout.get("median_mfe_5d_pct") or 0.0) >= 6.0,
        "holdout_alerts_per_day_at_most_5": float(holdout.get("signals_per_calendar_day", 999.0)) <= 5.0,
        "holdout_precision_drop_at_most_10pp": float(holdout.get("useful_precision_pct", 0.0))
        >= float(train.get("useful_precision_pct", 0.0)) - 10.0,
        "pol_detected_by_2026_07_02_close": bool(pol_detected_by_deadline),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "decision": "eligible_for_watch_only" if passed else "shadow_only",
        "checks": checks,
        "failed_checks": [name for name, ok in checks.items() if not ok],
        "buy_policy_changed": False,
    }


def build_report(
    histories: dict[str, dict[str, np.ndarray]],
    errors: dict[str, str],
    *,
    requested_symbols: Sequence[str],
    start_ms: int,
    end_ms: int,
    profile: RegimeStartProfile,
) -> dict[str, Any]:
    labeled: list[LabeledSignal] = []
    all_signals: list[tuple[str, RegimeStartSignal]] = []
    per_symbol: dict[str, dict[str, Any]] = {}
    for symbol, data in sorted(histories.items()):
        starts = detect_regime_starts(data["4h"], data["1d"], profile)
        labels = [label for signal in starts if (label := label_signal(symbol, signal, data["4h"])) is not None]
        labeled.extend(labels)
        all_signals.extend((symbol, signal) for signal in starts)
        per_symbol[symbol] = {
            "signals": len(starts),
            "labeled": len(labels),
            "useful": sum(1 for row in labels if row.useful),
            "first_signal_utc": _iso(starts[0].decision_ts_ms) if starts else None,
            "latest_signal_utc": _iso(starts[-1].decision_ts_ms) if starts else None,
        }

    split_ms = start_ms + int((end_ms - start_ms) * 0.70)
    train_rows = [row for row in labeled if row.decision_ts_ms < split_ms]
    holdout_rows = [row for row in labeled if row.decision_ts_ms >= split_ms]
    train_metrics = _metrics(train_rows, period_start_ms=start_ms, period_end_ms=split_ms)
    holdout_metrics = _metrics(holdout_rows, period_start_ms=split_ms, period_end_ms=end_ms)

    pol_start_ms = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)
    pol_deadline_ms = int(datetime(2026, 7, 3, tzinfo=timezone.utc).timestamp() * 1000)
    pol_signals = [
        signal for symbol, signal in all_signals
        if symbol == "POLUSDT" and pol_start_ms <= signal.decision_ts_ms <= pol_deadline_ms
    ]
    promotion = evaluate_promotion(
        train_metrics,
        holdout_metrics,
        pol_detected_by_deadline=bool(pol_signals),
    )
    examples = sorted(labeled, key=lambda row: row.decision_ts_ms)[-25:]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "audit": "regime_start_max_period",
        "profile": profile.to_dict(),
        "requested_period": {"start_utc": _iso(start_ms), "end_utc": _iso(end_ms), "split_utc": _iso(split_ms)},
        "coverage": {
            "requested_symbols": len(requested_symbols),
            "valid_symbols": len(histories),
            "failed_symbols": len(errors),
            "valid_symbol_ratio_pct": round(100.0 * len(histories) / max(1, len(requested_symbols)), 4),
            "errors": errors,
        },
        "train": train_metrics,
        "holdout": holdout_metrics,
        "promotion": promotion,
        "pol_case": {
            "detected_by_2026_07_02_close": bool(pol_signals),
            "signals": [signal.to_dict() | {"decision_utc": _iso(signal.decision_ts_ms)} for signal in pol_signals],
        },
        "per_symbol": per_symbol,
        "latest_labeled_examples": [asdict(row) | {"decision_utc": _iso(row.decision_ts_ms)} for row in examples],
    }


def render_text(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    train = report["train"]
    holdout = report["holdout"]
    promotion = report["promotion"]
    lines = [
        "Regime-start maximum-period audit",
        f"Period: {report['requested_period']['start_utc']} .. {report['requested_period']['end_utc']}",
        f"Coverage: {coverage['valid_symbols']}/{coverage['requested_symbols']} symbols ({coverage['valid_symbol_ratio_pct']:.2f}%)",
        (
            "Train: "
            f"signals={train['signals']} precision={train['useful_precision_pct']:.2f}% "
            f"ret5={train['median_ret_5d_pct']}% MFE5={train['median_mfe_5d_pct']}% "
            f"alerts/day={train['signals_per_calendar_day']:.2f}"
        ),
        (
            "Holdout: "
            f"signals={holdout['signals']} precision={holdout['useful_precision_pct']:.2f}% "
            f"ret5={holdout['median_ret_5d_pct']}% MFE5={holdout['median_mfe_5d_pct']}% "
            f"alerts/day={holdout['signals_per_calendar_day']:.2f}"
        ),
        f"POL by deadline: {report['pol_case']['detected_by_2026_07_02_close']}",
        f"Decision: {promotion['decision']}",
    ]
    if promotion["failed_checks"]:
        lines.append("Failed: " + ", ".join(promotion["failed_checks"]))
    return "\n".join(lines) + "\n"


def _symbols(raw: str | None) -> list[str]:
    if raw:
        return sorted({item.strip().upper() for item in raw.split(",") if item.strip()})
    return list(config.load_watchlist())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maximum-period causal audit for multi-day regime-start WATCH events")
    parser.add_argument("--symbols", help="comma-separated override; default is current watchlist")
    parser.add_argument("--start", default=DEFAULT_START.isoformat(), help="UTC ISO start; default is Binance launch")
    parser.add_argument("--end", default=datetime.now(timezone.utc).isoformat(), help="UTC ISO end")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = _symbols(args.symbols)
    start_ms = int(_parse_utc(args.start).timestamp() * 1000)
    end_ms = int(_parse_utc(args.end).timestamp() * 1000)
    if end_ms <= start_ms:
        raise SystemExit("end must be after start")
    histories, errors = asyncio.run(
        load_histories(
            symbols,
            start_ms=start_ms,
            end_ms=end_ms,
            cache_root=args.cache_root,
            refresh=bool(args.refresh),
            max_concurrency=max(1, int(args.max_concurrency)),
        )
    )
    report = build_report(
        histories,
        errors,
        requested_symbols=symbols,
        start_ms=start_ms,
        end_ms=end_ms,
        profile=profile_from_config(config),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    text_output = output.with_suffix(".txt")
    rendered = render_text(report)
    text_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
