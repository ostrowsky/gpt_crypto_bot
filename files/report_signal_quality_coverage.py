from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / ".runtime" / "reports"
CACHE_DIR = ROOT / ".runtime" / "signal_quality_cache"
WATCHLIST = ROOT / "files" / "watchlist.json"
DEFAULT_OUTPUT_JSON = REPORTS_DIR / "signal_quality_coverage_latest.json"
DEFAULT_OUTPUT_TXT = REPORTS_DIR / "signal_quality_coverage_latest.txt"
DEFAULT_EXCHANGE_STATUS_CACHE = ROOT / ".runtime" / "exchange_symbol_status.json"
BINANCE_EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo"
EXCHANGE_STATUS_CACHE_SCHEMA = 2
EXCHANGE_STATUS_TTL_HOURS = 6.0
INACTIVE_EXCHANGE_STATUSES = {"BREAK", "HALT", "END_OF_LIFE"}

BAR_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}
DEFAULT_HORIZON_BARS = {"15m": 32, "1h": 24, "4h": 12}


def build_report(
    *,
    signal_report: Path,
    watchlist_path: Path = WATCHLIST,
    cache_dir: Path = CACHE_DIR,
    exchange_status_cache: Path = DEFAULT_EXCHANGE_STATUS_CACHE,
    check_exchange_status: bool = True,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
    save: bool = True,
) -> dict[str, Any]:
    payload = _read_json(signal_report)
    scope = payload.get("scope") or {}
    window = payload.get("window") or {}
    coverage = payload.get("coverage") or {}
    tfs = [str(tf) for tf in scope.get("timeframes") or [] if str(tf) in BAR_MS]
    symbols = _symbols(scope, watchlist_path)
    start_ms = _iso_ms(str(window.get("start") or ""))
    end_ms = _iso_ms(str(window.get("end") or ""))
    expected = _expected_series(symbols, tfs, start_ms, end_ms)
    cache_index = _cache_index(cache_dir)
    rows = []
    for item in expected:
        key = (item["symbol"], item["tf"], item["fetch_start_ms"], item["fetch_end_ms"])
        cache_path = cache_index.get(key)
        status = "loaded" if _cache_has_rows(cache_path) else "missing"
        rows.append({**item, "status": status, "cache_path": str(cache_path) if cache_path else None})
    missing = [row for row in rows if row["status"] != "loaded"]
    exchange_statuses, exchange_status_provenance = _exchange_statuses(
        sorted({row["symbol"] for row in missing}),
        exchange_status_cache,
        enabled=check_exchange_status,
    )
    for row in missing:
        row["exchange_status"] = exchange_statuses.get(row["symbol"], "unknown")
    inactive_missing = [
        row for row in missing
        if str(row.get("exchange_status") or "unknown").upper() in INACTIVE_EXCHANGE_STATUSES
    ]
    active_missing = [row for row in missing if row not in inactive_missing]
    by_tf = {
        tf: {
            "requested": sum(1 for row in rows if row["tf"] == tf),
            "loaded": sum(1 for row in rows if row["tf"] == tf and row["status"] == "loaded"),
            "missing": sum(1 for row in rows if row["tf"] == tf and row["status"] != "loaded"),
        }
        for tf in tfs
    }
    result = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_report": str(signal_report),
        "window": window,
        "coverage": coverage,
        "requested_series": len(rows),
        "loaded_series_from_cache": len(rows) - len(missing),
        "missing_series_count": len(missing),
        "active_requested_series": len(rows) - len(inactive_missing),
        "active_loaded_series": len(rows) - len(missing),
        "active_missing_series_count": len(active_missing),
        "inactive_excluded_series_count": len(inactive_missing),
        "missing_symbol_status_counts": _status_counts(missing),
        "exchange_status_provenance": exchange_status_provenance,
        "by_timeframe": by_tf,
        "missing_series": missing,
        "assessment": _assessment(
            coverage,
            missing,
            statuses_trusted=bool(exchange_status_provenance.get("trusted")),
        ),
    }
    text = render_text(result)
    if save:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        output_txt.write_text(text, encoding="utf-8")
        result["files"] = {"json": str(output_json), "txt": str(output_txt)}
    return result


def render_text(report: dict[str, Any]) -> str:
    lines = [
        "Signal-quality coverage triage",
        f"source: {report.get('source_report')}",
        f"status: {(report.get('coverage') or {}).get('status')} — {report.get('assessment')}",
        f"series: loaded {report.get('loaded_series_from_cache')}/{report.get('requested_series')} missing {report.get('missing_series_count')}",
        f"active series: loaded {report.get('active_loaded_series')}/{report.get('active_requested_series')} missing {report.get('active_missing_series_count')}; inactive excluded {report.get('inactive_excluded_series_count')}",
        f"missing status counts: {report.get('missing_symbol_status_counts') or {}}",
        f"exchange status: {(report.get('exchange_status_provenance') or {}).get('freshness', 'unknown')}",
        "",
        "By timeframe:",
    ]
    for tf, row in (report.get("by_timeframe") or {}).items():
        lines.append(f"  {tf}: loaded {row['loaded']}/{row['requested']} missing {row['missing']}")
    missing = report.get("missing_series") or []
    if missing:
        lines.extend(["", "Missing series:"])
        for row in missing[:60]:
            lines.append(f"  {row['symbol']} {row['tf']} status={row.get('exchange_status', 'unknown')}")
        if len(missing) > 60:
            lines.append(f"  ... +{len(missing) - 60} more")
    return "\n".join(lines) + "\n"


def latest_signal_report(reports_dir: Path) -> Path | None:
    paths = sorted(reports_dir.glob("signal_quality_*_final.json"))
    return paths[-1] if paths else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _symbols(scope: dict[str, Any], watchlist_path: Path) -> list[str]:
    requested = scope.get("requested_symbol_filter")
    if isinstance(requested, list) and requested:
        return sorted({str(x).strip().upper() for x in requested if str(x).strip()})
    try:
        payload = json.loads(watchlist_path.read_text(encoding="utf-8"))
    except Exception:
        payload = []
    return sorted({str(x).strip().upper() for x in payload if str(x).strip()})


def _iso_ms(value: str) -> int:
    if not value:
        return 0
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def _expected_series(symbols: list[str], tfs: list[str], start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    if not symbols or not tfs or not start_ms or not end_ms:
        return []
    max_bar_ms = max(BAR_MS[tf] for tf in tfs)
    fetch_start_ms = start_ms - max_bar_ms * 4
    fetch_end_ms = end_ms + max_bar_ms * max(DEFAULT_HORIZON_BARS.values())
    return [
        {"symbol": sym, "tf": tf, "fetch_start_ms": fetch_start_ms, "fetch_end_ms": fetch_end_ms}
        for sym in symbols
        for tf in sorted(tfs)
    ]


def _cache_index(cache_dir: Path) -> dict[tuple[str, str, int, int], Path]:
    out: dict[tuple[str, str, int, int], Path] = {}
    pattern = re.compile(r"^([A-Z0-9]+)_([^_]+)_(\d+)_(\d+)\.json$")
    for path in cache_dir.glob("*.json") if cache_dir.exists() else []:
        match = pattern.match(path.name)
        if not match:
            continue
        sym, tf, start, end = match.groups()
        out[(sym, tf, int(start), int(end))] = path
    return out


def _cache_has_rows(path: Path | None) -> bool:
    if not path or not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, list) and bool(payload)


def _exchange_statuses(
    symbols: list[str],
    cache_path: Path,
    *,
    enabled: bool,
    now: datetime | None = None,
    ttl_hours: float = EXCHANGE_STATUS_TTL_HOURS,
) -> tuple[dict[str, str], dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    cached, generated_at = _decode_exchange_status_cache(_read_json(cache_path))
    if not symbols:
        return {}, _exchange_status_provenance(
            cache_path=cache_path,
            generated_at=generated_at,
            now=now,
            ttl_hours=ttl_hours,
            enabled=enabled,
            refreshed=False,
            refresh_error="",
        )

    out = {sym: str(cached.get(sym) or "unknown") for sym in symbols}
    fresh = _cache_is_fresh(generated_at, now=now, ttl_hours=ttl_hours)
    needs_refresh = enabled and (not fresh or any(status == "unknown" for status in out.values()))
    refreshed = False
    refresh_error = ""
    if needs_refresh:
        fetched, refresh_error = _fetch_exchange_statuses(symbols)
        if fetched:
            out.update(fetched)
        if fetched and not refresh_error:
            merged = dict(cached)
            merged.update(fetched)
            generated_at = now
            refreshed = True
            payload = {
                "schema_version": EXCHANGE_STATUS_CACHE_SCHEMA,
                "generated_at_utc": now.isoformat().replace("+00:00", "Z"),
                "source": BINANCE_EXCHANGE_INFO,
                "statuses": dict(sorted(merged.items())),
            }
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
                tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
                tmp.replace(cache_path)
            except Exception as exc:
                refresh_error = f"cache_write_failed: {exc}"

    provenance = _exchange_status_provenance(
        cache_path=cache_path,
        generated_at=generated_at,
        now=now,
        ttl_hours=ttl_hours,
        enabled=enabled,
        refreshed=refreshed,
        refresh_error=refresh_error,
    )
    return {sym: out.get(sym, "unknown") for sym in symbols}, provenance


def _decode_exchange_status_cache(payload: dict[str, Any]) -> tuple[dict[str, str], datetime | None]:
    statuses = payload.get("statuses") if isinstance(payload.get("statuses"), dict) else None
    if statuses is None:
        # Legacy caches have no generation timestamp and are always refreshed.
        statuses = payload
        generated_at = None
    else:
        generated_at = _parse_utc(payload.get("generated_at_utc"))
    out = {
        str(symbol).upper(): str(status)
        for symbol, status in statuses.items()
        if isinstance(status, str)
    }
    return out, generated_at


def _parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _cache_is_fresh(generated_at: datetime | None, *, now: datetime, ttl_hours: float) -> bool:
    if generated_at is None:
        return False
    age_hours = (now - generated_at).total_seconds() / 3600.0
    return 0.0 <= age_hours <= ttl_hours


def _exchange_status_provenance(
    *,
    cache_path: Path,
    generated_at: datetime | None,
    now: datetime,
    ttl_hours: float,
    enabled: bool,
    refreshed: bool,
    refresh_error: str,
) -> dict[str, Any]:
    fresh = _cache_is_fresh(generated_at, now=now, ttl_hours=ttl_hours)
    if not enabled:
        freshness = "status_check_disabled"
    elif fresh:
        freshness = "fresh"
    elif refresh_error:
        freshness = "stale_refresh_failed"
    else:
        freshness = "stale"
    return {
        "schema_version": EXCHANGE_STATUS_CACHE_SCHEMA,
        "source": BINANCE_EXCHANGE_INFO,
        "cache_path": str(cache_path),
        "generated_at_utc": generated_at.isoformat().replace("+00:00", "Z") if generated_at else None,
        "ttl_hours": ttl_hours,
        "freshness": freshness,
        "refreshed": refreshed,
        "refresh_error": refresh_error or None,
        "trusted": (not enabled) or fresh,
    }


def _fetch_exchange_statuses(symbols: list[str]) -> tuple[dict[str, str], str]:
    req = urllib.request.Request(BINANCE_EXCHANGE_INFO, headers={"User-Agent": "coverage-triage/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {}, f"exchange_info_failed: {exc}"
    rows = payload.get("symbols") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}, "exchange_info_invalid_payload"
    requested = set(symbols)
    out = {
        str(row.get("symbol")): str(row.get("status") or "unknown")
        for row in rows
        if isinstance(row, dict) and str(row.get("symbol")) in requested
    }
    error = "" if len(out) == len(requested) else "exchange_info_missing_requested_symbols"
    return out, error


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        status = str(row.get("exchange_status") or "unknown")
        out[status] = out.get(status, 0) + 1
    return dict(sorted(out.items()))


def _assessment(
    coverage: dict[str, Any],
    missing: list[dict[str, Any]],
    *,
    statuses_trusted: bool = True,
) -> str:
    if not missing and coverage.get("status") in {"ok", "complete"}:
        return "complete"
    if not missing:
        return "report_marked_partial_but_cache_complete"
    statuses = {str(row.get("exchange_status") or "unknown").upper() for row in missing}
    if statuses_trusted and statuses and statuses <= INACTIVE_EXCHANGE_STATUSES:
        return "partial_safe_inactive_symbols_only"
    if coverage.get("events_loaded", 0) and coverage.get("paired_trades", 0):
        return "metric_affecting_possible: candle series missing while events/trades exist"
    return "missing_candles_without_events: likely safe but incomplete"


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain signal-quality candle coverage gaps.")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--signal-report", type=Path, default=None)
    parser.add_argument("--watchlist", type=Path, default=WATCHLIST)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--exchange-status-cache", type=Path, default=DEFAULT_EXCHANGE_STATUS_CACHE)
    parser.add_argument("--no-exchange-status", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    signal_report = args.signal_report or latest_signal_report(args.reports_dir)
    if not signal_report:
        raise SystemExit(f"no signal_quality_*_final.json found in {args.reports_dir}")
    report = build_report(
        signal_report=signal_report,
        watchlist_path=args.watchlist,
        cache_dir=args.cache_dir,
        exchange_status_cache=args.exchange_status_cache,
        check_exchange_status=not args.no_exchange_status,
        output_json=args.output,
        output_txt=args.text_output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_text(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
