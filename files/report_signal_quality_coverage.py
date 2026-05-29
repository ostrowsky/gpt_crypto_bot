from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / ".runtime" / "reports"
CACHE_DIR = ROOT / ".runtime" / "signal_quality_cache"
WATCHLIST = ROOT / "files" / "watchlist.json"
DEFAULT_OUTPUT_JSON = REPORTS_DIR / "signal_quality_coverage_latest.json"
DEFAULT_OUTPUT_TXT = REPORTS_DIR / "signal_quality_coverage_latest.txt"

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
        "by_timeframe": by_tf,
        "missing_series": missing,
        "assessment": _assessment(coverage, missing),
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
        "",
        "By timeframe:",
    ]
    for tf, row in (report.get("by_timeframe") or {}).items():
        lines.append(f"  {tf}: loaded {row['loaded']}/{row['requested']} missing {row['missing']}")
    missing = report.get("missing_series") or []
    if missing:
        lines.extend(["", "Missing series:"])
        for row in missing[:60]:
            lines.append(f"  {row['symbol']} {row['tf']}")
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


def _assessment(coverage: dict[str, Any], missing: list[dict[str, Any]]) -> str:
    if not missing and coverage.get("status") in {"ok", "complete"}:
        return "complete"
    if not missing:
        return "report_marked_partial_but_cache_complete"
    if coverage.get("events_loaded", 0) and coverage.get("paired_trades", 0):
        return "metric_affecting_possible: candle series missing while events/trades exist"
    return "missing_candles_without_events: likely safe but incomplete"


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain signal-quality candle coverage gaps.")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--signal-report", type=Path, default=None)
    parser.add_argument("--watchlist", type=Path, default=WATCHLIST)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
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
        output_json=args.output,
        output_txt=args.text_output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_text(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
