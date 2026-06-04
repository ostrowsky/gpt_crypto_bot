from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
REPORT_DIR = WORKSPACE_ROOT / ".runtime" / "reports"
WATCHLIST_FILE = ROOT / "watchlist.json"
DEFAULT_OUTPUT_JSON = REPORT_DIR / "research_universe_impact_latest.json"
DEFAULT_OUTPUT_TXT = REPORT_DIR / "research_universe_impact_latest.txt"


def build_report(
    *,
    reports_dir: Path = REPORT_DIR,
    watchlist_file: Path = WATCHLIST_FILE,
    min_repeats_for_promotion: int = 3,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
    save: bool = True,
) -> dict[str, Any]:
    watchlist = _load_watchlist(watchlist_file)
    daily = []
    outside_symbols: Counter[str] = Counter()
    inside_symbols: Counter[str] = Counter()
    outside_quote_volume: dict[str, list[float]] = {}
    outside_day_change: dict[str, list[float]] = {}

    for path in sorted(reports_dir.glob("top_gainer_critic_*_final.json")):
        payload = _load_json(path)
        day = str(payload.get("target_day_local") or "")
        summary = payload.get("summary") or {}
        exchange_top = payload.get("exchange_top_gainers") or []
        watchlist_top = payload.get("watchlist_top_gainers") or []
        if not day or not exchange_top:
            continue

        exchange_top_count = int(summary.get("exchange_top_count") or len(exchange_top))
        exchange_top_in_watchlist = int(
            summary.get("exchange_top_in_watchlist")
            or sum(1 for row in exchange_top if str(row.get("symbol") or "") in watchlist)
        )
        watchlist_top_count = int(summary.get("watchlist_top_count") or len(watchlist_top))
        bought = int(summary.get("watchlist_top_bought") or 0)
        early = int(summary.get("watchlist_top_early_captured") or 0)
        false_positive = int(summary.get("bot_false_positive_buys") or 0)
        daily.append(
            {
                "day": day,
                "exchange_top_count": exchange_top_count,
                "exchange_top_in_watchlist": exchange_top_in_watchlist,
                "watchlist_top_count": watchlist_top_count,
                "watchlist_top_bought": bought,
                "watchlist_top_early_captured": early,
                "bot_false_positive_buys": false_positive,
            }
        )

        for row in exchange_top:
            symbol = str(row.get("symbol") or "").upper()
            if not symbol:
                continue
            target_counter = inside_symbols if symbol in watchlist else outside_symbols
            target_counter[symbol] += 1
            if symbol not in watchlist:
                outside_quote_volume.setdefault(symbol, []).append(_float(row.get("quote_volume_24h")))
                outside_day_change.setdefault(symbol, []).append(_float(row.get("day_change_pct")))

    totals = _totals(daily)
    promotion_candidates = _promotion_candidates(
        outside_symbols,
        outside_quote_volume,
        outside_day_change,
        min_repeats=min_repeats_for_promotion,
    )
    recommendation = _recommendation(totals, promotion_candidates)
    report = {
        "mode": "research_only",
        "watchlist_file": str(watchlist_file),
        "trade_watchlist_size": len(watchlist),
        "reports_dir": str(reports_dir),
        "days_loaded": len(daily),
        "date_range": {
            "first": daily[0]["day"] if daily else "",
            "last": daily[-1]["day"] if daily else "",
        },
        "summary": totals,
        "positive_label_expansion_factor": _safe_div(
            totals["exchange_top_events"],
            totals["exchange_top_in_watchlist_events"],
        ),
        "top_outside_watchlist_symbols": _counter_rows(outside_symbols, outside_quote_volume, outside_day_change, limit=25),
        "top_inside_watchlist_symbols": _counter_rows(inside_symbols, {}, {}, limit=15),
        "promotion_candidates": promotion_candidates,
        "recommendation": recommendation,
        "guardrails": [
            "do_not_change_live_watchlist",
            "do_not_change_buy_sell_gates",
            "use_research_universe_shadow_only",
            "promotion_requires_replay_and_operator_approval",
        ],
        "daily": daily,
    }
    if save:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        output_txt.write_text(render_text(report), encoding="utf-8")
        report["files"] = {"json": str(output_json), "txt": str(output_txt)}
    return report


def render_text(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    expansion = report.get("positive_label_expansion_factor")
    rec = report.get("recommendation") or {}
    lines = [
        "Research Universe Impact Audit",
        "",
        f"Mode: {report.get('mode', 'research_only')}",
        f"Window: {report.get('date_range', {}).get('first', '')} → {report.get('date_range', {}).get('last', '')} ({report.get('days_loaded', 0)} final days)",
        f"Trade watchlist size: {report.get('trade_watchlist_size', 0)}",
        "",
        "Summary:",
        f"  exchange top events: {summary.get('exchange_top_events', 0)}",
        f"  inside watchlist: {summary.get('exchange_top_in_watchlist_events', 0)} ({_fmt(summary.get('exchange_top_watchlist_coverage_pct'))}%)",
        f"  outside watchlist: {summary.get('exchange_top_outside_watchlist_events', 0)}",
        f"  watchlist capture: {_fmt(summary.get('watchlist_capture_pct'))}%",
        f"  watchlist early capture: {_fmt(summary.get('watchlist_early_capture_pct'))}%",
        f"  exchange-wide diagnostic capture: {_fmt(summary.get('exchange_diagnostic_capture_pct'))}%",
        f"  exchange-wide diagnostic early: {_fmt(summary.get('exchange_diagnostic_early_pct'))}%",
        f"  positive-label expansion factor: {_fmt(expansion, 2)}x",
        "",
        f"Recommendation: {rec.get('decision', 'unknown')}",
        f"Reason: {rec.get('reason', '')}",
        "",
        "Top outside-watchlist exchange movers:",
    ]
    outside = report.get("top_outside_watchlist_symbols") or []
    for idx, row in enumerate(outside[:10], start=1):
        lines.append(
            f"  {idx}. {row['symbol']} repeats={row['count']} "
            f"avg_change={_fmt(row.get('avg_day_change_pct'))}% "
            f"avg_quote_vol={_fmt(row.get('avg_quote_volume_24h'), 0)}"
        )
    candidates = report.get("promotion_candidates") or []
    lines.extend(["", "Promotion candidates for separate replay gate:"])
    if candidates:
        for idx, row in enumerate(candidates[:10], start=1):
            lines.append(f"  {idx}. {row['symbol']} repeats={row['count']} avg_change={_fmt(row.get('avg_day_change_pct'))}%")
    else:
        lines.append("  none")
    lines.extend([
        "",
        "Guardrails:",
        "  - research-only; live watchlist unchanged",
        "  - do not count outside-watchlist misses as live failures",
        "  - do not promote symbols without replay/liquidity/operator gates",
    ])
    return "\n".join(lines)


def _totals(daily: list[dict[str, Any]]) -> dict[str, Any]:
    exchange_top = sum(int(row["exchange_top_count"]) for row in daily)
    inside = sum(int(row["exchange_top_in_watchlist"]) for row in daily)
    watchlist_top = sum(int(row["watchlist_top_count"]) for row in daily)
    bought = sum(int(row["watchlist_top_bought"]) for row in daily)
    early = sum(int(row["watchlist_top_early_captured"]) for row in daily)
    return {
        "exchange_top_events": exchange_top,
        "exchange_top_in_watchlist_events": inside,
        "exchange_top_outside_watchlist_events": max(0, exchange_top - inside),
        "watchlist_top_events": watchlist_top,
        "watchlist_top_bought": bought,
        "watchlist_top_early_captured": early,
        "exchange_top_watchlist_coverage_pct": _safe_pct(inside, exchange_top),
        "watchlist_capture_pct": _safe_pct(bought, watchlist_top),
        "watchlist_early_capture_pct": _safe_pct(early, watchlist_top),
        "exchange_diagnostic_capture_pct": _safe_pct(bought, exchange_top),
        "exchange_diagnostic_early_pct": _safe_pct(early, exchange_top),
        "bot_false_positive_buys": sum(int(row["bot_false_positive_buys"]) for row in daily),
    }


def _promotion_candidates(
    outside_symbols: Counter[str],
    quote_volume: dict[str, list[float]],
    day_change: dict[str, list[float]],
    *,
    min_repeats: int,
) -> list[dict[str, Any]]:
    rows = _counter_rows(outside_symbols, quote_volume, day_change, limit=200)
    return [
        row
        for row in rows
        if int(row["count"]) >= min_repeats
        and _is_promotion_eligible_symbol(str(row.get("symbol") or ""))
    ]


def _counter_rows(
    counter: Counter[str],
    quote_volume: dict[str, list[float]],
    day_change: dict[str, list[float]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows = []
    for symbol, count in counter.most_common(limit):
        rows.append(
            {
                "symbol": symbol,
                "count": int(count),
                "avg_quote_volume_24h": _avg(quote_volume.get(symbol) or []),
                "avg_day_change_pct": _avg(day_change.get(symbol) or []),
            }
        )
    return rows


def _recommendation(totals: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, str]:
    exchange_top = int(totals.get("exchange_top_events") or 0)
    inside = int(totals.get("exchange_top_in_watchlist_events") or 0)
    outside = int(totals.get("exchange_top_outside_watchlist_events") or 0)
    if exchange_top <= 0:
        return {"decision": "insufficient_data", "reason": "no final top-gainer reports loaded"}
    if inside <= 0:
        return {
            "decision": "research_universe_shadow_required",
            "reason": "no exchange top movers are represented in the trade watchlist window",
        }
    if outside >= inside * 2:
        return {
            "decision": "add_research_universe_shadow_layer",
            "reason": (
                f"outside-watchlist top events ({outside}) are materially larger than "
                f"inside-watchlist events ({inside}); candidates={len(candidates)}"
            ),
        }
    return {
        "decision": "keep_trade_watchlist_only_for_now",
        "reason": "outside-watchlist label expansion is not large enough to justify extra collection cost",
    }


def _load_watchlist(path: Path) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return set()
    if isinstance(payload, dict):
        values = payload.get("symbols") or payload.get("watchlist") or []
    else:
        values = payload
    return {str(item).strip().upper().replace("/", "") for item in values if str(item).strip()}


def _is_promotion_eligible_symbol(symbol: str) -> bool:
    symbol = str(symbol or "").strip().upper()
    return symbol.endswith("USDT") and symbol.isascii() and symbol.isalnum()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _float(value: Any) -> float:
    try:
        out = float(value)
        return out if out == out else 0.0
    except Exception:
        return 0.0


def _avg(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def _safe_pct(num: float, den: float) -> float:
    return round((float(num) / float(den) * 100.0), 6) if den else 0.0


def _safe_div(num: float, den: float) -> float | None:
    return round(float(num) / float(den), 6) if den else None


def _fmt(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit impact of adding a Binance research universe.")
    parser.add_argument("--reports-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--watchlist", type=Path, default=WATCHLIST_FILE)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    report = build_report(
        reports_dir=args.reports_dir,
        watchlist_file=args.watchlist,
        min_repeats_for_promotion=args.min_repeats,
        output_json=args.output_json,
        output_txt=args.output_txt,
        save=True,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.as_json else render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
