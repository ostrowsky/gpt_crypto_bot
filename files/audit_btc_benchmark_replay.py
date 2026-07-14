from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import aiohttp

import config
from indicators import compute_features
from replay_backtest import (
    ReplayRunStats,
    ReplayTrade,
    _build_bull_day_context,
    build_replay_candidate_snapshot,
    fetch_klines,
    simulate_portfolio,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / ".runtime" / "reports" / "btc_benchmark_replay_latest.json"
BASELINE_VARIANT = "score_replace_cluster"
RESEARCH_VARIANTS = (
    "btc_cluster_exempt",
    "btc_1h_leader_admission",
    "btc_benchmark_combined",
    "btc_benchmark_rotation",
)
DEFAULT_DAYS = 44
DEFAULT_COST_BPS = 20.0
DEFAULT_HOLDOUT_FRACTION = 0.30


def _period_metrics(
    trades: Sequence[ReplayTrade],
    *,
    start_ms: int,
    end_ms: int,
    cost_bps: float,
) -> dict[str, Any]:
    selected = [trade for trade in trades if start_ms <= trade.entry_ts < end_ms]
    cost_pct = float(cost_bps) / 100.0
    returns = [float(trade.pnl_pct) - cost_pct for trade in selected]
    btc_returns = [
        float(trade.pnl_pct) - cost_pct
        for trade in selected
        if trade.sym.upper() == "BTCUSDT"
    ]
    return {
        "trades": len(selected),
        "net_pnl_pct": round(sum(returns), 4),
        "net_avg_pct": round(sum(returns) / len(returns), 4) if returns else 0.0,
        "net_win_rate": round(sum(value > 0 for value in returns) / len(returns), 6) if returns else 0.0,
        "btc_trades": len(btc_returns),
        "btc_net_pnl_pct": round(sum(btc_returns), 4),
        "btc_net_avg_pct": round(sum(btc_returns) / len(btc_returns), 4) if btc_returns else 0.0,
        "btc_net_win_rate": (
            round(sum(value > 0 for value in btc_returns) / len(btc_returns), 6)
            if btc_returns
            else 0.0
        ),
    }


def _period_delta(variant: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "trades": int(variant["trades"]) - int(baseline["trades"]),
        "net_pnl_pct": round(float(variant["net_pnl_pct"]) - float(baseline["net_pnl_pct"]), 4),
        "net_avg_pct": round(float(variant["net_avg_pct"]) - float(baseline["net_avg_pct"]), 4),
        "net_win_rate": round(float(variant["net_win_rate"]) - float(baseline["net_win_rate"]), 6),
        "btc_trades": int(variant["btc_trades"]) - int(baseline["btc_trades"]),
        "btc_net_pnl_pct": round(
            float(variant["btc_net_pnl_pct"]) - float(baseline["btc_net_pnl_pct"]),
            4,
        ),
    }


def _run_stats_payload(stats: ReplayRunStats) -> dict[str, int | float]:
    return {
        "candidates_total": stats.candidates_total,
        "skipped_portfolio_full": stats.skipped_portfolio_full,
        "skipped_top_gainer_score": stats.skipped_top_gainer_score,
        "skipped_cluster_cap": stats.skipped_cluster_cap,
        "replacements_total": stats.replacements_total,
        "btc_cluster_exemption_admitted": stats.btc_cluster_exemption_admitted,
        "btc_1h_leader_admitted": stats.btc_1h_leader_admitted,
        "btc_benchmark_policy_admitted": stats.btc_benchmark_policy_admitted,
        "btc_benchmark_replacements": stats.btc_benchmark_replacements,
    }


def _policy_admissions(variant: str, stats: dict[str, Any]) -> int:
    if variant == "btc_cluster_exempt":
        return int(stats.get("btc_cluster_exemption_admitted", 0))
    if variant == "btc_1h_leader_admission":
        return int(stats.get("btc_1h_leader_admitted", 0))
    if variant == "btc_benchmark_rotation":
        return int(stats.get("btc_benchmark_replacements", 0))
    return int(stats.get("btc_benchmark_policy_admitted", 0))


def _decision(
    *,
    variant: str,
    run_stats: dict[str, Any],
    all_metrics: dict[str, Any],
    holdout_metrics: dict[str, Any],
    holdout_delta: dict[str, Any],
) -> dict[str, Any]:
    admissions = _policy_admissions(variant, run_stats)
    gates = {
        "at_least_5_policy_admissions": admissions >= 5,
        "holdout_net_pnl_not_worse": float(holdout_delta["net_pnl_pct"]) >= 0.0,
        "holdout_net_avg_not_worse": float(holdout_delta["net_avg_pct"]) >= 0.0,
        "holdout_win_rate_within_1pp": float(holdout_delta["net_win_rate"]) >= -0.01,
        "whole_window_btc_net_positive": float(all_metrics["btc_net_pnl_pct"]) > 0.0,
        "holdout_has_trades": int(holdout_metrics["trades"]) > 0,
    }
    if not gates["at_least_5_policy_admissions"]:
        status = "insufficient_evidence"
    elif all(gates.values()):
        status = "advance_to_shadow_review"
    else:
        status = "rejected"
    return {
        "status": status,
        "policy_admissions": admissions,
        "gates": gates,
    }


async def _load_cache(
    symbols: Sequence[str],
    timeframes: Sequence[str],
    *,
    start_ms: int,
    end_ms: int,
    max_concurrency: int,
) -> tuple[dict, dict[str, list[str]]]:
    requested_timeframes = sorted(set(timeframes) | {"15m", "4h"})
    fetch_symbols = list(dict.fromkeys([*symbols, "BTCUSDT"]))
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
    cache: dict = {}
    missing: dict[str, list[str]] = {}

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=max_concurrency * 2)) as session:
        async def load_one(symbol: str, timeframe: str):
            async with semaphore:
                data = await fetch_klines(session, symbol, timeframe, start_ms, end_ms)
            return symbol, timeframe, data

        tasks = [load_one(symbol, timeframe) for symbol in fetch_symbols for timeframe in requested_timeframes]
        completed = 0
        for task in asyncio.as_completed(tasks):
            symbol, timeframe, data = await task
            completed += 1
            if data is None:
                missing.setdefault(symbol, []).append(timeframe)
            else:
                cache[(symbol, timeframe)] = (
                    data,
                    compute_features(
                        data["o"],
                        data["h"],
                        data["l"],
                        data["c"].astype(float),
                        data["v"],
                    ),
                )
            if completed % 50 == 0 or completed == len(tasks):
                print(f"history {completed}/{len(tasks)}", file=sys.stderr, flush=True)
    return cache, missing


async def build_audit(
    symbols: Sequence[str],
    *,
    days: int = DEFAULT_DAYS,
    timeframes: Sequence[str] = ("15m", "1h"),
    max_open_positions: int | None = None,
    replace_min_delta: float | None = None,
    top_gainer_score_min: float = 34.0,
    cost_bps: float = DEFAULT_COST_BPS,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    max_concurrency: int = 4,
    output: Path | None = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(days))
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    split_ms = int(start_ms + (end_ms - start_ms) * (1.0 - float(holdout_fraction)))
    max_open_positions = int(max_open_positions or getattr(config, "MAX_OPEN_POSITIONS", 10))
    replace_min_delta = float(
        replace_min_delta
        if replace_min_delta is not None
        else getattr(config, "PORTFOLIO_REPLACE_MIN_DELTA", 8.0)
    )

    cache, missing = await _load_cache(
        symbols,
        timeframes,
        start_ms=start_ms,
        end_ms=end_ms,
        max_concurrency=max_concurrency,
    )
    cache_15m = {symbol: cache[(symbol, "15m")] for symbol in symbols if (symbol, "15m") in cache}
    cache_4h = {symbol: cache[(symbol, "4h")] for symbol in symbols if (symbol, "4h") in cache}
    btc_1h = cache.get(("BTCUSDT", "1h"))
    market_ctx = _build_bull_day_context(btc_1h[0] if btc_1h else None)
    print("build shared causal candidate snapshot", file=sys.stderr, flush=True)
    candidate_snapshot = await build_replay_candidate_snapshot(
        list(symbols),
        list(timeframes),
        cache,
        cache_15m,
        cache_4h,
        market_ctx,
        variant=BASELINE_VARIANT,
    )

    variants: dict[str, Any] = {}
    for variant in (BASELINE_VARIANT, *RESEARCH_VARIANTS):
        print(f"simulate {variant}", file=sys.stderr, flush=True)
        trades, stats = await simulate_portfolio(
            list(symbols),
            list(timeframes),
            cache,
            cache_15m,
            cache_4h,
            market_ctx,
            max_open_positions=max_open_positions,
            enable_replacement=True,
            replace_min_delta=replace_min_delta,
            variant=variant,
            top_gainer_score_min=top_gainer_score_min,
            candidate_snapshot=candidate_snapshot,
        )
        variants[variant] = {
            "all": _period_metrics(trades, start_ms=start_ms, end_ms=end_ms, cost_bps=cost_bps),
            "train": _period_metrics(trades, start_ms=start_ms, end_ms=split_ms, cost_bps=cost_bps),
            "holdout": _period_metrics(trades, start_ms=split_ms, end_ms=end_ms, cost_bps=cost_bps),
            "run_stats": _run_stats_payload(stats),
        }

    baseline = variants[BASELINE_VARIANT]
    for variant in RESEARCH_VARIANTS:
        row = variants[variant]
        row["delta_vs_current"] = {
            period: _period_delta(row[period], baseline[period])
            for period in ("all", "train", "holdout")
        }
        row["decision"] = _decision(
            variant=variant,
            run_stats=row["run_stats"],
            all_metrics=row["all"],
            holdout_metrics=row["holdout"],
            holdout_delta=row["delta_vs_current"]["holdout"],
        )

    combined_gates = variants["btc_benchmark_combined"]["decision"]["gates"]
    combined_non_harmful = all(
        combined_gates[name]
        for name in (
            "holdout_net_pnl_not_worse",
            "holdout_net_avg_not_worse",
            "holdout_win_rate_within_1pp",
        )
    )
    if not combined_non_harmful:
        for variant in ("btc_cluster_exempt", "btc_1h_leader_admission"):
            if variants[variant]["decision"]["status"] == "advance_to_shadow_review":
                variants[variant]["decision"]["status"] = "blocked_by_combined_interaction_gate"
                variants[variant]["decision"]["gates"]["combined_variant_non_harmful"] = False

    loaded_by_tf = {
        timeframe: sum((symbol, timeframe) in cache for symbol in symbols)
        for timeframe in sorted(set(timeframes) | {"15m", "4h"})
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "research_only_production_unchanged",
        "window": {
            "days": int(days),
            "start_utc": start.isoformat().replace("+00:00", "Z"),
            "split_utc": datetime.fromtimestamp(split_ms / 1000.0, timezone.utc).isoformat().replace("+00:00", "Z"),
            "end_utc": end.isoformat().replace("+00:00", "Z"),
            "holdout_fraction": float(holdout_fraction),
        },
        "universe": {
            "symbols_requested": len(symbols),
            "symbols": list(symbols),
            "timeframes": list(timeframes),
            "loaded_by_timeframe": loaded_by_tf,
            "missing": missing,
        },
        "policy": {
            "current_variant": BASELINE_VARIANT,
            "research_variants": list(RESEARCH_VARIANTS),
            "max_open_positions": max_open_positions,
            "replace_min_delta": replace_min_delta,
            "top_gainer_score_min": float(top_gainer_score_min),
            "round_trip_cost_bps": float(cost_bps),
            "btc_1h_frozen_profile": {
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "mode": "breakout",
                "top_gainer_score_min": 32.0,
                "candidate_score_min": 120.0,
                "rsi_min": 55.0,
                "rsi_max": 70.0,
                "adx_min": 20.0,
                "vol_x_min": 2.0,
                "intraday_change_pct_min": 1.0,
                "daily_range_max": 5.0,
            },
            "btc_rotation_frozen_profile": {
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "mode": "breakout",
                "top_gainer_score_min": 32.0,
                "candidate_score_min": 120.0,
                "rsi_max": 72.0,
                "adx_min": 18.0,
                "vol_x_min": 2.0,
                "intraday_change_pct_min": 1.0,
                "daily_range_max": 5.0,
                "replaced_position_pnl_max": -0.25,
                "replaced_position_min_bars": 2,
            },
        },
        "variants": variants,
        "production_decision": "no_change",
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def render_text(payload: dict[str, Any]) -> str:
    window = payload["window"]
    lines = [
        "BTC benchmark admission replay",
        f"Window: {window['start_utc']} .. {window['end_utc']} (split {window['split_utc']})",
        (
            f"Universe: {payload['universe']['symbols_requested']} symbols, "
            f"loaded={payload['universe']['loaded_by_timeframe']}"
        ),
        f"Costs: {payload['policy']['round_trip_cost_bps']:.1f} bps round trip",
    ]
    baseline = payload["variants"][BASELINE_VARIANT]
    lines.append(
        f"Current: all net={baseline['all']['net_pnl_pct']:+.2f}% "
        f"holdout net={baseline['holdout']['net_pnl_pct']:+.2f}%"
    )
    for variant in RESEARCH_VARIANTS:
        row = payload["variants"][variant]
        delta = row["delta_vs_current"]["holdout"]
        lines.append(
            f"{variant}: decision={row['decision']['status']} "
            f"admissions={row['decision']['policy_admissions']} "
            f"holdout_net_delta={delta['net_pnl_pct']:+.2f}% "
            f"holdout_avg_delta={delta['net_avg_pct']:+.4f}% "
            f"btc_net={row['all']['btc_net_pnl_pct']:+.2f}%"
        )
    lines.append("Production: unchanged")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay BTC benchmark admission hypotheses")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframes", nargs="*", default=["15m", "1h"])
    parser.add_argument("--max-open-positions", type=int, default=getattr(config, "MAX_OPEN_POSITIONS", 10))
    parser.add_argument("--replace-min-delta", type=float, default=getattr(config, "PORTFOLIO_REPLACE_MIN_DELTA", 8.0))
    parser.add_argument("--top-gainer-score-min", type=float, default=34.0)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = args.symbols or config.load_watchlist()
    payload = asyncio.run(
        build_audit(
            symbols,
            days=args.days,
            timeframes=args.timeframes,
            max_open_positions=args.max_open_positions,
            replace_min_delta=args.replace_min_delta,
            top_gainer_score_min=args.top_gainer_score_min,
            cost_bps=args.cost_bps,
            holdout_fraction=args.holdout_fraction,
            max_concurrency=args.max_concurrency,
            output=args.output,
        )
    )
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
