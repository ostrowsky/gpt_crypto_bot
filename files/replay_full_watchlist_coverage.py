"""Structural maximum-period replay for full-watchlist poll coverage.

This replay proves scheduler coverage and bounded delay. It deliberately does
not claim causal PnL/recall uplift because historical shortlist membership was
not persisted and therefore cannot be reconstructed honestly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import config
from monitor import _select_poll_coins
from strategy import CoinReport


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / ".runtime" / "reports"
RUNTIME_WATCHLIST = ROOT / "files" / "watchlist.json"


def _report(symbol: str) -> CoinReport:
    return CoinReport(
        symbol=symbol,
        tf="15m",
        today_signals=0,
        today_accuracy={},
        today_confirmed=False,
        best_horizon=0,
        best_accuracy=0.0,
        in_play=False,
    )


def _canonical_reports(reports_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(reports_dir.glob("top_gainer_critic_????-??-??_final.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("phase") == "final":
            rows.append((path, payload))
    return rows


def _runtime_watchlist() -> list[str]:
    try:
        raw = json.loads(RUNTIME_WATCHLIST.read_text(encoding="utf-8-sig"))
    except Exception:
        raw = list(config.DEFAULT_WATCHLIST)
    result: list[str] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        symbol = str(item or "").strip().upper()
        if symbol and symbol not in seen:
            result.append(symbol)
            seen.add(symbol)
    return result


def _simulate(population: list[str], cap: int, held: set[str] | None = None) -> dict[str, Any]:
    coins = [_report(symbol) for symbol in population]
    held = held or set()
    cursor = 0
    seen: set[str] = set()
    max_cycles = max(1, math.ceil(max(1, len(coins) - len(held)) / max(1, cap - len(held))))
    held_every_cycle = True
    cycle_sizes: list[int] = []
    for _ in range(max_cycles):
        selected, cursor = _select_poll_coins(coins, held, cap, cursor)
        selected_symbols = {row.symbol for row in selected}
        held_every_cycle = held_every_cycle and held.issubset(selected_symbols)
        seen.update(selected_symbols)
        cycle_sizes.append(len(selected))
    return {
        "population": len(coins),
        "cycles": max_cycles,
        "seen": sorted(seen),
        "missing": sorted(set(population) - seen),
        "held_every_cycle": held_every_cycle,
        "max_cycle_size": max(cycle_sizes, default=0),
    }


def build(reports_dir: Path = REPORTS, cap: int | None = None) -> dict[str, Any]:
    cap = int(cap if cap is not None else getattr(config, "MAX_POLL_PER_CYCLE", 45))
    current_watchlist = _runtime_watchlist()
    reports = _canonical_reports(reports_dir)
    day_rows: list[dict[str, Any]] = []
    top_total = 0
    top_scheduled = 0
    fallback_days = 0
    for path, payload in reports:
        top_symbols = [
            str(row.get("symbol") or "")
            for row in (payload.get("watchlist_top_gainers") or [])
            if row.get("symbol")
        ]
        historical_population = [
            str(row.get("symbol") or "")
            for row in (payload.get("watchlist_universe_top_gainers") or [])
            if row.get("symbol")
        ]
        if historical_population:
            # The critic stores only a bounded ranking, not the immutable full
            # watchlist snapshot. Union with current watchlist is conservative
            # for load/delay and explicit about survivorship provenance.
            population = list(dict.fromkeys([*current_watchlist, *historical_population, *top_symbols]))
            provenance = "current_watchlist_plus_historical_ranked_members"
        else:
            population = list(dict.fromkeys([*current_watchlist, *top_symbols]))
            provenance = "current_watchlist_fallback"
            fallback_days += 1
        simulation = _simulate(population, cap)
        scheduled = sorted(set(top_symbols) & set(simulation["seen"]))
        top_total += len(top_symbols)
        top_scheduled += len(scheduled)
        day_rows.append({
            "day": payload.get("target_day_local"),
            "source": str(path),
            "population_provenance": provenance,
            "population": simulation["population"],
            "cycles_for_full_sweep": simulation["cycles"],
            "top_symbols": top_symbols,
            "top_scheduled": scheduled,
            "top_missing": sorted(set(top_symbols) - set(scheduled)),
        })

    current_simulation = _simulate(current_watchlist, cap, set(current_watchlist[:3]))
    poll_sec = int(getattr(config, "POLL_SEC", 60))
    status = "pass" if reports and top_scheduled == top_total and not current_simulation["missing"] else "fail"
    return {
        "schema_version": 1,
        "status": status,
        "claim_scope": "structural_scheduler_coverage_only",
        "causal_recall_uplift": None,
        "causal_recall_uplift_reason": "historical shortlist membership was not persisted",
        "period": {
            "first_day": day_rows[0]["day"] if day_rows else None,
            "last_day": day_rows[-1]["day"] if day_rows else None,
            "days": len(day_rows),
            "maximum_local_canonical_period": True,
        },
        "settings": {"cap": cap, "poll_sec": poll_sec},
        "current_watchlist": {
            "symbols": len(current_watchlist),
            "cycles_for_full_sweep": current_simulation["cycles"],
            "max_sweep_delay_sec": current_simulation["cycles"] * poll_sec,
            "missing": current_simulation["missing"],
            "held_every_cycle": current_simulation["held_every_cycle"],
            "max_cycle_size": current_simulation["max_cycle_size"],
        },
        "historical_top_coverage": {
            "numerator": top_scheduled,
            "denominator": top_total,
            "rate_pct": round(top_scheduled / top_total * 100.0, 2) if top_total else None,
            "fallback_days": fallback_days,
        },
        "days": day_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--cap", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build(args.reports_dir, args.cap)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
