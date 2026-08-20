"""Canonical capital-weighted portfolio alpha for causal replay trades."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from policy_provenance import current_policy_manifest, stable_hash


SCHEMA_VERSION = 1
METRIC_CONTRACT = "canonical_unified_ten_slot_alpha_v1"
BENCHMARK_NAME = "BTCUSDT_buy_and_hold_same_closed_bar_window"
MAX_REPLAY_DAYS = 30
MIN_PERIOD_COVERAGE = 0.95


def evaluator_source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _stream_hash(rows: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(repr(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _trade_stream_hash(trades: Sequence[Any]) -> str:
    fields = (
        "sym", "tf", "mode", "entry_ts", "entry_price", "exit_ts", "exit_price",
        "partial_exit_taken", "partial_exit_fraction", "partial_exit_ts", "partial_exit_price",
    )
    rows = [tuple(_trade_value(trade, field, None) for field in fields) for trade in trades]
    return _stream_hash(rows)


def _price_stream_hash(
    price_series_by_symbol: Mapping[str, Sequence[tuple[int, float]]],
    benchmark_series: Sequence[tuple[int, float]],
) -> str:
    digest = hashlib.sha256()
    for symbol in sorted(price_series_by_symbol):
        digest.update(str(symbol).upper().encode("utf-8"))
        digest.update(b"\n")
        for ts_ms, price in price_series_by_symbol[symbol]:
            digest.update(f"{int(ts_ms)}:{float(price):.16g}\n".encode("utf-8"))
    digest.update(b"BENCHMARK\n")
    for ts_ms, price in benchmark_series:
        digest.update(f"{int(ts_ms)}:{float(price):.16g}\n".encode("utf-8"))
    return digest.hexdigest()


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def closed_price_series(
    data: Any,
    *,
    bar_ms: int,
    start_ms: int,
    end_ms: int,
) -> list[tuple[int, float]]:
    """Return only prices whose candle close was observable inside the window."""
    if data is None:
        return []
    out: list[tuple[int, float]] = []
    try:
        timestamps = data["t"]
        closes = data["c"]
    except (KeyError, TypeError, ValueError):
        return out
    for raw_ts, raw_close in zip(timestamps, closes):
        close_ts = int(raw_ts) + int(bar_ms)
        price = float(raw_close)
        if int(start_ms) <= close_ts <= int(end_ms) and price > 0:
            out.append((close_ts, price))
    return out


@dataclass
class _Position:
    trade_id: int
    symbol: str
    qty: float


@dataclass
class _AccountResult:
    ending_equity: float
    return_pct: float
    fees_quote: float
    slippage_quote: float
    max_drawdown_pct: float
    max_concurrent_positions: int
    average_gross_utilization: float
    valuation_points: int
    fully_valued_points: int
    violations: list[str]


def _trade_value(trade: Any, name: str, default: Any = None) -> Any:
    if isinstance(trade, Mapping):
        return trade.get(name, default)
    return getattr(trade, name, default)


def _normalized_series(rows: Sequence[tuple[int, float]]) -> tuple[list[int], list[float]]:
    latest_by_ts: dict[int, float] = {}
    for raw_ts, raw_price in rows:
        ts = int(raw_ts)
        price = float(raw_price)
        if price > 0:
            latest_by_ts[ts] = price
    ordered = sorted(latest_by_ts.items())
    return [row[0] for row in ordered], [row[1] for row in ordered]


def _price_at_or_before(
    normalized: tuple[list[int], list[float]],
    ts_ms: int,
) -> float | None:
    timestamps, prices = normalized
    idx = bisect_right(timestamps, int(ts_ms)) - 1
    return prices[idx] if idx >= 0 else None


def _max_drawdown(values: Sequence[float]) -> float:
    peak = 0.0
    worst = 0.0
    for value in values:
        if value > peak:
            peak = value
        if peak > 0:
            worst = max(worst, (peak - value) / peak * 100.0)
    return worst


def _simulate_account(
    trades: Sequence[Any],
    *,
    price_series_by_symbol: Mapping[str, Sequence[tuple[int, float]]],
    valuation_timestamps: Sequence[int],
    capacity: int,
    initial_capital: float,
    fee_bps: float,
    slippage_bps: float,
) -> _AccountResult:
    fee_rate = max(0.0, float(fee_bps)) / 10_000.0
    slippage_rate = max(0.0, float(slippage_bps)) / 10_000.0
    cash = float(initial_capital)
    positions: dict[int, _Position] = {}
    symbol_to_trade: dict[str, int] = {}
    fees_quote = 0.0
    slippage_quote = 0.0
    violations: list[str] = []
    max_positions = 0
    utilization_sum = 0.0
    equity_curve: list[float] = []
    fully_valued_points = 0
    normalized = {
        str(symbol).upper(): _normalized_series(rows)
        for symbol, rows in price_series_by_symbol.items()
    }

    events: dict[int, list[tuple[int, int, str, float]]] = {}
    for trade_id, trade in enumerate(trades):
        entry_ts = int(_trade_value(trade, "entry_ts", 0) or 0)
        exit_ts = int(_trade_value(trade, "exit_ts", 0) or 0)
        entry_price = float(_trade_value(trade, "entry_price", 0.0) or 0.0)
        exit_price = float(_trade_value(trade, "exit_price", 0.0) or 0.0)
        symbol = str(_trade_value(trade, "sym", _trade_value(trade, "symbol", "")) or "").upper()
        if not symbol or entry_ts <= 0 or exit_ts < entry_ts or entry_price <= 0 or exit_price <= 0:
            violations.append(f"invalid_trade:{trade_id}")
            continue
        events.setdefault(entry_ts, []).append((2, trade_id, "entry", entry_price))
        partial_fraction = float(_trade_value(trade, "partial_exit_fraction", 0.0) or 0.0)
        partial_ts = int(_trade_value(trade, "partial_exit_ts", 0) or 0)
        partial_price = float(_trade_value(trade, "partial_exit_price", 0.0) or 0.0)
        partial_taken = bool(_trade_value(trade, "partial_exit_taken", False))
        if partial_taken and 0.0 < partial_fraction < 1.0 and entry_ts <= partial_ts <= exit_ts and partial_price > 0:
            events.setdefault(partial_ts, []).append((0, trade_id, "partial", partial_price))
        # Preserve the normal portfolio ordering (older exits before new entries),
        # but a boundary-liquidated trade must exist before its own same-timestamp
        # exit is booked. Otherwise the later entry becomes a phantom open position.
        exit_priority = 3 if exit_ts == entry_ts else 1
        events.setdefault(exit_ts, []).append((exit_priority, trade_id, "exit", exit_price))

    def liquidation_equity(ts_ms: int) -> tuple[float, float, bool]:
        equity = cash
        gross_exposure = 0.0
        complete = True
        for position in positions.values():
            price = _price_at_or_before(normalized.get(position.symbol, ([], [])), ts_ms)
            if price is None:
                complete = False
                continue
            gross_value = position.qty * price
            gross_exposure += gross_value
            exit_fill = price * (1.0 - slippage_rate)
            equity += position.qty * exit_fill * (1.0 - fee_rate)
        return equity, gross_exposure, complete

    valuation_set = {int(ts) for ts in valuation_timestamps}
    all_timestamps = sorted(valuation_set | set(events))
    for ts_ms in all_timestamps:
        # Partial/final exits are observable before admissions at the same close.
        for _, trade_id, event_type, raw_price in sorted(events.get(ts_ms, [])):
            trade = trades[trade_id]
            symbol = str(_trade_value(trade, "sym", _trade_value(trade, "symbol", "")) or "").upper()
            if event_type == "entry":
                if symbol in symbol_to_trade:
                    violations.append(f"duplicate_symbol:{symbol}:{ts_ms}")
                    continue
                if len(positions) >= int(capacity):
                    violations.append(f"capacity_exceeded:{ts_ms}")
                    continue
                equity, _, complete = liquidation_equity(ts_ms)
                if not complete:
                    violations.append(f"missing_mark_before_entry:{symbol}:{ts_ms}")
                    continue
                budget = min(cash, max(0.0, equity / float(capacity)))
                if budget <= 0:
                    violations.append(f"no_cash:{symbol}:{ts_ms}")
                    continue
                notional = budget / (1.0 + fee_rate)
                fill_price = float(raw_price) * (1.0 + slippage_rate)
                qty = notional / fill_price
                fee = notional * fee_rate
                slip = qty * (fill_price - float(raw_price))
                cash -= notional + fee
                fees_quote += fee
                slippage_quote += slip
                positions[trade_id] = _Position(trade_id=trade_id, symbol=symbol, qty=qty)
                symbol_to_trade[symbol] = trade_id
                max_positions = max(max_positions, len(positions))
                continue

            position = positions.get(trade_id)
            if position is None:
                # An entry rejected by a contract violation has no exit to book.
                continue
            fraction = 1.0
            if event_type == "partial":
                fraction = max(0.0, min(1.0, float(_trade_value(trade, "partial_exit_fraction", 0.0) or 0.0)))
            qty = position.qty * fraction
            fill_price = float(raw_price) * (1.0 - slippage_rate)
            proceeds = qty * fill_price
            fee = proceeds * fee_rate
            slip = qty * (float(raw_price) - fill_price)
            cash += proceeds - fee
            fees_quote += fee
            slippage_quote += slip
            position.qty -= qty
            if event_type == "exit" or position.qty <= 1e-15:
                positions.pop(trade_id, None)
                symbol_to_trade.pop(position.symbol, None)

        if ts_ms in valuation_set:
            equity, gross_exposure, complete = liquidation_equity(ts_ms)
            equity_curve.append(equity)
            if complete:
                fully_valued_points += 1
            utilization_sum += gross_exposure / equity if equity > 0 else 0.0

    final_ts = all_timestamps[-1] if all_timestamps else 0
    ending_equity, _, complete = liquidation_equity(final_ts)
    if positions:
        violations.append(f"open_positions_at_end:{len(positions)}")
    if not complete:
        violations.append("missing_final_mark")
    if not equity_curve:
        equity_curve = [float(initial_capital), ending_equity]
    return _AccountResult(
        ending_equity=ending_equity,
        return_pct=(ending_equity / float(initial_capital) - 1.0) * 100.0,
        fees_quote=fees_quote,
        slippage_quote=slippage_quote,
        max_drawdown_pct=_max_drawdown(equity_curve),
        max_concurrent_positions=max_positions,
        average_gross_utilization=(utilization_sum / len(valuation_timestamps)) if valuation_timestamps else 0.0,
        valuation_points=len(valuation_timestamps),
        fully_valued_points=fully_valued_points,
        violations=violations,
    )


def _benchmark_result(
    series: Sequence[tuple[int, float]],
    *,
    initial_capital: float,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    timestamps, prices = _normalized_series(series)
    if len(prices) < 2:
        return {"name": BENCHMARK_NAME, "status": "insufficient_data"}
    fee_rate = max(0.0, float(fee_bps)) / 10_000.0
    slippage_rate = max(0.0, float(slippage_bps)) / 10_000.0
    entry_fill = prices[0] * (1.0 + slippage_rate)
    entry_notional = float(initial_capital) / (1.0 + fee_rate)
    entry_fee = entry_notional * fee_rate
    qty = entry_notional / entry_fill
    exit_fill = prices[-1] * (1.0 - slippage_rate)
    exit_proceeds = qty * exit_fill
    exit_fee = exit_proceeds * fee_rate
    ending = exit_proceeds - exit_fee
    gross_return = (prices[-1] / prices[0] - 1.0) * 100.0
    return {
        "name": BENCHMARK_NAME,
        "symbol": "BTCUSDT",
        "policy": "buy first observable closed bar; sell last observable closed bar",
        "status": "complete",
        "entry_ts": _iso(timestamps[0]),
        "exit_ts": _iso(timestamps[-1]),
        "entry_price": _round(prices[0]),
        "exit_price": _round(prices[-1]),
        "gross_return_before_costs_pct": _round(gross_return),
        "net_return_after_costs_pct": _round((ending / float(initial_capital) - 1.0) * 100.0),
        "fees_quote": _round(entry_fee + exit_fee),
        "slippage_quote": _round(qty * (entry_fill - prices[0]) + qty * (prices[-1] - exit_fill)),
        "valuation_points": len(timestamps),
    }


def evaluate_portfolio_alpha(
    trades: Iterable[Any],
    *,
    price_series_by_symbol: Mapping[str, Sequence[tuple[int, float]]],
    benchmark_series: Sequence[tuple[int, float]],
    window_start_ms: int,
    window_end_ms: int,
    requested_days: int,
    universe: Sequence[str],
    variant: str,
    capacity: int = 10,
    initial_capital: float = 10_000.0,
    fee_bps: float = 7.5,
    slippage_bps: float = 5.0,
    policy_manifest: Mapping[str, Any] | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate one replay candidate stream as a capital-constrained account."""
    trade_rows = list(trades)
    benchmark_timestamps = _normalized_series(benchmark_series)[0]
    net = _simulate_account(
        trade_rows,
        price_series_by_symbol=price_series_by_symbol,
        valuation_timestamps=benchmark_timestamps,
        capacity=capacity,
        initial_capital=initial_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    gross = _simulate_account(
        trade_rows,
        price_series_by_symbol=price_series_by_symbol,
        valuation_timestamps=benchmark_timestamps,
        capacity=capacity,
        initial_capital=initial_capital,
        fee_bps=0.0,
        slippage_bps=0.0,
    )
    benchmark = _benchmark_result(
        benchmark_series,
        initial_capital=initial_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    manifest = dict(policy_manifest or current_policy_manifest())
    artifact_source_hashes = {str(key): str(value) for key, value in (source_hashes or {}).items()}
    artifact_source_hashes[Path(__file__).name] = evaluator_source_hash()
    requested_ms = max(1, int(requested_days) * 24 * 60 * 60 * 1000)
    observed_ms = max(0, (benchmark_timestamps[-1] - benchmark_timestamps[0])) if len(benchmark_timestamps) >= 2 else 0
    period_coverage = min(1.0, observed_ms / requested_ms)
    valuation_coverage = (
        net.fully_valued_points / net.valuation_points if net.valuation_points else 0.0
    )
    violations = sorted(set(net.violations))
    universe_clean = sorted({str(symbol).upper() for symbol in universe if str(symbol).strip()})
    benchmark_complete = benchmark.get("status") == "complete"
    provenance_complete = bool(manifest.get("policy_epoch") and manifest.get("policy_hash") and universe_clean)
    decision_grade = bool(
        int(capacity) == 10
        and int(requested_days) >= MAX_REPLAY_DAYS
        and period_coverage >= MIN_PERIOD_COVERAGE
        and len(trade_rows) > 0
        and not violations
        and float(fee_bps) > 0
        and float(slippage_bps) > 0
        and valuation_coverage == 1.0
        and benchmark_complete
        and provenance_complete
    )
    benchmark_return = float(benchmark.get("net_return_after_costs_pct") or 0.0)
    alpha = net.return_pct - benchmark_return if benchmark_complete else None
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_contract": METRIC_CONTRACT,
        "generated_at": generated_at,
        "status": "complete" if benchmark_complete and not violations else "incomplete",
        "evidence_grade": "decision_grade" if decision_grade else "diagnostic",
        "decision_grade": decision_grade,
        "claim_scope": "causal_replay_policy_alpha",
        "variant": str(variant),
        "window": {
            "requested_start": _iso(window_start_ms),
            "requested_end": _iso(window_end_ms),
            "requested_days": int(requested_days),
            "established_max_replay_days": MAX_REPLAY_DAYS,
            "observed_days": _round(observed_ms / 86_400_000.0),
            "period_coverage": _round(period_coverage),
        },
        "portfolio_contract": {
            "name": "single_unified_symbol_deduplicated_equal_slot_cash_account",
            "capacity": int(capacity),
            "initial_capital_quote": _round(initial_capital),
            "allocation": "min(available_cash, current_liquidation_equity/capacity)",
            "leverage": 0,
            "same_symbol_concurrency": 1,
            "candidate_source_scope": "causal_replay_policy_stream",
        },
        "cost_contract": {
            "name": "static_taker_fee_and_conservative_slippage_v1",
            "fee_bps_per_side": _round(fee_bps),
            "slippage_bps_per_side": _round(slippage_bps),
            "applies_to": ["entry", "partial_exit", "final_exit", "benchmark_entry", "benchmark_exit"],
        },
        "portfolio": {
            "trades": len(trade_rows),
            "ending_equity_after_costs": _round(net.ending_equity),
            "gross_return_before_costs_pct": _round(gross.return_pct),
            "net_return_after_costs_pct": _round(net.return_pct),
            "cost_drag_pct": _round(gross.return_pct - net.return_pct),
            "fees_quote": _round(net.fees_quote),
            "slippage_quote": _round(net.slippage_quote),
            "max_drawdown_after_costs_pct": _round(net.max_drawdown_pct),
            "max_concurrent_positions": net.max_concurrent_positions,
            "average_gross_utilization": _round(net.average_gross_utilization),
        },
        "benchmark": benchmark,
        "net_alpha_after_costs": _round(alpha) if alpha is not None else None,
        "coverage": {
            "valuation_points": net.valuation_points,
            "fully_valued_points": net.fully_valued_points,
            "valuation_coverage": _round(valuation_coverage),
            "contract_violations": violations,
        },
        "provenance": {
            "policy_epoch": manifest.get("policy_epoch"),
            "policy_hash": manifest.get("policy_hash"),
            "config_hash": manifest.get("config_hash"),
            "watchlist_hash": manifest.get("watchlist_hash"),
            "universe_hash": stable_hash(universe_clean),
            "universe_count": len(universe_clean),
            "universe": universe_clean,
            "price_contract": "closed candle prices observable at or before valuation time",
            "source_hashes": artifact_source_hashes,
            "trade_stream_hash": _trade_stream_hash(trade_rows),
            "price_stream_hash": _price_stream_hash(price_series_by_symbol, benchmark_series),
        },
        "limitations": [
            "static slippage is assumed because historical order-book depth is unavailable",
            "result evaluates a causal replay stream, not observed exchange fills",
        ],
    }
