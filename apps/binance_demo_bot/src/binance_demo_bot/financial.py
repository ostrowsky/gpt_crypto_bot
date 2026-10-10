"""Offline P0 financial projection. No I/O, exchange access or order sending.

The ledger is an in-memory projection, not the future durable event journal.
Portfolio marks and certified boundary snapshots must be supplied by callers.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from functools import wraps
from zoneinfo import ZoneInfo

ZERO = Decimal("0")
ONE = Decimal("1")
STATUSES = {"COMPLETE", "PENDING", "ERROR"}
OWNERSHIPS = {"SEED", "APPLICATION", "UNATTRIBUTED"}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class AccountingError(ValueError):
    """Invalid or insufficient inventory; no part of the fill is committed."""


class LedgerConflict(ValueError):
    """An accepted trade key has a different economic payload."""


def _arithmetic(function):
    @wraps(function)
    def call(*args, **kwargs):
        with localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN)):
            return function(*args, **kwargs)
    return call


def decimal_amount(value):
    if not isinstance(value, (str, Decimal)):
        raise ValueError("Money requires a Decimal or decimal string")
    try:
        amount = Decimal(value)
    except ArithmeticError:
        raise ValueError("Invalid decimal amount") from None
    if not amount.is_finite():
        raise ValueError("Money must be finite")
    return amount


def _nonnegative(value):
    amount = decimal_amount(value)
    if amount < ZERO:
        raise ValueError("Amount must be nonnegative")
    return amount


def _positive(value):
    amount = _nonnegative(value)
    if amount == ZERO:
        raise ValueError("Amount must be positive")
    return amount


def _status(value):
    if value not in STATUSES:
        raise ValueError("Unknown completeness status")
    return value


def _milliseconds(value):
    if type(value) is not int:
        raise ValueError("UTC milliseconds must be an integer")
    return value


def _local_date(value):
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("Expected an ISO calendar date")
    return parsed


def day_bounds(day, timezone):
    local_day = _local_date(day)
    zone = ZoneInfo(timezone)
    boundaries = (datetime.combine(local_day, datetime.min.time(), zone),
                  datetime.combine(local_day + timedelta(days=1), datetime.min.time(), zone))
    return tuple((bound.astimezone(EPOCH.tzinfo) - EPOCH) // timedelta(milliseconds=1)
                 for bound in boundaries)


def day_for_timestamp(timestamp_ms, timezone):
    instant = EPOCH + timedelta(milliseconds=_milliseconds(timestamp_ms))
    return instant.astimezone(ZoneInfo(timezone)).date().isoformat()


def _conversion(asset, quotes, routes, boundary, time_basis, max_age):
    if asset not in routes:
        return "PENDING", None, "MISSING_ROUTE"
    seen, current, mark = {asset}, asset, ONE
    for symbol, inverse in routes[asset]:
        if type(inverse) is not bool:
            return "ERROR", None, "INVALID_ROUTE"
        quote = quotes.get(symbol)
        if quote is None:
            return "PENDING", None, "MISSING_QUOTE"
        if quote.get("venue") != "BINANCE_SPOT_DEMO":
            return "ERROR", None, "FOREIGN_VENUE"
        source, target = ((quote["quote"], quote["base"]) if inverse
                          else (quote["base"], quote["quote"]))
        if source != current or target in seen:
            return "ERROR", None, "INVALID_ROUTE"
        bid, ask = _positive(quote["bid"]), _positive(quote["ask"])
        if bid > ask:
            return "ERROR", None, "CROSSED_QUOTE"
        if not quote.get("source_healthy"):
            return "PENDING", None, "UNHEALTHY_SOURCE"
        if time_basis == "EXCHANGE_EVENT":
            clock = quote.get("event_time_ms")
            if clock is None:
                return "PENDING", None, "MISSING_EXCHANGE_CLOCK"
        else:
            clock = quote.get("received_at_ms")
            started = quote.get("response_started_at_ms")
            if clock is None or started is None:
                return "PENDING", None, "MISSING_RESPONSE_INTERVAL"
            if _milliseconds(started) > _milliseconds(clock):
                return "ERROR", None, "INVALID_RESPONSE_INTERVAL"
        age = boundary - _milliseconds(clock)
        if age < 0 or age > max_age:
            return "PENDING", None, "FUTURE_OR_STALE_QUOTE"
        mid = (bid + ask) / Decimal("2")
        mark *= ONE / mid if inverse else mid
        current = target
        seen.add(target)
    if current != "USDT":
        return "ERROR", None, "INCOMPLETE_ROUTE"
    return "COMPLETE", mark, None


@_arithmetic
def value_portfolio(balances, quotes, routes, *, boundary_ms, time_basis, max_age_ms):
    boundary = _milliseconds(boundary_ms)
    if time_basis not in {"EXCHANGE_EVENT", "OBSERVED_RESPONSE"}:
        raise ValueError("A registered valuation clock is required")
    if type(max_age_ms) is not int or max_age_ms < 0:
        raise ValueError("Invalid quote age limit")
    equity, marks, reasons, status = ZERO, {"USDT": ONE}, [], "COMPLETE"
    for asset, balance in balances.items():
        try:
            quantity = _nonnegative(balance["free"]) + _nonnegative(balance["locked"])
            if quantity == ZERO:
                continue
            result, mark, reason = (("COMPLETE", ONE, None) if asset == "USDT" else
                                   _conversion(asset, quotes, routes, boundary, time_basis, max_age_ms))
        except (ValueError, KeyError, TypeError, ArithmeticError):
            result, mark, reason = "ERROR", None, "INVALID_ASSET_OR_QUOTE"
        if result != "COMPLETE":
            reasons.append(f"{asset}:{reason}")
            if result == "ERROR" or status == "COMPLETE":
                status = result
        else:
            marks[asset] = mark
            equity += quantity * mark
    return dict(status=status, equity_usdt=equity if status == "COMPLETE" else None,
                marks=marks, reason_codes=reasons)


@dataclass
class _Lot:
    quantity: Decimal
    basis: Decimal


class FinancialLedger:
    @_arithmetic
    def __init__(self, *, episode_id, opening_lots):
        if not isinstance(episode_id, str) or not episode_id:
            raise AccountingError("An account episode is required")
        self.episode_id = episode_id
        self._lots = defaultdict(list)
        self._managed = defaultdict(lambda: ZERO)
        self._fills = {}
        self._last_fill_clock = None
        self._realized, self._fees = ZERO, ZERO
        try:
            for lot in opening_lots:
                quantity, basis = _nonnegative(lot["quantity"]), _nonnegative(lot["basis_usdt"])
                if lot["ownership"] not in OWNERSHIPS or (quantity == ZERO and basis != ZERO):
                    raise ValueError("Invalid opening lot")
                if quantity:
                    self._lots[lot["asset"]].append(_Lot(quantity, basis))
                    if lot["ownership"] == "APPLICATION":
                        self._managed[lot["asset"]] += quantity
        except (ValueError, KeyError, TypeError, ArithmeticError):
            raise AccountingError("Invalid opening inventory") from None

    @_arithmetic
    def balances(self):
        return {asset: sum((lot.quantity for lot in lots), ZERO)
                for asset, lots in self._lots.items()}

    def managed_quantity(self, asset):
        return self._managed.get(asset, ZERO)

    @staticmethod
    def _mark(asset, marks):
        if asset == "USDT":
            if asset in marks and decimal_amount(marks[asset]) != ONE:
                raise AccountingError("USDT mark must equal one")
            return ONE
        return _positive(marks[asset])

    def _dispose(self, asset, quantity, mark):
        if quantity > self.balances().get(asset, ZERO):
            raise AccountingError("Insufficient inventory")
        remaining = quantity
        while remaining:
            lot = self._lots[asset][0]
            used = min(lot.quantity, remaining)
            cost = lot.basis if used == lot.quantity else lot.basis * used / lot.quantity
            self._realized += used * mark - cost
            lot.quantity -= used
            lot.basis -= cost
            remaining -= used
            if lot.quantity == ZERO:
                self._lots[asset].pop(0)
        self._managed[asset] = max(ZERO, self._managed.get(asset, ZERO) - quantity)

    @_arithmetic
    def apply_fill(self, fill, marks):
        try:
            required = ("application_id", "environment", "episode_id", "symbol", "trade_id",
                        "order_id", "client_order_id", "base_asset", "quote_asset", "side",
                        "quantity", "quote_quantity", "price", "commission_asset",
                        "commission_quantity", "exchange_event_time_ms", "ownership")
            payload = {key: fill[key] for key in required}
            if (payload["application_id"] != "binance_demo_bot" or
                    payload["environment"] != "BINANCE_SPOT_DEMO" or
                    payload["episode_id"] != self.episode_id):
                raise AccountingError("Foreign application, environment or episode")
            for key in ("quantity", "quote_quantity", "price"):
                payload[key] = _positive(payload[key])
            payload["commission_quantity"] = _nonnegative(payload["commission_quantity"])
            _milliseconds(payload["exchange_event_time_ms"])
            if (payload["side"] not in {"BUY", "SELL"} or payload["ownership"] not in OWNERSHIPS
                    or payload["ownership"] == "SEED"
                    or payload["base_asset"] == payload["quote_asset"]
                    or payload["quantity"] * payload["price"] != payload["quote_quantity"]):
                raise AccountingError("Invalid fill arithmetic or identity")
            key = (payload["symbol"], payload["trade_id"])
            if key in self._fills:
                if self._fills[key] != payload:
                    raise LedgerConflict("Conflicting payload for accepted trade")
                return "DUPLICATE"
            clock = payload["exchange_event_time_ms"]
            if self._last_fill_clock is not None and clock < self._last_fill_clock:
                raise AccountingError("Late fill requires ordered journal replay")
            candidate = deepcopy(self)
            base, quote = payload["base_asset"], payload["quote_asset"]
            quantity, quote_quantity = payload["quantity"], payload["quote_quantity"]
            quote_mark = self._mark(quote, marks)
            if payload["side"] == "BUY":
                candidate._dispose(quote, quote_quantity, quote_mark)
                received, received_quantity, basis = base, quantity, quote_quantity * quote_mark
            else:
                if payload["ownership"] == "APPLICATION" and quantity > candidate.managed_quantity(base):
                    raise AccountingError("Exit exceeds application-owned inventory")
                candidate._dispose(base, quantity, quote_quantity * quote_mark / quantity)
                received, received_quantity, basis = quote, quote_quantity, quote_quantity * quote_mark
            candidate._lots[received].append(_Lot(received_quantity, basis))
            if payload["ownership"] == "APPLICATION":
                candidate._managed[received] += received_quantity
            fee_quantity = payload["commission_quantity"]
            if fee_quantity:
                fee_asset = payload["commission_asset"]
                fee_mark = self._mark(fee_asset, marks)
                candidate._dispose(fee_asset, fee_quantity, fee_mark)
                candidate._fees += fee_quantity * fee_mark
            candidate._fills[key] = payload
            candidate._last_fill_clock = clock
        except LedgerConflict:
            raise
        except (ValueError, KeyError, TypeError, ArithmeticError):
            raise AccountingError("Fill violates the accounting contract") from None
        self._lots, self._managed, self._fills = candidate._lots, candidate._managed, candidate._fills
        self._realized, self._fees = candidate._realized, candidate._fees
        self._last_fill_clock = candidate._last_fill_clock
        return "APPLIED"

    @_arithmetic
    def attribution(self, marks):
        unrealized = ZERO
        try:
            for asset, lots in self._lots.items():
                if lots:
                    mark = self._mark(asset, marks)
                    unrealized += sum((lot.quantity * mark - lot.basis for lot in lots), ZERO)
        except (ValueError, KeyError, TypeError, ArithmeticError):
            raise AccountingError("Complete inventory marks are required") from None
        return dict(gross_realized_usdt=self._realized, fee_expenses_usdt=self._fees,
                    unrealized_usdt=unrealized)


@_arithmetic
def daily_performance(opening, closing, flows, *, reconciliation_status, economic_costs_usdt):
    start, end = _milliseconds(opening["boundary_ms"]), _milliseconds(closing["boundary_ms"])
    if end <= start:
        raise ValueError("Daily boundaries must be ordered and distinct")
    cost = None if economic_costs_usdt is None else _nonnegative(economic_costs_usdt)
    states = [_status(opening["status"]), _status(closing["status"]), _status(reconciliation_status)]
    reasons, net_flow, has_external_flow = [], ZERO, False
    if opening["episode_id"] != closing["episode_id"] or opening["scope_id"] != closing["scope_id"]:
        states.append("ERROR")
        reasons.append("EPISODE_OR_SCOPE_MISMATCH")
    for flow in flows:
        if not start <= _milliseconds(flow["effective_ms"]) < end:
            continue
        states.append(_status(flow["status"]))
        classification = flow["classification"]
        if classification not in {"DEPOSIT", "WITHDRAWAL", "INTERNAL_ACCOUNT_TRANSFER"}:
            raise ValueError("Unknown external flow classification")
        if flow["status"] == "COMPLETE":
            amount = decimal_amount(flow["signed_usdt"])
            if classification == "DEPOSIT" and amount < ZERO or classification == "WITHDRAWAL" and amount > ZERO:
                raise ValueError("External flow sign contradicts classification")
            if classification != "INTERNAL_ACCOUNT_TRANSFER":
                net_flow += amount
                has_external_flow |= amount != ZERO
    if opening["equity_usdt"] is None or closing["equity_usdt"] is None:
        states.append("PENDING")
    status = "ERROR" if "ERROR" in states else "PENDING" if "PENDING" in states else "COMPLETE"
    result = dict(status=status, net_pnl_usdt=None, net_external_flow_usdt=None, daily_return=None,
                  economic_pnl_usdt=None, economic_status="ERROR" if status == "ERROR" else "PENDING",
                  reason_codes=reasons)
    if status != "COMPLETE":
        reasons.append("INCOMPLETE_OR_ERROR_ACCOUNTING")
        return result
    initial, final = _nonnegative(opening["equity_usdt"]), _nonnegative(closing["equity_usdt"])
    pnl = final - initial - net_flow
    result.update(net_pnl_usdt=pnl, net_external_flow_usdt=net_flow)
    if not has_external_flow and initial > ZERO:
        result["daily_return"] = pnl / initial
    if cost is not None:
        result.update(economic_pnl_usdt=pnl - cost, economic_status="COMPLETE")
    return result


@_arithmetic
def time_weighted_return(segments):
    compounded = ONE
    if not segments:
        return dict(status="PENDING", value=None)
    for segment in segments:
        start, end = segment["start_equity_usdt"], segment["end_equity_usdt"]
        if start is None or end is None:
            return dict(status="PENDING", value=None)
        start, end = decimal_amount(start), _nonnegative(end)
        if start <= ZERO:
            return dict(status="PENDING", value=None)
        compounded *= end / start
    return dict(status="COMPLETE", value=compounded - ONE)


@_arithmetic
def reconcile_pnl(*, net_pnl_usdt, gross_realized_usdt, fee_expenses_usdt,
                  unrealized_delta_usdt, tolerance_usdt):
    tolerance = _nonnegative(tolerance_usdt)
    residual = decimal_amount(net_pnl_usdt) - (decimal_amount(gross_realized_usdt)
        - _nonnegative(fee_expenses_usdt) + decimal_amount(unrealized_delta_usdt))
    return dict(status="COMPLETE" if abs(residual) <= tolerance else "ERROR", residual_usdt=residual)


@_arithmetic
def aggregate_latest_daily(calendar_days, reports):
    days = list(calendar_days)
    if len(set(days)) != len(days):
        raise ValueError("Calendar manifest contains duplicate days")
    for day in days:
        _local_date(day)
    latest, seen = {}, set()
    for report in reports:
        day, revision = report["day"], report["revision"]
        if day not in days or type(revision) is not int or revision < 1 or (day, revision) in seen:
            raise ValueError("Invalid report day or duplicate revision")
        status = _status(report["status"])
        if status == "COMPLETE":
            decimal_amount(report["net_pnl_usdt"])
        seen.add((day, revision))
        if day not in latest or revision > latest[day]["revision"]:
            latest[day] = report
    counts = {status: 0 for status in STATUSES}
    outcomes = []
    for day in days:
        report = latest.get(day)
        status = report["status"] if report else "PENDING"
        counts[status] += 1
        if status == "COMPLETE":
            outcomes.append(decimal_amount(report["net_pnl_usdt"]))
    n, positive = len(outcomes), sum(value > ZERO for value in outcomes)
    return dict(N_calendar=len(days), N_complete=n, N_pending=counts["PENDING"], N_error=counts["ERROR"],
                positive_days=positive, mean_daily_net_pnl_usdt=sum(outcomes, ZERO) / n if n else None,
                positive_day_fraction=dict(n=positive, N=n, value=Decimal(positive) / n if n else None))
