"""RDE-1 offline shared-cash accounting and adverse OHLC execution.

This stdlib-only module has no exchange, credential or training capability.
An action is supplied before its selected/held execution bar is requested.
All fills, fractional quantities and close-based marks are SIMULATION.
"""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import math
from zoneinfo import ZoneInfo


STEP = 900_000
OBSERVATION_SIZE = 27
MARKET_VECTOR_SIZE = 20
ZONE = 'Europe/Budapest'


def day_bounds(day):
    """UTC millisecond boundaries of one Budapest calendar day, including DST."""
    parsed = date.fromisoformat(day)
    if parsed.isoformat() != day:
        raise ValueError('INVALID_CALENDAR_DAY')
    zone = ZoneInfo(ZONE)
    return tuple(int(datetime.combine(parsed + timedelta(days=offset), datetime.min.time(), zone)
                     .timestamp() * 1000) for offset in (0, 1))


def _day(at):
    return datetime.fromtimestamp(at / 1000, timezone.utc).astimezone(ZoneInfo(ZONE)).date().isoformat()


def _timestamp(value):
    if type(value) is not int or value % STEP:
        raise ValueError('INVALID_GRID_TIMESTAMP')


def _number(value):
    if isinstance(value, bool):
        raise ValueError('INVALID_FINITE_NUMBER')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('INVALID_FINITE_NUMBER')
    return result


def aggregate_days(samples, start_ms, end_ms):
    """Aggregate consecutive before/after MTM observations without imputing days.

    A sample marks the bar ``[origin_ms, origin_ms+STEP)``. The before mark
    belongs to that bar's opening boundary; the after mark is its last close.
    Complete/requested calendar denominators are separate from interval PnL.
    """
    _timestamp(start_ms)
    _timestamp(end_ms)
    if start_ms >= end_ms:
        raise ValueError('INVALID_REPORT_INTERVAL')
    days = {}
    previous = None
    for sample in samples:
        at = sample['origin_ms']
        _timestamp(at)
        if previous is not None and at <= previous:
            raise ValueError('NON_MONOTONE_MARKS')
        previous = at
        if not start_ms <= at < end_ms:
            continue
        day = _day(at)
        first, last = day_bounds(day)
        row = days.setdefault(day, dict(first_ms=at, last_ms=at - STEP,
            start_equity=sample['equity_before'], end_equity=sample['equity_before'],
            points_n=0, expected_points_n=(last - first) // STEP, contiguous=True))
        if row['points_n'] and (row['last_ms'] + STEP != at or
                not math.isclose(row['end_equity'], sample['equity_before'], rel_tol=0, abs_tol=1e-8)):
            row['contiguous'] = False
        row.update(last_ms=at, end_equity=sample['equity_after'], points_n=row['points_n'] + 1)
    requested = []
    cursor = date.fromisoformat(_day(start_ms))
    final = date.fromisoformat(_day(end_ms - 1))
    while cursor <= final:
        name = cursor.isoformat()
        first, last = day_bounds(name)
        if first >= start_ms and last <= end_ms:
            requested.append(name)
            if name not in days:
                days[name] = dict(first_ms=None, last_ms=None, start_equity=None,
                    end_equity=None, points_n=0, expected_points_n=(last - first) // STEP,
                    contiguous=False)
        cursor += timedelta(days=1)
    daily = {}
    for name, row in days.items():
        first, last = day_bounds(name)
        complete = (row['first_ms'] == first and row['last_ms'] is not None
                    and row['last_ms'] + STEP == last and row['contiguous']
                    and row['points_n'] == row['expected_points_n'])
        row['net_pnl_usdt'] = (row['end_equity'] - row['start_equity']
                               if row['points_n'] else None)
        row['calendar_status'] = ('COMPLETE_SIMULATION' if complete else
                                  'PARTIAL_SIMULATION' if row['points_n'] else 'MISSING_SIMULATION')
        if complete:
            daily[name] = row['net_pnl_usdt']
    values = list(daily.values())
    return dict(days=days, daily_pnl=daily, requested_complete_days=requested,
        requested_complete_days_N=len(requested), complete_days_n=len(daily),
        partial_days_n=sum(v['calendar_status'] == 'PARTIAL_SIMULATION' for v in days.values()),
        missing_days_n=sum(v['calendar_status'] == 'MISSING_SIMULATION' for v in days.values()),
        excluded_complete_days_n=len(set(requested) - set(daily)),
        profitable_days_n=sum(v > 0 for v in values),
        mean_daily_net_pnl_usdt=sum(values) / len(values) if values else None,
        complete_day_net_pnl_usdt=sum(values) if values else None)


class ResearchAccount:
    """One bounded position, continuous cash and interval-latched entry risk.

    Action 0 waits/holds; 1 buys10USDT; 2 buys30USDT; 3 exits at next open.
    Position BUY and flat EXIT are recorded invalid intents, converted to hold
    and wait respectively. They never authorize an additional fill.
    """

    def __init__(self, fee_bps=10, slippage_bps=5, exit_mode='fixed', initial=100):
        self.fee = _number(fee_bps) / 10000
        self.slip = _number(slippage_bps) / 10000
        self.initial = _number(initial)
        if not 0 <= self.fee <= .01 or not 0 <= self.slip <= .01 or self.initial <= 0:
            raise ValueError('INVALID_ACCOUNT_COSTS_OR_CAPITAL')
        if exit_mode not in ('fixed', 'trailing', 'learned'):
            raise ValueError('INVALID_EXIT_MODE')
        self.exit_mode = exit_mode
        self.cash = self.initial
        self.equity = self.initial
        self.peak_equity = self.initial
        self.max_drawdown = 0.
        self.position = None
        self.entry_killed = False
        self.valid = True
        self.reasons = []
        self.samples = []
        self.events = []
        self.trades = []
        self.last_origin = None
        self.fees = 0.
        self.turnover = 0.
        self.max_entry_debit = 0.
        self.counters = dict(steps_n=0, invalid_actions_n=0, entry_vetoes_n=0,
                             buy_intents_n=0, buys_n=0, sells_n=0)

    def observation(self, origin_ms, candidate, vector_lookup, remaining=1.):
        """Copy20 causal market values plus7 identical portfolio-state values.

        The held asset takes precedence over the next entry candidate. Marks
        and peaks come exclusively from already completed account steps.
        There is deliberately no execution-bar lookup in this method.
        """
        _timestamp(origin_ms)
        remaining = _number(remaining)
        if not 0 <= remaining <= 1:
            raise ValueError('INVALID_EPISODE_REMAINING')
        symbol = self.position['symbol'] if self.position else candidate
        raw = vector_lookup(symbol, origin_ms) if symbol is not None else None
        market = [0.] * MARKET_VECTOR_SIZE if raw is None else [_number(v) for v in raw]
        if len(market) != MARKET_VECTOR_SIZE:
            raise ValueError('INVALID_MARKET_VECTOR_SIZE')
        position = self.position
        marked = position['quantity'] * position['mark_price'] if position else 0.
        entry_return = position['mark_price'] / position['entry_price'] - 1 if position else 0.
        age = (origin_ms - position['entry_ms']) / STEP / 16 if position else 0.
        prior_peak = position['prior_peak'] / position['entry_price'] - 1 if position else 0.
        used_risk = max(self.initial - self.equity, self.peak_equity - self.equity, 0.)
        risk_margin = 0. if self.entry_killed else max(0., 1. - used_risk)
        return market + [self.cash / 100, marked / 100, entry_return, age,
                         prior_peak, risk_margin, remaining]

    def _mark(self, value):
        self.peak_equity = max(self.peak_equity, value)
        self.max_drawdown = max(self.max_drawdown, self.peak_equity - value)
        if self.initial - value >= 1 or self.peak_equity - value >= 1:
            self.entry_killed = True

    def _invalidate(self, reason):
        self.valid = False
        if reason not in self.reasons:
            self.reasons.append(reason)

    def _result(self, reward, events):
        return dict(reward=reward, equity=self.equity, cash=self.cash,
            events=deepcopy(events), valid=self.valid, entry_killed=self.entry_killed,
            position=deepcopy(self.position), reason=self.reasons[-1] if self.reasons else None)

    def _sell(self, raw_price, at, reason):
        position = self.position
        price = raw_price if reason == 'TP' else raw_price * (1 - self.slip)
        gross = position['quantity'] * price
        fee = gross * self.fee
        proceeds = gross - fee
        self.cash += proceeds
        self.fees += fee
        self.turnover += gross
        event = dict(side='SELL', symbol=position['symbol'], at_ms=at, price=price,
            quantity=position['quantity'], fee_usdt=fee, gross_usdt=gross,
            proceeds_usdt=proceeds, reason=reason)
        self.events.append(event)
        self.trades.append(dict(symbol=position['symbol'], entry_ms=position['entry_ms'],
            exit_ms=at, entry_price=position['entry_price'], exit_price=price,
            quantity=position['quantity'], entry_debit_usdt=position['debit_usdt'],
            net_usdt=proceeds - position['debit_usdt'], reason=reason))
        self.counters['sells_n'] += 1
        self.position = None
        return event

    def step(self, origin_ms, candidate, action, bar_lookup, allow_entry=True):
        """Resolve one15m transition after the caller has fixed its action.

        Flat WAIT needs no price, whereas a missing selected BUY or held bar
        invalidates the objective. No alternative symbol is looked up.
        """
        _timestamp(origin_ms)
        if type(action) is not int or action not in (0, 1, 2, 3) or type(allow_entry) is not bool:
            raise ValueError('INVALID_ACTION_CONTRACT')
        if candidate is not None and (not isinstance(candidate, str) or not candidate):
            raise ValueError('INVALID_CANDIDATE')
        if not self.valid:
            return self._result(0., [])
        if self.last_origin is not None and origin_ms != self.last_origin + STEP:
            self._invalidate('NON_CONSECUTIVE_STEP')
            return self._result(0., [])
        before = self.equity
        events = []
        holding = self.position is not None
        if action in (1, 2):
            self.counters['buy_intents_n'] += 1
        if holding and action in (1, 2):
            self.counters['invalid_actions_n'] += 1
            action = 0
        elif not holding and action == 3:
            self.counters['invalid_actions_n'] += 1
            action = 0
        debit = min(10. if action == 1 else 30., self.cash)
        if not holding and action in (1, 2) and (
                not allow_entry or self.entry_killed or candidate is None or debit < 5):
            self.counters['entry_vetoes_n'] += 1
            action = 0
        symbol = self.position['symbol'] if holding else candidate if action in (1, 2) else None
        row = None
        if symbol is not None:
            raw = bar_lookup(symbol, origin_ms)
            if raw is None:
                self._invalidate('HELD_EXECUTION_MISSING' if holding else 'SELECTED_EXECUTION_MISSING')
                return self._result(0., [])
            try:
                row = {key: _number(raw[key]) for key in ('open', 'high', 'low', 'close')}
                if (min(row.values()) <= 0 or row['high'] < max(row['open'], row['close'])
                        or row['low'] > min(row['open'], row['close']) or row['low'] > row['high']):
                    raise ValueError('INVALID_OHLC')
            except (KeyError, TypeError, ValueError, OverflowError):
                self._invalidate('EXECUTION_BAR_INVALID')
                return self._result(0., [])
        if holding:
            stop = self.position['entry_price'] * .995
            reason = 'SL'
            if self.exit_mode == 'trailing' and self.position['prior_peak'] >= self.position['entry_price'] * 1.005:
                trailing = self.position['prior_peak'] * .995
                if trailing > stop:
                    stop, reason = trailing, 'TRAIL'
            if row['open'] <= stop:
                events.append(self._sell(row['open'], origin_ms, reason))
            elif self.exit_mode == 'fixed' and row['open'] >= self.position['entry_price'] * 1.005:
                events.append(self._sell(self.position['entry_price'] * 1.005, origin_ms, 'TP'))
            if self.position is not None:
                self._mark(self.cash + self.position['quantity'] * row['open'])
        if holding and self.position is not None and action == 3:
            events.append(self._sell(row['open'], origin_ms, 'EXIT'))
        elif not holding and action in (1, 2):
            price = row['open'] * (1 + self.slip)
            gross = debit / (1 + self.fee)
            fee = debit - gross
            self.cash -= debit
            self.fees += fee
            self.turnover += gross
            self.max_entry_debit = max(self.max_entry_debit, debit)
            self.position = dict(symbol=symbol, quantity=gross / price,
                entry_price=price, entry_ms=origin_ms, prior_peak=price,
                mark_price=price, debit_usdt=debit)
            event = dict(side='BUY', symbol=symbol, at_ms=origin_ms, price=price,
                quantity=self.position['quantity'], fee_usdt=fee, gross_usdt=gross,
                debit_usdt=debit, reason='POLICY_ENTRY')
            events.append(event)
            self.events.append(event)
            self.counters['buys_n'] += 1
            self._mark(self.cash + self.position['quantity'] * row['open'])
        if self.position is not None:
            position = self.position
            stop = position['entry_price'] * .995
            stop_reason = 'SL'
            if self.exit_mode == 'trailing' and position['prior_peak'] >= position['entry_price'] * 1.005:
                trailing = position['prior_peak'] * .995
                if trailing > stop:
                    stop, stop_reason = trailing, 'TRAIL'
            target = position['entry_price'] * 1.005
            if row['low'] <= stop:
                events.append(self._sell(min(row['open'], stop), origin_ms, stop_reason))
            elif self.exit_mode == 'fixed' and row['high'] >= target:
                events.append(self._sell(target, origin_ms, 'TP'))
            elif origin_ms - position['entry_ms'] >= 15 * STEP:
                events.append(self._sell(row['close'], origin_ms + STEP - 1, 'HOLDING_LIMIT'))
            else:
                position['prior_peak'] = max(position['prior_peak'], row['high'])
                position['mark_price'] = row['close']
        self.equity = self.cash + (self.position['quantity'] * self.position['mark_price']
                                   if self.position else 0.)
        self._mark(self.equity)
        self.last_origin = origin_ms
        self.counters['steps_n'] += 1
        self.samples.append(dict(origin_ms=origin_ms, equity_before=before, equity_after=self.equity))
        return self._result((self.equity - before) / 100, events)

    def report(self, start_ms, end_ms):
        """Report interval MTM separately from complete calendar-day estimates."""
        daily = aggregate_days(self.samples, start_ms, end_ms)
        observed = [v for v in self.samples if start_ms <= v['origin_ms'] < end_ms]
        expected = (end_ms - start_ms) // STEP
        coverage = (len(observed) == expected and observed[0]['origin_ms'] == start_ms
                    and observed[-1]['origin_ms'] + STEP == end_ms)
        valid = self.valid and coverage
        reasons = list(self.reasons)
        if not coverage and 'INTERVAL_COVERAGE_INCOMPLETE' not in reasons:
            reasons.append('INTERVAL_COVERAGE_INCOMPLETE')
        interval = (observed[-1]['equity_after'] - observed[0]['equity_before']) if observed else None
        if not valid:
            daily.update(daily_pnl={}, complete_days_n=0, profitable_days_n=0,
                mean_daily_net_pnl_usdt=None, complete_day_net_pnl_usdt=None,
                excluded_complete_days_n=daily['requested_complete_days_N'])
        exposure = self.position['quantity'] * self.position['mark_price'] if self.position else 0.
        return dict(daily, state='COMPLETE' if valid else 'INCOMPLETE', evidence_kind='SIMULATION',
            promotion_allowed=False, start_ms=start_ms, end_ms=end_ms,
            initial_equity_usdt=self.initial, final_equity_usdt=self.equity,
            interval_net_pnl_usdt=interval if valid else None, net_pnl_usdt=interval if valid else None,
            observed_steps_n=len(observed), requested_steps_N=expected, reasons=reasons,
            fees_usdt=self.fees, turnover_usdt=self.turnover, max_drawdown_usdt=self.max_drawdown,
            max_entry_debit_usdt=self.max_entry_debit, max_concurrent_positions=int(self.counters['buys_n'] > 0),
            open_exposure_usdt=exposure, open_position=deepcopy(self.position), cash_usdt=self.cash,
            entry_killed=self.entry_killed, counters=dict(self.counters),
            trades_n=len(self.trades), trades=deepcopy(self.trades), events=deepcopy(self.events),
            equity_curve=[dict(v) for v in self.samples])
