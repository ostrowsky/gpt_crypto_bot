"""Fail-closed same-day signal accounting; not exchange fills or portfolio PnL."""
import math
from datetime import datetime, timezone


def _time(value):
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _price(value):
    try:
        x = float(value)
        return x if math.isfinite(x) and x > 0 else None
    except (ValueError, TypeError):
        return None


def measure(events, day_open, day_close):
    active, bad, seen, trades = {}, [], set(), []
    rows = [(kind, r) for kind in ('entries', 'exits') for r in events.get(kind, [])]
    if any(_time(r.get('ts')) is None for _, r in rows):
        bad.append('invalid_timestamp')
    rows.sort(key=lambda kr: str(kr[1].get('ts', '')))
    rows = sorted((kr for kr in rows if _time(kr[1].get('ts')) is not None),
                  key=lambda kr: _time(kr[1]['ts']))
    for kind, r in rows:
        key = (r.get('_log_source', r.get('source', 'unknown')),
               r.get('sym', r.get('symbol')), r.get('tf'))
        if key[0] == 'unknown' or not key[1] or not key[2]:
            bad.append('missing_identity')
        signature = (kind, key, r.get('ts'), r.get('price'), r.get('entry_price'),
                     r.get('exit_price'), r.get('position_id'))
        if signature in seen:
            continue
        seen.add(signature)
        if kind == 'entries':
            if key in active:
                bad.append('overlapping_entries')
            active[key] = r
            continue
        entry = active.pop(key, None)
        ep = _price((entry or {}).get('price'))
        stated, xp = _price(r.get('entry_price')), _price(r.get('exit_price'))
        if not entry or not ep or not stated or not xp:
            bad.append('unmatched_exit_or_missing_price')
            continue
        if not math.isclose(ep, stated, rel_tol=1e-6, abs_tol=1e-10):
            bad.append('entry_price_mismatch')
            continue
        if (entry.get('position_id') or r.get('position_id')) and entry.get('position_id') != r.get('position_id'):
            bad.append('position_id_mismatch')
            continue
        if _time(r['ts']) <= _time(entry['ts']):
            bad.append('nonpositive_duration')
            continue
        trades.append(dict(source=key[0], symbol=key[1], timeframe=key[2],
                           position_id=entry.get('position_id'),
                           entry_time=entry['ts'], exit_time=r['ts'],
                           entry_price=ep, exit_price=xp, price_gain=xp-ep))
    if active:
        bad.append('open_or_unmatched_entry')
    ordered = sorted(trades, key=lambda t: _time(t['entry_time']))
    if any(_time(b['entry_time']) < _time(a['exit_time']) for a, b in zip(ordered, ordered[1:])):
        bad.append('overlapping_positions')
    op, cl = _price(day_open), _price(day_close)
    denominator = cl-op if op and cl else None
    if denominator is None or denominator <= 0:
        bad.append('invalid_day_move')
    if not trades:
        bad.append('no_matched_trades')
    gain = sum(t['price_gain'] for t in trades) if not bad else None
    return dict(version='matched_signal_capture_v1', status='unknown' if bad else 'measured',
                reasons=sorted(set(bad)), matched_trades=len(trades),
                price_gain_sum=gain, day_price_move=denominator,
                realized_capture_ratio=gain/denominator if gain is not None else None,
                scope='same_day_signal_price_path_not_fills_or_capital_return',
                trades=trades, exit_efficiency=None, giveback_pct=None,
                exit_quality_status='unknown_no_verified_in_position_mfe',
                exit_quality_denominator=0)
