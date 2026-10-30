"""HT05..07 guarded public-trade compatibility and simulated first touch.

No network, trading, fitting or native-fill certification. Aggregate/raw-ID
continuity is an explicit strict coverage condition, not an API guarantee.
"""
from decimal import Context, Decimal, MAX_EMAX, MIN_EMIN, localcontext
import math

from .intrabar_data import (NUMBER_FIELDS, PRICE_FIELDS, VOLUME_FIELDS,
                           _decimal, _row, canonical_sha256, validate_window)

STEP = 900_000
FIELDS = frozenset(('a','p','q','f','l','T','m','M'))


def _trade(value):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError('TRADE_SCHEMA')
    if (any(type(value[k]) is not int or value[k] < 0 for k in ('a','f','l','T'))
            or value['f'] > value['l'] or type(value['m']) is not bool
            or type(value['M']) is not bool):
        raise ValueError('TRADE_SCHEMA')
    price, quantity = _decimal(value['p']), _decimal(value['q'])
    if price <= 0 or quantity <= 0:
        raise ValueError('TRADE_VALUES')
    return dict(value, price=price, quantity=quantity)


def _output(state, reasons, *, reference=None, minute_checks=None, **values):
    return dict(state=state,reasons=reasons,minute_reference=reference,
                minute_checks=[] if minute_checks is None else minute_checks,
                native_fill_certified=False,pnl_repriced=False,**values)


def _exact_context(minutes, records):
    numbers = [row[k] for row in minutes for k in NUMBER_FIELDS]
    high = max(v.adjusted() for v in numbers)
    low = min(v.as_tuple().exponent for v in numbers)
    for row in records:
        p, q = row['price'], row['quantity']
        high = max(high,p.adjusted(),q.adjusted(),p.adjusted()+q.adjusted()+1)
        low = min(low,p.as_tuple().exponent,q.as_tuple().exponent,
                  p.as_tuple().exponent+q.as_tuple().exponent)
    raw_count = sum(r['l']-r['f']+1 for r in records)
    return Context(prec=max(34,high-low+len(str(raw_count+len(records)+1))+10),
                   Emax=MAX_EMAX,Emin=MIN_EMIN)


def _aggregate_minute(reference, group, carried_price):
    """Compare one complete native minute under the fixed HT06 contract."""
    at=reference[0]
    raw_count=sum(t['l']-t['f']+1 for t in group)
    if group:
        prices=[t['price'] for t in group]
        o,h,l,c=prices[0],max(prices),min(prices),prices[-1]
        volume=sum((t['quantity'] for t in group),Decimal(0))
        quote=sum((t['quantity']*t['price'] for t in group),Decimal(0))
        buy=sum((t['quantity'] for t in group if not t['m']),Decimal(0))
        buy_quote=sum((t['quantity']*t['price'] for t in group if not t['m']),Decimal(0))
    else:
        # This is backed by the previous actual print and full guarded IDs,
        # rather than imputation of an empty/failed HTTP history response.
        o=h=l=c=carried_price
        volume=quote=buy=buy_quote=Decimal(0)
    aggregate=[at,o,h,l,c,volume,at+59_999,quote,raw_count,buy,buy_quote,Decimal(0)]
    differences={};reasons=[]
    for index,name in PRICE_FIELDS.items():
        expected,actual=reference[index],aggregate[index]
        difference=abs(expected-actual)
        differences[name]=dict(original=format(expected,'f'),aggregate=format(actual,'f'),
            difference=format(difference,'f'),bound='0',exact_equal=difference==0,within_bound=difference==0)
        if difference:
            reasons.append('FIELD_MISMATCH:'+name)
    for index,name in VOLUME_FIELDS.items():
        expected,actual=reference[index],aggregate[index]
        quantum=min(Decimal('1e-8'),Decimal((0,(1,),expected.as_tuple().exponent)))
        bound=Decimal(raw_count+1)/2*quantum
        difference=abs(expected-actual)
        differences[name]=dict(original=format(expected,'f'),aggregate=format(actual,'f'),
            difference=format(difference,'f'),signed_difference=format(actual-expected,'f'),
            bound=format(bound,'f'),quantum=format(quantum,'f'),exact_equal=difference==0,
            within_bound=difference<=bound,rounding_assumption='ASSUMED_NEAREST_ROUNDING')
        if difference>bound:
            reasons.append('FIELD_MISMATCH:'+name)
    difference=abs(raw_count-reference[8])
    differences['trade_count']=dict(original=str(reference[8]),aggregate=str(raw_count),
        difference=str(difference),bound='0',exact_equal=difference==0,within_bound=difference==0)
    if difference:
        reasons.append('FIELD_MISMATCH:trade_count')
    result=list(aggregate)
    for index in NUMBER_FIELDS:
        result[index]=format(result[index],'f')
    return dict(state='MISMATCH' if reasons else 'MATCH',reasons=reasons,
                origin_ms=at,aggregate=result,differences=differences,
                aggregate_records_n=len(group),raw_trades_n=raw_count,
                zero_trade_minute=not group),c


def validate_trade_window(original15m, minutes15, trades, *, previous, following,
                          received_at_ms):
    """Accept only a full guarded, source-compatible15m aggregate trade path.

    Supplied ``trades`` are in-window records only, already in source order;
    the preceding/following witness prints are separate and never aggregated.
    """
    try:
        reference=validate_window(original15m,minutes15,interval_ms=60_000,
                                  received_at_ms=received_at_ms)
    except (ValueError,TypeError,ArithmeticError):
        return _output('INCOMPLETE',['MINUTE_REFERENCE_INVALID'])
    if reference['state']!='MATCH':
        return _output(reference['state'],['MINUTE_REFERENCE_NOT_MATCHED']+reference['reasons'],
                       reference=reference)
    origin,end=original15m[0],original15m[0]+STEP
    if not isinstance(trades,list) or not trades:
        return _output('INCOMPLETE',['TRADE_WINDOW_EMPTY_OR_SCHEMA'],reference=reference,
                       origin_ms=origin,end_ms=end)
    try:
        before,after=_trade(previous),_trade(following)
        parsed=[_trade(t) for t in trades]
    except (ValueError,TypeError,ArithmeticError):
        return _output('INCOMPLETE',['TRADE_OR_GUARD_INVALID'],reference=reference,
                       origin_ms=origin,end_ms=end)
    if before['T']>=origin or after['T']<end:
        return _output('INCOMPLETE',['GUARD_TIME_NOT_OUTSIDE_WINDOW'],reference=reference,
                       origin_ms=origin,end_ms=end)
    if any(not origin<=t['T']<end for t in parsed):
        return _output('INCOMPLETE',['TRADE_OUTSIDE_WINDOW'],reference=reference,
                       origin_ms=origin,end_ms=end)
    sequence=[before]+parsed+[after]
    if any(b['a']!=a['a']+1 for a,b in zip(sequence,sequence[1:])):
        return _output('INCOMPLETE',['AGGREGATE_ID_GAP_OR_CONFLICT'],reference=reference,
                       origin_ms=origin,end_ms=end)
    if any(b['f']!=a['l']+1 for a,b in zip(sequence,sequence[1:])):
        return _output('INCOMPLETE',['RAW_ID_GAP_OR_CONFLICT'],reference=reference,
                       origin_ms=origin,end_ms=end)
    if any(b['T']<a['T'] for a,b in zip(sequence,sequence[1:])):
        return _output('INCOMPLETE',['TRADE_TIME_REVERSAL'],reference=reference,
                       origin_ms=origin,end_ms=end)
    if received_at_ms<after['T']:
        return _output('INCOMPLETE',['RECEIPT_BEFORE_GUARD_TRADE'],reference=reference,
                       origin_ms=origin,end_ms=end)
    minute_rows=[_row(row,60_000,origin+i*60_000) for i,row in enumerate(minutes15)]
    buckets=[[] for _ in range(15)]
    for row in parsed:
        buckets[(row['T']-origin)//60_000].append(row)
    checks=[];reasons=[];last=before['price']
    with localcontext(_exact_context(minute_rows,sequence)):
        for index,(row,group) in enumerate(zip(minute_rows,buckets)):
            checked,last=_aggregate_minute(row,group,last)
            checks.append(checked)
            if checked['state']!='MATCH':
                reasons.append('MINUTE_SOURCE_MISMATCH:'+str(index))
    return _output('MISMATCH' if reasons else 'MATCH',reasons,reference=reference,
        minute_checks=checks,origin_ms=origin,end_ms=end,received_at_ms=received_at_ms,
        aggregate_records_n=len(parsed),raw_trades_n=sum(t['l']-t['f']+1 for t in parsed),
        trades_sha256=canonical_sha256(trades),previous_sha256=canonical_sha256(previous),
        following_sha256=canonical_sha256(following),
        coverage_condition='STRICT_CONSECUTIVE_AGGREGATE_AND_RAW_IDS',
        rounding_contract='ASSUMED_NEAREST_ROUNDING_CAPPED_SOURCE_QUANTUM_1e-8')


def resolve_bracket(validation, trades, *, entry_price):
    """First public print touching a frozen simulated bracket; never price fills."""
    base=dict(state='ORDER_UNRESOLVED',trade_id=None,raw_id=None,at_ms=None,
              stop=None,target=None,native_fill_certified=False,pnl_repriced=False)
    if not isinstance(validation,dict) or validation.get('state')!='MATCH':
        return dict(base,reasons=['SOURCE_VALIDATION_NOT_MATCHED'])
    try:
        if validation.get('trades_sha256')!=canonical_sha256(trades):
            return dict(base,reasons=['TRADE_PATH_BINDING_CHANGED'])
        if isinstance(entry_price,bool):
            raise ValueError()
        entry=float(entry_price)
        if not math.isfinite(entry) or entry<=0:
            raise ValueError()
        # RDE-1 stores binary floats. Reproduce its boundary arithmetic first,
        # then compare the resulting decimal representation to native prints.
        stop_float,target_float=entry*.995,entry*1.005
        if not math.isfinite(target_float) or not 0<stop_float<target_float:
            raise ValueError()
        stop,target=Decimal(str(stop_float)),Decimal(str(target_float))
    except (ValueError,TypeError,OverflowError,ArithmeticError):
        return dict(base,reasons=['INVALID_ENTRY_OR_TRADE_PATH'])
    base.update(stop=str(stop),target=str(target))
    for trade in trades:
        price=_decimal(trade['p'])
        if price<=stop or price>=target:
            return dict(base,state='TRADE_SEQUENCE_SL_FIRST' if price<=stop else 'TRADE_SEQUENCE_TP_FIRST',
                        trade_id=trade['a'],raw_id=trade['f'],at_ms=trade['T'],reasons=[],
                        price=trade['p'],trades_sha256=validation['trades_sha256'])
    return dict(base,reasons=['NO_BRACKET_TOUCH_IN_COMPLETE_WINDOW'])
