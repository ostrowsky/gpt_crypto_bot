"""Independent RFP ledger arithmetic, source ownership and calendar audit."""
from collections import Counter
from decimal import Decimal as D
import hashlib
import json
from .financial import decimal_amount,day_bounds,_arithmetic


def _same(actual,expected,label):
    if actual is None or expected is None:
        if actual is not None or expected is not None:raise ValueError('AUDIT_'+label)
    elif abs(decimal_amount(actual)-expected)>D('1e-35'):raise ValueError('AUDIT_'+label)


@_arithmetic
def audit_events(events,result,policy,*,quote_at=None):
    cash=decimal_amount(policy.cash);initial=cash;qty=D(0);symbol=None;basis=D(0)
    fees=turnover=closed_net=D(0);entries=closed=0;counts=Counter();decisions={};boundaries=[]
    buy_orders={};sell_orders={}
    unknown=set();last_at=None;first_at=None;last_decision=None;decision_n=0
    fee_rate=decimal_amount(policy.fee_bps)/10000;slip=decimal_amount(policy.slippage_bps)/10000
    entry_time=origin_time=None;owner=None;peak=cash;drawdown=D(0);previous_close=initial

    def observe(value):
        nonlocal peak,drawdown
        if value is not None:peak=max(peak,value);drawdown=max(drawdown,peak-value)

    def mark(q,t):
        if symbol is None:
            if q is not None:raise ValueError('AUDIT_FLAT_QUOTE')
            return cash
        if quote_at is not None and q!=quote_at(symbol,t):raise ValueError('AUDIT_SOURCE_QUOTE')
        if q is None:unknown.add(t);return None
        if q['symbol']!=symbol or not 0<t-q['receipt_ms']<=1000:raise ValueError('AUDIT_SOURCE_CLOCK')
        b,a=decimal_amount(q['bid']),decimal_amount(q['ask'])
        if b<=0 or a<b:raise ValueError('AUDIT_SOURCE_PRICE')
        return cash+qty*(b+a)/2

    for e in events:
        t=e['at_ms'];kind=e['kind'];counts[kind]+=1
        if e['evidence_kind']!='SIMULATION' or last_at is not None and t<last_at:raise ValueError('AUDIT_EVENT_CLOCK')
        last_at=t
        if kind=='INITIAL_EQUITY':
            if first_at is not None:raise ValueError('AUDIT_DOUBLE_INITIAL')
            first_at=t;_same(e['cash'],cash,'INITIAL_CASH');_same(e['equity'],initial,'INITIAL_EQUITY')
        elif kind=='DECISION':
            if t%5000 or last_decision is not None and t<=last_decision:raise ValueError('AUDIT_DECISION_CLOCK')
            last_decision=t;decision_n+=1
            payload={k:v for k,v in e.items() if k not in ('kind','at_ms','evidence_kind','decision_id')}
            expected=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            if e['decision_id']!=expected or e['origin_ms']!=t:raise ValueError('AUDIT_DECISION_HASH')
            _same(e['cash'],cash,'DECISION_CASH')
            if e['position_symbol']!=symbol:raise ValueError('AUDIT_DECISION_INVENTORY')
            if e['selected'] is not None:decisions[e['decision_id']]=(t,e['selected'])
        elif kind=='BUY_INTENT':
            identity=e['decision_id']
            if (identity not in decisions or identity in buy_orders or symbol is not None or
                    e['origin_ms']!=t or e['eligible_ms']!=t+policy.latency_ms or
                    e['expires_ms']!=e['eligible_ms']+2000 or e['signal']!=decisions[identity][1]):
                raise ValueError('AUDIT_BUY_INTENT')
            buy_orders[identity]=e
        elif kind in ('BUY_CANCELLED','BUY_EXPIRED','BUY_REJECTED_CAPACITY'):
            if buy_orders.pop(e['decision_id'],None) is None:raise ValueError('AUDIT_UNOWNED_BUY_END')
        elif kind=='SELL_INTENT':
            if (symbol!=e['symbol'] or owner!=e['decision_id'] or owner in sell_orders or
                    e['decision_ms']!=t or e['eligible_ms']!=t+policy.latency_ms or t<entry_time):
                raise ValueError('AUDIT_SELL_INTENT')
            sell_orders[owner]={k:e[k] for k in ('decision_ms','eligible_ms','reason')}
        elif kind=='BUY_FILL':
            if symbol is not None or e['decision_id'] not in buy_orders:raise ValueError('AUDIT_UNOWNED_BUY')
            origin,selected=decisions[e['decision_id']]
            if (selected['symbol']!=e['symbol'] or e['origin_ms']!=origin or
                    not origin+policy.latency_ms<=t<origin+policy.latency_ms+2000):raise ValueError('AUDIT_BUY_TIMING')
            q=e['quote'];symbol=e['symbol'];qty=decimal_amount(e['quantity']);basis=decimal_amount(e['debit'])
            if quote_at is not None and q!=quote_at(symbol,t):raise ValueError('AUDIT_SOURCE_QUOTE')
            if q['symbol']!=symbol or not 0<t-q['receipt_ms']<=1000:raise ValueError('AUDIT_BUY_QUOTE')
            expected_price=decimal_amount(q['ask'])*(1+slip);notional=qty*expected_price;commission=notional*fee_rate
            if qty<=0 or qty>decimal_amount(q['ask_qty']) or basis>cash or basis>decimal_amount(policy.max_debit) or basis<decimal_amount(policy.min_debit):
                raise ValueError('AUDIT_BUY_CAPACITY')
            _same(e['price'],expected_price,'BUY_PRICE');_same(e['notional'],notional,'BUY_NOTIONAL')
            _same(e['fee'],commission,'BUY_FEE');_same(e['debit'],notional+commission,'BUY_DEBIT')
            cash-=basis;fees+=commission;turnover+=notional;entries+=1;entry_time=t;origin_time=origin;owner=e['decision_id']
            del buy_orders[owner]
            if e['entry_ms']!=t or e['deadline_ms']!=t+policy.horizon_seconds*1000:raise ValueError('AUDIT_HOLDING_DEADLINE')
            _same(e['cash'],cash,'BUY_CASH')
        elif kind=='SELL_FILL':
            if symbol!=e['symbol'] or owner!=e['decision_id']:raise ValueError('AUDIT_UNOWNED_SELL')
            _same(e['quantity'],qty,'SELL_QUANTITY');_same(e['entry_debit'],basis,'SELL_BASIS')
            intent=e['exit_intent'];due=intent['decision_ms']
            if (sell_orders.pop(owner,None)!=intent or due<entry_time or t<intent['eligible_ms'] or intent['eligible_ms']!=due+policy.latency_ms or
                    intent['reason']=='HOLDING_DEADLINE' and due!=entry_time+policy.horizon_seconds*1000 or
                    e['entry_ms']!=entry_time or e['entry_decision_ms']!=origin_time):raise ValueError('AUDIT_SELL_TIMING')
            q=e['quote']
            if quote_at is not None and q!=quote_at(symbol,t):raise ValueError('AUDIT_SOURCE_QUOTE')
            if q['symbol']!=symbol or not 0<t-q['receipt_ms']<=1000 or qty>decimal_amount(q['bid_qty']):raise ValueError('AUDIT_SELL_QUOTE')
            price=decimal_amount(q['bid'])*(1-slip);notional=qty*price;commission=notional*fee_rate
            _same(e['price'],price,'SELL_PRICE');_same(e['notional'],notional,'SELL_NOTIONAL');_same(e['fee'],commission,'SELL_FEE')
            net=notional-commission-basis;cash+=notional-commission;fees+=commission;turnover+=notional;closed_net+=net;closed+=1
            _same(e['cash'],cash,'SELL_CASH');_same(e['trade_net_usdt'],net,'TRADE_NET')
            symbol=None;qty=basis=D(0);owner=None;observe(cash)
        elif kind=='MARK':
            value=mark(e['quote'],t);_same(e['equity'],value,'MARK_EQUITY');_same(e['cash'],cash,'MARK_CASH');observe(value)
            if symbol is None or e['position']['symbol']!=symbol:raise ValueError('AUDIT_MARK_INVENTORY')
            _same(e['position']['quantity'],qty,'MARK_QUANTITY')
        elif kind=='BOUNDARY':
            value=mark(e['quote'],t);_same(e['closing_equity'],value,'BOUNDARY_EQUITY')
            _same(e['opening_equity'],previous_close,'BOUNDARY_OPEN')
            previous_close=value;observe(value);boundaries.append(e)

    if first_at is None or not boundaries:raise ValueError('AUDIT_MISSING_BOUNDARIES')
    final=previous_close;net=final-initial if final is not None else None
    for key,value in (('cash',cash),('fees_usdt',fees),('turnover_usdt',turnover),('closed_net_usdt',closed_net),
                      ('final_equity',final),('net_interval_usdt',net),('open_unrealized_usdt',net-closed_net if net is not None else None),
                      ('drawdown_observed_usdt',drawdown)):_same(result[key],value,key.upper())
    if (result['entries_n']!=entries or result['closed_cycles_n']!=closed or result['event_counts']!=dict(counts) or
            result['unmarked_held_ticks_n']!=len(unknown)):raise ValueError('AUDIT_COUNTERS')
    if (result['position'] is None)!=(symbol is None):raise ValueError('AUDIT_FINAL_INVENTORY')
    if symbol is not None:_same(result['position']['quantity'],qty,'FINAL_QUANTITY')
    expected_decisions=(last_at-1)//5000-((first_at+4999)//5000)+1
    if decision_n!=expected_decisions:raise ValueError('AUDIT_MISSING_DECISIONS')
    if len(boundaries)!=len(result['daily']):raise ValueError('AUDIT_DAILY_N')
    complete=[];requested=0
    for e,r in zip(boundaries,result['daily']):
        lo,hi=day_bounds(e['day'],'Europe/Budapest');full=e['start_ms']==lo and e['end_ms']==hi
        opening=D(e['opening_equity']) if e['opening_equity'] is not None else None
        closing=D(e['closing_equity']) if e['closing_equity'] is not None else None
        delta=closing-opening if closing is not None and opening is not None else None
        good=full and delta is not None and not any(e['start_ms']<=t<=e['end_ms'] for t in unknown)
        requested+=int(full)
        if good:complete.append(delta)
        if r['complete']!=good or r['full_calendar_day']!=full:raise ValueError('AUDIT_DAILY_COMPLETENESS')
        _same(r['pnl_usdt'],delta if good else None,'DAILY_PNL');_same(r['boundary_delta_usdt'],delta,'DAILY_DELTA')
    if (result['complete_days_n']!=len(complete) or result['requested_days_N']!=requested or
            result['profitable_days_n']!=sum(v>0 for v in complete)):raise ValueError('AUDIT_DAILY_DENOMINATOR')
    _same(result['mean_daily_net_pnl_usdt'],sum(complete,D(0))/len(complete) if complete else None,'DAILY_MEAN')
    return dict(state='PASS_LEDGER_ARTIFACT_AUDIT_ONLY',events_n=sum(counts.values()),entries_n=entries,
                closed_cycles_n=closed,complete_days_n=len(complete),financial_certification=False,promotion_allowed=False)
