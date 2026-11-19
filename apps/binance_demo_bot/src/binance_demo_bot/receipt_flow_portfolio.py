"""RFP-1 Decimal shared-cash conditional simulation. No exchange/credential I/O."""
from collections import Counter
from dataclasses import dataclass,asdict
from decimal import Decimal
import hashlib
import json
import math
from .financial import decimal_amount,day_bounds,day_for_timestamp,_arithmetic

D=Decimal


def _json_value(value):
    if isinstance(value,D):return str(value)
    if isinstance(value,dict):return {k:_json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_value(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(_json_value(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Policy:
    horizon_seconds:int=5
    admission:str='direction'
    fee_bps:str='10'
    slippage_bps:str='5'
    latency_ms:int=1000
    cash:str='100'
    max_debit:str='30'
    min_debit:str='5'
    day_loss:str='1'
    global_drawdown:str='5'
    cooldown_ms:int=5000

    def __post_init__(self):
        if self.horizon_seconds not in (5,10,30) or self.admission not in ('direction','cost-aware'):
            raise ValueError('POLICY_CONTRACT')
        if self.latency_ms not in (1000,2000) or self.cooldown_ms!=5000:raise ValueError('POLICY_CLOCK')
        for name in ('cash','max_debit','min_debit','day_loss','global_drawdown'):
            if decimal_amount(getattr(self,name))<=0:raise ValueError('POLICY_AMOUNT')
        for name in ('fee_bps','slippage_bps'):
            if not 0<=decimal_amount(getattr(self,name))<10000:raise ValueError('POLICY_COST')
        if decimal_amount(self.min_debit)>decimal_amount(self.max_debit):raise ValueError('POLICY_DEBIT')


def calendar_bounds(day):return day_bounds(day,'Europe/Budapest')


def select_signal(rows,policy):
    admissible=[]
    for row in rows:
        score=float(row['score_bps']);spread=float(row['spread_bps'])
        if not math.isfinite(score) or not math.isfinite(spread) or spread<0:raise ValueError('SIGNAL_NUMBER')
        threshold=(D(str(spread))+2*(decimal_amount(policy.fee_bps)+decimal_amount(policy.slippage_bps))+1
                   if policy.admission=='cost-aware' else D(0))
        if D(str(score))>threshold:admissible.append(row)
    return min(admissible,key=lambda r:(-r['score_bps'],r['symbol'],r['source_row'])) if admissible else None


def fit_microprice(rows,*,train_start_ms,train_end_ms):
    train=[]
    for r in rows:
        if not train_start_ms<=r['at_ms']<train_end_ms:continue
        if not r['feature_end_ms']<r['at_ms']<r['label_available_ms']:raise ValueError('BASELINE_TIMING')
        if r['label_available_ms']>=train_end_ms:continue
        if r['target_bps'] is None:continue
        x,y=float(r['micro_bps']),float(r['target_bps'])
        if not math.isfinite(x) or not math.isfinite(y):raise ValueError('BASELINE_NUMBER')
        train.append((r['at_ms'],x,y))
    if len(train)<2:raise ValueError('BASELINE_TRAIN_N')
    train.sort();n=len(train);mx=math.fsum(x for _,x,y in train)/n;my=math.fsum(y for _,x,y in train)/n
    variance=math.fsum((x-mx)**2 for _,x,y in train)
    slope=math.fsum((x-mx)*(y-my) for _,x,y in train)/variance if variance else 0.
    return dict(intercept=my-slope*mx,slope=slope,train_rows_n=n,
        features_sha256=digest([(t,x) for t,x,y in train]),labels_sha256=digest([(t,y) for t,x,y in train]))


@_arithmetic
def simulate(signals,*,start_ms,end_ms,quote_at,policy,emit=None):
    if (type(start_ms) is not int or type(end_ms) is not int or start_ms>=end_ms or
            start_ms%1000 or end_ms%1000):raise ValueError('INTERVAL_CLOCK')
    fee=decimal_amount(policy.fee_bps)/10000;slip=decimal_amount(policy.slippage_bps)/10000
    cash=initial=decimal_amount(policy.cash);max_debit=decimal_amount(policy.max_debit)
    minimum=decimal_amount(policy.min_debit);loss_limit=decimal_amount(policy.day_loss)
    global_limit=decimal_amount(policy.global_drawdown)
    position=buy=sell=None;cooldown=0;entries=closed=unmarked=0;last_unmarked_ms=None
    fees=turnover=closed_net=D(0);global_peak=initial;global_halt=False;max_dd=D(0)
    day=day_for_timestamp(start_ms,'Europe/Budapest');day_open=initial;day_peak=initial
    day_halt=False;day_gaps=0;day_begin=start_ms;daily=[];events=[] if emit is None else None;counts=Counter()

    def event(kind,t,**payload):
        counts[kind]+=1
        row=_json_value(dict(kind=kind,at_ms=t,evidence_kind='SIMULATION',**payload))
        if emit is None:events.append(row)
        else:emit(row)

    def get_quote(symbol,t):
        q=quote_at(symbol,t)
        if q is None:return None
        if q['symbol']!=symbol or not 0<t-q['receipt_ms']<=1000:raise ValueError('QUOTE_SOURCE_TIMING')
        b,a=decimal_amount(q['bid']),decimal_amount(q['ask'])
        if b<=0 or a<b or min(decimal_amount(q[k]) for k in ('bid_qty','ask_qty'))<=0:
            raise ValueError('QUOTE_SOURCE_PRICE')
        return q

    def equity(t):
        if position is None:return cash,None
        q=get_quote(position['symbol'],t)
        if q is None:return None,None
        return cash+position['quantity']*(decimal_amount(q['bid'])+decimal_amount(q['ask']))/2,q

    event('INITIAL_EQUITY',start_ms,cash=cash,equity=initial,policy=asdict(policy))
    t=start_ms;final_equity=initial;final_quote=None
    while True:
        boundary=calendar_bounds(day)[1]
        if t>start_ms and (t==boundary or t==end_ms):
            value,q=equity(t)
            if position is not None and value is None:
                day_gaps+=1
                if last_unmarked_ms!=t:unmarked+=1;last_unmarked_ms=t
            if value is not None:
                global_peak=max(global_peak,value);max_dd=max(max_dd,global_peak-value)
            lo,hi=calendar_bounds(day);delta=value-day_open if value is not None and day_open is not None else None
            full=day_begin==lo and t==hi
            complete=full and delta is not None and day_gaps==0
            record=dict(day=day,start_ms=day_begin,end_ms=t,full_calendar_day=full,
                opening_equity=day_open,closing_equity=value,boundary_delta_usdt=delta,
                pnl_usdt=delta if complete else None,complete=complete,unmarked_held_ticks_n=day_gaps)
            daily.append(_json_value(record));event('BOUNDARY',t,**record,quote=q)
            if t==end_ms:final_equity,final_quote=value,q;break
            day=day_for_timestamp(t,'Europe/Budapest');day_begin=t;day_open=value;day_peak=value
            boundary=calendar_bounds(day)[1]
            day_halt=value is None;day_gaps=0
            if day_halt:event('RISK_LATCH',t,reason='UNKNOWN_DAY_OPEN')

        if sell is not None and t>=sell['eligible_ms'] and position is not None:
            q=get_quote(position['symbol'],t)
            if q is not None and decimal_amount(q['bid_qty'])>=position['quantity']:
                notional=position['quantity']*decimal_amount(q['bid'])*(1-slip);commission=notional*fee
                proceeds=notional-commission;cash+=proceeds;fees+=commission;turnover+=notional
                net=proceeds-position['debit'];closed_net+=net;closed+=1
                event('SELL_FILL',t,symbol=position['symbol'],quantity=position['quantity'],
                    price=decimal_amount(q['bid'])*(1-slip),notional=notional,fee=commission,
                    cash=cash,trade_net_usdt=net,entry_debit=position['debit'],
                    entry_ms=position['entry_ms'],entry_decision_ms=position['origin_ms'],
                    decision_id=position['decision_id'],exit_intent=sell,quote=q)
                position=sell=None;cooldown=t+policy.cooldown_ms
            else:event('SELL_WAIT',t,reason='NO_FRESH_QUOTE' if q is None else 'BEST_BID_CAPACITY',quote=q)

        if buy is not None:
            if day_halt or global_halt:
                event('BUY_CANCELLED',t,reason='RISK_LATCH',decision_id=buy['decision_id']);buy=None
            elif t>=buy['expires_ms']:
                event('BUY_EXPIRED',t,decision_id=buy['decision_id']);buy=None;cooldown=t+policy.cooldown_ms
            elif t>=buy['eligible_ms']:
                q=get_quote(buy['signal']['symbol'],t)
                if q is None:event('BUY_WAIT',t,decision_id=buy['decision_id'],reason='NO_FRESH_QUOTE')
                else:
                    debit=min(max_debit,cash);price=decimal_amount(q['ask'])*(1+slip)
                    quantity=debit/(price*(1+fee))
                    if debit<minimum or decimal_amount(q['ask_qty'])<quantity:
                        event('BUY_REJECTED_CAPACITY',t,decision_id=buy['decision_id'],quote=q)
                        buy=None;cooldown=t+policy.cooldown_ms
                    else:
                        notional=quantity*price;commission=notional*fee
                        cash-=debit;fees+=commission;turnover+=notional;entries+=1
                        position=dict(symbol=buy['signal']['symbol'],quantity=quantity,debit=debit,
                            entry_ms=t,origin_ms=buy['origin_ms'],decision_id=buy['decision_id'],
                            deadline_ms=t+policy.horizon_seconds*1000)
                        event('BUY_FILL',t,**position,price=price,notional=notional,fee=commission,cash=cash,quote=q)
                        buy=None

        value,q=equity(t)
        if position is not None:
            if value is None:
                day_gaps+=1
                if last_unmarked_ms!=t:unmarked+=1;last_unmarked_ms=t
            event('MARK',t,equity=value,cash=cash,position=position,quote=q)
        if value is not None:
            global_peak=max(global_peak,value);max_dd=max(max_dd,global_peak-value)
            if day_peak is not None:day_peak=max(day_peak,value)
            if not global_halt and global_peak-value>=global_limit:
                global_halt=True;event('RISK_LATCH',t,reason='GLOBAL_DRAWDOWN')
            if not day_halt and day_open is not None and (day_open-value>=loss_limit or day_peak-value>=loss_limit):
                day_halt=True;event('RISK_LATCH',t,reason='DAILY_LOSS_OR_DRAWDOWN')
        if (day_halt or global_halt) and buy is not None:
            event('BUY_CANCELLED',t,reason='RISK_LATCH',decision_id=buy['decision_id']);buy=None
        if position is not None and sell is None and (t>=position['deadline_ms'] or day_halt or global_halt):
            sell=dict(decision_ms=t,eligible_ms=t+policy.latency_ms,
                      reason='RISK' if day_halt or global_halt else 'HOLDING_DEADLINE')
            event('SELL_INTENT',t,**sell,symbol=position['symbol'],decision_id=position['decision_id'])

        if t%5000==0:
            rows=signals.get(t,[]);selected=None
            if any(r.get('feature_end_ms',t-1)>=t or r.get('at_ms',t)!=t for r in rows):raise ValueError('SIGNAL_TIMING')
            reason=('POSITION_OR_INTENT' if position is not None or buy is not None or sell is not None else
                    'RISK_LATCH' if day_halt or global_halt else 'COOLDOWN' if t<cooldown else
                    'INSUFFICIENT_CASH' if cash<minimum else None)
            if reason is None:
                selected=select_signal(rows,policy)
                if selected is None:reason='NO_SIGNAL' if not rows else 'NO_ADMISSIBLE_EDGE'
            payload=dict(origin_ms=t,cash=cash,position_symbol=position['symbol'] if position else None,
                day_halt=day_halt,global_halt=global_halt,candidate_rows=[r['source_row'] for r in rows],
                selected=selected,reason=reason)
            decision_id=digest(payload);event('DECISION',t,decision_id=decision_id,**payload)
            if selected is not None:
                if 'origin_mid' in selected:
                    original=get_quote(selected['symbol'],t)
                    if original is None:raise ValueError('ORIGIN_QUOTE_COMPATIBILITY')
                    mid=(decimal_amount(original['bid'])+decimal_amount(original['ask']))/2
                    spread=(decimal_amount(original['ask'])-decimal_amount(original['bid']))/mid*10000
                    if (abs(mid-D(str(selected['origin_mid'])))>max(D('.00000001'),mid*D('1e-12')) or
                            abs(spread-D(str(selected['spread_bps'])))>D('1e-8')):raise ValueError('ORIGIN_QUOTE_COMPATIBILITY')
                buy=dict(origin_ms=t,eligible_ms=t+policy.latency_ms,
                    expires_ms=t+policy.latency_ms+2000,signal=selected,decision_id=decision_id)
                event('BUY_INTENT',t,**buy)
        next_t=t+1000 if position is not None or buy is not None or sell is not None else (t//5000+1)*5000
        t=min(next_t,boundary,end_ms)

    complete=[D(r['pnl_usdt']) for r in daily if r['complete']]
    net=final_equity-initial if final_equity is not None else None
    unrealized=net-closed_net if net is not None else None
    result=dict(state='CONDITIONAL_SIMULATION',cash=cash,position=position,pending_buy=buy,pending_sell=sell,
        initial_equity=initial,final_equity=final_equity,net_interval_usdt=net,closed_net_usdt=closed_net,
        open_unrealized_usdt=unrealized,fees_usdt=fees,turnover_usdt=turnover,entries_n=entries,
        closed_cycles_n=closed,daily=daily,complete_days_n=len(complete),requested_days_N=sum(r['full_calendar_day'] for r in daily),
        mean_daily_net_pnl_usdt=sum(complete,D(0))/len(complete) if complete else None,
        profitable_days_n=sum(v>0 for v in complete),drawdown_observed_usdt=max_dd,
        unmarked_held_ticks_n=unmarked,event_counts=dict(counts),events=events,
        liquidation_equity=(cash+position['quantity']*decimal_amount(final_quote['bid'])*(1-slip)*(1-fee)
                            if position is not None and final_quote is not None else cash if position is None else None),
        financial_certification=False,promotion_allowed=False)
    return _json_value(result)
