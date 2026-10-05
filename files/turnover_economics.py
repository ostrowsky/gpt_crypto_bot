"""Causal observed-amplitude screen and additive simulated cash attribution."""
from bisect import bisect_right
from collections import defaultdict
import math

import numpy as np

BAR=900_000
ARMS=('control','amplitude_cost','no_replacement','combined')


def roundtrip_hurdle_pct(fee_bps,slippage_bps):
    f,s=fee_bps/10000,slippage_bps/10000
    if not (0<=f<1 and 0<=s<1):raise ValueError('invalid costs')
    return 100*((1+f)*(1+s)/((1-f)*(1-s))-1)


def past_hour_range(data,clock):
    closes=data['t']+BAR;i=int(np.searchsorted(closes,clock))
    if i<3 or i>=len(data):return None
    rows=data[i-3:i+1]
    if not np.array_equal(rows['t']+BAR,clock-np.arange(3,-1,-1)*BAR):return None
    a=np.column_stack([rows[n] for n in ('o','h','l','c')])
    if (not np.isfinite(a).all() or (a<=0).any()
            or (a[:,1]<np.maximum(a[:,0],a[:,3])).any()
            or (a[:,2]>np.minimum(a[:,0],a[:,3])).any()
            or (a[:,1]<a[:,2]).any()):return None
    return float((rows['h'].max()/rows['l'].min()-1)*100)


def screened_snapshot(snapshot,cache,fee_bps,slippage_bps):
    threshold=2*roundtrip_hurdle_pct(fee_bps,slippage_bps)
    raw,times,n=snapshot;out={};unknown=0;below=0
    if n!=sum(map(len,raw.values())):raise ValueError('candidate count mismatch')
    for at,rows in raw.items():
        accepted=[]
        for c in rows:
            if c.ts_ms!=at:raise ValueError('candidate clock mismatch')
            value=past_hour_range(cache[c.sym,'15m'][0],at)
            if value is None:unknown+=1
            elif value<threshold:below+=1
            else:accepted.append(c)
        out[at]=accepted
    return (out,set(times),sum(map(len,out.values()))),{'input':n,'unknown_past':unknown,
        'below_cost_amplitude':below,'accepted':sum(map(len,out.values())),
        'threshold_pct':threshold}


def replacement_enabled(arm,control_enabled):
    if arm not in ARMS:raise ValueError('unknown arm')
    return bool(control_enabled) and arm not in ('no_replacement','combined')


def cash_ledger(trades,series,grid,fee_bps=7.5,slippage_bps=5.,initial=10000.,capacity=10):
    """After-cost quantities, independent event accounting, no actual-fill claims."""
    f,s=fee_bps/10000,slippage_bps/10000
    roundtrip_hurdle_pct(fee_bps,slippage_bps)
    if initial<=0 or capacity<=0 or not grid or grid!=list(range(grid[0],grid[-1]+1,BAR)):
        raise ValueError('invalid cash/grid contract')
    lookup={sym:([int(t) for t,p in rows],[float(p) for t,p in rows]) for sym,rows in series.items()}
    events=defaultdict(list);records=[]
    for i,t in enumerate(trades):
        if not grid[0]<=t.entry_ts<=t.exit_ts<=grid[-1] or not all(math.isfinite(v) and v>0 for v in (t.entry_price,t.exit_price)):
            raise ValueError('unknown/invalid trade')
        records.append({'trade_id':i,'symbol':t.sym,'mode':t.mode,'tf':t.tf,
            'entry_ts':t.entry_ts,'exit_ts':t.exit_ts,'exit_reason':t.exit_reason,
            'holding_minutes':(t.exit_ts-t.entry_ts)/60000,
            'budget':0.,'raw_price_pnl':0.,'fees':0.,'slippage':0.,'net_pnl':0.,'turnover':0.})
        events[t.entry_ts].append((2,i,'entry',t.entry_price))
        if t.partial_exit_taken:
            if not (t.entry_ts<t.partial_exit_ts<=t.exit_ts and 0<t.partial_exit_fraction<1 and t.partial_exit_price>0):
                raise ValueError('invalid partial exit')
            events[t.partial_exit_ts].append((0,i,'partial',t.partial_exit_price))
        events[t.exit_ts].append((3 if t.entry_ts==t.exit_ts else 1,i,'exit',t.exit_price))
    cash=initial;positions={};symbols=set();curve=[];exposure=[]
    def mark(at):
        value=cash;gross=0.
        for i,qty in positions.items():
            sym=trades[i].sym
            ts,ps=lookup[sym];j=bisect_right(ts,at)-1
            if j<0 or at-ts[j]>BAR or not math.isfinite(ps[j]) or ps[j]<=0:raise ValueError('missing/stale mark')
            gross+=qty*ps[j];value+=qty*ps[j]*(1-s)*(1-f)
        if not math.isfinite(value) or value<=0:raise ValueError('invalid liquidation equity')
        return value,gross
    grid_set=set(grid)
    for at in sorted(grid_set|set(events)):
        for _,i,kind,price in sorted(events.get(at,[])):
            t=trades[i];r=records[i]
            if kind=='entry':
                if t.sym in symbols or len(positions)>=capacity:raise ValueError('symbol/capacity violation')
                eq,_=mark(at);budget=min(cash,eq/capacity)
                if budget<=0:raise ValueError('no entry cash')
                notional=budget/(1+f);qty=notional/(price*(1+s))
                cash-=budget;positions[i]=qty;symbols.add(t.sym);r['budget']=budget
                r['fees']+=notional*f;r['slippage']+=qty*price*s
                r['raw_price_pnl']-=qty*price;r['turnover']+=qty*price;r['net_pnl']-=budget
            else:
                if i not in positions:raise ValueError('unmatched exit')
                qty=positions[i]*(t.partial_exit_fraction if kind=='partial' else 1.)
                proceeds=qty*price*(1-s);paid_fee=proceeds*f;cash+=proceeds-paid_fee
                r['fees']+=paid_fee;r['slippage']+=qty*price*s
                r['raw_price_pnl']+=qty*price;r['net_pnl']+=proceeds-paid_fee;r['turnover']+=qty*price
                if kind=='exit':del positions[i];symbols.remove(t.sym)
                else:positions[i]-=qty
            if cash < -1e-8:raise ValueError('borrowed cash')
        if at in grid_set:
            eq,gross=mark(at);curve.append((at,eq));exposure.append(gross/eq)
    if positions:raise ValueError('unclosed positions')
    total={k:sum(r[k] for r in records) for k in ('raw_price_pnl','fees','slippage','net_pnl','turnover')}
    if not math.isclose(total['net_pnl'],cash-initial,abs_tol=1e-7):raise ValueError('cash attribution mismatch')
    for r in records:
        if not math.isclose(r['net_pnl'],r['raw_price_pnl']-r['fees']-r['slippage'],abs_tol=1e-8):
            raise ValueError('trade attribution mismatch')
    values=np.array([v for _,v in curve]);peak=np.maximum(initial,np.maximum.accumulate(values))
    return {'ending_cash':cash,'net_return_pct':100*(cash/initial-1),'totals':total,
        'max_drawdown_pct':float((1-values/peak).max()*100),
        'average_gross_exposure_pct':float(np.mean(exposure)*100),'curve':curve,'ledger':records,
        'scope':'simulated additive cash attribution; idealized closed-bar fills'}


def grouped_attribution(ledger,key):
    groups={}
    for r in ledger:
        label=r[key];g=groups.setdefault(label,{'trades':0,'net_wins':0,'net_pnl':0.,'fees':0.,'slippage':0.,'raw_price_pnl':0.,'turnover':0.})
        g['trades']+=1;g['net_wins']+=r['net_pnl']>0
        for k in ('net_pnl','fees','slippage','raw_price_pnl','turnover'):g[k]+=r[k]
    return groups


def acceptance(control,candidate,full_mission,test_control,test_candidate,ci,test_days):
    base,target=full_mission
    checks={'sample':test_days>=30,'full_return_gain':candidate['net_return_pct']-control['net_return_pct']>=1,
        'test_return_gain':candidate['test_return_pct']-control['test_return_pct']>=1,
        'paired_corrected_lower':ci is not None and ci[0]>0,
        'turnover_reduction':candidate['trades']<=.9*control['trades'],
        'full_early_capture':target['early_pair_count']>=base['early_pair_count'],
        'full_capture':target['captured_pair_count']>=base['captured_pair_count'],
        'test_early_capture':test_candidate['early_pair_count']>=test_control['early_pair_count'],
        'test_capture':test_candidate['captured_pair_count']>=test_control['captured_pair_count'],
        'precision':target['objective_trade_count']/target['eligible_trade_count']>=base['objective_trade_count']/base['eligible_trade_count']-.005 if target['eligible_trade_count'] and base['eligible_trade_count'] else False,
        'drawdown':candidate['max_drawdown_pct']<=control['max_drawdown_pct']+1}
    return {'numerical_gate':'PASS' if all(checks.values()) else 'REJECTED' if checks['sample'] else 'INCONCLUSIVE',
            'checks':checks,'runtime_eligible':False,'achievement_claimed':False}
