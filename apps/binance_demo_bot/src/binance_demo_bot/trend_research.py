"""Frozen, credential-free trend challenger and shared-cash OHLC simulation.

Historical close availability and native matching cannot be certified by OHLC.
Nothing in this module authorizes an order, changes risk or trains on actual fills.
"""
from collections import deque
from datetime import datetime,timezone
import heapq
import math
from statistics import median
from zoneinfo import ZoneInfo
from .financial import day_bounds

STEP=900000
POLICY=dict(version='trend-breakout-v1',warmup=32,fast=8,slow=24,breakout=20,momentum_bars=4,
    volume_multiple=2,momentum_min=.005,chase_max=.08,stop=.01,take_profit=.04,
    trail=.015,trail_activation=.01,max_holding_bars=16,cooldown_bars=4,
    cash=100,entry_cash=30,positions=1,day_loss=1,fee_bps=10,slippage_bps=5)


def _valid(row):
    try:
        prices=[float(row[i]) for i in (1,2,3,4)]
        return (type(row[0]) is int and type(row[6]) is int and row[0]%STEP==0
            and row[6]==row[0]+STEP-1 and all(math.isfinite(v) and v>0 for v in prices)
            and prices[2]<=min(prices[0],prices[3])<=max(prices[0],prices[3])<=prices[1]
            and math.isfinite(float(row[7])) and float(row[7])>=0)
    except (IndexError,TypeError,ValueError):return False


def trend_signal(rows,*,now_ms,received_at_ms=None):
    if type(now_ms) is not int or (received_at_ms is not None and
        (type(received_at_ms) is not int or received_at_ms>now_ms)):
        return dict(action='BLOCK',reason='FEATURE_NOT_AVAILABLE')
    closed=[r for r in rows if r[6]<now_ms][-POLICY['warmup']:]
    if len(closed)<POLICY['warmup']:return dict(action='BLOCK',reason='WARMUP')
    if (not all(_valid(r) for r in closed) or
        any(b[0]-a[0]!=STEP for a,b in zip(closed,closed[1:])) or now_ms-closed[-1][6]>STEP):
        return dict(action='BLOCK',reason='INVALID_OR_GAPPED_BARS')
    prices=[float(r[4]) for r in closed]
    fast=sum(prices[-POLICY['fast']:])/POLICY['fast'];slow=sum(prices[-POLICY['slow']:])/POLICY['slow']
    prior=closed[-(POLICY['breakout']+1):-1]
    volume=median(float(r[7]) for r in prior);multiple=float(closed[-1][7])/volume if volume else 0
    momentum=prices[-1]/prices[-(POLICY['momentum_bars']+1)]-1
    buy=(prices[-1]>max(float(r[2]) for r in prior) and fast>slow and momentum>=POLICY['momentum_min']
        and multiple>=POLICY['volume_multiple'] and prices[-1]/slow-1<=POLICY['chase_max'])
    return dict(action='BUY' if buy else 'WAIT',reason='BREAKOUT_VOLUME' if buy else 'NO_EDGE',
        score=momentum*multiple,weakening=fast<slow,feature_end_ms=closed[-1][6])


def protective_exit(position,row):
    entry,peak=position['entry'],position['peak'];fixed=entry*(1-POLICY['stop']);stop=fixed
    if peak>=entry*(1+POLICY['trail_activation']):stop=max(stop,peak*(1-POLICY['trail']))
    reason='TRAIL' if stop>fixed else 'STOP'
    opening,high,low=map(float,(row[1],row[2],row[3]))
    if opening<=stop:return opening,reason
    if low<=stop:return stop,reason
    target=entry*(1+POLICY['take_profit'])
    if high>=target:return target,'TP'
    return None


def _events(series):
    heap=[];iterators={s:iter(v) for s,v in sorted(series.items())};previous={}
    def push(symbol):
        try:row=next(iterators[symbol])
        except StopIteration:return
        if symbol in previous and row[0]<=previous[symbol]:raise ValueError('NON_MONOTONE_HISTORY')
        previous[symbol]=row[0];heapq.heappush(heap,(row[0],symbol,row))
    for s in iterators:push(s)
    while heap:
        at=heap[0][0];group={}
        while heap and heap[0][0]==at:
            _,symbol,row=heapq.heappop(heap);group[symbol]=row;push(symbol)
        yield at,group


def _day(at):return datetime.fromtimestamp(at/1000,timezone.utc).astimezone(ZoneInfo('Europe/Budapest')).date().isoformat()


def portfolio_replay(series,*,start_ms,end_ms,fee_bps=10,slippage_bps=5):
    if (type(start_ms) is not int or type(end_ms) is not int or not start_ms<end_ms
        or not 0<=fee_bps<=100 or not 0<=slippage_bps<=100):raise ValueError('INVALID_REPLAY_CONTRACT')
    initial=float(POLICY['cash']);history={s:deque(maxlen=POLICY['warmup']) for s in series};cash=initial
    position=None;trades=[];fees=0.;turnover=0.
    high_water=initial;max_dd=0.;equity=initial;cooldown={};reasons=[];day_state={};max_concurrent=0;max_debit=0.
    fee,slip=fee_bps/10000,slippage_bps/10000
    def sell(price,at,why):
        nonlocal cash,position,fees,turnover
        gross=position['quantity']*price*(1-slip);cost=gross*fee
        cash+=gross-cost;fees+=cost;turnover+=gross
        trades.append(dict(symbol=position['symbol'],entry_ms=position['entry_ms'],exit_ms=at,
            entry_price=position['entry'],exit_price=price*(1-slip),reason=why,
            net_usdt=gross-cost-position['debit']))
        cooldown[position['symbol']]=at+POLICY['cooldown_bars']*STEP;position=None
    for at,group in _events(series):
        if at>=end_ms:break
        if at<start_ms:
            for s,row in group.items():history[s].append(row)
            continue
        if position and (position['symbol'] not in group or at-position['last_ms']!=STEP):
            reasons.append('HELD_PRICE_GAP');break
        day=_day(at)
        daily=day_state.setdefault(day,dict(start_equity=equity,peak=equity,end_equity=equity,first_ms=at,last_ms=at,points_n=0,killed=False))
        if position:
            s=position['symbol'];row=group[s]
            if not _valid(row):reasons.append('HELD_PRICE_INVALID');break
            signal=trend_signal(history[s],now_ms=at)
            # A previously known weakening/holding expiry exits at this open.
            if at-position['entry_ms']>=POLICY['max_holding_bars']*STEP or signal.get('weakening'):
                sell(float(row[1]),at,'HOLDING_LIMIT' if at-position['entry_ms']>=POLICY['max_holding_bars']*STEP else 'WEAKENING')
        opening_equity=cash+(position['quantity']*float(group[position['symbol']][1]) if position else 0)
        daily['peak']=max(daily['peak'],opening_equity)
        if daily['start_equity']-opening_equity>=POLICY['day_loss'] or daily['peak']-opening_equity>=POLICY['day_loss']:daily['killed']=True
        if position is None and not daily['killed']:
            candidates=[]
            for s,row in group.items():
                if at<cooldown.get(s,0):continue
                signal=trend_signal(history[s],now_ms=at)
                if signal['action']=='BUY':candidates.append((-signal['score'],s))
            if candidates:
                _,s=min(candidates)
                if not _valid(group[s]):reasons.append('ENTRY_BAR_INVALID');break
                price=float(group[s][1])*(1+slip);debit=min(float(POLICY['entry_cash']),cash)
                if debit>0:
                    gross=debit/(1+fee);quantity=gross/price;cash-=debit;fees+=debit-gross;turnover+=gross
                    position=dict(symbol=s,entry=price,peak=price,quantity=quantity,debit=debit,entry_ms=at,last_ms=at-STEP)
                    max_concurrent=1;max_debit=max(max_debit,debit)
        if position:
            s=position['symbol'];row=group[s];exit=protective_exit(position,row)
            if exit:sell(exit[0],at,exit[1])
            else:
                position['peak']=max(position['peak'],float(row[2]));position['last_ms']=at
        equity=cash+(position['quantity']*float(group[position['symbol']][4]) if position else 0)
        high_water=max(high_water,equity);max_dd=max(max_dd,high_water-equity)
        daily.update(end_equity=equity,last_ms=at,peak=max(daily['peak'],equity),points_n=daily['points_n']+1)
        if daily['start_equity']-equity>=POLICY['day_loss'] or daily['peak']-equity>=POLICY['day_loss']:daily['killed']=True
        for s,row in group.items():history[s].append(row)
    if position and position['last_ms']+STEP<end_ms and not reasons:reasons.append('HELD_PRICE_GAP')
    complete_days=[]
    for day,daily in day_state.items():
        first,last=day_bounds(day,'Europe/Budapest')
        complete=(daily['first_ms']==first and daily['last_ms']+STEP==last and daily['points_n']==(last-first)//STEP)
        daily.update(calendar_status='COMPLETE_SIMULATION' if complete else 'PARTIAL_SIMULATION',
            net_pnl_usdt=daily['end_equity']-daily['start_equity'])
        if complete:complete_days.append(daily['net_pnl_usdt'])
    profitable=sum(pnl>0 for pnl in complete_days)
    return dict(state='INCOMPLETE' if reasons else 'COMPLETE',policy=POLICY['version'],evidence_kind='SIMULATION',
        historical_receipts_certified=False,point_in_time_universe_certified=False,execution_parity_certified=False,
        start_ms=start_ms,end_ms=end_ms,fee_bps=fee_bps,slippage_bps=slippage_bps,initial_cash_usdt=initial,
        final_equity_usdt=None if reasons else equity,net_pnl_usdt=None if reasons else equity-initial,
        benchmark='USDT_NO_TRADE_SAME_100',benchmark_net_pnl_usdt=0,trades_n=len(trades),trades=trades,
        open_positions_n=int(position is not None),max_concurrent_positions=max_concurrent,max_entry_debit_usdt=max_debit,
        complete_days_N=len(complete_days),profitable_days_n=None if reasons else profitable,
        mean_daily_net_pnl_usdt=None if reasons or not complete_days else sum(complete_days)/len(complete_days),
        profitable_days_ratio=None if reasons or not complete_days else profitable/len(complete_days),
        fees_usdt=fees,turnover_usdt=turnover,max_drawdown_usdt=max_dd,days=day_state,reasons=reasons,
        promotion_allowed=False)


def time_splits(start,end):
    if type(start) is not int or type(end) is not int or end-start<20*STEP:raise ValueError('SPLIT_PERIOD_TOO_SHORT')
    duration=end-start;a=start+(duration*70//100//STEP)*STEP;b=start+(duration*85//100//STEP)*STEP
    return [('train',start,a),('tune',a,b),('test',b,end)]


def promotion_readiness(replay,*,cohort_complete):
    reasons=['FINANCIAL_CERTIFICATION_PENDING','HISTORICAL_PIT_RECEIPTS_UNCERTIFIED',
        'ACTUAL_FORWARD_FINANCIAL_COHORT_PENDING','FULL_TRUTH_HARNESS_PASS_REQUIRED']
    if not cohort_complete:reasons.append('MAXIMUM_COHORT_INCOMPLETE')
    if replay['state']!='COMPLETE':reasons.append('REPLAY_INCOMPLETE')
    if replay.get('net_pnl_usdt') is None or replay['net_pnl_usdt']<=0:reasons.append('POSITIVE_NET_RESULT_NOT_ESTABLISHED')
    return dict(state='RESEARCH_ONLY',promotion_allowed=False,training_state='NOT_TRAINED',reasons=reasons)
