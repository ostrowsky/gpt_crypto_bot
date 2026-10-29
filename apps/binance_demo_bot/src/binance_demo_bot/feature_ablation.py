"""FSA-1 causal features and matched, shared-cash research simulation.

Candle taker volume is an aggregate proxy. Neither retrospective publication
latency nor native fills are certified, and this module cannot place orders.
"""
from bisect import bisect_left
from collections import deque
from datetime import date, datetime, timedelta, timezone
import math
import random
from statistics import median
from zoneinfo import ZoneInfo
from .financial import day_bounds
from .trend_research import POLICY, STEP, _valid, protective_exit, trend_signal

CONTEXT = ('BTCUSDT', 'ETHUSDT')


def _volume_valid(row):
    try:
        base, quote, count, buy_base, buy_quote = (float(row[i]) for i in (5,7,8,9,10))
        return (all(math.isfinite(v) and v >= 0 for v in (base,quote,count,buy_base,buy_quote))
                and count.is_integer() and buy_base <= base and buy_quote <= quote)
    except (IndexError, TypeError, ValueError, OverflowError):
        return False


def _window(rows, now_ms):
    try:
        closed = [r for r in rows if r[6] < now_ms][-POLICY['warmup']:]
        if len(closed) != POLICY['warmup'] or closed[-1][0] != now_ms-STEP:
            return None
        if not all(_valid(r) and _volume_valid(r) for r in closed):
            return None
        if any(b[0]-a[0] != STEP for a,b in zip(closed,closed[1:])):
            return None
        return closed
    except (IndexError, TypeError, ValueError):
        return None


def _returns(rows):
    prices = [float(r[4]) for r in rows]
    returns = [math.log(b/a) for a,b in zip(prices,prices[1:])]
    result = {f'return_{n}':math.log(prices[-1]/prices[-n-1]) for n in (1,2,4,8,24)}
    for n in (8,24):
        result[f'volatility_{n}'] = math.sqrt(sum(v*v for v in returns[-n:])/n)
    return result


def build_features(history_by_symbol, symbol, now_ms):
    """Return nested causal feature dictionaries, or block all learned arms.

    ``now_ms`` is the next execution bar's opening time. Rows closing at or
    after it are ignored, including any caller-supplied future suffix.
    """
    if type(now_ms) is not int or now_ms % STEP or symbol not in history_by_symbol:
        return None
    windows = {s:_window(history_by_symbol.get(s,()),now_ms) for s in set(CONTEXT+(symbol,))}
    if any(rows is None for rows in windows.values()):
        return None
    rows = windows[symbol]
    prices = [float(r[4]) for r in rows]
    latest = rows[-1]
    close, high, low = (float(latest[i]) for i in (4,2,3))
    previous_volume = median(float(r[7]) for r in rows[-21:-1])
    a0 = _returns(rows)
    a0.update(range_relative=(high-low)/close,
              ma_gap_8=close/(sum(prices[-8:])/8)-1,
              ma_gap_24=close/(sum(prices[-24:])/24)-1,
              relative_volume=float(latest[7])/previous_volume if previous_volume else 0.,
              log_trade_count=math.log1p(float(latest[8])),
              close_location=(close-low)/(high-low) if high>low else .5)
    a1 = dict(a0)
    context_returns = []
    for instrument in CONTEXT:
        summary = _returns(windows[instrument])
        context_returns.append(summary['return_4'])
        prefix = instrument.lower()
        a1.update({f'{prefix}_{key}':value for key,value in summary.items()})
        a1[f'{prefix}_relative_return_4'] = a0['return_4']-summary['return_4']
    a1['context_breadth_positive_4'] = sum(v>0 for v in context_returns)/len(context_returns)
    a2 = dict(a1)
    for n in (1,4,8,24):
        total = sum(float(r[7]) for r in rows[-n:])
        buy = sum(float(r[10]) for r in rows[-n:])
        a2[f'flow_imbalance_{n}'] = 2*buy/total-1 if total else 0.
    a2['flow_zero_volume'] = float(float(latest[7])==0)
    return {'A0':a0,'A1':a1,'A2':a2}


def _sell_value(position, raw_price, why, fee, slip):
    # A TP limit cannot fill below its registered limit due to market slippage.
    price = raw_price if why == 'TP' else raw_price*(1-slip)
    gross = position['quantity']*price
    return gross*(1-fee), gross*fee, gross, price


def training_label(rows, origin_ms, fit_cutoff_ms, fee_bps=10, slippage_bps=5):
    """Mature candidate return under the fixed 16-bar counterfactual policy.

    Every label reserves the entire 16-bar horizon even when protection exits
    sooner. A missing/invalid bar in that horizon makes the label unavailable.
    This target is a training proxy, never the portfolio objective.
    """
    horizon = POLICY['max_holding_bars']
    if (type(origin_ms) is not int or type(fit_cutoff_ms) is not int or origin_ms % STEP
            or origin_ms+horizon*STEP>fit_cutoff_ms):
        return None
    if not 0<=fee_bps<=100 or not 0<=slippage_bps<=100:
        raise ValueError('INVALID_COST_CONTRACT')
    # The archive is ordered. Locate one 48-bar local window in logarithmic
    # time rather than re-scanning the maximum history for every candidate.
    index = bisect_left(rows,origin_ms,key=lambda row:row[0])
    future = rows[index:index+horizon]
    if (len(future)!=horizon or any(not _valid(r) for r in future)
            or any(r[0]!=origin_ms+i*STEP for i,r in enumerate(future))):
        return None
    past = rows[max(0,index-POLICY['warmup']):index]
    if len(past)!=POLICY['warmup'] or any(not _valid(r) for r in past):
        return None
    if past[-1][0]!=origin_ms-STEP or any(b[0]-a[0]!=STEP for a,b in zip(past,past[1:])):
        return None
    fee, slip = fee_bps/10000, slippage_bps/10000
    debit = float(POLICY['entry_cash'])
    entry = float(future[0][1])*(1+slip)
    position = {'entry':entry,'peak':entry,'quantity':debit/(1+fee)/entry}
    history = deque(past,maxlen=POLICY['warmup'])
    for i,row in enumerate(future):
        if i and trend_signal(history,now_ms=row[0]).get('weakening'):
            proceeds,_,_,_ = _sell_value(position,float(row[1]),'WEAKENING',fee,slip)
            return proceeds/debit-1
        exit_fill = protective_exit(position,row)
        if exit_fill:
            proceeds,_,_,_ = _sell_value(position,exit_fill[0],exit_fill[1],fee,slip)
            return proceeds/debit-1
        if i==horizon-1:
            proceeds,_,_,_ = _sell_value(position,float(row[4]),'HOLDING_LIMIT',fee,slip)
            return proceeds/debit-1
        position['peak'] = max(position['peak'],float(row[2]))
        history.append(row)
    raise AssertionError('UNREACHABLE_HORIZON')


def _day(at):
    return datetime.fromtimestamp(at/1000,timezone.utc).astimezone(ZoneInfo('Europe/Budapest')).date().isoformat()


def replay(series, start_ms, end_ms, scorer=None, fee_bps=10, slippage_bps=5):
    """Matched shared-cash simulation; decisions cannot inspect execution rows.

    ``scorer`` receives bounded closed histories. None means frozen rule C0;
    a learned score <=0 or None blocks the candidate. All input pairs must
    have a regular execution grid, even when no position is open.
    """
    if (type(start_ms) is not int or type(end_ms) is not int or start_ms>=end_ms
            or start_ms%STEP or end_ms%STEP or not series
            or not 0<=fee_bps<=100 or not 0<=slippage_bps<=100):
        raise ValueError('INVALID_REPLAY_CONTRACT')
    symbols = sorted(series)
    indices,history = {},{}
    for symbol in symbols:
        rows = series[symbol]
        timestamps = [r[0] for r in rows]
        if any(b<=a for a,b in zip(timestamps,timestamps[1:])):
            raise ValueError('NON_MONOTONE_HISTORY')
        indices[symbol] = dict(zip(timestamps,rows))
        before = bisect_left(timestamps,start_ms)
        history[symbol] = deque(rows[max(0,before-POLICY['warmup']):before],maxlen=POLICY['warmup'])
    initial = float(POLICY['cash']); cash=initial; equity=initial; high_water=initial
    fee,slip=fee_bps/10000,slippage_bps/10000
    position=None;trades=[];fees=0.;turnover=0.;max_dd=0.;cooldown={};days={};reasons=[]
    max_debit=0.;max_concurrent=0;scored_candidates=0;positive_candidates=0

    def sell(price, at, why):
        nonlocal cash,position,fees,turnover
        proceeds,cost,gross,filled = _sell_value(position,price,why,fee,slip)
        cash+=proceeds;fees+=cost;turnover+=gross
        trades.append({'symbol':position['symbol'],'entry_ms':position['entry_ms'],'exit_ms':at,
                       'entry_price':position['entry'],'exit_price':filled,'reason':why,
                       'net_usdt':proceeds-position['debit'],'entry_debit_usdt':position['debit'],
                       'quantity':position['quantity']})
        cooldown[position['symbol']] = at+POLICY['cooldown_bars']*STEP
        position=None

    for at in range(start_ms,end_ms,STEP):
        day = _day(at)
        first,last = day_bounds(day,'Europe/Budapest')
        daily = days.setdefault(day,{'start_equity':equity,'end_equity':equity,'peak':equity,
                                      'first_ms':at,'last_ms':at-STEP,'points_n':0,
                                      'expected_points_n':(last-first)//STEP,'killed':False})
        # Candidate construction only reads closed histories. In particular it
        # does not iterate the execution group, whose missingness is future data.
        candidates=[]
        if position is None and not daily['killed']:
            for symbol in symbols:
                if at<cooldown.get(symbol,0):
                    continue
                signal=trend_signal(history[symbol],now_ms=at)
                if signal['action']!='BUY':
                    continue
                if scorer is None:
                    score=signal['score']
                else:
                    scored_candidates+=1
                    score=scorer(symbol,history,at)
                    if score is None or not math.isfinite(score) or score<=0:
                        continue
                    positive_candidates+=1
                candidates.append((-score,symbol))
        selected=min(candidates)[1] if candidates else None
        group={s:indices[s].get(at) for s in symbols}
        if selected is not None and group[selected] is None:
            reasons.append('SELECTED_EXECUTION_MISSING');break
        if position is not None and group[position['symbol']] is None:
            reasons.append('HELD_EXECUTION_MISSING');break
        if any(row is None for row in group.values()):
            reasons.append('PAIR_EXECUTION_GRID_MISSING');break
        if any(not _valid(row) for row in group.values()):
            reasons.append('EXECUTION_BAR_INVALID');break
        if position:
            symbol=position['symbol']
            signal=trend_signal(history[symbol],now_ms=at)
            if signal.get('weakening'):
                sell(float(group[symbol][1]),at,'WEAKENING')
            # A weakening sale can free cash, but the next decision is made on
            # the next bar. This conservative rule is identical for every arm.
        opening_equity=cash+(position['quantity']*float(group[position['symbol']][1]) if position else 0.)
        high_water=max(high_water,opening_equity);max_dd=max(max_dd,high_water-opening_equity)
        daily['peak']=max(daily['peak'],opening_equity)
        if daily['start_equity']-opening_equity>=POLICY['day_loss'] or daily['peak']-opening_equity>=POLICY['day_loss']:
            daily['killed']=True
        if position is None and selected is not None and not daily['killed']:
            price=float(group[selected][1])*(1+slip)
            debit=min(float(POLICY['entry_cash']),cash)
            if debit>0:
                gross=debit/(1+fee);quantity=gross/price
                cash-=debit;fees+=debit-gross;turnover+=gross
                position={'symbol':selected,'entry':price,'peak':price,'quantity':quantity,
                          'debit':debit,'entry_ms':at,'last_ms':at-STEP}
                max_debit=max(max_debit,debit);max_concurrent=1
        if position:
            row=group[position['symbol']]
            exit_fill=protective_exit(position,row)
            if exit_fill:
                sell(exit_fill[0],at,exit_fill[1])
            elif at-position['entry_ms'] >= (POLICY['max_holding_bars']-1)*STEP:
                sell(float(row[4]),at+STEP-1,'HOLDING_LIMIT')
            else:
                position['peak']=max(position['peak'],float(row[2]));position['last_ms']=at
        equity=cash+(position['quantity']*float(group[position['symbol']][4]) if position else 0.)
        high_water=max(high_water,equity);max_dd=max(max_dd,high_water-equity)
        daily.update(end_equity=equity,last_ms=at,points_n=daily['points_n']+1,peak=max(daily['peak'],equity))
        if daily['start_equity']-equity>=POLICY['day_loss'] or daily['peak']-equity>=POLICY['day_loss']:
            daily['killed']=True
        for symbol,row in group.items():
            history[symbol].append(row)
    daily_pnl={}
    for day,daily in days.items():
        first,last=day_bounds(day,'Europe/Budapest')
        complete=(daily['first_ms']==first and daily['last_ms']+STEP==last
                  and daily['points_n']==daily['expected_points_n'])
        daily['net_pnl_usdt']=daily['end_equity']-daily['start_equity']
        daily['calendar_status']='COMPLETE_SIMULATION' if complete else 'PARTIAL_SIMULATION'
        if complete:
            daily_pnl[day]=daily['net_pnl_usdt']
    if reasons:
        daily_pnl={}
    requested_complete_days=[]
    cursor=date.fromisoformat(_day(start_ms))
    final_day=date.fromisoformat(_day(end_ms-1))
    while cursor<=final_day:
        name=cursor.isoformat();first,last=day_bounds(name,'Europe/Budapest')
        if first>=start_ms and last<=end_ms:
            requested_complete_days.append(name)
        cursor+=timedelta(days=1)
    pnl_values=list(daily_pnl.values())
    open_exposure=(position['quantity']*float(indices[position['symbol']][position['last_ms']][4]) if position else 0.)
    return {'state':'INCOMPLETE' if reasons else 'COMPLETE','evidence_kind':'SIMULATION',
            'policy':'FSA-1-trend-matched','start_ms':start_ms,'end_ms':end_ms,
            'historical_receipts_certified':False,'point_in_time_universe_certified':False,
            'execution_parity_certified':False,'promotion_allowed':False,
            'initial_cash_usdt':initial,'final_cash_usdt':cash,
            'final_equity_usdt':None if reasons else equity,
            'net_pnl_usdt':None if reasons else equity-initial,
            'benchmark':'USDT_NO_TRADE_SAME_100','benchmark_net_pnl_usdt':0.,
            'trades_n':len(trades),'trades':trades,'fees_usdt':fees,'turnover_usdt':turnover,
            'max_drawdown_usdt':max_dd,'max_entry_debit_usdt':max_debit,
            'max_concurrent_positions':max_concurrent,'open_positions_n':int(position is not None),
            'open_exposure_usdt':open_exposure,'fee_bps':fee_bps,'slippage_bps':slippage_bps,
            'requested_complete_days_N':len(requested_complete_days),
            'observed_complete_days_N':len(daily_pnl),'excluded_complete_days_N':len(requested_complete_days)-len(daily_pnl),
            'complete_days_N':len(daily_pnl),'profitable_days_n':sum(v>0 for v in pnl_values) if pnl_values else None,
            'profitable_days_ratio':sum(v>0 for v in pnl_values)/len(pnl_values) if pnl_values else None,
            'mean_daily_net_pnl_usdt':sum(pnl_values)/len(pnl_values) if pnl_values else None,
            'scored_candidates_n':scored_candidates,'positive_candidates_n':positive_candidates,
            'days':days,'daily_pnl':daily_pnl,'reasons':reasons}


def _quantile(sorted_values, probability):
    at=(len(sorted_values)-1)*probability
    lower=int(at);upper=min(lower+1,len(sorted_values)-1)
    return sorted_values[lower]+(sorted_values[upper]-sorted_values[lower])*(at-lower)


def paired_bootstrap(daily_candidate,daily_reference,draws=5000,seed=42,block_days=7,confidence=.9875):
    """Paired circular calendar-block interval; never intersect away missing days."""
    if type(draws) is not int or draws<1 or type(block_days) is not int or block_days<1 or not 0<confidence<1:
        raise ValueError('INVALID_BOOTSTRAP_CONTRACT')
    candidate_days=set(daily_candidate);reference_days=set(daily_reference)
    dates=sorted(candidate_days)
    reasons=[]
    if candidate_days!=reference_days:reasons.append('PAIRED_DATE_MISMATCH')
    if len(dates)<30:reasons.append('FEWER_THAN_30_COMPLETE_DAYS')
    try:
        parsed=[date.fromisoformat(name) for name in dates]
        if any((b-a).days!=1 for a,b in zip(parsed,parsed[1:])):
            reasons.append('CALENDAR_GAP')
        values=[float(daily_candidate[name])-float(daily_reference[name]) for name in dates]
        if not all(math.isfinite(v) for v in values):reasons.append('NONFINITE_DAILY_PNL')
    except (TypeError,ValueError,KeyError):
        values=[];reasons.append('INVALID_PAIRED_DAYS')
    result={'state':'INCONCLUSIVE' if reasons else 'COMPLETE','paired_days_N':len(dates),
            'candidate_days_N':len(candidate_days),'reference_days_N':len(reference_days),
            'mean_delta_usdt':None,'ci_low_usdt':None,'ci_high_usdt':None,
            'confidence':confidence,'draws':draws,'seed':seed,'block_days':block_days,'reasons':reasons}
    if reasons:return result
    n=len(values);randomizer=random.Random(seed)
    full,remainder=divmod(n,block_days)
    block_sums=[sum(values[(i+j)%n] for j in range(block_days)) for i in range(n)]
    remainder_sums=[sum(values[(i+j)%n] for j in range(remainder)) for i in range(n)]
    samples=[]
    for _ in range(draws):
        total=sum(block_sums[randomizer.randrange(n)] for _ in range(full))
        if remainder:total+=remainder_sums[randomizer.randrange(n)]
        samples.append(total/n)
    samples.sort();tail=(1-confidence)/2
    result.update(mean_delta_usdt=sum(values)/n,ci_low_usdt=_quantile(samples,tail),
                  ci_high_usdt=_quantile(samples,1-tail))
    return result
