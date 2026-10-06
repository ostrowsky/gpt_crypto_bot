"""Independent scalar prefix, actual action, raw leader/episode and cash audit."""
from __future__ import annotations
import argparse,json,math
from bisect import bisect_left,bisect_right
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import replay_backtest as rb
from historical_signal_evaluation import sha
from verify_leader_mission_reaudit import bounds,identity,TZ,BAR,scalar_episode,assert_values
from verify_exit_action_advantage import state_at
from verify_joint_direction_amplitude import assert_nested
from research_rocket_capture import policy
from compare_price_volatility_bot import reuse_features,reuse_candidate_snapshot
from portfolio_alpha import _simulate_account
from run_turnover_economics import publish


def scalar_confirmation(rows,times,ema,atr,entry,price,at):
    i=bisect_left(times,entry);j=bisect_left(times,at)
    if price<=0 or i<13 or j<=i or j>=len(times) or times[i:j+1]!=list(range(entry,at+1,BAR)):return None
    c=rows[j]['c'];e=ema[j];prev=ema[j-1];a=atr[j];ea=atr[i]
    peak=max(price,max(r['c'] for r in rows[i:j+1]));tol=price*1e-12
    if not all(math.isfinite(v) for v in (c,e,prev,a,ea,peak)) or min(a,ea)<=0:return None
    return dict(close=c,ema=e,previous_ema=prev,atr=a,entry_atr=ea,peak=peak,
        confirmed=c>e+tol and e>prev+tol and peak>=price+ea-tol and c>=peak-a-tol)


def compare_features(expected,actual):
    if expected is None:
        if actual is not None:raise ValueError('invalid prefix accepted')
    else:assert_values(expected,actual)


def block_interval(rows,key):
    days=sorted({r['day'] for r in rows})
    if len(days)<3:return None
    sums=np.array([sum(r[key] for r in rows if r['day']==d) for d in days]);count=np.array([sum(r['day']==d for r in rows) for d in days])
    rng=np.random.default_rng(42);st=rng.integers(0,len(days),(5000,math.ceil(len(days)/3)))
    idx=((st[:,:,None]+np.arange(3))%len(days)).reshape(5000,-1)[:,:len(days)]
    return np.quantile(sums[idx].sum(axis=1)/count[idx].sum(axis=1),[.05/48,1-.05/48]).tolist()


def terminal_recheck(tick,state,trade,data,end):
    """Only the inherited len(data)-2 boundary callback, never a live clock waiver."""
    return (not state['closed'] and state['last']==end and trade.exit_ts==end and
            trade.exit_reason=='open_at_end' and tick['at']==int(data['t'][-2])+rb.BAR_MS[trade.tf])


def verify(directory):
    reg=json.loads((directory/'registration.json').read_bytes());result=json.loads((directory/'result.json').read_bytes())
    for n,h in json.loads((directory/'receipt.json').read_bytes()).items():
        if Path(n).name!=n or sha(directory/n)!=h:raise ValueError('artifact drift')
    for n,h in reg['sources'].items():
        if sha(directory/'source_snapshot'/n)!=h or sha(Path(__file__).with_name(n))!=h:raise ValueError('source drift')
    if sha(directory/'registered_spec.md')!=reg['spec_sha256']:raise ValueError('spec drift')
    for n,v in json.loads((directory/'inheritance_receipt.json').read_bytes())['files'].items():
        if sha(directory/n)!=v['sha256'] or sha(Path(v['source']))!=v['sha256']:raise ValueError('inheritance drift')
    parent_audit=json.loads((directory/'parent_audit.json').read_bytes())
    if parent_audit['status']!='PASS' or parent_audit['result_sha256']!=sha(directory/'parent_result.json'):raise ValueError('parent audit')
    market=Path(reg['market']);manifest=json.loads((market/'manifest.json').read_bytes())
    if sha(market/'manifest.json')!=reg['market_manifest_sha256']:raise ValueError('manifest drift')
    audit_dir=directory/'verification_inputs_v2';audit_dir.mkdir(exist_ok=False)
    cache,frames,_=reuse_features(Path(reg['features']),manifest,audit_dir);del frames
    raw={};times={};ema={};atr={}
    for s in manifest['eligible_symbols']:
        for tf in ('15m','1h'):
            n=s+'_'+tf+'.json'
            if sha(market/'market'/n)!=manifest['input_hashes'][n]:raise ValueError('raw drift')
            rows=json.loads((market/'market'/n).read_bytes())
            for field in ('t','o','h','l','c','v'):np.testing.assert_array_equal(cache[s,tf][0][field],np.array([r[field] for r in rows]))
            if tf=='15m':raw[s]=rows
        times[s]=[r['t']+BAR for r in raw[s]];em=[];tr=[]
        for i,r in enumerate(raw[s]):
            em.append(r['c'] if i==0 else em[-1]+(2/21)*(r['c']-em[-1]))
            prior=raw[s][i-1]['c'] if i else r['c'];tr.append(max(r['h']-r['l'],abs(r['h']-prior),abs(r['l']-prior)))
        ema[s]=em;atr[s]=[float('nan')]*13+[sum(tr[i-13:i+1])/14 for i in range(13,len(tr))]
    labels=json.loads((directory/'daily_labels.json').read_bytes());expected={}
    date=datetime.fromtimestamp(reg['start_ms']/1000,timezone.utc).astimezone(TZ).date()
    while bounds(date.isoformat())[0]<=reg['end_ms']:
        day=date.isoformat();lo,hi=bounds(day)
        if reg['start_ms']<=lo and hi<=reg['end_ms']:
            vals={};ranks=[]
            for s in raw:
                i=bisect_left(times[s],lo+BAR);j=bisect_left(times[s],hi)
                if j>=len(times[s]) or times[s][i:j+1]!=list(range(lo+BAR,hi+1,BAR)):continue
                o,c=raw[s][i]['o'],raw[s][j]['c'];ret=100*(c/o-1);vals[s]=(o,c,hi);ranks.append((ret,s))
                assert_values(dict(open=o,close=c,cutoff=hi,return_pct=ret),labels[day]['values'][s])
            if len(vals)>=15:
                leaders=[s for v,s in sorted(ranks,reverse=True)[:15]]
                if labels[day]['leaders']!=leaders:raise ValueError('leader ranks')
                expected[day]=(leaders,vals)
        date+=timedelta(days=1)
    if expected.keys()!=labels.keys():raise ValueError('label days')
    all_trades={k:[rb.ReplayTrade(**r) for r in json.loads((directory/f'trades_{k}.json').read_bytes())] for k in ('control','trend')}
    if json.loads((directory/'control_parity.json').read_bytes())['status']!='PASS_ALL_FIELDS':raise ValueError('no baseline parity')
    trace=json.loads((directory/'policy_trace.json').read_bytes());by_key={(t.sym,t.tf,t.entry_ts):t for t in all_trades['trend']};states={};seen=set()
    progress=rb._update_trade_progress;terminal_checks=0
    with policy('baseline'):
        for r in trace['decisions']:
            key=tuple(r['key']);s,tf,entry=key
            if key in seen or key not in by_key:raise ValueError('soft population key')
            seen.add(key);tr=by_key[key];clone=state_at(tr,cache,r['at'],progress)
            if clone.exit_reason!=r['reason'] or clone.entry_price!=r['entry_price']:raise ValueError('original soft proposal')
            d,f=cache[s,tf];j=rb._find_last_closed_candle_index(d['t'],r['at'],rb.BAR_MS[tf])
            np.testing.assert_allclose(float(d['c'][j]),r['origin_price'],rtol=0,atol=1e-12)
            exact=int(d['t'][j])+rb.BAR_MS[tf]==r['at']
            x=scalar_confirmation(raw[s],times[s],ema[s],atr[s],entry,tr.entry_price,r['at']) if exact else None
            if x is not None and not math.isclose(x['close'],r['origin_price'],rel_tol=1e-10,abs_tol=1e-12):x=None
            compare_features(x,r['features'])
            if r['defer']!=bool(x is not None and x['confirmed']) or r['deadline']!=r['at']+4*BAR:raise ValueError('decision/deadline')
            if r['defer']:
                clone.exit_ts=0;clone.exit_price=0.;clone.exit_reason='';states[key]=dict(trade=clone,last=r['at'],deadline=r['deadline'],origin=r['reason'],closed=False)
        for tick in trace['ticks']:
            key=tuple(tick['key']);state=states[key];s,tf,entry=key;at=tick['at'];clone=state['trade']
            d,f=cache[s,tf]
            if state['closed'] or at!=state['last']+BAR:
                if not terminal_recheck(tick,state,by_key[key],d,reg['end_ms']):raise ValueError('active clock/reactivation')
                terminal_checks+=1
            j=rb._find_last_closed_candle_index(d['t'],at,rb.BAR_MS[tf]);micro=cache.get((s,'15m')) if tf=='1h' else None
            reason=progress(clone,d,f,j,ts_ms=at,micro_pack=micro)
            if reason!=tick['reason']:raise ValueError('hard/soft kernel mismatch')
            x=scalar_confirmation(raw[s],times[s],ema[s],atr[s],entry,clone.entry_price,at);compare_features(x,tick['features'])
            kind='HARD' if reason and not rb._is_weak_exit_reason(reason) else 'TIMEOUT' if at>=state['deadline'] else 'CONFIRMATION_LOST' if x is None or not x['confirmed'] else 'HOLD'
            if kind!=tick['kind'] or tick['deadline']!=state['deadline']:raise ValueError('active action contract')
            if kind=='HOLD':clone.exit_ts=0;clone.exit_price=0.
            else:
                state['closed']=True;live=by_key[key]
                if kind!='HARD':
                    expected_reason=state['origin']+' [trend continuation '+kind+']'
                    if live.exit_reason!=expected_reason or live.exit_ts!=at or live.exit_price!=raw[s][bisect_left(times[s],at)]['c']:raise ValueError('forced current15m fill')
                    if rb._cooldown_bars_after_exit(live.mode,live.exit_reason)!=rb._cooldown_bars_after_exit(live.mode,state['origin']):raise ValueError('cooldown category changed')
                elif live.exit_reason!=reason:raise ValueError('hard exit not preserved')
            state['last']=at
    if (len(trace['decisions']),len(states),len(trace['ticks']))!=(result['policy_counts']['first_soft_decisions'],result['policy_counts']['deferrals'],result['policy_counts']['active_ticks']):raise ValueError('policy counts')
    # Candidate identity checked against the producer/input/source-bound pinned native stream.
    snapshot=reuse_candidate_snapshot(Path(reg['candidates']),manifest,audit_dir)
    candidates={(c.sym,c.tf,c.mode,c.ts_ms,c.price,c.top_gainer_score) for group in snapshot[0].values() for c in group}
    daily={};episodes={};skips_checked=0;episodes_checked=0
    for arm,trades in all_trades.items():
        rows=[dict(vars(t)) for t in trades];recorded=json.loads((directory/f'leader_entries_{arm}.json').read_bytes());ep=json.loads((directory/f'episodes_{arm}.json').read_bytes())
        eligible=[];first={};by_symbol={s:[] for s in raw}
        for t in sorted(rows,key=lambda v:v['entry_ts']):
            by_symbol[t['sym']].append(t);day=identity(t['entry_ts'])
            if day in expected and t['entry_ts']<=bounds(day)[1] and t['sym'] in expected[day][1]:eligible.append(t);first.setdefault((day,t['sym']),t)
        leaders={p:t for p,t in first.items() if p[1] in expected[p[0]][0]}
        if len(recorded)!=len(leaders) or len(ep)!=len(recorded):raise ValueError('leader count')
        emap={(e['day'],e['symbol']):e for e in ep}
        for e in recorded:
            pair=(e['day'],e['symbol']);t=leaders[pair];o,c,cut=expected[pair[0]][1][pair[1]]
            if t!=e['trade']:raise ValueError('first BUY choice')
            capture=min(1.5,max(0,(c-t['entry_price'])/(c-o))) if c>o else None
            assert_values(dict(capture=capture,early=capture is not None and capture>=.35,lead_minutes=(cut-t['entry_ts'])/60000),e)
            calculated=scalar_episode(e,raw[pair[1]],ema[pair[1]],atr[pair[1]],by_symbol[pair[1]],times[pair[1]])
            assert_values(calculated,emap[pair]);episodes_checked+=1
        daily[arm]={}
        for cohort in ('full','test'):
            days=[d for d in expected if cohort=='full' or bounds(d)[0]>=reg['test_boundary_ms']]
            chosen=[t for t in eligible if identity(t['entry_ts']) in days];leads=[e for e in recorded if e['day'] in days]
            assert_values(dict(days=len(days),leader_pairs=len(days)*15,early=sum(e['early'] for e in leads),captured=len(leads),
                precision_n=sum(t['sym'] in expected[identity(t['entry_ts'])][0] for t in chosen),precision_N=len(chosen),
                unique_precision_n=len(leads),unique_precision_N=sum(p[0] in days for p in first)),result['arms'][arm][cohort])
            dayrows=[dict(day=d,early=sum(e['early'] for e in leads if e['day']==d),captured=sum(e['day']==d for e in leads),leaders=15,
                precision_n=sum(identity(t['entry_ts'])==d and t['sym'] in expected[d][0] for t in chosen),precision_N=sum(identity(t['entry_ts'])==d for t in chosen)) for d in sorted(days)]
            daily[arm][cohort]=dayrows
            cohort_ep=[e for e in ep if e['day'] in days]
            valid=[e for e in cohort_ep if e['state']=='CONFIRMED' and not e['terminal_forced']]
            rebound=[e for e in cohort_ep if e.get('new_high_after_first_exit') is not None and not e['terminal_forced']]
            state_counts={s:sum(e['state']==s for e in cohort_ep) for s in sorted({e['state'] for e in cohort_ep})}
            summary=dict(captured_pairs=len(cohort_ep),confirmed_n=len(valid),forced_n=sum(e['terminal_forced'] for e in cohort_ep),
                early_exit_n=sum(e['exit_before_marker'] for e in valid),first_early_exit_n=sum(e['first_exit_before_marker'] for e in valid),
                retention_mean=float(np.mean([e['retention'] for e in valid])) if valid else None,
                giveback_mean_pp=float(np.mean([e['giveback_pp'] for e in valid])) if valid else None,
                delay_bars_median=float(np.median([e['delay_bars'] for e in valid])) if valid else None,
                remaining_mean=float(np.mean([e['remaining_fraction_at_marker'] for e in valid])) if valid else None,
                accompaniment_mean=float(np.mean([e['accompaniment_fraction'] for e in valid])) if valid else None,
                any_remaining_n=sum(e['any_remaining_at_marker'] for e in valid),
                rebound_n=sum(e['new_high_after_first_exit'] for e in rebound),rebound_N=len(rebound))
            assert_values(summary,result['arms'][arm]['exits_'+cohort]);assert_nested(state_counts,result['arms'][arm]['exits_'+cohort]['states'])
        episodes[arm]=[e for e in ep if bounds(e['day'])[0]>=reg['test_boundary_ms']]
        skips=json.loads((directory/f'cooldown_{arm}.json').read_bytes())
        if len(skips)!=result['stats'][arm]['cooldown_skipped_candidates']:raise ValueError('skip total')
        exits={s:sorted([t for t in rows if t['sym']==s],key=lambda t:t['exit_ts']) for s in raw}
        clocks={s:[t['exit_ts'] for t in exits[s]] for s in raw}
        for skip in skips:
            s=skip['symbol'];key=(s,skip['tf'],skip['mode'],skip['at'],skip['price'],skip['score'])
            if key not in candidates:raise ValueError('skip not actual candidate')
            j=bisect_right(clocks[s],skip['at'])-1
            if j<0:raise ValueError('skip without previous exit')
            previous=exits[s][j]
            if (previous['entry_ts'],previous['exit_ts'],previous['exit_reason'])!=(skip['previous_entry'],skip['previous_exit'],skip['previous_reason']):raise ValueError('skip previous state')
            deadline=previous['exit_ts']+rb._cooldown_bars_after_exit(previous['mode'],previous['exit_reason'])*rb.BAR_MS[previous['tf']]
            if skip['cooldown_until']!=deadline or not previous['exit_ts']<=skip['at']<deadline:raise ValueError('skip cooldown')
            skips_checked+=1
        for cohort in ('full','test'):
            valid=[e for e in ep if e['state']=='CONFIRMED' and not e['terminal_forced'] and
                   (cohort=='full' or bounds(e['day'])[0]>=reg['test_boundary_ms'])]
            def kind(reason):
                if 'ATR trail' in reason:return 'ATR_TRAIL'
                if 'WEAK:' in reason:return 'WEAK'
                if 'time (' in reason:return 'TIME_LIMIT'
                if 'replace' in reason.lower():return 'REPLACEMENT'
                if 'open_at_end' in reason:return 'BOUNDARY'
                return 'OTHER_HARD'
            causes=result['causes'][arm][cohort];groups={};exact={};associated=[];pairs=set()
            for e in valid:
                name=kind(e['exit_reason']);group=groups.setdefault(name,dict(n=0,N=len(valid),before=0,rebound_n=0,rebound_N=0))
                group['n']+=1;group['before']+=int(e['first_exit_before_marker'])
                group['rebound_n']+=int(e.get('new_high_after_first_exit') is True);group['rebound_N']+=int(e.get('new_high_after_first_exit') is not None)
                exactrow=exact.setdefault(e['exit_reason'],dict(n=0,before=0));exactrow['n']+=1;exactrow['before']+=int(e['first_exit_before_marker'])
            for skip in skips:
                if cohort=='test' and skip['at']<reg['test_boundary_ms']:continue
                related=[e for e in valid if e['symbol']==skip['symbol'] and e['entry_ts']<=skip['at']<e['marker_ts']]
                if related:
                    associated.append(dict(**skip,leader_episodes=[e['day'] for e in related]));pairs.update((e['symbol'],e['day']) for e in related)
            assert_nested(groups,causes['categories']);assert_nested(exact,causes['exact_reasons']);assert_nested(associated,causes['cooldown_episode_events'])
            assert_values(dict(confirmed_n=len(valid),cooldown_events_in_confirmed_leader_episodes=len(associated),cooldown_unique_episode_pairs=len(pairs)),causes)
        series={s:list(zip(times[s],(r['c'] for r in raw[s]))) for s in raw}
        grid=list(range(reg['start_ms'],reg['end_ms']+1,BAR))
        account=_simulate_account(trades,price_series_by_symbol=series,valuation_timestamps=grid,capacity=10,initial_capital=10000,fee_bps=reg['fee_bps'],slippage_bps=reg['slippage_bps'])
        if account.violations or account.fully_valued_points!=len(grid):raise ValueError('account marks')
        np.testing.assert_allclose(account.equity_curve,result['accounts'][arm]['curve'],rtol=1e-10,atol=1e-7)
    # Independently paired entry intervals and same-entry exit/time effects.
    comp=result['comparison'];a,b=(np.array([[r['early']/15,r['captured']/15,r['precision_n'],r['precision_N']] for r in daily[k]['test']]) for k in ('control','trend'))
    n=len(a);rng=np.random.default_rng(42);st=rng.integers(0,n,(5000,math.ceil(n/3)));idx=((st[:,:,None]+np.arange(3))%n).reshape(5000,-1)[:,:n]
    aa,bb=a[idx],b[idx];delta=np.column_stack([bb[:,:,:2].mean(axis=1)-aa[:,:,:2].mean(axis=1),bb[:,:,2].sum(axis=1)/bb[:,:,3].sum(axis=1)-aa[:,:,2].sum(axis=1)/aa[:,:,3].sum(axis=1)])*100
    for j,key in enumerate(('early_pp','capture_pp','precision_pp')):np.testing.assert_allclose(np.quantile(delta[:,j],[.05/48,1-.05/48]),comp['entry']['intervals'][key],atol=1e-10)
    e0,e1=({(e['day'],e['symbol']):e for e in episodes[k]} for k in ('control','trend'));matched=[];different=0
    for p in sorted(e0.keys()&e1.keys()):
        x,y=e0[p],e1[p]
        if (x['entry_ts'],x['entry_price'])!=(y['entry_ts'],y['entry_price']):different+=1;continue
        if any(e['state']!='CONFIRMED' or e['terminal_forced'] for e in (x,y)):continue
        matched.append(dict(day=p[0],retention=y['retention']-x['retention'],held=y['accompaniment_fraction']-x['accompaniment_fraction'],early=int(y['first_exit_before_marker'])-int(x['first_exit_before_marker'])))
    assert_values(dict(common_pairs=len(e0.keys()&e1.keys()),different_entry_pairs=different,confirmed_same_entry_n=len(matched),
        retention_delta_mean=float(np.mean([r['retention'] for r in matched])) if matched else None,first_early_delta_n=sum(r['early'] for r in matched)),comp['exits'])
    assert_nested(block_interval(matched,'retention'),comp['exits']['retention_interval'])
    assert_values(dict(n=len(matched),days=len({r['day'] for r in matched}),mean=float(np.mean([r['held'] for r in matched])) if matched else None),comp['accompaniment'])
    assert_nested(block_interval(matched,'held'),comp['accompaniment']['interval'])
    checks={}
    for cohort in ('full','test'):
        a,b=(result['arms'][k][cohort] for k in ('control','trend'))
        checks[cohort+'_early']=b['early']>=a['early'];checks[cohort+'_coverage']=b['captured']>=a['captured']
        checks[cohort+'_precision']=bool(a['precision_N'] and b['precision_N'] and b['precision_n']/b['precision_N']>=a['precision_n']/a['precision_N'])
    p=comp['accompaniment'];e=comp['exits']
    checks.update(held_time_gain=p['mean'] is not None and p['mean']>0,
        held_time_confirmed=p['days']>=30 and p['interval'] is not None and p['interval'][0]>0,
        retention_not_worse=e['retention_delta_mean'] is not None and e['retention_delta_mean']>=0,no_more_early_exits=e['first_early_delta_n']<=0)
    status='RETROSPECTIVE_GAIN_REQUIRES_FORWARD' if all(checks.values()) else 'MISSION_TRADEOFF_OR_WORSE' if not all(v for k,v in checks.items() if k.startswith(('full_','test_'))) else 'NO_CONFIRMED_MISSION_GAIN'
    assert_nested(dict(checks=checks,status=status,runtime_eligible=False),result['verdict'])
    report=dict(status='PASS',result_sha256=sha(directory/'result.json'),verifier_sha256=sha(Path(__file__)),
        raw_symbols=len(raw),daily_labels=len(expected)*15,scalar_episode_checks=episodes_checked,
        first_soft_decisions=len(trace['decisions']),active_ticks=len(trace['ticks']),actual_cooldown_candidates=skips_checked,
        causal_active_ticks=len(trace['ticks'])-terminal_checks,terminal_legacy_rechecks=terminal_checks,
        limitations=['indicator kernel inherited/source-bound; rule EMA/ATR independently scalar reconstructed',
            'one inherited backdated boundary recheck excluded from causal action count; forced episodes excluded from exit quality',
            'no live/PIT/receive certification; causal marker operational; exposed retrospective'])
    publish(directory/'independent_verification.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
