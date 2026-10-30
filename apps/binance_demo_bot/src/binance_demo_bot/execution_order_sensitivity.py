"""EOS-1: hypothetical OHLC execution conventions with frozen-policy decisions.

The account transition is a versioned copy of RDE-1 with only its unresolved
fixed-TP/SL branch changed. Original modules/artifacts remain untouched.
"""
import hashlib
import math

from .rl_direction_core import ResearchAccount, STEP, _number, _timestamp
from .rl_direction_runner import threshold_action


class ScenarioAccount(ResearchAccount):
    def __init__(self,*,intrabar_order='STOP_FIRST',**kwargs):
        if intrabar_order not in ('STOP_FIRST','TARGET_FIRST'): raise ValueError('INVALID_EXECUTION_CONVENTION')
        super().__init__(**kwargs)
        self.intrabar_order=intrabar_order; self.ambiguities=[]

    def step(self,origin_ms,candidate,action,bar_lookup,allow_entry=True):
        _timestamp(origin_ms)
        if type(action) is not int or action not in (0,1,2,3) or type(allow_entry) is not bool:
            raise ValueError('INVALID_ACTION_CONTRACT')
        if candidate is not None and (not isinstance(candidate,str) or not candidate): raise ValueError('INVALID_CANDIDATE')
        if not self.valid: return self._result(0.,[])
        if self.last_origin is not None and origin_ms!=self.last_origin+STEP:
            self._invalidate('NON_CONSECUTIVE_STEP'); return self._result(0.,[])
        before=self.equity; events=[]; holding=self.position is not None
        if action in (1,2): self.counters['buy_intents_n']+=1
        if holding and action in (1,2):
            self.counters['invalid_actions_n']+=1; action=0
        elif not holding and action==3:
            self.counters['invalid_actions_n']+=1; action=0
        debit=min(10. if action==1 else 30.,self.cash)
        if not holding and action in (1,2) and (not allow_entry or self.entry_killed or candidate is None or debit<5):
            self.counters['entry_vetoes_n']+=1; action=0
        symbol=self.position['symbol'] if holding else candidate if action in (1,2) else None
        row=None
        if symbol is not None:
            raw=bar_lookup(symbol,origin_ms)
            if raw is None:
                self._invalidate('HELD_EXECUTION_MISSING' if holding else 'SELECTED_EXECUTION_MISSING')
                return self._result(0.,[])
            try:
                row={key:_number(raw[key]) for key in ('open','high','low','close')}
                if (min(row.values())<=0 or row['high']<max(row['open'],row['close']) or
                    row['low']>min(row['open'],row['close']) or row['low']>row['high']): raise ValueError('INVALID_OHLC')
            except (KeyError,TypeError,ValueError,OverflowError):
                self._invalidate('EXECUTION_BAR_INVALID'); return self._result(0.,[])
        if holding:
            stop=self.position['entry_price']*.995; reason='SL'
            if self.exit_mode=='trailing' and self.position['prior_peak']>=self.position['entry_price']*1.005:
                trailing=self.position['prior_peak']*.995
                if trailing>stop: stop,reason=trailing,'TRAIL'
            if row['open']<=stop: events.append(self._sell(row['open'],origin_ms,reason))
            elif self.exit_mode=='fixed' and row['open']>=self.position['entry_price']*1.005:
                events.append(self._sell(self.position['entry_price']*1.005,origin_ms,'TP'))
            if self.position is not None: self._mark(self.cash+self.position['quantity']*row['open'])
        if holding and self.position is not None and action==3:
            events.append(self._sell(row['open'],origin_ms,'EXIT'))
        elif not holding and action in (1,2):
            price=row['open']*(1+self.slip); gross=debit/(1+self.fee); fee=debit-gross
            self.cash-=debit; self.fees+=fee; self.turnover+=gross; self.max_entry_debit=max(self.max_entry_debit,debit)
            self.position=dict(symbol=symbol,quantity=gross/price,entry_price=price,entry_ms=origin_ms,
                prior_peak=price,mark_price=price,debit_usdt=debit)
            event=dict(side='BUY',symbol=symbol,at_ms=origin_ms,price=price,quantity=self.position['quantity'],
                fee_usdt=fee,gross_usdt=gross,debit_usdt=debit,reason='POLICY_ENTRY')
            events.append(event); self.events.append(event); self.counters['buys_n']+=1
            self._mark(self.cash+self.position['quantity']*row['open'])
        if self.position is not None:
            position=self.position; stop=position['entry_price']*.995; stop_reason='SL'
            if self.exit_mode=='trailing' and position['prior_peak']>=position['entry_price']*1.005:
                trailing=position['prior_peak']*.995
                if trailing>stop: stop,stop_reason=trailing,'TRAIL'
            target=position['entry_price']*1.005
            both=self.exit_mode=='fixed' and row['low']<=stop and row['high']>=target
            if both:
                self.ambiguities.append(dict(origin_ms=origin_ms,symbol=position['symbol'],stop=stop,target=target,
                    chosen_reason='TP' if self.intrabar_order=='TARGET_FIRST' else stop_reason))
            if both and self.intrabar_order=='TARGET_FIRST': events.append(self._sell(target,origin_ms,'TP'))
            elif row['low']<=stop: events.append(self._sell(min(row['open'],stop),origin_ms,stop_reason))
            elif self.exit_mode=='fixed' and row['high']>=target: events.append(self._sell(target,origin_ms,'TP'))
            elif origin_ms-position['entry_ms']>=15*STEP:
                events.append(self._sell(row['close'],origin_ms+STEP-1,'HOLDING_LIMIT'))
            else:
                position['prior_peak']=max(position['prior_peak'],row['high']); position['mark_price']=row['close']
        self.equity=self.cash+(self.position['quantity']*self.position['mark_price'] if self.position else 0.)
        self._mark(self.equity); self.last_origin=origin_ms; self.counters['steps_n']+=1
        self.samples.append(dict(origin_ms=origin_ms,equity_before=before,equity_after=self.equity))
        return self._result((self.equity-before)/100,events)


def make_signal_schedule(tape,start,end,threshold):
    digest=hashlib.sha256(); signals=[]; candidates=[]
    for at in range(start,end,STEP):
        candidate=tape.candidate(at); forecast=tape.forecast(candidate,at,'GRU') if candidate else None
        signal=bool(forecast and forecast[0]>=.5 and forecast[1]>.003+threshold)
        digest.update(f'{at}:{candidate}:{int(signal)}\n'.encode()); signals.append(signal); candidates.append(candidate)
    return dict(start=start,end=end,threshold=threshold,signals=signals,candidates=candidates,
                sha256=digest.hexdigest(),count=sum(signals))


def run_scenario(tape,start,end,*,kind,threshold=0.,policy=None,exit_mode=None,cost='base',
                 intrabar_order='STOP_FIRST',fast_forward=True,signals=None):
    if cost not in ('base','stress'): raise ValueError('INVALID_COST_CASE')
    if kind in ('SIGNAL','EXIT'):
        if signals is None: signals=make_signal_schedule(tape,start,end,threshold)
        if (signals['start'],signals['end'],signals['threshold'])!=(start,end,threshold):
            raise ValueError('SIGNAL_SCHEDULE_BINDING')
    mode=exit_mode or ('learned' if kind in ('MYOPIC','PPO','EXIT') else 'fixed')
    account=ScenarioAccount(fee_bps=10 if cost=='base' else 20,slippage_bps=5 if cost=='base' else 10,
        exit_mode=mode,intrabar_order=intrabar_order)
    actions=[]; locked=0; calls=0
    for i,at in enumerate(range(start,end,STEP)):
        if fast_forward and account.entry_killed and account.position is None:
            action=0; candidate=None; source='SYSTEM_RISK_LOCK'; locked+=1
        else:
            source='FROZEN_POLICY'; candidate=signals['candidates'][i] if signals else tape.candidate(at)
            if kind in ('SIGNAL','EXIT'):
                if not account.position: action=2 if signals['signals'][i] else 0
                elif kind=='EXIT':
                    obs=account.observation(at,candidate,tape.vector,remaining=(end-at)/(end-start))
                    action=3 if policy.predict(obs,deterministic=True)==3 else 0; calls+=1
                else: action=0
            elif kind in ('PPO','MYOPIC'):
                obs=account.observation(at,candidate,tape.vector,remaining=(end-at)/(end-start))
                action=policy.predict(obs,deterministic=True); calls+=1
            elif kind=='CASH': action=0
            else: action=threshold_action(account,tape,at,kind,threshold)
        result=account.step(at,candidate,int(action),tape.bar)
        actions.append(dict(origin_ms=at,action=None if source=='SYSTEM_RISK_LOCK' else int(action),source=source))
        if not result['valid']: break
    report=account.report(start,end)
    report.update(kind=kind,threshold=threshold,cost=cost,actions=actions,intrabar_order=intrabar_order,
        ambiguities=account.ambiguities,ambiguous_executions_n=len(account.ambiguities),
        risk_locked_steps_n=locked,policy_calls_n=calls,intent_counts_after_lock_comparable=not bool(locked),
        exit_mode=mode,scenario_kind='COUNTERFACTUAL_OHLC_CONVENTION',native_execution_certified=False)
    if signals: report.update(signal_sha256=signals['sha256'],entry_signals_n=signals['count'])
    return report


def compare_reference(reference,candidate):
    paths=[]; differences=0; max_abs=0.
    def mark(path):
        nonlocal differences
        differences+=1
        if len(paths)<20: paths.append(path)
    def walk(a,b,path):
        nonlocal max_abs
        if isinstance(a,dict):
            if not isinstance(b,dict): mark(path); return
            for key,value in a.items():
                if path=='' and key=='actions': continue
                if path=='counters' and key not in ('steps_n','buys_n','sells_n'): continue
                if key not in b: mark(path+'.'+key)
                else: walk(value,b[key],path+'.'+key if path else key)
        elif isinstance(a,list):
            if not isinstance(b,list) or len(a)!=len(b): mark(path); return
            for i,(x,y) in enumerate(zip(a,b)): walk(x,y,f'{path}[{i}]')
        elif type(a) in (int,float) and type(b) in (int,float):
            difference=abs(a-b); max_abs=max(max_abs,difference)
            if not math.isfinite(difference) or difference>1e-8: mark(path)
        elif a!=b: mark(path)
    walk(reference,candidate,'')
    return dict(state='MATCH' if not differences else 'MISMATCH',differences_n=differences,
                paths=paths,max_abs=max_abs,tolerance=1e-8)


def financial_check(report):
    reasons=[]
    if report['state']!='COMPLETE': return dict(state='INCOMPLETE',reasons=['INCOMPLETE_OBJECTIVE'])
    try:
        initial=report.get('initial_equity_usdt',100.); exposure=report['open_exposure_usdt']; cash=report['cash_usdt']
        net=report['net_pnl_usdt']; residual=cash+exposure-initial-net
        if not math.isfinite(residual) or abs(residual)>1e-8: reasons.append('CASH_EXPOSURE_NET_MISMATCH')
        if report['fees_usdt']<0: reasons.append('NEGATIVE_FEES')
        if not report.get('unmanaged_risk',False):
            events=report['events']
            if abs(sum(e['fee_usdt'] for e in events)-report['fees_usdt'])>1e-8:
                reasons.append('EXECUTION_FEE_LEDGER_MISMATCH')
            event_cash=initial-sum(e['debit_usdt'] for e in events if e['side']=='BUY')+sum(e['proceeds_usdt'] for e in events if e['side']=='SELL')
            if abs(event_cash-cash)>1e-8: reasons.append('EXECUTION_CASH_LEDGER_MISMATCH')
            position=report.get('open_position'); unrealized=exposure-position['debit_usdt'] if position else 0.
            ledger=sum(t['net_usdt'] for t in report['trades'])+unrealized
            if not math.isfinite(ledger-net) or abs(ledger-net)>1e-8: reasons.append('TRADE_MTM_NET_MISMATCH')
            if bool(position)!=bool(exposure): reasons.append('OPEN_EXPOSURE_BINDING')
            if len(report['trades'])!=report['trades_n']: reasons.append('TRADE_COUNT_MISMATCH')
    except (KeyError,TypeError,ValueError): reasons.append('FINANCIAL_SCHEMA_INVALID')
    return dict(state='FAIL' if reasons else 'PASS',reasons=reasons)
