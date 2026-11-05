"""MRP-1 native minute execution; frozen15m decisions and accounting grid.

Versioned transition from EOS-1. No training, exchange authority or synthetic path.
"""
from collections import OrderedDict, defaultdict
import math
import hashlib
import struct
from .rl_direction_core import ResearchAccount, STEP, _number, _timestamp
from .rl_direction_runner import threshold_action
from .execution_order_sensitivity import make_signal_schedule
from .intrabar_data import validate_window, canonical_sha256


class MinuteAccount(ResearchAccount):
    def __init__(self,*,intrabar_order='MINUTE_STOP_FIRST',**kwargs):
        if intrabar_order not in ('MINUTE_STOP_FIRST','MINUTE_TARGET_FIRST'): raise ValueError('INVALID_EXECUTION_CONVENTION')
        super().__init__(**kwargs)
        self.intrabar_order=intrabar_order; self.ambiguities=[]; self.execution_frames=[]

    def step(self,origin_ms,candidate,action,bar_lookup,minute_lookup,allow_entry=True):
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
        minutes=None
        if symbol is not None:
            try:
                frame=minute_lookup(symbol,origin_ms)
                if not isinstance(frame,dict) or frame.get('source_kind')!='BINANCE_SPOT_DEMO_1m':
                    raise ValueError('NATIVE_MINUTE_SOURCE_REQUIRED')
                check=validate_window(frame['original'],frame['rows'],received_at_ms=frame['received_at_ms'])
                if check['state']!='MATCH': raise ValueError('NATIVE_MINUTE_'+check['state'])
                if frame['original'][0]!=origin_ms or any(float(frame['original'][i+1])!=row[k]
                    for i,k in enumerate(('open','high','low','close'))): raise ValueError('NATIVE_PARENT_BINDING')
                minutes=[dict(at=r[0],open=float(r[1]),high=float(r[2]),low=float(r[3]),close=float(r[4])) for r in frame['rows']]
                self.execution_frames.append(dict(symbol=symbol,origin_ms=origin_ms,
                    source_ref=frame['source_ref'],source_kind=frame['source_kind'],rows_sha256=canonical_sha256(frame['rows']),
                    frozen_decision=frame.get('frozen_decision')))
            except (KeyError,TypeError,ValueError,OverflowError) as error:
                self._invalidate('MINUTE_EXECUTION_INVALID:'+str(error)); return self._result(0.,[])
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
            for minute in minutes:
                both=self.exit_mode=='fixed' and minute['low']<=stop and minute['high']>=target
                # Resting protection at a minute open fixes ordering before its high/low.
                if minute['open']<=stop:
                    events.append(self._sell(minute['open'],minute['at'],stop_reason)); break
                if self.exit_mode=='fixed' and minute['open']>=target:
                    events.append(self._sell(target,minute['at'],'TP')); break
                if both:
                    self.ambiguities.append(dict(origin_ms=origin_ms,symbol=position['symbol'],stop=stop,target=target,
                        bucket_origin_ms=minute['at'],resolution_ms=60000,
                        chosen_reason='TP' if self.intrabar_order=='MINUTE_TARGET_FIRST' else stop_reason))
                if both and self.intrabar_order=='MINUTE_TARGET_FIRST':
                    events.append(self._sell(target,minute['at'],'TP')); break
                if minute['low']<=stop:
                    events.append(self._sell(stop,minute['at'],stop_reason)); break
                if self.exit_mode=='fixed' and minute['high']>=target:
                    events.append(self._sell(target,minute['at'],'TP')); break
            if self.position is not None:
                if origin_ms-position['entry_ms']>=15*STEP:
                    events.append(self._sell(row['close'],origin_ms+STEP-1,'HOLDING_LIMIT'))
                else:
                    position['prior_peak']=max(position['prior_peak'],row['high']); position['mark_price']=row['close']
        self.equity=self.cash+(self.position['quantity']*self.position['mark_price'] if self.position else 0.)
        self._mark(self.equity); self.last_origin=origin_ms; self.counters['steps_n']+=1
        self.samples.append(dict(origin_ms=origin_ms,equity_before=before,equity_after=self.equity))
        return self._result((self.equity-before)/100,events)


def run_minute(tape,start,end,*,kind,threshold=0.,policy=None,exit_mode=None,cost='base',
                 intrabar_order='MINUTE_STOP_FIRST',fast_forward=True,signals=None,minute_lookup=None):
    if cost not in ('base','stress'): raise ValueError('INVALID_COST_CASE')
    if kind in ('SIGNAL','EXIT'):
        if signals is None: signals=make_signal_schedule(tape,start,end,threshold)
        if (signals['start'],signals['end'],signals['threshold'])!=(start,end,threshold):
            raise ValueError('SIGNAL_SCHEDULE_BINDING')
    mode=exit_mode or ('learned' if kind in ('MYOPIC','PPO','EXIT') else 'fixed')
    account=MinuteAccount(fee_bps=10 if cost=='base' else 20,slippage_bps=5 if cost=='base' else 10,
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
        decision=dict(origin_ms=at,candidate=candidate,action=int(action),source=source,
            state_sha256=canonical_sha256(dict(cash=account.cash,equity=account.equity,position=account.position,
                entry_killed=account.entry_killed,peak=account.peak_equity)))
        decision['sha256']=canonical_sha256(decision)
        result=account.step(at,candidate,int(action),tape.bar,
            lambda symbol,origin:dict(minute_lookup(symbol,origin,decision),frozen_decision=decision))
        actions.append(dict(origin_ms=at,action=None if source=='SYSTEM_RISK_LOCK' else int(action),source=source))
        if not result['valid']: break
    report=account.report(start,end)
    report.update(kind=kind,threshold=threshold,cost=cost,actions=actions,intrabar_order=intrabar_order,
        ambiguities=account.ambiguities,ambiguous_executions_n=len(account.ambiguities),
        risk_locked_steps_n=locked,policy_calls_n=calls,intent_counts_after_lock_comparable=not bool(locked),
        exit_mode=mode,scenario_kind='NATIVE_MINUTE_CONDITIONAL_SIMULATION',execution_frames=account.execution_frames,
        touch_timestamp_kind='MINUTE_BUCKET_PROXY',native_execution_certified=False)
    if signals: report.update(signal_sha256=signals['sha256'],entry_signals_n=signals['count'])
    return report


class CandidateTape:
    """Copy only already-frozen candidate features, grouped to avoid mmap churn.

    No bar/label lookup and no data-presence candidate admission rule. Held
    noncandidate forecasts keep original cache semantics. Copies preserve dtype.
    """
    def __init__(self,tape,start,end):
        self.tape=tape; self.start=start; self.end=end; self.candidates={}; self.rows={}
        grouped=defaultdict(list)
        for at in range(start,end,STEP):
            symbol=tape.candidate(at); self.candidates[at]=symbol
            if symbol is not None: grouped[symbol].append(at)
        for symbol,times in grouped.items():
            for at in times:
                row=tape.prediction(symbol,at)
                self.rows[at]=None if row is None else row.copy() if hasattr(row,'copy') else tuple(row)
    def identity(self):
        digest=hashlib.sha256();missing=0
        for at,symbol in self.candidates.items():
            row=self.rows.get(at)
            digest.update(canonical_sha256([at,symbol,str(getattr(row,'dtype','python_float64'))]).encode())
            if row is None:digest.update(b'NONE');missing+=int(symbol is not None)
            else:digest.update(struct.pack('<24d',*row))
        return dict(sha256=digest.hexdigest(),origins_n=len(self.candidates),missing_candidate_features_n=missing,
            contract='EXACT_DTYPE_MASK_FLOAT_VALUES_NO_EXECUTION_LOOKUP')
    def candidate(self,at):
        return self.candidates[at] if at in self.candidates else self.tape.candidate(at)
    def prediction(self,symbol,at):
        if symbol is None:return None
        if at in self.candidates and symbol==self.candidates[at]:return self.rows.get(at)
        return self.tape.prediction(symbol,at)
    def vector(self,symbol,at):
        row=self.prediction(symbol,at); return row[:20] if row is not None else None
    def forecast(self,symbol,at,kind):
        row=self.prediction(symbol,at)
        if row is None:return None
        return (float(row[21]),float(row[23])) if kind=='CatBoost' else (float(row[17]),float(row[19]))
    def bar(self,symbol,at):return self.tape.bar(symbol,at)


class MemoPolicy:
    """Exact finite observation cache, only for frozen stateless deterministic MLPs."""
    def __init__(self,policy,model_sha256,*,max_entries=100000):
        if len(model_sha256)!=64 or any(c not in '0123456789abcdef' for c in model_sha256):raise ValueError('MODEL_IDENTITY_REQUIRED')
        if type(max_entries) is not int or not 1<=max_entries<=100000:raise ValueError('MEMO_BUDGET')
        self.policy=policy;self.model_sha256=model_sha256;self.limit=max_entries;self.cache=OrderedDict();self.hits=0;self.calls=0
    @property
    def entries_n(self):return len(self.cache)
    def predict(self,observation,deterministic=True):
        if deterministic is not True or len(observation)!=27 or any(not math.isfinite(float(v)) for v in observation):
            raise ValueError('DETERMINISTIC_FINITE_MLP_OBSERVATION_REQUIRED')
        key=(self.model_sha256,struct.pack('<27d',*observation))
        if key in self.cache:
            self.hits+=1; self.cache.move_to_end(key); return self.cache[key]
        action=self.policy.predict(observation,deterministic=True); self.calls+=1
        if type(action) is not int or action not in (0,1,2,3):raise ValueError('POLICY_ACTION_CONTRACT')
        self.cache[key]=action
        if len(self.cache)>self.limit:self.cache.popitem(last=False)
        return action
