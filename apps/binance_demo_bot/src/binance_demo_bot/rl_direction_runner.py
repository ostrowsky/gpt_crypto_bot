"""RDE-1 research-only orchestration. No exchange client or live authority."""
from pathlib import Path
from collections import OrderedDict
import hashlib
import json
import math
import random

from .rl_direction_core import ResearchAccount, STEP, aggregate_days


def threshold_action(account,tape,at,kind,threshold):
    symbol=account.position['symbol'] if account.position else tape.candidate(at)
    forecast=tape.forecast(symbol,at,kind) if symbol else None
    if forecast is None:return 0
    p,mu=forecast
    if account.position:return 3 if mu <= 0 else 0
    return 2 if p >= .5 and mu > .003+threshold else 0


def run_portfolio(tape,start,end,*,kind,threshold=0.,policy=None,exit_mode=None,cost='base'):
    mode=exit_mode or ('learned' if kind in ('MYOPIC','PPO','EXIT') else 'fixed')
    account=ResearchAccount(fee_bps=10 if cost=='base' else 20,
        slippage_bps=5 if cost=='base' else 10,exit_mode=mode)
    signal_hash=hashlib.sha256();signals=0;actions=[]
    for at in range(start,end,STEP):
        candidate=tape.candidate(at)
        if kind in ('SIGNAL','EXIT'):
            forecast=tape.forecast(candidate,at,'GRU') if candidate else None
            signal=bool(forecast and forecast[0]>=.5 and forecast[1]>.003+threshold)
            signal_hash.update(f'{at}:{candidate}:{int(signal)}\n'.encode());signals+=signal
            if not account.position:action=2 if signal else 0
            elif kind=='EXIT':
                obs=account.observation(at,candidate,tape.vector,remaining=(end-at)/(end-start))
                action=3 if policy.predict(obs,deterministic=True)==3 else 0
            else:action=0
        elif kind in ('PPO','MYOPIC'):
            obs=account.observation(at,candidate,tape.vector,remaining=(end-at)/(end-start))
            action=policy.predict(obs,deterministic=True)
        elif kind=='CASH':action=0
        else:action=threshold_action(account,tape,at,kind,threshold)
        # Decision is frozen before any execution-bar lookup.
        result=account.step(at,candidate,int(action),tape.bar)
        actions.append([at,int(action)])
        if not result['valid']:break
    report=account.report(start,end)
    report.update(kind=kind,threshold=threshold,cost=cost,actions=actions)
    if kind in ('SIGNAL','EXIT'):
        report.update(signal_sha256=signal_hash.hexdigest(),entry_signals_n=signals)
    return report


def buy_hold(tape,start,end,cost='base'):
    fee=.001 if cost=='base' else .002;slip=.0005 if cost=='base' else .001
    first=tape.bar('BTCUSDT',start)
    if first is None:return dict(state='INCOMPLETE',net_pnl_usdt=None,daily_pnl={})
    qty=30/(1+fee)/(first['open']*(1+slip));cash=70.;equity=100.;peak=100.;dd=0.;samples=[]
    for at in range(start,end,STEP):
        row=tape.bar('BTCUSDT',at)
        if row is None:return dict(state='INCOMPLETE',net_pnl_usdt=None,daily_pnl={})
        after=cash+qty*row['close'];samples.append(dict(origin_ms=at,equity_before=equity,equity_after=after))
        equity=after;peak=max(peak,equity);dd=max(dd,peak-equity)
    return dict(aggregate_days(samples,start,end),state='COMPLETE',net_pnl_usdt=equity-100,
        interval_net_pnl_usdt=equity-100,cash_usdt=cash,open_exposure_usdt=qty*row['close'],
        equity_curve=samples,fees_usdt=30-30/(1+fee),max_drawdown_usdt=dd,
        trades_n=0,unmanaged_risk=True,entry_killed=False,cost=cost)


def make_training_set(manifest,train_start,train_end):
    import numpy as np
    from .rl_direction_data import training_indices,sequences,targets
    chunks=[];outcomes=[];provenance=[];counts={}
    for symbol in manifest['symbols']:
        entry=manifest['entries'][symbol];bars=np.load(entry['path'],mmap_mode='r')
        end=min(len(bars),(train_end-entry['first_ms'])//STEP)
        start=max(0,(train_start-entry['first_ms'])//STEP)
        indices=training_indices(bars,int(start),int(end)) if end>start else np.empty(0,dtype=np.int64)
        counts[symbol]=len(indices)
        if len(indices):
            chunks.append(sequences(bars,indices));outcomes.append(targets(bars,indices,fit_end_index=int(end))['returns'])
            for i in indices:
                at=entry['first_ms']+int(i)*STEP
                provenance.append(dict(symbol=symbol,split='TRAIN',feature_end_ms=at-1,origin_ms=at,
                    label_end_ms=at+4*STEP-1,label_available_at_ms=at+4*STEP))
        del bars
    if not chunks:raise ValueError('NO_MATURE_TRAINING_DATA')
    order=np.argsort([r['origin_ms'] for r in provenance],kind='stable')
    return (np.concatenate(chunks)[order],np.concatenate(outcomes)[order],
        [provenance[int(i)] for i in order],counts)


class MarketTape:
    """Read-only causal feature cache; selected/held execution queried separately."""
    def __init__(self,manifest,screen,cache):
        self.manifest=manifest;self.screen=screen;self.cache=Path(cache);self.opened=OrderedDict()
        self.first=manifest['global_first_ms'];self.symbols=screen['symbols']
    def _files(self,symbol):
        import numpy as np
        if symbol not in self.manifest['entries']:return None
        if symbol in self.opened:self.opened.move_to_end(symbol);return self.opened[symbol]
        entry=self.manifest['entries'][symbol];ordinal=self.symbols.index(symbol)
        arrays=(np.load(entry['path'],mmap_mode='r'),
            np.load(self.cache/f'{ordinal:04d}-indices.npy',mmap_mode='r'),
            np.load(self.cache/f'{ordinal:04d}-values.npy',mmap_mode='r'))
        self.opened[symbol]=arrays
        if len(self.opened)>24:self.opened.popitem(last=False)
        return arrays
    def candidate(self,at):
        i=(at-self.first)//STEP
        if i<0 or i>=len(self.screen['symbol_indices']):return None
        symbol=int(self.screen['symbol_indices'][i])
        return self.symbols[symbol] if symbol>=0 else None
    def close(self):
        for arrays in self.opened.values():
            for array in arrays:
                array._mmap.close()
        self.opened.clear()
    def prediction(self,symbol,at):
        if symbol is None:return None
        arrays=self._files(symbol)
        if arrays is None:return None
        _,indices,values=arrays;i=(at-self.manifest['entries'][symbol]['first_ms'])//STEP
        import numpy as np
        pos=int(np.searchsorted(indices,i))
        return values[pos] if pos<len(indices) and indices[pos]==i else None
    def vector(self,symbol,at):
        row=self.prediction(symbol,at)
        return row[:20] if row is not None else None
    def forecast(self,symbol,at,kind):
        row=self.prediction(symbol,at)
        if row is None:return None
        return (float(row[21]),float(row[23])) if kind=='CatBoost' else (float(row[17]),float(row[19]))
    def bar(self,symbol,at):
        import numpy as np
        arrays=self._files(symbol)
        if arrays is None:return None
        bars=arrays[0];i=(at-self.manifest['entries'][symbol]['first_ms'])//STEP
        if i<0 or i>=len(bars) or not np.isfinite(bars[i]).all():return None
        return dict(zip(('open','high','low','close'),map(float,bars[i,:4])))


def cache_predictions(manifest,screen,bundle,folder,progress=None):
    """Only candidate and up-to16 held origins; no future label eligibility."""
    import numpy as np
    from .rl_direction_data import eligible_origins,primitives,_file_sha
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);members={};total=0
    for ordinal,symbol in enumerate(screen['symbols']):
        entry=manifest['entries'][symbol];bars=np.load(entry['path'],mmap_mode='r')
        selected=screen['times_ms'][screen['symbol_indices']==ordinal]
        local=(selected-entry['first_ms'])//STEP
        needed=np.unique((local[:,None]+np.arange(17)[None,:]).ravel())
        indices=np.intersect1d(needed,eligible_origins(bars))
        values=np.empty((len(indices),24),dtype=np.float32);features=primitives(bars)
        for start in range(0,len(indices),2048):
            stop=min(start+2048,len(indices));x=features[indices[start:stop,None]+np.arange(-32,0)[None,:]]
            raw=bundle.predict(x)
            values[start:stop]=np.concatenate([raw[k] for k in
                ('latent','gru_proba','gru_return','cat_proba','cat_return')],axis=1)
        for suffix,value in (('indices',indices),('values',values)):
            path=folder/f'{ordinal:04d}-{suffix}.npy';np.save(path,value);members[path.name]=_file_sha(path)
        total+=len(indices)
        if progress:progress(symbol,ordinal+1,total)
        del bars,features,values
    bundle.assert_frozen()
    manifest_out=dict(rows_n=total,encoder_sha256=bundle.encoder_sha256,files=members,
        vector_order='latent16,p15,p60,mu15,mu60,catp15,catp60,catmu15,catmu60')
    (folder/'manifest.json').write_text(json.dumps(manifest_out,sort_keys=True),encoding='utf-8')
    return manifest_out


def make_training_env(tape,start,end,seed,episode_steps=512):
    import numpy as np
    import gymnasium as gym
    if end-start<episode_steps*STEP:raise ValueError('TRAIN_EPISODE_TOO_SHORT')
    class Environment(gym.Env):
        observation_space=gym.spaces.Box(-np.inf,np.inf,(27,),dtype=np.float32)
        action_space=gym.spaces.Discrete(4)
        def __init__(self):
            self.rng=random.Random(seed);self.invalid_episodes_n=0;self.episodes_n=0
            self.visited_starts=set();self.account=None
        def reset(self,*,seed=None,options=None):
            super().reset(seed=seed)
            i=self.rng.randrange((end-start)//STEP-episode_steps+1)
            self.at=start+i*STEP;self.finish=self.at+episode_steps*STEP
            self.visited_starts.add(self.at);self.episodes_n+=1
            self.account=ResearchAccount(exit_mode='learned')
            return self._obs(),{}
        def _obs(self):
            return np.asarray(self.account.observation(self.at,tape.candidate(self.at),tape.vector,
                remaining=max(0,(self.finish-self.at)/(episode_steps*STEP))),dtype=np.float32)
        def step(self,action):
            result=self.account.step(self.at,tape.candidate(self.at),int(action),tape.bar)
            self.at+=STEP;terminal=self.at>=self.finish or not result['valid']
            if not result['valid']:self.invalid_episodes_n+=1
            return self._obs(),result['reward'],terminal,False,dict(valid=result['valid'])
    return Environment()


def study_registration(manifest,sources,environment,attempt='RDE-1-MAX519-20261029'):
    from .trend_research import time_splits
    from .rl_direction_experiment import SEEDS,THRESHOLDS
    return dict(attempt=attempt,evidence_kind='SIMULATION',promotion_allowed=False,
        holdout_kind='RETROSPECTIVE_REUSED_HOLDOUT',data_sha256=manifest['data_sha256'],
        symbols=manifest['symbols'],source_sha256=sources,environment=environment,
        splits=time_splits(manifest['global_first_ms'],manifest['cutoff_ms']),
        seeds=list(SEEDS),thresholds=list(THRESHOLDS),policy_steps_per_seed=131072,
        candidate='causal_top1_abs4return_over_std32',sequence_shape=[32,5],
        episode_steps=512,primary_horizon_minutes=60,costs={'base':[10,5],'stress':[20,10]},
        max_entry_debit_usdt=30,initial_equity_usdt=100,interval_risk_kill_usdt=1,
        full_truth_harness='FAIL_TH11_CURRENT_SOURCE_HASH',historical_membership_certified=False,
        historical_receipts_certified=False,native_execution_parity_certified=False,
        forecast_heads='CatBoost,GRU;PPO inherits frozen GRU',
        contrasts=['R4-R3','R4-R2','R2-R1','learned-fixed','learned-trailing','R4-cash'])


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.next')
    temporary.write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def forecast_evaluation(tape,start,end):
    import numpy as np
    from .rl_direction_experiment import direction_metrics
    from .rl_direction_data import targets
    accumulator={(name,h):[[],[],[],[]] for name in ('CatBoost','GRU') for h in (0,1)}
    requested=0;missing=0;sample=None
    for at in range(start+16*STEP,end-16*STEP,STEP):
        symbol=tape.candidate(at)
        if symbol is None:continue
        requested+=1;raw=tape.prediction(symbol,at)
        if raw is None:missing+=1;continue
        bars=tape._files(symbol)[0];entry=tape.manifest['entries'][symbol]
        i=(at-entry['first_ms'])//STEP
        # Only five rows are needed for direct labels; scoring does not screen entries.
        tiny=bars[i-1:i+4]
        target=targets(tiny,np.array([1]),fit_end_index=len(tiny))
        for h in (0,1):
            if not target['valid'][0,h]:continue
            y=float(target['labels'][0,h]);actual=float(target['returns'][0,h])
            for name,pcol,mcol in (('GRU',16+h,18+h),('CatBoost',20+h,22+h)):
                row=accumulator[name,h];row[0].append(y);row[1].append(float(raw[pcol]))
                row[2].append(actual);row[3].append(float(raw[mcol]))
        if sample is None and target['valid'].all():
            history=bars[max(0,i-32):i,3]
            sample=dict(origin_ms=at,symbol=symbol,last_close=float(bars[i-1,3]),
                history_ms=[entry['first_ms']+j*STEP+STEP-1 for j in range(max(0,i-32),i)],
                history=list(map(float,history)),actual=list(map(float,bars[i:i+4,3])),
                CatBoost={'p_up':list(map(float,raw[20:22])),'return':list(map(float,raw[22:24]))},
                GRU={'p_up':list(map(float,raw[16:18])),'return':list(map(float,raw[18:20]))})
    return dict(candidate_origins_N=requested,missing_feature_cache_n=missing,
        scores=[dict(model=name,horizon_minutes=(15,60)[h],**direction_metrics(*values))
                for (name,h),values in accumulator.items()],sample=sample,
        omitted_label_counts={f'{n}-{(15,60)[h]}':requested-len(v[0]) for (n,h),v in accumulator.items()})


def execute_study(prepared,output):
    """Finite, credential-free run, fixed budgets and no trading promotion."""
    import sys
    import traceback
    import numpy as np
    from . import rl_direction_data as data,rl_direction_models as models
    from .rl_direction_experiment import register_run,SEEDS,THRESHOLDS,select_threshold,seed_daily_mean,comparison_verdict
    from .rl_direction_statistics import paired_block_intervals
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    def status(phase,**extra):
        from datetime import datetime,timezone
        row=dict(phase=phase,at_utc=datetime.now(timezone.utc).isoformat(),**extra)
        _write(output/'status.json',row);print(json.dumps(row),flush=True)
    try:
        manifest=json.loads((Path(prepared)/'manifest.json').read_text(encoding='utf-8'))
        # The loader population is also bound to the collector's independent registration.
        reference=json.loads((Path(manifest['archive']).parent/'history-registration.json').read_text(encoding='utf-8'))
        if set(reference['symbols'])!=set(manifest['symbols']):raise ValueError('REGISTERED_POPULATION_MISMATCH')
        sources={p.name:data._file_sha(p) for p in Path(__file__).parent.glob('rl_direction_*.py')}
        spec=Path(__file__).resolve().parents[4]/'docs/specs/binance-demo-rl-direction-experiment.md'
        sources['spec']=data._file_sha(spec)
        environment=models.environment_manifest()
        registration=study_registration(manifest,sources,environment)
        registration_sha=register_run(output,registration)
        _write(output/'environment.json',environment)
        for name,entry in manifest['entries'].items():
            if data._file_sha(entry['path'])!=entry['sha256']:raise ValueError('DATA_ARRAY_CHANGED:'+name)
        train,tune,test=registration['splits']
        status('CAUSAL_SCREEN',symbols_N=len(manifest['symbols']))
        screen=data.candidate_screen(manifest)
        np.savez(output/'screen.npz',times_ms=screen['times_ms'],symbol_indices=screen['symbol_indices'],scores=screen['scores'])
        _write(output/'screen.json',dict(symbols=screen['symbols'],eligible_counts=screen['eligible_counts'],
            total_selected_n=screen['total_selected_n'],sha256=data._file_sha(output/'screen.npz')))
        model_folder=output/'predictors'
        if model_folder.exists():bundle=models.load_predictors(model_folder)
        else:
            status('SUPERVISED_TRAIN_PREPARATION')
            x,y,provenance,counts=make_training_set(manifest,train[1],train[2])
            _write(output/'training-population.json',dict(samples_n=len(x),per_symbol=counts,
                maximum_per_symbol=256,provenance_sha256=data._sha(provenance),
                trained_symbols_n=sum(v>0 for v in counts.values()),source_symbols_N=len(counts)))
            status('CATBOOST_GRU_TRAINING',samples_n=len(x))
            bundle=models.train_predictors(x,y,model_folder,provenance=provenance,fit_cutoff_ms=train[2])
            del x,y,provenance
        binding=dict(registration_sha256=registration_sha,encoder_sha256=bundle.encoder_sha256,
            predictor_manifest_sha256=bundle.artifact_manifest['manifest_sha256'],data_sha256=manifest['data_sha256'])
        _write(output/'frozen-encoder.json',binding)
        cache=output/'features'
        if not cache.exists():
            status('FROZEN_FEATURE_CACHE')
            cache_predictions(manifest,screen,bundle,cache,
                lambda symbol,n,total:status('FROZEN_FEATURE_CACHE',symbol=symbol,symbols_done=n,cached_rows=total) if n%10==0 else None)
        cached=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
        if cached['encoder_sha256']!=bundle.encoder_sha256:raise ValueError('CACHE_ENCODER_MISMATCH')
        for name,sha in cached['files'].items():
            if data._file_sha(cache/name)!=sha:raise ValueError('CACHE_HASH')
        tape=MarketTape(manifest,screen,cache)
        status('FORECAST_EVALUATION')
        forecasts={segment[0]:forecast_evaluation(tape,segment[1],segment[2]) for segment in (tune,test)}
        _write(output/'forecasts.json',forecasts)
        thresholds={};tune_search=[]
        def save(label,segment,cost,report):
            filename=f'{segment}-{cost}-{label}.json'
            _write(output/'portfolios'/filename,report)
            return dict({k:v for k,v in report.items() if k not in ('events','trades','equity_curve','actions','days')},
                label=label,segment=segment,artifact=filename)
        for kind in ('CatBoost','GRU'):
            rows=[]
            for threshold in THRESHOLDS:
                report=run_portfolio(tape,tune[1],tune[2],kind=kind,threshold=threshold)
                row=dict(threshold=threshold,segment='tune',mean_daily_net_pnl_usdt=report['mean_daily_net_pnl_usdt'])
                rows.append(row);tune_search.append(dict(model=kind,**row))
                _write(output/'threshold-search.json',tune_search)
            thresholds[kind]=select_threshold(rows)
        _write(output/'frozen-thresholds.json',dict(thresholds=thresholds,source='TUNE_ONLY',binding=binding))
        policies={};training=[]
        for kind,gamma in (('MYOPIC',0),('PPO',1)):
            policies[kind]={}
            for seed in SEEDS:
                status('POLICY_TRAINING',model=kind,seed=seed,steps=131072)
                folder=output/'policies'/f'{kind}-{seed}'
                if folder.exists():
                    handle=models.load_policy(folder)
                    upstream=json.loads((folder/'upstream.json').read_text())
                    if upstream['binding']!=binding:raise ValueError('POLICY_UPSTREAM_CHANGED')
                    train_row=upstream['training']
                else:
                    env=make_training_env(tape,train[1]+33*STEP,train[2]-16*STEP,seed)
                    handle=models.fit_policy(env,folder,seed=seed,gamma=gamma)
                    train_row=dict(model=kind,seed=seed,episodes_n=env.episodes_n,
                        distinct_episode_starts_n=len(env.visited_starts),invalid_episodes_n=env.invalid_episodes_n,
                        actual_steps=handle.manifest['actual_steps'])
                    _write(folder/'upstream.json',dict(binding=binding,training=train_row))
                    env.close()
                training.append(train_row);policies[kind][seed]=handle
                _write(output/'policy-training.json',training);bundle.assert_frozen()
        results=[]
        for segment,start,end in (tune,test):
            for cost in ('base','stress'):
                status('CONTINUOUS_PORTFOLIO_REPLAY',segment=segment,cost=cost)
                for kind in ('CASH','CatBoost','GRU'):
                    results.append(save(kind,segment,cost,run_portfolio(tape,start,end,kind=kind,
                        threshold=thresholds.get(kind,0),cost=cost)))
                results.append(save('BTC_BUY_HOLD',segment,cost,buy_hold(tape,start,end,cost)))
                for kind in ('MYOPIC','PPO'):
                    for seed in SEEDS:
                        report=run_portfolio(tape,start,end,kind=kind,policy=policies[kind][seed],cost=cost)
                        report['seed']=seed;results.append(save(f'{kind}-{seed}',segment,cost,report))
                for mode in ('fixed','trailing'):
                    results.append(save('EXIT_'+mode,segment,cost,run_portfolio(tape,start,end,kind='SIGNAL',
                        threshold=thresholds['GRU'],exit_mode=mode,cost=cost)))
                for seed in SEEDS:
                    report=run_portfolio(tape,start,end,kind='EXIT',policy=policies['PPO'][seed],
                        threshold=thresholds['GRU'],cost=cost)
                    report['seed']=seed;results.append(save(f'EXIT_LEARNED-{seed}',segment,cost,report))
                _write(output/'portfolios.json',results)
        comparisons=[]
        base={r['label']:r for r in results if r['segment']=='test' and r['cost']=='base'}
        stress={r['label']:r for r in results if r['segment']=='test' and r['cost']=='stress'}
        def aggregate(prefix,rows):
            matched=[rows[f'{prefix}-{seed}'] for seed in SEEDS]
            if any(r['state']!='COMPLETE' for r in matched):raise ValueError('INCOMPLETE_SEED_REPLAY')
            return dict(daily_pnl=seed_daily_mean(matched,SEEDS),
                net_pnl_usdt=sum(r['net_pnl_usdt'] for r in matched)/len(SEEDS),
                max_drawdown_usdt=max(r['max_drawdown_usdt'] for r in matched))
        for rows in (base,stress):
            for prefix in ('MYOPIC','PPO','EXIT_LEARNED'):rows[prefix]=aggregate(prefix,rows)
        for candidate,reference in (('PPO','MYOPIC'),('PPO','GRU'),('GRU','CatBoost'),
                                    ('EXIT_LEARNED','EXIT_fixed'),('EXIT_LEARNED','EXIT_trailing'),('PPO','CASH')):
            effect=paired_block_intervals(base[candidate]['daily_pnl'],base[reference]['daily_pnl'])
            verdict=comparison_verdict(paired_n=effect['paired_n'],delta_lower=effect['delta_lower'],
                absolute_lower=effect['absolute_lower'],stress_net=stress[candidate]['net_pnl_usdt'],
                max_dd=base[candidate]['max_drawdown_usdt'],reference_dd=base[reference]['max_drawdown_usdt'])
            comparisons.append(dict(candidate=candidate,reference=reference,**effect,verdict=verdict))
        _write(output/'comparisons.json',comparisons)
        status('RESEARCH_REPORT')
        report_study(output,registration,results,comparisons,forecasts,training)
        tape.close();bundle.assert_frozen()
        state='COMPLETE_RETROSPECTIVE_RESEARCH' if all(r['state']=='COMPLETE' for r in results) and not any(r['invalid_episodes_n'] for r in training) else 'INCOMPLETE_RESEARCH'
        status(state,promotion_allowed=False,full_truth_harness='FAIL',models_trained=12)
    except BaseException as error:
        status('INCOMPLETE_RESEARCH',error=type(error).__name__+':'+str(error),traceback=traceback.format_exc(),promotion_allowed=False)
        raise


def report_study(output,registration,results,comparisons,forecasts,training):
    """Portable evidence tables and static graphs; historical research only."""
    import html
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime,timezone
    output=Path(output);plots=output/'plots';plots.mkdir(exist_ok=True)
    rows=[r for r in results if r['segment']=='test' and r['cost']=='base']
    lines=['# RDE-1: CatBoost / GRU / PPO',
        'SIMULATION · RETROSPECTIVE_REUSED_HOLDOUT · promotion_allowed=false',
        'Full Truth Harness: **FAIL TH-11**. PIT membership/receipt and native execution parity uncertified.',
        '100USDT shared cash; max entry30; latched interval risk kill1USDT;15m bars; direction15/60m.',
        'Historical studies reused this TEST. No actual Demo account outcome or profitability certification.',
        '', '| Model | Net interval USDT | Mean complete day | Trades | Drawdown | Entry killed |',
        '|---|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['label']} | {r.get('net_pnl_usdt')} | {r.get('mean_daily_net_pnl_usdt')} | {r.get('trades_n')} | {r.get('max_drawdown_usdt')} | {r.get('entry_killed')} |")
    lines+=['','## Direct direction forecasts','| Model | Horizon | N | Accuracy | Balanced accuracy | Brier | Return MAE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in forecasts['test']['scores']:
        lines.append(f"| {r['model']} | {r['horizon_minutes']} | {r['rows_N']} | {r['accuracy']} | {r['balanced_accuracy']} | {r['brier']} | {r['return_mae']} |")
    lines+=['','## Preregistered contrasts (all seeds)',
        '| Candidate / reference | N days | Mean delta/day |99.1667% lower | Absolute lower | Verdict |',
        '|---|---:|---:|---:|---:|---|']
    for r in comparisons:
        lines.append(f"| {r['candidate']} / {r['reference']} | {r['paired_n']} | {r['delta_mean']} | {r['delta_lower']} | {r['absolute_lower']} | {r['verdict']} |")
    lines+=['','## Limits','No independent evidence of daily profits. Five seeds retained; no TEST tuning. '
        'PPO inherits the GRU forecasts and optimizes control; action probability is not p_up. '
        'Learned exits transfer a jointly trained PPO. Fixed top1 momentum screen limits the population. '
        'Current-survivor archive omits delisted universe. OHLC execution assumes zero publication latency, '
        'fractional sizes and declared adverse costs; no queue/native filter parity. '
        'Training budget131072steps/seed is finite, not a conclusion about all RL. '
        'Any invalid training episode keeps study INCOMPLETE. All partial calendar days excluded with denominators in artifacts. '
        'Next: freeze prospective predictions/policies, collect later outcomes without promotion; assess power first.',
        '', 'Training diagnostics: '+json.dumps(training,sort_keys=True)]
    for r in rows:
        detail=json.loads((output/'portfolios'/r['artifact']).read_text())
        curve=detail.get('equity_curve',[])
        if not curve:continue
        fig,ax=plt.subplots(figsize=(11,3));ax.plot([datetime.fromtimestamp(v['origin_ms']/1000,timezone.utc) for v in curve],
            [v['equity_after'] for v in curve],lw=.8)
        ax.axhline(100,color='gray',ls='--');ax.set(title=r['label']+' | continuous TEST SIMULATION',ylabel='MTM equity USDT')
        ax.grid(alpha=.2);fig.tight_layout();fig.savefig(plots/(r['label']+'-equity.png'),dpi=130);plt.close(fig)
    sample=forecasts['test']['sample']
    if sample:
        for name,source in (('CatBoost','CatBoost'),('GRU','GRU'),('PPO','GRU')):
            at=sample['origin_ms'];origin=datetime.fromtimestamp(at/1000,timezone.utc)
            fig,ax=plt.subplots(figsize=(10,4))
            ax.plot([datetime.fromtimestamp(v/1000,timezone.utc) for v in sample['history_ms']],sample['history'],color='black',label='History known at origin')
            future=[datetime.fromtimestamp((at+j*STEP+STEP-1)/1000,timezone.utc) for j in range(4)]
            ax.plot([origin]+future,[sample['last_close']]+sample['actual'],color='gray',label='Later actual, withheld')
            points=[sample['last_close']*math.exp(v) for v in sample[source]['return']]
            ax.plot([origin,future[0],future[-1]],[sample['last_close']]+points,'o--',label='Direct15/60m estimates; guides between points')
            ax.axvline(origin,color='gray',ls=':');ax.set(title=f"{sample['symbol']} | {name}"+(' | shared frozen GRU forecast' if name=='PPO' else ''),ylabel='USDT',xlabel='UTC')
            ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(plots/(name+'-forecast.png'),dpi=140);plt.close(fig)
    _write(output/'report.json',dict(registration=registration,comparisons=comparisons,
        forecasts=forecasts,models=rows,training=training,promotion_allowed=False))
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    images=''.join(f'<h3>{html.escape(p.stem)}</h3><img src="plots/{p.name}" style="max-width:100%">' for p in sorted(plots.glob('*.png')))
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>RDE-1 research</title>'
        '<style>body{max-width:1200px;margin:auto;font:16px system-ui}pre{white-space:pre-wrap}</style>'
        '<h1>RDE-1 · SIMULATION · NO PROMOTION</h1><pre>'+html.escape('\n'.join(lines))+'</pre>'+images,encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();execute_study(args.prepared,args.output)
