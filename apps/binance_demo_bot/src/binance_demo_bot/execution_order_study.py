"""EOS-1 offline, frozen RDE-1 execution-convention sensitivity study."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import io
import json
import os
from pathlib import Path
import traceback
from zoneinfo import ZoneInfo

from .execution_order_sensitivity import run_scenario, make_signal_schedule, compare_reference, financial_check
from .intrabar_probe import _seal, _read_sealed, _write, _parent_binding, _single_writer
from .rl_direction_data import _file_sha


SEEDS=(11,23,37,53,71)
ORDERS=('STOP_FIRST','TARGET_FIRST')
AFFECTED=('CatBoost','GRU','EXIT_fixed')


def make_plan(parent,thresholds):
    if tuple(parent['seeds'])!=SEEDS: raise ValueError('SEED_POPULATION_CHANGED')
    if [s[0] for s in parent['splits']]!=['train','tune','test']: raise ValueError('SPLIT_POPULATION_CHANGED')
    arms=[('CASH','CASH',None,None),('CatBoost','CatBoost',None,None),('GRU','GRU',None,None),
          ('BTC_BUY_HOLD','BTC_BUY_HOLD',None,None)]
    arms += [(f'{kind}-{seed}',kind,seed,None) for kind in ('MYOPIC','PPO') for seed in SEEDS]
    arms += [('EXIT_fixed','SIGNAL',None,'fixed'),('EXIT_trailing','SIGNAL',None,'trailing')]
    arms += [(f'EXIT_LEARNED-{seed}','EXIT',seed,None) for seed in SEEDS]
    return [dict(id=f'{order}-{segment}-{cost}-{label}',order=order,segment=segment,start=start,end=end,
                 cost=cost,label=label,kind=kind,seed=seed,exit_mode=mode,
                 threshold=thresholds['GRU'] if kind in ('SIGNAL','EXIT') else thresholds.get(kind,0.))
        for order in ORDERS for segment,start,end in parent['splits'][1:] for cost in ('base','stress')
        for label,kind,seed,mode in arms]


def verify_files(files):
    total=0
    for name,expected in files.items():
        path=Path(name)
        if not path.is_file() or _file_sha(path)!=expected: raise ValueError('INPUT_FILE_HASH_CHANGED:'+name)
        total+=path.stat().st_size
    return dict(files_n=len(files),bytes_n=total)


def seal_registration(folder,binding):
    _seal(Path(folder)/'registration.json',binding)
    return _file_sha(Path(folder)/'registration.json')


def save_report(folder,task,registration_sha,detail):
    folder=Path(folder)/'portfolios'; folder.mkdir(parents=True,exist_ok=True)
    path=folder/(task['id']+'.json.gz'); temporary=path.with_suffix('.next')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8') as stream:
                json.dump(detail,stream,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    temporary.replace(path)
    _seal(path.with_suffix('.meta.json'),dict(id=task['id'],registration_sha256=registration_sha,
        sha256=_file_sha(path),format='gzip-json-v1'))


def load_report(folder,task,registration_sha):
    path=Path(folder)/'portfolios'/(task['id']+'.json.gz'); meta=path.with_suffix('.meta.json')
    if not meta.exists(): return None
    binding=_read_sealed(meta)
    if binding['id']!=task['id'] or binding['registration_sha256']!=registration_sha:
        raise ValueError('REPORT_REGISTRATION_BINDING_CHANGED')
    if not path.is_file() or _file_sha(path)!=binding['sha256']: raise ValueError('REPORT_INTEGRITY')
    with gzip.open(path,'rt',encoding='utf-8') as stream: return json.load(stream)


def group_results(rows):
    groups=defaultdict(list)
    for row in rows:
        label=row['label']
        family=label.rsplit('-',1)[0] if label.startswith(('MYOPIC-','PPO-','EXIT_LEARNED-')) else label
        groups[row['order'],row['segment'],row['cost'],family].append(row)
    result=[]
    fields=('net_pnl_usdt','mean_daily_net_pnl_usdt','trades_n','open_exposure_usdt','active_days_n',
            'ambiguous_executions_n','risk_locked_steps_n','fees_usdt')
    for (order,segment,cost,label),members in groups.items():
        if label in ('MYOPIC','PPO','EXIT_LEARNED'):
            if sorted(int(m['label'].rsplit('-',1)[1]) for m in members)!=list(SEEDS):
                raise ValueError('GROUP_SEED_POPULATION_CHANGED')
        if any(m['state']!='COMPLETE' for m in members): raise ValueError('GROUP_OBJECTIVE_INCOMPLETE')
        if len({(m['complete_days_n'],m['partial_days_n']) for m in members})!=1:
            raise ValueError('GROUP_CALENDAR_COVERAGE_MISMATCH')
        means={field:(sum(m[field] for m in members)/len(members) if all(m[field] is not None for m in members) else None)
               for field in fields}
        values=[m['net_pnl_usdt'] for m in members]
        result.append(dict(order=order,segment=segment,cost=cost,label=label,runs_n=len(members),
            **means,net_min_usdt=min(values),net_max_usdt=max(values),max_drawdown_usdt=max(m['max_drawdown_usdt'] for m in members),
            complete_days_n=members[0]['complete_days_n'],partial_days_n=members[0]['partial_days_n'],
            open_exposure_max_usdt=max(m['open_exposure_usdt'] for m in members),
            individual_net_usdt={m['label']:m['net_pnl_usdt'] for m in members}))
    return result


def capture_inputs(parent_folder,prepared_folder,spec_path,truth_report):
    from . import rl_direction_models as models
    from .rl_direction_runner import validate_policy_identity
    parent_folder,prepared_folder=Path(parent_folder).resolve(),Path(prepared_folder).resolve()
    spec_path=Path(spec_path).resolve(); files={}
    def add(path,expected=None):
        path=Path(path).resolve(); digest=_file_sha(path)
        if expected is not None and digest!=expected: raise ValueError('REGISTERED_INPUT_HASH:'+str(path))
        files[str(path)]=digest; return digest
    def read(path):
        add(path); return json.loads(Path(path).read_text(encoding='utf-8'))
    manifest=read(prepared_folder/'manifest.json')
    parent=_parent_binding(Path(manifest['archive']).resolve(),parent_folder,prepared_folder/'manifest.json')
    if len(parent['symbols'])!=519 or parent['attempt']!='RDE-1-MAX519-20261029': raise ValueError('PARENT_STUDY_SCOPE')
    registration_sha=add(parent_folder/'registration.json')
    for name,expected in parent['source_sha256'].items():
        path=(Path(__file__).parent/name) if name!='spec' else Path(__file__).resolve().parents[4]/'docs/specs/binance-demo-rl-direction-experiment.md'
        add(path,expected)
    for entry in manifest['entries'].values(): add(entry['path'],entry['sha256'])
    add(parent_folder/'screen.npz'); screen_meta=read(parent_folder/'screen.json')
    if screen_meta['symbols']!=manifest['symbols']: raise ValueError('SCREEN_POPULATION_MISMATCH')
    encoder=read(parent_folder/'frozen-encoder.json'); predictors=models._read_manifest(parent_folder/'predictors')
    read(parent_folder/'predictors/manifest.json')
    for name,expected in predictors['files'].items(): add(parent_folder/'predictors'/name,expected)
    if (encoder['registration_sha256']!=registration_sha or encoder['data_sha256']!=manifest['data_sha256'] or
        encoder['predictor_manifest_sha256']!=predictors['manifest_sha256'] or
        encoder['encoder_sha256']!=predictors['encoder_sha256'] or
        predictors['source_sha256']!=parent['source_sha256']['rl_direction_models.py']):
        raise ValueError('ENCODER_BINDING_CHANGED')
    cached=read(parent_folder/'features/manifest.json')
    if cached['encoder_sha256']!=encoder['encoder_sha256']: raise ValueError('CACHE_ENCODER_CHANGED')
    for name,expected in cached['files'].items(): add(parent_folder/'features'/name,expected)
    threshold_binding=read(parent_folder/'frozen-thresholds.json')
    if threshold_binding['source']!='TUNE_ONLY' or threshold_binding['binding']!=encoder:
        raise ValueError('THRESHOLD_UPSTREAM_CHANGED')
    thresholds=threshold_binding['thresholds']; plan=make_plan(parent,thresholds)
    episode_sha=add(parent_folder/'training-episode-starts.npy')
    expected_policy_binding=dict(encoder,train_episode_starts_sha256=episode_sha)
    for kind,gamma in (('MYOPIC',0),('PPO',1)):
        for seed in SEEDS:
            folder=parent_folder/'policies'/f'{kind}-{seed}'
            policy=models._read_manifest(folder); read(folder/'manifest.json')
            validate_policy_identity(policy,seed=seed,gamma=gamma)
            upstream=read(folder/'upstream.json'); sealed=read(parent_folder/'policy-bindings'/f'{kind}-{seed}.json')
            if (upstream['binding']!=expected_policy_binding or sealed['binding']!=expected_policy_binding or
                sealed['policy_manifest_sha256']!=policy['manifest_sha256'] or
                upstream['training']['invalid_episodes_n']!=0 or policy['source_sha256']!=parent['source_sha256']['rl_direction_models.py']):
                raise ValueError('POLICY_UPSTREAM_CHANGED')
            for name,expected in policy['files'].items(): add(folder/name,expected)
    for task in plan:
        add(parent_folder/'portfolios'/f"{task['segment']}-{task['cost']}-{task['label']}.json")
    add(spec_path)
    for name in ('execution_order_sensitivity.py','execution_order_study.py'): add(Path(__file__).with_name(name))
    truth=read(truth_report)
    if truth.get('profile')!='full': raise ValueError('FULL_TRUTH_REPORT_REQUIRED')
    return dict(protocol='EOS-1-FROZEN-RDE1-20261030',parent_folder=str(parent_folder),prepared_folder=str(prepared_folder),
        parent_registration_sha256=registration_sha,data_sha256=manifest['data_sha256'],
        symbols_N=len(parent['symbols']),raw_bars_n=sum(e['raw_rows_n'] for e in manifest['entries'].values()),
        splits=parent['splits'],thresholds=thresholds,seeds=list(SEEDS),plan=plan,files=files,
        truth_harness_at_registration=dict(path=str(Path(truth_report).resolve()),status=truth['status'].upper()),
        evidence_kind='COUNTERFACTUAL_SIMULATION',holdout='RETROSPECTIVE_REUSED_HOLDOUT',
        native_execution_certified=False,mathematical_pnl_bounds=False,promotion_allowed=False)


def _summary(task,report,check,parity):
    active={datetime.fromtimestamp(e['at_ms']/1000,timezone.utc).astimezone(ZoneInfo('Europe/Budapest')).date().isoformat()
            for e in report.get('events',[])}
    benchmark_entry=task['start'] if report.get('unmanaged_risk',False) else None
    if benchmark_entry is not None:
        active.add(datetime.fromtimestamp(benchmark_entry/1000,timezone.utc).astimezone(ZoneInfo('Europe/Budapest')).date().isoformat())
    fields=('state','net_pnl_usdt','mean_daily_net_pnl_usdt','trades_n','max_drawdown_usdt','open_exposure_usdt',
            'complete_days_n','partial_days_n','fees_usdt','entry_killed','observed_steps_n','requested_steps_N')
    return dict(task,**{k:report.get(k) for k in fields},active_days_n=len(active),
        ambiguous_executions_n=report.get('ambiguous_executions_n',0),risk_locked_steps_n=report.get('risk_locked_steps_n',0),
        financial_check=check,reference_parity=parity,
        first_fill_ms=min((e['at_ms'] for e in report.get('events',[])),default=benchmark_entry),
        last_fill_ms=max((e['at_ms'] for e in report.get('events',[])),default=benchmark_entry))


@_single_writer
def execute_study(parent_folder,prepared_folder,output,truth_report):
    output=Path(output).resolve(); parent_folder=Path(parent_folder).resolve(); prepared_folder=Path(prepared_folder).resolve()
    def status(phase,**extra):
        value=dict(phase=phase,pid=os.getpid(),at_utc=datetime.now(timezone.utc).isoformat(),promotion_allowed=False,**extra)
        _write(output/'status.json',value)
        with (output/'status-events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(value,ensure_ascii=False)+'\n')
        print(json.dumps(value),flush=True)
    tape=None
    try:
        status('VERIFYING_FROZEN_INPUTS')
        spec=Path(__file__).resolve().parents[4]/'docs/specs/binance-demo-execution-order-sensitivity.md'
        binding=capture_inputs(parent_folder,prepared_folder,spec,truth_report)
        import numpy as np
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        from . import rl_direction_models as models
        environment=models.environment_manifest()
        parent=json.loads((parent_folder/'registration.json').read_text(encoding='utf-8'))
        if environment['python']!=parent['environment']['python'] or environment['versions']!=parent['environment']['versions']:
            raise ValueError('FROZEN_RESEARCH_ENVIRONMENT_CHANGED')
        binding['environment']=environment
        registration_sha=seal_registration(output,binding)
        _write(output/'input-verification.json',dict(state='MATCH',files_n=len(binding['files']),symbols_N=binding['symbols_N']))
        manifest=json.loads((prepared_folder/'manifest.json').read_text(encoding='utf-8'))
        from .rl_direction_runner import MarketTape,buy_hold
        raw=np.load(parent_folder/'screen.npz'); screen_meta=json.loads((parent_folder/'screen.json').read_text(encoding='utf-8'))
        screen=dict(symbols=screen_meta['symbols'],times_ms=raw['times_ms'],symbol_indices=raw['symbol_indices'],scores=raw['scores'])
        tape=MarketTape(manifest,screen,parent_folder/'features')
        status('LOADING_FROZEN_POLICIES',models_n=10)
        policies={(kind,seed):models.load_policy(parent_folder/'policies'/f'{kind}-{seed}') for kind in ('MYOPIC','PPO') for seed in SEEDS}
        schedules={}
        for segment,start,end in parent['splits'][1:]:
            status('FROZEN_SIGNAL_SCHEDULE',segment=segment)
            schedule=make_signal_schedule(tape,start,end,binding['thresholds']['GRU']); schedules[segment]=schedule
            _seal(output/f'{segment}-signals.json',{k:v for k,v in schedule.items() if k not in ('signals','candidates')})
        summaries=[]
        for task in binding['plan']:
            report=load_report(output,task,registration_sha)
            if report is None:
                status('CONTINUOUS_REPLAY',done_n=len(summaries),total_N=168,task=task['id'])
                if task['kind']=='BTC_BUY_HOLD': report=buy_hold(tape,task['start'],task['end'],task['cost'])
                else:
                    policy_kind='PPO' if task['kind']=='EXIT' else task['kind']
                    policy=policies.get((policy_kind,task['seed']))
                    report=run_scenario(tape,task['start'],task['end'],kind=task['kind'],threshold=task['threshold'],
                        policy=policy,exit_mode=task['exit_mode'],cost=task['cost'],intrabar_order=task['order'],
                        signals=schedules[task['segment']] if task['kind'] in ('SIGNAL','EXIT') else None)
                if task['seed'] is not None: report['seed']=task['seed']
                check=financial_check(report)
                if check['state']!='PASS': raise ValueError('FINANCIAL_CHECK:'+task['id']+':'+str(check))
                reference=json.loads((parent_folder/'portfolios'/f"{task['segment']}-{task['cost']}-{task['label']}.json").read_text(encoding='utf-8'))
                required=task['order']=='STOP_FIRST' or task['label'] not in AFFECTED
                parity=compare_reference(reference,report) if required else dict(state='NOT_REQUIRED_CHANGED_CONVENTION')
                if required and parity['state']!='MATCH': raise ValueError('BASELINE_PARITY:'+task['id']+':'+str(parity))
                if task['kind'] in ('SIGNAL','EXIT') and (report['signal_sha256'],report['entry_signals_n'])!=(reference['signal_sha256'],reference['entry_signals_n']):
                    raise ValueError('FROZEN_SIGNAL_POPULATION_CHANGED')
                report['eos_financial_check']=check; report['eos_reference_parity']=parity
                save_report(output,task,registration_sha,report)
            summary=_summary(task,report,report['eos_financial_check'],report['eos_reference_parity'])
            summaries.append(summary); _write(output/'progress.json',summaries)
            del report
        status('FINAL_INPUT_RECHECK',done_n=len(summaries))
        verification=verify_files(binding['files']); _write(output/'final-input-verification.json',dict(state='MATCH',**verification))
        status('RENDERING_REPORT',done_n=len(summaries))
        render_study(output,binding,registration_sha,summaries)
        status('COMPLETE_COUNTERFACTUAL_STUDY',portfolios_n=len(summaries),baseline_matches_n=84,
               invariant_target_matches_n=72,financial_checks_passed_n=168,full_truth_harness=binding['truth_harness_at_registration']['status'])
    except BaseException as error:
        status('INCOMPLETE_STUDY',error=type(error).__name__+':'+str(error),traceback=traceback.format_exc())
        raise
    finally:
        if tape is not None: tape.close()


def render_study(output,binding,registration_sha,rows):
    groups=group_results(rows)
    parity_n=sum(r['reference_parity']['state']=='MATCH' for r in rows)
    report=dict(protocol=binding['protocol'],groups=groups,portfolios=rows,baseline_parity_n=parity_n,
        financial_passes_n=sum(r['financial_check']['state']=='PASS' for r in rows),
        parent_registration_sha256=binding['parent_registration_sha256'],full_portfolio_replayed=True,
        actual_execution_reconstructed=False,evidence_kind=binding['evidence_kind'],promotion_allowed=False)
    _write(output/'report.json',report)
    for filename,data in (('results.csv',rows),('groups.csv',groups)):
        fields=sorted({k for r in data for k,v in r.items() if not isinstance(v,(dict,list))})
        with (output/filename).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore'); writer.writeheader()
            writer.writerows({k:v for k,v in r.items() if k in fields} for r in data)
    labels=('CASH','CatBoost','GRU','MYOPIC','PPO','EXIT_fixed','EXIT_trailing','EXIT_LEARNED','BTC_BUY_HOLD')
    lines=['# EOS-1: чувствительность результата к порядку исполнения','',
        'COUNTERFACTUAL SIMULATION · RETROSPECTIVE REUSED HOLDOUT · без продвижения в торговлю.',
        f"Полный Truth Harness: **{binding['truth_harness_at_registration']['status']}**. Старый пробел TH-11 сохраняется.",
        f"Исходные данные: {binding['symbols_N']} пар, {binding['raw_bars_n']} свечей; те же TUNE/TEST и все пять инициализаций.",
        '100 USDT на независимый прогон; максимум входа30; непрерывный лимит потерь/просадки1 USDT.',
        f'Выполнено {len(rows)}/168 прогонов. Базовых финансовых совпадений84/84; инвариантных72/72; сверок баланса168/168.',
        '', 'STOP_FIRST: при обоих затронутых уровнях первым считается ограничение убытка.',
        'TARGET_FIRST: первым считается фиксация прибыли. Это идеализированное допущение, не установленная история.',
        'Сценарии не являются математическими границами портфельного результата: последующие сделки меняются.',
        'Для трёх семейств нейросетевых стратегий указано среднее пяти независимых прогонов, а не лучшая инициализация.']
    for segment,cost in (('test','base'),('test','stress'),('tune','base'),('tune','stress')):
        lookup={(r['label'],r['order']):r for r in groups if r['segment']==segment and r['cost']==cost}
        lines += ['',f'## {segment.upper()} / {cost}: чистый результат всего интервала, USDT','',
            '| Метод | Сначала убыток | Сначала прибыль | Разница | Сделок: убыток / прибыль | Неоднозначных: убыток / прибыль |',
            '|---|---:|---:|---:|---:|---:|']
        for label in labels:
            a,b=(lookup[label,o] for o in ORDERS)
            lines.append(f"| {label} | {a['net_pnl_usdt']:.6f} | {b['net_pnl_usdt']:.6f} | {b['net_pnl_usdt']-a['net_pnl_usdt']:.6f} | {a['trades_n']:.1f} / {b['trades_n']:.1f} | {a['ambiguous_executions_n']:.1f} / {b['ambiguous_executions_n']:.1f} |")
    lines += ['', '## Ограничения вывода','',
        'Полный интервал включает неполные календарные дни. Среднее полных дней не заменяет итог интервала.',
        'После срабатывания лимита и закрытия позиции последующие дни без сделок — бездействие, не доказательство ежедневной прибыли.',
        'TARGET_FIRST может противоречить уже известному минутному порядку в отдельных окнах; это проверка чувствительности15m модели.',
        'Исторические секундные свечи/сделки не восстановлены. Модели и пороги не переобучались/перенастраивались.',
        'Охват истории — текущие выжившие пары; исторический состав рынка и задержки публикации не сертифицированы.',
        'После блокировки входов подавленные намерения модели помечены SYSTEM_RISK_LOCK; их счётчики не сопоставимы с прежними.',
        '', '![Чистый результат TEST](plots/test-net.png)', '', '![Полная кривая TEST](plots/test-equity.png)']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plots=output/'plots'; plots.mkdir(exist_ok=True)
    lookup={(r['label'],r['order']):r for r in groups if r['segment']=='test' and r['cost']=='base'}
    fig,ax=plt.subplots(figsize=(13,5)); x=np.arange(len(labels))
    for offset,order in ((-.18,ORDERS[0]),(.18,ORDERS[1])):
        ax.bar(x+offset,[lookup[label,order]['net_pnl_usdt'] for label in labels],width=.36,label=order)
    ax.axhline(0,color='black',lw=.8); ax.set_xticks(x,labels,rotation=25,ha='right')
    ax.set(ylabel='Interval net PnL USDT / initial100',title='TEST base | counterfactual execution conventions; seed means')
    ax.legend(); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(plots/'test-net.png',dpi=140); plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(16,10),sharex=True)
    for ax,label in zip(axes.flat,labels):
        for order in ORDERS:
            selected=[r for r in rows if r['segment']=='test' and r['cost']=='base' and r['order']==order and
                (r['label']==label or label in ('MYOPIC','PPO','EXIT_LEARNED') and r['label'].startswith(label+'-'))]
            curves=[]
            for task in selected:
                detail=load_report(output,task,registration_sha)
                curve=detail['equity_curve']; curves.append(np.array([v['equity_after'] for v in curve]))
                times=[datetime.fromtimestamp(v['origin_ms']/1000,timezone.utc) for v in curve]
                del detail,curve
            array=np.stack(curves); ax.plot(times,array.mean(axis=0),label=order,lw=.9)
            if len(curves)>1: ax.fill_between(times,array.min(axis=0),array.max(axis=0),alpha=.1)
        ax.axhline(100,color='gray',lw=.6); ax.set_title(label); ax.set_ylabel('MTM USDT'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7); fig.autofmt_xdate(); fig.tight_layout()
    fig.suptitle('TEST base | full period | seed means; shaded bands=min/max across five seeds',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(plots/'test-equity.png',dpi=130); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True); parser.add_argument('--prepared',required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--truth-report',required=True)
    args=parser.parse_args(); app=Path(__file__).resolve().parents[2]
    if not Path(args.output).resolve().is_relative_to(app/'.runtime/execution-order-sensitivity'):
        raise SystemExit('OWN_SENSITIVITY_RUNTIME_REQUIRED')
    execute_study(args.parent,args.prepared,args.output,args.truth_report)


if __name__=='__main__': main()
