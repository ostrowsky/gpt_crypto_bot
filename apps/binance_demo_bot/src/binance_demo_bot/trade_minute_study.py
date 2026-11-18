"""MRP-2 complete frozen-policy native-minute portfolio recomputation.

Research-only CLI. Golden84 first;168 conditional reports, no training/promotion.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import traceback

from .execution_order_study import (make_plan,capture_inputs,verify_files,seal_registration,
    load_report,save_report,group_results,_summary,AFFECTED)
from .execution_order_sensitivity import run_scenario,make_signal_schedule,compare_reference,financial_check
from .intrabar_probe import _seal,_read_sealed,_write,_bytes_sha
from .trade_minute_core import CandidateTape,MemoPolicy,run_trade_minute
from .trade_minute_data import APP,TradeMinuteStore,_own
from .journal import _WriterLease


def minute_plan(parent,thresholds):
    result=make_plan(parent,thresholds)
    return [dict(t,order='MINUTE_'+t['order'],id='MINUTE_'+t['id']) for t in result]


def reuse_source(task):
    if task['order']=='MINUTE_TARGET_FIRST' and task['label'] not in AFFECTED:
        return task['id'].replace('MINUTE_TARGET_FIRST-','MINUTE_STOP_FIRST-',1)
    return None


def verify_mlp(handle):
    policy=handle.model.policy
    if type(policy).__name__ not in ('MlpPolicy','ActorCriticPolicy') or any(hasattr(policy,k) for k in ('lstm_actor','lstm_critic','recurrent_hidden_states')):
        raise ValueError('STATELESS_MLP_REQUIRED_FOR_EXACT_MEMO')


def complete_gate(golden_matches,rows,audit):
    if (golden_matches!=84 or len(rows)!=168 or any(r['state']!='COMPLETE' or r['financial_check']['state']!='PASS' for r in rows) or
            audit.get('state')!='VERIFIED_RECORDED_TRADE_MINUTE_DATA'):
        raise ValueError('FULL_MINUTE_STUDY_INCOMPLETE')
    return 'COMPLETE_CONDITIONAL_SIMULATION'



def verify_golden_reports(prior_folder,parent_folder):
    """Use exact old84 as prior evidence; no rewriting their metadata/registration."""
    prior=Path(prior_folder).resolve();parent=Path(parent_folder).resolve();reg=_read_sealed(prior/'registration.json')
    if reg.get('protocol')!='MRP-1-NATIVE-MINUTE-PORTFOLIO' or Path(reg['parent_folder']).resolve()!=parent:
        raise ValueError('GOLDEN_PRIOR_PARENT_IDENTITY')
    tasks=reg['original_golden_plan'];files={};items=[];sha=_bytes_sha(prior/'registration.json')
    if len(tasks)!=84 or len({t['id'] for t in tasks})!=84 or any(t['order']!='STOP_FIRST' for t in tasks):
        raise ValueError('GOLDEN_PRIOR_POPULATION')
    def pin(path,expected=None):
        path=Path(path).resolve();digest=_bytes_sha(path)
        if expected is not None and digest!=expected:raise ValueError('GOLDEN_PRIOR_INPUT_CHANGED:'+str(path))
        files[str(path)]=digest;return digest
    pin(prior/'registration.json')
    for name in ('rl_direction_core.py','rl_direction_runner.py','rl_direction_models.py','rl_direction_data.py',
                 'execution_order_sensitivity.py','minute_replay_core.py'):
        path=Path(__file__).with_name(name).resolve();expected=reg['files'].get(str(path))
        if expected is None:raise ValueError('GOLDEN_EXECUTABLE_NOT_PINNED:'+name)
        pin(path,expected)
    for task in tasks:
        report=load_report(prior/'golden',task,sha)
        if report is None:raise ValueError('GOLDEN_PRIOR_REPORT_MISSING')
        reference_path=parent/'portfolios'/f"{task['segment']}-{task['cost']}-{task['label']}.json"
        expected=reg['files'].get(str(reference_path))
        if expected is None:raise ValueError('GOLDEN_REFERENCE_NOT_PINNED')
        pin(reference_path,expected);reference=json.loads(reference_path.read_text(encoding='utf-8'))
        parity=compare_reference(reference,report);check=financial_check(report)
        if parity['state']!='MATCH' or check['state']!='PASS':raise ValueError('GOLDEN_PRIOR_FINANCIAL_PARITY:'+task['id'])
        path=prior/'golden/portfolios'/(task['id']+'.json.gz')
        for item in (path,path.with_suffix('.meta.json'),path.with_suffix('.meta.json.sha256')):pin(item)
        items.append(dict(id=task['id'],parity=parity,financial_check=check,prior_report_sha256=files[str(path)]))
    return dict(state='VERIFIED_PRIOR_GOLDEN84',portfolios_n=84,prior_registration_sha256=sha,
        evidence_kind='REUSED_PRIOR_EXECUTION_NOT_RERUN',items=items,files=files,promotion_allowed=False)


def collector_cli(operation,output,**options):
    command=[str(APP/'.venv/Scripts/python.exe'),'-m','binance_demo_bot.trade_minute_data',operation,'--folder',str(output)]
    for k,v in options.items():command+=['--'+k.replace('_','-'),str(v)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=180,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise ValueError('NATIVE_DATA_'+operation.upper()+':'+result.stdout[-1000:]+result.stderr[-1500:])
    return result.stdout


def _run(task,tape,policies,schedules,*,native=None):
    from .rl_direction_runner import buy_hold
    if task['kind']=='BTC_BUY_HOLD':
        report=buy_hold(tape,task['start'],task['end'],task['cost'])
        report.update(execution_scope='UNMANAGED_15M_VALUATION_BENCHMARK',native_execution_certified=False)
    else:
        policy=policies.get(('PPO' if task['kind']=='EXIT' else task['kind'],task['seed']))
        kwargs=dict(kind=task['kind'],threshold=task['threshold'],policy=policy,exit_mode=task['exit_mode'],cost=task['cost'],
                    signals=schedules[task['segment']] if task['kind'] in ('SIGNAL','EXIT') else None,intrabar_order=task['order'])
        report=run_trade_minute(tape,task['start'],task['end'],minute_lookup=native.get,**kwargs) if native else run_scenario(tape,task['start'],task['end'],**kwargs)
    if task['seed'] is not None:report['seed']=task['seed']
    return report


def execute(output,*,parent_folder,prepared_folder,ic_folder,truth_report,prior_folder):
    output=_own(output);output.mkdir(parents=True,exist_ok=True)
    lease=_WriterLease(output/'runner.writer.lock');raw_tape=None
    def status(phase,**extra):
        value=dict(phase=phase,pid=os.getpid(),at_utc=datetime.now(timezone.utc).isoformat(),promotion_allowed=False,**extra)
        _write(output/'status.json',value)
        with (output/'status-events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    try:
        status('VERIFYING_FROZEN_INPUTS')
        parent_folder=Path(parent_folder).resolve();prepared_folder=Path(prepared_folder).resolve()
        spec=APP.parents[1]/'docs/specs/binance-demo-trade-aware-replay.md'
        binding=capture_inputs(parent_folder,prepared_folder,spec,truth_report)
        import numpy as np
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        from . import rl_direction_models as models
        environment=models.environment_manifest();parent=json.loads((parent_folder/'registration.json').read_text(encoding='utf-8'))
        if environment['python']!=parent['environment']['python'] or environment['versions']!=parent['environment']['versions']:
            raise ValueError('FROZEN_RESEARCH_ENVIRONMENT_CHANGED')
        collector_cli('init',output,ic_folder=ic_folder,parent_folder=parent_folder,spec=spec,prior_folder=prior_folder)
        scope=_read_sealed(output/'data-scope.json')
        binding.update(protocol='MRP-2-TRADE-AWARE-EXECUTION',environment=environment,plan=minute_plan(parent,binding['thresholds']),
            original_golden_plan=[t for t in binding['plan'] if t['order']=='STOP_FIRST'],
            data_scope_sha256=_bytes_sha(output/'data-scope.json'),rounding_contract='ORIGINAL_FLOAT64_LEVELS_DECIMAL_DATA_CHECKS',
            uncertain_conventions=['MINUTE_STOP_FIRST','MINUTE_TARGET_FIRST'],native_execution_certified=False)
        golden=verify_golden_reports(prior_folder,parent_folder)
        binding['files'].update(golden['files'])
        binding['golden_evidence_kind']='REUSED_VERIFIED_MRP1_GOLDEN84'
        binding['golden_prior_registration_sha256']=golden['prior_registration_sha256']
        binding['files'].update(scope['files']);binding['files'][str(output/'data-scope.json')]=_bytes_sha(output/'data-scope.json')
        registration_sha=seal_registration(output,binding);_seal(output/'owner.json',dict(registration_sha256=registration_sha))
        _seal(output/'input-verification.json',dict(state='MATCH',files_n=len(binding['files']),symbols_N=519,raw_bars_n=binding['raw_bars_n']))
        manifest=json.loads((prepared_folder/'manifest.json').read_text(encoding='utf-8'))
        from .rl_direction_runner import MarketTape
        raw=np.load(parent_folder/'screen.npz');meta=json.loads((parent_folder/'screen.json').read_text(encoding='utf-8'))
        screen=dict(symbols=meta['symbols'],times_ms=raw['times_ms'],symbol_indices=raw['symbol_indices'],scores=raw['scores'])
        raw_tape=MarketTape(manifest,screen,parent_folder/'features')
        status('PREPARING_CAUSAL_CANDIDATE_CACHES')
        tapes={};schedules={}
        for segment,start,end in parent['splits'][1:]:
            tapes[segment]=CandidateTape(raw_tape,start,end)
            _seal(output/f'{segment}-candidate-cache.json',tapes[segment].identity())
            schedule=make_signal_schedule(tapes[segment],start,end,binding['thresholds']['GRU']);schedules[segment]=schedule
            _seal(output/f'{segment}-signals.json',{k:v for k,v in schedule.items() if k not in ('signals','candidates')})
        status('LOADING_FROZEN_STATELESS_POLICIES')
        policies={}
        for kind in ('MYOPIC','PPO'):
            for seed in parent['seeds']:
                handle=models.load_policy(parent_folder/'policies'/f'{kind}-{seed}');verify_mlp(handle)
                policies[kind,seed]=MemoPolicy(handle,handle.manifest['manifest_sha256'])
        _seal(output/'golden-prior-evidence.json',golden)
        native=TradeMinuteStore(output);rows=[]
        for task in binding['plan']:
            report=load_report(output,task,registration_sha)
            if report is None:
                status('NATIVE_MINUTE_PORTFOLIO',task=task['id'],done_n=len(rows),total_N=168,new_batches_n=native.fetches)
                source_id=reuse_source(task)
                if source_id:
                    source_task=next(t for t in binding['plan'] if t['id']==source_id)
                    report=load_report(output,source_task,registration_sha)
                    if report is None:raise ValueError('NATIVE_INVARIANT_SOURCE_MISSING')
                    report['intrabar_order']=task['order'];report['invariant_reuse']=dict(source_task=source_id,
                        source_report_sha256=_bytes_sha(output/'portfolios'/(source_id+'.json.gz')),
                        reason='NO_FIXED_TARGET_BRANCH_SAME_NATIVE_TRANSITIONS')
                else:report=_run(task,tapes[task['segment']],policies,schedules,native=native)
                check=financial_check(report)
                if check['state']!='PASS':
                    _seal(output/'failed-portfolios'/(task['id']+'.json'),dict(task=task,report=report,financial_check=check))
                    raise ValueError('NATIVE_FINANCIAL_CHECK:'+task['id']+':'+str(check)+':'+str(report.get('reasons')))
                report['mrp_financial_check']=check
                if task['kind'] in ('SIGNAL','EXIT'):
                    schedule=schedules[task['segment']]
                    if (report['signal_sha256'],report['entry_signals_n'])!=(schedule['sha256'],schedule['count']):raise ValueError('COMMON_SIGNAL_CHANGED')
                save_report(output,task,registration_sha,report)
            row=_summary(task,report,report['mrp_financial_check'],dict(state='NATIVE_RECOMPUTATION_NOT_GOLDEN_PARITY'))
            row['execution_frames_n']=len(report.get('execution_frames',[]));row['reused_from']=report.get('invariant_reuse',{}).get('source_task')
            row.update({k:report.get(k,0) for k in ('unfilled_intents_n','carry_exposure_steps_n','overdue_empty_steps_n')})
            rows.append(row);_write(output/'progress.json',rows);del report
        status('FINAL_NATIVE_AND_INPUT_AUDIT',portfolios_n=len(rows))
        collector_cli('audit',output)
        audit=_read_sealed(output/'native-data-audit.json');verification=verify_files(binding['files'])
        _seal(output/'final-input-verification.json',dict(state='MATCH',**verification))
        final=complete_gate(golden['portfolios_n'],rows,audit)
        render(output,binding,registration_sha,rows,audit)
        _seal(output/'memo-diagnostics.json',{f'{k}-{s}':dict(calls=p.calls,hits=p.hits,entries=p.entries_n) for (k,s),p in policies.items()})
        status(final,portfolios_n=len(rows),golden_matches_n=golden['portfolios_n'],native_frames_n=audit['frames_n'],
            native_attempts_n=audit['native_attempts_n'],full_truth_harness=binding['truth_harness_at_registration']['status'])
    except BaseException as error:
        status('INCOMPLETE_MINUTE_STUDY',error=type(error).__name__+':'+str(error),traceback=traceback.format_exc(),net_pnl_usdt=None)
        raise
    finally:
        if raw_tape is not None:raw_tape.close()
        lease.close()


def render(output,binding,registration_sha,rows,audit):
    groups=group_results(rows)
    report=dict(protocol=binding['protocol'],state='COMPLETE_CONDITIONAL_SIMULATION',groups=groups,portfolios=rows,
        golden_matches_n=84,golden_evidence_kind='REUSED_VERIFIED_MRP1_GOLDEN84',financial_checks_n=168,native_data_audit=audit,
        full_truth_harness=binding['truth_harness_at_registration']['status'],evidence_kind='SIMULATION',
        holdout='RETROSPECTIVE_REUSED_HOLDOUT',actual_demo_outcomes_n=0,promotion_allowed=False,
        native_execution_certified=False,mathematical_pnl_bounds=False,
        limitations=['CURRENT_SURVIVOR_COHORT','SAME_MINUTE_UNKNOWN_CONVENTIONS','15M_DECISION_AND_RISK_MARK_GRID',
            'UNMANAGED_BTC_15M_BENCHMARK','FIRST_TRADE_BUCKET_FILL_PROXY','CARRY_PRICE_MTM_NOT_EXECUTABLE_QUOTE','NO_NATIVE_TICK_FILL_RECONSTRUCTION','REUSED_TEST_NO_NEW_SIGNIFICANCE','NO_PRODUCTION_CHANGE'])
    _seal(output/'report.json',report)
    lines=['# MRP-2: полный портфельный пересчёт по минутам Demo','',
        f"84/84 прежних контрольных портфеля повторно верифицированы;168/168 финансовых проверок. Full Truth Harness: **{report['full_truth_harness']}**.",
        'Это условная симуляция на повторно использованном TEST, не фактическая прибыль Demo и не доказательство будущего дохода.',
        'Разные минуты задают порядок; два уровня внутри одной минуты оставлены двумя сценариями. Их итоги не являются математическими границами PnL.',
        'Исходный риск фиксируется по15m сетке; после потери/просадки1USDT входы прекращаются до конца интервала без ежедневного сброса.',
        '', '| Период | Издержки | Сценарий одной минуты | Метод | Net USDT | Среднее за полный день | Сделок | Неясных исполнений | Активных дней |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['segment']} | {g['cost']} | {g['order']} | {g['label']} | {g['net_pnl_usdt']:.6f} | {g['mean_daily_net_pnl_usdt']:.6f} | {g['trades_n']:.1f} | {g['ambiguous_executions_n']:.1f} | {g['active_days_n']:.1f} |")
    lines+=['','Пять seed: среднее пяти, диапазоны отдельных прогонов сохранены в report.json. Календарь включает дни без торговли; частичные дни отдельно.',
        f"Проверено {audit['frames_n']} минутных окон; новых GET-попыток {audit['native_attempts_n']}. Native1s gate остаётся незавершённым.",
        'Не включены реальные Demo сделки, переобучение, новые параметры или автоматическое продвижение моделей.']
    (output/'findings-ru.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict
    fig,axes=plt.subplots(2,2,figsize=(16,10),sharex=True)
    for i,cost in enumerate(('base','stress')):
        for j,order in enumerate(('MINUTE_STOP_FIRST','MINUTE_TARGET_FIRST')):
            ax=axes[i,j];families=defaultdict(list)
            for task in binding['plan']:
                if (task['segment'],task['cost'],task['order'])!=('test',cost,order):continue
                detail=load_report(output,task,registration_sha)
                family=task['label'].rsplit('-',1)[0] if task['seed'] is not None else task['label']
                marks={d:v['end_equity']-100 for d,v in detail['days'].items()}
                families[family].append(marks)
            for family,members in families.items():
                dates=sorted(members[0]);ys=[sum(m[d] for m in members)/len(members) for d in dates]
                ax.plot([datetime.fromisoformat(d) for d in dates],ys,label=family,linewidth=1.2)
            ax.set_title(f'TEST | {cost} | {order}');ax.set_ylabel('Interval cumulative MTM net, USDT');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=3)
    fig.suptitle('MRP-2 conditional native-minute simulation | same calendar,100 USDT initial | five-seed means | no promotion')
    fig.autofmt_xdate();fig.tight_layout();(output/'plots').mkdir(exist_ok=True);fig.savefig(output/'plots/test-equity.png',dpi=140);plt.close(fig)
    import html
    content='<meta charset="utf-8"><title>MRP-2</title><style>body{font:16px system-ui;margin:30px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:5px}img{max-width:100%}</style>'
    content+='<h1>MRP-2: native-minute conditional simulation</h1><p>Full Truth Harness: '+report['full_truth_harness']+'. Reused holdout. No actual Demo profit or promotion.</p><img src="plots/test-equity.png"><table>'
    content+='<tr>'+''.join('<th>'+x+'</th>' for x in ('segment','cost','order','method','net USDT','unresolved','active days'))+'</tr>'
    for g in groups:
        content+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (g['segment'],g['cost'],g['order'],g['label'],round(g['net_pnl_usdt'],6),g['ambiguous_executions_n'],g['active_days_n']))+'</tr>'
    (output/'comparison.html').write_text(content+'</table>',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',required=True);p.add_argument('--parent-folder',required=True)
    p.add_argument('--prior-folder',required=True);p.add_argument('--prepared-folder',required=True);p.add_argument('--ic-folder',required=True);p.add_argument('--truth-report',required=True);a=p.parse_args()
    execute(a.folder,parent_folder=a.parent_folder,prepared_folder=a.prepared_folder,ic_folder=a.ic_folder,truth_report=a.truth_report,prior_folder=a.prior_folder)

if __name__=='__main__':main()
