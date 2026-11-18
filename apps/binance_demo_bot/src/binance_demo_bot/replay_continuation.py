"""FMR-1: verified continuation without loading any neural policy or memo.

Prior COMPLETE portfolios remain explicit prior evidence. Economic code is
unchanged. A new finite read-only window inherits the old native GET usage.
"""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import traceback

from .execution_order_study import load_report,save_report,verify_files,_summary
from .execution_order_sensitivity import financial_check,make_signal_schedule
from .intrabar_probe import _read_sealed,_seal,_write,_bytes_sha
from .rl_direction_data import _file_sha
from .trade_minute_data import APP,_own,TradeMinuteStore,query_parameters
from .trade_minute_study import reuse_source,_run,render,complete_gate
from .minute_replay_core import CandidateTape
from .journal import _WriterLease


def continuation_scope(old,used,*,now_ms):
    if type(used) is not int or used<0 or used>=old['attempt_limit']:raise ValueError('CONTINUATION_BUDGET_EXHAUSTED')
    return dict(old,created_at_ms=now_ms,expires_at_ms=now_ms+7200000,
        attempt_limit=old['attempt_limit']-used,cumulative_attempt_limit=old['attempt_limit'],
        inherited_native_attempts_n=used,prior_imported_attempts_n=old.get('prior_imported_attempts_n',0)+used,
        prior_expiry_ms=old['expires_at_ms'],window_reason='NEW_READ_ONLY_CONTINUATION_AFTER_COMPUTATIONAL_MEMORY_FAILURE')


def remap_inputs(files,snapshots):
    mapped={}
    for name,expected in files.items():
        path=Path(name).resolve();actual=_file_sha(path)
        if actual==expected:mapped[str(path)]=expected;continue
        saved=snapshots.get(name)
        if not saved or path.suffix!='.md':raise ValueError('CONTINUATION_INPUT_CHANGED:'+name)
        snap=Path(saved['path']).resolve()
        if (_file_sha(snap)!=expected or saved['sha256']!=expected or path.read_bytes().rstrip()!=snap.read_bytes().rstrip()):
            raise ValueError('CONTINUATION_NOT_EOF_ONLY:'+name)
        mapped[str(snap)]=expected;mapped[str(path)]=actual
    return mapped


def remaining_computations(plan,completed):
    stop={t['id'] for t in plan if t['order']=='MINUTE_STOP_FIRST'}
    if len(plan)!=168 or len(stop)!=84 or not stop<=set(completed):raise ValueError('MISSING_STOP_FIRST_EVIDENCE')
    missing=[]
    for task in plan:
        if task['id'] in completed:continue
        source=reuse_source(task)
        if source:
            if source not in completed:raise ValueError('MISSING_NONFIXED_SOURCE')
        elif task['label'] not in ('CatBoost','GRU','EXIT_fixed'):raise ValueError('NEURAL_RECOMPUTATION_NOT_ALLOWED')
        else:missing.append(task)
    return missing


def request_guard_files(files,manifest_path):
    manifest_path=Path(manifest_path).resolve()
    _seal(manifest_path,dict(files=dict(files),verification='FULL_HASH_BEFORE_AND_AFTER_COMPUTATION'))
    guard={name:digest for name,digest in files.items() if Path(name).parent.name not in ('frames','raw-frames','responses')}
    guard[str(manifest_path)]=_bytes_sha(manifest_path);return guard


def make_binding(old,scope_path,**fields):
    value=dict(old);value.update(fields);value['data_scope_sha256']=_bytes_sha(scope_path);return value


def load_completed(folder,task,sha):
    report=load_report(folder,task,sha)
    if report is None:return None
    if report.get('state')!='COMPLETE' or financial_check(report)['state']!='PASS':raise ValueError('PRIOR_REPORT_INCOMPLETE:'+task['id'])
    return report


def prepare(folder,prior):
    """3.14 data-audit worker. Never load models, read keys or place orders."""
    from .public_research_audit import read_committed_records
    folder=_own(folder);prior=_own(prior)
    if (folder/'registration.json').exists():raise ValueError('CONTINUATION_ALREADY_REGISTERED')
    old=_read_sealed(prior/'registration.json');old_scope=_read_sealed(prior/'data-scope.json')
    proof=_read_sealed(prior/'native-data-audit.json')
    if old['protocol']!='MRP-2-TRADE-AWARE-EXECUTION' or proof['state']!='VERIFIED_RECORDED_TRADE_MINUTE_DATA':
        raise ValueError('VERIFIED_PRIOR_DATA_REQUIRED')
    _,begins,_=read_committed_records(prior/'rest-audit.sqlite');used=len(begins)
    if used!=proof['native_attempts_n']:raise ValueError('PRIOR_ATTEMPT_DENOMINATOR')
    snapshots=_read_sealed(prior/'frozen-source.json')
    if snapshots['registration_sha256']!=_bytes_sha(prior/'registration.json'):raise ValueError('PRIOR_SOURCE_ROOT')
    # The historical bytes remain in the immutable source bundle. Normalize only
    # the known trailing blank in the new documentation before pinning this run.
    doc=APP.parents[1]/'docs/specs/binance-demo-trade-aware-replay.md'
    raw=doc.read_bytes();normalized=raw.rstrip()+b'\n'
    if raw!=normalized:doc.write_bytes(normalized)
    binding_files=remap_inputs(old['files'],snapshots['files'])
    semantic=('trade_minute_core.py','trade_minute_data.py','minute_replay_core.py','rl_direction_core.py','rl_direction_runner.py','execution_order_sensitivity.py')
    for name in semantic:
        path=Path(__file__).with_name(name).resolve()
        if _bytes_sha(path)!=old['files'].get(str(path)):raise ValueError('ECONOMIC_SOURCE_CHANGED:'+name)
    # remap_inputs has already verified every full input hash.
    small=remap_inputs(old_scope['files'],snapshots['files'])
    pin_cache={}
    def pin(path):
        path=Path(path).resolve()
        if str(path) in pin_cache:return pin_cache[str(path)]
        digest=_bytes_sha(path);pin_cache[str(path)]=digest;small[str(path)]=digest;binding_files[str(path)]=digest;return digest
    for name in ('registration.json','data-scope.json','native-data-audit.json','frozen-source.json','rest-audit.sqlite','golden-prior-evidence.json','status.json'):
        pin(prior/name)
    for name in ('replay_continuation.py',):pin(Path(__file__).with_name(name))
    pin(APP.parents[1]/'docs/specs/binance-demo-replay-memory-continuation.md')
    responses={};raw_copied=frames_copied=0
    for kind in ('raw-frames','frames'):
        for path in sorted((prior/kind).glob('*.json')):
            value=_read_sealed(path);source=Path(value['source_ref']).resolve();source_sha=pin(source)
            if source_sha!=value['source_sha256']:raise ValueError('PRIOR_FRAME_RESPONSE_SHA')
            if str(source) not in responses:responses[str(source)]=_read_sealed(source)
            packet=responses[str(source)];at=value['origin_ms'];query_parameters(old_scope,value['symbol'],at)
            if (packet['parameters']['symbol']!=value['symbol'] or value['received_at_ms']!=packet['metadata']['received_at_ms'] or
                    value['rows']!=[r for r in packet['rows'] if at<=r[0]<at+900000]):raise ValueError('PRIOR_FRAME_EXACT_SLICE')
            pin(path)
            if kind=='frames':value=dict(value,imported=True,prior_frame_sha256=small[str(path)])
            _seal(folder/kind/path.name,value)
            if kind=='frames':frames_copied+=1
            else:raw_copied+=1
    del responses
    scope=continuation_scope(old_scope,used,now_ms=int(datetime.now(timezone.utc).timestamp()*1000))
    imports_path=folder/'imported-files.json';scope['files']=request_guard_files(small,imports_path)
    binding_files[str(imports_path)]=_bytes_sha(imports_path)
    scope['imported_frames_n']=frames_copied;scope['protocol']='FMR-1-CUMULATIVE-READ-ONLY-CONTINUATION'
    _seal(folder/'data-scope.json',scope);binding_files[str(folder/'data-scope.json')]=_bytes_sha(folder/'data-scope.json')
    old_sha=_bytes_sha(prior/'registration.json');completed=[]
    for task in old['plan']:
        report=load_completed(prior,task,old_sha)
        if report is None:continue
        path=prior/'portfolios'/(task['id']+'.json.gz')
        for item in (path,path.with_suffix('.meta.json'),path.with_suffix('.meta.json.sha256')):pin(item)
        completed.append(dict(id=task['id'],path=str(path),sha256=_bytes_sha(path),evidence_kind='REUSED_PRIOR_COMPLETE'))
        del report
    missing=remaining_computations(old['plan'],{r['id'] for r in completed})
    binding=make_binding(old,folder/'data-scope.json',protocol='FMR-1-MEMORY-BOUNDED-CONTINUATION',files=binding_files,
        prior_folder=str(prior),prior_registration_sha256=old_sha,prior_complete_reports=completed,
        recomputation_plan_ids=[t['id'] for t in missing],neural_policy_loads_n=0,
        continuation_window={k:scope[k] for k in ('created_at_ms','expires_at_ms','attempt_limit','inherited_native_attempts_n','cumulative_attempt_limit','prior_expiry_ms')})
    _seal(folder/'registration.json',binding);sha=_bytes_sha(folder/'registration.json');_seal(folder/'owner.json',dict(registration_sha256=sha))
    for item in completed:
        task=next(t for t in old['plan'] if t['id']==item['id']);source=Path(item['path']);target=folder/'portfolios'/source.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        _seal(target.with_suffix('.meta.json'),dict(id=task['id'],registration_sha256=sha,sha256=item['sha256'],format='gzip-json-v1',
            evidence_kind='REUSED_PRIOR_COMPLETE',prior_registration_sha256=old_sha,prior_report_path=str(source)))
    _seal(folder/'golden-prior-evidence.json',_read_sealed(prior/'golden-prior-evidence.json'))
    return dict(state='PREPARED_CONTINUATION',prior_complete_n=len(completed),remaining_computations_n=len(missing),
        inherited_attempts_n=used,remaining_attempt_limit=scope['attempt_limit'],frames_copied_n=frames_copied,raw_copied_n=raw_copied)


def execute(folder,prior):
    folder=_own(folder);folder.mkdir(parents=True,exist_ok=True);tape=None;lease=_WriterLease(folder/'continuation.writer.lock')
    def status(phase,**extra):
        value=dict(phase=phase,pid=os.getpid(),at_utc=datetime.now(timezone.utc).isoformat(),promotion_allowed=False,**extra)
        _write(folder/'status.json',value);print(json.dumps(value),flush=True)
    try:
        status('PREPARING_VERIFIED_CONTINUATION')
        result=subprocess.run([str(APP/'.venv/Scripts/python.exe'),'-m','binance_demo_bot.replay_continuation','prepare','--folder',str(folder),'--prior',str(prior)],
            capture_output=True,text=True,timeout=1800,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:raise ValueError('PREPARE_FAILED:'+result.stdout[-1000:]+result.stderr[-1500:])
        binding=_read_sealed(folder/'registration.json');sha=_bytes_sha(folder/'registration.json')
        import numpy as np
        if platform.python_version()!='3.11.9' or np.__version__!='2.4.4' or not Path(np.__file__).resolve().is_relative_to(Path(os.sys.executable).resolve().parent):
            raise ValueError('PINNED_NUMPY_RUNTIME_REQUIRED')
        from .rl_direction_runner import MarketTape
        parent=Path(binding['parent_folder']);manifest=json.loads((Path(binding['prepared_folder'])/'manifest.json').read_text())
        raw=np.load(parent/'screen.npz');meta=json.loads((parent/'screen.json').read_text())
        tape=MarketTape(manifest,dict(symbols=meta['symbols'],times_ms=raw['times_ms'],symbol_indices=raw['symbol_indices'],scores=raw['scores']),parent/'features')
        tapes={};schedules={}
        for segment,start,end in binding['splits'][1:]:
            tapes[segment]=CandidateTape(tape,start,end);identity=tapes[segment].identity()
            if identity!=_read_sealed(Path(prior)/f'{segment}-candidate-cache.json'):raise ValueError('FROZEN_CANDIDATE_CACHE_CHANGED')
            _seal(folder/f'{segment}-candidate-cache.json',identity)
            schedules[segment]=make_signal_schedule(tapes[segment],start,end,binding['thresholds']['GRU'])
        native=TradeMinuteStore(folder);rows=[];new_count=invariant_count=0;prior_ids={v['id'] for v in binding['prior_complete_reports']}
        for task in binding['plan']:
            report=load_completed(folder,task,sha)
            if report is None:
                source_id=reuse_source(task)
                if source_id:
                    source_task=next(t for t in binding['plan'] if t['id']==source_id);report=load_completed(folder,source_task,sha)
                    if report is None:raise ValueError('MISSING_INVARIANT_SOURCE')
                    report['intrabar_order']=task['order'];report['invariant_reuse']=dict(source_task=source_id,source_report_sha256=_bytes_sha(folder/'portfolios'/(source_id+'.json.gz')))
                    invariant_count+=1
                else:
                    if task['id'] not in binding['recomputation_plan_ids']:raise ValueError('UNREGISTERED_RECOMPUTATION')
                    status('RECOMPUTING_FIXED_TARGET_PORTFOLIO',done_n=len(rows),total_N=168,task=task['id'],neural_policy_loads_n=0)
                    report=_run(task,tapes[task['segment']],{},schedules,native=native);new_count+=1
                check=financial_check(report)
                if check['state']!='PASS':
                    _seal(folder/'failed-portfolios'/(task['id']+'.json'),dict(task=task,report=report,check=check));raise ValueError('INCOMPLETE_OBJECTIVE:'+str(report.get('reasons')))
                report['mrp_financial_check']=check;save_report(folder,task,sha,report)
            check=financial_check(report);row=_summary(task,report,check,dict(state='REUSED_PRIOR' if task['id'] in prior_ids else 'CONTINUATION_RESULT'))
            row.update(execution_frames_n=len(report.get('execution_frames',[])),reused_from=report.get('invariant_reuse',{}).get('source_task'),
                continuation_evidence_kind='REUSED_PRIOR_COMPLETE' if task['id'] in prior_ids else 'INVARIANT_REUSE' if reuse_source(task) else 'RECOMPUTED')
            row.update({k:report.get(k,0) for k in ('unfilled_intents_n','carry_exposure_steps_n','overdue_empty_steps_n')})
            rows.append(row);del report
        status('FINAL_DATA_AND_INPUT_AUDIT',portfolios_n=168)
        result=subprocess.run([str(APP/'.venv/Scripts/python.exe'),'-m','binance_demo_bot.trade_minute_data','audit','--folder',str(folder)],
            capture_output=True,text=True,timeout=1800,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:raise ValueError('FINAL_DATA_AUDIT_FAILED:'+result.stderr[-1500:])
        proof=_read_sealed(folder/'native-data-audit.json');verification=verify_files(binding['files'])
        _seal(folder/'final-input-verification.json',dict(state='MATCH',**verification))
        phase=complete_gate(84,rows,proof);render(folder,binding,sha,rows,proof)
        _seal(folder/'continuation-evidence.json',dict(prior_complete_n=len(prior_ids),recomputed_n=new_count,invariant_reuse_n=invariant_count,
            total_N=168,neural_policy_loads_n=0,window=binding['continuation_window'],actual_demo_outcomes_n=0,promotion_allowed=False))
        status(phase,portfolios_n=168,prior_complete_n=len(prior_ids),recomputed_n=new_count,invariant_reuse_n=invariant_count,neural_policy_loads_n=0)
    except BaseException as error:
        status('INCOMPLETE_CONTINUATION',error=type(error).__name__+':'+str(error),traceback=traceback.format_exc(),net_pnl_usdt=None);raise
    finally:
        if tape is not None:tape.close()
        lease.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['prepare','run']);p.add_argument('--folder',required=True);p.add_argument('--prior',required=True)
    a=p.parse_args()
    if a.operation=='prepare':print(json.dumps(prepare(a.folder,a.prior)),flush=True)
    else:execute(a.folder,a.prior)

if __name__=='__main__':main()
