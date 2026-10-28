"""Credential-free maximum-archive researcher. Never constructs an order sender."""
import argparse
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import zlib

from .adapter import _sha,AdapterError
from .isolation import AppPaths
from .journal import _WriterLease,WriterBusy
from .trend_research import POLICY,STEP,portfolio_replay,promotion_readiness,time_splits
from .usdt_history import read_ready,verify_reference


def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.next')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,separators=(',',':'),allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _rows(paths,symbol):
    """One compressed1000-row page per instrument, no unbounded full load."""
    path=paths.runtime_path('all-usdt/history.sqlite');cursor=-1
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5)) as con:
        while True:
            row=con.execute('SELECT start_ms,source_sha,blob_sha,body FROM pages WHERE symbol=? AND start_ms>? ORDER BY start_ms LIMIT 1',(symbol,cursor)).fetchone()
            if row is None:return
            cursor,digest,blob_digest,blob=row
            if hashlib.sha256(blob).hexdigest()!=blob_digest:raise AdapterError('USDT_HISTORY_INTEGRITY')
            try:page=json.loads(zlib.decompress(blob))
            except (ValueError,zlib.error):raise AdapterError('USDT_HISTORY_INTEGRITY') from None
            if _sha(page)!=digest:raise AdapterError('USDT_HISTORY_INTEGRITY')
            for bar in page:
                # Raw shortened candles stay in the archive; their missing
                # regular slot breaks feature/held-price continuity in replay.
                if bar[0]%STEP==0 and bar[6]==bar[0]+STEP-1:yield bar


def run_research(paths,*,allow_partial=False):
    folder=paths.runtime_path('trend-research');folder.mkdir(parents=True,exist_ok=True)
    try:
        with closing(_WriterLease(folder/'research.writer.lock')):
            return _run(paths,folder,allow_partial=allow_partial)
    except WriterBusy:raise AdapterError('RESEARCH_BUSY') from None


def _run(paths,folder,*,allow_partial):
    cohort=json.loads(paths.runtime_path('all-usdt/history-registration.json').read_text(encoding='utf-8'))
    symbols=cohort['symbols'];cutoff=cohort['cutoff_ms']
    if (not symbols or symbols!=sorted(set(symbols)) or _sha(symbols)!=cohort['cohort_sha256']
        or type(cutoff) is not int or cutoff%STEP):raise AdapterError('RESEARCH_COHORT_INVALID')
    source={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('trend_research.py','trend_runner.py','usdt_history.py')}
    registered=dict(schema_version=1,objective='daily_net_equity_pnl_v1',policy=deepcopy(POLICY),
        cohort_sha256=cohort['cohort_sha256'],symbols=symbols,cutoff_ms=cutoff,source_sha256=source,
        split_definition='global-calendar-duration-70-15-15',evidence_kind='SIMULATION',promotion_allowed=False)
    registration=folder/'registration.json'
    if registration.exists():
        if json.loads(registration.read_text(encoding='utf-8'))!=registered:raise AdapterError('RESEARCH_REGISTRATION_CHANGED')
    else:_write(registration,registered)
    ready=sorted(set(read_ready(paths))&set(symbols));complete=ready==symbols
    status=dict(state='WAITING_FOR_HISTORY',ready_n=len(ready),symbols_N=len(symbols),cohort_complete=complete,
        checked_at_ms=time.time_ns()//1000000,promotion_allowed=False,training_state='NOT_TRAINED',
        registration_sha256=_sha(registered))
    if not ready or not complete and not allow_partial:
        _write(folder/'status.json',status);return status
    target=folder/('full' if complete else 'partial-'+_sha(ready)[:16]);target.mkdir(exist_ok=True)
    manifests=[]
    _write(folder/'status.json',dict(status,state='VERIFYING_ARCHIVE'))
    for symbol in ready:
        manifest=verify_reference(paths,symbol)
        if manifest['cutoff_ms']!=cutoff:raise AdapterError('RESEARCH_ARCHIVE_CUTOFF_MISMATCH')
        manifests.append(manifest)
    data_sha=_sha(manifests);report_path=target/'report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text(encoding='utf-8'));digest=report.pop('report_sha256',None)
        if (_sha(report)!=digest or report.get('data_sha256')!=data_sha
            or report.get('registration_sha256')!=_sha(registered)):raise AdapterError('RESEARCH_RESULT_INTEGRITY')
        report['report_sha256']=digest;_write(folder/'status.json',report);return report
    first=min(m['first_open_ms'] for m in manifests)
    splits=time_splits(first,cutoff);results={}
    _write(target/'manifest.json',dict(registration=registered,manifests=manifests,data_sha256=data_sha,splits=splits))
    for cost,fee,slip in (('base',10,5),('stress',20,10)):
        for name,start,end in splits:
            key=name+'_'+cost
            _write(folder/'status.json',dict(status,state='EVALUATING',segment=key,evaluated_n=len(ready),data_sha256=data_sha))
            result=portfolio_replay({s:_rows(paths,s) for s in ready},start_ms=start,end_ms=end,fee_bps=fee,slippage_bps=slip)
            # Detailed trade evidence remains in the separate immutable segment
            # artifact; compact status does not repeatedly copy every fill.
            _write(target/(key+'.json'),dict(result,registration_sha256=_sha(registered),data_sha256=data_sha))
            results[key]={k:v for k,v in result.items() if k not in ('trades','days')}
            results[key]['observed_days_n']=len(result['days'])
            results[key]['complete_day_count_certified']=False
    readiness=promotion_readiness(results['test_base'],cohort_complete=complete)
    report=dict(status,state='RESEARCH_COMPLETE' if complete else 'PROVISIONAL_RESEARCH_COMPLETE',
        evaluated_n=len(ready),data_sha256=data_sha,splits=splits,results=results,readiness=readiness,
        limitations=['CURRENT_SURVIVOR_COHORT','HISTORICAL_RECEIPT_UNAVAILABLE','HISTORICAL_FILTERS_UNVERIFIED',
            'OHLC_ADVERSE_PATH_SIMULATION','CALENDAR_FINANCIAL_CERTIFICATION_PENDING','NO_PARAMETER_FIT_OR_MODEL_TRAINING'])
    report['report_sha256']=_sha(report);_write(report_path,report);_write(folder/'status.json',report);return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-history',action='store_true')
    parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    if args.wait_for_history and args.allow_partial:raise SystemExit('PARTIAL_RUN_MUST_BE_EXPLICIT_ONCE')
    paths=AppPaths(Path(__file__).resolve().parents[2])
    while True:
        try:report=run_research(paths,allow_partial=args.allow_partial)
        except Exception as error:
            _write(paths.runtime_path('trend-research/incident.json'),dict(state='FAILED',
                reason=error.reason if isinstance(error,AdapterError) else type(error).__name__,
                checked_at_ms=time.time_ns()//1000000,promotion_allowed=False))
            raise
        if report['state']!='WAITING_FOR_HISTORY' or not args.wait_for_history:break
        if paths.runtime_path('trend-research/stop').exists():break
        time.sleep(60)


if __name__=='__main__':main()
