"""RDE-2A bounded credential-free native candle compatibility experiment.

No trading, training, performance estimation or fallback data source. This
source gate must succeed before an intrabar portfolio replay can be trusted.
"""
import argparse
from contextlib import closing
from functools import wraps
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import zlib

from .adapter import AdapterError, DemoReadClient, DemoHttpTransport, _sha
from .intrabar_data import original_bar, validate_window, validate_seconds
from .journal import _WriterLease


STEP=900000
ENDPOINT='https://demo-api.binance.com/api/v3/klines'
AMBIGUOUS=(('TSTUSDT',1749763800000),('NXPCUSDT',1749808800000),
           ('NXPCUSDT',1749809700000),('PROMUSDT',1750638600000),
           ('DEXEUSDT',1750797000000),('GMXUSDT',1752068700000))


class PacedReadTransport:
    """Same existing Demo TLS engine, stricter public scope and native deadline."""
    def __init__(self,*,transport=None,deadline,monotonic=time.monotonic,sleep=time.sleep):
        self.transport=transport or DemoHttpTransport()
        self.deadline=deadline; self.monotonic=monotonic; self.sleep=sleep; self.last=None

    def request(self,method,path,query,headers,timeout_s):
        if method!='GET' or path!='/api/v3/klines' or headers:
            raise AdapterError('FORBIDDEN_ENDPOINT')
        if self.last is not None: self.sleep(max(0,.3-(self.monotonic()-self.last)))
        remaining=self.deadline-self.monotonic()
        if remaining<=0: raise AdapterError('READ_UNAVAILABLE')
        self.last=self.monotonic()
        return self.transport.request(method,path,query,headers,min(timeout_s,remaining))

    def close(self): self.transport.close()


def _single_writer(fn):
    @wraps(fn)
    def guarded(folder,*args,**kwargs):
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
        lease=_WriterLease(folder/'probe.writer.lock')
        try: return fn(folder,*args,**kwargs)
        finally: lease.close()
    return guarded


def _bytes_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
    tmp=path.with_suffix(path.suffix+'.next'); tmp.write_bytes(raw); tmp.replace(path)


def _seal(path,value):
    path=Path(path)
    if path.exists():
        old=_read_sealed(path)
        if old!=value: raise ValueError('INTRABAR_BINDING_CONFLICT')
        return
    _write(path,value)
    path.with_suffix(path.suffix+'.sha256').write_text(_bytes_sha(path),encoding='ascii')


def _read_sealed(path):
    path=Path(path)
    try:
        if _bytes_sha(path)!=path.with_suffix(path.suffix+'.sha256').read_text(encoding='ascii'):
            raise ValueError('INTRABAR_INTEGRITY')
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError): raise ValueError('INTRABAR_INTEGRITY') from None


def _anchor(archive,symbol,lower,upper,reverse=False):
    """Find a regular original candle; verify the selecting page before use."""
    with closing(sqlite3.connect(Path(archive).resolve().as_uri()+'?mode=ro',uri=True)) as con:
        order='DESC' if reverse else 'ASC'
        cursor=con.execute('SELECT source_sha,blob_sha,body FROM pages WHERE symbol=? '
            'AND last_ms>=? AND first_ms<? ORDER BY start_ms '+order,(symbol,lower,upper))
        for source_sha,blob_sha,blob in cursor:
            if hashlib.sha256(blob).hexdigest()!=blob_sha: raise ValueError('INTRABAR_ARCHIVE_INTEGRITY')
            try: rows=json.loads(zlib.decompress(blob))
            except (ValueError,zlib.error): raise ValueError('INTRABAR_ARCHIVE_INTEGRITY') from None
            if _sha(rows)!=source_sha: raise ValueError('INTRABAR_ARCHIVE_INTEGRITY')
            for row in reversed(rows) if reverse else rows:
                if (len(row)==12 and type(row[0]) is int and lower<=row[0]<upper
                        and row[0]%STEP==0 and row[6]==row[0]+STEP-1):
                    return row[0]
    raise ValueError('INTRABAR_ANCHOR_MISSING')


def _parent_binding(archive,rde_folder,prepared_path):
    parent=json.loads((rde_folder/'registration.json').read_text(encoding='utf-8'))
    prepared=json.loads(prepared_path.read_text(encoding='utf-8'))
    encoder=json.loads((rde_folder/'frozen-encoder.json').read_text(encoding='utf-8'))
    unsigned={k:v for k,v in prepared.items() if k!='data_sha256'}
    if (parent.get('attempt')!='RDE-1-MAX519-20261029' or parent.get('promotion_allowed') is not False or
        Path(prepared['archive']).resolve()!=archive or _sha(unsigned)!=prepared['data_sha256'] or
        prepared['data_sha256']!=parent['data_sha256'] or prepared['data_sha256']!=encoder['data_sha256'] or
        parent['symbols']!=prepared['symbols'] or
        _bytes_sha(rde_folder/'registration.json')!=encoder['registration_sha256']):
        raise ValueError('INTRABAR_PARENT_BINDING')
    with closing(sqlite3.connect(archive.as_uri()+'?mode=ro',uri=True)) as con:
        for symbol in parent['symbols']:
            vectors=[list(row) for row in con.execute('SELECT start_ms,rows_n,first_ms,last_ms,source_sha,blob_sha '
                'FROM pages WHERE symbol=? ORDER BY start_ms',(symbol,))]
            if _sha(vectors)!=prepared['entries'][symbol]['source_pages_sha256']:
                raise ValueError('INTRABAR_SOURCE_PAGES_BINDING')
    return parent


@_single_writer
def register_probe_run(folder,*,archive,rde_folder,spec_path,truth_report=None):
    folder,archive,rde_folder,spec_path=map(lambda p:Path(p).resolve(),(folder,archive,rde_folder,spec_path))
    prepared=rde_folder.parent/'prepared/manifest.json'
    parent=_parent_binding(archive,rde_folder,prepared)
    truth=dict(status='UNKNOWN',path=None,sha256=None)
    if truth_report is not None:
        path=Path(truth_report).resolve(); value=json.loads(path.read_text(encoding='utf-8'))
        if value.get('profile')!='full' or value.get('status') not in ('pass','fail','unknown'):
            raise ValueError('INTRABAR_TRUTH_BINDING')
        truth=dict(status=value['status'].upper(),path=str(path),sha256=_bytes_sha(path))
    start=parent['splits'][0][1]; test=parent['splits'][2][1]; end=parent['splits'][2][2]
    windows=set(AMBIGUOUS)
    for symbol in ('BTCUSDT','ETHUSDT','SOLUSDT'):
        windows.update((symbol,at) for at in (_anchor(archive,symbol,start,end),
            _anchor(archive,symbol,test,end),_anchor(archive,symbol,start,end,True)))
    if not {s for s,_ in windows}<=set(parent['symbols']): raise ValueError('INTRABAR_COHORT_BINDING')
    original=[]
    for symbol,at in sorted(windows):
        row=original_bar(archive,symbol,at)
        original.append(dict(symbol=symbol,origin_ms=at,**row,
                             seconds_required=(symbol,at) in AMBIGUOUS))
    binding=dict(protocol='RDE-2A-native-candle-compatibility-v1',endpoint=ENDPOINT,
        archive=str(archive),rde_folder=str(rde_folder),spec_path=str(spec_path),
        parent_registration_sha256=_bytes_sha(rde_folder/'registration.json'),
        prepared_manifest_path=str(prepared),prepared_manifest_sha256=_bytes_sha(prepared),
        frozen_encoder_sha256=_bytes_sha(rde_folder/'frozen-encoder.json'),
        parent_data_sha256=parent['data_sha256'],spec_sha256=_bytes_sha(spec_path),
        source_sha256={name:_bytes_sha(Path(__file__).with_name(name))
                      for name in ('intrabar_probe.py','intrabar_data.py','intrabar_audit.py')
                      if Path(__file__).with_name(name).exists()},
        cohort=parent['symbols'],splits=parent['splits'],windows=original,truth_harness_at_registration=truth,
        evidence_kind='SIMULATION',holdout_kind='RETROSPECTIVE_REUSED_HOLDOUT',
        rounding_contract='ASSUMED_NEAREST_ROUNDING_CAPPED_ORIGINAL_QUANTUM_1e-8',
        promotion_allowed=False,full_replay_complete=False)
    _seal(folder/'registration.json',binding)
    return binding


def _validate_binding(folder):
    registration=_read_sealed(Path(folder)/'registration.json')
    if (registration['endpoint']!=ENDPOINT or
        _bytes_sha(Path(registration['rde_folder'])/'registration.json')!=registration['parent_registration_sha256'] or
        _bytes_sha(registration['prepared_manifest_path'])!=registration['prepared_manifest_sha256'] or
        _bytes_sha(Path(registration['rde_folder'])/'frozen-encoder.json')!=registration['frozen_encoder_sha256'] or
        _bytes_sha(registration['spec_path'])!=registration['spec_sha256'] or
        any(_bytes_sha(Path(__file__).with_name(k))!=v for k,v in registration['source_sha256'].items())):
        raise ValueError('INTRABAR_BINDING_CHANGED')
    truth=registration['truth_harness_at_registration']
    if truth['path'] is not None and _bytes_sha(truth['path'])!=truth['sha256']:
        raise ValueError('INTRABAR_TRUTH_BINDING_CHANGED')
    _parent_binding(Path(registration['archive']),Path(registration['rde_folder']),Path(registration['prepared_manifest_path']))
    for window in registration['windows']:
        current=original_bar(registration['archive'],window['symbol'],window['origin_ms'])
        if any(current.get(k)!=window.get(k) for k in current): raise ValueError('INTRABAR_ORIGINAL_BINDING_CHANGED')
    return registration


@_single_writer
def collect_probe_run(folder,*,client,max_attempts=2000,max_wall_s=3600,
                      include_seconds=True,sleep=time.sleep,monotonic=time.monotonic):
    """A client call reserves all three possible native read attempts.

    Successful payloads are sealed and never refetched. Unavailable reads are
    append-only diagnostics, not empty-data responses or data absence claims.
    """
    if (type(max_attempts) is not int or not 3<=max_attempts<=2000 or
        type(max_wall_s) not in (int,float) or not 0<max_wall_s<=3600 or
        type(include_seconds) is not bool): raise ValueError('INTRABAR_INVALID_BUDGET')
    folder=Path(folder); registered=_validate_binding(folder)
    began=monotonic(); last_call=None; reserved=0; results=[]; minutes={}; fetched=cached=0
    attempt_folder=folder/'errors'; attempt_folder.mkdir(parents=True,exist_ok=True)

    def read(window,interval):
        nonlocal reserved,last_call,fetched,cached
        symbol,at=window['symbol'],window['origin_ms']; size=60000 if interval=='1m' else 1000
        params=dict(symbol=symbol,interval=interval,timeZone='0',startTime=at,endTime=at+STEP-1,limit=STEP//size)
        key=_sha(params); target=folder/'responses'/(key+'.json')
        base=dict(symbol=symbol,origin_ms=at,interval=interval,request_sha256=key)
        if target.exists():
            response=_read_sealed(target); cached+=1
            if response['parameters']!=params or response['endpoint']!=ENDPOINT or _sha(response['rows'])!=response['source_sha256']:
                raise ValueError('INTRABAR_RESPONSE_INTEGRITY')
        else:
            if reserved+3>max_attempts or monotonic()-began>=max_wall_s:
                results.append(dict(base,state='PENDING',reasons=['INVOCATION_BUDGET'])); return None
            if last_call is not None: sleep(max(0,.3-(monotonic()-last_call)))
            if monotonic()-began>=max_wall_s:
                results.append(dict(base,state='PENDING',reasons=['INVOCATION_BUDGET'])); return None
            last_call=monotonic(); reserved+=3
            try:
                rows,meta=client.request_public('/api/v3/klines',params,weight=2)
            except AdapterError as error:
                record=dict(base,state='UNKNOWN',reasons=[error.reason],endpoint=ENDPOINT,
                            parameters=params,observed_at_ms=time.time_ns()//1000000)
                _write(attempt_folder/(key+'-'+str(time.time_ns())+'.json'),record)
                results.append(record); return None
            response=dict(endpoint=ENDPOINT,parameters=params,rows=rows,metadata=meta,source_sha256=_sha(rows))
            _seal(target,response); fetched+=1
        received=response['metadata'].get('received_at_ms')
        validated=(validate_window(window['row'],response['rows'],interval_ms=size,received_at_ms=received)
            if interval=='1m' else validate_seconds(window['row'],minutes[(symbol,at)],response['rows'],received_at_ms=received))
        results.append(dict(base,**validated,response_file=str(target),source_sha256=response['source_sha256']))
        return response['rows'] if validated['state']=='MATCH' else None

    for window in registered['windows']:
        rows=read(window,'1m')
        if rows is not None: minutes[(window['symbol'],window['origin_ms'])]=rows
    seconds=0
    if len(minutes)==len(registered['windows']) and include_seconds:
        for window in registered['windows']:
            if window['seconds_required']: seconds+=1; read(window,'1s')
    count=lambda state:sum(r['state']==state for r in results)
    mismatch,incomplete,unknown,pending=map(count,('MISMATCH','INCOMPLETE','UNKNOWN','PENDING'))
    state=('SOURCE_INCOMPATIBLE' if mismatch else 'INCOMPLETE' if incomplete else
           'UNKNOWN' if unknown or pending else 'PASS_GATE_ONLY' if include_seconds else 'MINUTES_ONLY')
    report=dict(state=state,endpoint=ENDPOINT,registered_windows_n=len(registered['windows']),
        requested_n=len(results),match_n=count('MATCH'),mismatch_n=mismatch,
        incomplete_n=incomplete,unavailable_n=unknown,pending_n=pending,
        fetched_n=fetched,cached_n=cached,seconds_requested_n=seconds,
        reserved_transport_attempts_n=reserved,wall_seconds=monotonic()-began,
        evidence_kind='SIMULATION',full_replay_complete=False,promotion_allowed=False,
        truth_harness_at_registration=registered['truth_harness_at_registration'],
        native_transport_counters=dict(getattr(client,'_transport_counts',{})),
        native_audit_certification='UNKNOWN_NOT_YET_VERIFIED',
        rounding_contract=registered['rounding_contract'],
        scope='PREREQUISITE_WINDOWS_ONLY_NOT_PORTFOLIO_PERFORMANCE',results=results)
    _write(folder/'source-check.json',report)
    lines=['# RDE-2A: native execution data prerequisite','',
        f"State: **{state}**. MATCH {report['match_n']}/{len(results)}; mismatched {mismatch}; incomplete {incomplete}; unavailable {unknown}; pending {pending}.",
        '', 'SIMULATION. Source compatibility gate only; full frozen portfolio replay has not run.',
        f"No PnL improvement or trading readiness is established. Truth Harness at registration: {registered['truth_harness_at_registration']['status']}.",
        '', '| Symbol | UTC origin ms | Resolution | State | Reasons |','|---|---:|---|---|---|']
    lines.extend(f"| {r['symbol']} | {r['origin_ms']} | {r['interval']} | {r['state']} | {', '.join(r.get('reasons',[]))} |" for r in results)
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


@_single_writer
def finalize_probe_run(folder,audit_path,*,verifier=None):
    """Separate native evidence provenance from numerical candle compatibility."""
    _validate_binding(folder)
    if verifier is None:
        from .intrabar_audit import verify_probe_audit
        verifier=verify_probe_audit
    result=verifier(Path(folder),Path(audit_path))
    _write(Path(folder)/'audit-check.json',result)
    report=json.loads((Path(folder)/'source-check.json').read_text(encoding='utf-8'))
    report.update(native_audit_certification=result['state'],audit_path=str(Path(audit_path).resolve()),
        audit_check_sha256=_sha(result),overall_state=report['state'] if
        result['state']=='VERIFIED_RECORDED_PUBLIC_ATTEMPTS' else 'UNKNOWN')
    _write(Path(folder)/'source-check.json',report)
    with (Path(folder)/'report.md').open('a',encoding='utf-8') as out:
        out.write(f"\nNative recorded-attempt provenance: **{result['state']}**. Overall prerequisite: **{report['overall_state']}**.\n")
        out.write('This verifies only recorded public data requests, not historical exchange fills or profitability.\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True); parser.add_argument('--archive',required=True)
    parser.add_argument('--rde-folder',required=True); parser.add_argument('--register',action='store_true')
    parser.add_argument('--collect',action='store_true'); parser.add_argument('--max-attempts',type=int,default=2000)
    parser.add_argument('--truth-report')
    args=parser.parse_args()
    app=Path(__file__).resolve().parents[2]; folder=Path(args.folder).resolve()
    if not folder.is_relative_to(app/'.runtime'/'intrabar-execution'):
        raise SystemExit('OWN_INTRABAR_RUNTIME_REQUIRED')
    if args.register:
        register_probe_run(folder,archive=args.archive,rde_folder=args.rde_folder,
                           spec_path=app.parents[1]/'docs/specs/binance-demo-intrabar-execution.md',truth_report=args.truth_report)
    if args.collect:
        from .interaction_audit import AuditJournal
        from .isolation import AppPaths
        audit=AuditJournal(AppPaths(app),account_alias='public-research',episode_id=folder.name,
            scope_id='native-intrabar-source',database=str((folder/'rest-audit.sqlite').relative_to(app/'.runtime')),
            max_bytes=256*1024*1024)
        client=DemoReadClient(audit=audit,transport=PacedReadTransport(deadline=time.monotonic()+3600))
        try:
            result=collect_probe_run(folder,client=client,max_attempts=args.max_attempts)
        finally: client.close(); audit.close()
        result=finalize_probe_run(folder,folder/'rest-audit.sqlite')
        print(json.dumps({k:v for k,v in result.items() if k!='results'},ensure_ascii=False,indent=2))
    elif not args.register: raise SystemExit('REGISTER_OR_COLLECT_REQUIRED')


if __name__=='__main__': main()
