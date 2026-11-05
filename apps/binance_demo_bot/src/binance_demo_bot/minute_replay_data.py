"""MRP-1 bounded public native-minute store. No account or order capability.

The3.11 model runtime imports only stdlib helpers here. APSW-backed public
journaling is imported lazily by the separate app-owned3.14 collector CLI.
"""
import argparse
import base64
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time

from .intrabar_data import canonical_sha256, original_bar, validate_window, _row
from .intrabar_probe import _seal, _read_sealed, _write, _bytes_sha, _single_writer, STEP, ENDPOINT

APP=Path(__file__).resolve().parents[2]
SOURCES=('minute_replay_core.py','minute_replay_data.py','minute_replay_runner.py',
         'intrabar_data.py','intrabar_probe.py','public_research_audit.py','adapter.py','interaction_audit.py')


def query_parameters(scope,symbol,at):
    if (symbol not in scope['symbols'] or type(at) is not int or at%STEP or
            not scope['start']<=at<scope['end']):raise ValueError('MINUTE_QUERY_OUTSIDE_SCOPE')
    end=min(at+16*STEP,scope['end'])
    return dict(symbol=symbol,interval='1m',timeZone='0',startTime=at,endTime=end-1,limit=(end-at)//60000)


def scope_allows(scope,attempts,*,now_ms):
    if type(attempts) is not int or attempts<0 or attempts+3>scope['attempt_limit']:
        raise ValueError('MINUTE_ATTEMPT_BUDGET')
    if now_ms>=scope['expires_at_ms']:raise ValueError('MINUTE_SCOPE_EXPIRED')


def validate_decision(decision,at):
    if (not isinstance(decision,dict) or decision.get('origin_ms')!=at or
            type(decision.get('action')) is not int or decision['action'] not in (0,1,2,3) or
            decision.get('source')!='FROZEN_POLICY' or len(decision.get('state_sha256',''))!=64 or
            decision.get('sha256')!=canonical_sha256({k:v for k,v in decision.items() if k!='sha256'})):
        raise ValueError('MINUTE_FROZEN_DECISION_BINDING')


def accept_batch(rows,params,*,received_at_ms):
    if not isinstance(rows,list) or len(rows)>params['limit']:raise ValueError('MINUTE_BATCH_SCHEMA')
    grouped=defaultdict(list);previous=None
    for row in rows:
        _row(row,60000)
        if (not params['startTime']<=row[0]<=params['endTime'] or row[6]>=received_at_ms or
                previous is not None and row[0]<=previous):raise ValueError('MINUTE_BATCH_GRID_OR_RECEIPT')
        previous=row[0];grouped[(row[0]//STEP)*STEP].append(row)
    # An empty current frame must be retained as a failed response, not refetched
    # until one becomes conveniently complete. Future missing frames are opaque.
    grouped.setdefault(params['startTime'],[])
    return dict(grouped)


def verify_response_link(response,terminal):
    meta=response['metadata'];wrapper=terminal.get('payload',{})
    if (terminal.get('outcome')!='RESPONSE' or terminal.get('status')!=200 or
            any(meta['audit_'+k]!=terminal.get(k) for k in ('request_id','attempt_id')) or
            meta.get('received_at_ms')!=terminal.get('received_at_ms') or
            wrapper.get('representation')!='EXACT_PUBLIC_BYTES'):raise ValueError('MINUTE_RECEIPT_LINK')
    raw=base64.b64decode(wrapper['body_base64'],validate=True);digest=hashlib.sha256(raw).hexdigest()
    if (len(raw)!=wrapper['bytes_n'] or digest!=wrapper['content_sha256'] or digest!=meta['raw_body_sha256'] or
            json.loads(raw)!=response['rows'] or canonical_sha256(response['rows'])!=response['source_sha256']):
        raise ValueError('MINUTE_EXACT_WIRE_LINK')


def validated_frame(symbol,at,original,response,source_ref,source_sha256,*,imported):
    if original['row'][0]!=at or validate_window(original['row'],response['rows'],
            received_at_ms=response['metadata']['received_at_ms'])['state']!='MATCH':
        raise ValueError('NATIVE_FRAME_INCOMPLETE_OR_MISMATCH:'+symbol+':'+str(at))
    return dict(source_kind='BINANCE_SPOT_DEMO_1m',symbol=symbol,origin_ms=at,original=original['row'],
        rows=response['rows'],received_at_ms=response['metadata']['received_at_ms'],source_ref=str(source_ref),
        source_sha256=source_sha256,original_page_sha256=original['page_source_sha256'],imported=imported)


def frame_key(symbol,at):return canonical_sha256([symbol,at])


def _own(folder):
    folder=Path(folder).resolve()
    if not folder.is_relative_to(APP/'.runtime/minute-portfolio-replay') or folder==APP/'.runtime/minute-portfolio-replay':
        raise ValueError('OWN_MINUTE_REPLAY_RUNTIME_REQUIRED')
    return folder


def _scope(folder):
    scope=_read_sealed(Path(folder)/'data-scope.json')
    for name,digest in scope['files'].items():
        if _bytes_sha(name)!=digest:raise ValueError('MINUTE_SCOPE_INPUT_CHANGED:'+name)
    return scope


@_single_writer
def initialize(folder,*,ic_folder,parent_folder,spec_path):
    from .public_research_audit import verify_committed_probe_audit
    folder=_own(folder);ic_folder=Path(ic_folder).resolve();parent_folder=Path(parent_folder).resolve()
    if (folder/'data-scope.json').exists():return _scope(folder)
    proof=verify_committed_probe_audit(ic_folder,ic_folder/'rest-audit.sqlite')
    if proof['state']!='VERIFIED_RECORDED_PUBLIC_ATTEMPTS' or proof['linked_cached_responses_n']!=613 or proof['errors_n']:
        raise ValueError('MINUTE_PARENT_PUBLIC_PROVENANCE')
    ic=_read_sealed(ic_folder/'registration.json');census=_read_sealed(ic_folder/'census.json')
    parent=json.loads((parent_folder/'registration.json').read_text(encoding='utf-8'))
    if census['state']!='MINUTE_DATA_COMPLETE' or len(ic['windows'])!=613 or ic['cohort']!=parent['symbols']:
        raise ValueError('MINUTE_PARENT_SOURCE_INCOMPLETE')
    files={str(Path(__file__).with_name(n)):_bytes_sha(Path(__file__).with_name(n)) for n in SOURCES}
    for p in (Path(spec_path).resolve(),ic_folder/'registration.json',ic_folder/'census.json',ic_folder/'rest-audit.sqlite',
              parent_folder/'registration.json'):
        files[str(p)]=_bytes_sha(p)
    windows={(w['symbol'],w['origin_ms']):w for w in ic['windows']}
    copied=0
    for response_path in sorted((ic_folder/'responses').glob('*.json')):
        r=_read_sealed(response_path);p=r['parameters'];w=windows[(p['symbol'],p['startTime'])]
        if validate_window(w['row'],r['rows'],received_at_ms=r['metadata']['received_at_ms'])['state']!='MATCH':
            raise ValueError('MINUTE_IMPORTED_FRAME_INVALID')
        files[str(response_path)]=_bytes_sha(response_path)
        f=validated_frame(w['symbol'],w['origin_ms'],w,r,str(response_path),files[str(response_path)],imported=True)
        _seal(folder/'frames'/(frame_key(w['symbol'],w['origin_ms'])+'.json'),f);copied+=1
    if copied!=613:raise ValueError('MINUTE_IMPORTED_POPULATION')
    now=time.time_ns()//1000000
    scope=dict(protocol='MRP-1-NATIVE-MINUTE-ONLY',files=files,ic_folder=str(ic_folder),parent_folder=str(parent_folder),
        archive=ic['archive'],symbols=parent['symbols'],start=parent['splits'][1][1],end=parent['splits'][2][2],
        created_at_ms=now,expires_at_ms=now+7200000,attempt_limit=2000,imported_frames_n=copied,
        endpoint=ENDPOINT,promotion_allowed=False,evidence_kind='SIMULATION')
    _seal(folder/'data-scope.json',scope);_seal(folder/'parent-native-proof.json',proof)
    return scope


def _equal_rows(a,b):
    if len(a)!=len(b):return False
    return all(all((Decimal(x)==Decimal(y) if i not in (0,6,8) else x==y)
                   for i,(x,y) in enumerate(zip(ra,rb))) for ra,rb in zip(a,b))


@_single_writer
def fetch(folder,*,symbol,at,decision_path):
    from .adapter import DemoReadClient
    from .public_research_audit import PublicResearchJournal,read_committed_records
    from .intrabar_probe import PacedReadTransport
    from .isolation import AppPaths
    folder=_own(folder);scope=_scope(folder);decision_path=Path(decision_path).resolve()
    owner=_read_sealed(folder/'owner.json')
    if _bytes_sha(folder/'registration.json')!=owner['registration_sha256']:raise ValueError('MINUTE_OWNER_CHANGED')
    if not decision_path.is_relative_to(folder/'decisions'):raise ValueError('OWN_FROZEN_DECISION_REQUIRED')
    decision=_read_sealed(decision_path);validate_decision(decision,at)
    params=query_parameters(scope,symbol,at);q=canonical_sha256(params);response_path=folder/'responses'/(q+'.json')
    if response_path.exists():raise ValueError('MINUTE_REPEATED_UNCONSUMED_QUERY')
    if (folder/'queries'/(q+'.json')).exists():raise ValueError('MINUTE_PREVIOUS_QUERY_UNFINISHED')
    database=folder/'rest-audit.sqlite'
    prior=read_committed_records(database)[1] if database.exists() else {}
    attempts=len(prior)
    if prior:
        last=max(r['wall_ms'] for r in prior.values())
        time.sleep(max(0,(last+300-time.time_ns()//1000000)/1000))
    scope_allows(scope,attempts,now_ms=time.time_ns()//1000000)
    plan=dict(parameters=params,endpoint=ENDPOINT,owner_registration_sha256=owner['registration_sha256'],
              decision_path=str(decision_path),decision_sha256=_bytes_sha(decision_path),symbol=symbol,origin_ms=at)
    _seal(folder/'queries'/(q+'.json'),plan)
    journal=PublicResearchJournal(AppPaths(APP),account_alias='public-research',episode_id=folder.name,
        scope_id='native-minute-portfolio',database=str(database.relative_to(APP/'.runtime')),max_bytes=1024*1024*1024)
    deadline=time.monotonic()+min(75,(scope['expires_at_ms']-time.time_ns()//1000000)/1000)
    client=DemoReadClient(audit=journal,transport=PacedReadTransport(deadline=deadline))
    try:
        rows,meta=client.request_public('/api/v3/klines',params,weight=2)
        response=dict(endpoint=ENDPOINT,parameters=params,rows=rows,metadata=meta,source_sha256=canonical_sha256(rows))
        _seal(response_path,response)
    except BaseException as error:
        _write(folder/'errors'/(q+'-'+str(time.time_ns())+'.json'),dict(kind=type(error).__name__,query_sha256=q,at_ms=time.time_ns()//1000000))
        raise
    finally:client.close();journal.close()
    batches=accept_batch(rows,params,received_at_ms=meta['received_at_ms'])
    for origin,fine in batches.items():
        path=folder/'raw-frames'/(frame_key(symbol,origin)+'.json')
        if path.exists():
            old=_read_sealed(path)
            if not _equal_rows(old['rows'],fine):raise ValueError('MINUTE_OVERLAP_CONFLICT')
        else:
            _seal(path,dict(symbol=symbol,origin_ms=origin,rows=fine,received_at_ms=meta['received_at_ms'],
                source_ref=str(response_path),source_sha256=_bytes_sha(response_path)))
    return dict(state='FETCHED_NATIVE',rows_n=len(rows),parent_windows_n=len(batches),native_attempts_before_n=attempts)


class NativeMinuteStore:
    """Only called after action freeze. Future batches are never exposed to policy."""
    def __init__(self,folder):
        self.folder=_own(folder);self.scope=_read_sealed(self.folder/'data-scope.json');self.cache={};self.fetches=0
    def get(self,symbol,at,decision):
        validate_decision(decision,at);query_parameters(self.scope,symbol,at)
        key=frame_key(symbol,at);path=self.folder/'frames'/(key+'.json')
        if path.exists():return _read_sealed(path)
        rawpath=self.folder/'raw-frames'/(key+'.json')
        if not rawpath.exists():
            decision_path=self.folder/'decisions'/(decision['sha256']+'.json');_seal(decision_path,decision)
            command=[str(APP/'.venv/Scripts/python.exe'),'-m','binance_demo_bot.minute_replay_data','fetch',
                '--folder',str(self.folder),'--symbol',symbol,'--at',str(at),'--decision',str(decision_path)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=90,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode:raise ValueError('NATIVE_COLLECTOR_FAILED:'+result.stdout[-500:]+result.stderr[-500:])
            self.fetches+=1
        raw=_read_sealed(rawpath);original=original_bar(self.scope['archive'],symbol,at)
        check=validate_window(original['row'],raw['rows'],received_at_ms=raw['received_at_ms'])
        if check['state']!='MATCH':
            _write(self.folder/'rejected-frames'/(key+'.json'),dict(symbol=symbol,origin_ms=at,check=check,decision=decision))
            raise ValueError('NATIVE_FRAME_'+check['state']+':'+symbol+':'+str(at))
        f=validated_frame(symbol,at,original,dict(rows=raw['rows'],metadata=dict(received_at_ms=raw['received_at_ms'])),
            raw['source_ref'],raw['source_sha256'],imported=False)
        _seal(path,f);return f


def audit(folder):
    from .public_research_audit import read_committed_records,descriptor_matches
    folder=_own(folder);scope=_scope(folder);database=folder/'rest-audit.sqlite';linked=0;attempts=0;errors=0
    if database.exists():
        records,begins,ends=read_committed_records(database);attempts=len(begins);queries={}
        errors=sum(r['data'].get('outcome')!='RESPONSE' or r['data'].get('status')!=200 for r in ends.values())
        owner=_read_sealed(folder/'owner.json')
        for path in (folder/'queries').glob('*.json'):
            p=_read_sealed(path);params=query_parameters(scope,p['symbol'],p['origin_ms'])
            decision_path=Path(p['decision_path']);decision=_read_sealed(decision_path)
            validate_decision(decision,p['origin_ms'])
            if (p['parameters']!=params or p['endpoint']!=ENDPOINT or p['owner_registration_sha256']!=owner['registration_sha256'] or
                    not decision_path.is_relative_to(folder/'decisions') or _bytes_sha(decision_path)!=p['decision_sha256']):
                raise ValueError('MINUTE_QUERY_PLAN_CHANGED')
            queries[canonical_sha256(params)]=params
        for identifier,begin in begins.items():
            params=next((p for p in queries.values() if descriptor_matches(begin['data'],'/api/v3/klines',p)),None)
            if params is None or not descriptor_matches(ends[identifier]['data'],'/api/v3/klines'):raise ValueError('MINUTE_UNREGISTERED_REQUEST')
        used=set()
        for path in (folder/'responses').glob('*.json'):
            r=_read_sealed(path);meta=r['metadata'];begin=begins.get(meta['audit_begin_event_id']);end=records.get(meta['audit_end_event_id'])
            if (begin is None or end is None or ends.get(begin['event_id'])!=end or end['event_id'] in used or
                    not descriptor_matches(begin['data'],'/api/v3/klines',r['parameters']) or r['endpoint']!=ENDPOINT):
                raise ValueError('MINUTE_PUBLIC_CACHE_LINK')
            verify_response_link(r,end['data']);used.add(end['event_id']);linked+=1
        response_ids={r['event_id'] for r in ends.values() if r['data'].get('outcome')=='RESPONSE' and r['data'].get('status')==200}
        if used!=response_ids or attempts>scope['attempt_limit']:raise ValueError('MINUTE_UNBOUND_RESPONSE_OR_BUDGET')
    frames=imported=0
    for path in (folder/'frames').glob('*.json'):
        f=_read_sealed(path);source=Path(f['source_ref'])
        if _bytes_sha(source)!=f['source_sha256']:raise ValueError('MINUTE_FRAME_SOURCE_CHANGED')
        response=_read_sealed(source);p=response['parameters'];symbol=f['symbol'];at=f['origin_ms']
        if p['symbol']!=symbol or not p['startTime']<=at<=p['endTime']:raise ValueError('MINUTE_FRAME_QUERY_SCOPE')
        fine=[r for r in response['rows'] if at<=r[0]<at+STEP]
        original=original_bar(scope['archive'],symbol,at)
        if (fine!=f['rows'] or original['row']!=f['original'] or original['page_source_sha256']!=f['original_page_sha256'] or
                f['source_kind']!='BINANCE_SPOT_DEMO_1m' or validate_window(f['original'],fine,received_at_ms=f['received_at_ms'])['state']!='MATCH' or
                f['received_at_ms']!=response['metadata']['received_at_ms']):raise ValueError('MINUTE_FRAME_INTEGRITY')
        imported+=int(f['imported']);frames+=1
    proof=dict(state='VERIFIED_RECORDED_NATIVE_MINUTE_DATA',frames_n=frames,imported_frames_n=imported,new_frames_n=frames-imported,
        native_attempts_n=attempts,linked_new_responses_n=linked,errors_n=errors,promotion_allowed=False,financial_certification=False)
    _seal(folder/'native-data-audit.json',proof);return proof


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['init','fetch','audit']);p.add_argument('--folder',required=True)
    p.add_argument('--ic-folder');p.add_argument('--parent-folder');p.add_argument('--spec');p.add_argument('--symbol');p.add_argument('--at',type=int);p.add_argument('--decision')
    a=p.parse_args()
    if a.operation=='init':result=initialize(a.folder,ic_folder=a.ic_folder,parent_folder=a.parent_folder,spec_path=a.spec)
    elif a.operation=='fetch':result=fetch(a.folder,symbol=a.symbol,at=a.at,decision_path=a.decision)
    else:result=audit(a.folder)
    print(json.dumps({k:v for k,v in result.items() if k!='files' and k!='symbols'}),flush=True)

if __name__=='__main__':main()
