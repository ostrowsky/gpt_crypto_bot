"""TA12 read-only completed PFD audit: integrity is not continuous coverage.

Stored validation is used only when the exact producer source is pinned. This
is not an independent full raw semantic reparse and never admits training.
"""
import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
from datetime import datetime,timedelta,timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo
from . import public_flow_dataset as flow
from .intrabar_probe import _seal,_write,_bytes_sha



def iter_bounded_lines(stream,*,record_limit=flow.CHUNK_BYTES):
    """Preserve exact JSONL bytes using fixed64KiB reads and one bounded record."""
    if type(record_limit) is not int or not 1<=record_limit<=flow.CHUNK_BYTES:raise ValueError('FLOW_RECORD_LIMIT')
    pending=bytearray()
    while True:
        block=stream.read(65536)
        if not block:break
        for i,part in enumerate(block.split(b'\n')):
            if i:
                pending.extend(b'\n')
                if len(pending)>record_limit:raise ValueError('FLOW_RECORD_LIMIT')
                yield bytes(pending);pending.clear()
            pending.extend(part)
            if len(pending)>record_limit:raise ValueError('FLOW_RECORD_LIMIT')
    if pending:yield bytes(pending)


def _verified_chunk_lines(raw,item):
    digest=hashlib.sha256();total=0;count=0
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        class HashingStream:
            def read(self,size):
                nonlocal total
                block=stream.read(size);total+=len(block)
                if total>flow.CHUNK_BYTES:raise ValueError('FLOW_CHUNK_DECOMPRESSED_LIMIT')
                digest.update(block);return block
        for line in iter_bounded_lines(HashingStream()):count+=1;yield line
    if total!=item['uncompressed_bytes'] or digest.hexdigest()!=item['uncompressed_sha256']:raise ValueError('FLOW_CHUNK_UNCOMPRESSED_HASH')
    if count!=item['records_n']:raise ValueError('FLOW_CHUNK_RECORDS')


def audit_capture(folder,*,output=None):
    folder=Path(folder).resolve();output=Path(output).resolve() if output is not None else None
    files={}
    def read(name):
        path=folder/name;raw=path.read_bytes();files[str(path)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    registration=read('registration.json');flow._validate_registration(registration)
    if registration['source_sha256']!=_bytes_sha(flow.__file__):raise ValueError('FLOW_PRODUCER_SOURCE_MISMATCH')
    files[str(Path(flow.__file__).resolve())]=registration['source_sha256']
    manifest=read('manifest.json');status=read('status.json');reg_sha=flow._sha(registration)
    if manifest['registration_sha256']!=reg_sha or status['registration_sha256']!=reg_sha:raise ValueError('FLOW_REGISTRATION_BINDING')
    start,end=registration['full_days_start_ms'],registration['planned_end_ms'];zone=ZoneInfo(registration['timezone'])
    dates=[];boundaries=[];day=datetime.fromtimestamp(start/1000,timezone.utc).astimezone(zone).date()
    while True:
        at=int(datetime.combine(day,datetime.min.time(),zone).timestamp()*1000)
        if at>=end:break
        dates.append(day.isoformat());boundaries.append(at);day+=timedelta(days=1)
    days={name:dict(market_records_n=0,topic_records={},recorded_gaps_n=0) for name in dates}
    counts=Counter();topics=Counter();reasons=Counter();sessions=set();active=None;last_wall=last_mono=None
    first_market=last_market=None;gaps=invalid=duplicates=conflicts=controls=warmup=tail=0;compressed_n=records_n=0
    for index,item in enumerate(manifest['chunks']):
        if item['file']!='chunk-%08d.jsonl.gz'%index:raise ValueError('FLOW_CHUNK_ORDER')
        path=folder/item['file'];raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if len(raw)!=item['compressed_bytes'] or len(raw)>flow.CHUNK_BYTES+65536 or digest!=item['sha256']:raise ValueError('FLOW_CHUNK_HASH')
        files[str(path)]=digest;compressed_n+=len(raw)
        chunk_sessions=set();times=[]
        for line in _verified_chunk_lines(raw,item):
            record=json.loads(line);kind=record['kind'];wall=record['wall_ms'];mono=record['mono_ns'];sid=record['session_id']
            if (type(wall) is not int or type(mono) is not int or wall<registration['started_at_ms'] or
                    last_wall is not None and (wall<last_wall or mono<last_mono)):
                raise ValueError('FLOW_RECEIPT_REGRESSION')
            last_wall,last_mono=wall,mono;times.append(wall);chunk_sessions.add(sid);counts[kind]+=1;records_n+=1
            if kind=='SESSION_START':
                if active is not None or not isinstance(sid,str) or not sid or sid in sessions:raise ValueError('FLOW_SESSION_BOUNDARY')
                gaps+=int(bool(sessions));sessions.add(sid);active=sid;continue
            if sid!=active:raise ValueError('FLOW_SESSION_OWNERSHIP')
            if kind=='SESSION_END':active=None;continue
            if kind!='MARKET':
                invalid+=1;continue
            validation=record['validation'];topic=record['stream']
            if topic=='!serverShutdown':controls+=1;continue
            if topic not in registration['streams'] or not isinstance(record['raw'],str):raise ValueError('FLOW_TOPIC_SCHEMA')
            if type(validation['valid']) is not bool or not isinstance(validation['reasons'],list):raise ValueError('FLOW_VALIDATION_SCHEMA')
            topics[topic]+=1;reasons.update(validation['reasons'])
            gap=validation['continuity']=='GAP';gaps+=int(gap);invalid+=int(not validation['valid'])
            duplicates+=int(validation['continuity']=='DUPLICATE');conflicts+=int(any('CONFLICT' in r for r in validation['reasons']))
            first_market=wall if first_market is None else first_market;last_market=wall
            if wall<start:warmup+=1
            elif wall>=end:tail+=1
            else:
                row=days[dates[bisect_right(boundaries,wall)-1]];row['market_records_n']+=1;row['recorded_gaps_n']+=int(gap)
                row['topic_records'][topic]=row['topic_records'].get(topic,0)+1
        if (min(times,default=None)!=item['first_receipt_ms'] or max(times,default=None)!=item['last_receipt_ms'] or
                sorted(chunk_sessions)!=item['session_ids']):raise ValueError('FLOW_CHUNK_RECEIPT_METADATA')
        if output is not None and (index%128==0 or index+1==len(manifest['chunks'])):
            _write(output/'status.json',dict(phase='AUDITING_CAPTURE',chunks_done_n=index+1,chunks_total_N=len(manifest['chunks']),
                records_n=records_n,checked_at_ms=time.time_ns()//1000000,promotion_allowed=False))
    if (records_n!=manifest['records_n'] or compressed_n!=manifest['compressed_bytes'] or
            len(list(folder.glob('chunk-*.jsonl.gz')))!=len(manifest['chunks'])):raise ValueError('FLOW_MANIFEST_TOTALS')
    if any(topics.get(t,0)!=status['messages_per_topic'].get(t,0) for t in registration['streams']):raise ValueError('FLOW_STATUS_COUNTER_MISMATCH')
    if (len(sessions)!=status['sessions_n'] or len(sessions)-1!=status['reconnections_n'] or gaps!=status['gaps_n'] or
            invalid!=status['invalid_n'] or duplicates!=status['duplicates_n'] or conflicts!=status['conflicts_n'] or
            first_market!=status['first_receipt_ms'] or last_market!=status['last_receipt_ms']):raise ValueError('FLOW_STATUS_COUNTER_MISMATCH')
    if status['planned_end_ms']!=end or status['full_days_start_ms']!=start:raise ValueError('FLOW_STATUS_SCHEDULE')
    for name in ('registration.json','manifest.json','status.json'):
        path=folder/name
        if _bytes_sha(path)!=files[str(path)]:raise ValueError('FLOW_INPUT_CHANGED_DURING_AUDIT')
    report=dict(state='DATA_AUDIT_ONLY_NOT_TRAINED',integrity='PASS',records_n=records_n,market_records_n=counts['MARKET'],
        chunks_n=len(manifest['chunks']),compressed_bytes_n=compressed_n,record_kinds=dict(counts),
        recorded_gaps_n=gaps,stale_receipt_pauses_n=reasons['STALE_RECEIPT_GAP'],reason_counts=dict(reasons),
        sessions_n=len(sessions),reconnections_n=max(0,len(sessions)-1),unclosed_final_session=active is not None,
        invalid_records_n=invalid,controls_n=controls,topic_records=dict(topics),days=days,
        planned_days_N=len(dates),planned_days_with_records_n=sum(v['market_records_n']>0 for v in days.values()),
        first_market_receipt_ms=first_market,last_market_receipt_ms=last_market,warmup_market_n=warmup,after_end_market_n=tail,
        producer_calendar_schedule_finished=status.get('calendar_schedule_finished') is True,
        continuous_coverage=False if gaps or invalid or active is not None else None,
        stored_validation_profile='PINNED_PRODUCER_VALIDATION_NOT_INDEPENDENT_RAW_REPARSE',
        depth_semantics='STANDING_TOP20_SNAPSHOT_NOT_DIFF_ORDER_FLOW',exchange_age='UNKNOWN',
        financial_certification=False,net_pnl_usdt=None,model_training='NOT_TRAINED',promotion_allowed=False,
        files=files,auditor_source_sha256=_bytes_sha(__file__),registration_sha256=reg_sha)
    if output is not None:
        _seal(output/'report.json',report)
        _write(output/'status.json',dict(phase='COMPLETE_DATA_AUDIT_ONLY',records_n=records_n,model_training='NOT_TRAINED',promotion_allowed=False))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();app=Path(__file__).resolve().parents[2]
    folder=flow.own_runtime_path(app,args.folder);output=flow.own_runtime_path(app,args.output)
    try:report=audit_capture(folder,output=output)
    except BaseException as error:
        _write(output/'status.json',dict(phase='INCOMPLETE_AUDIT',exception_class=type(error).__name__,model_training='NOT_TRAINED',financial_certification=False,promotion_allowed=False))
        raise
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','days')}),flush=True)

if __name__=='__main__':main()
