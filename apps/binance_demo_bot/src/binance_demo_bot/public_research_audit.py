"""PRC-1: exact public GET commitments with existing redaction left intact."""
import base64
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .interaction_audit import AuditJournal, sanitize
from .intrabar_probe import _read_sealed


def descriptor_sha(method,endpoint,parameters=None):
    value=dict(method=method,endpoint=endpoint,private=False,parameters=parameters)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def descriptor_matches(data,endpoint,parameters=None):
    return (data.get('method')=='GET' and data.get('private') is False and
        data.get('endpoint')==sanitize(endpoint) and data.get('parameters')==sanitize(parameters) and
        data.get('public_descriptor_sha256')==descriptor_sha('GET',endpoint,parameters))


class PublicResearchJournal(AuditJournal):
    def append(self,kind,data,**kwargs):
        if kind in ('REST_BEGIN','REST_END'):
            endpoint=data.get('endpoint','')
            allowed=(endpoint=='/api/v3/klines' or isinstance(endpoint,str) and
                re.fullmatch(r'https://data\.binance\.vision/data/spot/daily/klines/[^/?#]+/1s/[^/?#]+\.zip(?:\.CHECKSUM)?',endpoint))
            if data.get('method')!='GET' or data.get('private') is not False or not allowed:
                raise ValueError('PUBLIC_RESEARCH_GET_ONLY')
            data=dict(data,public_descriptor_sha256=descriptor_sha('GET',endpoint,data.get('parameters')))
        return super().append(kind,data,**kwargs)


def read_committed_records(path):
    path=Path(path);wal=path.with_name(path.name+'-wal')
    if wal.exists() and wal.stat().st_size:raise ValueError('PUBLIC_AUDIT_WAL_OPEN')
    connection=sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    records={};begins={};ends={};runs={};attempts=set();tip='0'*64
    try:
        identity=json.loads(connection.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
        for seq,identifier,kind,payload,claimed in connection.execute('SELECT seq,id,kind,payload_json,content_sha256 FROM records ORDER BY seq'):
            record=json.loads(payload)
            digest=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
            if (seq!=len(records)+1 or identifier in records or record['seq']!=seq or record['event_id']!=identifier or
                record['kind']!=kind or record['previous_sha256']!=tip or digest!=claimed or
                any(record.get(k)!=v for k,v in identity.items()) or record.get('application_id')!='binance_demo_bot' or
                record.get('environment')!='BINANCE_SPOT_DEMO'):raise ValueError('PUBLIC_AUDIT_CHAIN')
            records[identifier]=record;tip=claimed;run=record['run_id'];data=record['data']
            if kind=='RUN_BEGIN':
                if run in runs:raise ValueError('PUBLIC_DUPLICATE_RUN')
                runs[run]=False;continue
            if run not in runs or runs[run]:raise ValueError('PUBLIC_RUN_BOUNDARY')
            if kind=='RUN_END':runs[run]=True;continue
            if (kind not in ('REST_BEGIN','REST_END') or data.get('method')!='GET' or data.get('private') is not False or
                not re.fullmatch('[a-f0-9]{64}',data.get('public_descriptor_sha256',''))):raise ValueError('PUBLIC_DESCRIPTOR_MISSING')
            if kind=='REST_BEGIN':
                attempt=data.get('attempt_id')
                if not isinstance(attempt,str) or not attempt or attempt in attempts:raise ValueError('PUBLIC_ATTEMPT_ID')
                attempts.add(attempt);begins[identifier]=record
            else:
                parent=data.get('parent_event_id');begin=begins.get(parent)
                if (begin is None or parent in ends or begin['run_id']!=run or
                    any(data.get(k)!=begin['data'].get(k) for k in ('request_id','attempt_id','method','endpoint','private'))):
                    raise ValueError('PUBLIC_ATTEMPT_PAIR')
                ends[parent]=record
        if not runs or not all(runs.values()) or set(begins)!=set(ends):raise ValueError('PUBLIC_AUDIT_INCOMPLETE')
        return records,begins,ends
    finally:connection.close()


def verify_committed_probe_audit(folder,audit_path):
    folder=Path(folder);linked=0;failures=[];attempts=responses=errors=0
    try:
        registration=_read_sealed(folder/'registration.json')
        expected={}
        for w in registration['windows']:
            params=dict(symbol=w['symbol'],interval='1m',timeZone='0',startTime=w['origin_ms'],endTime=w['origin_ms']+899999,limit=15)
            expected[descriptor_sha('GET','/api/v3/klines',params)]=params
        records,begins,ends=read_committed_records(audit_path);attempts=len(begins);used=set()
        for identifier,begin in begins.items():
            data=begin['data'];params=expected.get(data['public_descriptor_sha256']);terminal=ends[identifier]['data']
            if params is None or not descriptor_matches(data,'/api/v3/klines',params) or not descriptor_matches(terminal,'/api/v3/klines'):
                raise ValueError('PUBLIC_REGISTERED_QUERY_MISMATCH')
            responses+=int(terminal.get('outcome')=='RESPONSE')
            errors+=int(terminal.get('outcome')!='RESPONSE' or terminal.get('status')!=200)
        for path in sorted((folder/'responses').glob('*.json')):
            cached=_read_sealed(path);params=cached['parameters'];meta=cached['metadata']
            begin=begins.get(meta['audit_begin_event_id']);end=records.get(meta['audit_end_event_id'])
            if (begin is None or end is None or ends.get(begin['event_id'])!=end or end['event_id'] in used or
                not descriptor_matches(begin['data'],'/api/v3/klines',params) or
                cached['endpoint']!='https://demo-api.binance.com/api/v3/klines' or
                any(meta['audit_'+key]!=end['data'][key] for key in ('request_id','attempt_id')) or
                meta['received_at_ms']!=end['data'].get('received_at_ms') or end['data'].get('outcome')!='RESPONSE' or
                end['data'].get('status')!=200):raise ValueError('PUBLIC_CACHE_RECEIPT_BINDING')
            wrapper=end['data'].get('payload',{})
            if wrapper.get('representation')!='EXACT_PUBLIC_BYTES':raise ValueError('PUBLIC_EXACT_BYTES_MISSING')
            raw=base64.b64decode(wrapper['body_base64'],validate=True);digest=hashlib.sha256(raw).hexdigest()
            if (len(raw)!=wrapper['bytes_n'] or digest!=wrapper['content_sha256'] or digest!=meta['raw_body_sha256'] or
                json.loads(raw)!=cached['rows'] or cached['source_sha256']!=hashlib.sha256(json.dumps(cached['rows'],
                    sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()):
                raise ValueError('PUBLIC_WIRE_BODY_BINDING')
            used.add(end['event_id']);linked+=1
    except (ValueError,TypeError,KeyError,OSError,sqlite3.Error) as error:failures.append(str(error) if isinstance(error,ValueError) else type(error).__name__)
    return dict(state='FAIL' if failures else 'VERIFIED_RECORDED_PUBLIC_ATTEMPTS',reasons=failures,
        attempts_n=attempts,responses_n=responses,errors_n=errors,linked_cached_responses_n=linked,
        validation_profile='GET_DESCRIPTOR_COMMITMENT_V1',scope='REGISTERED_NATIVE_PUBLIC_GETS_AND_RAW_CACHE_ONLY',promotion_allowed=False)
