"""AUD-A/B1/B2/B3: app-local durable interaction/decision evidence, never an OMS.

The ws_audit bridge supplies application and pinned parsed-control observations.
Full wire capture and policy coverage remain incomplete. No credential discovery,
arbitrary endpoint capture, async backlog or automatic log deletion.
"""
import base64
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import unquote
import uuid

from . import market_sqlite as sqlite3
from .adapter import AdapterError
from .isolation import AppPaths
from .journal import _WriterLease

KINDS = ('RUN_BEGIN', 'RUN_END', 'REST_BEGIN', 'REST_END', 'DECISION','WS_BEGIN','WS_END','WS_PROCESSING','WS_CONTROL')
MAX_EVENT_BYTES = 48*1024*1024
SENSITIVE = re.compile(r'api[-_ ]?key|api[-_ ]?secret|signature|authorization|cookie|listen[-_ ]?key|access[-_ ]?token|password|secret|token', re.I)
REDACTED = '[REDACTED]'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _text(value):
    if not isinstance(value,str) or not value or len(value)>256 or any(ord(c)<33 for c in value):
        raise AdapterError('AUDIT_INVALID')
    return value


def _decoded(value):
    for _ in range(3):value=unquote(value)
    return value


def _sensitive(value,secrets):
    decoded=_decoded(value)
    return (SENSITIVE.search(decoded) or re.search(r'%[0-9a-f]{2}',decoded,re.I)
            or any(s and s in decoded for s in secrets))


def sanitize(value, secrets=()):
    """No dependency exception text is accepted; scrub structured/embedded auth."""
    if isinstance(value,dict):
        if any(not isinstance(k,str) for k in value):raise AdapterError('AUDIT_INVALID')
        if (value.get('representation')=='EXACT_PUBLIC_BYTES' and set(value)==
                {'representation','body_base64','content_sha256','bytes_n'}):
            # Encoding is not free text. Validate the original bytes rather than
            # accidentally masking an innocent base64 substring. Forged wrappers
            # still go through the same raw-content secret check.
            try:
                raw=base64.b64decode(value['body_base64'],validate=True)
                if len(raw)!=value['bytes_n'] or hashlib.sha256(raw).hexdigest()!=value['content_sha256']:
                    raise ValueError()
            except (ValueError,TypeError):raise AdapterError('AUDIT_INVALID') from None
            return response_payload(raw,private=False,secrets=secrets)
        return {k:sanitize(v,secrets) for k,v in value.items()
                if not _sensitive(k,secrets)}
    if isinstance(value,(list,tuple)):
        return [sanitize(v,secrets) for v in value]
    if isinstance(value,str):
        return REDACTED if _sensitive(value,secrets) else value
    if value is None or type(value) in (bool,int,float):return value
    raise AdapterError('AUDIT_INVALID')


def response_payload(body, *, private, secrets=()):
    """Exact public bytes only when auth-free; private raw is never persisted."""
    if not isinstance(body,bytes) or len(body)>32*1024*1024:
        return dict(representation='OMITTED',bytes_n=len(body) if isinstance(body,bytes) else None)
    base=dict(bytes_n=len(body))
    try:
        text=body.decode('utf-8')
        data=json.loads(text)
        clean=sanitize(data,secrets)
        # Inspect the text too: duplicate JSON keys must not hide secrets.
        unsafe=_sensitive(text,secrets)
        if private or clean!=data or unsafe:
            return dict(base,representation='SANITIZED',data=clean,content_sha256=_sha(clean))
    except (ValueError,UnicodeError,RecursionError,AdapterError):
        if private:return dict(base,representation='OMITTED')
        try:
            text=body.decode('utf-8')
            if sanitize(text,secrets)!=text:return dict(base,representation='OMITTED')
        except (UnicodeError,RecursionError):return dict(base,representation='OMITTED')
    return dict(base,representation='EXACT_PUBLIC_BYTES',body_base64=base64.b64encode(body).decode('ascii'),
                content_sha256=hashlib.sha256(body).hexdigest())


def rate_headers(headers):
    if not isinstance(headers,dict):return {}
    return {k.lower():v for k,v in headers.items() if isinstance(k,str) and isinstance(v,str)
            and (k.lower()=='retry-after' or re.fullmatch(r'x-mbx-used-weight-\d+[smhd]',k.lower()))
            and v.isascii() and v.isdecimal() and len(v)<=20}


class AuditJournal:
    def __init__(self,paths,*,account_alias,episode_id,scope_id,database='audit.sqlite',
                 max_bytes=512*1024*1024,max_inflight=8,fault_hook=None):
        if not isinstance(paths,AppPaths):raise AdapterError('AUDIT_IDENTITY_MISMATCH')
        for v in (account_alias,episode_id,scope_id):_text(v)
        if type(max_bytes) is not int or max_bytes<65536 or type(max_inflight) is not int or not 1<=max_inflight<=8:
            raise AdapterError('AUDIT_INVALID')
        self.identity=dict(application_id='binance_demo_bot',environment='BINANCE_SPOT_DEMO',
            objective_contract_id='daily_net_equity_pnl_v1',schema_version=1,account_alias=account_alias,
            account_episode_id=episode_id,portfolio_scope_id=scope_id)
        package=Path(__file__).resolve().parent
        self.source_code_sha256=_sha({n:hashlib.sha256((package/n).read_bytes()).hexdigest()
                                      for n in ('interaction_audit.py','adapter.py','ws_audit.py','market_stream.py','private_stream.py','market_runtime.py','operational_audit.py','protocol_audit.py')})
        self.path=paths.runtime_path(database)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.max_bytes,self.fault_hook=max_bytes,fault_hook
        self.run_id,self.process_id=uuid.uuid4().hex,uuid.uuid4().hex
        self._lock,self._metrics_lock=threading.RLock(),threading.Lock()
        self._slots=threading.BoundedSemaphore(max_inflight)
        self._expected,self._acked=Counter(),Counter()
        self._state,self._closed='READY',False
        self._failure_reason=None
        self._con,self._lease=None,None
        self._tip='0'*64
        try:
            self._lease=_WriterLease(paths.runtime_path(str(database)+'.writer.lock'))
            self._con=sqlite3.connect(self.path,timeout=0,isolation_level=None,check_same_thread=False)
            self._con.execute('PRAGMA journal_mode=WAL')
            self._con.execute('PRAGMA synchronous=FULL')
            tables={r[0] for r in self._con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not tables:
                self._con.execute('BEGIN IMMEDIATE')
                try:
                    self._con.execute('CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
                    self._con.execute('CREATE TABLE records (seq INTEGER PRIMARY KEY,id TEXT UNIQUE NOT NULL,kind TEXT NOT NULL,payload_json TEXT NOT NULL,content_sha256 TEXT NOT NULL)')
                    self._con.execute('INSERT INTO meta VALUES (?,?)',('identity',_json(self.identity)))
                    for table in ('meta','records'):
                        for op in ('UPDATE','DELETE'):
                            self._con.execute(f"CREATE TRIGGER immutable_{table}_{op} BEFORE {op} ON {table} BEGIN SELECT RAISE(ABORT,'immutable audit'); END")
                    self._con.execute("CREATE INDEX audit_attempt ON records(kind,json_extract(payload_json,'$.data.attempt_id'))")
                    self._con.commit()
                except BaseException:
                    self._con.rollback();raise
            self._verify()
            self._con.execute("CREATE INDEX IF NOT EXISTS audit_ws_operation ON records(kind,json_extract(payload_json,'$.data.operation_id'))")
            self.append('RUN_BEGIN',dict(scope='REST_ATTEMPTS_AND_EXPLICIT_DECISIONS'))
        except BaseException:
            self._state='FAILED'
            self.close()
            raise

    def _verify(self):
        try:
            identity=self._con.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
            if identity is None or json.loads(identity[0])!=self.identity:
                raise AdapterError('AUDIT_IDENTITY_MISMATCH')
            triggers={r[0] for r in self._con.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
            if not {f'immutable_{t}_{o}' for t in ('meta','records') for o in ('UPDATE','DELETE')}<=triggers:
                raise ValueError()
            tip='0'*64;seq=0
            for row in self._con.execute('SELECT seq,id,kind,payload_json,content_sha256 FROM records ORDER BY seq'):
                value=json.loads(row[3]);seq+=1
                if (row[0]!=seq or value['seq']!=seq or value['event_id']!=row[1]
                    or value['kind']!=row[2] or row[2] not in KINDS or value['previous_sha256']!=tip
                    or _sha(value)!=row[4] or any(value.get(k)!=v for k,v in self.identity.items())):
                    raise ValueError()
                tip=row[4]
            self._tip=tip;self._last_seq=seq
        except AdapterError:raise
        except (ValueError,KeyError,TypeError,sqlite3.Error):
            raise AdapterError('AUDIT_INTEGRITY_FAILED') from None

    def _failed(self,reason):
        with self._metrics_lock:
            self._state='FAILED'
            if self._failure_reason is None:self._failure_reason=reason

    def _disk_bytes(self):
        return sum(p.stat().st_size for p in (self.path,self.path.with_name(self.path.name+'-wal')) if p.exists())

    def append(self,kind,data,*,wall_ms=None,monotonic_ns=None,secrets=()):
        if kind not in KINDS or not isinstance(data,dict):raise AdapterError('AUDIT_INVALID')
        with self._metrics_lock:self._expected[kind]+=1
        if not self._slots.acquire(blocking=False):
            self._failed('AUDIT_BACKPRESSURE');raise AdapterError('AUDIT_BACKPRESSURE')
        try:
            with self._lock:
                if self._closed or self._state!='READY':raise AdapterError('AUDIT_UNAVAILABLE')
                phase='PREPARE'
                try:
                    wall=time.time_ns()//1000000 if wall_ms is None else wall_ms
                    mono=time.monotonic_ns() if monotonic_ns is None else monotonic_ns
                    if any(type(v) is not int or v<0 for v in (wall,mono)):raise AdapterError('AUDIT_INVALID')
                    value=dict(self.identity,seq=self._last_seq+1,event_id=uuid.uuid4().hex,kind=kind,
                        run_id=self.run_id,process_instance_id=self.process_id,source_code_sha256=self.source_code_sha256,
                        wall_ms=wall,monotonic_ns=mono,previous_sha256=self._tip,data=sanitize(data,secrets))
                    payload=_json(value);digest=hashlib.sha256(payload.encode()).hexdigest()
                    size=len(payload.encode())
                    if size>MAX_EVENT_BYTES:raise AdapterError('AUDIT_EVENT_LIMIT')
                    if self._disk_bytes()+2*size+65536>self.max_bytes:raise AdapterError('AUDIT_DISK_LIMIT')
                    self._con.execute('BEGIN IMMEDIATE')
                    phase='INSERT'
                    self._con.execute('INSERT INTO records VALUES (?,?,?,?,?)',
                        (value['seq'],value['event_id'],kind,payload,digest))
                    if self.fault_hook:self.fault_hook('before_commit')
                    phase='COMMIT'
                    self._con.commit()
                    if self.fault_hook:self.fault_hook('after_commit')
                    self._tip,self._last_seq=digest,value['seq']
                    with self._metrics_lock:self._acked[kind]+=1
                    return value['event_id']
                except BaseException as error:
                    try:self._con.rollback()
                    except Exception:pass
                    reason=('AUDIT_COMMIT_UNKNOWN' if phase=='COMMIT' else
                            error.reason if isinstance(error,AdapterError) else 'AUDIT_WRITE_FAILED')
                    self._failed(reason)
                    if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
                    raise AdapterError(reason) from None
        finally:self._slots.release()

    def record_decision(self,*,decision_type,action,reasons,cutoff_ms,sources,candidate_population,
                        policy_version,model_version,risk_version,intent=None):
        if (decision_type not in ('OPERATIONAL','TRADING') or action not in
            ('BUY','SELL','HOLD','WAIT','BLOCK','NO_ACTION','REFRESH','RECONNECT','BACKFILL')
            or type(cutoff_ms) is not int or cutoff_ms<0 or not isinstance(sources,list)
            or not isinstance(reasons,list) or not reasons or not isinstance(candidate_population,list)):
            raise AdapterError('AUDIT_INVALID')
        for value in reasons+candidate_population+ [policy_version,risk_version]:_text(value)
        if model_version is not None:_text(model_version)
        for source in sources:
            if (not isinstance(source,dict) or set(source)!={'snapshot_id','available_at_ms'}
                or type(source['available_at_ms']) is not int or source['available_at_ms']<0):
                raise AdapterError('AUDIT_INVALID')
            _text(source['snapshot_id'])
            if source['available_at_ms']>cutoff_ms:raise AdapterError('AUDIT_FUTURE_SOURCE')
        return self.append('DECISION',dict(decision_type=decision_type,action=action,reasons=reasons,
            cutoff_ms=cutoff_ms,sources=sources,candidate_population=candidate_population,
            policy_version=policy_version,model_version=model_version,risk_version=risk_version,intent=intent))

    def read(self,*,after_seq=0,limit=100):
        if type(after_seq) is not int or after_seq<0 or type(limit) is not int or not 1<=limit<=1000:
            raise AdapterError('AUDIT_INVALID')
        with self._lock:
            return [dict(json.loads(r[0]),content_sha256=r[1]) for r in self._con.execute(
                'SELECT payload_json,content_sha256 FROM records WHERE seq>? ORDER BY seq LIMIT ?', (after_seq,limit))]

    def manifest(self):
        with self._lock:
            self._verify()
            value=dict(identity=deepcopy(self.identity),first_seq=1 if self._last_seq else None,
                last_seq=self._last_seq,rows_n=self._last_seq,chain_tip=self._tip)
            return dict(value,manifest_sha256=_sha(value))

    def processing_report(self,*,through_seq=None):
        """Bounded Python memory; durable-prefix coverage is not transport coverage."""
        with self._lock:
            self._verify()
            cutoff=self._last_seq if through_seq is None else through_seq
            if type(cutoff) is not int or not 0<=cutoff<=self._last_seq:raise AdapterError('AUDIT_INVALID')
            tip=self._con.execute('SELECT content_sha256 FROM records WHERE seq=?',(cutoff,)).fetchone()
            result=dict(prefix_last_seq=cutoff,prefix_chain_tip=tip[0] if tip else '0'*64,
                recorded_receipts_N=0,processed_n=0,rejected_n=0,control_ack_n=0,unknown_n=0,conflict_n=0,
                invalid_marker_n=0,denominator_basis='DURABLE_COMPLETED_RECV',capture_coverage='UNKNOWN',
                full_audit_complete=False)
            links="""WITH links AS (
                SELECT m.seq AS marker_seq,m.payload_json AS marker_json,j.value AS receipt_id
                FROM records m,json_each(m.payload_json,'$.data.receipt_event_ids') j
                WHERE m.kind='WS_PROCESSING' AND m.seq<=?) """
            query=links+"""SELECT r.seq,COUNT(l.marker_seq),MAX(json_extract(l.marker_json,'$.data.outcome')),
                SUM(CASE WHEN l.marker_seq<=r.seq OR
                    json_type(l.marker_json,'$.data.receipt_event_ids')!='array' OR
                    json_extract(l.marker_json,'$.data.connection_id') IS NOT json_extract(r.payload_json,'$.data.connection_id') OR
                    json_extract(l.marker_json,'$.data.outcome') NOT IN ('PROCESSED','REJECTED','CONTROL_ACK')
                    THEN 1 ELSE 0 END)
                FROM records r LEFT JOIN links l ON l.receipt_id=r.id
                WHERE r.kind='WS_END' AND r.seq<=? AND json_extract(r.payload_json,'$.data.operation')='RECV'
                    AND json_extract(r.payload_json,'$.data.outcome')='COMPLETED'
                GROUP BY r.seq ORDER BY r.seq"""
            for _,count,outcome,invalid in self._con.execute(query,(cutoff,cutoff)):
                result['recorded_receipts_N']+=1
                if count==0:category='unknown_n'
                elif count!=1 or invalid or outcome not in ('PROCESSED','REJECTED','CONTROL_ACK'):category='conflict_n'
                else:category={'PROCESSED':'processed_n','REJECTED':'rejected_n','CONTROL_ACK':'control_ack_n'}[outcome]
                result[category]+=1
            query=links+"""SELECT COUNT(*) FROM links l LEFT JOIN records r ON r.id=l.receipt_id
                WHERE r.seq IS NULL OR r.seq> ? OR r.seq>=l.marker_seq OR r.kind!='WS_END'
                    OR json_extract(r.payload_json,'$.data.operation')!='RECV'
                    OR json_extract(r.payload_json,'$.data.outcome')!='COMPLETED'"""
            result['invalid_marker_n']=self._con.execute(query,(cutoff,cutoff)).fetchone()[0]
            empty=self._con.execute("SELECT COUNT(*) FROM records WHERE kind='WS_PROCESSING' AND seq<=? AND COALESCE(json_array_length(payload_json,'$.data.receipt_event_ids'),0)=0",(cutoff,)).fetchone()[0]
            result['invalid_marker_n']+=empty
            result['state']=('FAIL' if result['conflict_n'] or result['invalid_marker_n'] else
                'UNKNOWN' if not result['recorded_receipts_N'] or result['unknown_n'] else 'RESOLVED_RECORDED_PREFIX')
            return dict(result,report_sha256=_sha(result))

    def health(self):
        with self._metrics_lock:
            value=dict(state=self._state,scope='REST_ATTEMPTS_AND_EXPLICIT_DECISIONS',full_audit_complete=False,
                error=self._failure_reason,
                counter_basis='PROCESS_SUBMISSIONS',run_id=self.run_id,process_instance_id=self.process_id,
                channels={k:dict(expected_N=self._expected[k],recorded_n=self._acked[k],gap_n=self._expected[k]-self._acked[k]) for k in KINDS})
        with self._lock:
            if not self._closed:
                unresolved=self._con.execute("SELECT COUNT(*) FROM records b WHERE b.kind='REST_BEGIN' AND NOT EXISTS (SELECT 1 FROM records e WHERE e.kind='REST_END' AND json_extract(e.payload_json,'$.data.attempt_id')=json_extract(b.payload_json,'$.data.attempt_id'))").fetchone()[0]
                unresolved_ws=self._con.execute("SELECT COUNT(*) FROM records b WHERE b.kind='WS_BEGIN' AND NOT EXISTS (SELECT 1 FROM records e WHERE e.kind='WS_END' AND json_extract(e.payload_json,'$.data.operation_id')=json_extract(b.payload_json,'$.data.operation_id'))").fetchone()[0]
                incomplete=self._con.execute("SELECT COUNT(*) FROM records b WHERE b.kind='RUN_BEGIN' AND json_extract(b.payload_json,'$.run_id')!=? AND NOT EXISTS (SELECT 1 FROM records e WHERE e.kind='RUN_END' AND json_extract(e.payload_json,'$.run_id')=json_extract(b.payload_json,'$.run_id'))",(self.run_id,)).fetchone()[0]
                self._last_gaps=dict(unresolved_attempts_n=unresolved,unresolved_ws_operations_n=unresolved_ws,incomplete_prior_runs_n=incomplete)
            value.update(getattr(self,'_last_gaps',dict(unresolved_attempts_n=None,incomplete_prior_runs_n=None)))
        return value

    def close(self):
        with self._lock:
            if self._closed:return
            try:
                if self._con is not None and self._state=='READY':self.append('RUN_END',{})
            finally:
                self._closed=True
                if self._state!='FAILED':self._state='CLOSED'
                try:
                    if self._con is not None:self._con.close()
                finally:
                    if self._lease is not None:self._lease.close()
