"""Durable hold history. Missing authoritative certification never permits deletion."""
import json
import re

from .adapter import AdapterError
from .interaction_audit import AuditJournal, _sha, _json
from .journal import WriterBusy


CATEGORIES={'ACTIVE_EXPOSURE','UNFINALIZED_DAY','RESEARCH','INTEGRITY'}
SCOPE='RETENTION_HOLD'


def _hex(value,size):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{'+str(size)+'}',value) is None:
        raise AdapterError('AUDIT_HOLD_INVALID')


def _valid(data):
    try:
        common={'scope','operation','seal_sha256','evidence_sha256','request_id'}
        if not isinstance(data,dict) or data.get('scope')!=SCOPE:return False
        if data.get('operation')=='PLACE':
            if set(data)!=common|{'category','reference_id'} or data['category'] not in CATEGORIES:return False
            _hex(data['reference_id'],32)
        elif data.get('operation')=='RELEASE':
            if set(data)!=common|{'hold_event_id'}:return False
            _hex(data['hold_event_id'],32)
        else:return False
        for key,size in (('seal_sha256',64),('evidence_sha256',64),('request_id',32)):_hex(data[key],size)
        return True
    except (AdapterError,TypeError,KeyError):return False


class RetentionLedger:
    def __init__(self,paths,*,account_alias,episode_id,scope_id):
        try:
            self._journal=AuditJournal(paths,account_alias=account_alias,episode_id=episode_id,scope_id=scope_id,
                database='audit-retention.sqlite',max_bytes=16*1024*1024,max_inflight=4)
        except WriterBusy:raise AdapterError('AUDIT_HOLD_BUSY') from None

    def _ready(self):
        if self._journal._closed or self._journal._state!='READY':
            raise AdapterError(self._journal._failure_reason or 'AUDIT_HOLD_UNAVAILABLE')

    def _event(self,event_id):
        row=self._journal._con.execute("SELECT seq,id,payload_json FROM records WHERE id=? AND kind='DECISION'",(event_id,)).fetchone()
        if row is None:return None
        return dict(seq=row[0],event_id=row[1],data=json.loads(row[2])['data'])

    def _retry(self,data):
        row=self._journal._con.execute("SELECT id FROM records WHERE kind='DECISION' AND json_extract(payload_json,'$.data.scope')=? AND json_extract(payload_json,'$.data.request_id')=?",(SCOPE,data['request_id'])).fetchone()
        if row is None:return None
        event=self._event(row[0])
        if event['data']!=data:raise AdapterError('AUDIT_HOLD_CONFLICT')
        return event['event_id']

    def place(self,seal_sha256,*,category,reference_id,evidence_sha256,request_id):
        for value,size in ((seal_sha256,64),(reference_id,32),(evidence_sha256,64),(request_id,32)):_hex(value,size)
        if category not in CATEGORIES:raise AdapterError('AUDIT_HOLD_INVALID')
        data=dict(scope=SCOPE,operation='PLACE',seal_sha256=seal_sha256,category=category,
            reference_id=reference_id,evidence_sha256=evidence_sha256,request_id=request_id)
        with self._journal._lock:
            self._ready();prior=self._retry(data)
            return prior or self._journal.append('DECISION',data)

    def release(self,hold_event_id,*,evidence_sha256,request_id):
        for value,size in ((hold_event_id,32),(evidence_sha256,64),(request_id,32)):_hex(value,size)
        with self._journal._lock:
            self._ready();hold=self._event(hold_event_id)
            if hold is None or hold['data'].get('scope')!=SCOPE or hold['data'].get('operation')!='PLACE':
                raise AdapterError('AUDIT_HOLD_MISSING')
            if not _valid(hold['data']):raise AdapterError('AUDIT_HOLD_INTEGRITY')
            data=dict(scope=SCOPE,operation='RELEASE',seal_sha256=hold['data']['seal_sha256'],
                hold_event_id=hold_event_id,evidence_sha256=evidence_sha256,request_id=request_id)
            prior=self._retry(data)
            if prior:return prior
            released=self._journal._con.execute("SELECT 1 FROM records WHERE kind='DECISION' AND json_extract(payload_json,'$.data.scope')=? AND json_extract(payload_json,'$.data.operation')='RELEASE' AND json_extract(payload_json,'$.data.hold_event_id')=?",(SCOPE,hold_event_id)).fetchone()
            if released:raise AdapterError('AUDIT_HOLD_RELEASED')
            return self._journal.append('DECISION',data)

    def status(self,seal_sha256,*,through_seq=None):
        _hex(seal_sha256,64)
        with self._journal._lock:
            self._journal._verify()
            cutoff=self._journal._last_seq if through_seq is None else through_seq
            if type(cutoff) is not int or not 0<=cutoff<=self._journal._last_seq:raise AdapterError('AUDIT_HOLD_INVALID')
            prefix=self._journal._con.execute('SELECT content_sha256 FROM records WHERE seq=?',(cutoff,)).fetchone()
            invalid_ids=set();mutations=0
            for event_id,raw in self._journal._con.execute("SELECT id,payload_json FROM records WHERE kind='DECISION' AND seq<=? AND json_extract(payload_json,'$.data.scope')=? AND json_extract(payload_json,'$.data.seal_sha256')=?",(cutoff,SCOPE,seal_sha256)):
                mutations+=1
                if not _valid(json.loads(raw)['data']):invalid_ids.add(event_id)
            common="""WITH events AS (SELECT seq,id,json_extract(payload_json,'$.data.operation') AS op,
                json_extract(payload_json,'$.data.hold_event_id') AS hold_id,
                json_extract(payload_json,'$.data.seal_sha256') AS seal FROM records
                WHERE kind='DECISION' AND json_extract(payload_json,'$.data.scope')=? AND seq<=?
                    AND id NOT IN (SELECT value FROM json_each(?))),
                holds AS (SELECT p.id,p.seq,COUNT(r.id) AS releases FROM events p LEFT JOIN events r
                    ON r.op='RELEASE' AND r.hold_id=p.id AND r.seq>p.seq AND r.seal=p.seal
                    WHERE p.op='PLACE' AND p.seal=? GROUP BY p.id,p.seq) """
            args=(SCOPE,cutoff,_json(sorted(invalid_ids)),seal_sha256)
            placed,active,released,_conflicts=self._journal._con.execute(common+"""SELECT COUNT(*),
                COALESCE(SUM(releases!=1),0),COALESCE(SUM(releases=1),0),COALESCE(SUM(releases>1),0) FROM holds""",args).fetchone()
            bad_references=self._journal._con.execute(common+"""SELECT r.id FROM events r LEFT JOIN events p
                ON p.id=r.hold_id AND p.op='PLACE' AND p.seq<r.seq AND p.seal=r.seal
                WHERE r.op='RELEASE' AND r.seal=? AND p.id IS NULL""",args+(seal_sha256,))
            invalid_ids.update(row[0] for row in bad_references)
            duplicates=self._journal._con.execute(common+"SELECT r.id FROM events r JOIN holds h ON h.id=r.hold_id WHERE r.op='RELEASE' AND h.releases>1 AND r.seal=? AND r.seq>h.seq",args+(seal_sha256,))
            invalid_ids.update(row[0] for row in duplicates)
            invalid=len(invalid_ids)
            ids=[row[0] for row in self._journal._con.execute(common+'SELECT id FROM holds WHERE releases!=1 ORDER BY seq LIMIT 100',args)]
            result=dict(seal_sha256=seal_sha256,prefix_last_seq=cutoff,prefix_chain_tip=prefix[0] if prefix else '0'*64,
                mutations_N=mutations,placed_N=placed,active_n=active,released_n=released,invalid_n=invalid,active_ids=ids,
                returned_n=len(ids),truncated=active>len(ids),state='FAIL' if invalid else 'BLOCKED' if active else 'UNKNOWN',
                certification='UNAVAILABLE',deletion_eligible=False,denominator_basis='DURABLE_HOLD_PREFIX')
            return dict(result,report_sha256=_sha(result),ledger_state=self._journal._state,
                failure_reason=self._journal._failure_reason)

    def close(self):self._journal.close()
