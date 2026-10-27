"""Explicit quiesced rollover. Preserve all segments and ambiguity on restart."""
from contextlib import contextmanager
import json
import os
import re
import uuid

from .adapter import AdapterError
from .audit_archive import AuditArchive, _Summary, _identifier, _link, _digest
from .interaction_audit import AuditJournal, _json, _sha, MAX_EVENT_BYTES
from .isolation import AppPaths
from .journal import _WriterLease, WriterBusy
from . import market_sqlite as sqlite3


BASE_KEYS={'version','database','next_database','archive_id','archive_manifest_sha256',
    'prefix_last_seq','prefix_chain_tip','journal_limits'}
SEAL_KEYS=BASE_KEYS|{'seq','event_id','chain_tip','seal_sha256'}


def _database(paths,database):
    if (not isinstance(database,str) or (database!='audit.sqlite'
        and re.fullmatch(r'audit-segments/[0-9a-f]{32}\.sqlite',database) is None)):
        raise AdapterError('AUDIT_ROLLOVER_INVALID')
    try:
        lexical=paths.root/'.runtime'/database
        if _link(lexical):raise ValueError()
        return paths.runtime_path(database)
    except (ValueError,OSError):raise AdapterError('AUDIT_ROLLOVER_INVALID') from None


def _base(paths,value):
    if not isinstance(value,dict) or set(value)!=BASE_KEYS:raise ValueError()
    if type(value['version']) is not int or value['version']!=1:raise ValueError()
    _database(paths,value['database']);_database(paths,value['next_database'])
    if value['next_database']=='audit.sqlite' or value['next_database']==value['database']:raise ValueError()
    _identifier(value['archive_id'])
    if (type(value['prefix_last_seq']) is not int or value['prefix_last_seq']<1
        or not _digest(value['prefix_chain_tip']) or not _digest(value['archive_manifest_sha256'])):raise ValueError()
    limits=value['journal_limits']
    if (not isinstance(limits,dict) or set(limits)!={'max_bytes','max_inflight'}
        or type(limits['max_bytes']) is not int or limits['max_bytes']<65536
        or type(limits['max_inflight']) is not int or not 1<=limits['max_inflight']<=8):raise ValueError()


def _check_receipt(receipt,seal,first):
    expected=dict(version=1,next_database=seal['next_database'],seal_sha256=seal['seal_sha256'],
        first_record_sha256=first['content_sha256'] if first else None)
    expected['receipt_sha256']=_sha(expected)
    if receipt!=expected:raise AdapterError('AUDIT_ROLLOVER_INTEGRITY')


def _read_source(paths,database,identity,archives,*,immutable=False):
    path=_database(paths,database)
    if not path.is_file() or path.stat().st_size==0:raise AdapterError('AUDIT_ROLLOVER_MISSING')
    connection=None
    try:
        connection=sqlite3.connect(path.as_uri()+'?mode=ro'+('&immutable=1' if immutable else ''),uri=True);connection.execute('BEGIN')
        metadata={}
        for key,raw in connection.execute('SELECT key,value FROM meta'):
            if len(raw)>65536:raise ValueError()
            metadata[key]=json.loads(raw)
            if len(metadata)>4:raise ValueError()
        if metadata.get('identity')!=identity:raise ValueError()
        summary=_Summary(identity);first=last=None
        for seq,event_id,kind,raw,digest in connection.execute('SELECT seq,id,kind,payload_json,content_sha256 FROM records ORDER BY seq'):
            if len(raw.encode())>MAX_EVENT_BYTES:raise ValueError()
            value=json.loads(raw)
            if not isinstance(value,dict) or (value.get('seq'),value.get('event_id'),value.get('kind'))!=(seq,event_id,kind):
                raise ValueError()
            row=dict(value,content_sha256=digest);summary.accept(row)
            if first is None:first=row
            last=row
        parent=metadata.get('parent_segment')
        if parent is None and (database.startswith('audit-segments/')
            or (first is not None and 'parent_segment' in first['data'])):raise ValueError()
        if parent is not None and first is not None and first['data'].get('parent_segment')!=parent:raise ValueError()
        seal=metadata.get('sealed')
        if seal is None and connection.execute("SELECT 1 FROM records WHERE kind='RUN_END' AND json_extract(payload_json,'$.data.scope')='SEGMENT_SEAL' LIMIT 1").fetchone():
            raise ValueError()
        if seal is not None:
            if not isinstance(seal,dict) or set(seal)!=SEAL_KEYS:raise ValueError()
            base={key:seal[key] for key in BASE_KEYS};_base(paths,base)
            content=dict(seal);digest=content.pop('seal_sha256')
            if (not _digest(digest) or _sha(content)!=digest or seal['database']!=database
                or type(seal['seq']) is not int or seal['seq']!=seal['prefix_last_seq']+1
                or last is None or last['kind']!='RUN_END' or last['seq']!=seal['seq']
                or last['event_id']!=seal['event_id'] or last['content_sha256']!=seal['chain_tip']
                or last['previous_sha256']!=seal['prefix_chain_tip']
                or last['data']!={'scope':'SEGMENT_SEAL','rollover':base}):raise ValueError()
            archived=archives.verify(seal['archive_id'],expected_identity=identity,
                expected_manifest_sha256=seal['archive_manifest_sha256'])['manifest']
            if archived['last_seq']!=seal['prefix_last_seq'] or archived['chain_tip']!=seal['prefix_chain_tip']:
                raise ValueError()
        return dict(metadata=metadata,first=first,last=last)
    except AdapterError as error:
        if error.reason=='AUDIT_ROLLOVER_MISSING':raise
        raise AdapterError('AUDIT_ROLLOVER_INTEGRITY') from None
    except (ValueError,KeyError,TypeError,OSError,RecursionError,sqlite3.Error):
        raise AdapterError('AUDIT_ROLLOVER_INTEGRITY') from None
    finally:
        if connection is not None:connection.close()


def validate_parent(paths,parent,database,identity,archives=None):
    """Also called by ordinary journal startup before any RUN_BEGIN write."""
    archives=archives or AuditArchive(paths)
    visited=set()
    try:
        for _ in range(64):
            if not isinstance(parent,dict) or parent.get('next_database')!=database:raise ValueError()
            previous=parent['database']
            if previous in visited:raise ValueError()
            visited.add(previous)
            source=_read_source(paths,previous,identity,archives)
            if source['metadata'].get('sealed')!=parent:raise ValueError()
            receipt=source['metadata'].get('successor_opened')
            if receipt is not None:
                child=_read_source(paths,database,identity,archives)
                _check_receipt(receipt,parent,child['first'])
            database,parent=previous,source['metadata'].get('parent_segment')
            if parent is None:return len(visited)
        raise ValueError()
    except (ValueError,KeyError,TypeError,AdapterError):
        raise AdapterError('AUDIT_ROLLOVER_INTEGRITY') from None


class AuditRollover:
    def __init__(self,paths,*,account_alias,episode_id,scope_id,archives=None,max_segments=64,fault_hook=None):
        if not isinstance(paths,AppPaths):raise AdapterError('AUDIT_ROLLOVER_IDENTITY')
        if type(max_segments) is not int or not 1<=max_segments<=64:raise AdapterError('AUDIT_ROLLOVER_INVALID')
        if any(not isinstance(v,str) or not v for v in (account_alias,episode_id,scope_id)):
            raise AdapterError('AUDIT_ROLLOVER_IDENTITY')
        self.paths,self.max_segments,self.fault_hook=paths,max_segments,fault_hook
        self.arguments=dict(account_alias=account_alias,episode_id=episode_id,scope_id=scope_id)
        self.identity=dict(application_id='binance_demo_bot',environment='BINANCE_SPOT_DEMO',
            objective_contract_id='daily_net_equity_pnl_v1',schema_version=1,account_alias=account_alias,
            account_episode_id=episode_id,portfolio_scope_id=scope_id)
        self.archives=archives or AuditArchive(paths)
        if not isinstance(self.archives,AuditArchive) or self.archives.paths.root!=paths.root:
            raise AdapterError('AUDIT_ROLLOVER_IDENTITY')

    def _hook(self,phase):
        if self.fault_hook:self.fault_hook(phase)

    @contextmanager
    def _owner(self):
        lease=None
        try:
            try:
                lexical=self.paths.root/'.runtime'/'audit-rollover.writer.lock'
                if _link(lexical):raise ValueError()
                path=self.paths.runtime_path('audit-rollover.writer.lock');path.parent.mkdir(parents=True,exist_ok=True)
                lease=_WriterLease(path)
            except WriterBusy:raise AdapterError('AUDIT_ROLLOVER_BUSY') from None
            except (ValueError,OSError):raise AdapterError('AUDIT_ROLLOVER_INVALID') from None
            try:yield
            except WriterBusy:raise AdapterError('AUDIT_ROLLOVER_BUSY') from None
            except AdapterError:raise
            except Exception:raise AdapterError('AUDIT_ROLLOVER_TRANSITION_UNKNOWN') from None
        finally:
            if lease is not None:lease.close()

    def _child(self,seal):
        path=_database(self.paths,seal['next_database'])
        if not path.is_file():raise AdapterError('AUDIT_ROLLOVER_MISSING')
        parent=_read_source(self.paths,seal['database'],self.identity,self.archives)
        receipt=parent['metadata'].get('successor_opened')
        if path.stat().st_size==0:
            if receipt is not None:raise AdapterError('AUDIT_ROLLOVER_MISSING')
        else:
            existing=_read_source(self.paths,seal['next_database'],self.identity,self.archives)
            if existing['metadata'].get('parent_segment')!=seal:raise AdapterError('AUDIT_ROLLOVER_INTEGRITY')
            if receipt is not None:_check_receipt(receipt,seal,existing['first'])
        child=None
        try:
            child=AuditJournal(self.paths,**self.arguments,**seal['journal_limits'],
                database=seal['next_database'],parent_segment=seal)
            self._hook('after_child_open')
            self._activate(seal,child.read(limit=1)[0])
            self._hook('after_activation')
            return child
        except BaseException:
            if child is not None:child.close()
            raise

    def _activate(self,seal,first):
        parent=_read_source(self.paths,seal['database'],self.identity,self.archives)
        receipt=parent['metadata'].get('successor_opened')
        if receipt is not None:
            _check_receipt(receipt,seal,first);return
        receipt=dict(version=1,next_database=seal['next_database'],seal_sha256=seal['seal_sha256'],
            first_record_sha256=first['content_sha256'])
        receipt['receipt_sha256']=_sha(receipt)
        path=_database(self.paths,seal['database']);lease=None;connection=None
        try:
            lease=_WriterLease(self.paths.runtime_path(seal['database']+'.writer.lock'))
            connection=sqlite3.connect(path);connection.execute('PRAGMA synchronous=FULL')
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('INSERT INTO meta VALUES (?,?)',('successor_opened',_json(receipt)))
            connection.commit()
        except (WriterBusy,OSError,sqlite3.Error):raise AdapterError('AUDIT_ROLLOVER_ACTIVATION_UNKNOWN') from None
        finally:
            if connection is not None:connection.close()
            if lease is not None:lease.close()

    def rotate(self,journal,*,quiesced=False):
        if quiesced is not True:raise AdapterError('AUDIT_ROLLOVER_QUIESCENCE')
        if not isinstance(journal,AuditJournal) or journal.identity!=self.identity:
            raise AdapterError('AUDIT_ROLLOVER_IDENTITY')
        try:database=journal.path.relative_to(self.paths.runtime_path('.')).as_posix()
        except ValueError:raise AdapterError('AUDIT_ROLLOVER_IDENTITY') from None
        _database(self.paths,database)
        with self._owner(),journal._lock:
            if journal._closed or journal._state!='READY':raise AdapterError('AUDIT_ROLLOVER_SOURCE')
            health=journal.health();processing=journal.processing_report()
            if (any(health[k] for k in ('unresolved_attempts_n','unresolved_ws_operations_n','incomplete_prior_runs_n'))
                or any(c['gap_n'] for c in health['channels'].values())
                or any(processing[k] for k in ('unknown_n','conflict_n','invalid_marker_n'))):
                raise AdapterError('AUDIT_ROLLOVER_UNRESOLVED')
            source=_read_source(self.paths,database,self.identity,self.archives)
            parent=source['metadata'].get('parent_segment')
            depth=1 if parent is None else 1+validate_parent(self.paths,parent,database,self.identity,self.archives)
            if depth>=self.max_segments:raise AdapterError('AUDIT_ROLLOVER_LIMIT')
            result=self.archives.export(journal);manifest=result['manifest']
            self.archives.verify(result['archive_id'],expected_identity=self.identity,
                expected_manifest_sha256=manifest['manifest_sha256'])
            next_database='audit-segments/'+uuid.uuid4().hex+'.sqlite'
            target=_database(self.paths,next_database);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as reserved:reserved.flush();os.fsync(reserved.fileno())
            self._hook('after_reserve')
            seal=journal._seal_segment(dict(version=1,database=database,next_database=next_database,
                archive_id=result['archive_id'],archive_manifest_sha256=manifest['manifest_sha256'],
                prefix_last_seq=manifest['last_seq'],prefix_chain_tip=manifest['chain_tip'],
                journal_limits=dict(max_bytes=journal.max_bytes,max_inflight=journal.max_inflight)))
            self._hook('after_seal')
            return self._child(seal)

    def recover(self,*,database='audit.sqlite'):
        _database(self.paths,database)
        with self._owner():
            visited=set();parent=None
            for _ in range(self.max_segments):
                if database in visited:raise AdapterError('AUDIT_ROLLOVER_INTEGRITY')
                visited.add(database)
                source=_read_source(self.paths,database,self.identity,self.archives)
                if source['metadata'].get('parent_segment') is not None:
                    validate_parent(self.paths,source['metadata']['parent_segment'],database,self.identity,self.archives)
                seal=source['metadata'].get('sealed')
                if seal is None:
                    if parent is not None:return self._child(parent)
                    return AuditJournal(self.paths,**self.arguments,database=database)
                if len(visited)>=self.max_segments:raise AdapterError('AUDIT_ROLLOVER_LIMIT')
                target=_database(self.paths,seal['next_database'])
                if not target.is_file():raise AdapterError('AUDIT_ROLLOVER_MISSING')
                if target.stat().st_size==0:return self._child(seal)
                child=_read_source(self.paths,seal['next_database'],self.identity,self.archives)
                if child['metadata'].get('parent_segment')!=seal:raise AdapterError('AUDIT_ROLLOVER_INTEGRITY')
                receipt=source['metadata'].get('successor_opened')
                if receipt is not None:_check_receipt(receipt,seal,child['first'])
                parent,database=seal,seal['next_database']
            raise AdapterError('AUDIT_ROLLOVER_LIMIT')
