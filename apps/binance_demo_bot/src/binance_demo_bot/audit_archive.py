"""Opt-in frozen audit exports. Preserve all evidence; never reclaim the journal."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import time
import uuid
import zlib

from .adapter import AdapterError
from .interaction_audit import AuditJournal, KINDS, MAX_EVENT_BYTES, _json, _sha
from .isolation import AppPaths
from .journal import _WriterLease, WriterBusy
from . import market_sqlite as sqlite3


MIB=1024*1024
MANIFEST_LIMIT=65536
LINE_LIMIT=MAX_EVENT_BYTES+1024
FILES={'records.jsonl.gz','manifest.json'}
IDENTITY_KEYS={'application_id','environment','objective_contract_id','schema_version',
    'account_alias','account_episode_id','portfolio_scope_id'}
FORMAT='AUDIT_JSONL_GZIP_V1'
POLICY='PRESERVE_ALL_V1'
MANIFEST_KEYS=set('''format archive_id policy_version identity created_at_ms source_run_id
    source_state_at_pin exporter_code_sha256 compressed_sha256 compressed_bytes_n
    uncompressed_sha256 uncompressed_bytes_n denominator_basis capture_coverage
    full_audit_complete first_seq last_seq rows_n chain_tip kind_counts min_wall_ms
    max_wall_ms manifest_sha256'''.split())


def _identifier(value):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{32}',value) is None:
        raise AdapterError('AUDIT_ARCHIVE_INVALID')
    return value


def _digest(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
def _number(value):return type(value) is int and value>=0
def _link(path):return path.is_symlink() or path.is_junction()


class _LimitedWriter:
    def __init__(self,stream,limit):self.stream,self.limit,self.count=stream,limit,0
    def write(self,data):
        if self.count+len(data)>self.limit:raise AdapterError('AUDIT_ARCHIVE_LIMIT')
        size=self.stream.write(data)
        if size!=len(data):raise OSError('short archive write')
        self.count+=size
        return size
    def flush(self):return self.stream.flush()


class _Summary:
    def __init__(self,identity):
        self.identity=identity;self.count=0;self.tip='0'*64;self.kinds=Counter()
        self.low=self.high=None

    def accept(self,row):
        if not isinstance(row,dict):raise ValueError()
        value=dict(row);digest=value.pop('content_sha256')
        if (type(value['seq']) is not int or value['seq']!=self.count+1 or value['kind'] not in KINDS
            or not isinstance(value['data'],dict) or not _digest(digest) or _sha(value)!=digest
            or value['previous_sha256']!=self.tip or not _digest(value['source_code_sha256'])
            or not _number(value['wall_ms']) or not _number(value['monotonic_ns'])
            or any(value.get(key)!=v for key,v in self.identity.items())):raise ValueError()
        for key in ('event_id','run_id','process_instance_id'):_identifier(value[key])
        self.count+=1;self.tip=digest;self.kinds[value['kind']]+=1
        wall=value['wall_ms']
        self.low=wall if self.low is None else min(self.low,wall)
        self.high=wall if self.high is None else max(self.high,wall)

    def fields(self):
        return dict(first_seq=1 if self.count else None,last_seq=self.count,rows_n=self.count,
            chain_tip=self.tip,kind_counts={kind:self.kinds[kind] for kind in KINDS},
            min_wall_ms=self.low,max_wall_ms=self.high)


class AuditArchive:
    def __init__(self,paths,*,max_compressed_bytes=64*MIB,max_uncompressed_bytes=512*MIB,
                 max_total_bytes=512*MIB,max_archives=64,fault_hook=None):
        if not isinstance(paths,AppPaths):raise AdapterError('AUDIT_ARCHIVE_IDENTITY')
        for value in (max_compressed_bytes,max_uncompressed_bytes,max_total_bytes):
            if type(value) is not int or not 1<=value<=512*MIB:raise AdapterError('AUDIT_ARCHIVE_INVALID')
        if type(max_archives) is not int or not 1<=max_archives<=1024:
            raise AdapterError('AUDIT_ARCHIVE_INVALID')
        self.paths=paths
        self.max_compressed_bytes,self.max_uncompressed_bytes=max_compressed_bytes,max_uncompressed_bytes
        self.max_total_bytes,self.max_archives=max_total_bytes,max_archives
        self.fault_hook=fault_hook
        self._root().mkdir(parents=True,exist_ok=True)

    def _root(self):
        try:
            lexical=self.paths.root/'.runtime'/'audit-archives'
            if _link(lexical):raise ValueError()
            return self.paths.runtime_path('audit-archives')
        except (ValueError,OSError):raise AdapterError('AUDIT_ARCHIVE_LAYOUT') from None

    def _directory(self,identifier):
        _identifier(identifier)
        root=self._root();path=root/identifier
        if _link(path) or path.resolve().parent!=root:raise AdapterError('AUDIT_ARCHIVE_LAYOUT')
        return path

    def _hook(self,phase):
        if self.fault_hook:self.fault_hook(phase)

    def _scan(self):
        root=self._root();entries=list(islice(root.iterdir(),self.max_archives+3))
        if len(entries)>self.max_archives+1:raise AdapterError('AUDIT_ARCHIVE_LIMIT')
        directories=[];total=invalid=0
        for path in entries:
            if _link(path):raise AdapterError('AUDIT_ARCHIVE_LAYOUT')
            if path.name=='.writer.lock' and path.is_file():
                total+=path.stat().st_size;continue
            valid=re.fullmatch(r'(?:\.pending-)?[0-9a-f]{32}',path.name) is not None
            if not path.is_dir() or not valid:
                invalid+=1
                if path.is_file():total+=path.stat().st_size
                continue
            children=list(islice(path.iterdir(),4))
            for child in children:
                if _link(child) or not child.is_file():raise AdapterError('AUDIT_ARCHIVE_LAYOUT')
                total+=child.stat().st_size
                if child.name not in FILES:invalid+=1
            directories.append(path)
        return directories,total,invalid

    @contextmanager
    def _snapshot(self,journal,through_seq):
        if not isinstance(journal,AuditJournal):raise AdapterError('AUDIT_ARCHIVE_IDENTITY')
        try:
            runtime=self.paths.runtime_path('.')
            if not journal.path.resolve().is_relative_to(runtime) or _link(journal.path):
                raise AdapterError('AUDIT_ARCHIVE_IDENTITY')
        except ValueError:raise AdapterError('AUDIT_ARCHIVE_IDENTITY') from None
        connection=None
        try:
            with journal._lock:
                if journal._closed:raise AdapterError('AUDIT_ARCHIVE_SOURCE_CLOSED')
                connection=sqlite3.connect(journal.path.as_uri()+'?mode=ro',uri=True)
                connection.execute('BEGIN')
                identity=json.loads(connection.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
                if identity!=journal.identity:raise AdapterError('AUDIT_ARCHIVE_IDENTITY')
                last=connection.execute('SELECT COALESCE(MAX(seq),0) FROM records').fetchone()[0]
                cutoff=last if through_seq is None else through_seq
                if type(cutoff) is not int or not 0<=cutoff<=last:raise AdapterError('AUDIT_ARCHIVE_INVALID')
                with journal._metrics_lock:state=journal._state
                run=journal.run_id
            yield connection,identity,cutoff,state,run
        finally:
            if connection is not None:connection.close()

    def _file_hash(self,path):
        digest=hashlib.sha256();size=0
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(65536),b''):
                size+=len(chunk)
                if size>self.max_compressed_bytes:raise AdapterError('AUDIT_ARCHIVE_LIMIT')
                digest.update(chunk)
        return digest.hexdigest(),size

    def export(self,journal,*,through_seq=None,archive_id=None):
        identifier=_identifier(uuid.uuid4().hex if archive_id is None else archive_id)
        final=self._directory(identifier);pending=self._root()/('.pending-'+identifier)
        lease=None;published=False
        try:
            lock_path=self._root()/'.writer.lock'
            if _link(lock_path) or (lock_path.exists() and not lock_path.is_file()):
                raise AdapterError('AUDIT_ARCHIVE_LAYOUT')
            lease=_WriterLease(lock_path)
            if final.exists() or pending.exists():raise AdapterError('AUDIT_ARCHIVE_EXISTS')
            directories,used,invalid=self._scan()
            if invalid:raise AdapterError('AUDIT_ARCHIVE_LAYOUT')
            if len(directories)>=self.max_archives or used+MANIFEST_LIMIT>=self.max_total_bytes:
                raise AdapterError('AUDIT_ARCHIVE_LIMIT')
            with self._snapshot(journal,through_seq) as (connection,identity,cutoff,state,run):
                self._hook('snapshot_pinned')
                pending.mkdir()
                summary=_Summary(identity);plain_hash=hashlib.sha256();plain_size=0
                limit=min(self.max_compressed_bytes,self.max_total_bytes-used-MANIFEST_LIMIT)
                with (pending/'records.jsonl.gz').open('xb') as target:
                    writer=_LimitedWriter(target,limit)
                    with gzip.GzipFile(filename='',fileobj=writer,mode='wb',mtime=0) as compressed:
                        query='SELECT seq,id,kind,payload_json,content_sha256 FROM records WHERE seq<=? ORDER BY seq'
                        for seq,event_id,kind,payload,digest in connection.execute(query,(cutoff,)):
                            if len(payload.encode())>MAX_EVENT_BYTES:raise AdapterError('AUDIT_ARCHIVE_LIMIT')
                            value=json.loads(payload)
                            if (not isinstance(value,dict) or value.get('seq')!=seq
                                or value.get('event_id')!=event_id or value.get('kind')!=kind):
                                raise AdapterError('AUDIT_ARCHIVE_INTEGRITY')
                            row=dict(value,content_sha256=digest)
                            try:summary.accept(row)
                            except (ValueError,KeyError,TypeError,AdapterError):
                                raise AdapterError('AUDIT_ARCHIVE_INTEGRITY') from None
                            line=(_json(row)+'\n').encode()
                            plain_size+=len(line)
                            if len(line)>LINE_LIMIT or plain_size>self.max_uncompressed_bytes:
                                raise AdapterError('AUDIT_ARCHIVE_LIMIT')
                            plain_hash.update(line);compressed.write(line)
                    target.flush();os.fsync(target.fileno())
                if summary.count!=cutoff:raise AdapterError('AUDIT_ARCHIVE_INTEGRITY')
                compressed_hash,compressed_size=self._file_hash(pending/'records.jsonl.gz')
                manifest=dict(format=FORMAT,archive_id=identifier,policy_version=POLICY,
                    identity=identity,created_at_ms=time.time_ns()//1000000,source_run_id=run,
                    source_state_at_pin=state,exporter_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    compressed_sha256=compressed_hash,compressed_bytes_n=compressed_size,
                    uncompressed_sha256=plain_hash.hexdigest(),uncompressed_bytes_n=plain_size,
                    denominator_basis='DURABLE_EXPORTED_PREFIX',capture_coverage='UNKNOWN',
                    full_audit_complete=False,**summary.fields())
                manifest['manifest_sha256']=_sha(manifest)
                raw=_json(manifest).encode()
                if len(raw)>MANIFEST_LIMIT:raise AdapterError('AUDIT_ARCHIVE_LIMIT')
                with (pending/'manifest.json').open('xb') as target:
                    target.write(raw);target.flush();os.fsync(target.fileno())
            self._verify_directory(pending,identifier)
            self._hook('before_publish')
            pending.rename(final);published=True
            self._hook('after_publish')
            return self.verify(identifier)
        except WriterBusy:raise AdapterError('AUDIT_ARCHIVE_BUSY') from None
        except BaseException as error:
            if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
            if published:raise AdapterError('AUDIT_ARCHIVE_PUBLICATION_UNKNOWN') from None
            if isinstance(error,AdapterError):raise
            reason='AUDIT_ARCHIVE_INTEGRITY' if isinstance(error,(ValueError,KeyError,TypeError,sqlite3.Error)) else 'AUDIT_ARCHIVE_WRITE_FAILED'
            raise AdapterError(reason) from None
        finally:
            if lease is not None:lease.close()

    def _verify_directory(self,path,identifier):
        try:
            if _link(path) or not path.is_dir():raise AdapterError('AUDIT_ARCHIVE_LAYOUT')
            children=list(islice(path.iterdir(),3))
            if {p.name for p in children}!=FILES:raise AdapterError('AUDIT_ARCHIVE_LAYOUT')
            for child in children:
                if _link(child) or not child.is_file():raise AdapterError('AUDIT_ARCHIVE_LAYOUT')
            with (path/'manifest.json').open('rb') as stream:raw=stream.read(MANIFEST_LIMIT+1)
            if len(raw)>MANIFEST_LIMIT:raise AdapterError('AUDIT_ARCHIVE_LIMIT')
            manifest=json.loads(raw)
            if not isinstance(manifest,dict) or set(manifest)!=MANIFEST_KEYS:raise ValueError()
            digest=manifest['manifest_sha256'];value=dict(manifest);value.pop('manifest_sha256')
            if not _digest(digest) or _sha(value)!=digest:raise ValueError()
            identity=manifest['identity']
            if (not isinstance(identity,dict) or set(identity)!=IDENTITY_KEYS
                or identity['application_id']!='binance_demo_bot' or identity['environment']!='BINANCE_SPOT_DEMO'
                or identity['objective_contract_id']!='daily_net_equity_pnl_v1' or type(identity['schema_version']) is not int
                or identity['schema_version']!=1 or any(not isinstance(identity[k],str) or not identity[k]
                    for k in ('account_alias','account_episode_id','portfolio_scope_id'))):raise ValueError()
            if (manifest['format']!=FORMAT or manifest['archive_id']!=identifier or manifest['policy_version']!=POLICY
                or manifest['denominator_basis']!='DURABLE_EXPORTED_PREFIX' or manifest['capture_coverage']!='UNKNOWN'
                or manifest['full_audit_complete'] is not False or not _number(manifest['created_at_ms'])
                or not _digest(manifest['exporter_code_sha256']) or manifest['source_state_at_pin'] not in ('READY','FAILED')):
                raise ValueError()
            _identifier(manifest['source_run_id'])
            for key in ('compressed_bytes_n','uncompressed_bytes_n','rows_n','last_seq'):
                if not _number(manifest[key]):raise ValueError()
            for key in ('first_seq','min_wall_ms','max_wall_ms'):
                if manifest[key] is not None and not _number(manifest[key]):raise ValueError()
            if (not isinstance(manifest['kind_counts'],dict) or set(manifest['kind_counts'])!=set(KINDS)
                or any(not _number(count) for count in manifest['kind_counts'].values())):raise ValueError()
            if manifest['uncompressed_bytes_n']>self.max_uncompressed_bytes:raise AdapterError('AUDIT_ARCHIVE_LIMIT')
            compressed_hash,compressed_size=self._file_hash(path/'records.jsonl.gz')
            if compressed_hash!=manifest['compressed_sha256'] or compressed_size!=manifest['compressed_bytes_n']:
                raise ValueError()
            summary=_Summary(identity);plain_hash=hashlib.sha256();plain_size=0
            with gzip.open(path/'records.jsonl.gz','rb') as stream:
                while True:
                    line=stream.readline(min(LINE_LIMIT,self.max_uncompressed_bytes-plain_size)+1)
                    if not line:break
                    plain_size+=len(line)
                    if len(line)>LINE_LIMIT or plain_size>self.max_uncompressed_bytes:
                        raise AdapterError('AUDIT_ARCHIVE_LIMIT')
                    if not line.endswith(b'\n'):raise ValueError()
                    plain_hash.update(line);summary.accept(json.loads(line))
            if (plain_size!=manifest['uncompressed_bytes_n'] or plain_hash.hexdigest()!=manifest['uncompressed_sha256']
                or any(manifest[key]!=v for key,v in summary.fields().items())):raise ValueError()
            return deepcopy(manifest)
        except AdapterError as error:
            if error.reason in ('AUDIT_ARCHIVE_LIMIT','AUDIT_ARCHIVE_LAYOUT'):raise
            raise AdapterError('AUDIT_ARCHIVE_INTEGRITY') from None
        except (OSError,ValueError,KeyError,TypeError,RecursionError,EOFError,zlib.error):
            raise AdapterError('AUDIT_ARCHIVE_INTEGRITY') from None

    def verify(self,archive_id,*,expected_identity=None,expected_manifest_sha256=None):
        manifest=self._verify_directory(self._directory(archive_id),archive_id)
        if expected_identity is not None and manifest['identity']!=expected_identity:
            raise AdapterError('AUDIT_ARCHIVE_IDENTITY')
        if expected_manifest_sha256 is not None and manifest['manifest_sha256']!=expected_manifest_sha256:
            raise AdapterError('AUDIT_ARCHIVE_PIN_MISMATCH')
        return dict(archive_id=archive_id,manifest=manifest,integrity='INTERNAL_VERIFIED',
            authenticity='UNVERIFIED' if expected_manifest_sha256 is None else 'PINNED_MANIFEST',
            full_audit_complete=False)

    def inventory(self,*,protected_archive_ids=()):
        if not isinstance(protected_archive_ids,(list,tuple)) or len(protected_archive_ids)>self.max_archives:
            raise AdapterError('AUDIT_ARCHIVE_INVALID')
        protected={_identifier(identifier) for identifier in protected_archive_ids}
        if len(protected)!=len(protected_archive_ids):raise AdapterError('AUDIT_ARCHIVE_INVALID')
        directories,total,layout_errors=self._scan();entries=[];valid=invalid=incomplete=0;present=set()
        for path in sorted(directories):
            if path.name.startswith('.pending-'):
                incomplete+=1;entries.append(dict(archive_id=path.name[9:],state='INCOMPLETE'));continue
            present.add(path.name)
            try:
                result=self.verify(path.name);valid+=1
                entries.append(dict(archive_id=path.name,state='INTERNAL_VERIFIED',
                    manifest_sha256=result['manifest']['manifest_sha256'],protected=path.name in protected))
            except AdapterError as error:
                invalid+=1;entries.append(dict(archive_id=path.name,state='INVALID',error=error.reason))
        missing=sorted(protected-present)
        state=('FAIL' if invalid or layout_errors or total>self.max_total_bytes else
            'UNKNOWN' if incomplete or missing or not valid else 'VALID_PRESERVED')
        return dict(policy_version=POLICY,state=state,archives_expected_N=len(directories),valid_n=valid,
            invalid_n=invalid,incomplete_n=incomplete,layout_errors_n=layout_errors,
            stored_bytes_n=None if layout_errors else total,observed_file_bytes_n=total,
            max_total_bytes_N=self.max_total_bytes,
            protected_expected_N=len(protected),protected_present_n=len(protected)-len(missing),
            protected_missing_ids=missing,deletion_eligible_n=0,entries=entries,
            capture_coverage='UNKNOWN',full_audit_complete=False)
