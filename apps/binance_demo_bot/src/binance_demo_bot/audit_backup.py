"""Complete sealed-segment snapshots. No deletion or implicit runtime activation."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import time
import uuid

from .adapter import AdapterError
from .audit_archive import AuditArchive, _Summary, _identifier, _digest, _link
from .audit_rollover import AuditRollover, _database, _read_source, _check_receipt
from .interaction_audit import _json, _sha
from .isolation import AppPaths
from .journal import _WriterLease, WriterBusy
from . import market_sqlite as sqlite3


MIB=1024*1024
MANIFEST_LIMIT=65536
FORMAT='SEALED_AUDIT_BUNDLE_V1'
KEYS=set('''format backup_id identity database source_metadata successor_first records_n
    chain_tip scope files created_at_ms exporter_code_sha256 full_audit_complete manifest_sha256'''.split())


def _id(value):
    try:return _identifier(value)
    except AdapterError:raise AdapterError('AUDIT_BACKUP_INVALID') from None


def _safe(paths,relative):
    try:
        path=paths.root/'.runtime'
        if _link(path):raise ValueError()
        for part in Path(relative).parts:
            path=path/part
            if _link(path):raise ValueError()
        return paths.runtime_path(relative)
    except (ValueError,OSError):raise AdapterError('AUDIT_BACKUP_LAYOUT') from None


@contextmanager
def _lease(paths,relative):
    lease=None
    try:
        path=_safe(paths,relative);path.parent.mkdir(parents=True,exist_ok=True)
        lease=_WriterLease(path);yield
    except WriterBusy:raise AdapterError('AUDIT_BACKUP_BUSY') from None
    finally:
        if lease is not None:lease.close()


def _schema(path):
    """Only the app's fixed tables/triggers/indexes, before reading payload SQL."""
    connection=None
    try:
        connection=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
        connection.execute('PRAGMA trusted_schema=OFF')
        expected={
            ('table','meta'):'CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)',
            ('table','records'):'CREATE TABLE records (seq INTEGER PRIMARY KEY,id TEXT UNIQUE NOT NULL,kind TEXT NOT NULL,payload_json TEXT NOT NULL,content_sha256 TEXT NOT NULL)',
            ('index','audit_attempt'):"CREATE INDEX audit_attempt ON records(kind,json_extract(payload_json,'$.data.attempt_id'))",
            ('index','audit_ws_operation'):"CREATE INDEX audit_ws_operation ON records(kind,json_extract(payload_json,'$.data.operation_id'))",
        }
        for table in ('meta','records'):
            for operation in ('UPDATE','DELETE'):
                name=f'immutable_{table}_{operation}'
                expected[('trigger',name)]=f"CREATE TRIGGER {name} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT,'immutable audit'); END"
        observed={}
        for kind,name,sql in connection.execute('SELECT type,name,sql FROM sqlite_master'):
            if sql is None and kind=='index' and name in ('sqlite_autoindex_meta_1','sqlite_autoindex_records_1'):continue
            observed[(kind,name)]=' '.join(sql.split()).upper() if isinstance(sql,str) else None
        if observed!={key:' '.join(value.split()).upper() for key,value in expected.items()}:raise ValueError()
    finally:
        if connection is not None:connection.close()


class SealedAuditBackup:
    def __init__(self,paths,*,account_alias,episode_id,scope_id,max_database_bytes=512*MIB,
                 max_total_bytes=1024*MIB,max_backups=16,fault_hook=None):
        if not isinstance(paths,AppPaths):raise AdapterError('AUDIT_BACKUP_IDENTITY')
        if (type(max_database_bytes) is not int or not 1<=max_database_bytes<=512*MIB
            or type(max_total_bytes) is not int or not 1<=max_total_bytes<=2048*MIB
            or type(max_backups) is not int or not 1<=max_backups<=64):raise AdapterError('AUDIT_BACKUP_INVALID')
        self.paths=paths;self.arguments=dict(account_alias=account_alias,episode_id=episode_id,scope_id=scope_id)
        self.rollover=AuditRollover(paths,**self.arguments);self.identity=self.rollover.identity
        self.max_database_bytes,self.max_total_bytes,self.max_backups=max_database_bytes,max_total_bytes,max_backups
        self.fault_hook=fault_hook;self._root().mkdir(parents=True,exist_ok=True)

    def _root(self):return _safe(self.paths,'audit-sealed-backups')
    def _directory(self,identifier):return _safe(self.paths,'audit-sealed-backups/'+_id(identifier))
    def _hook(self,phase):
        if self.fault_hook:self.fault_hook(phase)

    def _hash(self,path,limit):
        digest=hashlib.sha256();size=0
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(65536),b''):
                size+=len(chunk)
                if size>limit:raise AdapterError('AUDIT_BACKUP_LIMIT')
                digest.update(chunk)
        return dict(bytes_n=size,sha256=digest.hexdigest())

    def _usage(self,root=None):
        entries=list(islice((root or self._root()).iterdir(),self.max_backups+3));count=total=0
        if len(entries)>self.max_backups+1:raise AdapterError('AUDIT_BACKUP_LIMIT')
        for entry in entries:
            if _link(entry):raise AdapterError('AUDIT_BACKUP_LAYOUT')
            if entry.name=='.writer.lock' and entry.is_file():total+=entry.stat().st_size;continue
            name=entry.name.removeprefix('.pending-');_id(name)
            if not entry.is_dir():raise AdapterError('AUDIT_BACKUP_LAYOUT')
            count+=1;stack=[entry];nodes=0
            while stack:
                for path in stack.pop().iterdir():
                    nodes+=1
                    if nodes>20 or _link(path):raise AdapterError('AUDIT_BACKUP_LAYOUT')
                    if path.is_dir():stack.append(path)
                    elif path.is_file():total+=path.stat().st_size
                    else:raise AdapterError('AUDIT_BACKUP_LAYOUT')
        return count,total

    def _copy(self,source,target,limit):
        target.parent.mkdir(parents=True,exist_ok=True);size=0
        with source.open('rb') as reader,target.open('xb') as writer:
            for chunk in iter(lambda:reader.read(65536),b''):
                size+=len(chunk)
                if size>limit:raise AdapterError('AUDIT_BACKUP_LIMIT')
                if writer.write(chunk)!=len(chunk):raise OSError('short backup write')
            writer.flush();os.fsync(writer.fileno())

    def _sqlite_copy(self,source,target):
        reader=writer=backup=None
        try:
            reader=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True);reader.execute('BEGIN')
            pages=reader.execute('PRAGMA page_count').fetchone()[0]
            page_size=reader.execute('PRAGMA page_size').fetchone()[0]
            if pages*page_size>self.max_database_bytes:raise AdapterError('AUDIT_BACKUP_LIMIT')
            target.parent.mkdir(parents=True,exist_ok=True)
            writer=sqlite3.connect(target)
            backup=writer.native.backup('main',reader.native,'main')
            while not backup.step(64):
                if target.stat().st_size>self.max_database_bytes:raise AdapterError('AUDIT_BACKUP_LIMIT')
            backup.finish();backup=None
            writer.execute('PRAGMA journal_mode=DELETE')
        finally:
            if backup is not None:backup.finish()
            if writer is not None:writer.close()
            if reader is not None:reader.close()
        with target.open('r+b') as file:os.fsync(file.fileno())

    def _relative_files(self,database,archive_id):
        return ('snapshot/.runtime/'+database,
            'snapshot/.runtime/audit-archives/'+archive_id+'/manifest.json',
            'snapshot/.runtime/audit-archives/'+archive_id+'/records.jsonl.gz')

    def create(self,database='audit.sqlite',*,backup_id=None):
        identifier=_id(uuid.uuid4().hex if backup_id is None else backup_id)
        _database(self.paths,database)
        final=self._directory(identifier);pending=self._root()/('.pending-'+identifier);published=False
        try:
            with self.rollover._owner(),ExitStack() as owners:
                owners.enter_context(_lease(self.paths,'audit-sealed-backups/.writer.lock'))
                if final.exists() or pending.exists():raise AdapterError('AUDIT_BACKUP_EXISTS')
                count,used=self._usage()
                if count>=self.max_backups or used+MANIFEST_LIMIT>=self.max_total_bytes:
                    raise AdapterError('AUDIT_BACKUP_LIMIT')
                source=_read_source(self.paths,database,self.identity,self.rollover.archives)
                metadata=source['metadata'];seal=metadata.get('sealed')
                if seal is None or metadata.get('successor_opened') is None:raise AdapterError('AUDIT_BACKUP_NOT_SEALED')
                owners.enter_context(_lease(self.paths,database+'.writer.lock'))
                connection=sqlite3.connect(_database(self.paths,database).as_uri()+'?mode=ro',uri=True)
                try:
                    logical_bytes=connection.execute('PRAGMA page_count').fetchone()[0]*connection.execute('PRAGMA page_size').fetchone()[0]
                finally:connection.close()
                prefix_root=_safe(self.paths,'audit-archives/'+seal['archive_id'])
                prefix_bytes=sum((prefix_root/name).stat().st_size for name in ('manifest.json','records.jsonl.gz'))
                if logical_bytes>self.max_database_bytes or used+logical_bytes+prefix_bytes+MANIFEST_LIMIT>self.max_total_bytes:
                    raise AdapterError('AUDIT_BACKUP_LIMIT')
                successor=_read_source(self.paths,seal['next_database'],self.identity,self.rollover.archives)
                _check_receipt(metadata['successor_opened'],seal,successor['first'])
                pending.mkdir();snapshot=AppPaths((pending/'snapshot').resolve())
                self._sqlite_copy(_database(self.paths,database),snapshot.runtime_path(database))
                relatives=self._relative_files(database,seal['archive_id'])
                for relative in relatives[1:]:
                    self._copy(self.paths.runtime_path(relative.removeprefix('snapshot/.runtime/')),pending/relative,64*MIB)
                files={relative:self._hash(pending/relative,self.max_database_bytes if i==0 else 64*MIB)
                    for i,relative in enumerate(relatives)}
                if used+sum(info['bytes_n'] for info in files.values())+MANIFEST_LIMIT>self.max_total_bytes:
                    raise AdapterError('AUDIT_BACKUP_LIMIT')
                value=dict(format=FORMAT,backup_id=identifier,identity=deepcopy(self.identity),database=database,
                    source_metadata=metadata,successor_first=successor['first'],records_n=seal['seq'],chain_tip=seal['chain_tip'],
                    scope='SEALED_SEGMENT_ONLY',files=files,created_at_ms=time.time_ns()//1000000,
                    exporter_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),full_audit_complete=False)
                value['manifest_sha256']=_sha(value);raw=_json(value).encode()
                if len(raw)>MANIFEST_LIMIT:raise AdapterError('AUDIT_BACKUP_LIMIT')
                with (pending/'manifest.json').open('xb') as writer:writer.write(raw);writer.flush();os.fsync(writer.fileno())
                self._verify_directory(pending,identifier)
                self._hook('before_publish');pending.rename(final);published=True;self._hook('after_publish')
                return self.verify(identifier)
        except BaseException as error:
            if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
            if published:raise AdapterError('AUDIT_BACKUP_PUBLICATION_UNKNOWN') from None
            if isinstance(error,AdapterError) and error.reason.startswith('AUDIT_BACKUP_'):raise
            reason=('AUDIT_BACKUP_INTEGRITY' if isinstance(error,AdapterError) and error.reason=='AUDIT_ROLLOVER_INTEGRITY'
                else 'AUDIT_BACKUP_WRITE_FAILED')
            raise AdapterError(reason) from None

    def _layout(self,directory,files):
        allowed_files=set(files)|{'manifest.json'};allowed_dirs=set()
        for name in allowed_files:
            allowed_dirs.update(str(p).replace('\\','/') for p in Path(name).parents if str(p)!='.')
        stack=[directory];seen=set();nodes=0
        while stack:
            for path in stack.pop().iterdir():
                nodes+=1
                if nodes>20 or _link(path):raise AdapterError('AUDIT_BACKUP_LAYOUT')
                name=path.relative_to(directory).as_posix()
                if path.is_dir() and name in allowed_dirs:stack.append(path)
                elif path.is_file() and name in allowed_files:seen.add(name)
                else:raise AdapterError('AUDIT_BACKUP_LAYOUT')
        if seen!=allowed_files:raise AdapterError('AUDIT_BACKUP_LAYOUT')

    def _verify_directory(self,directory,identifier):
        try:
            if _link(directory) or not directory.is_dir() or _link(directory/'manifest.json'):
                raise AdapterError('AUDIT_BACKUP_LAYOUT')
            with (directory/'manifest.json').open('rb') as reader:raw=reader.read(MANIFEST_LIMIT+1)
            if len(raw)>MANIFEST_LIMIT:raise AdapterError('AUDIT_BACKUP_LIMIT')
            value=json.loads(raw)
            if not isinstance(value,dict) or set(value)!=KEYS:raise ValueError()
            snapshot=AppPaths((directory/'snapshot').resolve());database=value['database'];_database(snapshot,database)
            seal=value['source_metadata']['sealed'];archive_id=_identifier(seal['archive_id'])
            files=self._relative_files(database,archive_id)
            if not isinstance(value['files'],dict) or set(value['files'])!=set(files):raise ValueError()
            self._layout(directory,files)
            body=dict(value);digest=body.pop('manifest_sha256')
            if not _digest(digest) or _sha(body)!=digest:raise ValueError()
            if value['identity']!=self.identity:raise AdapterError('AUDIT_BACKUP_IDENTITY')
            if (value['format']!=FORMAT or value['backup_id']!=identifier or value['scope']!='SEALED_SEGMENT_ONLY'
                or value['full_audit_complete'] is not False or type(value['records_n']) is not int
                or type(value['created_at_ms']) is not int or value['created_at_ms']<0
                or not _digest(value['exporter_code_sha256'])):raise ValueError()
            for i,relative in enumerate(files):
                info=value['files'][relative]
                if (not isinstance(info,dict) or set(info)!={'bytes_n','sha256'}
                    or type(info['bytes_n']) is not int or info['bytes_n']<0 or not _digest(info['sha256'])):raise ValueError()
                if self._hash(directory/relative,self.max_database_bytes if i==0 else 64*MIB)!=info:raise ValueError()
            if sum(info['bytes_n'] for info in value['files'].values())+len(raw)>self.max_total_bytes:
                raise AdapterError('AUDIT_BACKUP_LIMIT')
            _schema(snapshot.runtime_path(database))
            source=_read_source(snapshot,database,self.identity,AuditArchive(snapshot),immutable=True)
            if source['metadata']!=value['source_metadata'] or source['metadata'].get('successor_opened') is None:raise ValueError()
            if source['last']['seq']!=value['records_n'] or source['last']['content_sha256']!=value['chain_tip']:raise ValueError()
            witness=value['successor_first'];_Summary(self.identity).accept(witness)
            if witness['kind']!='RUN_BEGIN' or witness['data'].get('parent_segment')!=seal:raise ValueError()
            _check_receipt(source['metadata']['successor_opened'],seal,witness)
            return deepcopy(value)
        except AdapterError as error:
            if error.reason.startswith('AUDIT_BACKUP_'):raise
            raise AdapterError('AUDIT_BACKUP_INTEGRITY') from None
        except (OSError,ValueError,KeyError,TypeError,RecursionError,sqlite3.Error):
            raise AdapterError('AUDIT_BACKUP_INTEGRITY') from None

    def verify(self,backup_id,*,expected_manifest_sha256=None):
        value=self._verify_directory(self._directory(backup_id),backup_id)
        if expected_manifest_sha256 is not None and value['manifest_sha256']!=expected_manifest_sha256:
            raise AdapterError('AUDIT_BACKUP_PIN_MISMATCH')
        return dict(backup_id=backup_id,manifest=value,integrity='INTERNAL_VERIFIED',
            authenticity='UNVERIFIED' if expected_manifest_sha256 is None else 'PINNED_MANIFEST')

    def restore(self,backup_id,*,expected_manifest_sha256,target_paths=None,quiesced=False):
        if quiesced is not True:raise AdapterError('AUDIT_RESTORE_QUIESCENCE')
        if not _digest(expected_manifest_sha256):raise AdapterError('AUDIT_RESTORE_PIN_REQUIRED')
        value=self.verify(backup_id,expected_manifest_sha256=expected_manifest_sha256)['manifest']
        paths=self.paths if target_paths is None else target_paths
        if not isinstance(paths,AppPaths):raise AdapterError('AUDIT_RESTORE_INVALID')
        manager=AuditRollover(paths,**self.arguments);database=value['database'];seal=value['source_metadata']['sealed']
        published=False
        try:
            with manager._owner(),_lease(paths,'audit-archives/.writer.lock'),_lease(paths,database+'.writer.lock'):
                target=_safe(paths,database)
                if any(_safe(paths,database+suffix).exists() for suffix in ('','-wal','-shm')):
                    raise AdapterError('AUDIT_RESTORE_EXISTS')
                prefix=_safe(paths,'audit-archives/'+seal['archive_id'])
                reuse=prefix.exists()
                if reuse:
                    try:manager.archives.verify(seal['archive_id'],expected_identity=self.identity,
                        expected_manifest_sha256=seal['archive_manifest_sha256'])
                    except AdapterError:raise AdapterError('AUDIT_RESTORE_CONFLICT') from None
                next_path=_safe(paths,seal['next_database'])
                if next_path.exists():
                    try:
                        next_source=_read_source(paths,seal['next_database'],self.identity,manager.archives)
                        if next_source['first']!=value['successor_first']:raise ValueError()
                    except (AdapterError,ValueError):raise AdapterError('AUDIT_RESTORE_CONFLICT') from None
                stage_root=_safe(paths,'audit-restores');stage_root.mkdir(parents=True,exist_ok=True)
                try:count,used=self._usage(stage_root)
                except AdapterError as error:
                    if error.reason=='AUDIT_BACKUP_LIMIT':raise AdapterError('AUDIT_RESTORE_LIMIT') from None
                    raise
                if count>=self.max_backups or used+sum(info['bytes_n'] for info in value['files'].values())+MANIFEST_LIMIT>self.max_total_bytes:
                    raise AdapterError('AUDIT_RESTORE_LIMIT')
                stage=_safe(paths,'audit-restores/.pending-'+uuid.uuid4().hex);stage.mkdir(parents=True)
                for relative,info in value['files'].items():
                    self._copy(self._directory(backup_id)/relative,stage/relative,info['bytes_n'])
                self._copy(self._directory(backup_id)/'manifest.json',stage/'manifest.json',MANIFEST_LIMIT)
                self._verify_directory(stage,backup_id)
                staged_paths=AppPaths((stage/'snapshot').resolve())
                if not reuse:
                    staged_paths.runtime_path('audit-archives/'+seal['archive_id']).rename(prefix);published=True
                self._hook('after_restore_prefix')
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():raise AdapterError('AUDIT_RESTORE_EXISTS')
                staged_paths.runtime_path(database).rename(target);published=True
                self._hook('after_restore_database')
                restored=_read_source(paths,database,self.identity,manager.archives)
                if restored['metadata']!=value['source_metadata']:raise AdapterError('AUDIT_RESTORE_CONFLICT')
                receipt=dict(backup_id=backup_id,manifest_sha256=expected_manifest_sha256,database=database,
                    seal_sha256=seal['seal_sha256'],runtime_started=False,scope='SEALED_SEGMENT_ONLY')
                return dict(receipt,receipt_sha256=_sha(receipt))
        except BaseException as error:
            if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
            if published:raise AdapterError('AUDIT_RESTORE_UNKNOWN') from None
            if isinstance(error,AdapterError) and error.reason.startswith(('AUDIT_RESTORE_','AUDIT_BACKUP_')):raise
            raise AdapterError('AUDIT_RESTORE_WRITE_FAILED') from None
