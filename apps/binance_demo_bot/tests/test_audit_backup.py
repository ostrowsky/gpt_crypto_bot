"""C2b1 real SQLite/WAL sealed-copy/restore contracts; no exchange or keys."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import _support
from binance_demo_bot.adapter import AdapterError
from binance_demo_bot.audit_rollover import AuditRollover
from binance_demo_bot.interaction_audit import AuditJournal, _json, _sha
from binance_demo_bot.isolation import AppPaths
from binance_demo_bot import market_sqlite as sqlite3


IDENTITY=dict(account_alias='synthetic',episode_id='one',scope_id='all')
A='a'*32
B='b'*32


class SealedBackupTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.audit_backup')
        base=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures';base.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=base);self.addCleanup(temporary.cleanup)
        self.paths=AppPaths(Path(temporary.name).resolve())
        self.journal=AuditJournal(self.paths,**IDENTITY);self.addCleanup(self.journal.close)
        self.rollover=AuditRollover(self.paths,**IDENTITY);self.child=None
        self.backup=self.make()
    def make(self,**kwargs):return self.api.SealedAuditBackup(self.paths,**IDENTITY,**kwargs)
    def sealed(self):
        if self.child is None:
            self.child=self.rollover.rotate(self.journal,quiesced=True);self.addCleanup(self.child.close)
        return self.child.parent_segment
    def create(self,backup=None):
        self.sealed();return (backup or self.backup).create(backup_id=A)
    def directory(self):return self.paths.runtime_path('audit-sealed-backups/'+A)
    def snapshot(self):return AppPaths((self.directory()/'snapshot').resolve())
    def pin(self):return self.backup.verify(A)['manifest']['manifest_sha256']
    def reason(self,code,call,*args,**kwargs):
        with self.assertRaises(AdapterError) as caught:call(*args,**kwargs)
        self.assertEqual(caught.exception.reason,code)
    def remove_original(self):
        self.child.close()
        seal=self.sealed();root=self.paths.runtime_path('.').resolve()
        # Model loss of the entire SQLite file set. A different test explicitly
        # retains stale WAL and requires restoration to refuse it.
        paths=[self.journal.path]+[self.journal.path.with_name('audit.sqlite'+suffix)
            for suffix in ('-wal','-shm') if self.journal.path.with_name('audit.sqlite'+suffix).exists()]
        paths += [self.paths.runtime_path('audit-archives/'+seal['archive_id']+'/manifest.json'),
            self.paths.runtime_path('audit-archives/'+seal['archive_id']+'/records.jsonl.gz')]
        for path in paths:
            self.assertTrue(path.resolve().is_relative_to(root));path.unlink()
        paths[-1].parent.rmdir()
    def restore(self,**kwargs):
        args=dict(expected_manifest_sha256=self.pin(),quiesced=True);args.update(kwargs)
        return self.backup.restore(A,**args)
    def target(self):
        root=self.paths.root/'target';root.mkdir();return AppPaths(root)

    def test_c211_full_terminal_metadata_and_activation_roundtrip(self):
        seal=self.sealed();result=self.create();manifest=result['manifest']
        self.assertEqual(manifest['source_metadata']['sealed'],seal)
        self.assertIn('successor_opened',manifest['source_metadata'])
        self.assertEqual(manifest['records_n'],seal['seq'])
        self.assertEqual(manifest['chain_tip'],seal['chain_tip'])
        self.assertEqual(manifest['scope'],'SEALED_SEGMENT_ONLY')
        self.assertFalse(manifest['full_audit_complete'])
        verified=self.backup.verify(A,expected_manifest_sha256=manifest['manifest_sha256'])
        self.assertEqual(verified['integrity'],'INTERNAL_VERIFIED')
        self.assertEqual(verified['authenticity'],'PINNED_MANIFEST')

    def test_c211_online_backup_includes_committed_wal_pages(self):
        reader=sqlite3.connect(self.journal.path.as_uri()+'?mode=ro',uri=True)
        reader.execute('BEGIN');reader.execute('SELECT COUNT(*) FROM records').fetchone()
        try:
            seal=self.sealed()
            self.assertTrue(self.journal.path.with_name('audit.sqlite-wal').exists())
            result=self.create()
            self.assertEqual(result['manifest']['source_metadata']['sealed'],seal)
            self.assertEqual(result['manifest']['records_n'],seal['seq'])
        finally:reader.close()

    def test_c212_repeated_verification_does_not_mutate_or_add_sidecars(self):
        self.create()
        before={str(p.relative_to(self.directory())):p.read_bytes() for p in self.directory().rglob('*') if p.is_file()}
        self.backup.verify(A);self.backup.verify(A)
        after={str(p.relative_to(self.directory())):p.read_bytes() for p in self.directory().rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        self.assertFalse(any(name.endswith(('-wal','-shm')) for name in after))

    def test_c214_lost_sealed_db_and_prefix_restore_then_recover(self):
        seal=self.sealed();result=self.create();pin=result['manifest']['manifest_sha256']
        original_first=self.child.read(limit=1)[0]
        self.remove_original()
        restored=self.backup.restore(A,expected_manifest_sha256=pin,quiesced=True)
        self.assertEqual(restored['database'],'audit.sqlite')
        self.assertFalse(restored['runtime_started'])
        recovered=self.rollover.recover();self.addCleanup(recovered.close)
        self.assertEqual(recovered.parent_segment,seal)
        self.assertEqual(recovered.read(limit=1)[0],original_first)

    def test_c211_active_source_cannot_be_backed_up_as_sealed(self):
        self.reason('AUDIT_BACKUP_NOT_SEALED',self.backup.create,backup_id=A)
        self.assertEqual(self.journal.health()['state'],'READY')

    def test_c212_duplicate_backup_never_overwrites(self):
        self.create();raw=(self.directory()/'manifest.json').read_bytes()
        self.reason('AUDIT_BACKUP_EXISTS',self.backup.create,backup_id=A)
        self.assertEqual((self.directory()/'manifest.json').read_bytes(),raw)

    def test_c212_wrong_pin_and_identity_denied(self):
        self.create()
        self.reason('AUDIT_BACKUP_PIN_MISMATCH',self.backup.verify,A,expected_manifest_sha256='0'*64)
        other=self.api.SealedAuditBackup(self.paths,**dict(IDENTITY,scope_id='foreign'))
        self.reason('AUDIT_BACKUP_IDENTITY',other.verify,A)
        self.assertEqual(self.backup.verify(A)['authenticity'],'UNVERIFIED')

    def test_c212_corrupt_sqlite_and_extra_bundle_files_denied(self):
        self.create();path=self.snapshot().runtime_path('audit.sqlite');path.write_bytes(b'corrupt')
        self.reason('AUDIT_BACKUP_INTEGRITY',self.backup.verify,A)
        (self.directory()/'extra').write_text('synthetic')
        self.reason('AUDIT_BACKUP_LAYOUT',self.backup.verify,A)

    def test_c212_corrupt_included_prefix_denied(self):
        seal=self.sealed();self.create()
        self.snapshot().runtime_path('audit-archives/'+seal['archive_id']+'/records.jsonl.gz').write_bytes(b'bad')
        self.reason('AUDIT_BACKUP_INTEGRITY',self.backup.verify,A)

    def test_c213_database_limit_refuses_before_publication(self):
        self.sealed()
        self.reason('AUDIT_BACKUP_LIMIT',self.make(max_database_bytes=4096).create,backup_id=A)
        self.assertFalse(self.directory().exists())
        self.assertTrue(self.journal.path.exists())

    def test_c213_count_and_total_limits_preserve_original(self):
        self.create()
        self.reason('AUDIT_BACKUP_LIMIT',self.make(max_backups=1).create,backup_id=B)
        self.reason('AUDIT_BACKUP_LIMIT',self.make(max_total_bytes=1).create,backup_id=B)
        self.assertEqual(self.backup.verify(A)['integrity'],'INTERNAL_VERIFIED')

    def test_c214_quiescence_required_before_restore(self):
        self.create()
        self.reason('AUDIT_RESTORE_QUIESCENCE',self.backup.restore,A,expected_manifest_sha256=self.pin())

    def test_c214_existing_database_is_not_overwritten(self):
        self.create();raw=self.journal.path.read_bytes()
        self.reason('AUDIT_RESTORE_EXISTS',self.restore)
        self.assertEqual(self.journal.path.read_bytes(),raw)

    def test_c214_stale_sidecars_are_not_removed_or_reused(self):
        self.create();target=self.target();path=target.runtime_path('audit.sqlite-wal')
        path.parent.mkdir(parents=True);path.write_bytes(b'preserve')
        self.reason('AUDIT_RESTORE_EXISTS',self.restore,target_paths=target)
        self.assertEqual(path.read_bytes(),b'preserve')
        self.assertFalse(target.runtime_path('audit.sqlite').exists())

    def test_c214_conflicting_existing_prefix_blocks_restore(self):
        seal=self.sealed();self.create();target=self.target()
        path=target.runtime_path('audit-archives/'+seal['archive_id']);path.mkdir(parents=True)
        (path/'manifest.json').write_text('{}')
        self.reason('AUDIT_RESTORE_CONFLICT',self.restore,target_paths=target)
        self.assertFalse(target.runtime_path('audit.sqlite').exists())

    def test_c214_same_prefix_can_be_reused_after_partial_restore(self):
        self.create();self.remove_original()
        def fault(phase):
            if phase=='after_restore_prefix':raise OSError('private detail')
        broken=self.make(fault_hook=fault)
        self.reason('AUDIT_RESTORE_UNKNOWN',broken.restore,A,expected_manifest_sha256=self.pin(),quiesced=True)
        self.assertFalse(self.journal.path.exists())
        self.restore()
        recovered=self.rollover.recover();self.addCleanup(recovered.close)

    def test_c214_successor_witness_conflict_is_not_overwritten(self):
        self.create();self.remove_original()
        path=self.child.path;connection=sqlite3.connect(path)
        try:
            connection.execute('DROP TRIGGER immutable_records_UPDATE')
            connection.execute("UPDATE records SET payload_json='{}' WHERE seq=1")
        finally:connection.close()
        self.reason('AUDIT_RESTORE_CONFLICT',self.restore)
        self.assertFalse(self.journal.path.exists())

    def test_c212_invalid_ids_cannot_escape(self):
        for value in ('../outside','A'*32,'a'*31):
            self.reason('AUDIT_BACKUP_INVALID',self.backup.create,backup_id=value)
            self.reason('AUDIT_BACKUP_INVALID',self.backup.verify,value)

    def test_c213_prepublication_fault_leaves_incomplete_and_source(self):
        def fault(phase):
            if phase=='before_publish':raise OSError('do-not-expose-private-path')
        self.reason('AUDIT_BACKUP_WRITE_FAILED',self.create,self.make(fault_hook=fault))
        self.assertFalse(self.directory().exists())
        self.assertEqual(len(list(self.directory().parent.glob('.pending-*'))),1)
        self.assertTrue(self.journal.path.exists())

    def test_c213_postpublication_fault_is_unknown_and_verifiable(self):
        def fault(phase):
            if phase=='after_publish':raise OSError('do-not-expose-private-path')
        self.reason('AUDIT_BACKUP_PUBLICATION_UNKNOWN',self.create,self.make(fault_hook=fault))
        self.assertEqual(self.backup.verify(A)['integrity'],'INTERNAL_VERIFIED')

    def crash(self,phase):
        self.sealed();self.child.close()
        script='''import os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from binance_demo_bot.isolation import AppPaths
from binance_demo_bot.audit_backup import SealedAuditBackup
def fault(phase):
    if phase==sys.argv[3]:os._exit(73)
SealedAuditBackup(AppPaths(Path(sys.argv[2])),account_alias='synthetic',episode_id='one',scope_id='all',fault_hook=fault).create(backup_id='a'*32)
'''
        result=subprocess.run([sys.executable,'-I','-c',script,str(Path(__file__).resolve().parents[1]/'src'),
            str(self.paths.root),phase],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,73,result.stderr)

    def test_c213_real_process_death_before_publication(self):
        self.crash('before_publish')
        self.assertFalse(self.directory().exists())
        self.assertEqual(len(list(self.directory().parent.glob('.pending-*'))),1)

    def test_c213_real_process_death_after_publication(self):
        self.crash('after_publish')
        self.assertEqual(self.backup.verify(A)['integrity'],'INTERNAL_VERIFIED')

    def test_c214_restore_into_empty_runtime_is_segment_only(self):
        self.create();target=self.target();result=self.restore(target_paths=target)
        self.assertTrue(target.runtime_path('audit.sqlite').exists())
        self.assertFalse(result['runtime_started'])
        self.reason('AUDIT_ROLLOVER_MISSING',AuditRollover(target,**IDENTITY).recover)

    def test_c213_insufficient_quota_is_rejected_before_staging(self):
        self.sealed()
        self.reason('AUDIT_BACKUP_LIMIT',self.make(max_total_bytes=70000).create,backup_id=A)
        self.assertEqual(list(self.directory().parent.glob('.pending-*')),[])

    def test_c214_restore_staging_count_is_bounded(self):
        self.create();self.remove_original()
        def fault(phase):
            if phase=='after_restore_prefix':raise OSError('synthetic')
        limited=self.make(max_backups=1,fault_hook=fault)
        self.reason('AUDIT_RESTORE_UNKNOWN',limited.restore,A,expected_manifest_sha256=self.pin(),quiesced=True)
        self.reason('AUDIT_RESTORE_LIMIT',self.make(max_backups=1).restore,A,
            expected_manifest_sha256=self.pin(),quiesced=True)
        self.assertFalse(self.journal.path.exists())

    def test_c212_rehashed_unexpected_sqlite_schema_is_rejected(self):
        self.create();path=self.snapshot().runtime_path('audit.sqlite')
        connection=sqlite3.connect(path)
        try:connection.execute('CREATE VIEW unexpected AS SELECT 1')
        finally:connection.close()
        manifest_path=self.directory()/'manifest.json';value=json.loads(manifest_path.read_text())
        value['files']['snapshot/.runtime/audit.sqlite']=self.backup._hash(path,512*1024*1024)
        value.pop('manifest_sha256');value['manifest_sha256']=_sha(value)
        manifest_path.write_text(_json(value))
        self.reason('AUDIT_BACKUP_INTEGRITY',self.backup.verify,A)
