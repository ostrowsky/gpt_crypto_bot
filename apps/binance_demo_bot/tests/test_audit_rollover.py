"""AUD-C2a sealed ancestry/restart contracts, with own real SQLite and processes."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import _support
from binance_demo_bot.adapter import AdapterError
from binance_demo_bot.interaction_audit import AuditJournal, _json, _sha
from binance_demo_bot.isolation import AppPaths
from binance_demo_bot.journal import _WriterLease
from binance_demo_bot import market_sqlite as sqlite3


IDENTITY=dict(account_alias='synthetic',episode_id='one',scope_id='all')


class AuditRolloverTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.audit_rollover')
        base=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures';base.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=base);self.addCleanup(temporary.cleanup)
        self.paths=AppPaths(Path(temporary.name).resolve())
        self.journal=self.source();self.addCleanup(self.journal.close)
        self.rollover=self.manager()

    def source(self,**kwargs):
        journal=AuditJournal(self.paths,**dict(IDENTITY,**kwargs))
        self.addCleanup(journal.close);return journal
    def manager(self,**kwargs):return self.api.AuditRollover(self.paths,**dict(IDENTITY,**kwargs))
    def reason(self,code,call,*args,**kwargs):
        with self.assertRaises(AdapterError) as caught:
            result=call(*args,**kwargs)
            if isinstance(result,AuditJournal):self.addCleanup(result.close)
        self.assertEqual(caught.exception.reason,code)
    def rotate(self,manager=None):
        child=(manager or self.rollover).rotate(self.journal,quiesced=True)
        self.addCleanup(child.close);return child
    def meta(self,database,key):
        connection=sqlite3.connect(self.paths.runtime_path(database).as_uri()+'?mode=ro',uri=True)
        try:
            row=connection.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
            return json.loads(row[0]) if row else None
        finally:connection.close()
    def tamper_meta(self,database,key,value):
        connection=sqlite3.connect(self.paths.runtime_path(database))
        try:
            connection.execute('DROP TRIGGER immutable_meta_UPDATE')
            connection.execute('UPDATE meta SET value=? WHERE key=?',(json.dumps(value),key))
            # Isolate metadata validation from the separate missing-trigger guard.
            connection.execute("CREATE TRIGGER immutable_meta_UPDATE BEFORE UPDATE ON meta BEGIN SELECT RAISE(ABORT,'immutable audit'); END")
        finally:connection.close()

    def test_c201_explicit_quiescence_required(self):
        self.reason('AUDIT_ROLLOVER_QUIESCENCE',self.rollover.rotate,self.journal)
        self.assertEqual(self.journal.health()['state'],'READY')

    def test_c202_seal_and_successor_preserve_original_rows(self):
        self.journal.append('DECISION',dict(action='WAIT'))
        before=self.journal.read();child=self.rotate()
        parent=child.parent_segment
        self.assertEqual(parent['database'],'audit.sqlite')
        self.assertEqual(parent['prefix_last_seq'],len(before))
        self.assertEqual(parent['prefix_chain_tip'],before[-1]['content_sha256'])
        self.assertEqual(parent['seq'],len(before)+1)
        first=child.read()[0]
        self.assertEqual(first['seq'],1)
        self.assertEqual(first['data']['parent_segment'],parent)
        self.assertEqual(self.journal.health()['state'],'SEALED')
        self.reason('AUDIT_UNAVAILABLE',self.journal.append,'DECISION',dict(action='WAIT'))
        self.assertTrue(self.journal.path.exists())

    def test_c202_ordinary_reopen_cannot_append_to_sealed_source(self):
        self.rotate()
        self.reason('AUDIT_SEGMENT_SEALED',self.source)

    def test_c203_successor_receipt_freezes_first_record(self):
        child=self.rotate();receipt=self.meta('audit.sqlite','successor_opened')
        self.assertEqual(receipt['first_record_sha256'],child.read()[0]['content_sha256'])
        self.assertEqual(receipt['seal_sha256'],child.parent_segment['seal_sha256'])

    def test_c204_recovery_opens_same_child_with_existing_events(self):
        child=self.rotate();child.append('DECISION',dict(action='HOLD'))
        original=child.read();path=child.path;child.close()
        recovered=self.rollover.recover();self.addCleanup(recovered.close)
        self.assertEqual(recovered.path,path)
        self.assertEqual(recovered.read()[:len(original)],original)
        self.assertEqual(recovered.parent_segment['database'],'audit.sqlite')

    def test_c204_two_rotations_follow_bounded_chain(self):
        child=self.rotate()
        grandchild=self.rollover.rotate(child,quiesced=True);self.addCleanup(grandchild.close)
        target=grandchild.path;grandchild.close()
        recovered=self.rollover.recover();self.addCleanup(recovered.close)
        self.assertEqual(recovered.path,target)
        self.assertEqual(len(list(self.paths.runtime_path('audit-segments').glob('*.sqlite'))),2)

    def test_c204_chain_depth_limit_is_explicit(self):
        child=self.rotate();child.close()
        self.reason('AUDIT_ROLLOVER_LIMIT',self.manager(max_segments=1).recover)

    def test_c201_unfinished_rest_attempt_blocks_rotation(self):
        self.journal.append('REST_BEGIN',dict(attempt_id='unfinished'))
        self.reason('AUDIT_ROLLOVER_UNRESOLVED',self.rollover.rotate,self.journal,quiesced=True)
        self.assertEqual(self.journal.health()['state'],'READY')

    def test_c201_unfinished_ws_operation_blocks_rotation(self):
        self.journal.append('WS_BEGIN',dict(operation_id='unfinished'))
        self.reason('AUDIT_ROLLOVER_UNRESOLVED',self.rollover.rotate,self.journal,quiesced=True)

    def test_c201_unprocessed_recv_blocks_rotation(self):
        self.journal.append('WS_END',dict(operation='RECV',operation_id='one',outcome='COMPLETED'))
        self.reason('AUDIT_ROLLOVER_UNRESOLVED',self.rollover.rotate,self.journal,quiesced=True)

    def test_c201_failed_and_closed_sources_denied(self):
        self.journal._failed('AUDIT_COMMIT_UNKNOWN')
        self.reason('AUDIT_ROLLOVER_SOURCE',self.rollover.rotate,self.journal,quiesced=True)
        self.journal.close()
        self.reason('AUDIT_ROLLOVER_SOURCE',self.rollover.rotate,self.journal,quiesced=True)

    def test_c201_foreign_identity_and_runtime_denied(self):
        self.reason('AUDIT_ROLLOVER_IDENTITY',self.manager(scope_id='other').rotate,self.journal,quiesced=True)
        other=self.paths.root/'other';other.mkdir()
        self.reason('AUDIT_ROLLOVER_IDENTITY',self.api.AuditRollover(AppPaths(other),**IDENTITY).rotate,
            self.journal,quiesced=True)

    def test_c205_missing_activated_child_never_becomes_empty_history(self):
        child=self.rotate();path=child.path;child.close();path.unlink()
        self.reason('AUDIT_ROLLOVER_MISSING',self.rollover.recover)
        self.assertFalse(path.exists())

    def test_c205_empty_activated_child_denied_before_mutation(self):
        child=self.rotate();path=child.path;child.close();path.write_bytes(b'')
        self.reason('AUDIT_ROLLOVER_MISSING',self.rollover.recover)
        self.assertEqual(path.stat().st_size,0)

    def test_c205_missing_root_never_creates_fresh_episode(self):
        self.reason('AUDIT_ROLLOVER_MISSING',self.rollover.recover,database='audit-segments/'+'a'*32+'.sqlite')

    def test_c205_corrupt_archive_prevents_recovery(self):
        child=self.rotate();parent=child.parent_segment;child.close()
        path=self.paths.runtime_path('audit-archives/'+parent['archive_id']+'/records.jsonl.gz')
        path.write_bytes(b'bad')
        self.reason('AUDIT_ROLLOVER_INTEGRITY',self.rollover.recover)

    def test_c205_modified_seal_is_rejected(self):
        child=self.rotate();seal=child.parent_segment;child.close()
        seal['chain_tip']='0'*64;self.tamper_meta('audit.sqlite','sealed',seal)
        self.reason('AUDIT_ROLLOVER_INTEGRITY',self.rollover.recover)

    def test_c205_parent_mismatch_denies_ordinary_child_reopen(self):
        child=self.rotate();database=child.parent_segment['next_database'];child.close()
        parent=self.meta(database,'parent_segment');parent['prefix_chain_tip']='0'*64
        self.tamper_meta(database,'parent_segment',parent)
        self.reason('AUDIT_ROLLOVER_INTEGRITY',self.source,database=database)

    def test_c201_rollover_writer_lease_is_exclusive(self):
        lease=_WriterLease(self.paths.runtime_path('audit-rollover.writer.lock'))
        try:self.reason('AUDIT_ROLLOVER_BUSY',self.rollover.rotate,self.journal,quiesced=True)
        finally:lease.close()

    def test_c201_invalid_recovery_paths_denied(self):
        for path in ('../audit.sqlite','C:/outside','financial.sqlite'):
            self.reason('AUDIT_ROLLOVER_INVALID',self.rollover.recover,database=path)

    def test_c202_seal_transaction_rollback_does_not_publish_child(self):
        def fault(phase):
            if phase=='before_seal_commit':raise OSError('private detail')
        self.journal.fault_hook=fault
        self.reason('AUDIT_SEAL_WRITE_FAILED',self.rollover.rotate,self.journal,quiesced=True)
        self.assertIsNone(self.meta('audit.sqlite','sealed'))
        self.assertEqual(self.journal.health()['state'],'FAILED')

    def test_c202_seal_commit_unknown_is_recovered_without_duplicate_seal(self):
        def fault(phase):
            if phase=='after_seal_commit':raise OSError('private detail')
        self.journal.fault_hook=fault
        self.reason('AUDIT_SEAL_COMMIT_UNKNOWN',self.rollover.rotate,self.journal,quiesced=True)
        seal=self.meta('audit.sqlite','sealed')
        recovered=self.rollover.recover();self.addCleanup(recovered.close)
        self.assertEqual(recovered.parent_segment,seal)
        self.assertEqual(self.meta('audit.sqlite','sealed'),seal)

    def crash(self,phase):
        self.journal.close()
        package=Path(__file__).resolve().parents[1]/'src'
        script='''import os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from binance_demo_bot.isolation import AppPaths
from binance_demo_bot.interaction_audit import AuditJournal
from binance_demo_bot.audit_rollover import AuditRollover
paths=AppPaths(Path(sys.argv[2]))
identity=dict(account_alias='synthetic',episode_id='one',scope_id='all')
def fault(phase):
    if phase==sys.argv[3]:os._exit(73)
journal=AuditJournal(paths,**identity)
AuditRollover(paths,**identity,fault_hook=fault).rotate(journal,quiesced=True)
'''
        result=subprocess.run([sys.executable,'-I','-c',script,str(package),str(self.paths.root),phase],
            capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,73,result.stderr)

    def test_c205_process_death_after_reservation_preserves_unsealed_source(self):
        self.crash('after_reserve')
        self.assertIsNone(self.meta('audit.sqlite','sealed'))
        recovered=self.rollover.recover();self.addCleanup(recovered.close)
        self.assertEqual(recovered.path,self.journal.path)
        self.assertGreater(recovered.health()['incomplete_prior_runs_n'],0)
        self.assertEqual(len(list(self.paths.runtime_path('audit-segments').glob('*.sqlite'))),1)

    def test_c204_process_death_after_seal_initializes_reserved_child(self):
        self.crash('after_seal')
        recovered=self.rollover.recover();self.addCleanup(recovered.close)
        self.assertEqual(recovered.parent_segment,self.meta('audit.sqlite','sealed'))
        self.assertIsNotNone(self.meta('audit.sqlite','successor_opened'))

    def test_c204_process_death_after_child_open_retains_unknown_prior_run(self):
        self.crash('after_child_open')
        recovered=self.rollover.recover();self.addCleanup(recovered.close)
        self.assertGreater(recovered.health()['incomplete_prior_runs_n'],0)

    def test_c204_process_death_after_activation_reopens_same_segment(self):
        self.crash('after_activation')
        seal=self.meta('audit.sqlite','sealed')
        recovered=self.rollover.recover();self.addCleanup(recovered.close)
        self.assertEqual(recovered.path,self.paths.runtime_path(seal['next_database']))

    def test_c206_no_deletion_or_automatic_reclamation_api(self):
        self.rotate()
        self.assertFalse(hasattr(self.rollover,'delete'))
        self.assertFalse(hasattr(self.rollover,'reclaim'))
        self.assertTrue(self.journal.path.exists())

    def test_c203_ordinary_child_start_checks_external_first_record_receipt(self):
        child=self.rotate();database=child.parent_segment['next_database'];child.close()
        connection=sqlite3.connect(self.paths.runtime_path(database))
        try:
            rows=list(connection.execute('SELECT seq,payload_json FROM records ORDER BY seq'))
            connection.execute('DROP TRIGGER immutable_records_UPDATE');previous='0'*64
            for seq,raw in rows:
                value=json.loads(raw)
                if seq==1:value['run_id']='b'*32
                value['previous_sha256']=previous;previous=_sha(value)
                connection.execute('UPDATE records SET payload_json=?,content_sha256=? WHERE seq=?',(_json(value),previous,seq))
            connection.execute("CREATE TRIGGER immutable_records_UPDATE BEFORE UPDATE ON records BEGIN SELECT RAISE(ABORT,'immutable audit'); END")
        finally:connection.close()
        self.reason('AUDIT_ROLLOVER_INTEGRITY',self.source,database=database)

    def test_c204_reserved_child_cannot_bypass_depth_limit(self):
        self.crash('after_seal')
        self.reason('AUDIT_ROLLOVER_LIMIT',self.manager(max_segments=1).recover)

    def test_c205_unexpected_exception_after_seal_is_unknown_and_recoverable(self):
        def fault(phase):
            if phase=='after_seal':raise OSError('do-not-expose-private-path')
        self.reason('AUDIT_ROLLOVER_TRANSITION_UNKNOWN',self.manager(fault_hook=fault).rotate,self.journal,quiesced=True)
        recovered=self.rollover.recover();self.addCleanup(recovered.close)
        self.assertIsNotNone(recovered.parent_segment)

    def remove_meta(self,database,key):
        connection=sqlite3.connect(self.paths.runtime_path(database))
        try:
            connection.execute('DROP TRIGGER immutable_meta_DELETE')
            connection.execute('DELETE FROM meta WHERE key=?',(key,))
            connection.execute("CREATE TRIGGER immutable_meta_DELETE BEFORE DELETE ON meta BEGIN SELECT RAISE(ABORT,'immutable audit'); END")
        finally:connection.close()

    def test_c205_missing_seal_metadata_cannot_reopen_root(self):
        child=self.rotate();child.close();self.remove_meta('audit.sqlite','sealed')
        self.reason('AUDIT_ROLLOVER_INTEGRITY',self.rollover.recover)
        self.reason('AUDIT_ROLLOVER_INTEGRITY',self.source)

    def test_c205_missing_parent_metadata_cannot_reopen_child(self):
        child=self.rotate();database=child.parent_segment['next_database'];child.close()
        self.remove_meta(database,'parent_segment')
        self.reason('AUDIT_ROLLOVER_INTEGRITY',self.source,database=database)

    def test_c201_rotation_cannot_exceed_its_recovery_depth(self):
        self.reason('AUDIT_ROLLOVER_LIMIT',self.manager(max_segments=1).rotate,self.journal,quiesced=True)
        self.assertEqual(self.journal.health()['state'],'READY')

    def test_c203_tighter_resource_guards_survive_rotation_and_restart(self):
        self.journal.close()
        self.journal=self.source(max_bytes=1024*1024,max_inflight=2)
        child=self.rotate()
        self.assertEqual(child.max_bytes,1024*1024)
        self.assertEqual(child.max_inflight,2)
        child.close()
        recovered=self.rollover.recover();self.addCleanup(recovered.close)
        self.assertEqual(recovered.max_bytes,1024*1024)
        self.assertEqual(recovered.max_inflight,2)

    def test_c203_ordinary_child_open_preserves_tighter_ancestral_guards(self):
        self.journal.close();self.journal=self.source(max_bytes=1024*1024,max_inflight=2)
        child=self.rotate();database=child.parent_segment['next_database'];child.close()
        reopened=self.source(database=database)
        self.assertEqual(reopened.max_bytes,1024*1024)
        self.assertEqual(reopened.max_inflight,2)

    def test_c205_malformed_parent_json_has_fixed_startup_error(self):
        child=self.rotate();database=child.parent_segment['next_database'];child.close()
        connection=sqlite3.connect(self.paths.runtime_path(database))
        try:
            connection.execute('DROP TRIGGER immutable_meta_UPDATE')
            connection.execute("UPDATE meta SET value='{' WHERE key='parent_segment'")
            connection.execute("CREATE TRIGGER immutable_meta_UPDATE BEFORE UPDATE ON meta BEGIN SELECT RAISE(ABORT,'immutable audit'); END")
        finally:connection.close()
        self.reason('AUDIT_ROLLOVER_INTEGRITY',self.source,database=database)
