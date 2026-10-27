"""AUD-C1 spec-first real journal/gzip/filesystem tests; no keys or network."""
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import subprocess
import sys
import tempfile
import threading
import unittest

import _support
from binance_demo_bot.adapter import AdapterError
from binance_demo_bot.interaction_audit import AuditJournal, _json, _sha
from binance_demo_bot.isolation import AppPaths
from binance_demo_bot.journal import _WriterLease


A='a'*32
B='b'*32


class AuditArchiveTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.audit_archive')
        base=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures'
        base.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=base);self.addCleanup(temporary.cleanup)
        self.paths=AppPaths(Path(temporary.name).resolve())
        self.journal=AuditJournal(self.paths,account_alias='synthetic',episode_id='one',scope_id='all')
        self.addCleanup(self.journal.close)
        self.archive=self.make()

    def make(self,**kwargs):return self.api.AuditArchive(self.paths,**kwargs)
    def directory(self,identifier=A):return self.paths.runtime_path('audit-archives/'+identifier)
    def export(self,**kwargs):return self.archive.export(self.journal,archive_id=A,**kwargs)
    def reason(self,code,call,*args,**kwargs):
        with self.assertRaises(AdapterError) as caught:call(*args,**kwargs)
        self.assertEqual(caught.exception.reason,code)
    def records(self):
        with gzip.open(self.directory()/'records.jsonl.gz','rb') as stream:
            return [json.loads(line) for line in stream]
    def rewrite(self,*,rows=None,changes=None):
        directory=self.directory();path=directory/'manifest.json'
        value=json.loads(path.read_text(encoding='utf-8'))
        if rows is not None:
            raw=b''.join((_json(row)+'\n').encode() for row in rows)
            compressed=gzip.compress(raw,mtime=0)
            (directory/'records.jsonl.gz').write_bytes(compressed)
            value.update(compressed_sha256=hashlib.sha256(compressed).hexdigest(),
                compressed_bytes_n=len(compressed),uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                uncompressed_bytes_n=len(raw))
        value.update(changes or {})
        value.pop('manifest_sha256');value['manifest_sha256']=_sha(value)
        path.write_text(_json(value),encoding='utf-8')

    def test_c101_roundtrip_original_envelopes_and_external_pin(self):
        self.journal.append('DECISION',dict(action='WAIT'),wall_ms=10,monotonic_ns=4)
        expected=self.journal.read(limit=1000)
        result=self.export();manifest=result['manifest']
        self.assertEqual(self.records(),expected)
        self.assertEqual(manifest['rows_n'],2)
        self.assertEqual(manifest['chain_tip'],expected[-1]['content_sha256'])
        self.assertEqual(manifest['kind_counts']['DECISION'],1)
        self.assertEqual(manifest['denominator_basis'],'DURABLE_EXPORTED_PREFIX')
        self.assertEqual(manifest['capture_coverage'],'UNKNOWN')
        verified=self.archive.verify(A,expected_identity=self.journal.identity,
            expected_manifest_sha256=manifest['manifest_sha256'])
        self.assertEqual(verified['authenticity'],'PINNED_MANIFEST')
        self.assertEqual(verified['integrity'],'INTERNAL_VERIFIED')
        self.assertFalse(manifest['full_audit_complete'])

    def test_c101_explicit_prefix_excludes_later_rows(self):
        self.journal.append('DECISION',dict(action='WAIT'))
        self.journal.append('DECISION',dict(action='HOLD'))
        result=self.export(through_seq=2)
        self.assertEqual([r['seq'] for r in self.records()],[1,2])
        self.assertEqual(result['manifest']['last_seq'],2)

    def test_c101_empty_prefix_has_no_invented_coverage(self):
        result=self.export(through_seq=0)
        self.assertEqual(self.records(),[])
        self.assertEqual(result['manifest']['rows_n'],0)
        self.assertIsNone(result['manifest']['first_seq'])
        self.assertEqual(result['manifest']['chain_tip'],'0'*64)
        self.assertIsNone(result['manifest']['min_wall_ms'])

    def test_c101_concurrent_append_is_not_blocked_or_exported(self):
        completed=threading.Event();errors=[]
        def append():
            try:self.journal.append('DECISION',dict(action='WAIT'))
            except BaseException as error:errors.append(type(error).__name__)
            finally:completed.set()
        def hook(phase):
            if phase=='snapshot_pinned':
                worker=threading.Thread(target=append);worker.start()
                self.assertTrue(completed.wait(2),'export retained the journal lock')
                worker.join(2)
        result=self.make(fault_hook=hook).export(self.journal,archive_id=A)
        self.assertEqual(errors,[])
        self.assertEqual(result['manifest']['rows_n'],1)
        self.assertEqual(len(self.journal.read()),2)

    def test_c103_export_preserves_source_counters_rows_and_health(self):
        rows=self.journal.read();health=self.journal.health()
        self.export()
        self.assertEqual(self.journal.read(),rows)
        self.assertEqual(self.journal.health(),health)

    def test_c101_failed_source_is_forensic_not_healthy(self):
        self.journal._failed('AUDIT_COMMIT_UNKNOWN')
        result=self.export()
        self.assertEqual(result['manifest']['source_state_at_pin'],'FAILED')
        self.assertEqual(result['manifest']['capture_coverage'],'UNKNOWN')
        self.assertEqual(self.journal.health()['error'],'AUDIT_COMMIT_UNKNOWN')

    def test_c101_closed_source_denied(self):
        self.journal.close();self.reason('AUDIT_ARCHIVE_SOURCE_CLOSED',self.export)

    def test_c101_foreign_runtime_denied(self):
        other=self.paths.root/'other';other.mkdir()
        self.reason('AUDIT_ARCHIVE_IDENTITY',self.api.AuditArchive(AppPaths(other)).export,self.journal)

    def test_c101_invalid_cutoffs_denied(self):
        for cutoff in (-1,2,True,1.5,'1'):
            with self.subTest(cutoff=cutoff):self.reason('AUDIT_ARCHIVE_INVALID',self.export,through_seq=cutoff)

    def test_c103_duplicate_id_cannot_overwrite(self):
        first=self.export();original=(self.directory()/'manifest.json').read_bytes()
        self.reason('AUDIT_ARCHIVE_EXISTS',self.export)
        self.assertEqual((self.directory()/'manifest.json').read_bytes(),original)
        self.assertEqual(self.archive.verify(A)['manifest'],first['manifest'])

    def test_c104_truncated_compressed_file_is_rejected(self):
        self.export();path=self.directory()/'records.jsonl.gz'
        path.write_bytes(path.read_bytes()[:-7])
        self.reason('AUDIT_ARCHIVE_INTEGRITY',self.archive.verify,A)

    def test_c104_rehashed_wrong_row_count_is_rejected(self):
        self.export();self.rewrite(changes=dict(rows_n=9))
        self.reason('AUDIT_ARCHIVE_INTEGRITY',self.archive.verify,A)

    def test_c104_repacked_forged_event_chain_is_rejected(self):
        self.export();rows=self.records();rows[0]['data']['scope']='FORGED'
        self.rewrite(rows=rows)
        self.reason('AUDIT_ARCHIVE_INTEGRITY',self.archive.verify,A)

    def test_c104_wrong_identity_and_external_pin_are_rejected(self):
        self.export()
        self.reason('AUDIT_ARCHIVE_IDENTITY',self.archive.verify,A,expected_identity={'foreign':True})
        self.reason('AUDIT_ARCHIVE_PIN_MISMATCH',self.archive.verify,A,expected_manifest_sha256='0'*64)
        self.assertEqual(self.archive.verify(A)['authenticity'],'UNVERIFIED')

    def test_c104_extra_files_and_invalid_manifest_are_rejected(self):
        self.export();extra=self.directory()/'unexpected';extra.write_text('synthetic')
        self.reason('AUDIT_ARCHIVE_LAYOUT',self.archive.verify,A)
        extra.unlink();(self.directory()/'manifest.json').write_text('{}')
        self.reason('AUDIT_ARCHIVE_INTEGRITY',self.archive.verify,A)

    def test_c105_export_compressed_limit_is_explicit(self):
        self.reason('AUDIT_ARCHIVE_LIMIT',self.make(max_compressed_bytes=64).export,self.journal,archive_id=A)
        self.assertFalse(self.directory().exists())
        self.assertEqual(self.journal.health()['state'],'READY')

    def test_c105_export_uncompressed_limit_is_explicit(self):
        self.reason('AUDIT_ARCHIVE_LIMIT',self.make(max_uncompressed_bytes=64).export,self.journal,archive_id=A)

    def test_c105_verifier_decompression_budget_is_enforced(self):
        self.journal.append('DECISION',dict(padding='z'*10000));self.export()
        self.reason('AUDIT_ARCHIVE_LIMIT',self.make(max_uncompressed_bytes=2048).verify,A)

    def test_c105_storage_and_count_limits_preserve_existing_evidence(self):
        self.export();original=(self.directory()/'manifest.json').read_bytes()
        self.reason('AUDIT_ARCHIVE_LIMIT',self.make(max_total_bytes=1).export,self.journal,archive_id=B)
        self.reason('AUDIT_ARCHIVE_LIMIT',self.make(max_archives=1).export,self.journal,archive_id=B)
        self.assertEqual((self.directory()/'manifest.json').read_bytes(),original)

    def test_c103_prepublication_failure_leaves_incomplete_only(self):
        def fault(phase):
            if phase=='before_publish':raise OSError('do-not-expose-private-path')
        self.reason('AUDIT_ARCHIVE_WRITE_FAILED',self.make(fault_hook=fault).export,self.journal,archive_id=A)
        self.assertFalse(self.directory().exists())
        inventory=self.archive.inventory()
        self.assertEqual(inventory['incomplete_n'],1)
        self.assertEqual(inventory['state'],'UNKNOWN')
        self.assertEqual(inventory['deletion_eligible_n'],0)
        self.assertEqual(self.journal.health()['state'],'READY')

    def test_c103_postpublication_failure_is_unknown_and_recoverable(self):
        def fault(phase):
            if phase=='after_publish':raise OSError('do-not-expose-private-path')
        self.reason('AUDIT_ARCHIVE_PUBLICATION_UNKNOWN',self.make(fault_hook=fault).export,self.journal,archive_id=A)
        self.assertEqual(self.archive.verify(A)['integrity'],'INTERNAL_VERIFIED')
        self.assertEqual(self.archive.inventory()['valid_n'],1)

    def test_c106_inventory_preserves_missing_protected_ids_as_unknown(self):
        self.export();result=self.archive.inventory(protected_archive_ids=(A,B))
        self.assertEqual(result['policy_version'],'PRESERVE_ALL_V1')
        self.assertEqual(result['protected_present_n'],1)
        self.assertEqual(result['protected_expected_N'],2)
        self.assertEqual(result['protected_missing_ids'],[B])
        self.assertEqual(result['state'],'UNKNOWN')
        self.assertEqual(result['deletion_eligible_n'],0)
        self.assertFalse(hasattr(self.archive,'delete'))

    def test_c106_corrupt_inventory_is_fail(self):
        self.export();(self.directory()/'records.jsonl.gz').write_bytes(b'bad')
        result=self.archive.inventory()
        self.assertEqual(result['invalid_n'],1);self.assertEqual(result['state'],'FAIL')
        self.assertEqual(result['deletion_eligible_n'],0)

    def test_c104_ids_cannot_escape_runtime(self):
        for identifier in ('../escape','a/b','C:/outside','A'*32,'a'*31):
            with self.subTest(identifier=identifier):
                self.reason('AUDIT_ARCHIVE_INVALID',self.archive.export,self.journal,archive_id=identifier)
                self.reason('AUDIT_ARCHIVE_INVALID',self.archive.verify,identifier)

    def test_c104_junction_escape_is_rejected_without_target_writes(self):
        target=self.paths.root/'outside-runtime';target.mkdir()
        link=self.directory()
        if os.name=='nt':
            created=subprocess.run(['cmd.exe','/c','mklink','/J',str(link),str(target)],capture_output=True,text=True)
            self.assertEqual(created.returncode,0,created.stderr)
        else:link.symlink_to(target,target_is_directory=True)
        try:
            self.reason('AUDIT_ARCHIVE_LAYOUT',self.archive.verify,A)
            self.assertEqual(list(target.iterdir()),[])
        finally:
            if os.name=='nt':link.rmdir()
            else:link.unlink()

    def test_c103_archive_writer_lease_is_exclusive(self):
        lease=_WriterLease(self.paths.runtime_path('audit-archives/.writer.lock'))
        try:self.reason('AUDIT_ARCHIVE_BUSY',self.export)
        finally:lease.close()

    def test_c102_redacted_payload_roundtrip_never_restores_secrets(self):
        self.journal.append('DECISION',dict(api_key='synthetic-key',note='synthetic-secret'),secrets=('synthetic-secret',))
        result=self.export();raw=json.dumps(self.records())+json.dumps(result)
        self.assertNotIn('synthetic-key',raw);self.assertNotIn('synthetic-secret',raw)

    def test_c101_source_row_tampering_denies_publication(self):
        self.journal._con.execute('DROP TRIGGER immutable_records_UPDATE')
        self.journal._con.execute("UPDATE records SET payload_json='{}' WHERE seq=1")
        self.reason('AUDIT_ARCHIVE_INTEGRITY',self.export)
        self.assertFalse(self.directory().exists())

    def test_c104_invalid_deflate_has_safe_fixed_error(self):
        self.export();path=self.directory()/'records.jsonl.gz'
        # Reserved BTYPE=3, inside an otherwise real gzip framing.
        raw=path.read_bytes();raw=raw[:10]+b'\x07'+raw[-8:]
        path.write_bytes(raw)
        self.rewrite(changes=dict(compressed_sha256=hashlib.sha256(raw).hexdigest(),compressed_bytes_n=len(raw)))
        self.reason('AUDIT_ARCHIVE_INTEGRITY',self.archive.verify,A)

    def test_c104_unknown_manifest_schema_and_boolean_counts_denied(self):
        self.export();self.rewrite(changes=dict(unregistered_field='unexpected'))
        self.reason('AUDIT_ARCHIVE_INTEGRITY',self.archive.verify,A)
        manifest=json.loads((self.directory()/'manifest.json').read_text())
        manifest.pop('unregistered_field');manifest.pop('manifest_sha256');manifest['first_seq']=True
        manifest['manifest_sha256']=_sha(manifest)
        (self.directory()/'manifest.json').write_text(_json(manifest))
        self.reason('AUDIT_ARCHIVE_INTEGRITY',self.archive.verify,A)

    def test_c101_nonobject_source_row_has_integrity_error(self):
        self.journal._con.execute('DROP TRIGGER immutable_records_UPDATE')
        self.journal._con.execute("UPDATE records SET payload_json='[]' WHERE seq=1")
        self.reason('AUDIT_ARCHIVE_INTEGRITY',self.export)

    def test_c105_forged_size_does_not_bypass_actual_inflation_limit(self):
        self.journal.append('DECISION',dict(padding='z'*10000));self.export()
        self.rewrite(changes=dict(uncompressed_bytes_n=1))
        self.reason('AUDIT_ARCHIVE_LIMIT',self.make(max_uncompressed_bytes=2048).verify,A)

    def test_c101_real_commit_unknown_exports_committed_row_as_forensic(self):
        self.journal.fault_hook=lambda phase:(_ for _ in ()).throw(OSError()) if phase=='after_commit' else None
        self.reason('AUDIT_COMMIT_UNKNOWN',self.journal.append,'DECISION',dict(action='WAIT'))
        result=self.export()
        self.assertEqual(result['manifest']['rows_n'],2)
        self.assertEqual(result['manifest']['source_state_at_pin'],'FAILED')
        self.assertEqual(self.journal.health()['channels']['DECISION']['recorded_n'],0)

    def crash(self,phase):
        self.journal.close()
        package=Path(__file__).resolve().parents[1]/'src'
        script='''import os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from binance_demo_bot.isolation import AppPaths
from binance_demo_bot.interaction_audit import AuditJournal
from binance_demo_bot.audit_archive import AuditArchive
paths=AppPaths(Path(sys.argv[2]))
journal=AuditJournal(paths,account_alias='synthetic',episode_id='one',scope_id='all')
def fault(phase):
    if phase==sys.argv[3]:os._exit(73)
AuditArchive(paths,fault_hook=fault).export(journal,archive_id='a'*32)
'''
        result=subprocess.run([sys.executable,'-I','-c',script,str(package),str(self.paths.root),phase],
            capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,73,result.stderr)

    def test_c103_process_death_before_publish_retains_incomplete(self):
        self.crash('before_publish')
        result=self.archive.inventory()
        self.assertEqual(result['incomplete_n'],1)
        self.assertEqual(result['valid_n'],0)
        self.assertFalse(self.directory().exists())

    def test_c103_process_death_after_publish_is_verifiable_after_restart(self):
        self.crash('after_publish')
        result=self.archive.inventory()
        self.assertEqual(result['valid_n'],1)
        self.assertEqual(result['incomplete_n'],0)
        self.assertEqual(self.archive.verify(A)['integrity'],'INTERNAL_VERIFIED')

    def test_c106_layout_violations_do_not_inflate_archive_denominator(self):
        self.export();(self.directory()/'unexpected').write_text('extra')
        result=self.archive.inventory()
        self.assertEqual(result['valid_n']+result['invalid_n']+result['incomplete_n'],result['archives_expected_N'])
        self.assertEqual(result['layout_errors_n'],1)
        self.assertEqual(result['invalid_n'],1)
        self.assertIsNone(result['stored_bytes_n'])
        self.assertGreater(result['observed_file_bytes_n'],0)

    def test_c104_writer_lease_junction_is_denied_before_open(self):
        target=self.paths.root/'outside';target.mkdir()
        link=self.paths.runtime_path('audit-archives/.writer.lock')
        if os.name=='nt':
            created=subprocess.run(['cmd.exe','/c','mklink','/J',str(link),str(target)],capture_output=True,text=True)
            self.assertEqual(created.returncode,0,created.stderr)
        else:link.symlink_to(target,target_is_directory=True)
        try:
            self.reason('AUDIT_ARCHIVE_LAYOUT',self.export)
            self.assertEqual(list(target.iterdir()),[])
        finally:
            if os.name=='nt':link.rmdir()
            else:link.unlink()

    def test_c105_long_record_line_is_bounded_independently(self):
        self.journal.append('DECISION',dict(padding='z'*5000));self.export()
        with patch.object(self.api,'LINE_LIMIT',2048):
            self.reason('AUDIT_ARCHIVE_LIMIT',self.archive.verify,A)
