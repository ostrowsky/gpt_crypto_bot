"""C2b1 append-only hold history and conservative certification contracts."""
import importlib
from pathlib import Path
import tempfile
import unittest

import _support
from binance_demo_bot.adapter import AdapterError

from test_audit_backup import IDENTITY

SEAL='1'*64
EVIDENCE='2'*64
REF='a'*32
REQUEST='b'*32


class RetentionLedgerTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.audit_retention')
        from binance_demo_bot.isolation import AppPaths
        base=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures';base.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=base);self.addCleanup(temporary.cleanup)
        self.paths=AppPaths(Path(temporary.name).resolve());self.ledger=self.make()
    def make(self,**kwargs):
        value=self.api.RetentionLedger(self.paths,**dict(IDENTITY,**kwargs));self.addCleanup(value.close);return value
    def place(self,**kwargs):
        values=dict(category='RESEARCH',reference_id=REF,evidence_sha256=EVIDENCE,request_id=REQUEST)
        values.update(kwargs);return self.ledger.place(SEAL,**values)
    def reason(self,code,call,*args,**kwargs):
        with self.assertRaises(AdapterError) as caught:call(*args,**kwargs)
        self.assertEqual(caught.exception.reason,code)

    def test_c215_place_release_preserves_history_and_denominators(self):
        hold=self.place();before=self.ledger.status(SEAL)
        self.assertEqual((before['placed_N'],before['active_n'],before['released_n']),(1,1,0))
        self.assertEqual(before['active_ids'],[hold])
        self.ledger.release(hold,evidence_sha256='3'*64,request_id='c'*32)
        after=self.ledger.status(SEAL)
        self.assertEqual((after['placed_N'],after['active_n'],after['released_n']),(1,0,1))
        self.assertFalse(after['deletion_eligible'])
        self.assertEqual(after['certification'],'UNAVAILABLE')
        self.assertEqual(after['state'],'UNKNOWN')

    def test_c216_zero_holds_is_not_certified_clear(self):
        result=self.ledger.status(SEAL)
        self.assertEqual(result['placed_N'],0);self.assertEqual(result['state'],'UNKNOWN')
        self.assertFalse(result['deletion_eligible'])
        self.assertFalse(hasattr(self.ledger,'delete'))

    def test_c215_idempotent_requests_do_not_duplicate_hold(self):
        first=self.place();self.assertEqual(self.place(),first)
        self.assertEqual(self.ledger.status(SEAL)['placed_N'],1)

    def test_c215_conflicting_request_reuse_is_denied(self):
        self.place();self.reason('AUDIT_HOLD_CONFLICT',self.place,category='ACTIVE_EXPOSURE')

    def test_c215_release_request_is_idempotent_but_not_independent_duplicate(self):
        hold=self.place();kwargs=dict(evidence_sha256=EVIDENCE,request_id='c'*32)
        first=self.ledger.release(hold,**kwargs)
        self.assertEqual(self.ledger.release(hold,**kwargs),first)
        self.reason('AUDIT_HOLD_RELEASED',self.ledger.release,hold,evidence_sha256=EVIDENCE,request_id='d'*32)

    def test_c215_missing_hold_and_invalid_inputs_denied(self):
        self.reason('AUDIT_HOLD_MISSING',self.ledger.release,'f'*32,evidence_sha256=EVIDENCE,request_id='c'*32)
        self.reason('AUDIT_HOLD_INVALID',self.place,category='CLEAR')
        self.reason('AUDIT_HOLD_INVALID',self.place,reference_id='../outside')

    def test_c216_frozen_prefix_is_not_changed_by_later_release(self):
        hold=self.place();first=self.ledger.status(SEAL)
        self.ledger.release(hold,evidence_sha256=EVIDENCE,request_id='c'*32)
        self.assertEqual(self.ledger.status(SEAL,through_seq=first['prefix_last_seq']),first)

    def test_c215_restart_retains_holds_identity_and_release(self):
        hold=self.place();self.ledger.close();self.ledger=self.make()
        self.assertEqual(self.ledger.status(SEAL)['active_ids'],[hold])
        self.ledger.release(hold,evidence_sha256=EVIDENCE,request_id='c'*32)
        self.ledger.close();self.ledger=self.make()
        self.assertEqual(self.ledger.status(SEAL)['released_n'],1)

    def test_c215_unknown_commit_never_returns_same_process_success(self):
        def fault(phase):
            if phase=='after_commit':raise OSError('private detail')
        self.ledger._journal.fault_hook=fault
        self.reason('AUDIT_COMMIT_UNKNOWN',self.place)
        self.reason('AUDIT_COMMIT_UNKNOWN',self.place)
        self.ledger.close();self.ledger=self.make()
        self.assertIsInstance(self.place(),str)
        self.assertEqual(self.ledger.status(SEAL)['placed_N'],1)

    def test_c216_active_id_list_is_bounded_without_losing_counts(self):
        for i in range(105):self.place(request_id=f'{i:032x}')
        result=self.ledger.status(SEAL)
        self.assertEqual(result['active_n'],105);self.assertEqual(result['returned_n'],100)
        self.assertTrue(result['truncated']);self.assertFalse(result['deletion_eligible'])

    def test_c215_writer_ownership_is_exclusive(self):
        self.reason('AUDIT_HOLD_BUSY',self.make)

    def test_c216_phantom_release_cannot_clear_holds(self):
        self.place()
        self.ledger._journal.append('DECISION',dict(scope='RETENTION_HOLD',operation='RELEASE',
            seal_sha256=SEAL,hold_event_id='f'*32,evidence_sha256=EVIDENCE,request_id='c'*32))
        result=self.ledger.status(SEAL)
        self.assertEqual(result['invalid_n'],1);self.assertEqual(result['state'],'FAIL')
        self.assertEqual(result['active_n'],1);self.assertFalse(result['deletion_eligible'])

    def test_c216_status_exposes_unknown_persistence_failure(self):
        def fault(phase):
            if phase=='after_commit':raise OSError('synthetic')
        self.ledger._journal.fault_hook=fault
        self.reason('AUDIT_COMMIT_UNKNOWN',self.place)
        result=self.ledger.status(SEAL)
        self.assertEqual(result['ledger_state'],'FAILED')
        self.assertEqual(result['failure_reason'],'AUDIT_COMMIT_UNKNOWN')
        self.assertFalse(result['deletion_eligible'])

    def test_c216_schema_drift_is_fail_with_a_mutation_denominator(self):
        self.place()
        self.ledger._journal.append('DECISION',dict(scope='RETENTION_HOLD',operation='CLEAR',
            seal_sha256=SEAL,request_id='c'*32))
        result=self.ledger.status(SEAL)
        self.assertEqual(result['state'],'FAIL');self.assertEqual(result['invalid_n'],1)
        self.assertEqual(result['mutations_N'],2)

    def test_c216_malformed_release_does_not_clear_a_valid_hold(self):
        hold=self.place()
        self.ledger._journal.append('DECISION',dict(scope='RETENTION_HOLD',operation='RELEASE',
            seal_sha256=SEAL,hold_event_id=hold,request_id='c'*32))
        result=self.ledger.status(SEAL)
        self.assertEqual(result['active_n'],1);self.assertEqual(result['invalid_n'],1)
        self.assertEqual(result['state'],'FAIL')

    def test_c216_invalid_counts_do_not_include_another_seal(self):
        hold=self.place()
        self.ledger.release(hold,evidence_sha256=EVIDENCE,request_id='c'*32)
        for seal,request in ((SEAL,'d'*32),('3'*64,'e'*32)):
            self.ledger._journal.append('DECISION',dict(scope='RETENTION_HOLD',operation='RELEASE',
                seal_sha256=seal,hold_event_id=hold,evidence_sha256=EVIDENCE,request_id=request))
        result=self.ledger.status(SEAL)
        self.assertEqual(result['mutations_N'],3);self.assertEqual(result['invalid_n'],2)
        foreign=self.ledger.status('3'*64)
        self.assertEqual(foreign['mutations_N'],1);self.assertEqual(foreign['invalid_n'],1)
