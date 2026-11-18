"""TA12 complete capture audit, before implementation."""
import json
from pathlib import Path
import tempfile
import unittest
import _support
from test_public_flow_dataset import trade
from binance_demo_bot import public_flow_dataset as flow
from binance_demo_bot.flow_capture_audit import audit_capture

class FlowAuditTests(unittest.TestCase):
    def setUp(self):
        app=Path(__file__).resolve().parents[1]
        self.tmp=tempfile.TemporaryDirectory(dir=app/'.runtime');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'capture';self.reg=flow.make_registration(1793910000000)
        v=flow.FlowValidator();store=flow.ChunkStore(self.root,self.reg,free_bytes=lambda:40*1024**3)
        self.addCleanup(store.close)
        start=self.reg['started_at_ms'];at=self.reg['full_days_start_ms']+1000
        store.append(v.begin_session('one',wall_ms=start,mono_ns=1),monotonic_ns=1)
        store.append(v.ingest(trade(),wall_ms=at,mono_ns=2,session_id='one'),monotonic_ns=2)
        store.append(v.ingest(trade(2,12,13),wall_ms=at+6001,mono_ns=3,session_id='one'),monotonic_ns=3)
        store.append(dict(kind='SESSION_END',session_id='one',wall_ms=at+6002,mono_ns=4,raw=None,reason='PLANNED_END'),monotonic_ns=4)
        store.close();status=dict(v.summary(),state='CAPTURE_FINISHED',stop_reason='PLANNED_END',calendar_schedule_finished=True,
            full_days_start_ms=self.reg['full_days_start_ms'],planned_end_ms=self.reg['planned_end_ms'],registration_sha256=flow._sha(self.reg))
        flow._atomic_json(self.root/'status.json',status)

    def test_TA12_integrity_is_not_complete_coverage_or_income(self):
        p=audit_capture(self.root)
        self.assertEqual(p['integrity'],'PASS');self.assertEqual(p['records_n'],4)
        self.assertEqual(p['recorded_gaps_n'],1);self.assertEqual(p['market_records_n'],2)
        self.assertEqual(p['planned_days_with_records_n'],1)
        self.assertEqual(p['model_training'],'NOT_TRAINED');self.assertFalse(p['financial_certification'])
        self.assertFalse(p['continuous_coverage']);self.assertEqual(p['stored_validation_profile'],'PINNED_PRODUCER_VALIDATION_NOT_INDEPENDENT_RAW_REPARSE')

    def test_TA12_chunk_tampering_rejected(self):
        p=next(self.root.glob('chunk-*.gz'));p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaises(ValueError):audit_capture(self.root)

    def test_TA12_status_counts_cannot_hide_missing_committed_records(self):
        p=self.root/'status.json';s=json.loads(p.read_text());s['messages_per_topic']['btcusdt@aggTrade']+=1
        flow._atomic_json(p,s)
        with self.assertRaisesRegex(ValueError,'STATUS_COUNTER'):audit_capture(self.root)

    def test_TA12_producer_source_change_cannot_verify_stored_semantics(self):
        p=self.root/'registration.json';s=json.loads(p.read_text());s['source_sha256']='a'*64;flow._atomic_json(p,s)
        m=self.root/'manifest.json';r=json.loads(m.read_text());r['registration_sha256']=flow._sha(s);flow._atomic_json(m,r)
        with self.assertRaises(ValueError):audit_capture(self.root)


class FlowStreamingTests(unittest.TestCase):
    setUp=FlowAuditTests.setUp
    def test_FAS01_decompressed_reads_are_at_most_64KiB(self):
        from unittest.mock import patch
        import binance_demo_bot.flow_capture_audit as audit
        real=audit.gzip.GzipFile
        class Guard:
            def __init__(s,*a,**kw):s.inner=real(*a,**kw)
            def __enter__(s):s.inner.__enter__();return s
            def __exit__(s,*args):return s.inner.__exit__(*args)
            def read(s,size=-1):
                if not 0<size<=65536:raise AssertionError('DECOMPRESSED_READ_EXCEEDS_64K')
                return s.inner.read(size)
        with patch.object(audit.gzip,'GzipFile',Guard):
            p=audit.audit_capture(self.root)
        self.assertEqual(p['records_n'],4)

    def test_FAS02_records_crossing_read_boundaries_preserve_exact_bytes(self):
        import io
        from binance_demo_bot.flow_capture_audit import iter_bounded_lines
        raw=b'{"x":"'+b'a'*70000+b'"}\n{"last":1}'
        self.assertEqual(list(iter_bounded_lines(io.BytesIO(raw))),[raw[:70009],raw[70009:]])

    def test_FAS02_oversized_record_rejected(self):
        import io
        from binance_demo_bot.flow_capture_audit import iter_bounded_lines
        with self.assertRaisesRegex(ValueError,'RECORD_LIMIT'):
            list(iter_bounded_lines(io.BytesIO(b'x'*101),record_limit=100))

if __name__=='__main__':unittest.main()
