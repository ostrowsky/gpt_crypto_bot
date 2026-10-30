"""R2A05 native provenance: real app journal/client, entirely fake transport."""
from contextlib import closing
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import _support
from binance_demo_bot.adapter import DemoClock, DemoReadClient, HttpResponse
from binance_demo_bot.interaction_audit import AuditJournal
from binance_demo_bot.isolation import AppPaths


ENDPOINT='https://demo-api.binance.com/api/v3/klines'
PARAMS=dict(symbol='币安人生USDT',interval='1m',startTime=900000,endTime=1799999,limit=15,timeZone='0')
ROWS=[[900000,'100','101','99','100','1',959999,'100',2,'.4','40','0']]


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


class NativeProbeAuditTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.intrabar_audit')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.paths=AppPaths(Path(self.tmp.name));self.folder=self.paths.runtime_path('probe')
        self.folder.mkdir(parents=True);self.audit_path=self.paths.runtime_path('probe-audit.sqlite')

    def native(self, *, timeout_first=False, cache_name='one.json'):
        journal=AuditJournal(self.paths,account_alias='public-research',episode_id='synthetic-probe',
            scope_id='native-intrabar-source',database='probe-audit.sqlite')
        self.addCleanup(journal.close)
        clock_values=[1800000000000,10000000000]
        clock=DemoClock(wall_ms=lambda:clock_values[0],monotonic_ns=lambda:clock_values[1])
        replies=[TimeoutError('synthetic timeout')] if timeout_first else []
        replies.append(HttpResponse(200,json.dumps(ROWS,ensure_ascii=False,indent=1).encode(),{}))
        outer=self
        class Transport:
            def request(self,method,path,query,headers,timeout_s):
                outer.assertEqual((method,path,headers),('GET','/api/v3/klines',{}))
                clock_values[0]+=10;clock_values[1]+=10000000
                result=replies.pop(0)
                if isinstance(result,Exception):raise result
                return result
            def close(self):pass
        client=DemoReadClient(audit=journal,clock=clock,transport=Transport(),sleep=lambda _:None)
        rows,metadata=client.request_public('/api/v3/klines',dict(PARAMS),weight=2)
        client.close();journal.close()
        envelope=dict(endpoint=ENDPOINT,parameters=dict(PARAMS),rows=rows,metadata=metadata,source_sha256=digest(rows))
        target=self.folder/'responses'/cache_name;target.parent.mkdir(exist_ok=True)
        target.write_text(canonical(envelope),encoding='utf-8')
        return target

    def cache_edit(self,path,change):
        value=json.loads(path.read_text(encoding='utf-8'));change(value)
        path.write_text(canonical(value),encoding='utf-8')

    def records_edit(self,change):
        with closing(sqlite3.connect(self.audit_path)) as con,con:
            rows=[json.loads(r[0]) for r in con.execute('SELECT payload_json FROM records ORDER BY seq')]
            change(rows)
            for name, in con.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='records'").fetchall():
                con.execute('DROP TRIGGER '+name)
            con.execute('DELETE FROM records');previous='0'*64
            for i,value in enumerate(rows,1):
                value.update(seq=i,previous_sha256=previous);previous=digest(value)
                con.execute('INSERT INTO records VALUES (?,?,?,?,?)',(i,value['event_id'],value['kind'],canonical(value),previous))

    def verify(self):return self.api.verify_probe_audit(self.folder,self.audit_path)

    def test_R2A05_real_closed_public_unicode_run_is_verified_readonly(self):
        self.native();before=self.audit_path.read_bytes();names={p.name for p in self.audit_path.parent.iterdir()}
        result=self.verify()
        self.assertEqual(result['state'],'VERIFIED_RECORDED_PUBLIC_ATTEMPTS')
        self.assertEqual(result['attempts_n'],1);self.assertEqual(result['linked_cached_responses_n'],1)
        self.assertEqual(result['errors_n'],0);self.assertEqual(len(result['runids']),1)
        self.assertEqual(self.audit_path.read_bytes(),before)
        self.assertEqual({p.name for p in self.audit_path.parent.iterdir()},names)
        self.assertEqual(len(result['tip']),64)

    def test_R2A05_timeout_then_success_preserves_both_native_attempts(self):
        self.native(timeout_first=True)
        result=self.verify()
        self.assertEqual(result['state'],'VERIFIED_RECORDED_PUBLIC_ATTEMPTS')
        self.assertEqual(result['attempts_n'],2);self.assertEqual(result['errors_n'],1)
        self.assertEqual(result['linked_cached_responses_n'],1)

    def test_R2A05_two_closed_runs_are_distinct_and_complete(self):
        self.native();self.native(cache_name='two.json')
        result=self.verify()
        self.assertEqual(result['state'],'VERIFIED_RECORDED_PUBLIC_ATTEMPTS')
        self.assertEqual(len(result['runids']),2);self.assertEqual(result['attempts_n'],2)

    def test_R2A05_absent_native_journal_is_unknown_and_not_created(self):
        result=self.verify()
        self.assertEqual(result['state'],'UNKNOWN');self.assertFalse(self.audit_path.exists())

    def test_R2A05_open_run_is_unknown_even_with_complete_response_pair(self):
        self.native();self.records_edit(lambda rows:rows.pop())
        self.assertEqual(self.verify()['state'],'UNKNOWN')

    def test_R2A05_unfinalized_nonempty_wal_is_unknown_and_never_ignored(self):
        self.native()
        wal=self.audit_path.with_name(self.audit_path.name+'-wal')
        wal.write_bytes(b'pending-uncheckpointed-native-audit')
        self.assertEqual(self.verify()['state'],'UNKNOWN')
        self.assertEqual(wal.read_bytes(),b'pending-uncheckpointed-native-audit')

    def test_R2A05_unmatched_begin_is_unknown_not_a_response(self):
        self.native()
        self.records_edit(lambda rows:rows.pop(2))
        result=self.verify()
        self.assertNotEqual(result['state'],'VERIFIED_RECORDED_PUBLIC_ATTEMPTS')

    def test_R2A05_missing_cache_audit_metadata_is_unknown(self):
        path=self.native();self.cache_edit(path,lambda r:r['metadata'].pop('audit_attempt_id'))
        self.assertEqual(self.verify()['state'],'UNKNOWN')

    def test_R2A05_cache_receipt_or_raw_sha_contradiction_fails(self):
        for field,value in (('received_at_ms',1),('raw_body_sha256','0'*64)):
            with self.subTest(field=field):
                path=self.native(cache_name=field+'.json')
                self.cache_edit(path,lambda r:r['metadata'].__setitem__(field,value))
                self.assertEqual(self.verify()['state'],'FAIL')

    def test_R2A05_cache_rows_or_parameters_must_match_raw_native_attempt(self):
        path=self.native()
        self.cache_edit(path,lambda r:r['rows'][0].__setitem__(1,'999'))
        self.assertEqual(self.verify()['state'],'FAIL')

    def test_R2A05_request_parameters_cannot_borrow_another_native_receipt(self):
        path=self.native();self.cache_edit(path,lambda r:r['parameters'].__setitem__('symbol','OTHERUSDT'))
        self.assertEqual(self.verify()['state'],'FAIL')

    def test_R2A05_chain_or_sequence_tamper_fails(self):
        self.native()
        with closing(sqlite3.connect(self.audit_path)) as con,con:
            con.execute('DROP TRIGGER immutable_records_UPDATE')
            con.execute("UPDATE records SET content_sha256=? WHERE seq=2",('0'*64,))
        self.assertEqual(self.verify()['state'],'FAIL')

    def test_R2A05_unknown_endpoint_or_private_method_fails_even_valid_chain(self):
        self.native()
        def edit(rows):
            for row in rows:
                if row['kind'].startswith('REST_'):row['data'].update(method='POST',private=True,endpoint='/api/v3/order')
        self.records_edit(edit)
        self.assertEqual(self.verify()['state'],'FAIL')

    def test_R2A05_terminal_must_match_parent_attempt_and_request(self):
        self.native()
        self.records_edit(lambda rows:rows[2]['data'].__setitem__('attempt_id','wrong-attempt'))
        self.assertEqual(self.verify()['state'],'FAIL')

    def test_R2A05_double_terminal_and_malformed_public_bytes_fail(self):
        self.native()
        def duplicate(rows):
            extra=copy.deepcopy(rows[2]);extra['event_id']='different-event';rows.insert(3,extra)
        self.records_edit(duplicate)
        self.assertEqual(self.verify()['state'],'FAIL')

    def test_R2A05_exact_body_wrapper_is_verified_beyond_chain(self):
        self.native()
        self.records_edit(lambda rows:rows[2]['data']['payload'].__setitem__('body_base64','%%%'))
        self.assertEqual(self.verify()['state'],'FAIL')


if __name__=='__main__':unittest.main()
