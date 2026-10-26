"""AUD-A spec-first synthetic contracts; real own SQLite, no keys/network."""
import base64
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
import subprocess
import sys
from unittest.mock import patch

import _support
from binance_demo_bot.adapter import AdapterError, DemoClock, DemoReadClient, HttpResponse, BinanceDemoAdapter
from binance_demo_bot.isolation import AppPaths, DemoCredentials


class AuditCase(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('binance_demo_bot.interaction_audit')
        fixture_root=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures'
        fixture_root.mkdir(parents=True,exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=fixture_root)
        self.addCleanup(self.tmp.cleanup)
        self.paths = AppPaths(Path(self.tmp.name))
        self.journal = self.make()
        self.addCleanup(self.journal.close)
        self.wall, self.mono = 1800000000000, 10000000000
        self.clock = DemoClock(wall_ms=lambda:self.wall, monotonic_ns=lambda:self.mono)
        self.clock.observe(self.wall,self.wall,self.wall,self.mono,self.mono)

    def make(self, **changes):
        values = dict(account_alias='synthetic',episode_id='e1',scope_id='s1')
        values.update(changes)
        return self.api.AuditJournal(self.paths,**values)

    def rows(self, kind=None):
        rows = self.journal.read(limit=1000)
        return [r for r in rows if kind is None or r['kind']==kind]

    def reason(self, code, operation, *args, **kwargs):
        with self.assertRaises(AdapterError) as caught:operation(*args,**kwargs)
        self.assertEqual(caught.exception.reason,code)

    def client(self, replies):
        outer = self
        self.calls = []
        class Transport:
            def request(self, method,path,query,headers,timeout_s):
                # Real foreign connection verifies BEGIN is already durable.
                with outer.journal._lock:
                    foreign = outer.api.sqlite3.connect(outer.journal.path.as_uri()+'?mode=ro',uri=True)
                    try:
                        self.begin_count = foreign.execute("SELECT COUNT(*) FROM records WHERE kind='REST_BEGIN'").fetchone()[0]
                    finally:foreign.close()
                outer.calls.append((method,path,query,headers,self.begin_count))
                outer.wall += 10;outer.mono += 10000000
                reply = replies.pop(0)
                if isinstance(reply,BaseException):raise reply
                return reply
        return DemoReadClient(transport=Transport(),clock=self.clock,audit=self.journal,
            sleep=lambda seconds:None)

    def test_a01_durable_identity_chain_and_immutable_rows(self):
        eid = self.journal.append('DECISION',{'action':'WAIT'})
        rows = self.rows()
        self.assertEqual(rows[-1]['event_id'],eid)
        self.assertEqual(rows[-1]['previous_sha256'],rows[-2]['content_sha256'])
        self.assertEqual(rows[-1]['application_id'],'binance_demo_bot')
        self.assertEqual(len(rows[-1]['source_code_sha256']),64)
        self.assertEqual(self.journal._con.execute('PRAGMA synchronous').fetchone()[0],2)
        for sql in ('DELETE FROM records','UPDATE records SET kind=kind','DELETE FROM meta'):
            with self.assertRaises(self.api.sqlite3.IntegrityError):self.journal._con.execute(sql)
        self.journal.close()
        reopened = self.make();self.addCleanup(reopened.close)
        self.assertEqual(reopened.read(limit=1000)[:len(rows)],rows)
        reopened.close()
        self.reason('AUDIT_IDENTITY_MISMATCH',self.make,scope_id='other')

    def test_a01_tampering_is_rejected_at_reopen(self):
        self.journal.append('DECISION',{'action':'WAIT'})
        self.journal.close()
        con = self.api.sqlite3.connect(self.journal.path)
        con.execute('DROP TRIGGER immutable_records_UPDATE')
        con.execute("UPDATE records SET payload_json='{}' WHERE seq=2")
        con.close()
        self.reason('AUDIT_INTEGRITY_FAILED',self.make)

    def test_a03_retry_independent_attempts_and_raw_public_bytes(self):
        body = b' {"serverTime":1800000000000} '
        client = self.client([TimeoutError('signature=do-not-record'),HttpResponse(200,body,{})])
        data,meta = client.request_public('/api/v3/time',{},weight=1)
        begins,ends = self.rows('REST_BEGIN'),self.rows('REST_END')
        self.assertEqual([c[-1] for c in self.calls],[1,2])
        self.assertEqual(len(begins),2);self.assertEqual(len(ends),2)
        self.assertEqual(begins[0]['data']['request_id'],begins[1]['data']['request_id'])
        self.assertNotEqual(begins[0]['data']['attempt_id'],begins[1]['data']['attempt_id'])
        self.assertEqual(ends[0]['data']['outcome'],'TIMEOUT')
        self.assertEqual(ends[1]['data']['parent_event_id'],begins[1]['event_id'])
        payload = ends[1]['data']['payload']
        self.assertEqual(payload['representation'],'EXACT_PUBLIC_BYTES')
        self.assertEqual(base64.b64decode(payload['body_base64']),body)
        self.assertEqual(payload['content_sha256'],hashlib.sha256(body).hexdigest())
        self.assertEqual(meta['audit_attempt_id'],begins[1]['data']['attempt_id'])
        self.assertEqual(meta['request_started_at_ms'],ends[1]['data']['sent_at_ms'])
        counts = client.audit_health()['transport']
        self.assertEqual(counts['attempted_n'],2)
        self.assertEqual(counts['responses_n'],1)
        self.assertEqual(counts['errors_n'],1)
        self.assertEqual(self.journal.health()['unresolved_attempts_n'],0)
        self.assertEqual(data['serverTime'],1800000000000)

    def test_a03_5xx_malformed_and_rate_limit_are_recorded_before_validation(self):
        client = self.client([HttpResponse(503,b'busy',{}),HttpResponse(200,b'{',{})])
        self.reason('SCHEMA_INVALID',client.request_public,'/api/v3/time',{},weight=1)
        self.assertEqual([r['data']['status'] for r in self.rows('REST_END')],[503,200])
        client = self.client([HttpResponse(429,b'{"code":-1003}',{'Retry-After':'12','Set-Cookie':'secret'})])
        self.reason('RATE_LIMITED',client.request_public,'/api/v3/time',{},weight=1)
        end = self.rows('REST_END')[-1]['data']
        self.assertEqual(end['headers'],{'retry-after':'12'})
        self.assertEqual(end['status'],429)

    def test_a03_cancel_is_terminal_but_never_success(self):
        client = self.client([KeyboardInterrupt('private-key')])
        with self.assertRaises(KeyboardInterrupt):client.request_public('/api/v3/time',{},weight=1)
        end = self.rows('REST_END')[-1]['data']
        self.assertEqual(end['outcome'],'CANCELLED')
        self.assertNotIn('private-key',json.dumps(self.rows()))

    def test_a03_actual_send_clock_follows_begin_durable_ack(self):
        def delay(phase):
            if phase=='after_commit':self.wall+=70;self.mono+=70000000
        self.journal.fault_hook=delay
        client=self.client([HttpResponse(200,b'{}',{})])
        _,meta=client.request_public('/api/v3/ping',{},weight=1)
        begin,end=self.rows('REST_BEGIN')[-1],self.rows('REST_END')[-1]
        self.assertEqual(meta['request_started_at_ms'],begin['wall_ms']+70)
        self.assertEqual(end['data']['received_at_ms']-end['data']['sent_at_ms'],10)
        self.assertEqual(meta['received_at_ms'],end['data']['received_at_ms'])

    def test_a02_begin_failure_prevents_transport(self):
        self.journal._con.execute("CREATE TRIGGER reject_begin BEFORE INSERT ON records WHEN NEW.kind='REST_BEGIN' BEGIN SELECT RAISE(ABORT,'secret SQL text'); END")
        client=self.client([HttpResponse(200,b'{}',{})])
        self.reason('AUDIT_WRITE_FAILED',client.request_public,'/api/v3/ping',{},weight=1)
        self.assertEqual(self.calls,[])
        self.assertEqual(self.journal.health()['state'],'FAILED')
        self.assertEqual(self.journal.health()['channels']['REST_BEGIN'],{'expected_N':1,'recorded_n':0,'gap_n':1})
        self.assertEqual(client.audit_health()['transport']['attempted_n'],0)

    def test_a02_terminal_failure_prevents_success_and_keeps_unresolved_begin(self):
        self.journal._con.execute("CREATE TRIGGER reject_end BEFORE INSERT ON records WHEN NEW.kind='REST_END' BEGIN SELECT RAISE(ABORT,'private'); END")
        client=self.client([HttpResponse(200,b'{}',{})])
        self.reason('AUDIT_WRITE_FAILED',client.request_public,'/api/v3/ping',{},weight=1)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.journal.health()['unresolved_attempts_n'],1)
        self.assertEqual(client.audit_health()['transport']['responses_n'],1)
        self.assertEqual(self.journal.health()['channels']['REST_END']['gap_n'],1)
        self.journal.close()
        reopened=self.make();self.addCleanup(reopened.close)
        self.assertEqual(reopened.health()['unresolved_attempts_n'],1)
        self.assertEqual(reopened.health()['incomplete_prior_runs_n'],1)
        self.assertFalse(reopened.health()['full_audit_complete'])

    def test_a02_commit_ack_uncertainty_is_sticky_not_retried(self):
        self.journal.fault_hook=lambda phase: (_ for _ in ()).throw(OSError('secret')) if phase=='after_commit' else None
        self.reason('AUDIT_COMMIT_UNKNOWN',self.journal.append,'DECISION',{'action':'WAIT'})
        self.assertEqual(len(self.rows('DECISION')),1)  # Committed, caller never got ACK.
        self.reason('AUDIT_UNAVAILABLE',self.journal.append,'DECISION',{'action':'WAIT'})
        health=self.journal.health()
        self.assertEqual(health['state'],'FAILED')
        self.assertEqual(health['channels']['DECISION'],{'expected_N':2,'recorded_n':0,'gap_n':2})
        self.journal.close()
        self.assertEqual(self.journal.health()['state'],'FAILED')

    def test_a02_capacity_rejects_without_unbounded_wait(self):
        self.journal.close()
        self.journal=self.make(max_inflight=1);self.addCleanup(self.journal.close)
        entered,release=threading.Event(),threading.Event()
        def gate(phase):
            if phase=='before_commit':entered.set();release.wait(3)
        self.journal.fault_hook=gate
        errors=[]
        def first():
            try:self.journal.append('DECISION',{'action':'WAIT'})
            except BaseException as error:errors.append(error)
        worker=threading.Thread(target=first);worker.start()
        try:
            self.assertTrue(entered.wait(2))
            self.reason('AUDIT_BACKPRESSURE',self.journal.append,'DECISION',{'action':'WAIT'})
        finally:release.set();worker.join(3)
        self.assertFalse(worker.is_alive());self.assertEqual(errors,[])
        self.assertEqual(self.journal.health()['channels']['DECISION'],{'expected_N':2,'recorded_n':1,'gap_n':1})

    def test_a02_disk_budget_blocks_before_write(self):
        self.journal.max_bytes=1
        self.reason('AUDIT_DISK_LIMIT',self.journal.append,'DECISION',{'action':'WAIT'})
        self.assertEqual(self.rows('DECISION'),[])

    def test_a04_nested_secrets_and_encoded_urls_are_never_stored(self):
        data={'apiKey':'abc','nested':[{'API_SECRET':'xyz','signature':'zzz','listenKey':'listen'}],
              'message':'https://host/?signature%3Dabc%26apiKey%3Dxyz',
              'note':'echo TEST-CREDENTIAL in a message','good':'BTCUSDT'}
        self.journal.append('DECISION',data,secrets=('TEST-CREDENTIAL',))
        saved=self.rows('DECISION')[-1]['data']
        self.assertEqual(saved['good'],'BTCUSDT')
        text=json.dumps(saved)
        for secret in ('abc','xyz','zzz','listen','TEST-CREDENTIAL'):
            self.assertNotIn(secret,text)

    def test_a04_private_json_is_sanitized_and_invalid_body_omitted(self):
        creds=DemoCredentials('synthetic-key','synthetic-secret')
        client=self.client([HttpResponse(200,b'{"balance":"1","msg":"synthetic-secret","signature":"signed"}',{})])
        client._credentials=creds
        client._request('/api/v3/account',{},signed=True,weight=1)
        payload=self.rows('REST_END')[-1]['data']['payload']
        self.assertEqual(payload['representation'],'SANITIZED')
        self.assertNotIn('body_base64',payload)
        self.assertNotIn('synthetic-secret',json.dumps(self.rows()))
        self.assertNotIn('synthetic-key',json.dumps(self.rows()))
        self.assertNotIn('signed',json.dumps(payload))
        client=self.client([HttpResponse(200,b'not json synthetic-secret',{})]);client._credentials=creds
        self.reason('SCHEMA_INVALID',client._request,'/api/v3/account',{},signed=True,weight=1)
        self.assertEqual(self.rows('REST_END')[-1]['data']['payload']['representation'],'OMITTED')

    def test_a04_sensitive_public_response_is_not_labeled_exact(self):
        client=self.client([HttpResponse(200,b'{"apiKey":"DO-NOT-STORE"}',{})])
        client.request_public('/api/v3/ping',{},weight=1)
        payload=self.rows('REST_END')[-1]['data']['payload']
        self.assertEqual(payload['representation'],'SANITIZED')
        self.assertNotIn('DO-NOT-STORE',json.dumps(self.rows()))

    def test_a05_wait_and_no_action_have_causal_inputs_and_versions(self):
        for action in ('WAIT','NO_ACTION'):
            eid=self.journal.record_decision(decision_type='OPERATIONAL',action=action,
                reasons=['HISTORY_NOT_READY'],cutoff_ms=100,
                sources=[dict(snapshot_id='s1',available_at_ms=99)],candidate_population=['BTCUSDT'],
                policy_version='p1',model_version=None,risk_version='r1',intent=None)
            self.assertEqual(self.rows('DECISION')[-1]['event_id'],eid)
        data=self.rows('DECISION')[-1]['data']
        self.assertEqual(data['sources'],[dict(snapshot_id='s1',available_at_ms=99)])
        self.assertEqual(data['candidate_population'],['BTCUSDT'])
        self.assertEqual(data['policy_version'],'p1')
        self.reason('AUDIT_FUTURE_SOURCE',self.journal.record_decision,
            decision_type='TRADING',action='BUY',reasons=['SIGNAL'],cutoff_ms=100,
            sources=[dict(snapshot_id='future',available_at_ms=101)],candidate_population=['BTCUSDT'],
            policy_version='p1',model_version='m1',risk_version='r1',intent={'quantity':'1'})
        self.assertEqual(len(self.rows('DECISION')),2)

    def test_a06_pagination_frozen_manifest_and_detached_health(self):
        before=self.journal.manifest()
        self.journal.append('DECISION',{'action':'WAIT'})
        first=self.journal.read(limit=1);second=self.journal.read(after_seq=first[0]['seq'],limit=1)
        self.assertEqual(len(first),1);self.assertEqual(len(second),1)
        self.assertNotEqual(first[0]['event_id'],second[0]['event_id'])
        self.assertEqual(before['rows_n'],1)
        current=self.journal.manifest()
        self.assertEqual(current['rows_n'],2)
        self.assertEqual(current['chain_tip'],second[0]['content_sha256'])
        self.assertEqual(current,self.journal.manifest())
        health=self.journal.health();health['channels']['DECISION']['recorded_n']=0
        self.assertEqual(self.journal.health()['channels']['DECISION']['recorded_n'],1)
        self.assertFalse(health['full_audit_complete'])
        self.assertEqual(health['scope'],'REST_ATTEMPTS_AND_EXPLICIT_DECISIONS')
        self.reason('AUDIT_INVALID',self.journal.read,limit=1001)

    def test_a07_authenticated_identity_mismatch_denied_before_calls(self):
        self.reason('AUDIT_IDENTITY_MISMATCH',BinanceDemoAdapter,
            DemoCredentials('synthetic-key','synthetic-secret'),account_alias='foreign',
            credential_generation='g1',episode_id='e1',scope_id='s1',identity_key=b'x'*32,
            audit=self.journal)

    def test_a07_unconfigured_client_is_explicit_and_has_independent_counts(self):
        client=DemoReadClient(transport=object(),clock=self.clock)
        health=client.audit_health()
        self.assertEqual(health['state'],'UNCONFIGURED')
        self.assertFalse(health['full_audit_complete'])
        self.assertEqual(health['transport']['attempted_n'],0)

    def test_a03_deadline_consumed_by_begin_is_cancelled_without_send(self):
        def delay(phase):
            if phase=='after_commit':self.wall+=16000;self.mono+=16000000000
        self.journal.fault_hook=delay
        client=self.client([HttpResponse(200,b'{}',{})])
        self.reason('READ_UNAVAILABLE',client.request_public,'/api/v3/ping',{},weight=1)
        self.assertEqual(self.calls,[])
        self.assertEqual(client.audit_health()['transport']['attempted_n'],0)
        end=self.rows('REST_END')[-1]['data']
        self.assertEqual(end['outcome'],'CANCELLED');self.assertIsNone(end['sent_at_ms'])

    def test_a03_success_provenance_references_terminal_ack(self):
        client=self.client([HttpResponse(200,b'{}',{})])
        _,meta=client.request_public('/api/v3/ping',{},weight=1)
        self.assertEqual(meta['audit_end_event_id'],self.rows('REST_END')[-1]['event_id'])

    def test_a04_public_encoding_collision_preserves_original_bytes(self):
        body=b'{"serverTime":1800000000000}'
        client=self.client([HttpResponse(200,body,{})])
        client._credentials=DemoCredentials(base64.b64encode(body).decode()[:8],'synthetic-secret')
        client.request_public('/api/v3/time',{},weight=1)
        payload=self.rows('REST_END')[-1]['data']['payload']
        self.assertEqual(payload['representation'],'EXACT_PUBLIC_BYTES')
        self.assertEqual(base64.b64decode(payload['body_base64'],validate=True),body)

    def test_a02_event_size_failure_is_visible_and_write_free(self):
        with patch.object(self.api,'MAX_EVENT_BYTES',300):
            self.reason('AUDIT_EVENT_LIMIT',self.journal.append,'DECISION',{'action':'WAIT','x':'a'*500})
        self.assertEqual(self.rows('DECISION'),[])
        self.assertEqual(self.journal.health()['channels']['DECISION']['gap_n'],1)

    def test_a01_second_writer_is_denied_without_changing_owner(self):
        with self.assertRaises(ValueError):self.make()
        self.journal.append('DECISION',{'action':'WAIT'})
        self.assertEqual(len(self.rows('DECISION')),1)

    def test_a07_real_private_constructor_accepts_same_namespace(self):
        client=self.client([HttpResponse(200,b'{}',{})])
        adapter=BinanceDemoAdapter(DemoCredentials('synthetic-key','synthetic-secret'),
            account_alias='synthetic',credential_generation='g1',episode_id='e1',scope_id='s1',
            identity_key=b'x'*32,audit=self.journal,clock=self.clock,transport=client.transport)
        adapter.ping()
        self.assertEqual(len(self.rows('REST_END')),1)
        self.assertEqual(adapter.audit_health()['transport']['responses_n'],1)

    def test_a06_counter_basis_names_this_process_not_historical_rows(self):
        health=self.journal.health()
        self.assertEqual(health['counter_basis'],'PROCESS_SUBMISSIONS')
        self.assertEqual(health['run_id'],self.journal.run_id)
        self.assertEqual(health['process_instance_id'],self.journal.process_id)

    def test_a01_os_crash_reopens_durable_unknown_without_claiming_complete(self):
        source=str(Path(__file__).resolve().parents[1]/'src')
        program='''import os,sys
sys.path.insert(0,sys.argv[1])
from pathlib import Path
from binance_demo_bot.interaction_audit import AuditJournal
from binance_demo_bot.isolation import AppPaths
j=AuditJournal(AppPaths(Path(sys.argv[2])),account_alias='synthetic',episode_id='e1',scope_id='s1',database='crash.sqlite')
j.append('REST_BEGIN',dict(request_id='request',attempt_id='attempt'))
os._exit(0)
'''
        result=subprocess.run([sys.executable,'-I','-c',program,source,str(self.paths.root)],
                              capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        recovered=self.make(database='crash.sqlite');self.addCleanup(recovered.close)
        health=recovered.health()
        self.assertEqual(health['unresolved_attempts_n'],1)
        self.assertEqual(health['incomplete_prior_runs_n'],1)
        self.assertEqual(health['channels']['REST_BEGIN']['expected_N'],0)
        self.assertFalse(health['full_audit_complete'])

    def test_a04_encoded_sensitive_keys_are_removed_not_just_marked_sanitized(self):
        client=self.client([HttpResponse(200,b'{"api%5Fkey":"MUST-DISAPPEAR","good":1}',{})])
        client.request_public('/api/v3/ping',{},weight=1)
        data=self.rows('REST_END')[-1]['data']['payload']
        self.assertEqual(data['representation'],'SANITIZED')
        self.assertNotIn('MUST-DISAPPEAR',json.dumps(self.rows()))
        self.assertEqual(data['data'],{'good':1})
