"""REC-01/03/04/05: own journals and deterministic exchange, no live calls."""
from copy import deepcopy
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import json,unittest
import _support
from binance_demo_bot.adapter import AdapterError
from binance_demo_bot.execution_pilot import PilotJournal
from binance_demo_bot.usdt_scope import usdt_catalog
import test_pilot_service as pilot_fixture
import test_usdt_service as usdt_fixture


class RecoveryLifecycleTests(unittest.TestCase):
    setUp=pilot_fixture.ServiceTests.setUp

    def controller(self):
        return self.api.PilotService(self.book,self.exchange,clock_ms=lambda:self.exchange.now)

    def unknown_buy(self):
        self.exchange.unknown_buy=True
        self.service.tick()
        self.exchange.unknown_buy=False
        self.assertEqual(self.book.actions()['buy-1']['state'],'QUERY_REQUIRED')

    def test_rec03_not_found_is_durable_and_restart_obeys_next_query(self):
        self.unknown_buy();queries=[]
        def missing(*args):queries.append(deepcopy(args));raise AdapterError('PILOT_ORDER_NOT_FOUND')
        self.exchange.query_action=missing
        binding=deepcopy(self.book._binding)
        state=self.controller().recover_once()
        self.assertEqual(state['recovery_attempts'],1)
        self.assertEqual(state['recovery_last_checked_ms'],2000)
        self.assertEqual(state['recovery_last_reason'],'PILOT_ORDER_NOT_FOUND')
        self.assertEqual(state['recovery_next_query_ms'],12000)
        self.book.close()
        self.book=PilotJournal(self.paths,account_identity='a'*64,start_ms=1000,expires_ms=86401000,
            initial_balances={'USDT':{'free':'100','locked':'0'}},enabled=True)
        self.addCleanup(self.book.close)
        self.exchange.now=11999;self.controller().recover_once()
        self.assertEqual(len(queries),1)
        self.exchange.now=12000;state=self.controller().recover_once()
        self.assertEqual(len(queries),2);self.assertEqual(state['recovery_attempts'],2)
        self.assertEqual(state['recovery_next_query_ms'],32000)
        self.assertEqual(state['phase'],'RECOVERY_REQUIRED');self.assertTrue(state['entry_killed'])
        self.assertEqual(self.book.actions()['buy-1']['state'],'QUERY_REQUIRED')
        self.assertEqual(self.book._binding,binding)
        self.assertEqual(sum(p.get('side')=='BUY' for _,_,p in self.exchange.calls),1)
        decisions=[r['data']['value'] for r in self.book._audit.read(limit=1000)
            if r['kind']=='DECISION' and r['data'].get('operation')=='DECISION']
        self.assertTrue(any(d.get('action')=='RECOVERY_QUERY' for d in decisions))

    def test_rec03_original_incident_and_exact_error_are_separate(self):
        self.unknown_buy();service=self.controller()
        for reason in ('CLOCK_QUALITY','AUTH_DENIED','AUDIT_WRITE_FAILED','READ_UNAVAILABLE','RATE_LIMITED'):
            with self.subTest(reason=reason):
                def broken(*args):raise AdapterError(reason)
                self.exchange.query_action=broken
                state=service.recover_once()
                self.assertEqual(state['reason'],'PILOT_QUERY_REQUIRED')
                self.assertEqual(state['recovery_last_reason'],reason)
                self.assertGreaterEqual(state['recovery_next_query_ms']-self.exchange.now,
                    60000 if reason=='RATE_LIMITED' else 10000)
                self.exchange.now=state['recovery_next_query_ms']
        self.assertEqual(self.book.actions()['buy-1']['state'],'QUERY_REQUIRED')

    def test_rec03_one_lookup_per_pass_and_capped_cadence_after_expiry(self):
        self.unknown_buy();queries=[]
        def missing(*args):queries.append(self.exchange.now);return None
        self.exchange.query_action=missing
        # Two unknown logical actions, each with a real write-ahead send record.
        p=dict(symbol='BTCUSDT',side='SELL',quantity='0.1',type='MARKET',
            newClientOrderId=self.book.client_id('sell-1'))
        for op,value in [('PREPARED',dict(action='sell-1',method='POST',path='/api/v3/order',params=p)),
            ('START',dict(action='sell-1')),('UNKNOWN',dict(action='sell-1'))]:self.book._save(op,value)
        service=self.controller();self.exchange.now=self.book._binding['expires_ms']+1
        for attempt in range(1,10):
            state=service.recover_once()
            self.assertEqual(len(queries),attempt)
            expected=min(300000,10000*2**min(attempt-1,5))
            self.assertEqual(state['recovery_next_query_ms']-self.exchange.now,expected)
            service.recover_once();self.assertEqual(len(queries),attempt)
            self.exchange.now=state['recovery_next_query_ms']
        self.assertEqual(state['phase'],'RECOVERY_REQUIRED')
        for at in queries:self.assertLessEqual(sum(at<=t<at+60000 for t in queries),6)

    def test_rec04_missing_phase_cannot_become_open(self):
        self.book.save_state(dict(phase='RECOVERY_REQUIRED',cycles=1,entry_killed=True))
        state=self.controller().recover_once()
        self.assertEqual(state['phase'],'RECOVERY_REQUIRED')
        self.assertEqual(state['recovery_last_reason'],'PILOT_RECOVERY_PHASE_INVALID')
        self.assertEqual(self.exchange.calls,[])

    def test_rec04_crash_before_send_cannot_create_buy_during_recovery(self):
        self.book.save_state(dict(phase='RECOVERY_REQUIRED',recovery_phase='BUY_PENDING',cycles=1,
            origin_ms=2000,feature_end_ms=1999,features=[0.,0.,.01,2.],snapshot_id='frozen',
            start_equity='100',entry_killed=True,symbol='BTCUSDT'))
        state=self.controller().recover_once()
        self.assertEqual(self.exchange.calls,[])
        self.assertEqual(state['recovery_last_reason'],'PILOT_RECOVERY_ENTRY_UNVERIFIED')
        self.assertEqual(state['phase'],'RECOVERY_REQUIRED')

    def test_rec04_unsent_pending_entry_checks_expiry_and_kill(self):
        for expired in (False,True):
            with self.subTest(expired=expired):
                self.exchange.now=86401001 if expired else 2000
                self.book.save_state(dict(phase='BUY_PENDING',cycles=1,entry_killed=not expired,
                    origin_ms=2000,feature_end_ms=1999,features=[0.,0.,.01,2.],snapshot_id='frozen',
                    start_equity='100',symbol='BTCUSDT'))
                state=self.service.tick()
                self.assertEqual(self.exchange.calls,[])
                self.assertEqual(state['phase'],'RECOVERY_REQUIRED')

    def test_rec04_stop_marker_blocks_unsent_pending_entry(self):
        self.paths.runtime_path('pilot/stop-entries').write_text('stop')
        self.book.save_state(dict(phase='BUY_PENDING',cycles=1,entry_killed=False,
            origin_ms=2000,feature_end_ms=1999,features=[0.,0.,.01,2.],snapshot_id='frozen',
            start_equity='100',symbol='BTCUSDT'))
        self.service.tick();self.assertEqual(self.exchange.calls,[])

    def test_rec04_malformed_buy_ack_cannot_resume(self):
        self.book.save_state(dict(phase='RECOVERY_REQUIRED',recovery_phase='BUY_PENDING',cycles=1,
            symbol='BTCUSDT',entry_killed=True))
        with patch.object(self.book,'actions',return_value={'buy-1':dict(state='ACKNOWLEDGED',method='DELETE',
            path='/api/v3/orderList',params=dict(symbol='BTCUSDT',side='BUY'))}):
            state=self.controller().recover_once()
        self.assertEqual(state['recovery_last_reason'],'PILOT_RECOVERY_ENTRY_UNVERIFIED')
        self.assertEqual(state['phase'],'RECOVERY_REQUIRED');self.assertEqual(self.exchange.calls,[])

    def test_rec03_actual_worker_keeps_scheduling_recovery_past_first_pass(self):
        self.unknown_buy();queries=[];sleeps=[]
        class EndLoop(BaseException):pass
        def missing(*args):
            queries.append(self.exchange.now)
            if len(queries)==2:raise EndLoop()
            return None
        def sleep(seconds):
            sleeps.append(seconds);self.exchange.now+=round(seconds*1000)
            if len(sleeps)>8:raise EndLoop()
        self.exchange.query_action=missing;self.exchange.close=lambda:None
        self.service._clock_ms=lambda:self.exchange.now
        self.paths.runtime_path('pilot-research').mkdir(parents=True,exist_ok=True)
        self.paths.runtime_path('pilot-research/manifest.json').write_text(json.dumps(
            dict(state='MAXIMUM_AVAILABLE_FETCH_COMPLETE',bars_N=1,replay_verdict='DIAGNOSTIC')))
        self.paths.runtime_path('account-identity.key').write_bytes(b'synthetic-identity')
        self.paths.runtime_path('pilot/profile.json').write_text(json.dumps(self.book._binding))
        close=lambda:None
        dummy=SimpleNamespace(close=close)
        client=SimpleNamespace(close=close,sync_clock=lambda:None,
            read_account=lambda:dict(data=dict(account_identity='a'*64)))
        opening=SimpleNamespace(close=close,summary=lambda:dict(attestation=dict(account_identity='a'*64)))
        reader=SimpleNamespace(close=close,start=lambda:None,health=lambda:{})
        replacements={'AppPaths':lambda _:self.paths,'AuditJournal':lambda *a,**k:dummy,
            'load_demo_credentials':lambda _:object(),'BinanceDemoAdapter':lambda *a,**k:client,
            'AccountOpeningManifest':lambda *a,**k:opening,'PilotJournal':lambda *a,**k:self.book,
            'AccountEvidenceStore':lambda *a,**k:dummy,'PrivateEventReader':lambda *a,**k:reader,
            'LiveGateway':lambda *a,**k:self.exchange,'PilotService':lambda *a,**k:self.service}
        with ExitStack() as stack:
            for name,value in replacements.items():stack.enter_context(patch.object(self.api,name,value))
            stack.enter_context(patch('sys.argv',['pilot','--enable-demo-qualification']))
            stack.enter_context(patch.object(self.api.time,'sleep',sleep))
            stack.enter_context(patch.object(self.api.time,'time_ns',lambda:self.exchange.now*1000000))
            with self.assertRaises(EndLoop):self.api.main()
        self.assertEqual(queries,[2000,12000]);self.assertEqual(sleeps,[10])

    def test_rec04_delayed_buy_ack_protected_and_closed_after_expiry_without_rebuy(self):
        self.unknown_buy();self.exchange.now=86401001
        state=self.controller().recover_once()
        self.assertEqual(state['phase'],'OPEN');self.assertTrue(state['entry_killed'])
        self.assertEqual(sum(p.get('side')=='BUY' for _,_,p in self.exchange.calls),1)
        self.assertEqual(self.exchange.calls[-1][1],'/api/v3/orderList/oco')
        state=self.controller().tick()
        self.assertEqual(state['phase'],'IDLE');self.assertEqual(len(self.book.outcomes()),1)
        self.assertTrue(state['entry_killed'])

    def test_rec04_unknown_protection_recovers_without_second_oco(self):
        self.exchange.unknown_protection=True;self.service.tick()
        self.assertEqual(self.book.actions()['protect-1']['state'],'QUERY_REQUIRED')
        # Exchange accepted the OCO, although the original response was lost.
        params=self.book.actions()['protect-1']['params'];self.exchange.unknown_protection=False
        response=self.exchange.mutate('POST','/api/v3/orderList/oco',params)
        sent=len(self.exchange.calls)
        state=self.controller().recover_once()
        self.assertEqual(state['phase'],'OPEN');self.assertEqual(len(self.exchange.calls),sent)
        self.assertEqual(self.book.actions()['protect-1']['response'],response)

    def test_rec03_worker_sleep_obeys_persisted_query_deadline(self):
        state=dict(phase='RECOVERY_REQUIRED',reason='PILOT_QUERY_REQUIRED',recovery_next_query_ms=121000)
        self.assertEqual(self.api.pilot_sleep_seconds(state,1000),120)
        self.assertEqual(self.api.pilot_sleep_seconds(state,122000),.1)


class CycleScopedRecoveryTests(unittest.TestCase):
    setUp=usdt_fixture.UsdtServiceTests.setUp

    def test_rec01_new_symbol_unknown_buy_cannot_inherit_closed_position(self):
        self.service.tick();self.exchange.now+=61000;self.service.tick()
        previous=self.book.state();outcomes=deepcopy(self.book.outcomes());binding=deepcopy(self.book._binding)
        self.assertIn('take_profit',previous)
        state=dict(previous,symbol='SOLUSDT',base_asset='SOL',recovery_phase='OPEN',recovery_attempts=3,
            recovery_last_reason='CLOCK_QUALITY',recovery_next_query_ms=999999)
        self.book.save_state(state);self.exchange.symbol='SOLUSDT';self.exchange.now=previous['next_entry_ms']
        self.book.authorize_universe(usdt_catalog(
            {'symbols':list(self.exchange.metadata.values())},['SPOT'],can_trade=True),received_at_ms=self.exchange.now,
            source_sha256='b'*64,validated_symbols=list(self.exchange.metadata))
        def lost(method,path,p):self.exchange.calls.append((method,path,deepcopy(p)));raise TimeoutError()
        self.exchange.mutate=lost
        current=self.service.tick()
        for key in ('fill_price','protected_quantity','take_profit','stop_loss','deadline_ms',
            'unprotected_duration_ms','protection_action','recovery_attempts','recovery_last_reason','recovery_next_query_ms'):
            self.assertNotIn(key,current,key)
        self.assertEqual(current['recovery_phase'],'BUY_PENDING');self.assertEqual(current['cycles'],2)
        self.assertEqual(current['symbol'],'SOLUSDT');self.assertEqual(self.book.outcomes(),outcomes)
        self.assertEqual(self.book._binding,binding);self.assertEqual(self.book.actions()['buy-1']['state'],'ACKNOWLEDGED')
        self.assertEqual(self.book.actions()['buy-2']['state'],'QUERY_REQUIRED')

    def test_rec05_legacy_stale_protection_is_hidden_without_rewriting_state(self):
        self.service.tick();state=self.book.state()
        state.update(phase='RECOVERY_REQUIRED',recovery_phase='BUY_PENDING',cycles=2,symbol='SOLUSDT',base_asset='SOL')
        before=deepcopy(state)
        report=self.api.operational_report(state,dict(quotes_received_at_ms=2000,selected_symbol='SOLUSDT'),
            pid=1,now_ms=2100,expires_ms=86401000,actual_outcomes_n=0,stream={},actions=self.book.actions())
        self.assertEqual(report['position_status'],'UNVERIFIED')
        self.assertEqual(report['current_exposure_status'],'UNKNOWN')
        self.assertEqual(report['operational_status'],'BLOCKED')
        for key in ('fill_price','protected_quantity','take_profit','stop_loss','deadline_ms','protection_action'):
            self.assertNotIn(key,report['state'])
        self.assertEqual(state,before)

    def test_rec05_current_protection_ack_is_visible_but_not_live_certified(self):
        self.service.tick();state=self.book.state()
        report=self.api.operational_report(state,dict(quotes_received_at_ms=2000),pid=1,now_ms=2100,
            expires_ms=86401000,actual_outcomes_n=0,stream={},actions=self.book.actions())
        self.assertEqual(report['position_status'],'PROTECTION_ACKNOWLEDGED')
        self.assertEqual(report['state']['take_profit'],state['take_profit'])
        self.assertEqual(report['current_exposure_status'],'UNKNOWN')
        state['entry_killed']=False;state.update(phase='RECOVERY_REQUIRED',recovery_phase='OPEN')
        report=self.api.operational_report(state,dict(quotes_received_at_ms=2000),pid=1,now_ms=2100,
            expires_ms=86401000,actual_outcomes_n=0,stream={},actions=self.book.actions())
        self.assertEqual(report['operational_status'],'BLOCKED')
        self.assertEqual(report['position_status'],'PROTECTION_ACKNOWLEDGED')

    def test_rec05_closed_state_has_no_active_position_even_with_historical_fields(self):
        self.service.tick();self.exchange.now+=61000;self.service.tick()
        from binance_demo_bot.pilot_service import cycle_status
        projection=cycle_status(self.book.state(),self.book.actions())
        self.assertEqual(projection['position_status'],'CLOSED')
        self.assertNotIn('fill_price',projection['state']);self.assertNotIn('take_profit',projection['state'])


if __name__=='__main__':unittest.main()
