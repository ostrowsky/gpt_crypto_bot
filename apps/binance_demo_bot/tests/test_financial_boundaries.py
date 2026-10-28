"""FBO-01..04 synthetic sources and real own SQLite; no network/credentials."""
from copy import deepcopy
from decimal import Decimal
import importlib,tempfile,unittest
from pathlib import Path
import _support
import test_account_opening as source
from binance_demo_bot.account_opening import AccountOpeningManifest
from binance_demo_bot.adapter import AdapterError
from binance_demo_bot.financial import day_bounds
from binance_demo_bot.isolation import AppPaths

class FinancialBoundaryTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures';root.mkdir(parents=True,exist_ok=True)
        tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(tmp.cleanup);self.paths=AppPaths(Path(tmp.name).resolve())
        self.opening=AccountOpeningManifest(self.paths,**source.IDENTITY);self.addCleanup(self.opening.close)
        self.opening.attest(account_identity=source.HMAC,declaration_id='exclusive',declared_at_ms=8500,exclusive=True,now_ms=10000)
        self.seed=self.opening.capture(**source.inputs());self.boundary=day_bounds('1970-01-02','Europe/Budapest')[0]
    def api(self):return importlib.import_module('binance_demo_bot.financial_boundaries')
    def store(self,**kwargs):
        value=self.api().FinancialBoundaryStore(self.paths,opening_manifest=self.opening,candidate_id='candidate-1',**kwargs)
        self.addCleanup(value.close);return value
    def args(self,offset,key):
        value=source.inputs();delta=self.boundary+offset-value['cutoff_ms'];value.pop('candidate_id');value['observation_id']=key;value['cutoff_ms']+=delta
        for name in ('account_before','account_after','orders','order_lists','metadata','tickers'):
            for field in ('request_started_at_ms','received_at_ms'):value[name][field]+=delta
            for field in ('request_started_monotonic_ns','received_monotonic_ns'):value[name][field]+=delta*1000000
        return value
    def test_fbo01_original_seed_immutable_and_reopen_idempotent(self):
        store=self.store();args=self.args(-1000,'before');saved=store.observe(**args)
        self.assertEqual(saved,store.observe(**args));store.close();store=self.store()
        self.assertEqual(store.observations(),[saved]);self.assertEqual(self.opening.summary()['candidates']['candidate-1'],self.seed)
        self.assertFalse(self.paths.runtime_path('account-ledger.sqlite').exists());self.assertNotIn('seed_lots',saved)
        self.assertEqual(saved['opening_sha256'],self.seed['manifest_sha256'])
    def test_fbo01_conflicting_observation_and_other_seed_rejected(self):
        store=self.store();store.observe(**self.args(-1000,'same'))
        with self.assertRaises(AdapterError):store.observe(**self.args(-1100,'same'))
        store.close();args=source.inputs();args['candidate_id']='candidate-2';self.opening.capture(**args)
        with self.assertRaises(AdapterError):self.api().FinancialBoundaryStore(self.paths,opening_manifest=self.opening,candidate_id='candidate-2')
    def test_fbo02_hash_namespace_clock_and_bracket_validation(self):
        store=self.store()
        for name,field,value in [('tickers','source_sha256','f'*64),('account_after','account_alias','foreign'),('orders','clock_uncertainty_ms',251),('tickers','request_started_at_ms',self.boundary-2000)]:
            args=self.args(-1000,'bad');args[name][field]=value
            with self.subTest(field=field),self.assertRaises(AdapterError):store.observe(**args)
        self.assertEqual(store.observations(),[])
    def test_fbo03_complete_bracket_is_provisional_never_daily_pnl(self):
        store=self.store();store.observe(**self.args(-1000,'before'));store.observe(**self.args(1000,'after'))
        report=store.boundary_report(self.boundary)
        self.assertEqual(report['state'],'PROVISIONAL_BRACKET');self.assertEqual(report['before_id'],'before');self.assertEqual(report['after_id'],'after')
        self.assertEqual(report['financial_status'],'PENDING');self.assertIsNone(report['daily_net_pnl_usdt']);self.assertFalse(report['execution_authorized'])
        self.assertIn('EXACT_ECONOMIC_CUT_UNCERTIFIED',report['reasons'])
    def test_fbo03_future_observation_never_backfills_missing_before(self):
        store=self.store();store.observe(**self.args(1000,'future'));report=store.boundary_report(self.boundary)
        self.assertEqual(report['state'],'PENDING');self.assertIsNone(report['before_id']);self.assertIn('BEFORE_MISSING_OR_STALE',report['reasons'])
    def test_fbo03_quote_stale_at_midnight_and_missing_after(self):
        store=self.store();store.observe(**self.args(-6000,'too-old'));report=store.boundary_report(self.boundary)
        self.assertIsNone(report['before_id']);self.assertIsNone(report['after_id']);self.assertEqual(report['state'],'PENDING')
    def test_fbo03_select_last_before_first_after_not_cherry_picked_value(self):
        store=self.store()
        for offset,key in [(-2000,'earlier'),(2000,'later'),(-1000,'before'),(1000,'after')]:store.observe(**self.args(offset,key))
        report=store.boundary_report(self.boundary);self.assertEqual((report['before_id'],report['after_id']),('before','after'))
    def test_fbo03_changed_or_locked_holdings_retained_as_pending(self):
        store=self.store();store.observe(**self.args(-1000,'before'));args=self.args(1000,'after')
        for field in ('account_before','account_after'):
            args[field]['data']['balances'][0].update(free='99',locked='1');source.rehash(args[field])
        saved=store.observe(**args);self.assertEqual(saved['holdings'][1 if saved['holdings'][0]['asset']=='USDC' else 0]['locked'],'1')
        report=store.boundary_report(self.boundary);self.assertEqual(report['state'],'PENDING');self.assertIn('HOLDINGS_CHANGED',report['reasons'])
    def test_fbo02_sensitive_public_asset_roundtrips_without_redaction_relaxation(self):
        store=self.store();args=self.args(-1000,'public-assets')
        for name in ('account_before','account_after'):
            args[name]['data']['balances'][1]['asset']='TOKEN';args[name]['data']['returned_assets']=sorted(r['asset'] for r in args[name]['data']['balances']);source.rehash(args[name])
        args['metadata']['data']['symbols'][0].update(symbol='TOKENUSDT',baseAsset='TOKEN');source.rehash(args['metadata'])
        args['tickers']['data'][0]['symbol']='TOKENUSDT';source.rehash(args['tickers'])
        saved=store.observe(**args);store.close();store=self.store();self.assertEqual(store.observations(),[saved])
        self.assertIn('TOKEN',{r['asset'] for r in saved['holdings']})
    def test_fbo01_native_commit_failure_never_returns_success(self):
        store=self.store()
        def fault(phase):
            if phase=='before_commit':raise OSError('synthetic commit failure')
        store._audit.fault_hook=fault
        with self.assertRaises(AdapterError):store.observe(**self.args(-1000,'failed'))
        store.close();store=self.store();self.assertEqual(store.observations(),[])
    def test_fbo03_non_midnight_rejected_and_dst_schedule_uses_calendar(self):
        store=self.store()
        with self.assertRaises(AdapterError):store.boundary_report(self.boundary+1)
        api=self.api();start,end=day_bounds('2026-10-25','Europe/Budapest');self.assertEqual(end-start,25*3600000)
        spring=day_bounds('2026-03-29','Europe/Budapest');self.assertEqual(spring[1]-spring[0],23*3600000)
        self.assertEqual(api.next_boundaries(start+1000,count=2)[0],end)
        self.assertEqual(api.capture_slots(end,end+2000),[end+10000])



class FinancialBoundaryServiceTests(unittest.TestCase):
    setUp=FinancialBoundaryTests.setUp
    api=FinancialBoundaryTests.api
    store=FinancialBoundaryTests.store
    args=FinancialBoundaryTests.args
    def service(self):return importlib.import_module('binance_demo_bot.financial_boundary_service')
    def test_fbo01_frozen_schedule_survives_restart(self):
        store=self.store();first=store.schedule(self.boundary-10000);store.close();store=self.store()
        self.assertEqual(store.schedule(self.boundary+2000),first);self.assertEqual(first['boundaries_ms'][0],self.boundary)
    def test_fbo04_native_capture_only_get_bracket_order(self):
        store=self.store();args=self.args(-1000,'native');calls=[]
        class Client:
            def sync_clock(c):calls.append('clock')
            def read_account(c):
                before='account-before' not in calls;calls.append('account-before' if before else 'account-after')
                return args['account_before' if before else 'account_after']
            def read_open_orders(c):calls.append('orders');return args['orders']
            def read_open_order_lists(c):calls.append('lists');return args['order_lists']
            def mutate(c,*a,**k):raise AssertionError('mutation forbidden')
        class Market:
            def read_book_tickers(m):calls.append('tickers');return args['tickers']
        report,raw=self.service().capture_native(Client(),Market(),store,args['metadata'],'native',lambda:args['cutoff_ms'])
        self.assertEqual(calls,['clock','account-before','orders','lists','tickers','account-after']);self.assertEqual(report['observation_id'],'native')
        self.assertEqual(raw['account_after'],args['account_after']);self.assertIsNone(report['daily_net_pnl_usdt'])
    def test_fbo04_finite_scheduler_skips_past_offsets_and_retains_failure(self):
        now=[self.boundary+2000];events=[];captures=[]
        def capture(b,slot):captures.append(slot);raise AdapterError('CLOCK_QUALITY')
        self.service().run_schedule([self.boundary],now_ms=lambda:now[0],sleep=lambda seconds:now.__setitem__(0,now[0]+round(seconds*1000)),
            warm=lambda b:None,capture=capture,emit=events.append,stopped=lambda:False)
        self.assertEqual(captures,[self.boundary+10000]);self.assertEqual(events[-1]['state'],'COMPLETED')
        self.assertEqual(sum(e['state']=='SKIPPED_PAST_SLOT' for e in events),2)
        self.assertTrue(any(e.get('reason')=='CLOCK_QUALITY' for e in events))
    def test_fbo04_stop_during_wait_does_not_capture_or_affect_trading(self):
        now=[self.boundary-100000];events=[];captures=[]
        self.service().run_schedule([self.boundary],now_ms=lambda:now[0],sleep=lambda seconds:now.__setitem__(0,now[0]+round(seconds*1000)),
            warm=lambda b:None,capture=lambda *a:captures.append(a),emit=events.append,stopped=lambda:now[0]>=self.boundary-90000)
        self.assertEqual(captures,[]);self.assertEqual(events[-1]['state'],'STOPPED')



class BoundaryPublicGapTests(unittest.TestCase):
    setUp=FinancialBoundaryTests.setUp
    api=FinancialBoundaryTests.api
    store=FinancialBoundaryTests.store
    args=FinancialBoundaryTests.args
    def test_fbo02_unknown_long_public_asset_preserves_missing_route_reason(self):
        store=self.store();args=self.args(-1000,'unpriced');asset='LONG_TOKEN_ASSET_12345678901234567890'
        for name in ('account_before','account_after'):
            args[name]['data']['balances'][1]['asset']=asset;args[name]['data']['returned_assets']=sorted(r['asset'] for r in args[name]['data']['balances']);source.rehash(args[name])
        saved=store.observe(**args)
        self.assertEqual(saved['valuation']['status'],'PENDING');self.assertIsNone(saved['valuation']['equity_usdt'])
        self.assertIn(dict(asset=asset,reason='MISSING_ROUTE'),saved['valuation']['reason_rows'])
        store.close();store=self.store();self.assertEqual(store.observations(),[saved])



class BoundaryManifestLeaseTests(unittest.TestCase):
    setUp=FinancialBoundaryTests.setUp
    api=FinancialBoundaryTests.api
    store=FinancialBoundaryTests.store
    args=FinancialBoundaryTests.args
    def test_fbo01_observer_uses_frozen_validation_basis_without_holding_original_lease(self):
        store=self.store();self.opening.close()
        other=AccountOpeningManifest(self.paths,**source.IDENTITY);self.addCleanup(other.close)
        self.assertEqual(other.summary()['candidates']['candidate-1'],self.seed)
        saved=store.observe(**self.args(-1000,'no-opening-lease'))
        self.assertEqual(saved['opening_sha256'],self.seed['manifest_sha256'])



if __name__=='__main__':unittest.main()
