"""USD-02/03/06: shared portfolio, scoped canary and safe operating lifecycle."""
from copy import deepcopy
from decimal import Decimal
import importlib,tempfile,unittest
from pathlib import Path
import _support
from binance_demo_bot.usdt_scope import UsdtPilotJournal,usdt_catalog,portfolio_inventory
from binance_demo_bot.adapter import AdapterError
from binance_demo_bot.isolation import AppPaths
from test_usdt_scope import info,fill
from test_pilot_service import Exchange


class ScopedExchange(Exchange):
    def __init__(self):
        super().__init__();self.symbol='ETHUSDT';self.metadata={s:info(s) for s in ('BTCUSDT','ETHUSDT','SOLUSDT')}
    def snapshot(self):
        inv=portfolio_inventory(self.rows,{(r['symbol'],r['order_id']) for r in self.rows},{s:s[:-4] for s in self.metadata})
        balances={'USDT':{'free':str(Decimal('100')+Decimal(inv['deltas'].get('USDT','0'))),'locked':'0'}}
        for asset,value in inv['deltas'].items():
            if asset!='USDT':balances[asset]=dict(free=value,locked='0')
        return dict(now_ms=self.now,bid='99.99',ask='100',average='100',metadata=self.metadata[self.symbol],balances=balances,
            marks={s[:-4]:('99.99','100') for s in self.metadata},stream_ok=True,
            open_orders=[dict(symbol=self.symbol,order_id=999)] if self.foreign else [],
            features=[0.,0.,.01,2.],feature_end_ms=self.now-1,snapshot_id='scoped-source',account_identity='a'*64)
    def fill_rows(self,keys):return [deepcopy(r) for r in self.rows if (r['symbol'],r['order_id']) in keys]
    def mutate(self,method,path,p):
        response=super().mutate(method,path,p)
        # The synthetic parent models fixed price100. Rebind its newly created
        # native objects to the requested instrument, without changing assertions.
        if path.endswith('/order') and method=='POST':self.rows[-1]['symbol']=p['symbol'];self.rows[-1]['commission_asset']=p['symbol'][:-4] if p['side']=='BUY' else 'USDT'
        response['symbol']=p['symbol']
        for o in response.get('orders',[]):o['symbol']=p['symbol']
        if 'orderId' in response:self.orders[response['orderId']]=response
        return response
    def query_action(self,method,path,p):
        result=super().query_action(method,path,p)
        if result is not None:result['symbol']=p['symbol']
        return result


class UsdtServiceTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.usdt_service')
        root=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures';root.mkdir(parents=True,exist_ok=True)
        tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(tmp.cleanup);self.paths=AppPaths(Path(tmp.name).resolve())
        self.exchange=ScopedExchange();self.book=UsdtPilotJournal(self.paths,account_identity='a'*64,start_ms=1000,
            expires_ms=86401000,initial_balances={'USDT':{'free':'100','locked':'0'}},enabled=True,clock_ms=lambda:self.exchange.now)
        self.addCleanup(self.book.close)
        c=usdt_catalog({'symbols':list(self.exchange.metadata.values())},['SPOT'],can_trade=True)
        self.book.authorize_universe(c,received_at_ms=2000,source_sha256='b'*64,validated_symbols=list(self.exchange.metadata))
        self.book.save_state(dict(phase='IDLE',cycles=0,qualified=False,entry_killed=False,peak_equity='100',
            last_equity='100',next_entry_ms=0,symbol='ETHUSDT',base_asset='ETH'))
        self.service=self.api.UsdtPilotService(self.book,self.exchange)
    def test_usd02_non_btc_buy_gets_native_protection(self):
        s=self.service.tick();self.assertEqual(s['phase'],'OPEN')
        self.assertTrue(all(p['symbol']=='ETHUSDT' for _,_,p in self.exchange.calls))
    def test_usd03_bad_pair_before_send_is_quarantined_without_global_halt(self):
        next(f for f in self.exchange.metadata['ETHUSDT']['filters'] if f['filterType']=='LOT_SIZE')['minQty']='100'
        s=self.service.tick();self.assertEqual(s['phase'],'IDLE');self.assertEqual(self.exchange.calls,[])
        self.assertEqual(s['pair_blocks'][0]['symbol'],'ETHUSDT');self.assertFalse(s['entry_killed'])
    def test_usd03_balance_failure_is_not_downgraded_to_pair_pause(self):
        original=self.exchange.snapshot
        def changed():
            s=original();s['balances']['USDT']['free']='101';return s
        self.exchange.snapshot=changed
        self.assertEqual(self.service.tick()['phase'],'HALTED')
    def test_usd02_non_btc_cycle_closes_and_exports_symbol(self):
        self.service.tick();self.exchange.now+=61000;s=self.service.tick()
        self.assertEqual(s['phase'],'IDLE');self.assertEqual(self.book.outcomes()[0]['symbol'],'ETHUSDT')
        self.assertTrue(self.book.outcomes()[0]['private_delivery_matched'])
    def test_usd02_numeric_id_collision_cannot_hide_foreign_order(self):
        self.service.tick();snapshot=self.exchange.snapshot()
        snapshot['open_orders']=[dict(symbol='SOLUSDT',order_id=1)]
        self.assertTrue(self.service._foreign_orders(snapshot))
    def test_usd02_previous_asset_dust_stays_in_portfolio_valuation(self):
        self.service.tick();self.exchange.now+=61000;self.service.tick()
        original=self.book.state()['last_equity'];state=self.book.state();state.update(symbol='SOLUSDT',base_asset='SOL')
        self.book.save_state(state);self.exchange.symbol='SOLUSDT'
        snapshot=self.exchange.snapshot();inventory=self.service._inventory()
        self.assertEqual(self.service._equity(inventory,snapshot),Decimal(original))
    def test_usd06_expiry_stops_all_symbols_without_new_budget(self):
        self.exchange.now=86402000;s=self.service.tick()
        self.assertEqual(s['phase'],'FINISHED');self.assertEqual(self.exchange.calls,[])
    def test_usd02_one_shared_attempt_limit_for_entire_universe(self):
        s=self.book.state();s.update(cycles=8,qualified=True);self.book.save_state(s)
        self.assertEqual(self.service.tick()['phase'],'FINISHED');self.assertEqual(self.exchange.calls,[])
    def test_usd06_missing_mark_cannot_invent_flatness(self):
        self.service.tick();snapshot=self.exchange.snapshot();snapshot['marks']={}
        with self.assertRaises(AdapterError):self.service._equity(self.service._inventory(),snapshot)

    def test_usd03_owned_exit_reuses_frozen_features_without_new_candles(self):
        state=dict(phase='OPEN',features=[1.,2.,3.,4.],feature_end_ms=1000,feature_available_at_ms=1500)
        values,end,available=self.api.forecast_features(state,None,None,now_ms=3000)
        self.assertEqual((values,end,available),([1.,2.,3.,4.],1000,1500))

    def test_usd03_read_failure_publishes_fresh_paused_status(self):
        report=self.api.operational_report(dict(phase='IDLE',entry_killed=False),
            dict(monitored_pairs_N=519,quote_ready_n=519,quotes_received_at_ms=1000),pid=1,now_ms=40000,
            expires_ms=100000,actual_outcomes_n=1,stream={},error='MARKET_RATE_BUDGET')
        self.assertEqual(report['operational_status'],'PAUSED');self.assertFalse(report['entry_enabled'])
        self.assertIsNone(report['monitoring']['quote_ready_n']);self.assertEqual(report['checked_at_ms'],40000)


if __name__=='__main__':unittest.main()
