"""USD-01..06: source-scoped discovery and account-wide ownership, no network."""
from copy import deepcopy
import importlib,tempfile,unittest
from pathlib import Path
from decimal import Decimal
import _support
from binance_demo_bot.adapter import AdapterError
from binance_demo_bot.isolation import AppPaths
from test_execution_pilot import metadata,Broker


def info(symbol='ETHUSDT'):
    m=metadata();m.update(symbol=symbol,baseAsset=symbol[:-4],quoteAsset='USDT',permissionSets=[['SPOT','TRD_GRP_271']]);return m


def fill(symbol='ETHUSDT',oid=1,tid=1,side='BUY',quantity='0.3',quote='30',fee='0.0003'):
    return dict(symbol=symbol,order_id=oid,trade_id=tid,side=side,quantity=quantity,quote_quantity=quote,
        commission_asset=symbol[:-4],commission_quantity=fee)


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.usdt_scope')
        root=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures';root.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=root);self.addCleanup(temporary.cleanup);self.paths=AppPaths(Path(temporary.name).resolve())
    def book(self):
        b=self.api.UsdtPilotJournal(self.paths,account_identity='a'*64,start_ms=1000,expires_ms=86401000,
            initial_balances={'USDT':{'free':'100','locked':'0'}},enabled=True,clock_ms=lambda:2000)
        self.addCleanup(b.close);return b
    def catalog(self,*rows):return self.api.usdt_catalog({'symbols':list(rows)},['SPOT'],can_trade=True)
    def test_usd01_all_usdt_not_a_fixed_coin_list(self):
        rows=[info('X'+str(i)+'USDT') for i in range(300)]
        c=self.catalog(*rows);self.assertEqual(len(c),300);self.assertTrue(all(r['monitored'] for r in c))
    def test_usd01_unicode_native_instrument_is_not_silently_excluded(self):
        m=info('\u5e01\u5b89\u4eba\u751fUSDT');r=self.catalog(m)[0]
        self.assertTrue(r['execution_supported'])
        from binance_demo_bot.execution_pilot import execution_plan
        self.assertEqual(execution_plan(m,'99.99','100','100',symbol=m['symbol'])['quote'],'30')
    def test_usd01_exact_quote_and_account_permission(self):
        btc=info('ETHBTC');btc.update(quoteAsset='BTC',baseAsset='ETH')
        denied=info('ABCUSDT');denied['permissionSets']=[['SPOT'],['TRD_GRP_271']]
        c=self.catalog(info(),btc,denied);self.assertEqual([r['symbol'] for r in c if r['monitored']],['ETHUSDT'])
    def test_usd01_oco_unsupported_stays_monitored(self):
        m=info();m['ocoAllowed']=False;c=self.catalog(m)[0]
        self.assertTrue(c['monitored']);self.assertFalse(c['execution_supported']);self.assertIn('OCO_UNSUPPORTED',c['reasons'])
    def test_usd01_conflicting_metadata_does_not_silently_pick_last(self):
        m=info();other=deepcopy(m);other['baseAsset']='FAKE'
        with self.assertRaises(AdapterError):self.catalog(m,other)
    def test_usd02_composite_order_and_trade_ids_do_not_collide(self):
        rows=[fill('ETHUSDT'),fill('SOLUSDT')];keys={('ETHUSDT',1),('SOLUSDT',1)}
        r=self.api.portfolio_inventory(rows,keys,{'ETHUSDT':'ETH','SOLUSDT':'SOL'})
        self.assertEqual(r['deltas'],{'ETH':'0.2997','SOL':'0.2997','USDT':'-60'})
        self.assertEqual(r['fills_n'],2)
    def test_usd02_foreign_pair_same_numeric_id_is_rejected(self):
        with self.assertRaises(AdapterError):self.api.portfolio_inventory([fill()],{('BTCUSDT',1)},{'ETHUSDT':'ETH'})
    def test_usd02_duplicate_fee_charged_once(self):
        row=fill();r=self.api.portfolio_inventory([row,row],{('ETHUSDT',1)},{'ETHUSDT':'ETH'})
        self.assertEqual(r['deltas']['ETH'],'0.2997');self.assertEqual(r['fills_n'],1)
    def test_usd02_conflicting_fill_is_rejected(self):
        r=fill();other=dict(r,commission_quantity='0.001')
        with self.assertRaises(AdapterError):self.api.portfolio_inventory([r,other],{('ETHUSDT',1)},{'ETHUSDT':'ETH'})
    def test_usd02_aggregate_marks_include_prior_btc_dust(self):
        value=self.api.portfolio_value({'USDT':'-30','ETH':'0.2997','BTC':'0.001'},{'ETH':('99','101'),'BTC':('999','1001')})
        self.assertEqual(value,Decimal('100.9700'))
    def test_usd02_missing_mark_is_not_zero(self):
        with self.assertRaises(AdapterError):self.api.portfolio_value({'ETH':'1'},{})
    def test_usd06_unvalidated_symbol_cannot_send(self):
        b=self.book();b.authorize_universe(self.catalog(info()),received_at_ms=2000,source_sha256='b'*64,validated_symbols=[])
        p=dict(symbol='ETHUSDT',side='BUY',type='MARKET',quoteOrderQty='30',newClientOrderId=b.client_id('buy-1'))
        with self.assertRaises(AdapterError):b.call('buy-1','POST','/api/v3/order',p,Broker())
    def test_usd03_authorization_ledger_does_not_repeat_unrelated_markets(self):
        b=self.book();other=info('SOLUSDC');other.update(baseAsset='SOL',quoteAsset='USDC')
        b.authorize_universe(self.catalog(info(),other),received_at_ms=2000,source_sha256='b'*64,validated_symbols=['ETHUSDT'])
        self.assertEqual(len(b._usdt_scope['rows']),1)
    def test_usd06_validated_non_btc_order_and_ack_are_scoped(self):
        b=self.book();b.authorize_universe(self.catalog(info()),received_at_ms=2000,source_sha256='b'*64,validated_symbols=['ETHUSDT'])
        broker=Broker();broker.result=dict(symbol='ETHUSDT',clientOrderId=b.client_id('buy-1'),orderId=1,status='FILLED')
        b.call('buy-1','POST','/api/v3/order',dict(symbol='ETHUSDT',side='BUY',type='MARKET',quoteOrderQty='30',
            newClientOrderId=b.client_id('buy-1')),broker)
        self.assertEqual(b.owned_order_keys(),[('ETHUSDT',1)])
    def test_usd06_revoked_scope_cannot_hide_known_ack_needed_for_protection(self):
        b=self.book();b.authorize_universe(self.catalog(info()),received_at_ms=2000,source_sha256='b'*64,validated_symbols=['ETHUSDT'])
        broker=Broker();broker.result=dict(symbol='ETHUSDT',clientOrderId=b.client_id('buy-1'),orderId=1,status='FILLED')
        p=dict(symbol='ETHUSDT',side='BUY',type='MARKET',quoteOrderQty='30',newClientOrderId=b.client_id('buy-1'))
        response=b.call('buy-1','POST','/api/v3/order',p,broker)
        b.authorize_universe([],received_at_ms=2000,source_sha256='c'*64,validated_symbols=[])
        self.assertEqual(b.call('buy-1','POST','/api/v3/order',p,broker),response)
        self.assertEqual(len(broker.calls),1)
    def test_usd06_public_sensitive_word_symbol_roundtrip_without_weakening_redaction(self):
        b=self.book();b.save_state(dict(phase='IDLE',symbol='COOKIEUSDT',base_asset='COOKIE',fees={'COOKIE':'0.1'}))
        self.assertEqual(b.state()['symbol'],'COOKIEUSDT');self.assertEqual(b.state()['fees'],{'COOKIE':'0.1'})
        with self.assertRaises(AdapterError):b.save_state(dict(phase='IDLE',password='forbidden'))
        b.close();b=self.book();self.assertEqual(b.state()['symbol'],'COOKIEUSDT')
    def test_usd06_native_ack_fee_asset_identifier_is_lossless(self):
        b=self.book();b.authorize_universe(self.catalog(info('COOKIEUSDT')),received_at_ms=2000,
            source_sha256='b'*64,validated_symbols=['COOKIEUSDT'])
        broker=Broker();broker.result=dict(symbol='COOKIEUSDT',clientOrderId=b.client_id('buy-1'),orderId=1,
            status='FILLED',fills=[dict(commissionAsset='COOKIE',commission='0.001')])
        b.call('buy-1','POST','/api/v3/order',dict(symbol='COOKIEUSDT',side='BUY',type='MARKET',quoteOrderQty='30',
            newClientOrderId=b.client_id('buy-1')),broker)
        self.assertEqual(b.actions()['buy-1']['response']['fills'][0]['commissionAsset'],'COOKIE')
    def test_usd03_selector_is_round_robin_and_not_a_shortlist(self):
        c=self.catalog(info('SOLUSDT'),info('ETHUSDT'));quotes={r['symbol']:{'bidPrice':'99.99','askPrice':'100','bidQty':'100','askQty':'100'} for r in c}
        pick=self.api.select_round_robin(c,quotes,{'ETHUSDT':2},validated={'ETHUSDT','SOLUSDT'})
        self.assertEqual(pick,'SOLUSDT')
    def test_usd03_selector_blocks_missing_quote_or_replay(self):
        self.assertIsNone(self.api.select_round_robin(self.catalog(info()),{}, {},validated=set()))


if __name__=='__main__':unittest.main()
