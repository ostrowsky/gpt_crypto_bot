"""Native adapter/store coverage of monitoring and composite execution reads."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from _p2_support import Fixture,book
from binance_demo_bot.adapter import BinanceDemoAdapter
from binance_demo_bot.isolation import DemoCredentials
from binance_demo_bot.usdt_service import UsdtGateway
from binance_demo_bot.pilot_service import LiveGateway
from test_adapter import order,trade
from test_adapter import account
from binance_demo_bot.market_data import DemoMarketClient
from test_usdt_scope import info


class UsdtGatewayTests(Fixture,unittest.TestCase):
    def setUp(self):
        self.setup_market()
        self.gateway=UsdtGateway.__new__(UsdtGateway);self.gateway.client=self.read_client
        self.gateway.store=self.store;self.gateway.quotes={};self.gateway.quote_at=0;self.gateway.cache={}
        self.gateway.market=DemoMarketClient(self.read_client)
    def test_usd03_bulk_monitoring_uses_one_request_for_hundreds_of_pairs(self):
        self.replies.append([book('X'+str(i)+'USDT') for i in range(519)])
        self.gateway.poll_quotes(force=True)
        self.assertEqual(len(self.calls),1);self.assertEqual(len(self.gateway.quotes),519)
    def test_usd03_bad_quote_does_not_count_as_fresh_coverage(self):
        self.replies.append([book('ETHUSDT'),book('SOLUSDT',bid='0',ask='0')])
        self.gateway.poll_quotes(force=True)
        self.assertEqual(set(self.gateway.quotes),{'ETHUSDT'})
    def test_usd02_terminal_fill_cache_is_scoped_by_symbol(self):
        c=BinanceDemoAdapter(DemoCredentials('synthetic-key','synthetic-secret'),account_alias='demo-one',credential_generation='g1',
            episode_id='episode-one',scope_id='whole-account',identity_key=b'i'*32,transport=self.read_client.transport,clock=self.clock)
        self.addCleanup(c.close);self.gateway.client=c
        for s in ('ETHUSDT','SOLUSDT'):
            self.replies.extend([order(symbol=s,orderId=1,origQty='1',executedQty='1',status='FILLED'),
                [trade(symbol=s,orderId=1,commissionAsset=s[:-4])]])
        keys=[('ETHUSDT',1),('SOLUSDT',1)]
        rows=self.gateway.fill_rows(keys)
        self.assertEqual({r['symbol'] for r in rows},{'ETHUSDT','SOLUSDT'})
        self.assertEqual(len(self.calls),4)
        self.assertEqual(self.gateway.fill_rows(keys),rows);self.assertEqual(len(self.calls),4)
    def test_usd02_private_fee_asset_mismatch_is_not_delivery_proof(self):
        row=dict(symbol='ETHUSDT',order_id=1,trade_id=1,side='BUY',quantity='1',quote_quantity='100',
            commission_quantity='0.001',commission_asset='ETH')
        self.gateway.evidence=SimpleNamespace(fills=lambda:[dict(data=dict(row,commission_asset='BNB'))])
        self.assertFalse(LiveGateway.private_matches(self.gateway,[row]))
    def test_usd02_private_order_or_side_mismatch_is_not_delivery_proof(self):
        row=dict(symbol='ETHUSDT',order_id=1,trade_id=1,side='BUY',quantity='1',quote_quantity='100',
            commission_quantity='0',commission_asset=None)
        self.gateway.evidence=SimpleNamespace(fills=lambda:[dict(data=dict(row,order_id=2,side='SELL'))])
        self.assertFalse(LiveGateway.private_matches(self.gateway,[row]))
    def test_usd03_preflight_rechecks_spread_after_source_refresh(self):
        self.gateway.verified_histories={'ETHUSDT':'a'*64}
        snapshot=dict(metadata=dict(symbol='ETHUSDT'),bid='99',ask='100',bid_qty='100',ask_qty='100')
        from binance_demo_bot.adapter import AdapterError
        with patch.object(LiveGateway,'preflight',return_value=True):
            with self.assertRaises(AdapterError) as caught:self.gateway.preflight(snapshot)
        self.assertEqual(caught.exception.reason,'USDT_MARKET_NOT_READY')
    def test_usd03_owned_snapshot_keeps_request_headroom_for_protection(self):
        c=BinanceDemoAdapter(DemoCredentials('synthetic-key','synthetic-secret'),account_alias='demo-one',credential_generation='g1',
            episode_id='episode-one',scope_id='whole-account',identity_key=b'i'*32,transport=self.read_client.transport,clock=self.clock)
        self.addCleanup(c.close);g=self.gateway;g.client=c;g.symbol='ETHUSDT';g.market=DemoMarketClient(c)
        lower_priority=[]
        g.refresh_catalog=lambda:lower_priority.append('catalog');g.poll_quotes=lambda **kw:lower_priority.append('bulk')
        g.metadata={'ETHUSDT':info()};g.catalog=[dict(symbol='ETHUSDT',base_asset='ETH',metadata=info())]
        g.quotes={'ETHUSDT':book('ETHUSDT')};g.quote_at=self.wall;g.quote_source='source'
        g.selected_metadata_at={'ETHUSDT':self.wall}
        g.book=SimpleNamespace(state=lambda:dict(phase='OPEN',features=[0.,0.,1.,2.],feature_end_ms=self.wall-1000,origin_ms=self.wall-1),
            owned_order_keys=lambda:[('ETHUSDT',1)])
        g.reader=SimpleNamespace(health=lambda:dict(transport_healthy=True))
        self.replies.extend([account(),dict(price='100'),[],[book('ETHUSDT')]])
        with patch('binance_demo_bot.usdt_service.time.time_ns',return_value=self.wall*1000000):g.snapshot()
        self.assertLessEqual(sum(w for _,w in c._reservations),40)
        self.assertEqual(lower_priority,[])


if __name__=='__main__':unittest.main()
