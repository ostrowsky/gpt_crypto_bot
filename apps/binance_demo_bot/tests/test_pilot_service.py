"""PIL-04/05/08 full controller cycles on a deterministic synthetic exchange."""
from copy import deepcopy
from pathlib import Path
from decimal import Decimal
import importlib,tempfile,unittest
import _support
from test_execution_pilot import metadata
from binance_demo_bot.execution_pilot import PilotJournal,fill_inventory


class Exchange:
    def __init__(self):
        self.now=2000;self.calls=[];self.rows=[];self.orders={};self.lists={};self.counter=0
        self.foreign=False;self.stream=True;self.delivered=True;self.reject_protection=False;self.unknown_protection=False
        self.unknown_buy=False
        self.partial_buy=False
        self.trade_permission=True
    def preflight(self,snapshot):return self.trade_permission
    def snapshot(self):
        inv=fill_inventory(self.rows,set(self.orders))
        return dict(now_ms=self.now,bid='99.99',ask='100',average='100',metadata=metadata(),
            balances={'USDT':{'free':str(Decimal('100')+Decimal(inv['cash_delta'])),'locked':'0'},
                      'BTC':{'free':inv['base'],'locked':'0'}},stream_ok=self.stream,
            open_orders=[{'order_id':999}] if self.foreign else [],
            features=[0.,0.,.01,2.],feature_end_ms=self.now-1,snapshot_id='market-fixture',account_identity='a'*64)
    def mutate(self,method,path,p):
        from binance_demo_bot.adapter import AdapterError
        self.calls.append((method,path,deepcopy(p)));self.counter+=1
        if path.endswith('/oco'):
            if self.reject_protection:raise AdapterError('PILOT_REJECTED')
            if self.unknown_protection:raise TimeoutError()
            orders=[dict(symbol='BTCUSDT',orderId=self.counter*10+i,clientOrderId=p[k])
                    for i,k in enumerate(('aboveClientOrderId','belowClientOrderId'))]
            r=dict(symbol='BTCUSDT',orderListId=self.counter,listClientOrderId=p['listClientOrderId'],
                   listOrderStatus='EXECUTING',orders=orders)
            self.lists[p['listClientOrderId']]=r
            for order in orders:self.orders[order['orderId']]=dict(order,status='NEW',executedQty='0')
            return r
        if method=='DELETE':
            if path.endswith('/order'):
                r=deepcopy(self.orders[p['orderId']]);r['status']='CANCELED';self.orders[p['orderId']]=r;return r
            r=deepcopy(self.lists[p['listClientOrderId']]);r['listOrderStatus']='ALL_DONE'
            self.lists[p['listClientOrderId']]=r
            for o in r['orders']:self.orders[o['orderId']]['status']='CANCELED'
            return r
        qty=Decimal(p['quoteOrderQty'])/100 if p['side']=='BUY' else Decimal(p['quantity'])
        if self.partial_buy and p['side']=='BUY':qty/=2
        fee=qty*Decimal('.001') if p['side']=='BUY' else qty*100*Decimal('.001')
        row=dict(symbol='BTCUSDT',trade_id=self.counter,order_id=self.counter,side=p['side'],quantity=str(qty),
            quote_quantity=str(qty*100),commission_asset='BTC' if p['side']=='BUY' else 'USDT',commission_quantity=str(fee))
        self.rows.append(row)
        r=dict(symbol='BTCUSDT',clientOrderId=p['newClientOrderId'],orderId=self.counter,
               status='PARTIALLY_FILLED' if self.partial_buy and p['side']=='BUY' else 'FILLED',
               executedQty=str(qty),cummulativeQuoteQty=str(qty*100))
        self.orders[self.counter]=r
        if self.unknown_buy and p['side']=='BUY':raise TimeoutError()
        return r
    def query_action(self,method,path,p):
        if 'orderList' in path:return self.lists.get(p['listClientOrderId'])
        return next((r for r in self.orders.values() if r.get('clientOrderId')==p.get('newClientOrderId')),None)
    def fill_rows(self,ids):return [deepcopy(r) for r in self.rows if r['order_id'] in ids]
    def protection_ok(self,response,quantity):return response['listOrderStatus']=='EXECUTING'
    def private_matches(self,rows):return self.delivered and bool(rows)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.pilot_service')
        from binance_demo_bot.isolation import AppPaths
        root=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures';root.mkdir(parents=True,exist_ok=True)
        temp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(temp.cleanup)
        self.paths=AppPaths(Path(temp.name).resolve());self.exchange=Exchange()
        self.book=PilotJournal(self.paths,account_identity='a'*64,start_ms=1000,expires_ms=86401000,
            initial_balances={'USDT':{'free':'100','locked':'0'}},enabled=True);self.addCleanup(self.book.close)
        self.service=self.api.PilotService(self.book,self.exchange)
    def test_pil04_buy_immediately_protected(self):
        state=self.service.tick();self.assertEqual(state['phase'],'OPEN')
        self.assertEqual(len(self.exchange.calls),2)
        self.assertEqual(self.exchange.calls[1][1],'/api/v3/orderList/oco')
    def test_pil04_first_probe_time_exit_cancels_only_owned_then_sells(self):
        self.service.tick();self.exchange.now+=61000;state=self.service.tick()
        self.assertEqual(state['phase'],'IDLE');self.assertEqual(len(self.book.outcomes()),1)
        self.assertEqual([x[0] for x in self.exchange.calls],['POST','POST','DELETE','POST'])
        row=self.book.outcomes()[0];self.assertEqual(row['state'],'CLOSED')
        self.assertFalse(row['daily_objective_eligible']);self.assertTrue(row['technical_probe'])
    def test_pil05_foreign_order_blocks_before_buy(self):
        self.exchange.foreign=True;state=self.service.tick()
        self.assertEqual(state['phase'],'HALTED');self.assertEqual(self.exchange.calls,[])
    def test_pil05_no_private_connection_blocks_before_buy(self):
        self.exchange.stream=False;state=self.service.tick()
        self.assertEqual(self.exchange.calls,[]);self.assertEqual(state['phase'],'IDLE')
    def test_pil01_trade_permission_preflight_before_any_buy(self):
        self.exchange.trade_permission=False;self.service.tick();self.assertEqual(self.exchange.calls,[])
    def test_pil05_protection_rejection_emergency_sale_not_second_buy(self):
        self.exchange.reject_protection=True;state=self.service.tick()
        self.assertIn(state['phase'],('HALTED','IDLE'))
        self.assertEqual(sum(p.get('side')=='BUY' for m,path,p in self.exchange.calls),1)
        self.assertTrue(any(p.get('side')=='SELL' and path.endswith('/order') for m,path,p in self.exchange.calls))
    def test_pil05_unknown_protection_no_blind_sell(self):
        self.exchange.unknown_protection=True;state=self.service.tick()
        self.assertEqual(state['phase'],'RECOVERY_REQUIRED')
        self.assertFalse(any(p.get('side')=='SELL' and path.endswith('/order') for m,path,p in self.exchange.calls))
    def test_pil05_source_balance_change_not_adopted(self):
        original=self.exchange.snapshot
        def changed():
            s=original();s['balances']['USDT']['free']='101';return s
        self.exchange.snapshot=changed;state=self.service.tick()
        self.assertEqual(state['phase'],'HALTED');self.assertEqual(self.exchange.calls,[])
    def test_pil05_after_first_cycle_requires_actual_delivery(self):
        self.exchange.delivered=False;self.service.tick();self.exchange.now+=61000;state=self.service.tick()
        self.assertFalse(state.get('qualified',False))
        self.exchange.now+=10800000;self.service.tick()
        self.assertEqual(sum(p.get('side')=='BUY' for m,path,p in self.exchange.calls),1)
    def test_pil06_stop_marker_does_not_remove_existing_protection(self):
        self.service.tick();self.paths.runtime_path('pilot/stop-entries').write_text('stop')
        self.exchange.now+=1000;state=self.service.tick()
        self.assertEqual(state['phase'],'OPEN');self.assertEqual(len(self.exchange.calls),2)
    def test_pil08_outcome_features_frozen_before_future_changes(self):
        self.service.tick();self.exchange.now+=61000
        original=self.exchange.snapshot
        def changed():
            s=original();s['features']=[99.]*4;return s
        self.exchange.snapshot=changed;self.service.tick()
        self.assertEqual(self.book.outcomes()[0]['features'],[0.,0.,.01,2.])

    def test_pil03_unknown_accepted_buy_recovered_without_second_buy(self):
        self.exchange.unknown_buy=True;self.service.tick();self.exchange.unknown_buy=False
        state=self.service.recover_once()
        self.assertEqual(state['phase'],'OPEN')
        self.assertEqual(sum(p.get('side')=='BUY' for m,path,p in self.exchange.calls),1)

    def test_pil05_invalid_protection_prices_after_known_buy_emergency_exit(self):
        original=self.exchange.snapshot
        def changed():
            s=original()
            if self.exchange.rows:s.update(bid='100.59',ask='100.6')
            return s
        self.exchange.snapshot=changed;state=self.service.tick()
        self.assertIn(state['phase'],('IDLE','HALTED'))
        self.assertTrue(any(p.get('side')=='SELL' and path.endswith('/order') for m,path,p in self.exchange.calls))
    def test_pil05_partial_entry_cancelled_then_net_fill_protected(self):
        self.exchange.partial_buy=True;state=self.service.tick()
        self.assertEqual(state['phase'],'OPEN');self.assertTrue(state['entry_killed'])
        self.assertEqual([path for method,path,p in self.exchange.calls],
            ['/api/v3/order','/api/v3/order','/api/v3/orderList/oco'])
        self.assertEqual(self.exchange.calls[1][0],'DELETE')


if __name__=='__main__':unittest.main()
