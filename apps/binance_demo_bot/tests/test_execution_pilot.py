"""PIL-01..06: native own journals and synthetic transport, never credentials."""
from copy import deepcopy
from decimal import Decimal,localcontext
import importlib
from pathlib import Path
import tempfile
import unittest

import _support
from binance_demo_bot.adapter import AdapterError,DemoHttpTransport
from binance_demo_bot.isolation import AppPaths


def metadata():
    return dict(symbol='BTCUSDT',baseAsset='BTC',quoteAsset='USDT',status='TRADING',
        isSpotTradingAllowed=True,ocoAllowed=True,quoteOrderQtyMarketAllowed=True,
        orderTypes=['MARKET','LIMIT_MAKER','STOP_LOSS'],filters=[
        dict(filterType='PRICE_FILTER',minPrice='0.01',maxPrice='1000000',tickSize='0.01'),
        dict(filterType='LOT_SIZE',minQty='0.001',maxQty='100',stepSize='0.001'),
        dict(filterType='MARKET_LOT_SIZE',minQty='0',maxQty='100',stepSize='0'),
        dict(filterType='NOTIONAL',minNotional='5',maxNotional='1000000',applyMinToMarket=True,applyMaxToMarket=False),
        dict(filterType='PERCENT_PRICE_BY_SIDE',bidMultiplierDown='0.5',bidMultiplierUp='1.2',
            askMultiplierDown='0.8',askMultiplierUp='2',avgPriceMins=5),
        dict(filterType='MAX_NUM_ORDERS',maxNumOrders=200),dict(filterType='MAX_NUM_ALGO_ORDERS',maxNumAlgoOrders=5),
        dict(filterType='MAX_NUM_ORDER_LISTS',maxNumOrderLists=20)])


class Broker:
    def __init__(self):self.calls=[];self.result=None;self.error=None;self.queries=[]
    def mutate(self,method,path,params):
        self.calls.append((method,path,deepcopy(params)))
        if self.error:raise self.error
        return self.result or dict(symbol='BTCUSDT',clientOrderId=params['newClientOrderId'],
            orderId=1,status='FILLED',executedQty='0.3',cummulativeQuoteQty='30')
    def query_action(self,method,path,params):
        self.queries.append((method,path,deepcopy(params)))
        return self.result


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.execution_pilot')
        root=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures';root.mkdir(parents=True,exist_ok=True)
        tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(tmp.cleanup)
        self.paths=AppPaths(Path(tmp.name).resolve());self.broker=Broker();self.book=self.make()
    def make(self,**changes):
        args=dict(account_identity='a'*64,start_ms=1000,expires_ms=86_401_000,
                  initial_balances={'USDT':{'free':'100','locked':'0'}},enabled=True)
        args.update(changes);b=self.api.PilotJournal(self.paths,**args);self.addCleanup(b.close);return b
    def reason(self,reason,call,*args,**kwargs):
        with self.assertRaises(AdapterError) as caught:call(*args,**kwargs)
        self.assertEqual(caught.exception.reason,reason)
    def buy(self,**changes):
        params=dict(symbol='BTCUSDT',side='BUY',type='MARKET',quoteOrderQty='30',
                    newClientOrderId=self.book.client_id('buy-1'),newOrderRespType='FULL')
        params.update(changes)
        return self.book.call('buy-1','POST','/api/v3/order',params,self.broker)

    def test_pil01_default_matching_disabled(self):
        self.book.close();self.book=self.make(enabled=False)
        self.reason('PILOT_DISABLED',self.buy);self.assertEqual(self.broker.calls,[])
    def test_pil01_read_transport_still_denies_matching(self):
        self.reason('FORBIDDEN_ENDPOINT',DemoHttpTransport()._request_https,'POST','/api/v3/order','',{},1)
    def test_pil01_matching_transport_pins_demo_host(self):
        self.reason('FORBIDDEN_ENDPOINT',self.api.DemoMatchingTransport,'https://api.binance.com/api/v3/')
        transport=self.api.DemoMatchingTransport()
        self.reason('FORBIDDEN_ENDPOINT',transport.request,'POST','/api/v3/withdraw','',{},1)
    def test_pil02_plan_exact_decimal_and_stop_relationship(self):
        with localcontext() as ctx:
            ctx.prec=2
            p=self.api.execution_plan(metadata(),'99.99','100','100',owned='0.3',fill_price='100')
        self.assertEqual(p,dict(quantity='0.299',take_profit='100.5',stop_loss='99.5',quote='30'))
    def test_pil02_unknown_filter_blocks(self):
        info=metadata();info['filters'].append(dict(filterType='FUTURE_FILTER'))
        self.reason('PILOT_FILTER_UNSUPPORTED',self.api.execution_plan,info,'99','100','100')
    def test_pil02_invalid_money_or_book_blocks(self):
        for bid,ask in ((101,'100'),('101','100'),('NaN','100'),('0','100')):
            self.reason('PILOT_INVALID',self.api.execution_plan,metadata(),bid,ask,'100')
    def test_pil02_nontrading_foreign_symbol_or_no_stop_refused(self):
        for change in (dict(symbol='ETHUSDT'),dict(status='BREAK'),dict(ocoAllowed=False),dict(orderTypes=['MARKET'])):
            m=metadata();m.update(change)
            self.reason('PILOT_FILTER_UNSUPPORTED',self.api.execution_plan,m,'99','100','100')
    def test_pil02_notional_and_percent_bounds(self):
        self.reason('PILOT_FILTER_INVALID',self.api.execution_plan,metadata(),'999','1000','1',owned='0.001',fill_price='1000')
    def test_pil02_native_order_count_must_fit_both_protection_legs(self):
        info=metadata();next(f for f in info['filters'] if f['filterType']=='MAX_NUM_ORDERS')['maxNumOrders']=1
        self.reason('PILOT_FILTER_INVALID',self.api.execution_plan,info,'99.99','100','100')
    def test_pil02_quote_budget_cannot_be_overridden(self):
        self.reason('PILOT_RISK_LIMIT',self.buy,quoteOrderQty='31');self.assertEqual(self.broker.calls,[])
    def test_pil03_repeated_logical_action_never_resends(self):
        first=self.buy();self.assertEqual(self.buy(),first);self.assertEqual(len(self.broker.calls),1)
    def test_pil03_conflicting_action_rejected(self):
        self.buy();self.reason('PILOT_ACTION_CONFLICT',self.buy,quoteOrderQty='29')
        self.assertEqual(len(self.broker.calls),1)
    def test_pil03_unknown_network_has_write_ahead_and_no_resend(self):
        self.broker.error=TimeoutError();self.reason('PILOT_QUERY_REQUIRED',self.buy)
        self.reason('PILOT_QUERY_REQUIRED',self.buy)
        self.assertEqual(len(self.broker.calls),1)
        self.assertEqual(self.book.actions()['buy-1']['state'],'QUERY_REQUIRED')
    def test_pil03_not_found_is_not_nonacceptance_proof(self):
        self.broker.error=TimeoutError();self.reason('PILOT_QUERY_REQUIRED',self.buy)
        self.reason('PILOT_QUERY_REQUIRED',self.book.recover,'buy-1',self.broker)
        self.assertEqual(len(self.broker.calls),1)
    def test_pil03_recover_exact_client_id_after_restart(self):
        self.broker.error=TimeoutError();self.reason('PILOT_QUERY_REQUIRED',self.buy)
        client=self.book.client_id('buy-1');self.book.close();self.book=self.make()
        self.broker.result=dict(symbol='BTCUSDT',clientOrderId=client,orderId=1,status='FILLED',executedQty='0.3',cummulativeQuoteQty='30')
        self.book.recover('buy-1',self.broker)
        self.assertEqual(self.book.actions()['buy-1']['state'],'ACKNOWLEDGED')
        self.assertEqual(len(self.broker.calls),1)
    def test_pil03_foreign_response_not_owned(self):
        self.broker.result=dict(symbol='BTCUSDT',clientOrderId='foreign',orderId=1,status='FILLED')
        self.reason('PILOT_QUERY_REQUIRED',self.buy)
        self.assertEqual(self.book.owned_order_ids(),[])
    def test_pil04_oco_exact_children_and_ownership(self):
        self.buy()
        p=dict(symbol='BTCUSDT',side='SELL',quantity='0.2',listClientOrderId=self.book.client_id('protect'),
            aboveClientOrderId=self.book.client_id('protect','tp'),belowClientOrderId=self.book.client_id('protect','sl'),
            aboveType='LIMIT_MAKER',abovePrice='100.5',belowType='STOP_LOSS',belowStopPrice='99.5')
        self.broker.result=dict(symbol='BTCUSDT',listClientOrderId=p['listClientOrderId'],orderListId=2,
            listOrderStatus='EXECUTING',orders=[dict(symbol='BTCUSDT',orderId=3,clientOrderId=p['aboveClientOrderId']),
                                              dict(symbol='BTCUSDT',orderId=4,clientOrderId=p['belowClientOrderId'])])
        self.book.call('protect','POST','/api/v3/orderList/oco',p,self.broker)
        self.assertEqual(self.book.owned_order_ids(),[1,3,4])
    def test_pil05_protection_without_owned_buy_refused(self):
        self.reason('PILOT_NOT_OWNED',self.book.call,'protect','POST','/api/v3/orderList/oco',
            dict(symbol='BTCUSDT',side='SELL',quantity='0.2'),self.broker)
    def test_pil05_unowned_cancel_refused(self):
        self.reason('PILOT_NOT_OWNED',self.book.call,'cancel','DELETE','/api/v3/orderList',
            dict(symbol='BTCUSDT',listClientOrderId='foreign'),self.broker)
    def test_pil05_fill_dedup_and_actual_fee(self):
        row=dict(symbol='BTCUSDT',trade_id=1,order_id=1,side='BUY',quantity='0.3',quote_quantity='30',
            commission_asset='BTC',commission_quantity='0.0003')
        result=self.api.fill_inventory([row,row],{1})
        self.assertEqual(result['base'],'0.2997');self.assertEqual(result['cash_delta'],'-30');self.assertEqual(result['fills_n'],1)
    def test_pil05_foreign_fill_not_adopted(self):
        row=dict(symbol='BTCUSDT',trade_id=1,order_id=999,side='BUY',quantity='0.3',quote_quantity='30',commission_asset='USDT',commission_quantity='0')
        self.reason('PILOT_FOREIGN_FILL',self.api.fill_inventory,[row],{1})
    def test_pil05_conflicting_duplicate_not_silently_deduped(self):
        row=dict(symbol='BTCUSDT',trade_id=1,order_id=1,side='BUY',quantity='0.3',quote_quantity='30',commission_asset='USDT',commission_quantity='0')
        other=dict(row,commission_quantity='0.1')
        self.reason('PILOT_FILL_CONFLICT',self.api.fill_inventory,[row,other],{1})
    def test_pil06_audit_failure_prevents_network(self):
        def fault(phase):
            if phase=='before_commit':raise OSError()
        self.book._audit.fault_hook=fault
        self.reason('AUDIT_WRITE_FAILED',self.buy);self.assertEqual(self.broker.calls,[])
    def test_pil06_frozen_identity_not_reset_on_restart(self):
        self.book.close();self.reason('PILOT_IDENTITY',self.make,account_identity='b'*64)
    def test_pil06_second_writer_refused(self):
        self.reason('PILOT_BUSY',self.make)
    def test_pil06_runtime_state_durable_detached(self):
        self.book.save_state({'phase':'IDLE','cycles':0});state=self.book.state();state['phase']='WRONG'
        self.book.close();self.book=self.make();self.assertEqual(self.book.state()['phase'],'IDLE')


if __name__=='__main__':unittest.main()
