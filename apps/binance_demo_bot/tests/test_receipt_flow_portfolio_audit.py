"""RFP09 independent arithmetic from fabricated economic facts, not model scores."""
from collections import Counter
from copy import deepcopy
import unittest
from binance_demo_bot import receipt_flow_portfolio_audit as audit
from binance_demo_bot.receipt_flow_portfolio import Policy,digest


def example():
    def event(kind,t,**kw):return dict(kind=kind,at_ms=t,evidence_kind='SIMULATION',**kw)
    selected=dict(symbol='BTCUSDT',source_row=0,feature_sha256='f'*64,score_bps=2.,spread_bps=0.)
    payload=dict(origin_ms=0,cash='100',position_symbol=None,day_halt=False,global_halt=False,
                 candidate_rows=[0],selected=selected,reason=None)
    identity=digest(payload)
    def quote(t,price):return dict(symbol='BTCUSDT',bid=price,ask=price,bid_qty='1',ask_qty='1',
                                   receipt_ms=t-1,session_id='s',chunk='fixture',chunk_sha256='a'*64,record_index=0)
    daily=dict(day='1970-01-01',start_ms=0,end_ms=9000,full_calendar_day=False,
               opening_equity='100',closing_equity='103',boundary_delta_usdt='3',pnl_usdt=None,
               complete=False,unmarked_held_ticks_n=0)
    exit_intent=dict(decision_ms=6000,eligible_ms=7000,reason='HOLDING_DEADLINE')
    busy=dict(origin_ms=5000,cash='70',position_symbol='BTCUSDT',day_halt=False,global_halt=False,
              candidate_rows=[],selected=None,reason='POSITION_OR_INTENT')
    events=[event('INITIAL_EQUITY',0,cash='100',equity='100'),
        event('DECISION',0,decision_id=identity,**payload),
        event('BUY_INTENT',0,origin_ms=0,eligible_ms=1000,expires_ms=3000,signal=selected,decision_id=identity),
        event('BUY_FILL',1000,symbol='BTCUSDT',quantity='.3',debit='30',entry_ms=1000,origin_ms=0,
              deadline_ms=6000,decision_id=identity,price='100',notional='30',fee='0',cash='70',quote=quote(1000,'100')),
        event('MARK',1000,equity='100',cash='70',position=dict(symbol='BTCUSDT',quantity='.3'),quote=quote(1000,'100')),
        event('DECISION',5000,decision_id=digest(busy),**busy),
        event('SELL_INTENT',6000,**exit_intent,symbol='BTCUSDT',decision_id=identity),
        event('SELL_FILL',7000,symbol='BTCUSDT',quantity='.3',price='110',notional='33',fee='0',cash='103',
              trade_net_usdt='3',entry_debit='30',entry_ms=1000,entry_decision_ms=0,
              decision_id=identity,exit_intent=exit_intent,quote=quote(7000,'110')),
        event('BOUNDARY',9000,**daily,quote=None)]
    result=dict(cash='103',position=None,final_equity='103',net_interval_usdt='3',closed_net_usdt='3',
                open_unrealized_usdt='0',fees_usdt='0',turnover_usdt='63',entries_n=1,closed_cycles_n=1,
                initial_equity='100',daily=[daily],complete_days_n=0,requested_days_N=0,profitable_days_n=0,
                mean_daily_net_pnl_usdt=None,drawdown_observed_usdt='0',unmarked_held_ticks_n=0,
                event_counts=dict(Counter(e['kind'] for e in events)))
    return events,result,Policy(fee_bps='0',slippage_bps='0')


class IndependentLedgerContract(unittest.TestCase):
    def test_RFP09_reconstructs_cash_fee_inventory_and_partial_day(self):
        events,result,policy=example()
        verified=audit.audit_events(iter(events),result,policy)
        self.assertEqual(verified['state'],'PASS_LEDGER_ARTIFACT_AUDIT_ONLY')
        self.assertEqual(verified['closed_cycles_n'],1)

    def test_RFP09_fee_or_cash_tampering_is_rejected(self):
        events,result,policy=example();events[3]['fee']='1'
        with self.assertRaises(ValueError):audit.audit_events(iter(events),result,policy)

    def test_RFP09_future_or_unowned_fill_is_rejected(self):
        events,result,policy=example();events[3]['symbol']='ETHUSDT'
        with self.assertRaises(ValueError):audit.audit_events(iter(events),result,policy)

    def test_RFP09_partial_day_cannot_become_zero_objective(self):
        events,result,policy=example();result['daily'][0]['pnl_usdt']='0'
        with self.assertRaises(ValueError):audit.audit_events(iter(events),result,policy)

    def test_RFP09_sell_fill_cannot_replace_logged_intent_with_past_risk(self):
        events,result,policy=example()
        sell=next(e for e in events if e['kind']=='SELL_FILL')
        sell['exit_intent']=dict(decision_ms=-1000,eligible_ms=0,reason='RISK')
        with self.assertRaises(ValueError):audit.audit_events(iter(events),result,policy)

    def test_RFP09_cancelled_buy_cannot_later_be_counted_as_fill(self):
        events,result,policy=example();identity=events[1]['decision_id']
        events.insert(3,dict(kind='BUY_CANCELLED',at_ms=500,evidence_kind='SIMULATION',
                             decision_id=identity,reason='RISK_LATCH'))
        result['event_counts']=dict(Counter(e['kind'] for e in events))
        with self.assertRaises(ValueError):audit.audit_events(iter(events),result,policy)


if __name__=='__main__':unittest.main()
