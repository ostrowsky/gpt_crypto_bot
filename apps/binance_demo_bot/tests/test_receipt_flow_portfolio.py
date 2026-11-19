"""RFP02..09: causal shared cash, delayed fills and honest daily marks."""
from copy import deepcopy
from decimal import Decimal as D
import unittest
from binance_demo_bot import receipt_flow_portfolio as p
from binance_demo_bot.financial import day_bounds


def signal(symbol='BTCUSDT',score=2.,spread=0.,row=0):
    return dict(symbol=symbol,score_bps=score,spread_bps=spread,source_row=row,
                feature_sha256='f'*64)


def quote(symbol,t,*,bid='100',ask=None,bid_qty='1000000',ask_qty='1000000'):
    return dict(symbol=symbol,bid=bid,ask=ask or bid,bid_qty=bid_qty,ask_qty=ask_qty,
        receipt_ms=t-1,session_id='s',chunk='fixture',chunk_sha256='a'*64,record_index=0)


def policy(**kwargs):
    return p.Policy(**dict(dict(fee_bps='0',slippage_bps='0',day_loss='99',
                              global_drawdown='99'),**kwargs))


class PortfolioContracts(unittest.TestCase):
    def test_RFP02_future_feature_timestamp_is_rejected(self):
        row=signal();row['feature_end_ms']=1
        with self.assertRaisesRegex(ValueError,'SIGNAL_TIMING'):
            p.simulate({0:[row]},start_ms=0,end_ms=9000,
                       quote_at=lambda s,t:quote(s,t),policy=policy())

    def test_RFP02_selection_is_past_only_cost_gate_and_stable_tie(self):
        rows=[signal('SOLUSDT',5),signal('BTCUSDT',5),signal('ETHUSDT',-2)]
        self.assertEqual(p.select_signal(rows,policy())['symbol'],'BTCUSDT')
        self.assertIsNone(p.select_signal([signal(score=3,spread=3)],policy(admission='cost-aware')))
        self.assertIsNotNone(p.select_signal([signal(score=5,spread=3)],policy(admission='cost-aware')))
        self.assertIsNone(p.select_signal([signal(score=0)],policy()))

    def test_RFP02_missing_selected_quote_never_substitutes_other_coin(self):
        lookups=[]
        def source(s,t):
            lookups.append((s,t));return None if s=='BTCUSDT' else quote(s,t)
        r=p.simulate({0:[signal('BTCUSDT',3),signal('ETHUSDT',2)]},start_ms=0,end_ms=10000,
                     quote_at=source,policy=policy())
        self.assertEqual(r['entries_n'],0)
        self.assertTrue(all(s=='BTCUSDT' for s,t in lookups))
        self.assertTrue(any(e['kind']=='BUY_EXPIRED' for e in r['events']))

    def test_RFP02_future_prices_do_not_change_frozen_decision(self):
        signals={0:[signal()]}
        a=p.simulate(signals,start_ms=0,end_ms=10000,quote_at=lambda s,t:quote(s,t),policy=policy())
        b=p.simulate(signals,start_ms=0,end_ms=10000,
                     quote_at=lambda s,t:quote(s,t,bid='120' if t>=4000 else '100'),policy=policy())
        first=lambda r:next(e for e in r['events'] if e['kind']=='DECISION' and e.get('selected'))
        self.assertEqual(first(a),first(b))

    def test_RFP03_decimal_fee_cash_and_quantity_reconcile_once(self):
        cfg=policy(fee_bps='10',slippage_bps='5')
        r=p.simulate({0:[signal()]},start_ms=0,end_ms=9000,
                     quote_at=lambda s,t:quote(s,t,bid='100',ask='101'),policy=cfg)
        qty=D('30')/(D('101')*D('1.0005')*D('1.001'))
        buy_notional=qty*D('101')*D('1.0005')
        sell_notional=qty*D('100')*D('.9995')
        expected=D('70')+sell_notional*D('.999')
        self.assertAlmostEqual(D(r['cash']),expected,places=22)
        self.assertAlmostEqual(D(r['fees_usdt']),(buy_notional+sell_notional)*D('.001'),places=22)
        self.assertEqual(r['closed_cycles_n'],1)
        self.assertIsNone(r['position'])
        self.assertAlmostEqual(D(r['net_interval_usdt']),D(r['closed_net_usdt']),places=22)

    def test_RFP03_insufficient_ask_capacity_rejects_without_debit(self):
        r=p.simulate({0:[signal()]},start_ms=0,end_ms=9000,
                     quote_at=lambda s,t:quote(s,t,ask_qty='.01'),policy=policy())
        self.assertEqual(r['entries_n'],0);self.assertEqual(D(r['cash']),100)
        self.assertTrue(any(e['kind']=='BUY_REJECTED_CAPACITY' for e in r['events']))

    def test_RFP03_exit_waits_for_full_bid_capacity_and_never_oversells(self):
        r=p.simulate({0:[signal()]},start_ms=0,end_ms=10000,
            quote_at=lambda s,t:quote(s,t,bid_qty='.01' if 7000<=t<9000 else '100'),policy=policy())
        exits=[e for e in r['events'] if e['kind']=='SELL_FILL']
        self.assertEqual(len(exits),1);self.assertEqual(exits[0]['at_ms'],9000)
        self.assertEqual(D(r['cash']),100)

    def test_RFP04_latency_holding_from_fill_and_exit_latency(self):
        r=p.simulate({0:[signal()]},start_ms=0,end_ms=10000,
                     quote_at=lambda s,t:quote(s,t),policy=policy())
        fills=[e for e in r['events'] if e['kind'] in ('BUY_FILL','SELL_FILL')]
        self.assertEqual([(e['kind'],e['at_ms']) for e in fills],[('BUY_FILL',1000),('SELL_FILL',7000)])
        intent=next(e for e in r['events'] if e['kind']=='SELL_INTENT')
        self.assertEqual(intent['at_ms'],6000)

    def test_RFP04_no_future_forced_liquidation_at_exclusive_end(self):
        r=p.simulate({0:[signal()]},start_ms=0,end_ms=7000,
                     quote_at=lambda s,t:quote(s,t),policy=policy())
        self.assertEqual(r['closed_cycles_n'],0);self.assertIsNotNone(r['position'])
        self.assertEqual(D(r['cash']),70)
        self.assertEqual(D(r['net_interval_usdt']),0)

    def test_RFP04_buy_ttl_exclusive_and_cooldown(self):
        r=p.simulate({0:[signal()],5000:[signal()]},start_ms=0,end_ms=8000,
            quote_at=lambda s,t:quote(s,t) if t>=3000 else None,policy=policy())
        self.assertEqual(r['entries_n'],0)
        self.assertEqual(next(e for e in r['events'] if e['kind']=='BUY_EXPIRED')['at_ms'],3000)

    def test_RFP05_midnight_mark_precedes_fill_and_next_day_has_fee(self):
        start,end=day_bounds('2026-11-11','Europe/Budapest');cut=end
        r=p.simulate({cut-10000:[signal()]},start_ms=cut-15000,end_ms=cut+10000,
            quote_at=lambda s,t:quote(s,t,bid_qty='.01' if cut-3000<=t<cut else '100'),
            policy=policy(fee_bps='10'))
        first=r['daily'][0];second=r['daily'][1]
        one_fee=D('30')/D('1.001')*D('.001')
        self.assertAlmostEqual(D(first['boundary_delta_usdt']),-one_fee,places=22)
        self.assertAlmostEqual(D(second['boundary_delta_usdt']),-one_fee,places=22)
        self.assertEqual(next(e for e in r['events'] if e['kind']=='SELL_FILL')['at_ms'],cut)

    def test_RFP07_shared_unknown_midnight_is_one_unmarked_clock(self):
        _,cut=day_bounds('2026-11-10','Europe/Budapest')
        r=p.simulate({cut-5000:[signal()]},start_ms=cut-10000,end_ms=cut+20000,
            quote_at=lambda s,t:None if t==cut else quote(s,t),policy=policy(horizon_seconds=30))
        self.assertEqual(r['unmarked_held_ticks_n'],1)

    def test_RFP07_final_mark_is_included_in_observed_drawdown(self):
        r=p.simulate({0:[signal()]},start_ms=0,end_ms=7000,
            quote_at=lambda s,t:quote(s,t,bid='90' if t==7000 else '100'),policy=policy())
        self.assertEqual(D(r['net_interval_usdt']),-3)
        self.assertEqual(D(r['drawdown_observed_usdt']),3)

    def test_RFP05_dst_calendar_bounds_are_not_24hour_utc(self):
        self.assertEqual(p.calendar_bounds('2026-03-29')[1]-p.calendar_bounds('2026-03-29')[0],23*3600000)
        self.assertEqual(p.calendar_bounds('2026-10-25')[1]-p.calendar_bounds('2026-10-25')[0],25*3600000)

    def test_RFP05_midnight_clock_is_processed_exactly_once(self):
        _,cut=day_bounds('2026-11-10','Europe/Budapest')
        r=p.simulate({},start_ms=cut-5000,end_ms=cut+5000,
                     quote_at=lambda s,t:quote(s,t),policy=policy())
        times=[e['at_ms'] for e in r['events'] if e['kind']=='DECISION']
        self.assertEqual(times,[cut-5000,cut])

    def test_RFP06_daily_risk_resets_only_midnight_without_cash_reset(self):
        start,cut=day_bounds('2026-11-10','Europe/Budapest')
        end=day_bounds('2026-11-11','Europe/Budapest')[1]
        def source(s,t):
            day_start=start if t<cut else cut
            return quote(s,t,bid='100' if t-day_start<2000 else '90')
        r=p.simulate({start:[signal()],start+5000:[signal()],cut:[signal()]},
            start_ms=start,end_ms=end,quote_at=source,policy=policy(day_loss='1',horizon_seconds=30))
        self.assertEqual(r['entries_n'],2)
        self.assertEqual(r['closed_cycles_n'],2)
        self.assertEqual(D(r['cash']),94)
        self.assertEqual(r['complete_days_n'],2)

    def test_RFP06_risk_exit_continues_after_entry_latch(self):
        r=p.simulate({0:[signal()],5000:[signal()]},start_ms=0,end_ms=20000,
            quote_at=lambda s,t:quote(s,t,bid='90' if t>=2000 else '100'),
            policy=policy(day_loss='.01',horizon_seconds=30))
        self.assertEqual(r['entries_n'],1);self.assertEqual(r['closed_cycles_n'],1)
        self.assertEqual(next(e for e in r['events'] if e['kind']=='SELL_FILL')['at_ms'],3000)

    def test_RFP06_missing_midnight_anchor_cannot_use_later_quote(self):
        _,cut=day_bounds('2026-11-10','Europe/Budapest')
        r=p.simulate({cut-5000:[signal()],cut+10000:[signal()]},start_ms=cut-10000,end_ms=cut+20000,
            quote_at=lambda s,t:None if t==cut else quote(s,t),policy=policy(horizon_seconds=30))
        self.assertEqual(r['entries_n'],1)
        self.assertTrue(all(d['pnl_usdt'] is None for d in r['daily']))
        self.assertTrue(any(e['kind']=='RISK_LATCH' and e['reason']=='UNKNOWN_DAY_OPEN' for e in r['events']))

    def test_RFP07_missing_held_mark_excludes_day_not_zero(self):
        start,end=day_bounds('2026-11-11','Europe/Budapest')
        r=p.simulate({start:[signal()]},start_ms=start,end_ms=end,
            quote_at=lambda s,t:None if t==start+2000 else quote(s,t),policy=policy())
        self.assertIsNone(r['daily'][0]['pnl_usdt']);self.assertEqual(r['complete_days_n'],0)
        self.assertIsNone(r['mean_daily_net_pnl_usdt']);self.assertEqual(D(r['net_interval_usdt']),0)

    def test_RFP07_flat_cash_needs_no_quote_and_is_known_zero(self):
        start,end=day_bounds('2026-11-11','Europe/Budapest')
        def fail(s,t):raise AssertionError('Flat cash does not need market quotes')
        r=p.simulate({},start_ms=start,end_ms=end,quote_at=fail,policy=policy())
        self.assertEqual(r['complete_days_n'],1)
        self.assertEqual(D(r['mean_daily_net_pnl_usdt']),0)
        self.assertEqual(r['profitable_days_n'],0)

    def test_RFP08_micro_calibration_uses_only_mature_train(self):
        rows=[dict(at_ms=t,feature_end_ms=t-1,label_available_ms=t+1000,micro_bps=x,target_bps=2+3*x)
              for t,x in ((0,0.),(5000,1.),(10000,2.),(20000,100.))]
        a=p.fit_microprice(rows,train_start_ms=0,train_end_ms=20000)
        rows[-1]['target_bps']=-999999
        b=p.fit_microprice(rows,train_start_ms=0,train_end_ms=20000)
        self.assertEqual(a,b);self.assertAlmostEqual(a['intercept'],2.);self.assertAlmostEqual(a['slope'],3.)
        self.assertEqual(a['train_rows_n'],3)

    def test_RFP08_zero_variance_micro_calibration_is_train_mean(self):
        rows=[dict(at_ms=t,feature_end_ms=t-1,label_available_ms=t+1,micro_bps=1.,target_bps=y)
              for t,y in ((0,2.),(5000,4.))]
        r=p.fit_microprice(rows,train_start_ms=0,train_end_ms=10000)
        self.assertEqual(r['slope'],0.);self.assertEqual(r['intercept'],3.)

    def test_RFP09_external_journal_preserves_decisions_without_memory_copy(self):
        events=[]
        r=p.simulate({0:[signal()]},start_ms=0,end_ms=9000,
            quote_at=lambda s,t:quote(s,t),policy=policy(),emit=events.append)
        self.assertIsNone(r['events'])
        self.assertEqual(r['event_counts']['BUY_FILL'],1)
        self.assertTrue(any(e['kind']=='DECISION' for e in events))
        self.assertTrue(all(e.get('evidence_kind')=='SIMULATION' for e in events))


if __name__=='__main__':unittest.main()
