"""FRI-04..06 causal research contracts; no network/credentials."""
import importlib,unittest
from copy import deepcopy
import _support

STEP=900000
def bars(n=40,start=0,price=100):
    return [[start+i*STEP,str(price),str(price+0.1),str(price-0.1),str(price),'1',start+(i+1)*STEP-1,'100',10,'0.5','50','0'] for i in range(n)]
def breakout():
    rows=bars(32)
    rows[-1][2:5]=['102.1','99.9','102'];rows[-1][7]='300'
    return rows

class TrendResearchTests(unittest.TestCase):
    def setUp(self):self.api=importlib.import_module('binance_demo_bot.trend_research')
    def signal(self,rows,**kw):return self.api.trend_signal(rows,now_ms=rows[-1][6]+1,**kw)
    def test_fri04_breakout_is_causal_ranked_candidate(self):
        result=self.signal(breakout());self.assertEqual(result['action'],'BUY');self.assertGreater(result['score'],0)
    def test_fri04_flat_or_low_volume_waits(self):
        self.assertEqual(self.signal(bars(32))['action'],'WAIT')
        rows=breakout();rows[-1][7]='100';self.assertEqual(self.signal(rows)['action'],'WAIT')
    def test_fri04_future_prefix_invariance(self):
        rows=breakout();now=rows[-1][6]+1
        result=self.api.trend_signal(rows,now_ms=now)
        future=bars(5,start=now,price=10000)
        self.assertEqual(result,self.api.trend_signal(rows+future,now_ms=now))
    def test_fri04_delayed_feature_receipt_blocks(self):
        rows=breakout();self.assertEqual(self.signal(rows,received_at_ms=rows[-1][6]+2)['action'],'BLOCK')
    def test_fri04_gap_and_bad_ohlc_block(self):
        rows=breakout();del rows[-3];self.assertEqual(self.signal(rows)['action'],'BLOCK')
        rows=breakout();rows[-1][2]='1';self.assertEqual(self.signal(rows)['action'],'BLOCK')
    def test_fri04_chasing_far_above_slow_mean_waits(self):
        rows=breakout();rows[-1][2:5]=['130','99','129'];self.assertEqual(self.signal(rows)['action'],'WAIT')
    def test_fri05_stop_first_when_intrabar_order_unknown(self):
        result=self.api.protective_exit(dict(entry=100,peak=100),[0,'100','106','98','104',0,STEP-1])
        self.assertEqual(result,(99.0,'STOP'))
    def test_fri05_current_bar_high_cannot_raise_its_own_stop(self):
        result=self.api.protective_exit(dict(entry=100,peak=100),[0,'100','103','99.5','102',0,STEP-1])
        self.assertIsNone(result)
    def test_fri05_gap_stop_uses_worse_open(self):
        self.assertEqual(self.api.protective_exit(dict(entry=100,peak=100),[0,'97','100','96','99',0,STEP-1]),(97.0,'STOP'))
    def test_fri05_trailing_uses_prior_peak(self):
        self.assertEqual(self.api.protective_exit(dict(entry=100,peak=103),[0,'103','103.1','101','102',0,STEP-1]),(101.455,'TRAIL'))
    def test_fri05_no_trade_control_is_same_cash(self):
        report=self.api.portfolio_replay({'AAAUSDT':bars()},start_ms=0,end_ms=40*STEP)
        self.assertEqual(report['net_pnl_usdt'],0);self.assertEqual(report['benchmark_net_pnl_usdt'],0)
        self.assertEqual(report['trades_n'],0);self.assertEqual(report['evidence_kind'],'SIMULATION')
    def test_fri05_shared_capacity_never_opens_two_positions(self):
        rows=breakout()+bars(20,start=32*STEP,price=102)
        report=self.api.portfolio_replay({'AAAUSDT':rows,'BBBUSDT':rows},start_ms=0,end_ms=52*STEP)
        self.assertEqual(report['max_concurrent_positions'],1)
        self.assertEqual(report['trades'][0]['symbol'],'AAAUSDT')
        self.assertEqual(report['trades'][0]['entry_ms'],32*STEP)
        self.assertGreater(report['fees_usdt'],0)
        self.assertLessEqual(report['max_entry_debit_usdt'],30.000000001)
    def test_fri05_stress_costs_reduce_same_path_result(self):
        rows=breakout()+bars(20,start=32*STEP,price=102)
        a=self.api.portfolio_replay({'AAAUSDT':rows},start_ms=0,end_ms=52*STEP)
        b=self.api.portfolio_replay({'AAAUSDT':rows},start_ms=0,end_ms=52*STEP,fee_bps=20,slippage_bps=10)
        self.assertLess(b['net_pnl_usdt'],a['net_pnl_usdt'])
    def test_fri05_held_data_gap_blocks_numeric_verdict(self):
        rows=breakout()+bars(20,start=32*STEP,price=102);del rows[34]
        report=self.api.portfolio_replay({'AAAUSDT':rows,'BBBUSDT':bars(52)},start_ms=0,end_ms=52*STEP)
        self.assertEqual(report['state'],'INCOMPLETE');self.assertIsNone(report['net_pnl_usdt'])
        self.assertIn('HELD_PRICE_GAP',report['reasons'])
    def test_fri05_events_processed_chronologically_regardless_symbol_input_order(self):
        rows=breakout()+bars(20,start=32*STEP,price=102)
        a=self.api.portfolio_replay({'AAAUSDT':rows,'BBBUSDT':rows},start_ms=0,end_ms=52*STEP)
        b=self.api.portfolio_replay({'BBBUSDT':rows,'AAAUSDT':rows},start_ms=0,end_ms=52*STEP)
        self.assertEqual(a,b)
    def test_fri05_date_splits_use_global_clock_not_row_index(self):
        result=self.api.time_splits(0,100*STEP)
        self.assertEqual(result,[('train',0,70*STEP),('tune',70*STEP,85*STEP),('test',85*STEP,100*STEP)])
    def test_fri05_invalid_entry_bar_blocks_verdict_instead_of_selecting_future_valid_bars(self):
        rows=breakout()+bars(20,start=32*STEP,price=102);rows[32][2]='1'
        report=self.api.portfolio_replay({'AAAUSDT':rows},start_ms=0,end_ms=52*STEP)
        self.assertEqual(report['state'],'INCOMPLETE');self.assertIn('ENTRY_BAR_INVALID',report['reasons'])
    def test_fri05_exit_gap_loss_stops_replacement_at_same_open(self):
        first=breakout()+bars(17,start=32*STEP,price=102)
        first[48][1:5]=['90','90.1','89.9','90']
        second=bars(49);second[47][2:5]=['102.1','99.9','102'];second[47][7]='300'
        second[48][1:5]=['102','102.1','101.9','102']
        report=self.api.portfolio_replay({'AAAUSDT':first,'BBBUSDT':second},start_ms=0,end_ms=49*STEP)
        self.assertEqual(report['trades_n'],1);self.assertEqual(report['open_positions_n'],0)
    def test_fri06_positive_simulation_cannot_grant_live_promotion(self):
        report=self.api.promotion_readiness(dict(state='COMPLETE',net_pnl_usdt=10),cohort_complete=True)
        self.assertFalse(report['promotion_allowed']);self.assertIn('ACTUAL_FORWARD_FINANCIAL_COHORT_PENDING',report['reasons'])
    def test_fri05_simulated_daily_metric_excludes_partial_days(self):
        report=self.api.portfolio_replay({'AAAUSDT':bars(40)},start_ms=0,end_ms=40*STEP)
        self.assertEqual(report['complete_days_N'],0);self.assertIsNone(report['mean_daily_net_pnl_usdt'])
        self.assertIsNone(report['profitable_days_ratio'])
    def test_fri05_simulated_complete_dst_day_has_correct_denominator(self):
        from binance_demo_bot.financial import day_bounds
        for day,count in [('2026-03-29',92),('2026-10-25',100)]:
            start,end=day_bounds(day,'Europe/Budapest')
            report=self.api.portfolio_replay({'AAAUSDT':bars(count,start=start)},start_ms=start,end_ms=end)
            self.assertEqual(report['complete_days_N'],1)
            self.assertEqual(report['mean_daily_net_pnl_usdt'],0)
            self.assertEqual(report['profitable_days_n'],0)
    def test_fri05_missing_calendar_slot_does_not_become_zero_profit_complete_day(self):
        from binance_demo_bot.financial import day_bounds
        start,end=day_bounds('2026-10-28','Europe/Budapest');rows=bars(96,start=start);del rows[50]
        report=self.api.portfolio_replay({'AAAUSDT':rows},start_ms=start,end_ms=end)
        self.assertEqual(report['complete_days_N'],0);self.assertIsNone(report['mean_daily_net_pnl_usdt'])
    def test_fri06_partial_cohort_cannot_pass_maximum_history_gate(self):
        report=self.api.promotion_readiness(dict(state='COMPLETE',net_pnl_usdt=10),cohort_complete=False)
        self.assertIn('MAXIMUM_COHORT_INCOMPLETE',report['reasons'])

if __name__=='__main__':unittest.main()
