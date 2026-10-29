"""FSA01..FSA07: preregistered causal feature and financial contracts."""
import copy
import math
import unittest
from unittest.mock import patch
from binance_demo_bot.feature_ablation import build_features, training_label, replay, paired_bootstrap
from binance_demo_bot.financial import day_bounds
from binance_demo_bot.trend_research import STEP


def bar(at, price=100., high=None, low=None, quote=100., buy=50.):
    return [at,str(price),str(price if high is None else high),str(price if low is None else low),
            str(price),str(quote/price),at+STEP-1,str(quote),10,str(buy/price),str(buy),'0']


def windows(n=32):
    return {s:[bar(i*STEP,100+i*.1) for i in range(n)] for s in ('BTCUSDT','ETHUSDT')}


def always_buy(rows, *, now_ms):
    return {'action':'BUY','score':1.,'weakening':False}


class FeatureTests(unittest.TestCase):
    def test_FSA01_suffix_and_current_bar_cannot_change_features(self):
        source=windows(); origin=32*STEP
        expected=build_features(source,'BTCUSDT',origin)
        self.assertIsNotNone(expected)
        source['BTCUSDT'] += [bar(origin,10000),bar(origin+STEP,1)]
        source['ETHUSDT'] += [bar(origin,1),bar(origin+STEP,10000)]
        self.assertEqual(expected,build_features(source,'BTCUSDT',origin))
        self.assertEqual(set(expected),{'A0','A1','A2'})
        self.assertTrue(set(expected['A0']) < set(expected['A1']) < set(expected['A2']))
        self.assertTrue(all(math.isfinite(v) for arm in expected.values() for v in arm.values()))

    def test_FSA02_gaps_stale_or_unclosed_context_block_common_rows(self):
        source=windows(); source['ETHUSDT'].pop(14)
        self.assertIsNone(build_features(source,'BTCUSDT',32*STEP))
        self.assertIsNone(build_features(windows(),'BTCUSDT',33*STEP))
        source=windows(); source['ETHUSDT'][-1][6] += STEP
        self.assertIsNone(build_features(source,'BTCUSDT',32*STEP))

    def test_FSA03_zero_valid_invalid_and_missing_flow_block(self):
        source=windows(); source['BTCUSDT'][-1]=bar(31*STEP,quote=0,buy=0)
        features=build_features(source,'BTCUSDT',32*STEP)
        self.assertEqual(features['A2']['flow_zero_volume'],1.)
        self.assertEqual(features['A2']['flow_imbalance_1'],0.)
        for invalid in (None,'nan','-1','101'):
            broken=windows();broken['ETHUSDT'][-1][10]=invalid
            self.assertIsNone(build_features(broken,'BTCUSDT',32*STEP))

    def test_FSA03_imbalance_uses_quote_volume_weighting(self):
        source=windows(); source['BTCUSDT'][-1]=bar(31*STEP,quote=300,buy=300)
        features=build_features(source,'BTCUSDT',32*STEP)
        self.assertAlmostEqual(features['A2']['flow_imbalance_4'],.5)


class LabelTests(unittest.TestCase):
    def test_FSA02_fixed_maturity_no_bar17(self):
        rows=[bar(i*STEP) for i in range(48)];origin=32*STEP
        self.assertIsNone(training_label(rows,origin,48*STEP-1))
        result=training_label(rows,origin,48*STEP)
        self.assertIsNotNone(result); self.assertLess(result,0)
        changed=copy.deepcopy(rows)+[bar(48*STEP,10000)]
        self.assertEqual(result,training_label(changed,origin,48*STEP))
        self.assertIsNone(training_label(rows[:42]+rows[43:],origin,48*STEP))

    def test_FSA04_label_stop_first_and_tp_limit(self):
        rows=[bar(i*STEP) for i in range(48)]; origin=32*STEP
        rows[32]=bar(origin,100,high=105,low=98)
        stopped=training_label(rows,origin,48*STEP,fee_bps=0,slippage_bps=0)
        self.assertAlmostEqual(stopped,-.01)
        rows[32]=bar(origin,100,high=105,low=100)
        target=training_label(rows,origin,48*STEP,fee_bps=0,slippage_bps=5)
        self.assertAlmostEqual(target,.04)


class ReplayTests(unittest.TestCase):
    def test_FSA05_missing_selected_execution_no_future_filter(self):
        source=windows(35); del source['BTCUSDT'][32]
        with patch('binance_demo_bot.feature_ablation.trend_signal',always_buy):
            result=replay(source,32*STEP,35*STEP)
        self.assertEqual(result['state'],'INCOMPLETE')
        self.assertIsNone(result['net_pnl_usdt'])
        self.assertIn('SELECTED_EXECUTION_MISSING',result['reasons'])
        self.assertEqual(result['trades_n'],0)

    def test_FSA04_cash_fees_tp_conservation(self):
        source=windows(33)
        for symbol in source:
            source[symbol]=[bar(i*STEP) for i in range(33)]
            source[symbol][32]=bar(32*STEP,100,high=105,low=100)
        with patch('binance_demo_bot.feature_ablation.trend_signal',always_buy):
            result=replay(source,32*STEP,33*STEP)
        self.assertEqual(result['state'],'COMPLETE')
        self.assertEqual(result['max_concurrent_positions'],1)
        self.assertEqual(result['max_entry_debit_usdt'],30.)
        self.assertEqual(result['trades_n'],1)
        trade=result['trades'][0]
        self.assertEqual(trade['reason'],'TP')
        self.assertAlmostEqual(trade['exit_price'],trade['entry_price']*1.04)
        self.assertAlmostEqual(result['net_pnl_usdt'],trade['net_usdt'])
        self.assertGreater(result['fees_usdt'],0)

    def test_FSA04_prior_peak_does_not_trail_from_current_high(self):
        source={s:[bar(i*STEP) for i in range(34)] for s in ('BTCUSDT','ETHUSDT')}
        for s in source:source[s][32]=bar(32*STEP,100,high=103,low=99.8)
        with patch('binance_demo_bot.feature_ablation.trend_signal',always_buy):
            result=replay(source,32*STEP,33*STEP,fee_bps=0,slippage_bps=0)
        self.assertEqual(result['trades_n'],0)
        self.assertEqual(result['open_positions_n'],1)

    def test_FSA04_drawdown_includes_open_and_close_marks(self):
        source={s:[bar(i*STEP) for i in range(34)] for s in ('BTCUSDT','ETHUSDT')}
        for s in source:
            source[s][33]=bar(33*STEP,101,high=103,low=101)
            source[s][33][1]='103'
        with patch('binance_demo_bot.feature_ablation.trend_signal',always_buy):
            result=replay(source,32*STEP,34*STEP,fee_bps=0,slippage_bps=0)
        self.assertAlmostEqual(result['max_drawdown_usdt'],.6)
    def test_FSA01_replay_scorer_only_observes_closed_history(self):
        source=windows(36); calls=[]
        def scorer(symbol,histories,at):
            self.assertTrue(all(r[6]<at for rows in histories.values() for r in rows))
            calls.append((symbol,at));return -1.
        with patch('binance_demo_bot.feature_ablation.trend_signal',always_buy):
            result=replay(source,32*STEP,36*STEP,scorer=scorer)
        self.assertTrue(calls);self.assertEqual(result['trades_n'],0)
        self.assertAlmostEqual(result['net_pnl_usdt'],0)

    def test_FSA06_dst_complete_no_trade_day_and_reconciliation(self):
        start,end=day_bounds('2026-10-25','Europe/Budapest')
        self.assertEqual((end-start)//STEP,100)
        source={s:[bar(at) for at in range(start-32*STEP,end,STEP)] for s in ('BTCUSDT','ETHUSDT')}
        result=replay(source,start,end,scorer=lambda *_:-1.)
        self.assertEqual(result['state'],'COMPLETE')
        self.assertEqual(result['complete_days_N'],1)
        self.assertEqual(result['daily_pnl'],{'2026-10-25':0.})
        self.assertEqual(result['days']['2026-10-25']['expected_points_n'],100)
        self.assertAlmostEqual(sum(d['net_pnl_usdt'] for d in result['days'].values()),result['net_pnl_usdt'])

    def test_FSA06_missing_unheld_pair_still_incomplete(self):
        source=windows(35);del source['ETHUSDT'][33]
        result=replay(source,32*STEP,35*STEP,scorer=lambda *_:-1.)
        self.assertEqual(result['state'],'INCOMPLETE')
        self.assertIsNone(result['mean_daily_net_pnl_usdt'])


class BootstrapTests(unittest.TestCase):
    def test_FSA07_paired_block_deterministic_and_adequate(self):
        from datetime import date,timedelta
        dates=[(date(2026,1,1)+timedelta(days=i)).isoformat() for i in range(35)]
        base={d:.01 for d in dates};new={d:.11 for d in dates}
        one=paired_bootstrap(new,base,draws=100)
        self.assertEqual(one,paired_bootstrap(new,base,draws=100))
        self.assertEqual(one['state'],'COMPLETE');self.assertEqual(one['paired_days_N'],35)
        self.assertEqual(one['confidence'],.9875)
        self.assertAlmostEqual(one['mean_delta_usdt'],.1)
        self.assertAlmostEqual(one['ci_low_usdt'],.1)

    def test_FSA07_no_intersection_cherry_pick_or_gap_bridge(self):
        small={f'2026-01-{i:02}':0. for i in range(1,20)}
        self.assertEqual(paired_bootstrap(small,small)['state'],'INCONCLUSIVE')
        from datetime import date,timedelta
        full={(date(2026,1,1)+timedelta(days=i)).isoformat():0. for i in range(35)}
        broken=dict(full);broken.pop('2026-01-15')
        self.assertEqual(paired_bootstrap(full,broken)['state'],'INCONCLUSIVE')
        self.assertEqual(paired_bootstrap(broken,broken)['state'],'INCONCLUSIVE')

if __name__=='__main__':unittest.main()
