from types import SimpleNamespace
import unittest
import numpy as np

from turnover_economics import (BAR,roundtrip_hurdle_pct,past_hour_range,screened_snapshot,
    replacement_enabled,cash_ledger,grouped_attribution,acceptance)
from portfolio_alpha import _simulate_account


def trade(sym='A',begin=BAR,end=3*BAR,entry=100,exit=101,**kw):
    values=dict(sym=sym,tf='15m',mode='trend',entry_ts=begin,exit_ts=end,entry_price=entry,
        exit_price=exit,exit_reason='test',partial_exit_taken=False,partial_exit_ts=0,
        partial_exit_price=0.,partial_exit_fraction=0.)
    values.update(kw);return SimpleNamespace(**values)


class EconomicsTests(unittest.TestCase):
    def bars(self):
        d=np.zeros(10,dtype=[(k,'i8' if k=='t' else 'f8') for k in ('t','o','h','l','c')])
        d['t']=np.arange(10)*BAR;d['o']=d['c']=100;d['h']=101;d['l']=99
        return d

    def test_cost_hurdle_is_exact_break_even(self):
        h=roundtrip_hurdle_pct(7.5,5)
        self.assertAlmostEqual((1+h/100)*(1-.0005)*(1-.00075)/((1+.0005)*(1+.00075)),1.)
        with self.assertRaises(ValueError):roundtrip_hurdle_pct(-1,5)

    def test_future_mutation_and_prefix_cannot_change_screen(self):
        d=self.bars();v=past_hour_range(d,5*BAR);b=d.copy();b[5:]['h']=9999
        self.assertEqual(v,past_hour_range(b,5*BAR));self.assertEqual(v,past_hour_range(d[:5],5*BAR))
        self.assertIsNone(past_hour_range(np.delete(d,3),5*BAR))
        self.assertIsNone(past_hour_range(d,5*BAR+1))

    def test_invalid_ohlc_fails_closed_and_candidate_objects_are_preserved(self):
        d=self.bars();c=SimpleNamespace(sym='A',ts_ms=5*BAR)
        result,a=screened_snapshot(({5*BAR:[c]},{5*BAR},1),{('A','15m'):(d,{})},7.5,5)
        self.assertIs(result[0][5*BAR][0],c);self.assertEqual(a['accepted'],1)
        d['h'][3]=1
        result,a=screened_snapshot(({5*BAR:[c]},{5*BAR},1),{('A','15m'):(d,{})},7.5,5)
        self.assertEqual(a['unknown_past'],1);self.assertEqual(result[2],0)

    def test_replacement_is_only_disabled_in_registered_arms(self):
        self.assertTrue(replacement_enabled('control',True))
        self.assertFalse(replacement_enabled('combined',True))
        self.assertFalse(replacement_enabled('control',False))

    def compare(self,trades):
        grid=list(range(BAR,6*BAR+1,BAR));series={s:[(t,100+t/BAR) for t in grid] for s in ('A','B')}
        actual=cash_ledger(trades,series,grid)
        reference=_simulate_account(trades,price_series_by_symbol=series,valuation_timestamps=grid,
            initial_capital=10000,capacity=10,fee_bps=7.5,slippage_bps=5)
        self.assertEqual(reference.violations,[])
        np.testing.assert_allclose(actual['curve'],reference.equity_curve,rtol=1e-12,atol=1e-9)
        self.assertAlmostEqual(actual['totals']['fees'],reference.fees_quote)
        self.assertAlmostEqual(actual['totals']['slippage'],reference.slippage_quote)
        self.assertAlmostEqual(sum(r['net_pnl'] for r in actual['ledger']),actual['ending_cash']-10000)
        return actual

    def test_cash_attribution_and_partial_exit_equal_canonical(self):
        r=self.compare([trade(partial_exit_taken=True,partial_exit_ts=2*BAR,
            partial_exit_fraction=.5,partial_exit_price=102),trade('B',begin=3*BAR,end=6*BAR)])
        g=grouped_attribution(r['ledger'],'mode')
        self.assertAlmostEqual(g['trend']['net_pnl'],r['totals']['net_pnl'])

    def test_same_clock_boundary_and_initial_peak(self):
        r=self.compare([trade(begin=6*BAR,end=6*BAR,entry=100,exit=100)])
        self.assertLess(r['net_return_pct'],0);self.assertGreater(r['max_drawdown_pct'],0)

    def test_missing_marks_and_duplicate_symbol_cannot_pass(self):
        with self.assertRaisesRegex(ValueError,'missing/stale'):
            cash_ledger([trade(end=6*BAR)],{'A':[(BAR,100)]},list(range(BAR,6*BAR+1,BAR)))
        with self.assertRaisesRegex(ValueError,'symbol/capacity'):
            self.compare([trade(),trade()])

    def test_better_pnl_but_lost_early_leaders_is_rejected(self):
        c=dict(net_return_pct=-50,test_return_pct=-20,trades=100,max_drawdown_pct=55)
        t=dict(net_return_pct=-30,test_return_pct=-10,trades=50,max_drawdown_pct=35)
        b=dict(early_pair_count=10,captured_pair_count=20,objective_trade_count=50,eligible_trade_count=100)
        d={**b,'early_pair_count':9}
        r=acceptance(c,t,(b,d),b,d,[.01,.02],35)
        self.assertEqual(r['numerical_gate'],'REJECTED');self.assertFalse(r['runtime_eligible'])


if __name__=='__main__':unittest.main()
