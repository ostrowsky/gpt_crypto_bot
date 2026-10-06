import unittest
from types import SimpleNamespace
import numpy as np
from impulse_entry_catboost import (BAR, HORIZON, DAY, features, net_target,
                                   fit_cohorts, admit, screen, gate)


def data(n=40):
    a = np.zeros(n, dtype=[(k, 'i8' if k == 't' else 'f8') for k in ('t','o','h','l','c','v')])
    a['t'] = np.arange(n)*BAR
    a['o'] = a['c'] = 100+np.arange(n)*.1
    a['h'] = a['c']+1; a['l'] = a['c']-1; a['v'] = 3
    return a


class CausalTests(unittest.TestCase):
    def test_future_mutation_and_prefix_equivalence(self):
        a = data(); at = 20*BAR
        expected = features(a, at, '15m')
        np.testing.assert_array_equal(expected, features(a[:20], at, '15m'))
        a['c'][20:] = 100000
        np.testing.assert_array_equal(expected, features(a, at, '15m'))
        self.assertEqual(len(expected), 10)

    def test_gap_invalid_and_insufficient_history(self):
        a = data(); self.assertIsNone(features(a, 10*BAR, '1h'))
        a['t'][16] += 1
        self.assertIsNone(features(a, 20*BAR, '1h'))
        a = data(); a['v'][18] = -1
        self.assertIsNone(features(a, 20*BAR, '1h'))

    def test_target_multiplicative_cost_and_future_tail(self):
        a = data(); at = 20*BAR
        expected = 100*(a['c'][24]/a['c'][19]*(1-.00075)*(1-.0005)/((1+.00075)*(1+.0005))-1)
        self.assertAlmostEqual(net_target(a, at, 7.5, 5), expected)
        self.assertIsNone(net_target(a[:22], at, 7.5, 5))
        self.assertIsNotNone(features(a[:22], at, '1h'))
        with self.assertRaises(ValueError): net_target(a, at, -1, 5)

    def test_purge_and_maturity(self):
        clocks = np.arange(0, 31*DAY, BAR)
        train, val, cut = fit_cohorts(clocks, np.ones(len(clocks),bool), 30*DAY, 0)
        self.assertTrue(np.all(clocks[train]+HORIZON < cut))
        self.assertTrue(np.all(clocks[val] >= cut))
        self.assertTrue(np.all(clocks[val]+HORIZON < 30*DAY))
        changed_valid = np.ones(len(clocks),bool); changed_valid[clocks >= 30*DAY] = False
        t2,v2,_ = fit_cohorts(clocks,changed_valid,30*DAY,0)
        np.testing.assert_array_equal(train,t2); np.testing.assert_array_equal(val,v2)

    def test_strict_gate_and_other_modes(self):
        self.assertFalse(admit(0)); self.assertFalse(admit(np.nan)); self.assertTrue(admit(.01))
        rows = [SimpleNamespace(mode=x) for x in ('impulse_speed','trend','impulse_speed')]
        result = screen(({1:rows},{1,2},3),{(1,0):False,(1,1):False})
        self.assertEqual(result[0][1],rows[1:]); self.assertEqual(result[1],{1,2})
        self.assertEqual(result[2],2)

    def test_no_turnover_hurdle_but_capture_and_risk_required(self):
        base = dict(net_return_pct=0,test_return_pct=0,trades=100,max_drawdown_pct=5)
        target = dict(base,net_return_pct=2,test_return_pct=2)
        m = dict(early_pair_count=10,captured_pair_count=15,objective_trade_count=20,eligible_trade_count=40)
        v = gate(base,target,(m,m),(m,m),[.1,1],35)
        self.assertEqual(v['numerical_gate'],'PASS'); self.assertNotIn('turnover_reduction',v['checks'])
        self.assertFalse(v['runtime_eligible'])
        v = gate(base,target,(m,dict(m,early_pair_count=9)),(m,m),[.1,1],35)
        self.assertEqual(v['numerical_gate'],'REJECTED')


if __name__ == '__main__': unittest.main()
