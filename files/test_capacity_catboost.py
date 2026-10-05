import unittest

import numpy as np

from capacity_catboost import (BAR, causal_features, forward_return, cohort,
                               relevance, score_actions, evaluate, split_boundaries)


def series(n=40):
    a = np.zeros(n, dtype=[(x, 'i8' if x == 't' else 'f8') for x in ('t','o','h','l','c','v')])
    a['t'] = np.arange(n)*BAR
    a['o'] = a['c'] = np.linspace(100, 110, n)
    a['h'] = a['c']+1; a['l'] = a['c']-1; a['v'] = 10
    return a


class CapacityTest(unittest.TestCase):
    def test_future_mutation_cannot_change_features(self):
        a = series(); clock = 20*BAR
        expected = causal_features(a, clock, 1)
        b = a.copy(); b[20:]['c'] = 100000; b[20:]['v'] = 0
        np.testing.assert_array_equal(expected, causal_features(b, clock, 1))
        np.testing.assert_array_equal(expected, causal_features(a[:20], clock, 1))

    def test_no_stale_or_gap_features(self):
        a = series()
        self.assertIsNone(causal_features(a, 20*BAR+1, 1))
        self.assertIsNone(causal_features(np.delete(a, 10), 20*BAR, 1))

    def test_nonfinite_and_invalid_ohlc(self):
        for field, value in [('v', -1), ('c', np.nan), ('h', 1)]:
            a = series(); a[field][15] = value
            self.assertIsNone(causal_features(a, 20*BAR, 1))

    def test_all_five_future_closes_required(self):
        a = series()
        self.assertIsNone(forward_return(a[:24], 20*BAR))
        self.assertIsNone(forward_return(np.delete(a, 22), 20*BAR))
        self.assertAlmostEqual(forward_return(a, 20*BAR), (a['c'][24]/a['c'][19]-1)*100)

    def test_label_day_cutoff_purges_cross_boundary(self):
        self.assertEqual(cohort(99, 100, (100, 200)), 'purged')
        self.assertEqual(cohort(99, 99, (100, 200)), 'train')
        self.assertEqual(cohort(199, 201, (100, 200)), 'purged')
        self.assertEqual(cohort(200, 205, (100, 200)), 'test')
        with self.assertRaises(ValueError): cohort(100, 99, (200, 300))
        a, b = split_boundaries(1774994400000, 1790632800000)
        self.assertLess(a, b)

    def test_switching_cost_changes_winner_and_ties_keep(self):
        y = np.array([[.2-.25, 0], [.5-.25, 0], [0, 0]])
        np.testing.assert_array_equal(relevance(y), [[0,1], [1,0], [0,0]])
        np.testing.assert_array_equal(score_actions([[0,0], [2,1]]), [False,True])
        with self.assertRaises(ValueError): score_actions([[np.nan, 0]])

    def test_partial_population_never_authorizes_promotion(self):
        n = 200; days = [f'2026-09-{i//10+1:02}' for i in range(n)]
        scores = np.tile([2., 1.], (n,1)); y = np.tile([1., 0.], (n,1))
        leader = np.ones((n,2), dtype=bool); cap = np.ones((n,2))*.5
        report = evaluate(scores,y,days,leader,cap,np.ones(n,dtype=bool),population_complete=False)
        self.assertEqual(report['verdict'], 'INCONCLUSIVE')
        self.assertEqual(report['replacements'], n)
        self.assertFalse(report['runtime_eligible'])
        self.assertEqual(report['candidate_better_count'], n)

    def test_bad_proxy_is_rejected_even_with_partial_population(self):
        n = 200; days = [str(i//10) for i in range(n)]
        r = evaluate(np.tile([1,0],(n,1)),np.tile([-1,0],(n,1)),days,
                     np.ones((n,2),bool),np.ones((n,2))*.5,np.ones(n,bool),population_complete=False)
        self.assertEqual(r['verdict'], 'REJECTED')
        self.assertEqual(r['decision_correct_count'], 0)


if __name__ == '__main__': unittest.main()
