import unittest
import numpy as np
from joint_direction_amplitude import (BAR,DAY,HORIZON,net_from_gross,cohorts,
    calibrated,mixture,probability_metrics)


class JointTests(unittest.TestCase):
    def test_probability_above_half_does_not_mean_economic_edge(self):
        self.assertLess(mixture(.6,.2,.4,7.5,5),0)
        self.assertGreater(mixture(.7,2,1,7.5,5),0)
        self.assertAlmostEqual(mixture(.7,-2,-1,7.5,5),net_from_gross(0,7.5,5))
        with self.assertRaises(ValueError):mixture(1.1,1,1,7.5,5)

    def test_multiplicative_roundtrip_inversion(self):
        gross=np.array([-1.,0.,1.]);net=net_from_gross(gross,7.5,5)
        k=(1-.00075)*(1-.0005)/((1+.00075)*(1+.0005))
        np.testing.assert_allclose(100*((1+net/100)/k-1),gross,atol=1e-12)
        self.assertLess(net[1],0)

    def test_three_cohorts_disjoint_purged_and_mature(self):
        c=np.arange(0,45*DAY,BAR);valid=np.ones(len(c),bool)
        t,v,k,cuts=cohorts(c,valid,30*DAY,0)
        self.assertFalse(np.any(t&v|t&k|v&k))
        self.assertTrue(np.all(c[t]+HORIZON<cuts[0]));self.assertTrue(np.all(c[v]+HORIZON<cuts[1]))
        self.assertTrue(np.all(c[k]+HORIZON<30*DAY));self.assertTrue(np.all(c[k]>=cuts[1]))

    def test_platt_identity_constant_and_extreme_logits(self):
        p=np.array([.1,.5,.9]);np.testing.assert_allclose(calibrated(p,1,0),p)
        np.testing.assert_allclose(calibrated(p,0,0),.5)
        self.assertTrue(np.isfinite(calibrated([0,1],1,0)).all())

    def test_reliability_denominators_include_edges_and_empty_bins_unknown(self):
        r=probability_metrics([0,.5,1],[False,True,True])
        self.assertEqual(sum(b['n'] for b in r['bins']),3);self.assertEqual(r['up_n'],2)
        self.assertEqual(r['direction_correct_n'],2)
        self.assertEqual(r['bins'][1]['mean_probability'],None)
        self.assertAlmostEqual(r['brier'],.25/3)
        with self.assertRaises(ValueError):probability_metrics([np.nan],[True])


if __name__=='__main__':unittest.main()
