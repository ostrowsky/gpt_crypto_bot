import unittest
import numpy as np
import order_flow_execution as m


def book():return np.array([[100-i,2,101+i,2] for i in range(5)]).ravel().astype(float)


class ExecutionTest(unittest.TestCase):
    def test_fees_sides_and_depth(self):
        q=book()
        self.assertAlmostEqual(m.walk(q,3,1),(101*2+102)/3*(1+m.FEE))
        self.assertAlmostEqual(m.walk(q,3,-1),(100*2+99)/3*(1-m.FEE))
        self.assertTrue(np.isnan(m.walk(q,11,1)))
        q[0]=102;self.assertTrue(np.isnan(m.walk(q,1,1)))

    def test_threshold_side_and_missing(self):
        np.testing.assert_array_equal(m.decide([-.0002,-.0001,0,.0002],1,5),[5,0,0,0])
        np.testing.assert_array_equal(m.decide([-.0002,.0002],-1,10),[0,10])
        with self.assertRaises(ValueError):m.decide([np.nan],1,5)

    def test_arrival_gaps_and_exact_clock(self):
        a=dict(time=np.arange(8)*1000,book=np.tile(book(),(8,1)),segment=np.zeros(8,int),age=np.zeros(8))
        self.assertTrue(np.isfinite(m.arrival(a,[0],1,1,6)[0]))
        a['segment'][3]=-1;self.assertTrue(np.isnan(m.arrival(a,[0],1,1,6)[0]))
        self.assertTrue(np.isnan(m.arrival(a,[501],1,1,1)[0]))

    def test_future_does_not_change_selection(self):
        p=np.array([-.0002,.0003]);before=m.decide(p,1,5)
        future=np.array([99.,101.]);future[:]=np.nan
        np.testing.assert_array_equal(before,m.decide(p,1,5))

    def test_unknowns_and_sign_arithmetic(self):
        z=m.metrics(np.array([0,5000]),[100,100],[[101,99],[101,99]],[[100,100],[np.nan,98]],[[True,True],[True,False]])
        self.assertEqual((z['issued'],z['deferred'],z['action_known'],z['action_unknown'],z['matched']), (4,3,3,1,3))
        self.assertAlmostEqual(z['mean_gain_bp'],200/3)
        self.assertEqual(z['benefited'],2)
        z=m.metrics(np.array([0]),[100],[[np.nan,np.nan]],[[np.nan,np.nan]],[[False,False]])
        self.assertIsNone(z['mean_gain_bp']);self.assertIsNone(z['p99_shortfall_bp'])

    def test_order_extraction_full_partial_and_protection(self):
        t=dict(sym='BTCUSDT',entry_ts=1000,exit_ts=3000,entry_price=100,exit_reason='hard ATR',
            partial_exit_taken=True,partial_exit_fraction=.25,partial_exit_ts=2000)
        l=dict(trade_id=0,symbol='BTCUSDT',entry_ts=1000,exit_ts=3000,budget=1000)
        a=m.extract_orders([t],[l]);self.assertEqual(len(a),3)
        self.assertAlmostEqual(a[0]['quantity'],a[1]['quantity']+a[2]['quantity']);self.assertTrue(a[2]['protected'])
        t['exit_reason']='WEAK: trend';a=m.extract_orders([t],[l]);self.assertTrue(a[1]['protected']);self.assertFalse(a[2]['protected'])
        l['symbol']='ETHUSDT'
        with self.assertRaises(ValueError):m.extract_orders([t],[l])

    def test_one_side_denominator_and_interval(self):
        z=m.metrics(np.array([0]),[100],[[99]],[[100]],[[True]],(-1,))
        self.assertEqual(z['issued'],1);self.assertEqual(z['mean_gain_bp'],100)
        self.assertIsNone(m.interval([1,2]));self.assertEqual(m.interval([1,1,1]),[1.,1.])


if __name__=='__main__':unittest.main()
