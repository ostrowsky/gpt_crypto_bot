import unittest
from unittest.mock import patch
import audit_btc_leader_entry_exit as a


def history(n=100):
    return {s:{i*a.BAR:(100+i,102+i,99+i,101+i) for i in range(n)}
            for s in ('BTCUSDT',*a.TARGETS)}


class LeaderTests(unittest.TestCase):
    def test_zero(self):
        self.assertIsNone(a.summarize([])['mean_pct'])
        self.assertIsNone(a.basket(history(),[],'fixed24')['basket_compounded_pct'])

    def test_conflicting_duplicate(self):
        r=[0,100,101,99,100,0,a.BAR-1]
        self.assertEqual(len(a.merge_rows([[r],[r]])),1)
        with self.assertRaises(ValueError):
            a.merge_rows([[r],[[0,100,101,99,101,0,a.BAR-1]]])

    def test_open_candle(self):
        with self.assertRaises(ValueError):
            a.merge_rows([[[0,100,101,99,100,0,0]]])

    def test_next_open(self):
        d=history(); t=10*a.BAR
        r=a.trade(d,'SOLUSDT',t,'fixed24')
        self.assertEqual(r['entry_ms'],11*a.BAR)
        self.assertEqual(r['entry'],111)
        self.assertEqual(r['exit_ms'],35*a.BAR)

    def test_exit_next_open(self):
        d=history(); d['BTCUSDT'][11*a.BAR]=(111,112,90,90)
        r=a.trade(d,'SOLUSDT',10*a.BAR,'btc_nonpositive')
        self.assertEqual(r['exit_ms'],12*a.BAR)

    def test_gap(self):
        d=history(); del d['ETHUSDT'][20*a.BAR]
        self.assertFalse(a.valid_window(d,10*a.BAR))

    def test_nonoverlap_and_purge(self):
        stamps=a.events(history(),0,70*a.BAR,.25)
        self.assertTrue(all(y-x>=25*a.BAR for x,y in zip(stamps,stamps[1:])))
        self.assertTrue(all(t+25*a.BAR<70*a.BAR for t in stamps))

    def test_costs(self):
        self.assertLess(a.net_return(100,100),-.24)

    def test_selection_uses_train_bounds(self):
        bounds=[]
        def event(d,start,end,q):
            bounds.append(end); return []
        with patch.object(a,'events',event):
            self.assertIsNone(a.choose_threshold(history(),0,50*a.BAR))
        self.assertEqual(bounds,[50*a.BAR]*3)

    def test_amplification_denominator(self):
        d=history(); r=a.forward_stats(d,[10*a.BAR])['SOLUSDT']['1']
        self.assertEqual(r['btc_positive_denominator'],1)
        self.assertEqual(r['target_at_least_2x_btc'],0)


if __name__=='__main__':
    unittest.main()
