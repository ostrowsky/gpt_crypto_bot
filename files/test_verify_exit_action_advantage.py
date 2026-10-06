import unittest
import numpy as np
from test_impulse_entry_catboost import data
from test_exit_action_advantage import trade
from verify_exit_action_advantage import state_at,BAR


class StateTests(unittest.TestCase):
    def test_reconstruct_does_not_use_final_trade_peak_or_trail(self):
        d=data();t=trade();t.max_price_since_entry=99999;t.trail_stop=99999
        cache={('A','15m'):(d,{'atr':np.ones(len(d))})}
        def progress(t,d,f,i,**kw):
            t.trail_stop=max(t.trail_stop,float(d['c'][i])-2)
            return 'WEAK: x' if kw['ts_ms']==20*BAR else None
        state=state_at(t,cache,20*BAR,progress)
        self.assertLess(state.trail_stop,200);self.assertLess(state.max_price_since_entry,200)
        self.assertEqual(state.exit_ts,20*BAR)

    def test_earlier_sell_invalidates_recorded_first_soft_state(self):
        d=data();t=trade();cache={('A','15m'):(d,{'atr':np.ones(len(d))})}
        with self.assertRaisesRegex(ValueError,'earlier'):
            state_at(t,cache,20*BAR,lambda *a,**k:'ATR trail stop')


if __name__=='__main__':unittest.main()
