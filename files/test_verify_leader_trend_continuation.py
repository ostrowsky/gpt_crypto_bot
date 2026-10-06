import unittest
from verify_leader_trend_continuation import scalar_confirmation,block_interval,terminal_recheck
from types import SimpleNamespace
import numpy as np
from test_leader_trend_continuation import market
from leader_trend_continuation import confirmation,BAR
import leader_mission_metrics as m


class VerifyTests(unittest.TestCase):
    def test_scalar_prefix_matches_native_and_future_mutation(self):
        d=market();f=m.indicators(d);rows=[dict(c=c) for c in d['close']];times=d['time'].tolist()
        x=scalar_confirmation(rows,times,f['ema'].tolist(),f['atr'].tolist(),17*BAR,d['close'][16],22*BAR)
        self.assertEqual(x,confirmation(d,f,17*BAR,d['close'][16],22*BAR))
        rows[22]['c']=999999
        self.assertEqual(x,scalar_confirmation(rows,times,f['ema'].tolist(),f['atr'].tolist(),17*BAR,d['close'][16],22*BAR))

    def test_identical_pair_bootstrap_zero_and_short_unknown(self):
        rows=[dict(day=str(i),delta=0) for i in range(38)]
        self.assertEqual(block_interval(rows,'delta'),[0,0]);self.assertIsNone(block_interval(rows[:2],'delta'))

    def test_boundary_exception_cannot_allow_live_backdated_or_reactivated_ticks(self):
        data=np.zeros(3,dtype=[('t','i8')]);data['t']=[0,BAR,2*BAR]
        trade=SimpleNamespace(exit_ts=3*BAR,exit_reason='open_at_end',tf='15m')
        tick=dict(at=2*BAR);state=dict(closed=False,last=3*BAR)
        self.assertTrue(terminal_recheck(tick,state,trade,data,3*BAR))
        trade.exit_reason='WEAK: x';self.assertFalse(terminal_recheck(tick,state,trade,data,3*BAR))
        trade.exit_reason='open_at_end';state['last']=2*BAR;self.assertFalse(terminal_recheck(tick,state,trade,data,3*BAR))
        state.update(last=3*BAR,closed=True);self.assertFalse(terminal_recheck(tick,state,trade,data,3*BAR))


if __name__=='__main__':unittest.main()
