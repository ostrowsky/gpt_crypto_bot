import unittest
from types import SimpleNamespace
import numpy as np
from exit_action_advantage import BAR,ExitPolicy,position_features,cash_advantage,label_action
from replay_backtest import ReplayTrade,_cooldown_bars_after_exit
from test_impulse_entry_catboost import data


def trade():
    return ReplayTrade('A','15m','trend',17*BAR,101.6,16,2.,48,95.,
                       exit_ts=20*BAR,exit_price=101.9,exit_reason='WEAK: example')


class ActionTests(unittest.TestCase):
    def test_observed_mfe_prefix_and_future_mutation(self):
        d=data();t=trade();x=position_features(t,d,20*BAR,101.9)
        np.testing.assert_array_equal(x,position_features(t,d[:20],20*BAR,101.9))
        d['h'][20:]=99999
        np.testing.assert_array_equal(x,position_features(t,d,20*BAR,101.9))
        self.assertEqual(len(x),22)
        expected=100*(max(101.6,float(d['h'][17:20].max()))/101.6-1)
        self.assertAlmostEqual(x[11],expected)

    def test_action_uses_one_sell_cost_not_roundtrip(self):
        self.assertAlmostEqual(cash_advantage(100,101,7.5,5),1*.99925*.9995)
        self.assertLess(cash_advantage(100,99,7.5,5),0)

    def test_deadline_label_and_missing_future(self):
        d=data();t=trade();cache={('A','15m'):(d,{})}
        progress=lambda *args,**kw:'WEAK: still'
        y=label_action(t,cache,25*BAR,7.5,5,progress)
        self.assertEqual(y['available_at'],21*BAR);self.assertEqual(y['hold_exit_at'],21*BAR)
        self.assertAlmostEqual(y['hold_exit_price'],d['c'][20])
        self.assertIsNone(label_action(t,cache,20*BAR,7.5,5,progress))

    def test_hard_exit_never_suppressed_and_timeout_no_recursive_extension(self):
        d=data();cache={('A','15m'):(d,{})};t=trade()
        p=ExitPolicy(lambda *a,**k:'WEAK: x',cache,lambda at,x:(.1,0))
        self.assertIsNone(p(t,d,{},19,ts_ms=20*BAR));self.assertEqual(len(p.decisions),1)
        timeout=p(t,d,{},20,ts_ms=21*BAR)
        self.assertEqual(timeout,'WEAK: x [action one-bar deadline]')
        for mode in ('trend','impulse_speed','retest'):
            self.assertEqual(_cooldown_bars_after_exit(mode,timeout),_cooldown_bars_after_exit(mode,'WEAK: x'))
        self.assertEqual(p(t,d,{},21,ts_ms=22*BAR),'WEAK: x');self.assertEqual(len(p.decisions),1)
        t=trade();p=ExitPolicy(lambda *a,**k:'ATR trail stop',cache,lambda at,x:(100.,0))
        self.assertEqual(p(t,d,{},19,ts_ms=20*BAR),'ATR trail stop');self.assertEqual(p.decisions,[])

    def test_active_deferral_respects_new_hard_exit(self):
        d=data();t=trade();cache={('A','15m'):(d,{})}
        def progression(*a,**k):return 'WEAK: x' if k['ts_ms']==20*BAR else 'ATR trail stop'
        p=ExitPolicy(progression,cache,lambda at,x:(.1,0))
        self.assertIsNone(p(t,d,{},19,ts_ms=20*BAR))
        self.assertEqual(p(t,d,{},20,ts_ms=21*BAR),'ATR trail stop')
        self.assertEqual(p.active,{})

    def test_strict_threshold_and_stale_origin_retains_sell(self):
        d=data();t=trade();cache={('A','15m'):(d,{})}
        p=ExitPolicy(lambda *a,**k:'WEAK: x',cache,lambda at,x:(.05,0))
        self.assertEqual(p(t,d,{},19,ts_ms=20*BAR),'WEAK: x')
        p=ExitPolicy(lambda *a,**k:'WEAK: x',cache,lambda at,x:(None,None))
        self.assertEqual(p(t,d,{},19,ts_ms=20*BAR+1),'WEAK: x')
        self.assertIsNone(p.decisions[0]['features'])


if __name__=='__main__':unittest.main()
