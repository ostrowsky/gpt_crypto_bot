import unittest
from types import SimpleNamespace
import numpy as np
import leader_mission_metrics as m
from leader_trend_continuation import confirmation,TrendPolicy,paired_accompaniment,diagnostics,verdict,BAR
from replay_backtest import _cooldown_bars_after_exit


def market(n=60):
    c=100+np.arange(n)*.2
    return dict(time=np.arange(1,n+1)*BAR,close=c,open=c-.1,high=c+.1,low=c-.1)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.d=market();self.f=m.indicators(self.d)
        self.t=SimpleNamespace(sym='A',tf='15m',entry_ts=17*BAR,entry_price=float(self.d['close'][16]),exit_ts=0,exit_price=0)
        self.native=np.zeros(60,dtype=[('t','i8'),('c','f8')]);self.native['t']=self.d['time']-BAR;self.native['c']=self.d['close']

    def make(self,progress=lambda *a,**k:'WEAK: x'):
        return TrendPolicy(progress,{'A':self.d},{'A':self.f})

    def test_prefix_invariance_and_future_mutation(self):
        x=confirmation(self.d,self.f,self.t.entry_ts,self.t.entry_price,22*BAR);self.assertTrue(x['confirmed'])
        prefix={k:v[:22] for k,v in self.d.items()};self.assertEqual(x,confirmation(prefix,m.indicators(prefix),self.t.entry_ts,self.t.entry_price,22*BAR))
        for k in ('close','high','low'):self.d[k][22:]=9999
        self.assertEqual(x,confirmation(self.d,m.indicators(self.d),self.t.entry_ts,self.t.entry_price,22*BAR))

    def test_no_final_trade_fields_used(self):
        self.t.capture_ratio_at_entry=999;self.t.max_price_since_entry=999999
        p=self.make();self.assertIsNone(p(self.t,self.native,{},21,ts_ms=22*BAR))
        self.assertTrue(p.decisions[0]['defer'])

    def test_original_hard_exit_never_considered(self):
        p=self.make(lambda *a,**k:'ATR trail stop')
        self.assertEqual(p(self.t,self.native,{},21,ts_ms=22*BAR),'ATR trail stop')
        self.assertEqual(p.decisions,[]);self.assertEqual(p.active,{})

    def test_insufficient_rise_and_stale_1h_price_retain_sell(self):
        self.t.entry_price=200
        p=self.make();self.assertEqual(p(self.t,self.native,{},21,ts_ms=22*BAR),'WEAK: x')
        self.t.entry_price=float(self.d['close'][16]);self.t.tf='1h'
        self.native['t'][23]=20*BAR;self.native['c'][23]=999
        p=self.make();self.assertEqual(p(self.t,self.native,{},23,ts_ms=24*BAR),'WEAK: x')
        self.assertIsNone(p.decisions[0]['features'])

    def test_hard_priority_even_active_and_no_reactivation(self):
        p=self.make(lambda *a,**k:'WEAK: x' if k['ts_ms']==22*BAR else 'ATR trail stop')
        self.assertIsNone(p(self.t,self.native,{},21,ts_ms=22*BAR))
        self.assertEqual(p(self.t,self.native,{},22,ts_ms=23*BAR),'ATR trail stop');self.assertFalse(p.active)
        p.progress=lambda *a,**k:'WEAK: again'
        self.assertEqual(p(self.t,self.native,{},23,ts_ms=24*BAR),'WEAK: again');self.assertEqual(len(p.decisions),1)

    def test_nonextension_and_original_weak_cooldown(self):
        p=self.make();self.assertIsNone(p(self.t,self.native,{},21,ts_ms=22*BAR))
        for at in range(23,26):self.assertIsNone(p(self.t,self.native,{},at-1,ts_ms=at*BAR))
        reason=p(self.t,self.native,{},25,ts_ms=26*BAR);self.assertIn('TIMEOUT',reason)
        self.assertEqual(self.t.exit_ts,26*BAR);self.assertEqual(self.t.exit_price,self.d['close'][25])
        for mode in ('trend','impulse_speed'):self.assertEqual(_cooldown_bars_after_exit(mode,reason),_cooldown_bars_after_exit(mode,'WEAK: x'))

    def test_confirmation_loss_and_1h_current_closed_price(self):
        self.t.tf='1h';self.native['t'][23]=20*BAR
        p=self.make();self.assertIsNone(p(self.t,self.native,{},23,ts_ms=24*BAR))
        self.d['close'][24]=90;self.f=m.indicators(self.d);p.features={'A':self.f}
        reason=p(self.t,self.native,{},23,ts_ms=25*BAR)
        self.assertIn('CONFIRMATION_LOST',reason);self.assertEqual(self.t.exit_price,90);self.assertEqual(self.t.exit_ts,25*BAR)

    def test_stale_origin_missing_grid_and_invalid_atr_retain_sell(self):
        p=self.make();self.assertEqual(p(self.t,self.native,{},21,ts_ms=22*BAR+1),'WEAK: x')
        self.d['time'][18]+=1;self.assertIsNone(confirmation(self.d,self.f,17*BAR,self.t.entry_price,22*BAR))
        self.d=market();self.f=m.indicators(self.d);self.f['atr'][21]=np.nan
        self.assertEqual(self.make()(self.t,self.native,{},21,ts_ms=22*BAR),'WEAK: x')

    def test_zero_effect_and_pair_alignment(self):
        rows=[dict(day=f'2026-09-{d:02}',symbol='A',entry_ts=d,entry_price=100,state='CONFIRMED',terminal_forced=False,marker_ts=d+10,accompaniment_fraction=.5) for d in range(1,31)]
        v=paired_accompaniment(rows,rows);self.assertEqual(v['n'],30);self.assertEqual(v['mean'],0);self.assertEqual(v['interval'],[0,0])
        altered=[dict(r,entry_price=101) for r in rows];self.assertEqual(paired_accompaniment(rows,altered)['n'],0)

    def test_missing_denominator_cannot_pass(self):
        arm=dict(early=1,captured=1,precision_n=0,precision_N=0)
        arms={k:dict(full=arm,test=arm) for k in ('control','trend')}
        result=verdict(arms,dict(accompaniment=dict(mean=0,days=38,interval=[0,0]),exits=dict(retention_delta_mean=0,first_early_delta_n=0)))
        self.assertEqual(result['status'],'MISSION_TRADEOFF_OR_WORSE');self.assertFalse(result['runtime_eligible'])

    def test_actual_skip_episode_association_not_counterfactual_success(self):
        ep=dict(day='2026-09-01',symbol='A',entry_ts=10,marker_ts=30,state='CONFIRMED',terminal_forced=False,
            exit_reason='WEAK: x',first_exit_before_marker=True,new_high_after_first_exit=True)
        r=diagnostics([ep],[dict(symbol='A',at=20),dict(symbol='A',at=30)])
        self.assertEqual(r['cooldown_events_in_confirmed_leader_episodes'],1);self.assertEqual(r['categories']['WEAK']['before'],1)


if __name__=='__main__':unittest.main()
