import unittest
import numpy as np
import leader_mission_metrics as m


def trade(clock,price=100,exit_clock=None,exit_price=104):
    return dict(sym='S00',tf='1h',entry_ts=clock,entry_price=price,exit_ts=exit_clock or clock+6*m.BAR,
        exit_price=exit_price,exit_reason='WEAK: fixture',partial_exit_taken=False,
        partial_exit_ts=0,partial_exit_fraction=0,partial_exit_price=0,capture_ratio_at_entry=0.)


def episode_fixture():
    n=130;t=np.arange(n)*m.BAR;c=np.full(n,100.)
    c[21:28]=[101,102,103,104,105,103,102];c[28:]=101
    d=dict(time=t,close=c,open=c.copy(),high=c+.5,low=c-.5)
    ema=np.full(n,100.);ema[21:26]=[101,102,103,104,105];ema[26:]=np.arange(n-26)*(-.01)+104
    f=dict(ema=ema,atr=np.ones(n));tr=trade(int(t[20]),exit_clock=int(t[26]))
    r=dict(day='2026-04-01',symbol='S00',entry_ts=int(t[20]),entry_price=100.,trade=tr)
    return r,d,f


class MissionTests(unittest.TestCase):
    def test_cutoff_close_and_same_decision_day(self):
        day='2026-04-01';start,cut=m.day_bounds(day);clock=np.arange(start+m.BAR,cut+m.BAR+1,m.BAR)
        c=np.full(len(clock),110.);c[-1]=200
        d=dict(time=clock,open=np.full(len(clock),100.),close=c)
        market={f'S{i:02}':d for i in range(15)};labels,_=m.daily_labels(market,start,cut)
        self.assertEqual(labels[day]['values']['S00']['close'],110)
        tr=trade(start,price=102);s,_,_=m.mission([tr],labels)
        self.assertEqual((s['early'],s['captured'],s['precision_N']),(1,1,1))
        self.assertEqual(s['corrected_early_disagreements'],1)
        tr['entry_ts']=cut+1;self.assertEqual(m.mission([tr],labels)[0]['captured'],0)

    def test_dst_and_missing_label_grid(self):
        a,b=m.day_bounds('2026-03-29');self.assertEqual((b-a)//3600000,21)
        clock=np.arange(a+m.BAR,b+1,m.BAR);d=dict(time=clock,open=np.ones(len(clock)),close=np.ones(len(clock))*2)
        market={f'S{i:02}':dict(d) for i in range(15)}
        self.assertEqual(len(m.daily_labels(market,a,b)[1]),1)
        market['S00']['time']=clock[:-1];market['S00']['open']=d['open'][:-1];market['S00']['close']=d['close'][:-1]
        self.assertEqual(m.daily_labels(market,a,b)[1],[])

    def test_confirmed_marker_and_partial_retention(self):
        r,d,f=episode_fixture();ep=m.episode(r,d,f)
        self.assertEqual(ep['marker_ts'],27*m.BAR);self.assertTrue(ep['exit_before_marker'])
        self.assertAlmostEqual(ep['retention'],.8)
        r['trade'].update(partial_exit_taken=True,partial_exit_ts=25*m.BAR,partial_exit_fraction=.5,partial_exit_price=103,exit_ts=28*m.BAR)
        ep=m.episode(r,d,f);self.assertAlmostEqual(ep['retention'],.7);self.assertEqual(ep['remaining_fraction_at_marker'],.5)

    def test_tail_and_no_trend_not_zero_success(self):
        r,d,f=episode_fixture();d['time']=d['time'][:100];d['close']=d['close'][:100]
        self.assertEqual(m.episode(r,d,f)['state'],'UNKNOWN_FOLLOWUP')
        r,d,f=episode_fixture();d['close'][:]=100;ep=m.episode(r,d,f)
        self.assertEqual(ep['state'],'NO_ESTABLISHED_UPSWING');self.assertNotIn('retention',ep)
        self.assertIsNone(m.exit_summary([ep])['retention_mean'])

    def test_future_prefix_marker_invariance(self):
        r,d,f=episode_fixture();before=m.episode(r,d,f)['marker_ts'];d['close'][50:]=10000
        self.assertEqual(m.episode(r,d,f)['marker_ts'],before)
        data=dict(close=np.arange(100.)+100,high=np.arange(100.)+101,low=np.arange(100.)+99)
        full=m.indicators(data);short=m.indicators({k:v[:50] for k,v in data.items()})
        np.testing.assert_allclose(full['ema'][:50],short['ema']);np.testing.assert_allclose(full['atr'][:50],short['atr'],equal_nan=True)

    def test_threshold_roundoff_does_not_delay_marker(self):
        r,d,f=episode_fixture();f['atr'][26]=1+1e-13;f['atr'][27]=1+1e-13
        self.assertEqual(m.episode(r,d,f)['marker_ts'],27*m.BAR)
        d['close'][27]=100.;f['ema'][27]=f['ema'][26]+1e-13
        self.assertNotEqual(m.episode(r,d,f)['marker_ts'],27*m.BAR)

    def test_common_pair_different_entry_excluded(self):
        r,d,f=episode_fixture();a=m.episode(r,d,f);b=dict(a,retention=a['retention']+.1)
        out=m.compare_exits([a],[b]);self.assertEqual(out['confirmed_same_entry_n'],1);self.assertAlmostEqual(out['retention_delta_mean'],.1)
        b['entry_ts']+=m.BAR;out=m.compare_exits([a],[b]);self.assertEqual(out['different_entry_pairs'],1);self.assertIsNone(out['retention_delta_mean'])

    def test_verdict_has_no_cost_inputs(self):
        s=dict(early=10,captured=15,precision_n=20,precision_N=30)
        ci=dict(days=38,intervals={'early_pp':[0,0],'capture_pp':[0,0],'precision_pp':[0,0]})
        ex=dict(retention_delta_mean=0,first_early_delta_n=0,days=38,retention_interval=[0,0])
        self.assertEqual(m.verdict(s,s,ci,ex),'NO_MISSION_GAIN')
        self.assertEqual(m.verdict(s,dict(s,early=9),ci,ex),'MISSION_TRADEOFF_OR_WORSE')

    def test_bootstrap_identity_and_improvement(self):
        days=[dict(day=f'2026-04-{i:02}',early=5,captured=10,leaders=15,precision_n=10,precision_N=20) for i in range(1,10)]
        out=m.paired_intervals(days,days)
        self.assertEqual(out['intervals']['early_pp'],[0.,0.])
        better=[dict(r,early=6) for r in days]
        self.assertGreater(m.paired_intervals(days,better)['intervals']['early_pp'][0],0)

    def test_reentries_and_partial_accompaniment(self):
        a=trade(0,exit_clock=2*m.BAR);b=trade(3*m.BAR,exit_clock=6*m.BAR)
        value=m.accompaniment([a,b],0,5*m.BAR)
        self.assertAlmostEqual(value['accompaniment_fraction'],.8);self.assertTrue(value['any_remaining_at_marker'])
        b.update(partial_exit_taken=True,partial_exit_ts=4*m.BAR,partial_exit_fraction=.5)
        self.assertAlmostEqual(m.accompaniment([a,b],0,5*m.BAR)['accompaniment_fraction'],.7)
        b['entry_ts']=m.BAR
        with self.assertRaises(ValueError):m.accompaniment([a,b],0,5*m.BAR)


if __name__=='__main__':unittest.main()
