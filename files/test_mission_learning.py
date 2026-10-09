import unittest
import numpy as np
from mission_learning import entry_features,continuation_target,fit_cohorts,mission_metrics,release_readiness,BAR
from mission_contract import window,daily_label

def candles(n=50):
    a=np.zeros(n,dtype=[(k,'i8' if k=='t' else 'f8') for k in ('t','o','h','l','c','v')]);a['t']=np.arange(n)*BAR
    a['o']=a['c']=100+np.arange(n)*.1;a['h']=a['c']+1;a['l']=a['c']-1;a['v']=1;return a

class LearningTests(unittest.TestCase):
    def test_future_prefix_invariance_and_target_maturity(self):
        d=candles();at=20*BAR;x=entry_features(d,at,'15m',40,'trend');other=d.copy();other['c'][20:]*=10
        np.testing.assert_array_equal(x,entry_features(other,at,'15m',40,'trend'))
        y=continuation_target(d,at,4*BAR);self.assertEqual(y['available_at'],at+4*BAR)
        self.assertIsNone(continuation_target(d[:22],at,4*BAR))

    def test_purge_label_boundary_and_future_rows(self):
        start=window('2026-04-01')[0];fit=window('2026-05-01')[0];cut=window('2026-04-25')[0]
        c=np.array([start+BAR,cut-BAR,cut+BAR,fit]);avail=np.array([start+2*BAR,cut,fit,fit+BAR])
        t,v,_=fit_cohorts(c,avail,np.ones(4,bool),fit,start)
        self.assertEqual(t.tolist(),[True,False,False,False]);self.assertFalse(v.any())

    def test_unique_precision_not_trade_events(self):
        lo,hi=window('2026-10-08');cov=dict(missing=[],historical_PIT_certified=False)
        label=daily_label('2026-10-08',{'AUSDT':[lo,100,110,99,110,0,hi-1,2000000],'BUSDT':[lo,100,110,99,110,0,hi-1,1]},{'AUSDT'},hi,cov)
        trades=[dict(sym=s,entry_ts=lo+i*BAR,entry_price=101) for i,s in enumerate(['AUSDT','AUSDT','BUSDT'])]
        result,_,_=mission_metrics(trades,{'2026-10-08':label})
        self.assertEqual((result['early'],result['captured'],result['leader_pairs']), (1,1,1));self.assertEqual(result['unique_precision_N'],2)

    def test_proxy_and_history_cannot_unlock_canary(self):
        r=release_readiness(dict(retrospective_mission_pass=True,teacher_pass=True,fresh_completed_days=999,fresh_leader_pairs=999))
        self.assertEqual(r['state'],'SHADOW_ONLY');self.assertFalse(r['runtime_eligible']);self.assertIn('acceptance_clock_verified',r['blockers'])

if __name__=='__main__':unittest.main()
