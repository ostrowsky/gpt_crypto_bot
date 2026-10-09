"""Preregistered3-calendar-day block evaluation; unknown days stay masked."""
from datetime import date,timedelta
import numpy as np


def interval(a,b):
    by0={r['day']:r for r in a};by1={r['day']:r for r in b}
    if by0.keys()!=by1.keys() or len(by0)!=len(a) or len(by1)!=len(b):raise ValueError('unequal/duplicate days')
    if len(a)<3:return None
    first,last=(date.fromisoformat(d) for d in (min(by0),max(by0)))
    calendar=[(first+timedelta(days=i)).isoformat() for i in range((last-first).days+1)]
    delta=np.array([by1[d]['early']-by0[d]['early'] if d in by0 else np.nan for d in calendar]);n=len(delta)
    rng=np.random.default_rng(42);starts=rng.integers(0,n,size=(5000,int(np.ceil(n/3))))
    idx=((starts[:,:,None]+np.arange(3))%n).reshape(5000,-1)[:,:n];samples=delta[idx];known=np.isfinite(samples).sum(axis=1)
    sums=np.nansum(samples,axis=1);scaled=np.divide(sums,known,out=np.full(len(sums),np.nan),where=known>0)*len(a)
    return dict(known_days=len(a),calendar_days=n,unknown_calendar_days=n-len(a),block_calendar_days=3,draws=5000,known_early_delta=int(np.nansum(delta)),early_count_delta95=np.nanquantile(scaled,[.025,.975]).tolist(),scope='common observed days only; missing days masked, not scored as misses')


def evaluate(base,target,ci,companion_exit_pass=False):
    den0=base['unique_precision_N'];den1=target['unique_precision_N']
    checks=dict(early_count_gain=target['early']>base['early'],early_ci_positive=ci is not None and ci['early_count_delta95'][0]>0,coverage_not_worse=target['captured']>=base['captured'],precision_not_worse=den0>0 and den1>0 and target['unique_precision_n']*den0>=base['unique_precision_n']*den1,companion_exit_pass=companion_exit_pass)
    return dict(state='RETROSPECTIVE_GATE_PASS' if all(checks.values()) else 'RETROSPECTIVE_NOT_ACCEPTED',checks=checks,runtime_eligible=False)
