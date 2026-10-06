import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from exit_action_advantage import cohorts,MODEL_PARAMETERS,BAR
from impulse_entry_catboost import DAY
from run_exit_action_advantage import fit_blocks,action_diagnostics


class FitTests(unittest.TestCase):
    def test_variable_action_deadlines_purged(self):
        clock=np.arange(0,40*DAY,BAR);available=clock+np.where(np.arange(len(clock))%2,4*BAR,BAR)
        train,val,cut=cohorts(clock,available,np.ones(len(clock),bool),30*DAY,0)
        self.assertTrue(np.all(available[train]<cut));self.assertTrue(np.all(available[val]<30*DAY))

    def test_native_future_labels_do_not_affect_models_or_inference(self):
        rng=np.random.default_rng(42);clock=np.arange(0,45*DAY,3*BAR)
        x=rng.normal(size=(len(clock),22));y=.3*x[:,0]+rng.normal(size=len(clock))
        def payload(target):return dict(clock=clock,available=clock+BAR,x=x.copy(),y=target.copy())
        changed=y.copy();changed[clock>=30*DAY]=np.nan
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,patch('run_exit_action_advantage.MODEL_PARAMETERS',dict(MODEL_PARAMETERS,iterations=8,depth=2,thread_count=1)):
            za=payload(y);zb=payload(changed);pa,fa=fit_blocks(za,0,45*DAY,Path(a));pb,fb=fit_blocks(zb,0,45*DAY,Path(b))
            np.testing.assert_array_equal(za['prediction'],zb['prediction'])
            self.assertEqual(pa(31*DAY,x[-1]),pb(31*DAY,x[-1]));self.assertEqual(pa(31*DAY,None),(None,0))

    def test_action_selected_losses_counted(self):
        z=dict(fold=np.array([0,0]),clock=np.array([1,2]),y=np.array([1.,-1.]),prediction=np.array([.2,.3]))
        d=action_diagnostics(z,0)['test'];self.assertEqual(d['selected_known_n'],2)
        self.assertEqual(d['selected_hurt_n'],1);self.assertEqual(d['selected_mean_advantage_pp'],0)


if __name__=='__main__':unittest.main()
