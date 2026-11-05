"""MR01/02/11/12 full-population replay orchestration contracts, first RED."""
import copy
import unittest
import _support
from binance_demo_bot.minute_replay_runner import minute_plan, reuse_source, complete_gate, verify_mlp

class MinuteRunnerTests(unittest.TestCase):
    def setUp(self):
        self.parent=dict(seeds=[11,23,37,53,71],splits=[['train',0,900000],['tune',900000,1800000],['test',1800000,2700000]])
    def test_MR02_exact_168_native_and_84_separate_golden(self):
        plan=minute_plan(self.parent,dict(CatBoost=.002,GRU=.002))
        self.assertEqual(len(plan),168);self.assertEqual(len({t['id'] for t in plan}),168)
        self.assertEqual(sum(t['order']=='MINUTE_STOP_FIRST' for t in plan),84)
        self.assertEqual(sum(reuse_source(t) is not None for t in plan),72)
        self.assertTrue(all(reuse_source(t) is None for t in plan if t['label'] in ('CatBoost','GRU','EXIT_fixed')))
        self.assertFalse(any(t['segment']=='train' for t in plan))

    def test_MR12_partial_or_failed_evidence_cannot_complete(self):
        good=[dict(state='COMPLETE',financial_check=dict(state='PASS'))]*168
        for count in (0,83):
            with self.assertRaises(ValueError):complete_gate(count,good,dict(state='VERIFIED_RECORDED_NATIVE_MINUTE_DATA'))
        with self.assertRaises(ValueError):complete_gate(84,good[:-1],dict(state='VERIFIED_RECORDED_NATIVE_MINUTE_DATA'))
        with self.assertRaises(ValueError):complete_gate(84,good,dict(state='FAIL'))
        self.assertEqual(complete_gate(84,good,dict(state='VERIFIED_RECORDED_NATIVE_MINUTE_DATA')),'COMPLETE_CONDITIONAL_SIMULATION')
        bad=copy.deepcopy(good);bad[0]['financial_check']['state']='FAIL'
        with self.assertRaises(ValueError):complete_gate(84,bad,dict(state='VERIFIED_RECORDED_NATIVE_MINUTE_DATA'))

    def test_MR11_memo_rejects_nonstateless_policy(self):
        class Handle:pass
        class Model:pass
        class MlpPolicy:pass
        h=Handle();h.model=Model();h.model.policy=MlpPolicy();verify_mlp(h)
        class RecurrentActorCriticPolicy:pass
        h.model.policy=RecurrentActorCriticPolicy()
        with self.assertRaises(ValueError):verify_mlp(h)


if __name__=='__main__':unittest.main()
