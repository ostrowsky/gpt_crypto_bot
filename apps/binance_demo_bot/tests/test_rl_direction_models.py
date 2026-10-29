"""RDE02/03/06/07/08: model contracts; real fitting is an explicit research check."""
import copy
import importlib
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def examples(n=3):
    return [[[float(i+j+k) for k in range(5)] for j in range(32)] for i in range(n)]


def provenance(n=3):
    return [dict(split='TRAIN', origin_ms=100+i, feature_end_ms=99+i,
                 label_end_ms=190+i, label_available_at_ms=200+i) for i in range(n)]


class ScalerContracts(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('binance_demo_bot.rl_direction_models')

    def test_RDE02_train_channel_statistics_and_clipping(self):
        scaler = self.api.fit_train_scaler(examples(), provenance=provenance(), fit_cutoff_ms=300)
        self.assertEqual(len(scaler.mean), 5)
        self.assertAlmostEqual(scaler.mean[0], 16.5)
        standardized = scaler.transform(examples())
        values = [v[0] for seq in standardized for v in seq]
        self.assertAlmostEqual(sum(values)/len(values), 0.)
        huge = [[[1e9]*5 for _ in range(32)]]
        self.assertTrue(all(x == 10. for row in scaler.transform(huge)[0] for x in row))

    def test_RDE02_constant_channel_finite(self):
        data = [[[2.]*5 for _ in range(32)] for _ in range(3)]
        scaler = self.api.fit_train_scaler(data, provenance=provenance(), fit_cutoff_ms=300)
        self.assertEqual(scaler.scale, [1.]*5)
        self.assertEqual(scaler.transform(data), [[[0.]*5 for _ in range(32)] for _ in range(3)])

    def test_RDE03_future_label_rejected(self):
        rows = provenance(); rows[-1]['label_available_at_ms'] = 301
        with self.assertRaisesRegex(ValueError, 'TRAIN_ONLY'):
            self.api.fit_train_scaler(examples(), provenance=rows, fit_cutoff_ms=300)

    def test_RDE03_tune_or_future_feature_rejected(self):
        for changes in ({'split':'TUNE'}, {'feature_end_ms':101}):
            rows = provenance(); rows[0].update(changes)
            with self.assertRaisesRegex(ValueError, 'TRAIN_ONLY'):
                self.api.fit_train_scaler(examples(), provenance=rows, fit_cutoff_ms=300)

    def test_RDE03_no_unbound_or_mismatched_provenance(self):
        for rows in ([], provenance(2), None):
            with self.assertRaisesRegex(ValueError, 'TRAIN_ONLY'):
                self.api.fit_train_scaler(examples(), provenance=rows, fit_cutoff_ms=300)

    def test_RDE02_reject_nonfinite_and_wrong_shape(self):
        bad = examples(); bad[0][0][0] = float('nan')
        for data in (bad, examples()[0], [], [[[1.]*4 for _ in range(32)]]):
            with self.assertRaisesRegex(ValueError, 'SEQUENCE'):
                self.api.fit_train_scaler(data, provenance=provenance(len(data)), fit_cutoff_ms=300)

    def test_RDE12_scaler_hash_input_sensitive_and_roundtrip_tamper(self):
        source = examples(); scaler = self.api.fit_train_scaler(source, provenance=provenance(), fit_cutoff_ms=300)
        changed = copy.deepcopy(source); changed[0][0][0] += 1.
        other = self.api.fit_train_scaler(changed, provenance=provenance(), fit_cutoff_ms=300)
        self.assertNotEqual(scaler.fit_sha256, other.fit_sha256)
        restored = self.api.TrainScaler.from_manifest(scaler.manifest())
        self.assertEqual(restored.transform(source), scaler.transform(source))
        manifest = scaler.manifest(); manifest['mean'][0] += 1.
        with self.assertRaisesRegex(ValueError, 'SCALER_HASH'):
            self.api.TrainScaler.from_manifest(manifest)


class PolicyInputContracts(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('binance_demo_bot.rl_direction_models')

    def test_RDE07_policy_rejects_nonregistered_seed_gamma_and_budget(self):
        for args in ({'seed':42,'gamma':1.,'total_timesteps':512},
                     {'seed':11,'gamma':.99,'total_timesteps':512},
                     {'seed':11,'gamma':1.,'total_timesteps':0}):
            with self.assertRaisesRegex(ValueError, 'POLICY_CONTRACT'):
                self.api.fit_policy(None, '.', **args)

    def test_RDE07_model_module_import_has_no_optional_runtime_dependency(self):
        import subprocess
        script = ("import sys; sys.path.insert(0,"+repr(str(Path(__file__).resolve().parents[1]/'src'))+
                  "); import binance_demo_bot.rl_direction_models; "
                  "assert not any(k in sys.modules for k in ('torch','numpy','catboost','gymnasium','stable_baselines3'))")
        subprocess.run([sys.executable,'-c',script],check=True,capture_output=True,text=True)

    def test_RDE06_probability_names_are_not_forecast(self):
        result = self.api.policy_probability_record([.1,.2,.3,.4])
        self.assertEqual(result, {'action_probabilities':{'WAIT_HOLD':.1,'BUY10':.2,'BUY30':.3,'EXIT':.4}})
        self.assertNotIn('p_up', result)
        for bad in ([.5,.5], [.5,.5,.5,.5], [float('nan'),0.,0.,1.]):
            with self.assertRaisesRegex(ValueError, 'ACTION_PROBABILITIES'):
                self.api.policy_probability_record(bad)


def run_real_integration(folder):
    """Executed explicitly with isolated ML Python; no skipped/mocked training."""
    import numpy as np
    from binance_demo_bot import rl_direction_models as api
    rng = np.random.default_rng(42)
    x = rng.normal(size=(32,32,5)).astype(np.float32)
    y = np.column_stack((np.where(x[:,-1,0]>0,.002,-.002),
                         np.where(x[:,-1,1]>0,.004,-.004)))
    rows = provenance(32)
    bundle = api.train_predictors(x,y,Path(folder)/'predictors',provenance=rows,fit_cutoff_ms=500)
    # Assert the prediction resource contract while still calling real CatBoost.
    for model,method in [(m,'predict_proba') for m in bundle.classifiers]+[(m,'predict') for m in bundle.regressors]:
        original=getattr(model,method)
        def checked(*args,_original=original,**kwargs):
            assert kwargs.get('thread_count')==2,'CATBOOST_PREDICTION_THREAD_BUDGET'
            return _original(*args,**kwargs)
        setattr(model,method,checked)
    first = bundle.predict(x[:7]); encoder_sha = bundle.encoder_sha256
    assert first['latent'].shape == (7,16)
    assert set(first) == {'cat_proba','cat_return','gru_proba','gru_return','latent'}
    for name in ('cat_proba','gru_proba'):
        assert first[name].shape == (7,2) and np.all((first[name]>=0)&(first[name]<=1))
    for name in first:
        assert np.isfinite(first[name]).all()
    loaded = api.load_predictors(Path(folder)/'predictors')
    for name, value in loaded.predict(x[:7]).items():
        np.testing.assert_allclose(value,first[name],rtol=1e-6,atol=1e-7)
    bundle.assert_frozen(); assert bundle.encoder_sha256 == encoder_sha

    import torch
    parameter = next(bundle.gru.parameters())
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(1.)
    try:
        bundle.assert_frozen()
    except ValueError as error:
        assert 'ENCODER_NOT_FROZEN' in str(error)
    else:
        raise AssertionError('modified frozen weights accepted')
    with torch.no_grad():
        parameter.copy_(original)
    bundle.assert_frozen()

    import gymnasium as gym
    class Tiny(gym.Env):
        observation_space = gym.spaces.Box(-10.,10.,shape=(27,),dtype=np.float32)
        action_space = gym.spaces.Discrete(4)
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed); self.at = 0
            return np.zeros(27,dtype=np.float32), {}
        def step(self, action):
            self.at += 1
            obs = np.zeros(27,dtype=np.float32); obs[-1] = 1-self.at/8
            return obs, float(action==self.at%4)*.001, self.at==8, False, {}
    policy = api.fit_policy(Tiny(),Path(folder)/'ppo',seed=11,gamma=1.,total_timesteps=512)
    obs = np.zeros(27,dtype=np.float32)
    before = policy.predict(obs)
    after = api.load_policy(Path(folder)/'ppo').predict(obs)
    assert before == after and before in (0,1,2,3)
    assert abs(sum(policy.action_probabilities(obs).values())-1.)<1e-6
    bundle.assert_frozen(); assert bundle.encoder_sha256 == encoder_sha
    # Sealed manifest and member hashes must reject modified artifacts.
    member = Path(folder)/'predictors'/'gru.pt'
    member.write_bytes(member.read_bytes()+b'tamper')
    try:
        api.load_predictors(Path(folder)/'predictors')
    except ValueError as error:
        assert 'ARTIFACT_HASH' in str(error)
    else:
        raise AssertionError('tampered artifact accepted')
    return dict(state='REAL_CPU_TRAIN_SAVE_LOAD_PASS',predictors_N=5,ppo_steps=512,
                encoder_sha256=encoder_sha,source='SYNTHETIC_TEST_ONLY')


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--integration':
        import json
        print(json.dumps(run_real_integration(sys.argv[2]),sort_keys=True))
    else:
        unittest.main()
