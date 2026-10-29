"""RDE-1 isolated supervised/frozen-encoder/PPO models, optional imports delayed.

This module has no adapter, credential or order dependency. Artifacts produced
here are SIMULATION research, never an authorized deployed trading model.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

SEEDS = (11, 23, 37, 53, 71)
CHANNELS = 5
LENGTH = 32
OBSERVATION_SIZE = 27
CAT_PARAMS = dict(iterations=150, depth=5, learning_rate=.05, l2_leaf_reg=10,
                  random_seed=42, thread_count=2, bootstrap_type='No',
                  has_time=True, allow_writing_files=False, verbose=False)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'),
                               allow_nan=False)+'\n', encoding='utf-8')


def _validate_sequences(data):
    try:
        if not len(data):
            raise ValueError('SEQUENCE_EMPTY')
        if hasattr(data, 'shape'):
            import numpy as np
            if len(data.shape) != 3 or tuple(data.shape[1:]) != (LENGTH, CHANNELS):
                raise ValueError('SEQUENCE_SHAPE')
            for start in range(0, len(data), 256):
                if not np.isfinite(data[start:start+256]).all():
                    raise ValueError('SEQUENCE_NONFINITE')
        else:
            for sequence in data:
                if len(sequence) != LENGTH:
                    raise ValueError('SEQUENCE_SHAPE')
                for row in sequence:
                    if len(row) != CHANNELS or any(not math.isfinite(float(v)) for v in row):
                        raise ValueError('SEQUENCE_NONFINITE_OR_SHAPE')
    except (TypeError, OverflowError, IndexError) as error:
        raise ValueError('SEQUENCE_SHAPE') from error


class TrainScaler:
    def __init__(self, mean, scale, fit_sha256, provenance_sha256, fit_cutoff_ms, samples_n):
        self.mean = list(mean)
        self.scale = list(scale)
        self.fit_sha256 = fit_sha256
        self.provenance_sha256 = provenance_sha256
        self.fit_cutoff_ms = fit_cutoff_ms
        self.samples_n = samples_n

    def manifest(self):
        value = dict(mean=self.mean[:], scale=self.scale[:], fit_sha256=self.fit_sha256,
                     provenance_sha256=self.provenance_sha256, fit_cutoff_ms=self.fit_cutoff_ms,
                     samples_n=self.samples_n, sequence_length=LENGTH, channels=CHANNELS)
        return dict(value, scaler_sha256=_digest(value))

    @classmethod
    def from_manifest(cls, value):
        value = dict(value)
        expected = value.pop('scaler_sha256', None)
        if expected != _digest(value):
            raise ValueError('SCALER_HASH')
        if (value.get('sequence_length') != LENGTH or value.get('channels') != CHANNELS
                or len(value.get('mean', ())) != CHANNELS or len(value.get('scale', ())) != CHANNELS
                or any(not math.isfinite(float(v)) for v in value['mean'])
                or any(not math.isfinite(float(v)) or float(v) <= 0 for v in value['scale'])):
            raise ValueError('SCALER_HASH')
        return cls(*(value[k] for k in ('mean', 'scale', 'fit_sha256', 'provenance_sha256',
                                       'fit_cutoff_ms', 'samples_n')))

    def transform(self, data):
        _validate_sequences(data)
        if hasattr(data, 'shape'):
            import numpy as np
            result = np.array(data, dtype=np.float32, copy=True)
            result -= np.asarray(self.mean, dtype=np.float32)
            result /= np.asarray(self.scale, dtype=np.float32)
            np.clip(result, -10., 10., out=result)
            return result
        return [[[max(-10., min(10., (float(v)-self.mean[k])/self.scale[k]))
                  for k, v in enumerate(row)] for row in seq] for seq in data]


def fit_train_scaler(data, *, provenance, fit_cutoff_ms):
    """Bind normalization to mature TRAIN features only, before any transform."""
    _validate_sequences(data)
    if (type(fit_cutoff_ms) is not int or provenance is None or len(provenance) != len(data)):
        raise ValueError('TRAIN_ONLY_PROVENANCE')
    for row in provenance:
        try:
            valid = (row['split'] == 'TRAIN' and row['feature_end_ms'] < row['origin_ms']
                     and row['origin_ms'] <= row['label_end_ms'] <= row['label_available_at_ms']
                     <= fit_cutoff_ms)
        except (KeyError, TypeError):
            valid = False
        if not valid:
            raise ValueError('TRAIN_ONLY_PROVENANCE')
    count = 0
    means = [0.]*CHANNELS
    squared = [0.]*CHANNELS
    source_hash = hashlib.sha256()
    # Streaming Welford avoids extra full-history copies and is stable for
    # constant channels. Numerical fitting cost is bounded by sampled TRAIN.
    for sequence in data:
        for row in sequence:
            values = [float(v) for v in row]
            source_hash.update(struct.pack('<5d', *values))
            count += 1
            for k, value in enumerate(values):
                difference = value-means[k]
                means[k] += difference/count
                squared[k] += difference*(value-means[k])
    scales = [math.sqrt(max(0., v/count)) if v > 1e-24 else 1. for v in squared]
    return TrainScaler(means, scales, source_hash.hexdigest(), _digest(provenance),
                       fit_cutoff_ms, len(data))


def _torch_setup(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    # set_num_interop_threads may be called only before parallel work begins.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def environment_manifest():
    """Exact measured ML versions/origins, called only in isolated research."""
    import importlib
    import importlib.metadata
    import platform
    modules = {name:importlib.import_module(name) for name in
               ('numpy','catboost','torch','stable_baselines3','gymnasium')}
    versions = {name:str(module.__version__) for name,module in modules.items()}
    expected = dict(numpy='2.4.4',catboost='1.2.10',torch='2.9.1+cpu',
                    stable_baselines3='2.7.1',gymnasium='1.2.3')
    if versions != expected or platform.python_version() != '3.11.9':
        raise ValueError('UNPINNED_RESEARCH_ENVIRONMENT')
    roots = {name:str(Path(module.__file__).resolve()) for name,module in modules.items()}
    own_root = Path(sys.executable).resolve().parent
    if any(not Path(path).is_relative_to(own_root) for path in roots.values()):
        raise ValueError('FOREIGN_RESEARCH_MODULE')
    return dict(python=platform.python_version(),executable=str(Path(sys.executable).resolve()),
                versions=versions,module_origins=roots,
                distributions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()})


def _gru_model():
    import torch
    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.GRU(CHANNELS, 16, batch_first=True)
            self.direction = torch.nn.Linear(16, 2)
            self.returns = torch.nn.Linear(16, 2)

        def forward(self, values):
            _, state = self.encoder(values)
            latent = state[-1]
            return self.direction(latent), self.returns(latent), latent
    return Predictor()


def _state_sha(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PredictorBundle:
    def __init__(self, scaler, classifiers, regressors, gru, return_scale, manifest):
        self.scaler = scaler
        self.classifiers = classifiers
        self.regressors = regressors
        self.gru = gru.eval()
        for parameter in self.gru.parameters():
            parameter.requires_grad_(False)
        self.return_scale = return_scale
        self.artifact_manifest = manifest
        self.encoder_sha256 = manifest['encoder_sha256']

    def assert_frozen(self):
        if (any(p.requires_grad for p in self.gru.parameters()) or self.gru.training
                or _state_sha(self.gru) != self.encoder_sha256
                or self.scaler.manifest()['scaler_sha256'] != self.artifact_manifest['scaler']['scaler_sha256']):
            raise ValueError('ENCODER_NOT_FROZEN')
        return True

    def predict(self, data, batch_size=256):
        import numpy as np
        import torch
        self.assert_frozen()
        _validate_sequences(data)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('BATCH_SIZE')
        result = {name:np.empty((len(data), width), dtype=np.float32) for name, width in
                  (('cat_proba', 2), ('cat_return', 2), ('gru_proba', 2),
                   ('gru_return', 2), ('latent', 16))}
        with torch.inference_mode():
            for start in range(0, len(data), batch_size):
                stop = min(len(data), start+batch_size)
                values = self.scaler.transform(np.asarray(data[start:stop], dtype=np.float32))
                flattened = values.reshape(len(values), -1)
                for horizon in range(2):
                    result['cat_proba'][start:stop, horizon] = self.classifiers[horizon].predict_proba(flattened,thread_count=2)[:, 1]
                    result['cat_return'][start:stop, horizon] = self.regressors[horizon].predict(flattened,thread_count=2)
                logits, returns, latent = self.gru(torch.from_numpy(values))
                result['gru_proba'][start:stop] = torch.sigmoid(logits).numpy()
                result['gru_return'][start:stop] = returns.numpy()*self.return_scale
                result['latent'][start:stop] = latent.numpy()
        return result


def train_predictors(data, targets, model_folder, *, provenance, fit_cutoff_ms):
    """Fixed RDE08 training: four CatBoost heads and one frozen GRU bundle."""
    import numpy as np
    from catboost import CatBoostClassifier, CatBoostRegressor
    torch = _torch_setup(42)
    environment = environment_manifest()
    scaler = fit_train_scaler(data, provenance=provenance, fit_cutoff_ms=fit_cutoff_ms)
    targets = np.asarray(targets, dtype=np.float32)
    if targets.shape != (len(data), 2) or not np.isfinite(targets).all():
        raise ValueError('TARGET_SHAPE_OR_NONFINITE')
    labels = targets > 0
    if any(len(np.unique(labels[:, k])) != 2 for k in range(2)):
        raise ValueError('TARGET_SINGLE_CLASS')
    folder = Path(model_folder)
    folder.mkdir(parents=True, exist_ok=False)
    values = scaler.transform(np.asarray(data, dtype=np.float32))
    flattened = values.reshape(len(values), -1)
    classifiers, regressors, members = [], [], []
    for index, horizon in enumerate((15, 60)):
        classifier = CatBoostClassifier(**CAT_PARAMS, loss_function='Logloss')
        regressor = CatBoostRegressor(**CAT_PARAMS, loss_function='RMSE')
        classifier.fit(flattened, labels[:, index].astype(np.int8))
        regressor.fit(flattened, targets[:, index])
        for model, kind in ((classifier, 'classifier'), (regressor, 'regressor')):
            name = f'cat_{kind}_{horizon}.cbm'
            model.save_model(str(folder/name)); members.append(name)
        classifiers.append(classifier); regressors.append(regressor)
    return_scale = np.maximum(targets.std(axis=0, dtype=np.float64), 1e-8).astype(np.float32)
    model = _gru_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    x_tensor = torch.from_numpy(values)
    y_direction = torch.from_numpy(labels.astype(np.float32))
    y_returns = torch.from_numpy(targets/return_scale)
    epoch_losses = []
    model.train()
    for _ in range(8):
        epoch_loss = 0.
        for start in range(0, len(values), 256):
            optimizer.zero_grad(set_to_none=True)
            logits, returns, _ = model(x_tensor[start:start+256])
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, y_direction[start:start+256])
                    + torch.nn.functional.huber_loss(returns, y_returns[start:start+256]))
            if not torch.isfinite(loss):
                raise ValueError('TRAIN_NONFINITE_LOSS')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            epoch_loss += float(loss.detach())*len(x_tensor[start:start+256])
        epoch_losses.append(epoch_loss/len(values))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.save(model.state_dict(), folder/'gru.pt'); members.append('gru.pt')
    manifest = dict(schema='RDE1_MODELS_V1', source='SIMULATION', promotion_allowed=False,
                    horizons_minutes=[15,60], scaler=scaler.manifest(),
                    return_scale=return_scale.tolist(), encoder_sha256=_state_sha(model),
                    cat_params=CAT_PARAMS, gru=dict(hidden=16,epochs=8,batch=256,seed=42,
                                                   losses=epoch_losses),
                    environment=environment,source_sha256=_file_sha(__file__),
                    targets_sha256=hashlib.sha256(targets.tobytes()).hexdigest(),
                    files={name:_file_sha(folder/name) for name in members})
    manifest['manifest_sha256'] = _digest(manifest)
    _write(folder/'manifest.json', manifest)
    return PredictorBundle(scaler,classifiers,regressors,model,return_scale,manifest)


def _read_manifest(folder, name='manifest.json'):
    folder = Path(folder)
    manifest = json.loads((folder/name).read_text(encoding='utf-8'))
    claimed = manifest.get('manifest_sha256')
    if claimed != _digest({k:v for k,v in manifest.items() if k != 'manifest_sha256'}):
        raise ValueError('ARTIFACT_HASH_MANIFEST')
    for member, digest in manifest['files'].items():
        if Path(member).name != member or not (folder/member).is_file() or _file_sha(folder/member) != digest:
            raise ValueError('ARTIFACT_HASH_MEMBER')
    return manifest


def load_predictors(folder):
    import numpy as np
    from catboost import CatBoostClassifier, CatBoostRegressor
    torch = _torch_setup(42)
    folder = Path(folder)
    manifest = _read_manifest(folder)
    if manifest['schema'] != 'RDE1_MODELS_V1':
        raise ValueError('ARTIFACT_SCHEMA')
    scaler = TrainScaler.from_manifest(manifest['scaler'])
    classifiers, regressors = [], []
    for horizon in (15,60):
        classifier, regressor = CatBoostClassifier(), CatBoostRegressor()
        classifier.load_model(str(folder/f'cat_classifier_{horizon}.cbm'))
        regressor.load_model(str(folder/f'cat_regressor_{horizon}.cbm'))
        classifiers.append(classifier); regressors.append(regressor)
    model = _gru_model()
    model.load_state_dict(torch.load(folder/'gru.pt', map_location='cpu', weights_only=True))
    bundle = PredictorBundle(scaler,classifiers,regressors,model,
                             np.asarray(manifest['return_scale'], dtype=np.float32),manifest)
    bundle.assert_frozen()
    return bundle


def policy_probability_record(probabilities):
    try:
        values = [float(v) for v in probabilities]
        if (len(values) != 4 or any(not math.isfinite(v) or v < 0 or v > 1 for v in values)
                or abs(sum(values)-1.) > 1e-5):
            raise ValueError('ACTION_PROBABILITIES')
    except (TypeError, OverflowError):
        raise ValueError('ACTION_PROBABILITIES') from None
    return {'action_probabilities':dict(zip(('WAIT_HOLD','BUY10','BUY30','EXIT'),values))}


class PolicyHandle:
    def __init__(self, model, manifest):
        self.model = model
        self.manifest = manifest

    def predict(self, observation, deterministic=True):
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def action_probabilities(self, observation):
        import torch
        with torch.inference_mode():
            tensor, _ = self.model.policy.obs_to_tensor(observation)
            values = self.model.policy.get_distribution(tensor).distribution.probs[0].cpu().numpy()
        return policy_probability_record(values)['action_probabilities']


def fit_policy(env, folder, *, seed, gamma, total_timesteps=131072):
    if (seed not in SEEDS or gamma not in (0, 1) or type(total_timesteps) is not int
            or total_timesteps <= 0 or total_timesteps % 512):
        raise ValueError('POLICY_CONTRACT')
    _torch_setup(seed)
    environment = environment_manifest()
    from stable_baselines3 import PPO
    if env.observation_space.shape != (OBSERVATION_SIZE,) or env.action_space.n != 4:
        raise ValueError('POLICY_CONTRACT_OBSERVATIONS')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    params = dict(n_steps=512,batch_size=128,n_epochs=5,learning_rate=.0003,
                  gamma=float(gamma),gae_lambda=.95,clip_range=.2,ent_coef=.01,
                  vf_coef=.5,max_grad_norm=.5,target_kl=.03,device='cpu',seed=seed,
                  policy_kwargs=dict(net_arch=dict(pi=[32,32],vf=[32,32])),verbose=0)
    model = PPO('MlpPolicy',env,**params)
    model.learn(total_timesteps=total_timesteps)
    model.save(str(folder/'policy.zip'))
    manifest = dict(schema='RDE1_POLICY_V1',source='SIMULATION',promotion_allowed=False,
                    seed=seed,gamma=float(gamma),requested_steps=total_timesteps,
                    actual_steps=model.num_timesteps,params=params,
                    environment=environment,source_sha256=_file_sha(__file__),
                    files={'policy.zip':_file_sha(folder/'policy.zip')})
    manifest['manifest_sha256'] = _digest(manifest)
    _write(folder/'manifest.json',manifest)
    return PolicyHandle(model,manifest)


def load_policy(folder):
    from stable_baselines3 import PPO
    folder = Path(folder); manifest = _read_manifest(folder)
    if manifest['schema'] != 'RDE1_POLICY_V1':
        raise ValueError('ARTIFACT_SCHEMA')
    return PolicyHandle(PPO.load(str(folder/'policy.zip'),device='cpu'),manifest)
