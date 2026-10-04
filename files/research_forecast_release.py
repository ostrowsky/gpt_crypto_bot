"""Portable numeric/native model IO; no pickle or inference fitting."""
import base64
import hashlib
import json
import zlib


def pack_release(experiment, cfg, provenance):
    import numpy as np
    from dataclasses import asdict
    payload=dict(schema=1,features=FEATURES,config=asdict(cfg),provenance=provenance,
                 evidence={k:experiment.get(k,[]) for k in ('results','status','metadata','choices','direction_baselines')},
                 states={},widths={})
    for symbol in cfg.symbols:
        payload['states'][symbol]={};payload['widths'][symbol]={}
        names=tuple(dict.fromkeys(cfg.models+(('Persistence',) if 'Persistence' in experiment['policies'][symbol] else ())))
        for name in names:
            policy=experiment['policies'][symbol][name]
            state=dict(name=name,n_fit=getattr(policy,'n_fit',None))
            if name=='Persistence':
                pass
            elif name=='Ridge':
                scaler,regressor=policy.model.steps[0][1],policy.model.steps[1][1]
                state.update(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                             coef=regressor.coef_.tolist(),intercept=regressor.intercept_.tolist())
            elif name=='XGBoost':
                state['boosters']=[base64.b64encode(m.get_booster().save_raw(raw_format='json')).decode()
                                   for m in policy.model.estimators_]
            elif name in ('SARIMA','SARIMAX'):
                state.update(phi=float(policy.phi),seasonal=float(policy.seasonal),beta=policy.beta.tolist(),
                             scale=policy.scale,exog_scale=policy.exog_scale.tolist())
            elif name=='ARIMA':
                pass
            elif name=='ETS':
                state.update(alpha=policy.alpha,beta=policy.beta,phi=policy.phi,scale=policy.scale)
            elif name=='Prophet':
                from prophet.serialize import model_to_json
                state.update(model_json=model_to_json(policy.model),center=policy.center,scale=policy.scale)
            elif name in ('LSTM','TFT'):
                from safetensors.torch import save
                # TFT ties prescalers and gating weights. Clone each serialized
                # view; the reconstructed network restores the same sharing.
                state['tensors']=base64.b64encode(save({k:v.detach().cpu().contiguous().clone() for k,v in policy.model.state_dict().items()})).decode()
                if name=='LSTM':
                    state.update(xmean=policy.xscale.mean_.tolist(),xscale=policy.xscale.scale_.tolist(),
                                 ymean=policy.yscale.mean_.tolist(),yscale=policy.yscale.scale_.tolist())
                else:
                    state.update(xmean=policy.xmean.tolist(),xscale=policy.xscale.tolist(),
                                 ymean=policy.ymean,yscale=policy.yscale,template=policy.template)
            else:raise ValueError('Unsupported portable model: '+name)
            payload['states'][symbol][name]=state
            q=np.asarray(experiment['widths'][symbol][name])
            if q.shape!=(cfg.horizon,) or not np.isfinite(q).all() or (q<0).any():
                raise ValueError('Invalid release intervals')
            payload['widths'][symbol][name]=q.tolist()
    raw=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
    return dict(sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(zlib.compress(raw,9)).decode())


class PortableRegressor:
    def __init__(self,state,cfg):
        self.state,self.cfg,self.name=state,cfg,state['name']
        self.n_fit=state['n_fit']
        if self.name=='XGBoost':
            from xgboost import Booster
            self.boosters=[]
            for raw in state['boosters']:
                model=Booster(params={'nthread':2});model.load_model(bytearray(base64.b64decode(raw)))
                self.boosters.append(model)
        elif self.name=='LSTM':
            import torch
            from torch import nn
            from safetensors.torch import load
            torch.set_num_threads(2)
            class Net(nn.Module):
                def __init__(self):
                    super().__init__();self.lstm=nn.LSTM(len(FEATURES),32,batch_first=True);self.head=nn.Linear(32,cfg.horizon)
                def forward(self,x):return self.head(self.lstm(x)[0][:,-1])
            self.model=Net();self.model.load_state_dict(load(base64.b64decode(state['tensors'])));self.model.eval()

    def predict(self,full,rows):
        import numpy as np
        state=self.state
        if self.name=='Ridge':
            x=(rows[FEATURES].to_numpy()-np.asarray(state['mean']))/np.asarray(state['scale'])
            return x@np.asarray(state['coef']).T+np.asarray(state['intercept'])
        if self.name=='XGBoost':
            from xgboost import DMatrix
            data=DMatrix(rows[FEATURES])
            return np.column_stack([m.predict(data) for m in self.boosters])
        import torch
        x=(sequence_inputs(full,rows,self.cfg)-np.asarray(state['xmean']))/np.asarray(state['xscale'])
        with torch.no_grad():pred=self.model(torch.tensor(x,dtype=torch.float32)).numpy()
        return pred*np.asarray(state['yscale'])+np.asarray(state['ymean'])


def unpack_release(envelope):
    import numpy as np
    import pandas as pd
    raw=zlib.decompress(base64.b64decode(envelope['data'],validate=True))
    if hashlib.sha256(raw).hexdigest()!=envelope['sha256']:raise ValueError('Release checksum mismatch')
    payload=json.loads(raw)
    if payload['schema']!=1 or payload['features']!=FEATURES:raise ValueError('Release schema mismatch')
    config=payload['config']
    for key in ('symbols','models','fractions'):config[key]=tuple(config[key])
    cfg=ForecastConfig(**config)
    experiment=dict(payload['evidence'],policies={},widths={},predictions={})
    for symbol in cfg.symbols:
        experiment['policies'][symbol]={};experiment['widths'][symbol]={};experiment['predictions'][symbol]={}
        for name in payload['states'][symbol]:
            state=payload['states'][symbol][name]
            if state['name']!=name:raise ValueError('Release model mismatch')
            for k in ('scale','xscale','yscale','exog_scale'):
                if k in state:
                    values=np.asarray(state[k],dtype=float)
                    if not np.isfinite(values).all() or (values<=0).any():raise ValueError('Invalid release scale')
            for k in ('mean','coef','intercept','xmean','ymean','alpha','phi','seasonal','beta','center'):
                if k in state and not np.isfinite(np.asarray(state[k],dtype=float)).all():raise ValueError('Nonfinite release state')
            if name=='Persistence':policy=None
            elif name in ('Ridge','XGBoost','LSTM'):policy=PortableRegressor(state,cfg)
            elif name=='ARIMA':policy=ClassicalPolicy(name,None,None,None,cfg)
            elif name in ('ETS','SARIMA','SARIMAX'):
                cls=ETSPolicy if name=='ETS' else SeasonalPolicy
                policy=cls.__new__(cls);policy.cfg,policy.name=cfg,name
                for k in ('alpha','beta','phi','scale','seasonal','exog_scale'):
                    if k in state:setattr(policy,k,np.asarray(state[k]) if isinstance(state[k],list) else state[k])
            elif name=='Prophet':
                from prophet.serialize import model_from_json
                policy=ProphetPolicy.__new__(ProphetPolicy);policy.cfg,policy.name=cfg,name
                policy.scale,policy.center=state['scale'],state['center'];policy.model=model_from_json(state['model_json'])
            elif name=='TFT':
                from safetensors.torch import load
                policy=TFTPolicy.__new__(TFTPolicy);policy.cfg,policy.name=cfg,name;policy._init_runtime()
                policy.xmean,policy.xscale=np.asarray(state['xmean']),np.asarray(state['xscale'])
                policy.ymean,policy.yscale=state['ymean'],state['yscale']
                policy.template=state['template'];policy.training=policy._dataset(pd.DataFrame(state['template']))
                policy.model=policy._network();policy.model.load_state_dict(load(base64.b64decode(state['tensors'])));policy.model.eval()
            else:raise ValueError('Unknown model in release')
            if policy is not None:policy.n_fit=state['n_fit']
            q=np.asarray(payload['widths'][symbol][name],dtype=float)
            if q.shape!=(cfg.horizon,) or not np.isfinite(q).all() or (q<0).any():raise ValueError('Invalid release intervals')
            experiment['policies'][symbol][name]=policy;experiment['widths'][symbol][name]=q
            experiment['predictions'][symbol][name]=np.empty((0,cfg.horizon))
        if not set(cfg.models).issubset(experiment['policies'][symbol]):raise ValueError('Incomplete model release')
    return experiment,cfg,payload['provenance']


# Imported normally from the repo, or concatenated after the engine in a notebook.
if 'ForecastConfig' not in globals():
    from research_forecast import (ForecastConfig,FEATURES,sequence_inputs,ClassicalPolicy,
                                   SeasonalPolicy,ETSPolicy,ProphetPolicy,TFTPolicy)
