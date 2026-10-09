"""Actual candidate generation prefix-invariance probe, with neutral unknown sentinels."""
import argparse,asyncio,json,pickle,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
import replay_backtest as rb
from compare_price_volatility_bot import read_feature_pack
from mission_rule_policy import rules_only

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def closed_prefix(pack,tf,clock):
    step={'15m':900000,'1h':3600000,'4h':14400000}[tf];data=pack[0];prefix=data[data['t']+step<=clock].copy()
    sentinel=np.zeros(1,dtype=data.dtype);sentinel['t']=clock//step*step
    for k in ('o','h','l','c'):sentinel[k]=prefix['c'][-1]
    out=np.concatenate([prefix,sentinel]);f=rb.compute_features(out['o'],out['h'],out['l'],out['c'],out['v']);return out,f

async def audit(experiment):
    reg=json.loads((experiment/'registration.json').read_bytes());prepared=json.loads((experiment/'prepared_parent_receipt.json').read_bytes());features=Path(prepared['parent'])/'features'
    with (experiment/'candidate_snapshot.pkl').open('rb') as f:snapshot=pickle.load(f)
    packs={s:read_feature_pack(features/(s+'.npz')) for s in ('BTCUSDT','ETHUSDT','AMPUSDT','SOLUSDT')}
    records=[]
    with rules_only():
        for s in packs:
            selected=next(c for at in sorted(snapshot[0]) if at>=reg['test_boundary'] for c in snapshot[0][at] if c.sym==s)
            clock=selected.ts_ms;prefix={tf:closed_prefix(pack,tf,clock) for tf,pack in packs[s].items()}
            btc=closed_prefix(packs['BTCUSDT']['1h'],'1h',clock);context=rb._build_bull_day_context(btc[0])
            data,feat=prefix[selected.tf]
            candidates=await rb._build_candidates_for_symbol(s,selected.tf,data,feat,{s:prefix['15m']},{s:prefix['4h']},context,variant='score_replace_cluster',include_trend_start=False)
            same=[c for c in candidates if c.ts_ms==clock and c.mode==selected.mode]
            if len(same)!=1 or asdict(same[0])!=asdict(selected):raise ValueError('future truncation changes candidate '+s+' '+str(clock))
            records.append(dict(symbol=s,tf=selected.tf,clock=clock,mode=selected.mode,state='PASS_ALL_CANDIDATE_FIELDS'))
    result=dict(status='PASS',result_sha256=digest(experiment/'result.json'),probes=records,scope='real prefix-only regeneration with neutral unclosed sentinel; all candidate fields match; finite probe, not historical PIT certification')
    (experiment/'prefix_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();asyncio.run(audit(a.experiment))
