"""Independent result reconciliation and sampled raw closed-candle screen audit."""
import argparse
import json
import math
from pathlib import Path
import pickle

import numpy as np
import replay_backtest as rb
from historical_signal_evaluation import sha,freeze


def verify(directory,market,candidates):
    receipt=json.loads((directory/'receipt.json').read_bytes())
    for name,value in receipt.items():
        if Path(name).name!=name or sha(directory/name)!=value:raise ValueError('result receipt drift')
    r=json.loads((directory/'result.json').read_bytes());reg=json.loads((directory/'registration.json').read_bytes())
    for name,value in reg['sources'].items():
        if Path(name).name!=name or sha(directory/'source_snapshot'/name)!=value:
            raise ValueError('source snapshot drift')
    results={}
    for arm,a in r['accounts'].items():
        ledger=json.loads((directory/f'ledger_{arm}.json').read_bytes())
        curve=np.array(a['curve']);times=curve[:,0].astype('int64');equity=curve[:,1]
        if not np.array_equal(times,np.arange(r['start_ms'],r['end_ms']+1,900000)):
            raise ValueError('equity grid gap')
        if not np.isfinite(equity).all() or (equity<=0).any():raise ValueError('unknown equity')
        if len(ledger)!=a['trades']:raise ValueError('ledger trade denominator mismatch')
        for row in ledger:
            if not math.isclose(row['raw_price_pnl']-row['fees']-row['slippage'],row['net_pnl'],abs_tol=1e-8):
                raise ValueError('per-trade money mismatch')
        summed={k:sum(v[k] for v in ledger) for k in ('raw_price_pnl','fees','slippage','net_pnl','turnover')}
        for key,value in summed.items():
            if not math.isclose(value,a['totals'][key],abs_tol=1e-7):raise ValueError('attribution sum mismatch')
        if not math.isclose(summed['net_pnl'],a['ending_cash']-10000,abs_tol=1e-7):raise ValueError('cash reconciliation mismatch')
        net=(equity[-1]/10000-1)*100;peak=np.maximum.accumulate(np.r_[10000,equity])[1:]
        if not math.isclose(equity[-1],a['ending_cash'],abs_tol=1e-7):raise ValueError('terminal cash/equity mismatch')
        if not math.isclose(net-r['benchmark']['net_return_after_costs_pct'],a['alpha_pp'],abs_tol=1e-8):
            raise ValueError('named benchmark alpha mismatch')
        dd=float(((1-equity/peak)*100).max())
        test_start=np.where(times==r['split_boundaries_ms'][1])[0]
        if len(test_start)!=1:raise ValueError('missing common TEST origin')
        test=(equity[-1]/equity[test_start[0]]-1)*100
        for key,value in [('net_return_pct',net),('max_drawdown_pct',dd),('test_return_pct',test)]:
            if not math.isclose(value,a[key],abs_tol=1e-8):raise ValueError('portfolio metric mismatch: '+key)
        for group in r['attribution'][arm].values():
            for k in summed:
                if not math.isclose(sum(v[k] for v in group.values()),summed[k],abs_tol=1e-7):
                    raise ValueError('grouped attribution mismatch')
        results[arm]={'ledger_rows':len(ledger),'equity_points':len(equity),'net_return_pct':net,
            'test_return_pct':test,'max_drawdown_pct':dd}
    if not candidates.resolve().is_relative_to(Path(__file__).resolve().parent.parent/'.runtime'):
        raise ValueError('trusted local candidate checkpoint required')
    cr=json.loads((candidates/'candidate_checkpoint_receipt.json').read_bytes())
    if sha(candidates/'candidate_snapshot.pkl')!=cr['snapshot_sha256']:raise ValueError('candidate snapshot drift')
    with (candidates/'candidate_snapshot.pkl').open('rb') as f:raw,times,n=pickle.load(f)
    rows=[c for at in sorted(raw) for c in raw[at]]
    if len(rows)!=n:raise ValueError('candidate denominator mismatch')
    fee=r['costs']['fee_bps']/10000;slip=r['costs']['slippage_bps']/10000
    threshold=200*((1+fee)*(1+slip)/((1-fee)*(1-slip))-1)
    manifest=json.loads((market/'manifest.json').read_bytes())
    by_symbol={}
    for c in rows:by_symbol.setdefault(c.sym,[]).append(c)
    examined=0;accepted=0;unknown=0;allowed=set()
    for sym,candidates_for_symbol in by_symbol.items():
        name=sym+'_15m.json';p=market/'market'/name
        if sha(p)!=manifest['input_hashes'][name]:raise ValueError('raw price drift')
        data={int(x['t'])+900000:(x['h'],x['l']) for x in json.loads(p.read_bytes())}
        for c in candidates_for_symbol:
            examined+=1;past=[data.get(c.ts_ms-j*900000) for j in range(4)]
            if any(v is None for v in past):unknown+=1;continue
            value=100*(max(v[0] for v in past)/min(v[1] for v in past)-1)
            if value>=threshold:
                accepted+=1;allowed.add((c.sym,c.tf,c.ts_ms))
    audit=r['filter_audit']
    if (examined,accepted,unknown)!=(audit['input'],audit['accepted'],audit['unknown_past']):
        raise ValueError('raw complete screen audit mismatch')
    for arm in ('amplitude_cost','combined'):
        trades=json.loads((directory/f'trades_{arm}.json').read_bytes())
        if any((t['sym'],t['tf'],t['entry_ts']) not in allowed for t in trades):
            raise ValueError('filtered arm admitted an ineligible candidate')
    if not math.isclose(threshold,r['filter_audit']['threshold_pct'],abs_tol=1e-12):
        raise ValueError('cost screen threshold mismatch')
    full_denominators={m['label_pair_count'] for m in r['missions'].values()}
    test_denominators={m['label_pair_count'] for m in r['test_missions'].values()}
    if len(full_denominators)!=1 or len(test_denominators)!=1:
        raise ValueError('mission denominator differs between arms')
    return {'status':'PASS','accounts':results,'raw_screen_candidates_checked':examined,
        'raw_screen_accepted':int(accepted),'raw_screen_unknown':unknown,'verifier_sha256':sha(Path(__file__)),
        'scope':'numeric/provenance verification; no live parity or deployment authority'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','market','candidates'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=verify(a.run,a.market,a.candidates)
    freeze(a.run/'independent_verification.json',json.dumps(r,indent=2,allow_nan=False).encode())
    print(json.dumps(r,indent=2))
