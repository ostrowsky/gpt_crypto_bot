"""Freeze baseline inputs from verified complete archive; disclose fallback gaps."""
import argparse
import json
from pathlib import Path

import numpy as np

import config
import replay_backtest as rb
from historical_signal_evaluation import freeze, sha
from recover_rocket_history import valid


def prepare(archive, local, output, symbols, start, end):
    manifest=json.loads((archive/'manifest.json').read_bytes())
    output.mkdir(parents=True,exist_ok=False)
    index=rb._build_market_cache_index(local)
    summary={'start_ms':start,'end_ms':end,'requested_symbols':symbols,
             'archive_manifest_sha256':sha(archive/'manifest.json'),'series':[],
             'scope':'closed-candle cache repair; no historical universe certification'}
    for sym in symbols:
        for tf in ('15m','1h'):
            step=rb.BAR_MS[tf]; left=(start-step)//step*step; right=end//step*step
            name=f"{sym}_{tf}_{manifest['archive_start_ms']}_{manifest['archive_end_ms']}.json"
            if name in manifest['input_hashes']:
                source=archive/'market'/name
                if sha(source)!=manifest['input_hashes'][name]:
                    raise ValueError('archive SHA mismatch')
                rows=[r for r in json.loads(source.read_bytes()) if left<=r['t']<right]
                origin={'kind':'verified_archive','path':str(source),'sha256':sha(source)}
            else:
                data=rb._load_cached_klines(local,sym,tf,left,right,cache_index=index)
                rows=[] if data is None else [dict(t=int(r['t']),**{k:float(r[k]) for k in ('o','h','l','c','v')}) for r in data]
                # Preserve all source-file hashes because the local loader merges overlaps.
                origin={'kind':'partial_local_fallback','inputs':{
                    str(p):sha(p) for a,b,p in index.get((sym,tf),[]) if a<right and b>left}}
            times=[int(r['t']) for r in rows]
            if (times!=sorted(set(times)) or any(not valid(r,step) or r['t']+step>right for r in rows)):
                raise ValueError('invalid closed cache input')
            expected=set(range(left,right,step));missing=len(expected-set(times))
            dest=output/f'{sym}_{tf}_{left}_{right}.json'
            freeze(dest,json.dumps(rows,allow_nan=False,separators=(',',':')).encode())
            summary['series'].append({'symbol':sym,'tf':tf,'observed':len(rows),
                'expected':len(expected),'missing':missing,'origin':origin,'output_sha256':sha(dest)})
    freeze(output/'manifest.json',json.dumps(summary,indent=2,allow_nan=False).encode())
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--local',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--start-ms',type=int,required=True)
    p.add_argument('--end-ms',type=int,required=True)
    a=p.parse_args();r=prepare(a.archive,a.local,a.output,config.load_watchlist(),a.start_ms,a.end_ms)
    print(json.dumps({'series':len(r['series']),'complete':sum(x['missing']==0 for x in r['series']),
                      'missing':[{k:x[k] for k in ('symbol','tf','missing')} for x in r['series'] if x['missing']]}))
