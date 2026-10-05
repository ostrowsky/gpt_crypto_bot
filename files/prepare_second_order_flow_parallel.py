"""Parallelize independent asset samplers; preserve the exact causal algorithm."""
from __future__ import annotations
import argparse,ast,json,shutil,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from minute_direction_data import sha,SYMBOLS
import second_order_flow_data as d


def worker(raw,output,symbol):
    # A private process owns each asset's carry/session; no shared mutable sampler.
    d.SYMBOLS=(symbol,)
    d.prepare(raw,output)


def completed_prefix(raw,root,log,symbol='BTCUSDT'):
    import pyarrow.parquet as pq
    files=sorted((raw/symbol).glob('*.parquet'));lines=log.read_text(encoding='utf-8').splitlines()
    relevant=[line for line in lines if line.startswith('One-second OFI '+symbol+' ')]
    if len(relevant)!=len(files) or files[-1].name not in relevant[-1]:raise ValueError('Sequential asset prefix incomplete')
    audit=ast.literal_eval(relevant[-1].split(' audit ',1)[1])
    if audit['raw_rows']!=sum(pq.ParquetFile(p).metadata.num_rows for p in files):raise ValueError('Raw prefix denominator mismatch')
    chunks=sorted((root/symbol).glob('*.npz'))
    if len(chunks)!=len(files):raise ValueError('Prepared prefix parts incomplete')
    parts=[];last=None
    for i,p in enumerate(chunks):
        if p.name!=f'{i:03d}.npz':raise ValueError('Prepared prefix part order mismatch')
        with np.load(p,allow_pickle=False) as a:
            t=a['time']
            if not len(t) or np.any(np.diff(t)!=d.STEP) or (last is not None and t[0]!=last+d.STEP):raise ValueError('Prepared prefix clock broken')
            last=int(t[-1]);parts.append(dict(file=p.name,sha256=sha(p),n=len(t),start=int(t[0]),end=last,valid=int((a['segment']>=0).sum())))
    return dict(parts=parts,audit=audit)


def assemble(raw,output,roots,prefix_log=None):
    if (output/'coverage.json').exists():raise ValueError('Prepared coverage already published')
    assembler_hash=sha(__file__)
    manifest=json.loads((raw/'manifest.json').read_text(encoding='utf-8'));manifest_hash=sha(raw/'manifest.json');source_hash=sha(d.__file__)
    if len(manifest['files'])!=90 or manifest['dataset']!='predict-quant/binance-future-orderbook' or manifest['revision']!='b8590b83452d7a32fbb274ff7741b6db000b3984':raise ValueError('Wrong archive')
    if {r['path'] for r in manifest['files']}!={p.relative_to(raw).as_posix() for s in SYMBOLS for p in (raw/s).glob('*.parquet')}:raise ValueError('Archive inventory drift')
    for row in manifest['files']:
        if sha(raw/row['path'])!=row['lfs']['oid']:raise ValueError('Raw source drift')
    assets={};receipts={}
    for s in SYMBOLS:
        root=roots[s]
        if prefix_log is not None and s=='BTCUSDT':
            asset=completed_prefix(raw,root,prefix_log);receipts[s]=dict(status='REUSED_COMPLETED_ASSET_FROM_STOPPED_SEQUENTIAL_RUN',log_sha256=sha(prefix_log),source_hash=source_hash)
        else:
            report=json.loads((root/'coverage.json').read_text(encoding='utf-8'))
            if report['status']!='PREPARED_CAUSAL_BOOK_OFI' or report['source_hash']!=source_hash or report['source_manifest_sha256']!=manifest_hash or set(report['assets'])!={s}:raise ValueError('Worker provenance mismatch')
            for key,value in [('step_ms',d.STEP),('max_quote_age_ms',d.MAX_AGE),('max_source_gap_ms',d.MAX_GAP)]:
                if report[key]!=value:raise ValueError('Worker sampling policy differs')
            asset=report['assets'][s];receipts[s]=dict(status='COMPLETE_INDEPENDENT_ASSET_WORKER',coverage_sha256=sha(root/'coverage.json'),source_hash=source_hash)
        if len(asset['parts'])!=sum(r['path'].startswith(s+'/') for r in manifest['files']):raise ValueError('Missing prepared raw file')
        (output/s).mkdir(parents=True,exist_ok=True)
        for part in asset['parts']:
            src=root/s/part['file'];dst=output/s/part['file']
            if sha(src)!=part['sha256']:raise ValueError('Worker part drift')
            if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
            if sha(dst)!=part['sha256']:raise ValueError('Assembled part drift')
        if len(list((output/s).glob('*.npz')))!=len(asset['parts']):raise ValueError('Unexpected assembled parts')
        assets[s]=asset
    if sha(d.__file__)!=source_hash or sha(raw/'manifest.json')!=manifest_hash or sha(__file__)!=assembler_hash:raise ValueError('Preparation source changed')
    report=dict(status='PREPARED_CAUSAL_BOOK_OFI',source_manifest_sha256=manifest_hash,source_hash=source_hash,step_ms=d.STEP,
        max_quote_age_ms=d.MAX_AGE,max_source_gap_ms=d.MAX_GAP,assets=assets,preparation_execution=dict(mode='independent parallel assets',receipts=receipts,assembler_sha256=assembler_hash))
    temporary=output/'coverage.json.tmp';temporary.write_text(json.dumps(report,indent=2),encoding='utf-8');temporary.replace(output/'coverage.json')
    print('ASSEMBLED all 90 source files',output,flush=True)


def run(raw,output):
    source_hash=sha(__file__)
    output.mkdir(parents=True,exist_ok=False);roots={s:output/'workers'/s for s in SYMBOLS}
    paths=[str(p) for p in sys.path]
    def launch(s):
        command=f"import sys;from pathlib import Path;sys.path[:0]={paths!r};from prepare_second_order_flow_parallel import worker;worker(Path({str(raw)!r}),Path({str(roots[s])!r}),{s!r})"
        with (output/(s+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-c',command],stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(launch,SYMBOLS))
    if sha(__file__)!=source_hash:raise ValueError('Parallel preparation code changed during run')
    assemble(raw,output,roots)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.data,a.output)
