"""Synthetic temporary-file CPU benchmark; no production dataset reads/writes."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import tempfile
import time
from unittest.mock import patch

import ml_dataset as ml


def run(rows=2000,repeats=3):
    if rows<2 or repeats<1: raise ValueError('invalid sample')
    seq=[[1.01,1.02,.99,1.,1.1,.4,25.,55.,.003,.02] for _ in range(30)]
    raw=''.join(json.dumps({'id':str(i),'seq':seq,'labels':{}})+'\n' for i in range(rows))
    target=str(rows-1)
    def mutate(row):
        if row.get('id')==target: row['labels']['ret_3']=1.25; return True
        return False
    times={'legacy':[],'scoped':[]};expected=None
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'benchmark.jsonl'
        with patch.object(ml,'ML_FILE',path):
            for repeat in range(repeats):
                order=('legacy','scoped') if repeat%2==0 else ('scoped','legacy')
                for arm in order:
                    path.write_text(raw,encoding='utf-8')
                    start=time.process_time()
                    counts=ml._rewrite_records(mutate,raw_required_tokens=(f'"id": "{target}"',),
                        mutation_scope=(lambda r:r.get('id')==target) if arm=='scoped' else None)
                    times[arm].append(time.process_time()-start)
                    decoded=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
                    if counts!=(1,0) or len(decoded)!=rows: raise AssertionError('count drift')
                    if expected is None: expected=decoded
                    elif decoded!=expected: raise AssertionError('semantic drift')
    a,b=median(times['legacy']),median(times['scoped'])
    return {'scope':'synthetic_writer_not_whole_bot','rows':rows,'bytes':len(raw.encode()),
        'repeats':repeats,'semantic_parity':'PASS','legacy_cpu_median_sec':a,
        'scoped_cpu_median_sec':b,'cpu_reduction_pct':100*(a-b)/a,
        'source_sha256':hashlib.sha256(Path(ml.__file__).read_bytes()).hexdigest()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=int,default=2000)
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    print(json.dumps(run(a.rows,a.repeats)))
