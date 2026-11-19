"""MR-1 lazy lossless signal lookup and bounded-memory portfolio continuation."""
import argparse
from bisect import bisect_left,bisect_right
import gc
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time
from . import receipt_flow_direction as io
from . import receipt_flow_portfolio_study as original
from .receipt_flow_portfolio import Policy,simulate,digest
from .receipt_flow_portfolio_audit import audit_events
from .receipt_flow_quotes import ReceiptQuotes
from .receipt_flow_journal import DurableGzipText
from .receipt_flow_continuation import VerifiedParent,_decompressed_proof


class LazySignals:
    def __init__(self,data,symbol_names,feature_names,start_ms,end_ms):
        self.data=data;self.names=symbol_names;self.start=start_ms;self.end=end_ms
        self.times=data['times'];self.symbols=data['symbols'];self.x=data['features_F'];self.mids=data['mids']
        self.micro=feature_names.index('micro_bps');self.spread=feature_names.index('spread_bps')
        self.lo=bisect_left(self.times,start_ms);self.hi=bisect_left(self.times,end_ms)
        self.hashes=bytearray((self.hi-self.lo)*64);body=hashlib.sha256();body.update(b'[')
        previous=None
        for i in range(self.lo,self.hi):
            t=int(self.times[i]);symbol=self.names[int(self.symbols[i])]
            if previous is not None and t<previous:raise ValueError('LAZY_SOURCE_ORDER')
            previous=t
            feature_sha=digest(dict(source_row=i,at_ms=t,symbol=symbol,features=[float(v) for v in self.x[i]]))
            self.hashes[(i-self.lo)*64:(i-self.lo+1)*64]=feature_sha.encode('ascii')
            if i>self.lo:body.update(b',')
            body.update(json.dumps([t,symbol,i,feature_sha],sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        body.update(b']');self.provenance=body.hexdigest();self.active_key=None;self.vector=None

    def for_case(self,arm,horizon,coefficients):
        key=f'{arm}_{horizon}' if arm in ('P','B','F') else None
        if key!=self.active_key:
            self.vector=None;gc.collect();self.vector=self.data[key] if key is not None else None;self.active_key=key
        return _AtTime(self,arm,horizon,coefficients,self.vector)

    def evidence(self):
        return dict(issued_rows_n=self.hi-self.lo,provenance_sha256=self.provenance,
            retained_origin_dicts_n=0,feature_hash_bytes=len(self.hashes),
            future_targets_loaded=False,precision='ORIGINAL_FLOAT64',max_active_prediction_vectors=1)


class _AtTime:
    def __init__(self,source,arm,horizon,coefficients,vector):
        self.source=source;self.arm=arm;self.horizon=horizon;self.coefficients=coefficients;self.vector=vector

    def get(self,t,default=None):
        s=self.source
        if not s.start<=t<s.end:return default
        lo=max(s.lo,bisect_left(s.times,t));hi=min(s.hi,bisect_right(s.times,t))
        if lo==hi:return default
        result=[]
        for i in range(lo,hi):
            row=dict(at_ms=int(s.times[i]),feature_end_ms=int(s.times[i])-1,
                symbol=s.names[int(s.symbols[i])],source_row=i,
                features_sha256=s.hashes[(i-s.lo)*64:(i-s.lo+1)*64].decode('ascii'),
                micro_bps=float(s.x[i][s.micro]),spread_bps=float(s.x[i][s.spread]),origin_mid=float(s.mids[i]),
                predictions={f'{self.arm}_{self.horizon}':float(self.vector[i])} if self.vector is not None else {})
            result.extend(original.signal_rows([row],arm=self.arm,horizon=self.horizon,coefficients=self.coefficients))
        return result


def _events(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:yield json.loads(line)


def run(parent,rfd,capture,folder):
    import numpy as np
    parent=Path(parent).resolve();rfd=Path(rfd).resolve();capture=Path(capture).resolve();folder=Path(folder).resolve()
    if folder.exists():raise ValueError('OUTPUT_EXISTS')
    verified=VerifiedParent(parent);contract=verified.registration
    if contract['cases']!=original.case_definitions():raise ValueError('CASE_DEFINITIONS_CHANGED')
    if contract['environment']!=dict(python='3.11.9',numpy=np.__version__):raise ValueError('MEMORY_ENVIRONMENT')
    import platform
    if platform.python_version()!='3.11.9':raise ValueError('MEMORY_PYTHON')
    incident_path=parent/'incident.json'
    if not incident_path.exists():
        outer=io.read_frozen(parent.parent/'continuation-registration.json')
        if outer['child']!=parent.name:raise ValueError('PARENT_INCIDENT_SCOPE')
        incident_path=parent.parent/'incident.json'
    source=Path(__file__).parent;spec=source.parents[3]/'docs/specs/binance-demo-receipt-flow-memory.md'
    folder.mkdir(parents=True);child=folder/'economic';child.mkdir()
    supplemental=dict(protocol='RFP-MR-1',parent=str(parent),parent_incident=json.loads(incident_path.read_text()),
        parent_status=json.loads((parent/'status.json').read_text()),
        economic_registration_sha256=io.file_sha(parent/'registration.json'),
        reused_case_vector={k:v['manifest'] for k,v in verified.cases.items()},
        active_sources={name:io.file_sha(source/name) for name in
            ('receipt_flow_lazy.py','receipt_flow_journal.py','receipt_flow_continuation.py')},
        memory_spec_sha256=io.file_sha(spec),quote_cache_max=4,
        economics='ORIGINAL_POLICY_SIMULATOR_AND_SIGNAL_HELPER',no_training=True,
        no_future_target_reads=True,financial_certification=False,promotion_allowed=False)
    io.freeze(folder/'memory-registration.json',supplemental)
    snapshot=folder/'source-snapshot';snapshot.mkdir()
    for name in supplemental['active_sources']:(snapshot/name).write_bytes((source/name).read_bytes())
    (snapshot/spec.name).write_bytes(spec.read_bytes())
    io.freeze(child/'registration.json',contract)
    for name in ('microprice-calibration.json','signals-manifest.json'):
        io.freeze(child/name,io.read_frozen(parent/name))
        if io.file_sha(child/name)!=io.file_sha(parent/name):raise ValueError('COPIED_CONTRACT_CHANGED')
    expected_signals=io.read_frozen(child/'signals-manifest.json');coefficients=io.read_frozen(child/'microprice-calibration.json')
    rfd_registration=io.read_frozen(rfd/'registration.json');predicted=io.read_frozen(rfd/'predictions.json')
    data_path=rfd/'frozen-predictions.npz'
    if (str(data_path) not in contract['inputs'] or io.file_sha(data_path)!=contract['inputs'][str(data_path)] or
            predicted['sha256']!=contract['inputs'][str(data_path)]):raise ValueError('MEMORY_FORECAST_INPUT')
    data=np.load(data_path,allow_pickle=False)
    signals=LazySignals(data,contract['source_symbols'],rfd_registration['feature_names']['F'],contract['start_ms'],contract['end_ms'])
    evidence=signals.evidence()
    if (evidence['issued_rows_n']!=expected_signals['issued_rows_n'] or
            evidence['provenance_sha256']!=expected_signals['provenance_sha256']):raise ValueError('LOSSLESS_SIGNAL_PROVENANCE')
    io.freeze(folder/'signal-memory-proof.json',evidence)
    manifest=json.loads((capture/'manifest.json').read_text())
    expected={item['file']:rfd_registration['inputs'].get(str(capture/item['file'])) for item in manifest['chunks']}
    quotes=ReceiptQuotes(capture,manifest,expected,cache_limit=4)
    results={};summaries=[];audits={};first_trades={};journals={};recomputed=[]
    try:
        for case in contract['cases']:
            key=case['case_id'];out=child/'cases'/key;out.mkdir(parents=True)
            policy=Policy() if case['arm']=='CASH' else Policy(horizon_seconds=case['horizon_seconds'],
                admission=case['admission'],**original.PROFILES[case['profile']])
            io.write_json(folder/'status.json',dict(phase='MEMORY_BOUNDED_PORTFOLIO_REPLAY',case_id=key,
                complete_n=len(summaries),cases_N=61,reusable_cases_n=len(verified.cases),promotion_allowed=False))
            io.write_json(child/'status.json',dict(phase='MEMORY_BOUNDED_PORTFOLIO_REPLAY',case_id=key,complete_n=len(summaries),promotion_allowed=False))
            first=None;latest=None;pending=out/'events.jsonl.gz.partial'
            with DurableGzipText(pending) as stream:
                def emit(e):
                    nonlocal first,latest
                    stream.write(json.dumps(e,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
                    if e['kind']=='DECISION' and e['selected'] is not None:latest=(e['decision_id'],e['selected'])
                    elif e['kind']=='BUY_FILL' and first is None:
                        if latest is None or latest[0]!=e['decision_id']:raise ValueError('FIRST_TRADE_SOURCE')
                        first=dict(buy=e,selected=latest[1])
                    elif e['kind']=='SELL_FILL' and first is not None and 'sell' not in first and e['decision_id']==first['buy']['decision_id']:first['sell']=e
                if key in verified.cases:result=verified.replay(key,child,policy,emit)
                else:
                    recomputed.append(key)
                    view={} if case['arm']=='CASH' else signals.for_case(case['arm'],case['horizon_seconds'],coefficients)
                    result=simulate(view,start_ms=contract['start_ms'],end_ms=contract['end_ms'],quote_at=quotes.quote_at,policy=policy,emit=emit)
            journal=out/'events.jsonl.gz';pending.replace(journal)
            proof=stream.proof();actual=_decompressed_proof(journal)
            if not proof['complete'] or any(actual[k]!=proof[k] for k in actual):raise ValueError('MEMORY_JOURNAL_PROOF')
            if key in verified.cases and actual!=_decompressed_proof(verified.cases[key]['path']):raise ValueError('REUSED_EVENT_BYTES_CHANGED')
            io.freeze(out/'result.json',result)
            checked=audit_events(_events(journal),result,policy,quote_at=quotes.quote_at)
            io.freeze(out/'audit.json',checked);audits[key]=checked;journals[key]=proof
            io.freeze(out/'manifest.json',dict(case=case,policy=asdict(policy),events_sha256=io.file_sha(journal),
                result_sha256=io.file_sha(out/'result.json'),audit_sha256=io.file_sha(out/'audit.json'),
                registration_sha256=io.file_sha(child/'registration.json'),microprice_calibration_sha256=verified.coefficients))
            summaries.append(original._summary(result,case,contract['test_start_ms']));results[key]=result
            if first is not None:first_trades[key]=first
            gc.collect()
        comparisons=[]
        for h in (5,10,30):
            for admission in ('direction','cost-aware'):
                for profile in ('base','stress'):
                    left=f'F-h{h}-{admission}-{profile}'
                    for ref in ('M1','B'):
                        right=f'{ref}-h{h}-{admission}-{profile}'
                        comparisons.append(dict(left=left,right=right,**original.paired_daily(results[left]['daily'],results[right]['daily'],test_start_ms=contract['test_start_ms'])))
        report=dict(state='COMPLETE_CONDITIONAL_SIMULATION',cases=summaries,start_ms=contract['start_ms'],end_ms=contract['end_ms'],
            test_start_ms=contract['test_start_ms'],prior_issued_rows_n=evidence['issued_rows_n'],raw_validation_proof_reused=True,
            paired_test_comparisons=comparisons,truth_harness=contract['truth_harness'],
            limitations=['CONDITIONAL_SIMULATION_NOT_ACTUAL_ACCOUNT_PNL','EXPOSED_TEST_TWO_DAYS_NO_INDEPENDENT_SIGNIFICANCE',
                'THREE_LIQUID_DEMO_ASSETS_NOT_ALL_USDT_POPULATION','EXCHANGE_AGE_UNKNOWN_RECEIPT_CLOCK',
                'TOP_OF_BOOK_AND_SLIPPAGE_FILL_PROXY_NOT_QUEUE_CERTIFICATION','FRACTIONAL_QUANTITY_HISTORICAL_FILTERS_NOT_CERTIFIED',
                'DAYS_WITH_UNMARKED_HELD_EXPOSURE_EXCLUDED_NOT_ZERO','NO_MODEL_OR_POLICY_PROMOTION'],
            financial_certification=False,promotion_allowed=False)
        original._render(child,report,results,first_trades,quotes)
        quotes.verify_used()
        for path,sha in contract['inputs'].items():
            if io.file_sha(path)!=sha:raise ValueError('MEMORY_INPUT_CHANGED')
        for name,sha in supplemental['active_sources'].items():
            if io.file_sha(source/name)!=sha:raise ValueError('MEMORY_SOURCE_CHANGED')
        io.freeze(child/'quote-source-evidence.json',quotes.evidence());io.freeze(child/'report.json',report)
        io.freeze(child/'ledger-audits.json',dict(state='PASS_LEDGER_ARTIFACT_AUDIT_ONLY',cases_n=len(audits),cases=audits,
            financial_certification=False,promotion_allowed=False))
        io.freeze(folder/'memory-evidence.json',dict(state='COMPLETE_LOSSLESS_MEMORY_CONTINUATION',cases_n=len(summaries),
            reused_cases=verified.reused,recomputed_cases=recomputed,signal_proof=evidence,journals=journals,
            parent_stays_incomplete=True,financial_certification=False,promotion_allowed=False))
        io.write_json(child/'status.json',dict(phase='COMPLETE_CONDITIONAL_SIMULATION',cases_n=61,promotion_allowed=False))
        io.write_json(folder/'status.json',dict(phase='COMPLETE_CONDITIONAL_SIMULATION',cases_n=61,reused_cases_n=len(verified.reused),
            recomputed_cases_n=len(recomputed),promotion_allowed=False))
        return report
    except BaseException as error:
        incident=dict(exception=type(error).__name__,detail=str(error),at_ms=time.time_ns()//1000000,promotion_allowed=False)
        io.write_json(folder/'incident.json',incident);io.write_json(folder/'status.json',dict(phase='INCOMPLETE',**incident))
        io.write_json(child/'status.json',dict(phase='INCOMPLETE',**incident));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('parent','rfd','capture','output'):parser.add_argument('--'+arg,required=True)
    args=parser.parse_args()
    from .public_flow_dataset import own_runtime_path
    app=Path(__file__).resolve().parents[2]
    parent,rfd,capture,folder=[own_runtime_path(app,getattr(args,n)) for n in ('parent','rfd','capture','output')]
    if any(folder==p or folder.is_relative_to(p) or p.is_relative_to(folder) for p in (parent,rfd,capture)):
        raise ValueError('OVERLAPPING_PATHS')
    report=run(parent,rfd,capture,folder)
    print(json.dumps(dict(state=report['state'],cases_n=len(report['cases']),promotion_allowed=False)),flush=True)


if __name__=='__main__':main()
