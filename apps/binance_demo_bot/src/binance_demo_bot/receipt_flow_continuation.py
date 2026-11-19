"""JR03..05 continuation: original economics, durable I/O and sealed case replay."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time
from . import receipt_flow_direction as io
from . import receipt_flow_portfolio_study as study
from .receipt_flow_journal import DurableGzipText


class VerifiedParent:
    def __init__(self,parent):
        self.parent=Path(parent).resolve();self.registration=io.read_frozen(self.parent/'registration.json')
        repo=Path(__file__).resolve().parents[4]
        for path,digest in self.registration['inputs'].items():
            p=Path(path).resolve()
            if not p.is_relative_to(repo) or p.suffix.lower() not in ('.py','.md','.json','.npz'):
                raise ValueError('UNSAFE_PARENT_INPUT')
            if io.file_sha(p)!=digest:raise ValueError('PARENT_INPUT_CHANGED:'+str(p))
        self.coefficients=io.file_sha(self.parent/'microprice-calibration.json')
        self.signals=io.file_sha(self.parent/'signals-manifest.json');self.cases={};self.reused=[]
        for case in self.registration['cases']:
            out=self.parent/'cases'/case['case_id'];manifest_path=out/'manifest.json'
            if not manifest_path.with_suffix('.json.sha256').exists():continue
            m=io.read_frozen(manifest_path);result=io.read_frozen(out/'result.json');audit=io.read_frozen(out/'audit.json')
            if (m['case']!=case or m['registration_sha256']!=io.file_sha(self.parent/'registration.json') or
                    m['microprice_calibration_sha256']!=self.coefficients or
                    io.file_sha(out/'result.json')!=m['result_sha256'] or io.file_sha(out/'audit.json')!=m['audit_sha256'] or
                    io.file_sha(out/'events.jsonl.gz')!=m['events_sha256'] or audit['state']!='PASS_LEDGER_ARTIFACT_AUDIT_ONLY'):
                raise ValueError('PARENT_COMPLETED_CASE_CHANGED')
            self.cases[case['case_id']]=dict(manifest=m,result=result,path=out/'events.jsonl.gz')

    def replay(self,case_id,child,policy,emit):
        child=Path(child)
        if io.read_frozen(child/'registration.json')!=self.registration:raise ValueError('ECONOMIC_REGISTRATION_CHANGED')
        if (io.file_sha(child/'microprice-calibration.json')!=self.coefficients or
                io.file_sha(child/'signals-manifest.json')!=self.signals):raise ValueError('FROZEN_SIGNALS_CHANGED')
        case=self.cases[case_id]
        if case['manifest']['policy']!=asdict(policy):raise ValueError('ECONOMIC_POLICY_CHANGED')
        with gzip.open(case['path'],'rt',encoding='utf-8') as stream:
            for line in stream:emit(json.loads(line))
        self.reused.append(case_id);return deepcopy(case['result'])


class JournalFacade:
    def __init__(self,on_open):self.on_open=on_open;self.writers={}

    def open(self,path,mode='rb',*args,**kwargs):
        if mode=='wt' and str(path).endswith('events.jsonl.gz.partial'):
            if kwargs.get('encoding','utf-8')!='utf-8':raise ValueError('JOURNAL_ENCODING')
            writer=DurableGzipText(path,compresslevel=kwargs.get('compresslevel',3))
            case=Path(path).parent.name;self.writers[case]=writer;self.on_open(case,writer);return writer
        return gzip.open(path,mode,*args,**kwargs)


def _decompressed_proof(path):
    digest=hashlib.sha256();records=0;size=0
    with gzip.open(path,'rb') as stream:
        while block:=stream.read(65536):digest.update(block);records+=block.count(b'\n');size+=len(block)
    return dict(uncompressed_sha256=digest.hexdigest(),records_n=records,uncompressed_bytes=size)


def run(parent,rfd,capture,folder,truth_report):
    parent=Path(parent).resolve();folder=Path(folder).resolve();child=folder/'economic'
    if folder.exists():raise ValueError('OUTPUT_EXISTS')
    verified=VerifiedParent(parent)
    if verified.registration['cases']!=study.case_definitions():raise ValueError('CASE_DEFINITIONS_CHANGED')
    source=Path(__file__).parent;spec=source.parents[3]/'docs/specs/binance-demo-receipt-flow-journal-recovery.md'
    supplemental=dict(protocol='RFP-JR-1',parent=str(parent),parent_registration_sha256=io.file_sha(parent/'registration.json'),
        parent_incident=json.loads((parent/'incident.json').read_text()),parent_status=json.loads((parent/'status.json').read_text()),
        verified_parent_cases={key:value['manifest'] for key,value in verified.cases.items()},
        child='economic',bindings=['study.gzip.open WRITE_MODE_ONLY','study.simulate VERIFIED_COMPLETE_CASES_ONLY'],
        source_sha256={name:io.file_sha(source/name) for name in ('receipt_flow_journal.py','receipt_flow_continuation.py')},
        recovery_spec_sha256=io.file_sha(spec),economic_policy_changed=False,promotion_allowed=False)
    folder.mkdir(parents=True);io.freeze(folder/'continuation-registration.json',supplemental)
    snapshots=folder/'source-snapshot';snapshots.mkdir()
    for name in ('receipt_flow_journal.py','receipt_flow_continuation.py'):(snapshots/name).write_bytes((source/name).read_bytes())
    (snapshots/spec.name).write_bytes(spec.read_bytes())
    current={};recomputed=[]
    def opened(case,writer):current['case']=case
    facade=JournalFacade(opened);original_simulate=study.simulate;original_gzip=study.gzip
    def simulate_bound(signals,*,start_ms,end_ms,quote_at,policy,emit):
        case=current['case']
        if io.read_frozen(child/'registration.json')!=verified.registration:raise ValueError('ECONOMIC_REGISTRATION_CHANGED')
        if case in verified.cases:return verified.replay(case,child,policy,emit)
        recomputed.append(case)
        return original_simulate(signals,start_ms=start_ms,end_ms=end_ms,quote_at=quote_at,policy=policy,emit=emit)
    io.write_json(folder/'status.json',dict(phase='RUNNING_ORIGINAL_ECONOMIC_STUDY',reusable_cases_n=len(verified.cases),promotion_allowed=False))
    try:
        study.gzip=facade;study.simulate=simulate_bound
        report=study.run(rfd,capture,child,truth_report)
        evidence={}
        for case,writer in facade.writers.items():
            proof=writer.proof();actual=_decompressed_proof(child/'cases'/case/'events.jsonl.gz')
            if not proof['complete'] or any(actual[k]!=proof[k] for k in actual):raise ValueError('JOURNAL_FINAL_PROOF')
            if io.file_sha(child/'cases'/case/'events.jsonl.gz')!=proof['compressed_sha256']:raise ValueError('JOURNAL_FINAL_HASH')
            if case in verified.cases and actual!=_decompressed_proof(verified.cases[case]['path']):raise ValueError('REUSED_EVENT_BYTES_CHANGED')
            evidence[case]=proof
        if len(evidence)!=61 or set(verified.reused)!=set(verified.cases):raise ValueError('CONTINUATION_SCOPE')
        io.freeze(folder/'continuation-evidence.json',dict(state='COMPLETE_JOURNAL_CONTINUATION',
            cases_n=len(evidence),reused_cases=verified.reused,recomputed_cases=recomputed,journals=evidence,
            economic_registration_sha256=io.file_sha(child/'registration.json'),
            original_parent_stays_incomplete=True,financial_certification=False,promotion_allowed=False))
        io.write_json(folder/'status.json',dict(phase='COMPLETE_CONDITIONAL_SIMULATION',cases_n=61,
            reused_cases_n=len(verified.reused),recomputed_cases_n=len(recomputed),promotion_allowed=False))
        return report
    except BaseException as error:
        incident=dict(exception=type(error).__name__,detail=str(error),at_ms=time.time_ns()//1000000,promotion_allowed=False)
        io.write_json(folder/'incident.json',incident);io.write_json(folder/'status.json',dict(phase='INCOMPLETE',**incident))
        if child.exists():io.write_json(child/'status.json',dict(phase='INCOMPLETE',**incident))
        raise
    finally:study.gzip=original_gzip;study.simulate=original_simulate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('parent','rfd','capture','output','truth-report'):parser.add_argument('--'+arg,required=True)
    args=parser.parse_args()
    from .public_flow_dataset import own_runtime_path
    app=Path(__file__).resolve().parents[2]
    values=[own_runtime_path(app,getattr(args,name)) for name in ('parent','rfd','capture','output','truth_report')]
    parent,rfd,capture,folder,truth=values
    if any(folder==p or folder.is_relative_to(p) or p.is_relative_to(folder) for p in (parent,rfd,capture,truth)):
        raise ValueError('OVERLAPPING_PATHS')
    report=run(parent,rfd,capture,folder,truth)
    print(json.dumps(dict(state=report['state'],cases_n=len(report['cases']),promotion_allowed=False)),flush=True)


if __name__=='__main__':main()
