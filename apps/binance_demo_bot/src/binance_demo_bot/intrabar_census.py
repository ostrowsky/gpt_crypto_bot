"""IC-1: complete sealed EOS ambiguity population, public Demo minute evidence."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path
import time

from .intrabar_data import original_bar, _row
from .intrabar_probe import (_single_writer, _seal, _read_sealed, _bytes_sha, _write,
    _parent_binding, collect_probe_run, finalize_probe_run, PacedReadTransport, ENDPOINT)
from .execution_order_study import load_report, verify_files


def collector_registration(value):
    contract='ASSUMED_NEAREST_ROUNDING_CAPPED_ORIGINAL_QUANTUM_1e-8'
    if value.get('rounding_contract',contract)!=contract:raise ValueError('CENSUS_ROUNDING_CONTRACT')
    return dict(value,rounding_contract=contract)


def bind_cases(task,detail):
    if detail['state']!='COMPLETE' or detail['ambiguous_executions_n']!=len(detail['ambiguities']):
        raise ValueError('CENSUS_PARENT_INCOMPLETE_OR_COUNT')
    result=[]
    for index,ambiguity in enumerate(detail['ambiguities']):
        matches=[t for t in detail['trades'] if t['symbol']==ambiguity['symbol'] and
                 t['entry_ms']<=ambiguity['origin_ms']<=t['exit_ms']]
        if len(matches)!=1: raise ValueError('CENSUS_TRADE_BINDING')
        trade=matches[0]; entry=trade['entry_price']
        reason='TP' if task['order']=='TARGET_FIRST' else 'SL'
        if (ambiguity['stop']!=entry*.995 or ambiguity['target']!=entry*1.005 or
            ambiguity['chosen_reason']!=reason or trade['reason']!=reason):
            raise ValueError('CENSUS_LEVEL_OR_CONVENTION_BINDING')
        result.append(dict(case_id=f"{task['id']}:{index}",portfolio_id=task['id'],
            **{k:task[k] for k in ('order','segment','cost','label')},**ambiguity,
            entry_ms=trade['entry_ms'],entry_price=entry))
    return result


def first_touch(rows,stop,target,*,resolution_ms=60000,verified=True):
    unknown=dict(state='ORDER_UNRESOLVED_DATA',bucket_origin_ms=None,resolution_ms=resolution_ms,trigger=None)
    if not verified or not rows: return unknown
    try:
        low,high=Decimal(str(stop)),Decimal(str(target))
        if not low.is_finite() or not high.is_finite() or not 0<low<high: return unknown
        parsed=[_row(r,resolution_ms,rows[0][0]+i*resolution_ms) for i,r in enumerate(rows)]
    except (ValueError,TypeError,KeyError,ArithmeticError): return unknown
    for r in parsed:
        state,trigger=None,None
        if r[1]<=low: state,trigger='SL_FIRST','OPEN'
        elif r[1]>=high: state,trigger='TP_FIRST','OPEN'
        elif r[3]<=low and r[2]>=high: state,trigger='ORDER_UNRESOLVED_SAME_BUCKET','HIGH_LOW'
        elif r[3]<=low: state,trigger='SL_FIRST','LOW'
        elif r[2]>=high: state,trigger='TP_FIRST','HIGH'
        if state: return dict(state=state,bucket_origin_ms=r[0],resolution_ms=resolution_ms,trigger=trigger)
    return dict(unknown,state='ORDER_UNRESOLVED_NO_TOUCH')


def census_counts(cases):
    counts=Counter(r['ordering']['state'] for r in cases)
    resolved=counts['SL_FIRST']+counts['TP_FIRST']
    return dict(cases_n=len(cases),resolved_n=resolved,unresolved_n=len(cases)-resolved,
        sl_first_n=counts['SL_FIRST'],tp_first_n=counts['TP_FIRST'],states=dict(counts),
        contradicted_n=sum(r['ordering']['state'] in ('SL_FIRST','TP_FIRST') and
            r['chosen_reason']!=('SL' if r['ordering']['state']=='SL_FIRST' else 'TP') for r in cases))


@_single_writer
def register_census(folder,eos_folder,truth_report):
    eos_folder=Path(eos_folder).resolve(); truth_report=Path(truth_report).resolve()
    eos=_read_sealed(eos_folder/'registration.json')
    report=json.loads((eos_folder/'report.json').read_text(encoding='utf-8'))
    status=json.loads((eos_folder/'status.json').read_text(encoding='utf-8'))
    if (len(eos['plan'])!=168 or report['financial_passes_n']!=168 or report['baseline_parity_n']!=156 or
        status['phase']!='COMPLETE_COUNTERFACTUAL_STUDY' or eos['symbols_N']!=519):
        raise ValueError('CENSUS_EOS_SCOPE')
    verify_files(eos['files'])
    parent_files={str(eos_folder/name):_bytes_sha(eos_folder/name)
                  for name in ('registration.json','registration.json.sha256','report.json','status.json')}
    cases=[]; registration_sha=_bytes_sha(eos_folder/'registration.json')
    for task in eos['plan']:
        detail=load_report(eos_folder,task,registration_sha)
        if detail is None: raise ValueError('CENSUS_PORTFOLIO_MISSING')
        if detail['state']!='COMPLETE' or detail['eos_financial_check']['state']!='PASS':
            raise ValueError('CENSUS_PORTFOLIO_FINANCIAL_CHECK')
        if detail.get('ambiguities'): cases.extend(bind_cases(task,detail))
        path=eos_folder/'portfolios'/(task['id']+'.json.gz')
        for p in (path,path.with_suffix('.meta.json'),path.with_suffix('.meta.json.sha256')):
            parent_files[str(p)]=_bytes_sha(p)
        del detail
    unique=sorted({(c['symbol'],c['origin_ms']) for c in cases})
    if len(unique)!=613 or len(cases)!=1457 or len({s for s,_ in unique})!=223:
        raise ValueError('CENSUS_EXPECTED_POPULATION_CHANGED')
    rde=Path(eos['parent_folder']); prepared=Path(eos['prepared_folder'])/'manifest.json'
    manifest=json.loads(prepared.read_text(encoding='utf-8')); archive=Path(manifest['archive']).resolve()
    parent=_parent_binding(archive,rde,prepared)
    windows=[dict(symbol=symbol,origin_ms=at,seconds_required=False,**original_bar(archive,symbol,at)) for symbol,at in unique]
    spec=Path(__file__).resolve().parents[4]/'docs/specs/binance-demo-intrabar-census.md'
    audit_spec=spec.with_name('binance-demo-public-request-commitments.md')
    parent_files[str(audit_spec)]=_bytes_sha(audit_spec)
    truth=json.loads(truth_report.read_text(encoding='utf-8'))
    if truth['profile']!='full': raise ValueError('CENSUS_FULL_TRUTH_REQUIRED')
    registration=dict(protocol='IC-1-MINUTE-CENSUS-20261105',endpoint=ENDPOINT,archive=str(archive),
        rde_folder=str(rde),prepared_manifest_path=str(prepared),prepared_manifest_sha256=_bytes_sha(prepared),
        parent_registration_sha256=_bytes_sha(rde/'registration.json'),
        frozen_encoder_sha256=_bytes_sha(rde/'frozen-encoder.json'),parent_data_sha256=parent['data_sha256'],
        spec_path=str(spec),spec_sha256=_bytes_sha(spec),
        source_sha256={name:_bytes_sha(Path(__file__).with_name(name)) for name in
            ('intrabar_census.py','intrabar_probe.py','intrabar_data.py','intrabar_audit.py','public_research_audit.py','interaction_audit.py')},
        eos_folder=str(eos_folder),eos_registration_sha256=registration_sha,
        census_parent_files=parent_files,eos_input_files=eos['files'],
        cohort=parent['symbols'],splits=parent['splits'],windows=windows,cases=cases,
        truth_harness_at_registration=dict(path=str(truth_report),sha256=_bytes_sha(truth_report),status=truth['status'].upper()),
        evidence_kind='SIMULATION',holdout_kind='RETROSPECTIVE_REUSED_HOLDOUT',
        promotion_allowed=False,actual_execution_reconstructed=False,full_replay_complete=False)
    registration=collector_registration(registration)
    _seal(Path(folder)/'registration.json',registration)
    return registration


@_single_writer
def render_census(folder):
    folder=Path(folder); registered=_read_sealed(folder/'registration.json')
    verify_files(registered['census_parent_files']); verify_files(registered['eos_input_files'])
    source=json.loads((folder/'source-check.json').read_text(encoding='utf-8'))
    audit=json.loads((folder/'audit-check.json').read_text(encoding='utf-8'))
    checks={(r['symbol'],r['origin_ms']):r for r in source['results'] if r['interval']=='1m'}
    verified=audit['state']=='VERIFIED_RECORDED_PUBLIC_ATTEMPTS'; results=[]
    for case in registered['cases']:
        check=checks[case['symbol'],case['origin_ms']]
        accepted=verified and check['state']=='MATCH'
        response=_read_sealed(check['response_file']) if accepted else None
        ordering=first_touch(response['rows'] if accepted else [],case['stop'],case['target'],verified=accepted)
        results.append(dict(case,ordering=ordering,source_state=check['state'],source_kind='BINANCE_DEMO_1m',
            response_file=check.get('response_file'),source_sha256=check.get('source_sha256')))
    groups=defaultdict(list)
    for row in results: groups[row['segment'],row['cost'],row['order'],row['label']].append(row)
    grouped=[dict(segment=k[0],cost=k[1],order=k[2],label=k[3],**census_counts(v)) for k,v in sorted(groups.items())]
    report=dict(protocol=registered['protocol'],state='MINUTE_DATA_COMPLETE' if verified and source['match_n']==len(registered['windows']) else 'INCOMPLETE_SOURCE',
        **census_counts(results),windows_n=len(registered['windows']),symbols_n=223,matched_windows_n=source['match_n'],
        native_provenance=audit['state'],cases=results,groups=grouped,
        full_truth_harness=registered['truth_harness_at_registration']['status'],
        evidence_kind='SIMULATION_DATA_DIAGNOSTIC',actual_execution_reconstructed=False,
        full_replay_complete=False,promotion_allowed=False)
    _seal(folder/'census.json',report)
    lines=['# IC-1: полный минутный аудит порядка исполнения','',
        f"**{report['state']}**. Окон {report['matched_windows_n']}/{report['windows_n']}; случаев {report['cases_n']}; разрешено {report['resolved_n']}; не разрешено {report['unresolved_n']}.",
        f"Источник: Binance Demo1m; provenance {audit['state']}. Полный Truth Harness: {report['full_truth_harness']}.",
        '', 'Все неоднозначности обоих EOS-сценариев, TUNE/TEST и издержек; отбора по результату нет.',
        '', '| Период | Издержки | Допущение | Метод | N | Стоп раньше | Цель раньше | Неясно | Противоречий допущению |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for g in grouped:
        lines.append(f"| {g['segment']} | {g['cost']} | {g['order']} | {g['label']} | {g['cases_n']} | {g['sl_first_n']} | {g['tp_first_n']} | {g['unresolved_n']} | {g['contradicted_n']} |")
    lines+=['', 'Это диагностика данных и допущений. PnL не пересчитан; новое закрытие изменяет последующие входы.',
        'Оба уровня в одной минуте остаются неизвестными. Временная отметка — начало ведра, а не время исполнения ордера.',
        'Секундный официальный архив исследуется отдельно после проверки контрольных сумм и сверки с Demo1m.',
        'Ни прибыльность, ни готовность к торговле, ни старый RDE-2 gate этим отчётом не сертифицированы.']
    (folder/'findings-ru.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True);parser.add_argument('--eos-folder',required=True)
    parser.add_argument('--truth-report',required=True);args=parser.parse_args()
    app=Path(__file__).resolve().parents[2];folder=Path(args.folder).resolve()
    if not folder.is_relative_to(app/'.runtime/intrabar-execution') or not folder.name.startswith('IC-1-20261105'):
        raise SystemExit('OWN_CENSUS_RUNTIME_REQUIRED')
    from .adapter import DemoReadClient
    from .public_research_audit import PublicResearchJournal, verify_committed_probe_audit
    from .isolation import AppPaths
    def status(phase,**extra):
        value=dict(phase=phase,at_ms=time.time_ns()//1000000,promotion_allowed=False,**extra)
        _write(folder/'status.json',value);print(json.dumps(value),flush=True)
    try:
        folder.mkdir(parents=True,exist_ok=True); status('REGISTERING')
        register_census(folder,args.eos_folder,args.truth_report);status('COLLECTING_MINUTES',windows_n=613,cases_n=1457)
        audit=PublicResearchJournal(AppPaths(app),account_alias='public-research',episode_id=folder.name,
            scope_id='complete-intrabar-census',database=str((folder/'rest-audit.sqlite').relative_to(app/'.runtime')),max_bytes=256*1024*1024)
        client=DemoReadClient(audit=audit,transport=PacedReadTransport(deadline=time.monotonic()+3600))
        try: collect_probe_run(folder,client=client,max_attempts=2000,max_wall_s=3600,include_seconds=False)
        finally: client.close();audit.close()
        status('VERIFYING_NATIVE_SOURCE');finalize_probe_run(folder,folder/'rest-audit.sqlite',verifier=verify_committed_probe_audit)
        report=render_census(folder);status('COMPLETE_DATA_DIAGNOSTIC',state=report['state'],resolved_n=report['resolved_n'],unresolved_n=report['unresolved_n'])
    except BaseException as error:
        status('INCOMPLETE',error_kind=type(error).__name__);raise


if __name__=='__main__':main()
