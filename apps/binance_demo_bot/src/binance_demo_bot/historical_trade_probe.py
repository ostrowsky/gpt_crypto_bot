"""RDE-2C public historical-trade availability, coverage and ordering diagnostic.

No model fitting, portfolio repricing, credentials, orders or worker controls.
"""
import argparse
from decimal import Decimal
import json
from pathlib import Path

from .intrabar_data import validate_window
from .intrabar_probe import (_bytes_sha, _read_sealed, _seal, _sha, _single_writer,
                             _validate_binding, _write)
from .intrabar_audit import verify_probe_audit
from .historical_trade_source import TradeSourceError
from .historical_trade_validation import validate_trade_window, resolve_bracket


STEP=900000
ENDPOINT='https://demo-api.binance.com/api/v3/aggTrades'
WINDOWS=(('DEXEUSDT',1750797000000),('GMXUSDT',1752068700000),('NXPCUSDT',1749808800000))


def _parent_snapshot(parent_folder):
    parent_folder=Path(parent_folder).resolve()
    parent=_validate_binding(parent_folder)
    check=json.loads((parent_folder/'source-check.json').read_text(encoding='utf-8'))
    for source in check['results']:
        if 'response_file' in source:
            sealed=_read_sealed(source['response_file'])
            if sealed['source_sha256']!=source['source_sha256']:
                raise ValueError('HISTORICAL_PARENT_RESPONSE_BINDING')
    audit=verify_probe_audit(parent_folder,parent_folder/'rest-audit.sqlite')
    if (check['native_audit_certification']!='VERIFIED_RECORDED_PUBLIC_ATTEMPTS' or
            audit['state']!='VERIFIED_RECORDED_PUBLIC_ATTEMPTS'):
        raise ValueError('HISTORICAL_PARENT_AUDIT_BINDING')
    rde=Path(parent['rde_folder']); portfolios={}
    for model in ('CatBoost','GRU'):
        path=rde/'portfolios'/('test-base-'+model+'.json')
        portfolios[model]=(path,json.loads(path.read_text(encoding='utf-8')))
    windows=[]
    for symbol,at in WINDOWS:
        original=next(w for w in parent['windows'] if (w['symbol'],w['origin_ms'])==(symbol,at))
        reference=next(r for r in check['results'] if (r['symbol'],r['origin_ms'],r['interval'])==(symbol,at,'1m'))
        path=Path(reference['response_file']); minutes=_read_sealed(path)
        if (reference['state']!='MATCH' or minutes['source_sha256']!=reference['source_sha256'] or
            _sha(minutes['rows'])!=minutes['source_sha256'] or
            validate_window(original['row'],minutes['rows'],received_at_ms=minutes['metadata']['received_at_ms'])['state']!='MATCH'):
            raise ValueError('HISTORICAL_MINUTE_BINDING')
        brackets=[]
        for model,(portfolio_path,portfolio) in portfolios.items():
            for index,trade in enumerate(portfolio['trades']):
                if (trade['symbol'],trade['entry_ms'],trade['exit_ms'])!=(symbol,at,at): continue
                entry=trade['entry_price']; stop=Decimal(str(entry*.995)); target=Decimal(str(entry*1.005))
                if trade['reason']!='SL' or Decimal(original['row'][3])>stop or Decimal(original['row'][2])<target:
                    raise ValueError('HISTORICAL_ORIGINAL_BRACKET_BINDING')
                brackets.append(dict(model=model,portfolio_path=str(portfolio_path),
                    portfolio_sha256=_bytes_sha(portfolio_path),trade_index=index,trade=trade,trade_sha256=_sha(trade)))
        windows.append(dict(symbol=symbol,origin_ms=at,original=original,
            minutes=minutes,minute_response_path=str(path),minute_response_sha256=_bytes_sha(path),brackets=brackets))
    if sum(len(w['brackets']) for w in windows)!=4: raise ValueError('HISTORICAL_ORIGINAL_POPULATION_BINDING')
    return dict(parent_folder=str(parent_folder),parent_registration_sha256=_bytes_sha(parent_folder/'registration.json'),
        parent_source_check_sha256=_bytes_sha(parent_folder/'source-check.json'),
        parent_audit_sha256=_bytes_sha(parent_folder/'rest-audit.sqlite'),windows=windows)


@_single_writer
def register_trade_run(folder,*,parent_folder,spec_path,truth_report=None):
    spec_path=Path(spec_path).resolve()
    truth=dict(status='UNKNOWN',path=None,sha256=None)
    if truth_report is not None:
        path=Path(truth_report).resolve(); value=json.loads(path.read_text(encoding='utf-8'))
        if value.get('profile')!='full' or value.get('status') not in ('pass','fail','unknown'):
            raise ValueError('HISTORICAL_TRUTH_BINDING')
        truth=dict(status=value['status'].upper(),path=str(path),sha256=_bytes_sha(path))
    registered=dict(protocol='RDE-2C-historical-trade-resolution-v1',endpoint=ENDPOINT,
        parent=_parent_snapshot(parent_folder),spec_path=str(spec_path),spec_sha256=_bytes_sha(spec_path),
        source_sha256={name:_bytes_sha(Path(__file__).with_name(name)) for name in
            ('historical_trade_probe.py','historical_trade_source.py','historical_trade_validation.py')},
        truth_harness_at_registration=truth,evidence_kind='SIMULATION_POST_HOC_SOURCE_DIAGNOSTIC',
        promotion_allowed=False,pnl_repriced=False,full_portfolio_replayed=False)
    _seal(Path(folder)/'registration.json',registered)
    return dict(registered,windows=registered['parent']['windows'])


def _binding(folder):
    r=_read_sealed(Path(folder)/'registration.json')
    if (r['endpoint']!=ENDPOINT or _bytes_sha(r['spec_path'])!=r['spec_sha256'] or
        any(_bytes_sha(Path(__file__).with_name(k))!=v for k,v in r['source_sha256'].items()) or
        _parent_snapshot(r['parent']['parent_folder'])!=r['parent']):
        raise ValueError('HISTORICAL_BINDING_CHANGED')
    truth=r['truth_harness_at_registration']
    if truth['path'] is not None and _bytes_sha(truth['path'])!=truth['sha256']:
        raise ValueError('HISTORICAL_TRUTH_BINDING_CHANGED')
    return r


def _ordered_page(rows):
    if not isinstance(rows,list) or len(rows)>1000: return False
    previous=None
    for row in rows:
        if (not isinstance(row,dict) or type(row.get('a')) is not int or row['a']<0 or
            type(row.get('T')) is not int or row['T']<0 or
            previous is not None and (row['a']!=previous['a']+1 or row['T']<previous['T'])): return False
        previous=row
    return True


@_single_writer
def collect_trade_run(folder,*,client,max_pages=100):
    if type(max_pages) is not int or not 1<=max_pages<=100: raise ValueError('HISTORICAL_INVALID_PAGE_BUDGET')
    folder=Path(folder); r=_binding(folder); fetched=cached=0; outcomes=[]
    def read(params):
        nonlocal fetched,cached
        key=_sha(params); target=folder/'responses'/(key+'.json')
        if target.exists(): response=_read_sealed(target); cached+=1
        else:
            rows,meta=client.read(params)
            response=dict(endpoint=ENDPOINT,parameters=params,rows=rows,metadata=meta,source_sha256=_sha(rows))
            _seal(target,response); fetched+=1
        if (response['endpoint']!=ENDPOINT or response['parameters']!=params or
            _sha(response['rows'])!=response['source_sha256']): raise ValueError('HISTORICAL_RESPONSE_INTEGRITY')
        return response['rows']

    for window in r['parent']['windows']:
        symbol,at=window['symbol'],window['origin_ms']; end=at+STEP
        result=dict(symbol=symbol,origin_ms=at,state='INCOMPLETE',reasons=[],received_rows_n=0,
            brackets=[dict(model=b['model'],trade_sha256=b['trade_sha256'],state='ORDER_UNRESOLVED') for b in window['brackets']])
        try:
            first=read(dict(symbol=symbol,startTime=at,endTime=end-1,limit=1000))
            if not _ordered_page(first): result['reasons']=['TRADE_PAGE_SCHEMA_OR_ORDER']
            elif not first:
                result.update(availability='HISTORY_EMPTY_FOR_REQUESTED_WINDOW',reasons=['HISTORY_EMPTY_FOR_REQUESTED_WINDOW'])
            elif any(not at<=x['T']<end for x in first): result['reasons']=['INITIAL_TIME_QUERY_RANGE_CONTRADICTION']
            else:
                result['availability']='NONEMPTY_RESPONSE'
                before=read(dict(symbol=symbol,fromId=first[0]['a']-1,limit=1)) if first[0]['a'] else []
                previous=before[0] if isinstance(before,list) and len(before)==1 else None
                trades=list(first); following=None; pages=1; error=None
                while pages<max_pages and len(trades)<100000:
                    page=read(dict(symbol=symbol,fromId=trades[-1]['a']+1,limit=1000)); pages+=1
                    if not _ordered_page(page): error='TRADE_PAGE_SCHEMA_OR_ORDER'; break
                    if not page: error='FOLLOWING_GUARD_MISSING'; break
                    if page[0]['a']!=trades[-1]['a']+1 or page[0]['T']<trades[-1]['T']:
                        error='TRADE_PAGE_BOUNDARY_GAP_OR_ORDER'; break
                    for row in page:
                        if row['T']>=end: following=row; break
                        trades.append(row)
                    if following is not None: break
                result.update(received_rows_n=len(trades),data_pages_n=pages,
                    previous_guard=previous,following_guard=following)
                if error: result['reasons']=[error]
                elif following is None: result['reasons']=['DATA_PAGE_OR_ROW_BUDGET']
                else:
                    validation=validate_trade_window(window['original']['row'],window['minutes']['rows'],trades,
                        previous=previous,following=following,received_at_ms=window['minutes']['metadata']['received_at_ms'])
                    result['validation']=validation; result['reasons']=validation['reasons']
                    result['state']=('SOURCE_INCOMPATIBLE' if validation['state']=='MISMATCH' else
                                     'COMPLETE' if validation['state']=='MATCH' else 'INCOMPLETE')
                    result['brackets']=[dict(model=b['model'],trade_sha256=b['trade_sha256'],
                        **resolve_bracket(validation,trades,entry_price=b['trade']['entry_price'])) for b in window['brackets']]
        except TradeSourceError as error:
            result.update(state='UNKNOWN',availability='API_OR_NETWORK_ERROR',reasons=[error.reason],
                          http_status=getattr(error,'status',None),api_code=getattr(error,'api_code',None))
        outcomes.append(result)
    resolved=sum(b['state']!='ORDER_UNRESOLVED' for w in outcomes for b in w['brackets'])
    state=('SOURCE_INCOMPATIBLE' if any(w['state']=='SOURCE_INCOMPATIBLE' for w in outcomes) else
           'UNKNOWN' if any(w['state']=='UNKNOWN' for w in outcomes) else
           'INCOMPLETE' if any(w['state']!='COMPLETE' for w in outcomes) else 'SOURCE_GATE_ONLY_COMPLETE')
    report=dict(state=state,registered_windows_N=3,original_trade_bindings_N=4,
        matched_windows_n=sum(w['state']=='COMPLETE' for w in outcomes),
        history_empty_n=sum(w.get('availability')=='HISTORY_EMPTY_FOR_REQUESTED_WINDOW' for w in outcomes),
        api_error_n=sum(w['state']=='UNKNOWN' for w in outcomes),resolved_bindings_n=resolved,
        unresolved_bindings_n=4-resolved,received_rows_n=sum(w['received_rows_n'] for w in outcomes),
        fetched_n=fetched,cached_n=cached,windows=outcomes,
        truth_harness_at_registration=r['truth_harness_at_registration'],
        native_audit_certification='UNKNOWN_PENDING_FINALIZATION',
        evidence_kind=r['evidence_kind'],promotion_allowed=False,pnl_repriced=False,full_portfolio_replayed=False)
    _write(folder/'trade-check.json',report)
    _report(folder,report)
    return report


def _report(folder,report):
    lines=['# RDE-2C: historical Demo trade resolution','',f"Source state: **{report['state']}**.",
        f"Native recorded-attempt provenance: **{report['native_audit_certification']}**.",
        f"Resolved original trade bindings: {report['resolved_bindings_n']}/{report['original_trade_bindings_N']}.",
        '', 'Post-hoc source diagnostic only. Original portfolios/PnL unchanged.',
        '', '| Symbol | Origin ms | Rows | Availability | State | Reasons |','|---|---:|---:|---|---|---|']
    lines.extend(f"| {w['symbol']} | {w['origin_ms']} | {w['received_rows_n']} | {w.get('availability','')} | {w['state']} | {', '.join(w['reasons'])} |" for w in report['windows'])
    lines.extend(['',f"Full Truth Harness at registration: {report['truth_harness_at_registration']['status']}.",
        'No native historical fill, queue, latency or profitability certification.'])
    (Path(folder)/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


@_single_writer
def finalize_trade_run(folder,audit_path):
    from .historical_trade_source import verify_trade_audit
    _binding(folder)
    audit=verify_trade_audit(folder,audit_path); _write(Path(folder)/'audit-check.json',audit)
    report=json.loads((Path(folder)/'trade-check.json').read_text(encoding='utf-8'))
    report.update(native_audit_certification=audit['state'],audit_check_sha256=_sha(audit),
        overall_state=report['state'] if audit['state']=='VERIFIED_RECORDED_PUBLIC_ATTEMPTS' else 'UNKNOWN')
    _write(Path(folder)/'trade-check.json',report); _report(folder,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True); parser.add_argument('--parent-folder',required=True)
    parser.add_argument('--truth-report'); parser.add_argument('--register',action='store_true')
    parser.add_argument('--collect',action='store_true'); args=parser.parse_args()
    app=Path(__file__).resolve().parents[2]; folder=Path(args.folder).resolve()
    if not folder.is_relative_to(app/'.runtime/intrabar-execution'): raise SystemExit('OWN_INTRABAR_RUNTIME_REQUIRED')
    if args.register:
        register_trade_run(folder,parent_folder=args.parent_folder,
            spec_path=app.parents[1]/'docs/specs/binance-demo-historical-trade-resolution.md',truth_report=args.truth_report)
    if args.collect:
        from .historical_trade_source import HistoricalTradeClient
        from .interaction_audit import AuditJournal
        from .isolation import AppPaths
        audit=AuditJournal(AppPaths(app),account_alias='public-research',episode_id=folder.name,
            scope_id='historical-trade-source',database=str((folder/'rest-audit.sqlite').relative_to(app/'.runtime')),
            max_bytes=512*1024*1024)
        client=HistoricalTradeClient(audit)
        try: collect_trade_run(folder,client=client)
        finally: client.close(); audit.close()
        result=finalize_trade_run(folder,folder/'rest-audit.sqlite')
        print(json.dumps({k:v for k,v in result.items() if k!='windows'},indent=2))
    elif not args.register: raise SystemExit('REGISTER_OR_COLLECT_REQUIRED')


if __name__=='__main__': main()
