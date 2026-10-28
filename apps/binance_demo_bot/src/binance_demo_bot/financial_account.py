"""P0-C2: pinned seed, source-backed accounting imports and residual evidence.

No exchange transport, OMS ownership, certified financial cut or order authority.
"""
from contextlib import closing
from copy import deepcopy
from decimal import Context, Decimal, localcontext
import hashlib
import json
from pathlib import Path
import sqlite3

from .account_evidence import AccountEvidenceStore
from .account_opening import AccountOpeningManifest, _balances, _identifier, _strings, _source, _asset, _market_symbol
from .adapter import AdapterError, _sha
from .financial import AccountingError, LedgerConflict
from .interaction_audit import AuditJournal, sanitize
from .journal import JournalStore, JournalIntegrityError, WriterBusy, _accounting_code_hash, _digest


def _fail(reason):
    raise AdapterError(reason)


class FinancialAccount:
    def __init__(self,paths,*,opening_manifest,candidate_id,database='financial-account-audit.sqlite',
                 ledger_database='account-ledger.sqlite',fault_hook=None,ledger_fault_hook=None):
        self._audit=self._ledger=None
        self._failed=False
        self._binding=None
        self._ready_record=False
        self._imports={}
        self._checks={}
        self.paths,self.opening=paths,opening_manifest
        self._source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        try:
            if not isinstance(opening_manifest,AccountOpeningManifest):_fail('FINANCE_OPENING_INVALID')
            with opening_manifest._journal._lock:
                opening_manifest._journal.manifest()
                summary=opening_manifest.summary()
                seed=summary['candidates'].get(candidate_id)
                if (not seed or seed['state']!='PROVISIONAL_OPENING' or seed['reasons']
                        or seed['valuation']['status']!='COMPLETE' or seed['execution_authorized'] is not False
                        or any(lot['ownership']!='SEED' for lot in seed['seed_lots'])):
                    _fail('FINANCE_OPENING_INVALID')
                namespace=opening_manifest._journal.identity
                if not opening_manifest._journal.path.resolve().is_relative_to(paths.runtime_path('x').parent):
                    _fail('FINANCE_OPENING_INVALID')
            ledger_path=paths.runtime_path(ledger_database)
            if ledger_path==paths.runtime_path(database) or ledger_path==opening_manifest._journal.path:
                _fail('FINANCE_OPENING_INVALID')
            self._audit=AuditJournal(paths,account_alias=namespace['account_alias'],
                episode_id=namespace['account_episode_id'],scope_id=namespace['portfolio_scope_id'],
                database=database,max_bytes=64*1024*1024,fault_hook=fault_hook)
            with self._audit._lock:
                count=self._audit._con.execute("SELECT COUNT(*) FROM records WHERE kind='DECISION'").fetchone()[0]
                if count>31002:_fail('FINANCE_CAPACITY')
                for kind,raw in self._audit._con.execute('SELECT kind,payload_json FROM records ORDER BY seq'):
                    if kind=='DECISION':self._apply(json.loads(raw)['data'])
                    elif kind not in ('RUN_BEGIN','RUN_END'):_fail('FINANCE_INTEGRITY')
                binding=dict(opening_sha256=seed['manifest_sha256'],candidate_id=candidate_id,
                    account_identity=seed['account_identity'],opening=seed,ledger_database=ledger_database,
                    accounting_code_sha256=_accounting_code_hash())
                if self._binding is None:
                    if ledger_path.exists() or self._audit._last_seq!=1:_fail('FINANCE_INTEGRITY')
                    self._save('BIND',binding)
                elif self._binding!=binding:_fail('FINANCE_BINDING_CONFLICT')
                if self._ready_record and not ledger_path.is_file():_fail('FINANCE_LEDGER_MISSING')
                if self._ready_record:
                    # The frozen core may initialize a truly new file. A ready
                    # binding must first prove an existing identity read-only.
                    try:
                        with closing(sqlite3.connect(ledger_path.as_uri()+'?mode=ro',uri=True)) as con:
                            if con.execute("SELECT value FROM meta WHERE key='identity'").fetchone() is None:
                                _fail('FINANCE_INTEGRITY')
                    except sqlite3.Error:_fail('FINANCE_INTEGRITY')
                self._ledger=JournalStore(paths,episode_id=namespace['account_episode_id'],
                    scope_id=namespace['portfolio_scope_id'],opening_lots=seed['seed_lots'],
                    database=ledger_database,fault_hook=ledger_fault_hook)
                if not self._ready_record:
                    if self._ledger.events():_fail('FINANCE_INTEGRITY')
                    self._save('LEDGER_READY',dict(opening_sha256=binding['opening_sha256']))
                for key,entry in list(self._imports.items()):
                    if entry['outcome'] is None:self._execute(key)
        except BaseException as error:
            self.close()
            if isinstance(error,WriterBusy):raise AdapterError('FINANCE_BUSY') from None
            if isinstance(error,JournalIntegrityError):raise AdapterError('FINANCE_INTEGRITY') from None
            raise

    def _ready(self):
        if self._audit is None or self._audit._closed or self._audit._state!='READY' or self._failed:
            _fail('FINANCE_UNAVAILABLE')

    def _apply(self,event):
        try:
            if (set(event)!={'operation','version','source_code_sha256','value','value_sha256'}
                    or event['version']!=1 or event['value_sha256']!=_sha(event['value'])):
                _fail('FINANCE_INTEGRITY')
            operation,value=event['operation'],event['value']
            if operation=='BIND' and self._binding is None:
                self._binding=deepcopy(value)
            elif operation=='LEDGER_READY' and self._binding and not self._ready_record:
                if value!={'opening_sha256':self._binding['opening_sha256']}:_fail('FINANCE_INTEGRITY')
                self._ready_record=True
            elif operation=='IMPORT_BEGIN' and self._ready_record:
                key=value['import_id']
                if key in self._imports or key!=_digest(value['recipe']) or len(self._imports)>=10000:
                    _fail('FINANCE_INTEGRITY')
                self._imports[key]=dict(recipe=deepcopy(value['recipe']),outcome=None)
            elif operation=='IMPORT_END':
                entry=self._imports[value['import_id']]
                if entry['outcome'] is not None or value['outcome'] not in ('APPLIED','DUPLICATE','ACCOUNTING_REJECTED','FILL_CONFLICT'):
                    _fail('FINANCE_INTEGRITY')
                entry['outcome']=value['outcome']
            elif operation=='CHECK' and self._ready_record:
                key=value['check_id']
                if key in self._checks or len(self._checks)>=1000:_fail('FINANCE_INTEGRITY')
                if (value['daily_net_pnl_usdt'] is not None or value['execution_authorized'] is not False
                        or value['exchange_reconciled'] is not False or value['financial_episode_opened'] is not False):
                    _fail('FINANCE_INTEGRITY')
                self._checks[key]=deepcopy(value)
            else:_fail('FINANCE_INTEGRITY')
        except (KeyError,TypeError,ValueError):_fail('FINANCE_INTEGRITY')

    def _save(self,operation,value):
        value=_strings(value)
        event=dict(operation=operation,version=1,source_code_sha256=self._source,value=value,value_sha256=_sha(value))
        if sanitize(event)!=event:_fail('FINANCE_SOURCE_INVALID')
        self._audit.append('DECISION',event)
        self._apply(event)
        return deepcopy(value)

    def _execute(self,key):
        recipe=self._imports[key]['recipe']
        try:
            outcome=self._ledger.append_fill(recipe['envelope'],recipe['fill'],recipe['marks'])
        except LedgerConflict:outcome='FILL_CONFLICT'
        except (AccountingError,ValueError):outcome='ACCOUNTING_REJECTED'
        except Exception:
            self._failed=True
            self._audit._failed('FINANCE_JOURNAL_UNKNOWN')
            _fail('FINANCE_JOURNAL_UNKNOWN')
        self._save('IMPORT_END',dict(import_id=key,outcome=outcome))
        return outcome

    def _prefix(self,evidence):
        if not isinstance(evidence,AccountEvidenceStore):_fail('FINANCE_EVIDENCE_INVALID')
        expected=dict(self._audit.identity,account_identity=self._binding['account_identity'])
        if (evidence.identity!=expected or not evidence.path.resolve().is_relative_to(self.paths.runtime_path('x').parent)):
            _fail('FINANCE_EVIDENCE_INVALID')
        try:
            with evidence._lock:
                evidence._verify()
                return dict(generation=evidence.generation(),rows=evidence._rows(),fills=evidence.fills(),
                            incidents=evidence.incidents())
        except Exception:_fail('FINANCE_EVIDENCE_INVALID')

    def _metadata_sources(self,metadata,history,cutoff):
        try:
            if not isinstance(history,(list,tuple)) or len(history)>64:_fail('FINANCE_SOURCE_INVALID')
            sources=[];mappings={}
            for observation in [metadata,*history]:
                _source(observation,'exchangeInfo',cutoff,2**63-1,{},private=False)
                members={}
                for member in observation['data']['symbols']:
                    symbol=_market_symbol(member['symbol']);mapping=(_asset(member['baseAsset']),_asset(member['quoteAsset']))
                    if symbol in members:_fail('FINANCE_SOURCE_INVALID')
                    members[symbol]=member
                    key=(observation['received_at_ms'],symbol)
                    if key in mappings and mappings[key]!=mapping:_fail('FINANCE_METADATA_CONFLICT')
                    mappings[key]=mapping
                sources.append((observation,members))
            return sorted(sources,key=lambda r:(r[0]['received_at_ms'],r[0]['source_sha256']),reverse=True)
        except AdapterError as error:
            if error.reason=='FINANCE_METADATA_CONFLICT':raise
            _fail('FINANCE_SOURCE_INVALID')
        except (KeyError,TypeError,ValueError,ArithmeticError):_fail('FINANCE_SOURCE_INVALID')

    def _import(self,prefix,metadata_sources,cutoff):
        rows={row['seq']:row for row in prefix['rows']}
        counts=dict(stored_fills_N=len(prefix['fills']),imported_n=0,unimported_n=0,pre_anchor_n=0,
                    deferred_n=0,global_fill_denominator=None,dispositions=[])
        reasons=[]
        accepted={(entry['recipe']['fill']['symbol'],entry['recipe']['fill']['trade_id'],
                   entry['recipe']['source_fill_sha256']) for entry in self._imports.values()
                  if entry['outcome'] in ('APPLIED','DUPLICATE')}
        for item in sorted(prefix['fills'],key=lambda r:(r['data']['exchange_event_time_ms'],r['data']['symbol'],r['data']['trade_id'])):
            fill=item['data'];source=rows[item['record_seq']]
            disposition=dict(symbol=fill['symbol'],trade_id=fill['trade_id'],source_event_id=source['id'])
            at=fill['exchange_event_time_ms']
            if at<=self._binding['opening']['cutoff_ms']:
                counts['pre_anchor_n']+=1;disposition['state']='PRE_ANCHOR'
            elif at>cutoff or source['received_at_ms']>cutoff:
                counts['deferred_n']+=1;disposition['state']='DEFERRED';reasons.append('FUTURE_FILL_OR_RECEIPT')
            elif (fill['symbol'],fill['trade_id'],item['source_sha256']) in accepted:
                counts['imported_n']+=1;disposition['state']='IMPORTED';reasons.append('UNATTRIBUTED_ACTIVITY')
            else:
                selected=next(((source,members[fill['symbol']]) for source,members in metadata_sources
                    if source['received_at_ms']<=at and fill['symbol'] in members),None)
                metadata,member=selected if selected else (None,None)
                reason=None
                if not member:reason='HISTORICAL_SYMBOL_MAPPING_MISSING'
                elif member['quoteAsset']!='USDT':reason='NON_USDT_FILL_UNSUPPORTED'
                elif Decimal(fill['commission_quantity'])!=0 and fill['commission_asset'] not in ('USDT',member['baseAsset']):
                    reason='HISTORICAL_FEE_MARK_MISSING'
                if reason:
                    counts['unimported_n']+=1;reasons.append(reason);disposition.update(state='UNIMPORTED',reason=reason)
                else:
                    payload={k:fill[k] for k in ('symbol','trade_id','order_id','side','quantity','quote_quantity',
                        'price','commission_asset','commission_quantity','exchange_event_time_ms','ownership')}
                    payload.update(application_id='binance_demo_bot',environment='BINANCE_SPOT_DEMO',
                        episode_id=self._audit.identity['account_episode_id'],base_asset=member['baseAsset'],quote_asset='USDT',
                        client_order_id=source['data'].get('order',{}).get('client_order_id'))
                    envelope=dict(self._audit.identity,event_type='ExchangeFillRecorded',
                        event_id=_digest(dict(source_event_id=source['id'],fill_sha256=item['source_sha256'])),
                        process_instance_id=source['process_instance_id'],exchange_event_time_ms=at,
                        received_at_ms=source['received_at_ms'],received_monotonic_ns=source['received_monotonic_ns'],
                        source_sha256=source['source_sha256'],causation_id=source['id'])
                    recipe=dict(fill=payload,envelope=envelope,marks={'USDT':'1',member['baseAsset']:fill['price']},
                        metadata_sha256=metadata['source_sha256'],source_fill_sha256=item['source_sha256'])
                    recipe=_strings(recipe);key=_digest(recipe)
                    if key not in self._imports:
                        if len(self._imports)>=10000:_fail('FINANCE_CAPACITY')
                        self._save('IMPORT_BEGIN',dict(import_id=key,recipe=recipe))
                    outcome=self._imports[key]['outcome'] or self._execute(key)
                    if outcome in ('APPLIED','DUPLICATE'):
                        counts['imported_n']+=1;disposition['state']='IMPORTED'
                    else:
                        counts['unimported_n']+=1;disposition.update(state='UNIMPORTED',reason=outcome);reasons.append(outcome)
                reasons.append('UNATTRIBUTED_ACTIVITY')
            counts['dispositions'].append(disposition)
        return counts,reasons

    def reconcile(self,*,check_id,account_before,account_after,orders,order_lists,metadata,tickers,cutoff_ms,evidence,metadata_history=()):
        with self._audit._lock:
            self._ready();_identifier(check_id)
            try:
                with self.opening._journal._lock,localcontext(Context(prec=96)):
                    self.opening.summary()
                    observed=self.opening._build(check_id,account_before,account_after,orders,order_lists,metadata,tickers,cutoff_ms)
                    if account_before['request_started_at_ms']<=self._binding['opening']['cutoff_ms']:
                        _fail('FINANCE_SOURCE_INVALID')
                    returned=_balances(account_after['data'],self._binding['account_identity'])
            except (AdapterError,KeyError,TypeError,ValueError,ArithmeticError):_fail('FINANCE_SOURCE_INVALID')
            metadata_sources=self._metadata_sources(metadata,metadata_history,cutoff_ms)
            prefix=self._prefix(evidence)
            request=dict(observed=observed,evidence_generation=prefix['generation'])
            if metadata_history:request['historical_metadata_pins']=[_sha(_strings(row)) for row in metadata_history]
            request_hash=_sha(_strings(request))
            if check_id in self._checks:
                previous=self._checks[check_id]
                if previous['request_sha256']!=request_hash:_fail('FINANCE_CHECK_CONFLICT')
                return deepcopy(previous)
            if len(self._checks)>=1000:_fail('FINANCE_CAPACITY')
            with localcontext(Context(prec=96)):
                fills,import_reasons=self._import(prefix,metadata_sources,cutoff_ms)
                marks=observed['valuation']['marks']
                projection=self._ledger._cache(self._ledger._event_rows())
                if all(Decimal(q)==0 or asset=='USDT' or asset in marks for asset,q in projection['balances'].items()):
                    projection=self._ledger.projection(marks)
                required={a for a,q in projection['balances'].items() if Decimal(q)!=0}
                required.update(a for a,row in returned.items() if Decimal(row['free'])+Decimal(row['locked'])!=0)
                assets=[]
                for asset in sorted(required):
                    expected=Decimal(projection['balances'].get(asset,'0'))
                    actual=None if asset not in returned else Decimal(returned[asset]['free'])+Decimal(returned[asset]['locked'])
                    delta=None if actual is None else actual-expected
                    reason='ASSET_COVERAGE_MISSING' if actual is None else 'BALANCE_MISMATCH' if delta!=0 else None
                    assets.append(dict(asset=asset,journal_quantity=expected,exchange_quantity=actual,delta=delta,reason=reason))
                reasons=observed['reasons']+import_reasons+[r['data']['reason'] for r in prefix['incidents']]
                reasons.extend(r['reason'] for r in assets if r['reason'])
                if evidence.generation()!=prefix['generation']:reasons.append('EVIDENCE_CHANGED')
                reasons+=['CERTIFIED_CUT_PENDING','FILL_FLOW_COVERAGE_PENDING','PRIVATE_DELIVERY_PENDING']
                unknown=any(row['exchange_quantity'] is None for row in assets)
                different=any(row['delta'] is not None and row['delta']!=0 for row in assets)
                equity=observed['valuation']['equity_usdt']
                result=dict(check_id=check_id,request_sha256=request_hash,opening_sha256=self._binding['opening_sha256'],
                    cutoff_ms=cutoff_ms,status='MISMATCH' if different else 'PENDING',
                    balance_check='UNKNOWN' if unknown else 'BALANCES_DIFFER' if different else 'BALANCES_MATCH',
                    assets=assets,matched_assets_n=sum(row['reason'] is None for row in assets),assets_N=len(assets),
                    fills=fills,evidence_generation=prefix['generation'],evidence_prefix_sha256=_digest(prefix),
                    journal_source_sha256=projection['source_sha256'],source_pins=observed['sources'],
                    returned_assets_N=len(returned),returned_asset_sha256=sorted(_sha(a) for a in returned),
                    reasons=sorted(set(reasons)),
                    valuation=observed['valuation'],observed_equity_usdt=equity,unadjusted_equity_change_usdt=None if equity is None else
                    equity-Decimal(self._binding['opening']['valuation']['equity_usdt']),
                    daily_net_pnl_usdt=None,exchange_reconciled=False,financial_episode_opened=False,execution_authorized=False)
            return self._save('CHECK',result)

    def interval_report(self,check_id):
        with self._audit._lock,localcontext(Context(prec=96)):
            self._ready()
            if check_id not in self._checks:_fail('FINANCE_CHECK_MISSING')
            check=self._checks[check_id];reasons=list(check['reasons'])+['CALENDAR_BOUNDARIES_PENDING']
            result=dict(check_id=check_id,opening_sha256=self._binding['opening_sha256'],
                cutoff_ms=check['cutoff_ms'],evidence_kind='LOCAL_ACCOUNT_DIAGNOSTIC',financial_status='PENDING',
                attribution_status='BLOCKED',observed_equity_usdt=check['observed_equity_usdt'],
                projected_equity_usdt=None,projected_equity_change_usdt=None,attribution=None,
                attribution_residual_usdt=None,daily_net_pnl_usdt=None,execution_authorized=False,reasons=reasons)
            marks=check.get('valuation',{}).get('marks',{})
            cached=self._ledger._cache(self._ledger._event_rows())
            if cached['source_sha256']!=check['journal_source_sha256']:
                reasons.append('JOURNAL_CHANGED');return _strings(result)
            if (check['balance_check']!='BALANCES_MATCH' or check['observed_equity_usdt'] is None
                or check['fills']['unimported_n'] or check['fills']['deferred_n']):return _strings(result)
            try:
                projection=self._ledger.projection(marks)
                equity=sum((Decimal(q)*Decimal(marks[a]) for a,q in projection['balances'].items() if Decimal(q)),Decimal(0))
            except (KeyError,ValueError,ArithmeticError):
                reasons.append('VALUATION_INCOMPLETE');return _strings(result)
            change=equity-Decimal(self._binding['opening']['valuation']['equity_usdt'])
            attribution=projection['attribution']
            residual=change-(attribution['gross_realized_usdt']-attribution['fee_expenses_usdt']+attribution['unrealized_usdt'])
            if equity!=Decimal(check['observed_equity_usdt']) or residual!=0:
                reasons.append('ATTRIBUTION_MISMATCH');return _strings(result)
            result.update(attribution_status='MATCH',projected_equity_usdt=equity,
                projected_equity_change_usdt=change,attribution=attribution,attribution_residual_usdt=residual)
            return _strings(result)

    def projection(self,marks):
        with self._audit._lock:
            self._ready()
            return self._ledger.projection(marks)

    def summary(self):
        with self._audit._lock:
            self._ready()
            return dict(opening_sha256=self._binding['opening_sha256'],checks_n=len(self._checks),
                        imports_n=len(self._imports),checks=deepcopy(self._checks))

    def close(self):
        if self._ledger is not None:self._ledger.close()
        if self._audit is not None:self._audit.close()
