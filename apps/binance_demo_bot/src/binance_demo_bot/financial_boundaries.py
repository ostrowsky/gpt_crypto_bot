"""Native observed-response brackets; never a certified economic cut or sender."""
from copy import deepcopy
from decimal import Context,localcontext
import hashlib,json
from pathlib import Path
from .account_opening import AccountOpeningManifest,_identifier,_time,_strings,_asset,_market_symbol,_build_observation
from .adapter import AdapterError,_sha
from .financial import day_bounds,day_for_timestamp
from .interaction_audit import AuditJournal,sanitize

ZONE='Europe/Budapest'

def fail(reason):raise AdapterError(reason)

def encode_public(value):
    if isinstance(value,list):return [encode_public(v) for v in value]
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            if key in ('asset','symbol') and isinstance(item,str):
                (_asset if key=='asset' else _market_symbol)(item)
                result[key+'_points']=[ord(c) for c in item]
            else:result[key]=encode_public(item)
        return result
    return value

def decode_public(value):
    if isinstance(value,list):return [decode_public(v) for v in value]
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            if key in ('asset_points','symbol_points'):
                if not isinstance(item,list) or not 1<=len(item)<=64 or any(type(c) is not int for c in item):fail('FBO_INTEGRITY')
                try:public=''.join(chr(c) for c in item)
                except ValueError:fail('FBO_INTEGRITY')
                target=key[:-7];(_asset if target=='asset' else _market_symbol)(public)
                if target in value or target in result:fail('FBO_INTEGRITY')
                result[target]=public
            else:result[key]=decode_public(item)
        return result
    return value

def next_boundaries(now_ms,*,count=2):
    _time(now_ms)
    if type(count) is not int or not 1<=count<=2:fail('FBO_INVALID')
    result=[]
    for _ in range(count):
        boundary=day_bounds(day_for_timestamp(now_ms,ZONE),ZONE)[1]
        result.append(boundary);now_ms=boundary
    return result

def capture_slots(boundary_ms,now_ms):
    _time(boundary_ms);_time(now_ms)
    return [boundary_ms+offset for offset in (-6000,1000,10000) if boundary_ms+offset>=now_ms]

class FinancialBoundaryStore:
    def __init__(self,paths,*,opening_manifest,candidate_id,database='financial-boundaries/observations.sqlite'):
        self._audit=None;self._binding=None;self._plan=None;self._observations={}
        self._source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        try:
            if not isinstance(opening_manifest,AccountOpeningManifest):fail('FBO_OPENING_INVALID')
            with opening_manifest._journal._lock:
                opening_manifest._journal.manifest();summary=opening_manifest.summary();seed=summary['candidates'].get(candidate_id)
                if (not seed or seed['state']!='PROVISIONAL_OPENING' or seed['reasons'] or seed['valuation']['status']!='COMPLETE'
                    or not opening_manifest._journal.path.resolve().is_relative_to(paths.runtime_path('x').parent)):
                    fail('FBO_OPENING_INVALID')
                ns=opening_manifest._journal.identity
                self._namespace=deepcopy(ns);self._attestation=deepcopy(summary['attestation'])
            binding=dict(opening_sha256=seed['manifest_sha256'],account_identity=seed['account_identity'],
                candidate_id=candidate_id,timezone=ZONE,execution_authorized=False)
            self._audit=AuditJournal(paths,account_alias=ns['account_alias'],episode_id=ns['account_episode_id'],
                scope_id=ns['portfolio_scope_id'],database=database,max_bytes=64*1024*1024)
            with self._audit._lock:
                if self._audit._con.execute("SELECT COUNT(*) FROM records WHERE kind='DECISION'").fetchone()[0]>1026:fail('FBO_CAPACITY')
                for kind,raw in self._audit._con.execute('SELECT kind,payload_json FROM records ORDER BY seq'):
                    if kind=='DECISION':self._apply(json.loads(raw)['data'])
                    elif kind not in ('RUN_BEGIN','RUN_END'):fail('FBO_INTEGRITY')
                if self._binding is None:self._save('BIND',binding)
                elif self._binding!=binding:fail('FBO_BINDING_CONFLICT')
        except BaseException:
            self.close();raise
    def _apply(self,event):
        try:
            if (set(event)!={'operation','version','source_code_sha256','value','value_sha256'}
                or event['version']!=1 or event['value_sha256']!=_sha(event['value'])):fail('FBO_INTEGRITY')
            value=decode_public(event['value'])
            if event['operation']=='BIND' and self._binding is None:
                if value['execution_authorized'] is not False or value['timezone']!=ZONE:fail('FBO_INTEGRITY')
                self._binding=deepcopy(value)
            elif event['operation']=='PLAN' and self._binding and self._plan is None:
                if (value['opening_sha256']!=self._binding['opening_sha256']
                    or value['boundaries_ms']!=next_boundaries(value['created_at_ms'])):fail('FBO_INTEGRITY')
                self._plan=deepcopy(value)
            elif event['operation']=='OBSERVE' and self._binding:
                key=_identifier(value['observation_id'])
                if (key in self._observations or len(self._observations)>=1024 or value['opening_sha256']!=self._binding['opening_sha256']
                    or value['account_identity']!=self._binding['account_identity'] or value['daily_net_pnl_usdt'] is not None
                    or value['financial_status']!='PENDING' or value['execution_authorized'] is not False):fail('FBO_INTEGRITY')
                self._observations[key]=deepcopy(value)
            else:fail('FBO_INTEGRITY')
        except (KeyError,TypeError,ValueError):fail('FBO_INTEGRITY')
    def _save(self,operation,value):
        encoded=encode_public(_strings(value))
        event=dict(operation=operation,version=1,source_code_sha256=self._source,value=encoded,value_sha256=_sha(encoded))
        if sanitize(event)!=event:fail('FBO_SOURCE_INVALID')
        self._audit.append('DECISION',event);self._apply(event)
        return deepcopy(self._binding if operation=='BIND' else self._plan if operation=='PLAN' else self._observations[value['observation_id']])
    def observe(self,*,observation_id,account_before,account_after,orders,order_lists,metadata,tickers,cutoff_ms):
        with self._audit._lock:
            _identifier(observation_id);_time(cutoff_ms)
            try:
                with localcontext(Context(prec=96)):
                    built=_build_observation(self._attestation,self._namespace,observation_id,account_before,account_after,orders,order_lists,metadata,tickers,cutoff_ms)
            except (AdapterError,KeyError,TypeError,ValueError,ArithmeticError):fail('FBO_SOURCE_INVALID')
            valuation=deepcopy(built['valuation']);marks=valuation.pop('marks');routes=valuation.pop('routes')
            valuation['reason_rows']=[dict(asset=code.split(':',1)[0],reason=code.split(':',1)[1]) for code in valuation.pop('reason_codes')]
            # Public identifiers are row values, never redactor-sensitive map keys.
            valuation['asset_rows']=[dict(asset=a,mark_usdt=marks.get(a),route=[dict(symbol=s,inverse=i) for s,i in routes.get(a,[])])
                for a in sorted(built['balances'])]
            value=_strings(dict(observation_id=observation_id,cutoff_ms=cutoff_ms,opening_sha256=self._binding['opening_sha256'],
                account_identity=built['account_identity'],sources=built['sources'],reasons=built['reasons'],
                holdings=[dict(asset=a,**r) for a,r in sorted(built['balances'].items())],valuation=valuation,
                financial_status='PENDING',daily_net_pnl_usdt=None,execution_authorized=False))
            old=self._observations.get(observation_id)
            if old is not None:
                if old!=value:fail('FBO_OBSERVATION_CONFLICT')
                return deepcopy(old)
            if len(self._observations)>=1024:fail('FBO_CAPACITY')
            return self._save('OBSERVE',value)
    def schedule(self,now_ms):
        with self._audit._lock:
            _time(now_ms)
            if self._plan is not None:return deepcopy(self._plan)
            return self._save('PLAN',dict(created_at_ms=now_ms,boundaries_ms=next_boundaries(now_ms),
                opening_sha256=self._binding['opening_sha256']))
    def observations(self):
        with self._audit._lock:
            self._audit.manifest();return deepcopy(list(self._observations.values()))
    def boundary_report(self,boundary_ms):
        _time(boundary_ms)
        if day_bounds(day_for_timestamp(boundary_ms,ZONE),ZONE)[0]!=boundary_ms:fail('FBO_NOT_MIDNIGHT')
        rows=self.observations()
        pre=[r for r in rows if r['cutoff_ms']<=boundary_ms
             and 0<=boundary_ms-r['sources']['tickers']['received_at_ms']<=5000
             and all(0<=boundary_ms-r['sources'][name]['received_at_ms']<=30000 for name in ('account_before','account_after'))]
        post=[r for r in rows if 0<r['cutoff_ms']-boundary_ms<=30000
              and r['sources']['account_before']['request_started_at_ms']>=boundary_ms]
        before=max(pre,key=lambda r:(r['cutoff_ms'],r['observation_id']),default=None)
        after=min(post,key=lambda r:(r['cutoff_ms'],r['observation_id']),default=None)
        reasons=[]
        if not before:reasons.append('BEFORE_MISSING_OR_STALE')
        if not after:reasons.append('AFTER_MISSING_OR_STALE')
        if before and after:
            if before['holdings']!=after['holdings']:reasons.append('HOLDINGS_CHANGED')
            if any(r['reasons'] or r['valuation']['status']!='COMPLETE' for r in (before,after)):reasons.append('OBSERVATION_INCOMPLETE')
        return dict(boundary_ms=boundary_ms,timezone=ZONE,state='PENDING' if reasons else 'PROVISIONAL_BRACKET',
            before_id=before['observation_id'] if before else None,after_id=after['observation_id'] if after else None,
            opening_sha256=self._binding['opening_sha256'],reasons=reasons+['EXACT_ECONOMIC_CUT_UNCERTIFIED',
                'GLOBAL_FILL_FLOW_COVERAGE_PENDING','PRIVATE_DELIVERY_PENDING'],financial_status='PENDING',
            daily_net_pnl_usdt=None,execution_authorized=False)
    def close(self):
        if self._audit:self._audit.close()
