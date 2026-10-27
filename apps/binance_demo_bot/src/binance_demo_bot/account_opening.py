"""Dedicated-account declaration and provisional opening evidence; never a sender."""
from collections import defaultdict, deque
from copy import deepcopy
from decimal import Context, Decimal, localcontext
import hashlib
import json
from pathlib import Path
import re

from .adapter import AdapterError, _sha
from .financial import value_portfolio
from .interaction_audit import AuditJournal, sanitize
from .journal import WriterBusy

GATES = ['CERTIFIED_CUT_PENDING', 'FILL_FLOW_COVERAGE_PENDING',
         'PRIVATE_DELIVERY_PENDING', 'RISK_PROTECTION_SENDER_PENDING']


def _fail(reason='OPENING_INVALID'):
    raise AdapterError(reason)


def _identifier(value):
    if not isinstance(value, str) or re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', value) is None:
        _fail()
    return value


def _time(value):
    if type(value) is not int or not 0 <= value <= 2**63-1:
        _fail()
    return value


def _hex(value, length=64):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{'+str(length)+'}', value) is None:
        _fail()
    return value


def _asset(value):
    if not isinstance(value, str) or re.fullmatch(r'\w{1,64}', value) is None:
        _fail()
    return value


def _market_symbol(value):
    if not isinstance(value, str) or re.fullmatch(r'\w{2,64}', value) is None:
        _fail()
    return value


def _money(value):
    if not isinstance(value, (str, Decimal)) or isinstance(value, str) and len(value)>100:
        _fail()
    amount = Decimal(value)
    if (not amount.is_finite() or amount < 0 or len(amount.as_tuple().digits)>38
            or abs(amount.as_tuple().exponent)>36):
        _fail()
    return amount


def _text(amount):
    result = format(amount, 'f')
    return result.rstrip('0').rstrip('.') if '.' in result else result


def _strings(value):
    if isinstance(value, Decimal): return _text(value)
    if isinstance(value, dict): return {k:_strings(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [_strings(v) for v in value]
    return value


def _source(value, endpoint, cutoff, max_age, identity, *, private):
    if (not isinstance(value, dict) or value.get('environment')!='BINANCE_SPOT_DEMO'
            or value.get('source_method')!='GET' or value.get('source_endpoint')!='/api/v3/'+endpoint
            or value.get('source_sha256')!=_sha(value['data'])):
        _fail()
    wall0, wall1, mono0, mono1 = (_time(value[k]) for k in ('request_started_at_ms',
        'received_at_ms','request_started_monotonic_ns','received_monotonic_ns'))
    uncertainty=value.get('clock_uncertainty_ms')
    if type(uncertainty) not in (int,float,Decimal):_fail()
    uncertainty=Decimal(str(uncertainty))
    if (not wall0<=wall1<=cutoff or mono0>mono1 or cutoff-wall1>max_age
            or abs((wall1-wall0)*1_000_000-(mono1-mono0))>250_000_000
            or not uncertainty.is_finite() or not 0<=uncertainty<=250):
        _fail()
    for key in ('audit_begin_event_id','audit_end_event_id'):_hex(value[key],32)
    if private:
        for key, expected in identity.items():
            if value.get(key)!=expected: _fail()
        if type(value.get('schema_version')) is not int or value['schema_version']!=1:_fail()
        _identifier(value['credential_generation'])
    return {k:value[k] for k in ('source_endpoint','source_sha256','audit_begin_event_id',
        'audit_end_event_id','request_started_at_ms','received_at_ms',
        'request_started_monotonic_ns','received_monotonic_ns','clock_uncertainty_ms')}


def _balances(value, account_identity):
    if (not isinstance(value,dict) or value.get('account_identity')!=account_identity
            or value.get('coverage')!='RETURNED_BALANCES_ONLY' or not isinstance(value.get('balances'),list)
            or not isinstance(value.get('returned_assets'),list)):
        _fail()
    balances={}
    for row in value['balances']:
        asset=_asset(row['asset'])
        if asset in balances: _fail()
        balances[asset]=dict(free=_text(_money(row['free'])),locked=_text(_money(row['locked'])))
    if sorted(balances)!=value['returned_assets']: _fail()
    return dict(sorted(balances.items()))


def _valuation(balances, metadata, tickers, cutoff):
    if not isinstance(metadata,dict) or not isinstance(metadata.get('symbols'),list) or not isinstance(tickers['data'],list):
        _fail()
    members={}
    for row in metadata['symbols']:
        symbol=_market_symbol(row['symbol'])
        if symbol in members: _fail()
        members[symbol]=dict(base=_asset(row['baseAsset']),quote=_asset(row['quoteAsset']),status=_identifier(row['status']))
    quotes, adjacency, seen = {}, defaultdict(list), set()
    for row in tickers['data']:
        symbol=_market_symbol(row['symbol'])
        if symbol in seen: _fail()
        seen.add(symbol)
        member=members.get(symbol)
        bid,ask=_money(row['bidPrice']),_money(row['askPrice'])
        if not member or member['status']!='TRADING' or bid==0 or ask==0 or bid>ask: continue
        base,target=member['base'],member['quote']
        quotes[symbol]=dict(base=base,quote=target,bid=bid,ask=ask,venue='BINANCE_SPOT_DEMO',
            source_healthy=True,event_time_ms=None,received_at_ms=tickers['received_at_ms'],
            response_started_at_ms=tickers['request_started_at_ms'])
        adjacency[base].append((symbol,False,target));adjacency[target].append((symbol,True,base))
    routes={}
    for asset in balances:
        if asset=='USDT':routes[asset]=[];continue
        queue=deque([(asset,[],frozenset((asset,)))])
        while queue:
            current,path,visited=queue.popleft()
            if current=='USDT':routes[asset]=path;break
            if len(path)==3:continue
            for symbol,inverse,target in sorted(adjacency[current]):
                if target not in visited:queue.append((target,path+[[symbol,inverse]],visited|{target}))
    result=value_portfolio(balances,quotes,routes,boundary_ms=cutoff,time_basis='OBSERVED_RESPONSE',max_age_ms=5000)
    return dict(result,routes=routes,time_basis='OBSERVED_RESPONSE',convention_id='demo-mid-observed-v1',
                exchange_quote_age_unknown=True,liquidation_certified=False)


class AccountOpeningManifest:
    """Immutable audit-backed candidates. No financial journal initialization."""

    def __init__(self,paths,*,account_alias,episode_id,scope_id,database='account-opening.sqlite',fault_hook=None):
        for item in (account_alias,episode_id,scope_id):_identifier(item)
        self._journal=None
        self._attestation=None
        self._candidates={}
        self._source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        try:
            self._journal=AuditJournal(paths,account_alias=account_alias,episode_id=episode_id,
                scope_id=scope_id,database=database,max_bytes=64*1024*1024,fault_hook=fault_hook)
            with self._journal._lock:
                if self._journal._con.execute("SELECT COUNT(*) FROM records WHERE kind='DECISION'").fetchone()[0]>101:
                    _fail('OPENING_CAPACITY')
                for kind,raw in self._journal._con.execute('SELECT kind,payload_json FROM records ORDER BY seq'):
                    if kind=='DECISION':self._apply(json.loads(raw)['data'])
                    elif kind not in ('RUN_BEGIN','RUN_END'):_fail('OPENING_INTEGRITY')
                if self._attestation is None and self._journal._last_seq!=1:_fail('OPENING_INTEGRITY')
        except BaseException as error:
            self.close()
            if isinstance(error,WriterBusy):raise AdapterError('OPENING_BUSY') from None
            if isinstance(error,AdapterError):
                raise AdapterError({'AUDIT_IDENTITY_MISMATCH':'OPENING_IDENTITY',
                    'AUDIT_INTEGRITY_FAILED':'OPENING_INTEGRITY'}.get(error.reason,error.reason)) from None
            raise

    def _ready(self):
        if self._journal is None or self._journal._closed or self._journal._state!='READY':
            _fail('OPENING_UNAVAILABLE')

    def _apply(self,event):
        try:
            if set(event)!={'operation','version','source_code_sha256','value'} or event['version']!=1:_fail()
            _hex(event['source_code_sha256'])
            value=event['value']
            if value['manifest_sha256']!=_sha({k:v for k,v in value.items() if k!='manifest_sha256'}):_fail()
            if event['operation']=='ATTEST' and self._attestation is None:
                if value['exclusive'] is not True:_fail()
                self._attestation=deepcopy(value)
            elif event['operation']=='CAPTURE' and self._attestation is not None:
                key=_identifier(value['candidate_id'])
                if (key in self._candidates or len(self._candidates)>=100
                        or value['attestation_sha256']!=self._attestation['manifest_sha256']
                        or value['exchange_reconciled'] is not False or value['execution_authorized'] is not False
                        or value['financial_episode_opened'] is not False or value['daily_net_pnl_usdt'] is not None):_fail()
                self._candidates[key]=deepcopy(value)
            else:_fail()
        except (AdapterError,KeyError,TypeError,ValueError):_fail('OPENING_INTEGRITY')

    def _save(self,operation,value):
        value=_strings(value)
        value['manifest_sha256']=_sha(value)
        event=dict(operation=operation,version=1,source_code_sha256=self._source,value=value)
        if sanitize(event)!=event:_fail()
        self._journal.append('DECISION',event)
        self._apply(event)
        return deepcopy(value)

    def attest(self,*,account_identity,declaration_id,declared_at_ms,exclusive,now_ms):
        with self._journal._lock:
            self._ready()
            _hex(account_identity);_identifier(declaration_id);_time(declared_at_ms);_time(now_ms)
            if exclusive is not True or declared_at_ms>now_ms:_fail()
            value=dict(account_identity=account_identity,declaration_id=declaration_id,
                       declared_at_ms=declared_at_ms,exclusive=True)
            if self._attestation is not None:
                if {k:v for k,v in self._attestation.items() if k!='manifest_sha256'}!=value:_fail('OPENING_CONFLICT')
                return deepcopy(self._attestation)
            return self._save('ATTEST',value)

    def capture(self,*,candidate_id,account_before,account_after,orders,order_lists,metadata,tickers,cutoff_ms):
        with self._journal._lock:
            self._ready()
            if self._attestation is None:_fail('OPENING_ATTESTATION_REQUIRED')
            _identifier(candidate_id);_time(cutoff_ms)
            try:
                with localcontext(Context(prec=96)):
                    value=self._build(candidate_id,account_before,account_after,orders,order_lists,metadata,tickers,cutoff_ms)
            except (AdapterError,KeyError,TypeError,ValueError,ArithmeticError):
                _fail('OPENING_SOURCE_INVALID')
            value=_strings(value)
            old=self._candidates.get(candidate_id)
            if old is not None:
                if {k:v for k,v in old.items() if k!='manifest_sha256'}!=value:_fail('OPENING_CONFLICT')
                return deepcopy(old)
            if len(self._candidates)>=100:_fail('OPENING_CAPACITY')
            return self._save('CAPTURE',value)

    def _build(self,key,before,after,orders,lists,metadata,tickers,cutoff):
        if self._attestation['declared_at_ms']>cutoff:_fail()
        identity={k:v for k,v in self._journal.identity.items() if k!='schema_version'}
        source_inputs=dict(account_before=(before,'account',30000,True),
            account_after=(after,'account',30000,True),orders=(orders,'openOrders',30000,True),
            order_lists=(lists,'openOrderList',30000,True),metadata=(metadata,'exchangeInfo',900000,False),
            tickers=(tickers,'ticker/bookTicker',5000,False))
        pins={name:_source(source,endpoint,cutoff,age,identity,private=private)
              for name,(source,endpoint,age,private) in source_inputs.items()}
        if len({r['credential_generation'] for r in (before,after,orders,lists)})!=1:_fail()
        for source in (orders,lists,tickers):
            if (before['received_at_ms']>source['request_started_at_ms']
                    or source['received_at_ms']>after['request_started_at_ms']
                    or before['received_monotonic_ns']>source['request_started_monotonic_ns']
                    or source['received_monotonic_ns']>after['request_started_monotonic_ns']):_fail()
        if not isinstance(orders['data'],list) or not isinstance(lists['data'],list):_fail()
        initial=_balances(before['data'],self._attestation['account_identity'])
        final=_balances(after['data'],self._attestation['account_identity'])
        reasons=[]
        if initial!=final:reasons.append('BALANCES_CHANGED')
        if any(Decimal(row['locked'])!=0 for row in final.values()):reasons.append('LOCKED_INVENTORY')
        if orders['data']:reasons.append('OPEN_ORDERS_PRESENT')
        if lists['data']:reasons.append('OPEN_ORDER_LISTS_PRESENT')
        if any(r['data'].get('can_trade') is not True or not isinstance(r['data'].get('permissions'),list)
               or 'SPOT' not in r['data']['permissions'] for r in (before,after)):
            reasons.append('SPOT_PERMISSION_UNCONFIRMED')
        held={asset:row for asset,row in final.items() if Decimal(row['free'])+Decimal(row['locked'])!=0}
        valuation=_valuation(held,metadata['data'],tickers,cutoff)
        if valuation['status']!='COMPLETE':reasons.append('VALUATION_INCOMPLETE')
        lots=[]
        if not reasons:
            for asset,row in held.items():
                quantity=Decimal(row['free'])+Decimal(row['locked'])
                lots.append(dict(asset=asset,quantity=quantity,basis_usdt=quantity*valuation['marks'][asset],ownership='SEED'))
        return dict(candidate_id=key,cutoff_ms=cutoff,attestation_sha256=self._attestation['manifest_sha256'],
            credential_generation=before['credential_generation'],account_identity=self._attestation['account_identity'],
            state='BLOCKED' if reasons else 'PROVISIONAL_OPENING',reasons=reasons,sources=pins,
            balances=held,held_assets_N=len(held),returned_assets_N=len(final),
            inventory_coverage='RETURNED_BALANCES_ONLY',seed_lots=lots,valuation=valuation,
            managed_inventory={},exchange_reconciled=False,financial_episode_opened=False,
            execution_authorized=False,daily_net_pnl_usdt=None,remaining_gates=GATES[:])

    def summary(self):
        with self._journal._lock:
            self._ready()
            return dict(attestation=deepcopy(self._attestation),candidates_n=len(self._candidates),
                        candidates=deepcopy(self._candidates))

    def close(self):
        if self._journal is not None:self._journal.close()
