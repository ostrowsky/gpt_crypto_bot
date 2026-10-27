"""Opt-in, audited order-test permission probe. No matching-engine submission."""
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path
import re
import uuid

from .adapter import (AdapterError, BinanceDemoAdapter, DemoHttpTransport, HttpResponse,
                      _json, _sha, sign_rest)
from .interaction_audit import AuditJournal


ENDPOINT='/api/v3/order/test'
AUDIT_ERRORS=frozenset(('AUDIT_UNAVAILABLE','AUDIT_WRITE_FAILED','AUDIT_COMMIT_UNKNOWN',
                       'AUDIT_DISK_LIMIT','AUDIT_EVENT_LIMIT','AUDIT_BACKPRESSURE','AUDIT_INVALID'))
UNAVAILABLE=frozenset(('ORDER_TEST_UNAVAILABLE','CLOCK_JUMP','CLOCK_QUALITY','CLOCK_STALE',
    'CLOCK_UNSYNCED','READ_UNAVAILABLE','RATE_LIMITED','IP_BANNED','LOCAL_RATE_BUDGET',
    'TIMESTAMP_REJECTED','ACCOUNT_NOT_READY','ADAPTER_DISABLED','AUDIT_UNAVAILABLE'))|AUDIT_ERRORS


def _amount(value):
    if not isinstance(value,(str,Decimal)) or isinstance(value,str) and len(value)>96:
        raise AdapterError('INVALID_PARAMETERS')
    try:
        amount=Decimal(value)
        if (not amount.is_finite() or amount<=0 or len(amount.as_tuple().digits)>28
                or abs(amount.as_tuple().exponent)>18):
            raise ValueError()
        text=format(amount,'f')
        return text.rstrip('0').rstrip('.') if '.' in text else text
    except (InvalidOperation,ValueError,TypeError):
        raise AdapterError('INVALID_PARAMETERS') from None


class DemoOrderTestTransport(DemoHttpTransport):
    def request(self,method,path,query,headers,timeout_s):
        if method!='POST' or not isinstance(path,str) or path!=ENDPOINT:
            raise AdapterError('FORBIDDEN_ENDPOINT')
        return self._request_https(method,path,query,headers,timeout_s)


class DemoTradeValidator:
    def __init__(self,client,*,allow_order_validation=False,transport=None):
        if not isinstance(client,BinanceDemoAdapter) or type(allow_order_validation) is not bool:
            raise AdapterError('INVALID_PARAMETERS')
        self.client=client
        self._audit_ready()
        self._enabled,self._closed=allow_order_validation,False
        self.transport=DemoOrderTestTransport() if transport is None else transport
        self._source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def _audit_ready(self):
        audit=self.client.audit
        if not isinstance(audit,AuditJournal) or audit._closed or audit._state!='READY':
            raise AdapterError('AUDIT_UNAVAILABLE')
        if any(audit.identity.get(k)!=self.client.identity[k]
               for k in ('account_alias','account_episode_id','portfolio_scope_id')):
            raise AdapterError('AUDIT_IDENTITY_MISMATCH')
        if audit.health()['unresolved_attempts_n']:
            raise AdapterError('AUDIT_UNAVAILABLE')

    def _cap(self,state,params=None,response_hash=None):
        digest=_sha(params) if params is not None else None
        self.client._set_cap('KEY_TRADE_VALIDATED',state,
            _sha(dict(request_sha256=digest,response_sha256=response_hash)) if digest else None)
        self.client._cap_states['KEY_TRADE_VALIDATED'].update(source='P1_ORDER_TEST',
            tested_request_sha256=digest,tested_symbol=params.get('symbol') if params else None,
            tested_side=params.get('side') if params else None,tested_type='LIMIT' if params else None,
            validation_code_sha256=self._source)

    def _decision(self,action,reason,params):
        self._audit_ready()
        return self.client.audit.record_decision(decision_type='OPERATIONAL',action=action,
            reasons=[reason],cutoff_ms=self.client.clock.wall_ms(),sources=[],
            candidate_population=[params['symbol']] if params else [],policy_version='trade-validation-v1',
            model_version=None,risk_version='no-submission-v1',intent=dict(operation='ORDER_TEST_VALIDATION',
                tested_parameters=params,tested_request_sha256=_sha(params) if params else None,
                validation_code_sha256=self._source,submitted=False))

    def disable(self):
        self._enabled=False
        self._cap('UNVERIFIED')
        self._decision('BLOCK','ORDER_VALIDATION_DISABLED',None)

    def close(self):
        self._closed=True
        if hasattr(self.transport,'close'):self.transport.close()

    def _rates(self,response,received_mono):
        client=self.client
        lower={k.lower():v for k,v in response.headers.items()}
        weight=lower.get('x-mbx-used-weight-1m')
        if weight is not None and weight.isascii() and weight.isdecimal() and len(weight)<=20:
            client._shared_weight=max(client._shared_weight,int(weight))
            client._shared_weight_expires_ns=received_mono+60000000000
        if response.status in (429,418):
            retry=lower.get('retry-after','60')
            wait=int(retry) if retry.isascii() and retry.isdecimal() and len(retry)<=10 else 60
            client.cooldown_until_ns=received_mono+max(1,wait)*1000000000
            if response.status==418:client.blocked='IP_BANNED'
            raise AdapterError('IP_BANNED' if response.status==418 else 'RATE_LIMITED')

    def validate_limit(self,symbol,*,side,quantity,price):
        client=self.client;params=None
        self._cap('UNVERIFIED')
        try:
            self._audit_ready()
            if self._closed or not self._enabled:raise AdapterError('ORDER_VALIDATION_DISABLED')
            if (not isinstance(symbol,str) or re.fullmatch('[A-Z0-9]{1,20}USDT',symbol) is None
                    or side not in ('BUY','SELL')):raise AdapterError('INVALID_PARAMETERS')
            params=dict(symbol=symbol,side=side,type='LIMIT',timeInForce='GTC',
                        quantity=_amount(quantity),price=_amount(price),newClientOrderId='bdv_'+uuid.uuid4().hex)
            deadline=client.clock.monotonic_ns()+15000000000
            client._guard(1)
            try:timestamp=client.clock.timestamp_ms()
            except AdapterError as error:
                if error.reason not in ('CLOCK_UNSYNCED','CLOCK_STALE'):raise
                client.sync_clock(deadline_ns=deadline);timestamp=client.clock.timestamp_ms()
            caps=client.capabilities()
            if any(caps[k]['state']!='SUPPORTED' for k in ('USER_DATA_READ','ACCOUNT_SPOT_ALLOWED')):
                raise AdapterError('ACCOUNT_NOT_READY')
            self._decision('WAIT','ORDER_TEST_REQUESTED',params)
            signed_params=dict(params,timestamp=timestamp,recvWindow=5000)
            query=sign_rest(signed_params,client._credentials.api_secret)
            try:
                result=client._transport_attempt(ENDPOINT,query,{'X-MBX-APIKEY':client._credentials.api_key},
                    signed_params,True,deadline,uuid.uuid4().hex,0,method='POST',transport=self.transport)
            except AdapterError as error:
                if error.reason in AUDIT_ERRORS:raise
                raise AdapterError('ORDER_TEST_UNAVAILABLE') from None
            except (TimeoutError,OSError):raise AdapterError('ORDER_TEST_UNAVAILABLE') from None
            except Exception:raise AdapterError('ORDER_TEST_UNAVAILABLE') from None
            response,wall,mono,received_wall,received_mono,attempt_id,begin,end=result
            if (received_mono<mono or received_mono>deadline or client.clock.monotonic_ns()>deadline):
                raise AdapterError('ORDER_TEST_UNAVAILABLE')
            if (not isinstance(response,HttpResponse) or type(response.status) is not int
                    or not isinstance(response.headers,dict) or any(not isinstance(k,str) or not isinstance(v,str)
                    for k,v in response.headers.items())):raise AdapterError('SCHEMA_INVALID')
            self._rates(response,received_mono)
            if 300<=response.status<400:raise AdapterError('ORDER_TEST_UNAVAILABLE')
            if response.status>=500:raise AdapterError('ORDER_TEST_UNAVAILABLE')
            if response.status==401:raise AdapterError('AUTH_DENIED')
            try:data=_json(response.body)
            except AdapterError:
                if response.status==403:raise AdapterError('ORDER_TEST_UNAVAILABLE') from None
                raise
            code=data.get('code') if isinstance(data,dict) else None
            if code in (-2014,-2015,-1022):raise AdapterError('AUTH_DENIED')
            if response.status==403:raise AdapterError('ORDER_TEST_UNAVAILABLE')
            if code==-1021:raise AdapterError('TIMESTAMP_REJECTED')
            if response.status!=200:raise AdapterError('ORDER_TEST_REJECTED')
            if not isinstance(data,dict) or data:raise AdapterError('SCHEMA_INVALID')
            self._decision('NO_ACTION','ORDER_TEST_VALIDATED',params)
            if client.clock.monotonic_ns()>deadline:
                raise AdapterError('ORDER_TEST_UNAVAILABLE')
            self._cap('SUPPORTED',params,_sha(data))
            return dict(client.identity,schema_version=1,state='VALIDATED_NOT_SUBMITTED',submitted=False,
                order_send_enabled=False,source_method='POST',source_endpoint=ENDPOINT,
                tested_parameters=dict(params),tested_request_sha256=_sha(params),source_sha256=_sha(data),
                validation_code_sha256=self._source,request_started_at_ms=wall,request_started_monotonic_ns=mono,
                received_at_ms=received_wall,received_monotonic_ns=received_mono,audit_attempt_id=attempt_id,
                audit_begin_event_id=begin,audit_end_event_id=end,clock_uncertainty_ms=client.clock.uncertainty_ms)
        except AdapterError as error:
            state='DENIED' if error.reason=='AUTH_DENIED' else 'UNAVAILABLE' if error.reason in UNAVAILABLE else 'UNVERIFIED'
            self._cap(state)
            if client.audit is not None and not client.audit._closed and client.audit._state=='READY':
                try:self._decision('BLOCK',error.reason,params)
                except AdapterError:
                    self._cap('UNAVAILABLE');raise
            raise
        except BaseException:
            self._cap('UNAVAILABLE')
            raise
