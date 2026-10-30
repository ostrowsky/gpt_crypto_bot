"""HT02/03/08 isolated public Demo aggregate-trade source and evidence audit.

No existing adapter allowlist is changed. This client has no credentials,
orders, alternate host, proxy discovery or historical completeness authority.
"""
import base64
from collections import Counter
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import sqlite3
import ssl
import time
from urllib.parse import parse_qsl, urlencode
import uuid

from .adapter import HttpResponse
from .interaction_audit import AuditJournal, rate_headers, response_payload


ENDPOINT = 'https://demo-api.binance.com/api/v3/aggTrades'
PATH = '/api/v3/aggTrades'
BODY_LIMIT = 8 * 1024 * 1024
KINDS = {'RUN_BEGIN', 'RUN_END', 'REST_BEGIN', 'REST_END'}


class TradeSourceError(Exception):
    def __init__(self, reason, api_code=None, status=None):
        self.reason, self.api_code, self.status = reason, api_code, status
        super().__init__(reason)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _parameters(params):
    if not isinstance(params, dict) or not set(params) <= {'symbol', 'startTime', 'endTime', 'fromId', 'limit'}:
        raise TradeSourceError('INVALID_PARAMETERS')
    if (not isinstance(params.get('symbol'), str) or not params['symbol'] or len(params['symbol']) > 64
            or any(ord(c) < 33 for c in params['symbol'])
            or type(params.get('limit')) is not int or not 1 <= params['limit'] <= 1000):
        raise TradeSourceError('INVALID_PARAMETERS')
    keys = set(params)
    if keys == {'symbol', 'startTime', 'endTime', 'limit'}:
        first, last = params['startTime'], params['endTime']
        if type(first) is not int or type(last) is not int or not 0 <= first <= last <= first + 900_000:
            raise TradeSourceError('INVALID_PARAMETERS')
    elif keys == {'symbol', 'fromId', 'limit'}:
        if type(params['fromId']) is not int or params['fromId'] < 0:
            raise TradeSourceError('INVALID_PARAMETERS')
    else:
        raise TradeSourceError('INVALID_PARAMETERS')
    return dict(params)


def _query_parameters(query):
    if not isinstance(query, str) or any(c in query for c in '\r\n?#'):
        raise TradeSourceError('INVALID_PARAMETERS')
    try:
        pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=True)
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValueError('DUPLICATE')
        params = dict(pairs)
        for key in ('startTime', 'endTime', 'fromId', 'limit'):
            if key in params:
                if not re.fullmatch(r'[0-9]+', params[key]):
                    raise ValueError('INTEGER')
                params[key] = int(params[key])
        return _parameters(params)
    except (ValueError, TypeError):
        raise TradeSourceError('INVALID_PARAMETERS') from None


class HistoricalTradeTransport:
    """Exactly one fixed TLS host/path; no redirects or library-level retries."""
    def request(self, method, path, query, headers, timeout_s):
        if method != 'GET' or path != PATH or headers != {}:
            raise TradeSourceError('FORBIDDEN_ENDPOINT')
        _query_parameters(query)
        if (type(timeout_s) not in (int, float) or not math.isfinite(timeout_s)
                or not 0 < timeout_s <= 15):
            raise TradeSourceError('INVALID_TIMEOUT')
        deadline = time.monotonic() + timeout_s
        def remaining():
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError()
            return value
        connection = None
        try:
            connection = http.client.HTTPSConnection('demo-api.binance.com', port=443,
                timeout=min(5, remaining()), context=ssl.create_default_context())
            connection.connect()
            connection.sock.settimeout(remaining())
            connection.request('GET', PATH + '?' + query, headers={})
            connection.sock.settimeout(remaining())
            response = connection.getresponse()
            if hasattr(response, 'read1'):
                chunks, size = [], 0
                while True:
                    connection.sock.settimeout(remaining())
                    block = response.read1(min(65536, BODY_LIMIT + 1 - size))
                    if not block:
                        break
                    chunks.append(block)
                    size += len(block)
                    if size > BODY_LIMIT:
                        raise TradeSourceError('RESPONSE_TOO_LARGE', status=response.status)
                body = b''.join(chunks)
            else:
                body = response.read(BODY_LIMIT + 1)
            remaining()
            if not isinstance(body, bytes) or len(body) > BODY_LIMIT:
                raise TradeSourceError('RESPONSE_TOO_LARGE', status=response.status)
            # http.client never follows3xx. Return its status/body to the
            # journal before the client rejects the redirect response.
            return HttpResponse(response.status, body, dict(response.getheaders()))
        finally:
            if connection is not None:
                connection.close()

    def close(self):
        pass  # Each native attempt owns and closes its single TLS connection.


class HistoricalTradeClient:
    """One bounded invocation, audited before and after every native attempt."""
    def __init__(self, audit, transport=None, *, sleep=time.sleep, monotonic=time.monotonic,
                 wall_ms=None, max_attempts=300, max_wall_s=900):
        if (not isinstance(audit, AuditJournal) or type(max_attempts) is not int
                or not 1 <= max_attempts <= 300 or type(max_wall_s) not in (int, float)
                or not math.isfinite(max_wall_s) or not 0 < max_wall_s <= 900):
            raise TradeSourceError('INVALID_INVOCATION_OR_AUDIT')
        self.audit, self.transport = audit, transport or HistoricalTradeTransport()
        self.sleep, self.monotonic = sleep, monotonic
        self.wall_ms = wall_ms or (lambda: time.time_ns() // 1_000_000)
        self.max_attempts, self.deadline = max_attempts, monotonic() + max_wall_s
        self.attempts_n, self.last_start, self.closed = 0, None, False
        self.blocked = None
        self.cooldown_until = 0.
        self._transport_counts = dict(attempted_n=0, responses_n=0, errors_n=0)

    def _wait(self, delay):
        remaining = self.deadline - self.monotonic()
        if remaining <= 0:
            raise TradeSourceError('INVOCATION_DEADLINE')
        if delay > 0:
            self.sleep(min(delay, remaining))
        if self.monotonic() >= self.deadline:
            raise TradeSourceError('INVOCATION_DEADLINE')

    def _attempt(self, params, request_id, number):
        if self.closed:
            raise TradeSourceError('CLIENT_CLOSED')
        if self.attempts_n >= self.max_attempts:
            raise TradeSourceError('ATTEMPT_BUDGET_EXHAUSTED')
        self._wait(max(0.,self.cooldown_until-self.monotonic()))
        self._wait(max(0., .3 - (self.monotonic() - self.last_start)) if self.last_start is not None else 0.)
        if self.last_start is not None and self.monotonic() - self.last_start < .3 - 1e-9:
            raise TradeSourceError('MONOTONIC_PACING_UNAVAILABLE')
        attempt_id = uuid.uuid4().hex
        parent = self.audit.append('REST_BEGIN', dict(request_id=request_id, attempt_id=attempt_id,
            attempt_number=number, method='GET', endpoint=PATH, parameters=params, private=False),
            wall_ms=self.wall_ms(), monotonic_ns=int(self.monotonic() * 1_000_000_000))
        response, error, sent = None, None, False
        started_wall, started = self.wall_ms(), self.monotonic()
        try:
            remaining = self.deadline - self.monotonic()
            if remaining <= 0:
                raise TradeSourceError('INVOCATION_DEADLINE')
            self.last_start = self.monotonic()
            self.attempts_n += 1
            self._transport_counts['attempted_n'] += 1
            sent = True
            response = self.transport.request('GET', PATH, urlencode(sorted(params.items())), {}, min(15, remaining))
            self._transport_counts['responses_n'] += 1
            if not isinstance(response, HttpResponse) or type(response.status) is not int or not isinstance(response.body, bytes):
                raise TradeSourceError('MALFORMED_RESPONSE')
            if len(response.body) > BODY_LIMIT:
                raise TradeSourceError('RESPONSE_TOO_LARGE', status=response.status)
        except BaseException as caught:
            error = caught
            self._transport_counts['errors_n'] += 1
            raise
        finally:
            received_wall, received = self.wall_ms(), self.monotonic()
            valid = isinstance(response, HttpResponse)
            outcome = ('RESPONSE' if error is None else 'TIMEOUT' if isinstance(error, (TimeoutError, OSError))
                       else 'CANCELLED' if not sent or isinstance(error, (KeyboardInterrupt, SystemExit)) else 'TRANSPORT_ERROR')
            payload = (response_payload(response.body, private=False) if valid and isinstance(response.body, bytes)
                and len(response.body) <= BODY_LIMIT else dict(representation='OMITTED', bytes_n=len(response.body))
                if valid and isinstance(response.body, bytes) else None)
            terminal = self.audit.append('REST_END', dict(request_id=request_id, attempt_id=attempt_id,
                parent_event_id=parent, outcome=outcome, method='GET', endpoint=PATH, private=False,
                status=response.status if valid and type(response.status) is int else getattr(error, 'status', None),
                headers=rate_headers(response.headers) if valid else {}, sent_at_ms=started_wall if sent else None,
                sent_monotonic_ns=int(started * 1_000_000_000) if sent else None,
                received_at_ms=received_wall, received_monotonic_ns=int(received * 1_000_000_000),
                latency_ns=max(0, int((received-started) * 1_000_000_000)) if sent else None,
                error_reason=(error.reason if isinstance(error, TradeSourceError) else 'NETWORK_ERROR' if error else None),
                payload=payload), wall_ms=received_wall, monotonic_ns=int(received * 1_000_000_000))
        return response, dict(raw_body_sha256=hashlib.sha256(response.body).hexdigest(),
            received_at_ms=received_wall, received_monotonic_ns=int(received * 1_000_000_000),
            request_started_at_ms=started_wall, audit_request_id=request_id, audit_attempt_id=attempt_id,
            audit_begin_event_id=parent, audit_end_event_id=terminal, source_endpoint=PATH, http_status=response.status)

    def read(self, params):
        params = _parameters(params)
        if self.blocked is not None:
            raise self.blocked
        request_id, last = uuid.uuid4().hex, None
        for number in range(1, 4):
            retry, delay = False, .3 * number
            try:
                response, metadata = self._attempt(params, request_id, number)
            except (TimeoutError, OSError):
                last, retry = TradeSourceError('READ_UNAVAILABLE'), True
            except TradeSourceError:
                raise
            except Exception:
                raise TradeSourceError('TRANSPORT_OR_AUDIT_UNAVAILABLE') from None
            else:
                try:
                    data = json.loads(response.body)
                except (ValueError, UnicodeError):
                    data = None
                code = data.get('code') if isinstance(data, dict) and type(data.get('code')) is int else None
                status = response.status
                if status == 418:
                    self.blocked=TradeSourceError('IP_BANNED', api_code=code, status=status)
                    raise self.blocked
                if status == 429 or status in (500, 502, 503, 504):
                    last = TradeSourceError('RATE_LIMITED' if status == 429 else 'READ_UNAVAILABLE', api_code=code, status=status)
                    retry = True
                    header = next((v for k, v in response.headers.items() if str(k).lower() == 'retry-after'), None)
                    if isinstance(header, str) and re.fullmatch(r'[0-9]{1,20}', header):
                        requested=int(header)
                        if requested>60 or requested>=self.deadline-self.monotonic():
                            self.blocked=TradeSourceError('RATE_LIMITED_LONG_COOLDOWN',api_code=code,status=status)
                            raise self.blocked
                        delay = max(delay, requested)
                    self.cooldown_until=max(self.cooldown_until,self.monotonic()+delay)
                elif 300 <= status < 400:
                    raise TradeSourceError('REDIRECT_BLOCKED', api_code=code, status=status)
                elif status != 200 or code is not None and code < 0:
                    raise TradeSourceError('API_REJECTED', api_code=code, status=status)
                elif not isinstance(data, list):
                    raise TradeSourceError('SCHEMA_INVALID', status=status)
                else:
                    return data, metadata
            if retry and number < 3:
                self._wait(delay)
                continue
            raise last or TradeSourceError('READ_UNAVAILABLE')

    def close(self):
        self.closed = True
        self.transport.close()


def verify_trade_audit(folder, audit_path):
    """Verify a closed journal and exact public source/cache ownership read-only."""
    folder,audit_path=Path(folder),Path(audit_path)
    failures=[]; unknown=[]; counts=Counter(); records={}; begins={}; ends={}; runs={}; raw={}; tip='0'*64
    linked=0
    def issue(reason,fail=False):
        target=failures if fail else unknown
        if reason not in target: target.append(reason)
    def result():
        return dict(state='FAIL' if failures else 'UNKNOWN' if unknown else 'VERIFIED_RECORDED_PUBLIC_ATTEMPTS',
            reasons=failures+unknown,failures=failures,unknowns=unknown,counts=dict(counts),
            attempts_n=counts['REST_BEGIN'],responses_n=sum(e['data'].get('outcome')=='RESPONSE' for e in ends.values()),
            errors_n=sum(e['data'].get('outcome')!='RESPONSE' or e['data'].get('status')!=200 for e in ends.values()),
            linked_cached_responses_n=linked,records_n=len(records),tip=tip,runids=list(runs),
            scope='RECORDED_PUBLIC_AGGREGATE_TRADE_ATTEMPTS_AND_CACHE_LINKS_ONLY',promotion_allowed=False)
    if not audit_path.is_file(): issue('AUDIT_MISSING'); return result()
    wal=audit_path.with_name(audit_path.name+'-wal')
    if wal.is_file() and wal.stat().st_size: issue('AUDIT_WAL_NOT_FINALIZED'); return result()
    connection=None
    try:
        connection=sqlite3.connect(audit_path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
        row=connection.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        identity=json.loads(row[0]) if row else None
        if identity is None: issue('AUDIT_IDENTITY_MISSING')
        for expected,(seq,identifier,kind,payload,digest) in enumerate(connection.execute(
                'SELECT seq,id,kind,payload_json,content_sha256 FROM records ORDER BY seq'),1):
            value=json.loads(payload)
            if (seq!=expected or value['seq']!=seq or value['event_id']!=identifier or identifier in records
                or kind not in KINDS or value['kind']!=kind or value['previous_sha256']!=tip or _sha(value)!=digest):
                issue('AUDIT_CHAIN_INVALID',True); return result()
            if (value.get('application_id')!='binance_demo_bot' or value.get('environment')!='BINANCE_SPOT_DEMO'
                or identity is not None and any(value.get(k)!=v for k,v in identity.items())):
                issue('AUDIT_IDENTITY_CONTRADICTION',True)
            if not isinstance(value.get('data'),dict): issue('AUDIT_DATA_INVALID',True); return result()
            tip=digest; records[identifier]=value; counts[kind]+=1
            rid=value.get('run_id'); data=value['data']
            if not isinstance(rid,str) or not rid: issue('AUDIT_RUN_ID_MISSING'); continue
            if kind=='RUN_BEGIN':
                if rid in runs: issue('AUDIT_DUPLICATE_RUN',True)
                runs[rid]=dict(begin=seq,end=None); continue
            if rid not in runs: issue('AUDIT_RUN_BEGIN_MISSING',True); continue
            if runs[rid]['end'] is not None: issue('AUDIT_EVENT_AFTER_RUN_END',True)
            if kind=='RUN_END': runs[rid]['end']=seq; continue
            if data.get('method')!='GET' or data.get('endpoint')!=PATH or data.get('private') is not False:
                issue('AUDIT_FORBIDDEN_REQUEST',True)
            if not all(isinstance(data.get(k),str) and data[k] for k in ('request_id','attempt_id')):
                issue('AUDIT_ATTEMPT_IDS_MISSING'); continue
            if kind=='REST_BEGIN':
                if not isinstance(data.get('parameters'),dict): issue('AUDIT_PARAMETERS_MISSING')
                if any(b['data']['attempt_id']==data['attempt_id'] for b in begins.values()): issue('AUDIT_DUPLICATE_ATTEMPT',True)
                begins[identifier]=value; continue
            parent=data.get('parent_event_id'); begin=begins.get(parent)
            if begin is None: issue('AUDIT_ORPHAN_END',True); continue
            if (begin['run_id']!=rid or any(data.get(k)!=begin['data'].get(k) for k in
                ('request_id','attempt_id','method','endpoint','private')) or
                any(e['data'].get('parent_event_id')==parent for e in ends.values())):
                issue('AUDIT_ATTEMPT_CONTRADICTION',True)
            ends[identifier]=value
            if data.get('outcome') not in ('RESPONSE','TIMEOUT','TRANSPORT_ERROR','CANCELLED'):
                issue('AUDIT_OUTCOME_MISSING')
            wrapper=data.get('payload')
            if wrapper is not None and wrapper.get('representation')=='EXACT_PUBLIC_BYTES':
                body=base64.b64decode(wrapper['body_base64'],validate=True)
                if len(body)!=wrapper['bytes_n'] or hashlib.sha256(body).hexdigest()!=wrapper['content_sha256']:
                    issue('AUDIT_PUBLIC_BODY_CONTRADICTION',True)
                raw[identifier]=body
            elif data.get('outcome')=='RESPONSE': issue('AUDIT_EXACT_BODY_UNAVAILABLE')
    except (sqlite3.Error,OSError,ValueError,TypeError,KeyError,AttributeError): issue('AUDIT_SCHEMA_OR_CHAIN_INVALID',True)
    finally:
        if connection is not None: connection.close()
    if any(v['end'] is None for v in runs.values()): issue('AUDIT_RUN_OPEN')
    paired={e['data'].get('parent_event_id') for e in ends.values()}
    if set(begins)-paired: issue('AUDIT_UNPAIRED_ATTEMPTS')
    if not begins: issue('AUDIT_NO_ATTEMPTS')
    used=set()
    for path in sorted((folder/'responses').glob('*.json')):
        try:
            cached=json.loads(path.read_text(encoding='utf-8')); meta=cached['metadata']
            keys=('audit_request_id','audit_attempt_id','audit_begin_event_id','audit_end_event_id','raw_body_sha256','received_at_ms')
            if any(meta.get(k) is None for k in keys): issue('CACHE_PROVENANCE_MISSING'); continue
            begin=begins.get(meta['audit_begin_event_id']); end=ends.get(meta['audit_end_event_id'])
            if begin is None or end is None: issue('CACHE_NATIVE_RECORD_NOT_FOUND',True); continue
            bd,ed=begin['data'],end['data']; body=raw.get(end['event_id'])
            if (cached['endpoint']!=ENDPOINT or cached['parameters']!=bd.get('parameters') or
                meta['audit_request_id']!=bd['request_id'] or meta['audit_attempt_id']!=bd['attempt_id'] or
                ed.get('parent_event_id')!=begin['event_id'] or ed.get('request_id')!=meta['audit_request_id'] or
                ed.get('attempt_id')!=meta['audit_attempt_id'] or ed.get('outcome')!='RESPONSE' or
                ed.get('status')!=200 or meta['received_at_ms']!=ed.get('received_at_ms') or end['event_id'] in used):
                issue('CACHE_NATIVE_OWNERSHIP_CONTRADICTION',True); continue
            if body is None: issue('CACHE_NATIVE_BODY_MISSING'); continue
            if (hashlib.sha256(body).hexdigest()!=meta['raw_body_sha256'] or json.loads(body)!=cached['rows'] or
                'source_sha256' in cached and cached['source_sha256']!=_sha(cached['rows'])):
                issue('CACHE_NATIVE_BODY_CONTRADICTION',True); continue
            used.add(end['event_id']); linked+=1
        except (ValueError,TypeError,KeyError,OSError): issue('CACHE_PROVENANCE_UNREADABLE')
    return result()
