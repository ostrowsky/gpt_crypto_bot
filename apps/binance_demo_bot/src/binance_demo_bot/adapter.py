"""P1-A single-threaded, read-only Spot Demo account adapter.

No discovery of credentials, legacy state, proxies or production endpoints.
Returned observations do not establish reconciliation or execution ownership.
"""

from dataclasses import dataclass, field
from collections import deque
from decimal import Decimal, InvalidOperation
from functools import wraps
import hashlib
import hmac
import http.client
import json
import re
import ssl
import time
import uuid
import zlib
from urllib.parse import urlencode, urlsplit

from .isolation import DemoCredentials


BASE_URL = "https://demo-api.binance.com/api/v3/"
PUBLIC_PATHS = frozenset(("/api/v3/ping", "/api/v3/time", "/api/v3/exchangeInfo",
                         "/api/v3/ticker/bookTicker", "/api/v3/klines", "/api/v3/depth", "/api/v3/avgPrice"))
READ_PATHS = PUBLIC_PATHS | frozenset(("/api/v3/account",
                        "/api/v3/openOrders", "/api/v3/order", "/api/v3/myTrades", "/api/v3/openOrderList", "/api/v3/orderList"))
BODY_LIMIT = 8 * 1024 * 1024
METADATA_LIMIT = 32 * 1024 * 1024


def _body_limit(path):
    return METADATA_LIMIT if path == "/api/v3/exchangeInfo" else BODY_LIMIT


class AdapterError(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _integer(value, minimum=0, reason="SCHEMA_INVALID"):
    if type(value) is not int or value < minimum:
        raise AdapterError(reason)
    return value


def time_ms(value, unit="ms"):
    _integer(value, reason="INVALID_TIME")
    if unit == "ms":
        return value
    if unit == "us" and value % 1000 == 0:
        return value // 1000
    raise AdapterError("INVALID_TIME")


def _value(value):
    if isinstance(value, str):
        return value
    if type(value) is bool:
        return str(value).lower()
    if type(value) is int:
        return str(value)
    if isinstance(value, Decimal) and value.is_finite():
        return format(value, "f")
    raise AdapterError("INVALID_PARAMETERS")


def sign_rest(parameters, secret):
    try:
        pairs = list(parameters.items()) if isinstance(parameters, dict) else list(parameters)
        names = [pair[0] for pair in pairs]
        if any(not isinstance(name, str) or not name or name == "signature" for name in names):
            raise AdapterError("INVALID_PARAMETERS")
        if len(set(names)) != len(names) or not isinstance(secret, str) or not secret:
            raise AdapterError("INVALID_PARAMETERS")
        query = urlencode(sorted((name, _value(value)) for name, value in pairs), encoding="utf-8")
        signature = hmac.new(secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
        return query + "&signature=" + signature
    except AdapterError:
        raise
    except (TypeError, ValueError, UnicodeError):
        raise AdapterError("INVALID_PARAMETERS") from None


class DemoClock:
    def __init__(self, wall_ms=None, monotonic_ns=None):
        self.wall_ms = wall_ms or (lambda: time.time_ns() // 1_000_000)
        self.monotonic_ns = monotonic_ns or time.monotonic_ns
        self.sample = None

    def observe(self, server_time_ms, started_wall_ms, ended_wall_ms,
                started_monotonic_ns, ended_monotonic_ns):
        self.sample = None
        values = (server_time_ms, started_wall_ms, ended_wall_ms,
                  started_monotonic_ns, ended_monotonic_ns)
        if any(type(v) is not int or v < 0 for v in values):
            raise AdapterError("CLOCK_QUALITY")
        elapsed = ended_monotonic_ns - started_monotonic_ns
        wall_delta = ended_wall_ms - started_wall_ms
        if elapsed < 0 or elapsed > 1_000_000_000 or wall_delta < 0:
            raise AdapterError("CLOCK_QUALITY")
        uncertainty_ns = (elapsed + 1) // 2
        if uncertainty_ns > 250_000_000 or abs(wall_delta * 1_000_000 - elapsed) > 250_000_000:
            raise AdapterError("CLOCK_QUALITY")
        offset_ms = server_time_ms - (started_wall_ms + ended_wall_ms) // 2
        self.sample = (ended_wall_ms, ended_monotonic_ns, offset_ms, uncertainty_ns)

    def timestamp_ms(self):
        if self.sample is None:
            raise AdapterError("CLOCK_UNSYNCED")
        wall, mono, offset, _ = self.sample
        current_wall, current_mono = self.wall_ms(), self.monotonic_ns()
        age = current_mono - mono
        if age < 0 or abs((current_wall - wall) * 1_000_000 - age) > 250_000_000:
            self.sample = None
            raise AdapterError("CLOCK_JUMP")
        if age > 60_000_000_000:
            raise AdapterError("CLOCK_STALE")
        return current_wall + offset

    @property
    def uncertainty_ms(self):
        if self.sample is None:
            return None
        whole, fraction = divmod(self.sample[3], 1_000_000)
        return Decimal(f"{whole}.{fraction:06d}")


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes = field(repr=False)
    headers: dict = field(default_factory=dict, repr=False)


class DemoHttpTransport:
    def __init__(self, base_url=BASE_URL):
        try:
            parsed = urlsplit(base_url)
            valid = (isinstance(base_url, str)
                     and base_url in (BASE_URL, "https://demo-api.binance.com:443/api/v3/")
                     and parsed.scheme == "https" and parsed.hostname == "demo-api.binance.com"
                     and parsed.port in (None, 443) and not parsed.username and not parsed.password
                     and parsed.path == "/api/v3/" and not parsed.query and not parsed.fragment)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise AdapterError("FORBIDDEN_ENDPOINT")
        self._connection = None

    def close(self):
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                # A failed socket close must not expose a dependency exception.
                pass

    def request(self, method, path, query, headers, timeout_s):
        if method != "GET" or not isinstance(path, str) or path not in READ_PATHS:
            raise AdapterError("FORBIDDEN_ENDPOINT")
        return self._request_https(method, path, query, headers, timeout_s)

    def _request_https(self, method, path, query, headers, timeout_s, *, matching_engine=False):
        """Shared bounded TLS implementation; public transports own their allowlist."""
        if (not isinstance(path,str) or not ((method=='GET' and path in READ_PATHS)
                or (method=='POST' and path=='/api/v3/order/test')
                or (matching_engine is True and (method,path) in {('POST','/api/v3/order'),
                    ('POST','/api/v3/orderList/oco'),('DELETE','/api/v3/order'),('DELETE','/api/v3/orderList')}))):
            raise AdapterError('FORBIDDEN_ENDPOINT')
        allowed_request_headers = {"X-MBX-APIKEY"} | ({"Accept-Encoding"} if path == "/api/v3/exchangeInfo" else set())
        if (not isinstance(query, str) or any(c in query for c in "\r\n?#")
                or not isinstance(headers, dict) or set(headers) - allowed_request_headers
                or ("Accept-Encoding" in headers and headers["Accept-Encoding"] != "gzip")
                or any(not isinstance(v, str) or any(c in v for c in "\r\n") for v in headers.values())):
            raise AdapterError("INVALID_PARAMETERS")
        if type(timeout_s) not in (int, float) or not 0 < timeout_s <= 15:
            raise AdapterError("INVALID_PARAMETERS")
        connection = None
        completed = False
        body_limit = _body_limit(path)
        deadline = time.monotonic() + timeout_s

        def remaining():
            available = deadline - time.monotonic()
            if available <= 0:
                raise TimeoutError()
            return available

        try:
            if self._connection is None:
                self._connection = http.client.HTTPSConnection("demo-api.binance.com", port=443,
                    timeout=min(5, remaining()), context=ssl.create_default_context())
                self._connection.connect()
            connection = self._connection
            if connection.sock is None:
                connection.timeout = min(5, remaining())
                connection.connect()
            connection.sock.settimeout(remaining())
            target = path + ("?" + query if query else "")
            connection.request(method, target, headers=headers)
            connection.sock.settimeout(remaining())
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise AdapterError("REDIRECT_BLOCKED")
            if hasattr(response, "read1"):
                chunks, size = [], 0
                while True:
                    connection.sock.settimeout(remaining())
                    chunk = response.read1(min(65536, body_limit + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > body_limit:
                        raise AdapterError("RESPONSE_TOO_LARGE")
                body = b"".join(chunks)
            else:
                # Equivalent bound for simple injected protocol-test responses.
                body = response.read(body_limit + 1)
            remaining()
            if len(body) > body_limit:
                raise AdapterError("RESPONSE_TOO_LARGE")
            response_headers = response.getheaders()
            encoding = next((v.lower() for k, v in response_headers if k.lower() == "content-encoding"), "identity")
            if encoding == "gzip" and path == "/api/v3/exchangeInfo":
                try:
                    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
                    body = decoder.decompress(body, body_limit + 1)
                    if len(body) > body_limit or decoder.unconsumed_tail:
                        raise AdapterError("RESPONSE_TOO_LARGE")
                    if not decoder.eof or decoder.unused_data:
                        raise AdapterError("SCHEMA_INVALID")
                except zlib.error:
                    raise AdapterError("SCHEMA_INVALID") from None
            elif encoding != "identity":
                raise AdapterError("SCHEMA_INVALID")
            allowed_headers = {k: v for k, v in response.getheaders()
                               if k.lower() == "retry-after" or k.lower().startswith("x-mbx-used-weight-")}
            completed = True
            return HttpResponse(response.status, body, allowed_headers)
        except AdapterError:
            raise
        except (TimeoutError, OSError, http.client.HTTPException):
            raise AdapterError("TRANSPORT_UNAVAILABLE") from None
        except Exception:
            raise AdapterError("TRANSPORT_FAILURE") from None
        finally:
            if not completed:
                self.close()


def _json(body, max_bytes=BODY_LIMIT):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError()
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError()

    try:
        if not isinstance(body, bytes) or len(body) > max_bytes:
            raise ValueError()
        return json.loads(body, object_pairs_hook=unique, parse_constant=bad_constant)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise AdapterError("SCHEMA_INVALID") from None


def _text(value):
    if not isinstance(value, str) or not value or len(value) > 256 or any(c.isspace() for c in value):
        raise AdapterError("SCHEMA_INVALID")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise AdapterError("SCHEMA_INVALID")
    return value


def _money(value, positive=False):
    if (not isinstance(value, str) or len(value) > 256
            or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) is None):
        raise AdapterError("SCHEMA_INVALID")
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise AdapterError("SCHEMA_INVALID") from None
    if not result.is_finite() or result < 0 or (positive and result == 0) or len(value) > 256:
        raise AdapterError("SCHEMA_INVALID")
    return result


def _bool(value):
    if type(value) is not bool:
        raise AdapterError("SCHEMA_INVALID")
    return value


def _sha(data):
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                            default=lambda v: format(v, "f") if isinstance(v, Decimal) else None)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _signed_read(operation):
    """Every failure closes previous evidence, including normalization errors."""
    @wraps(operation)
    def guarded(self, *args, **kwargs):
        try:
            return operation(self, *args, **kwargs)
        except AdapterError:
            self._invalidate_reads()
            raise
        except Exception:
            self._invalidate_reads()
            raise AdapterError("SCHEMA_INVALID") from None
    return guarded


class DemoReadClient:
    """Credential-free GET engine, optionally shared by the private adapter."""
    def __init__(self, *, transport=None, clock=None, enabled=True, sleep=None, audit=None):
        if type(enabled) is not bool:
            raise AdapterError("INVALID_PARAMETERS")
        self._credentials = None
        self.transport = transport or DemoHttpTransport()
        self.clock = clock or DemoClock()
        self.enabled = enabled
        self.sleep = sleep or time.sleep
        self.blocked = None
        self.cooldown_until_ns = 0
        self._reservations = deque()
        self._shared_weight = 0
        self._shared_weight_expires_ns = 0
        if audit is not None:
            from .interaction_audit import AuditJournal
            if not isinstance(audit, AuditJournal):
                raise AdapterError('AUDIT_IDENTITY_MISMATCH')
        self.audit = audit
        self._transport_counts = dict(attempted_n=0,responses_n=0,errors_n=0)

    def audit_health(self):
        result = (self.audit.health() if self.audit is not None else
                  dict(state='UNCONFIGURED',full_audit_complete=False))
        return dict(result,transport=dict(self._transport_counts))

    def _transport_attempt(self, path, query, headers, params, signed, deadline, request_id, attempt,
                           *, method='GET', transport=None):
        """Intent ACK before send; receipt captured before terminal audit latency."""
        if (not isinstance(path,str) or not ((method=='GET' and path in READ_PATHS)
                or (method=='POST' and path=='/api/v3/order/test' and signed is True))):
            raise AdapterError('FORBIDDEN_ENDPOINT')
        audit, parent, terminal = self.audit, None, None
        attempt_id = uuid.uuid4().hex
        secrets = (() if self._credentials is None else
                   (self._credentials.api_key,self._credentials.api_secret))
        if audit is not None:
            parent = audit.append('REST_BEGIN',dict(request_id=request_id,attempt_id=attempt_id,
                attempt_number=attempt+1,method=method,endpoint=path,parameters=params,private=signed),
                wall_ms=self.clock.wall_ms(),monotonic_ns=self.clock.monotonic_ns(),secrets=secrets)
        response, error = None, None
        wall, mono = self.clock.wall_ms(), self.clock.monotonic_ns()
        sent = False
        try:
            available = (deadline-mono)/1_000_000_000
            if available<=0:raise AdapterError('READ_UNAVAILABLE')
            self._transport_counts['attempted_n']+=1
            sent=True
            target=self.transport if transport is None else transport
            response=target.request(method,path,query,headers,min(15,available))
            self._transport_counts['responses_n']+=1
        except BaseException as caught:
            error=caught
            if sent:self._transport_counts['errors_n']+=1
            raise
        finally:
            received_wall, received_mono = self.clock.wall_ms(), self.clock.monotonic_ns()
            if audit is not None:
                from .interaction_audit import response_payload, rate_headers
                valid=isinstance(response,HttpResponse)
                outcome=('RESPONSE' if error is None else 'TIMEOUT' if isinstance(error,(TimeoutError,OSError))
                         else 'CANCELLED' if isinstance(error,(KeyboardInterrupt,SystemExit)) or not sent
                         else 'TRANSPORT_ERROR')
                terminal = audit.append('REST_END',dict(request_id=request_id,attempt_id=attempt_id,parent_event_id=parent,
                    outcome=outcome,method=method,endpoint=path,private=signed,
                    status=response.status if valid and type(response.status) is int else None,
                    headers=rate_headers(response.headers) if valid else {},
                    sent_at_ms=wall if sent else None,sent_monotonic_ns=mono if sent else None,
                    received_at_ms=received_wall,received_monotonic_ns=received_mono,
                    latency_ns=received_mono-mono if sent and received_mono>=mono else None,
                    payload=response_payload(response.body,private=signed,secrets=secrets) if valid else None),
                    wall_ms=received_wall,monotonic_ns=received_mono,secrets=secrets)
        return response,wall,mono,received_wall,received_mono,attempt_id,parent,terminal

    def request_public(self, path, parameters, *, weight):
        if path not in PUBLIC_PATHS:
            raise AdapterError("FORBIDDEN_ENDPOINT")
        if not isinstance(parameters, dict) or set(parameters) & {"signature", "timestamp", "recvWindow"}:
            raise AdapterError("INVALID_PARAMETERS")
        _integer(weight, 1, "INVALID_PARAMETERS")
        return self._request(path, parameters, signed=False, weight=weight)

    def close(self):
        if hasattr(self.transport, "close"):
            self.transport.close()

    def _guard(self, weight):
        if not self.enabled:
            raise AdapterError("ADAPTER_DISABLED")
        if self.blocked:
            raise AdapterError(self.blocked)
        now = self.clock.monotonic_ns()
        if now < self.cooldown_until_ns:
            raise AdapterError("RATE_LIMITED")
        while self._reservations and now - self._reservations[0][0] >= 60_000_000_000:
            self._reservations.popleft()
        if now >= self._shared_weight_expires_ns:
            self._shared_weight = 0
        local_weight = sum(w for _, w in self._reservations)
        if max(local_weight, self._shared_weight) + weight > 1200:
            raise AdapterError("LOCAL_RATE_BUDGET")
        self._reservations.append((now, weight))
        if self._shared_weight_expires_ns > now:
            self._shared_weight += weight

    def _invalidate_reads(self, state="UNAVAILABLE"):
        self._set_cap("USER_DATA_READ", state)
        self._set_cap("ACCOUNT_SPOT_ALLOWED", "UNVERIFIED")

    def _request(self, path, parameters, *, signed, weight, deadline_ns=None):
        started = self.clock.monotonic_ns()
        request_id = uuid.uuid4().hex
        deadline = deadline_ns or started + 15_000_000_000
        resynced = False
        for attempt in range(3):
            available = (deadline - self.clock.monotonic_ns()) / 1_000_000_000
            if available <= 0:
                raise AdapterError("READ_UNAVAILABLE")
            self._guard(weight)
            params = dict(parameters)
            headers = {}
            if signed:
                if self._credentials is None:
                    raise AdapterError("PRIVATE_READ_UNAVAILABLE")
                try:
                    timestamp = self.clock.timestamp_ms()
                except AdapterError as error:
                    if error.reason not in ("CLOCK_UNSYNCED", "CLOCK_STALE"):
                        raise
                    self.sync_clock(deadline_ns=deadline)
                    timestamp = self.clock.timestamp_ms()
                params.update(timestamp=timestamp, recvWindow=5000)
                query = sign_rest(params, self._credentials.api_secret)
                headers["X-MBX-APIKEY"] = self._credentials.api_key
            else:
                query = urlencode(sorted((k, _value(v)) for k, v in params.items()))
                if path == "/api/v3/exchangeInfo":
                    headers["Accept-Encoding"] = "gzip"
            request_wall, request_mono = self.clock.wall_ms(), self.clock.monotonic_ns()
            available = (deadline - request_mono) / 1_000_000_000
            if available <= 0:
                raise AdapterError("READ_UNAVAILABLE")
            try:
                response,request_wall,request_mono,received_wall,received_mono,attempt_id,audit_parent,audit_end = self._transport_attempt(
                    path,query,headers,params,signed,deadline,request_id,attempt)
            except (TimeoutError, OSError):
                response = None
            except AdapterError as error:
                if error.reason.startswith('AUDIT_') or error.reason=='READ_UNAVAILABLE':
                    raise
                if error.reason in ("REDIRECT_BLOCKED", "RESPONSE_TOO_LARGE", "FORBIDDEN_ENDPOINT", "INVALID_PARAMETERS"):
                    raise
                if error.reason == "TRANSPORT_UNAVAILABLE":
                    response = None
                else:
                    raise AdapterError("TRANSPORT_FAILURE") from None
            except Exception:
                raise AdapterError("TRANSPORT_FAILURE") from None
            if response is None:
                received_wall, received_mono = self.clock.wall_ms(), self.clock.monotonic_ns()
            if received_mono > deadline or received_mono < request_mono:
                raise AdapterError("READ_UNAVAILABLE")
            if response is not None:
                if (not isinstance(response, HttpResponse) or type(response.status) is not int
                        or not isinstance(response.headers, dict)
                        or any(not isinstance(k, str) or not isinstance(v, str)
                               for k, v in response.headers.items())):
                    raise AdapterError("SCHEMA_INVALID")
                lower = {k.lower(): v for k, v in response.headers.items() if isinstance(k, str)}
                weight_header = lower.get("x-mbx-used-weight-1m")
                if isinstance(weight_header, str) and weight_header.isdecimal():
                    self._shared_weight = max(self._shared_weight, int(weight_header))
                    self._shared_weight_expires_ns = received_mono + 60_000_000_000
                if response.status in (429, 418):
                    try:
                        wait = int(lower.get("retry-after", "60"))
                        if wait < 0:
                            raise ValueError()
                    except (TypeError, ValueError):
                        wait = 60
                    self.cooldown_until_ns = received_mono + max(1, wait) * 1_000_000_000
                    if response.status == 418:
                        self.blocked = "IP_BANNED"
                    raise AdapterError(self.blocked or "RATE_LIMITED")
                if 300 <= response.status < 400:
                    raise AdapterError("REDIRECT_BLOCKED")
                if 500 <= response.status <= 599:
                    response = None
                else:
                    data = _json(response.body, _body_limit(path))
                    code = data.get("code") if isinstance(data, dict) else None
                    if response.status in (401, 403) or code in (-2014, -2015, -1022):
                        self.blocked = "SIGNATURE_REJECTED" if code == -1022 else "AUTH_DENIED"
                        raise AdapterError(self.blocked)
                    if code == -1021:
                        if resynced or not signed:
                            raise AdapterError("TIMESTAMP_REJECTED")
                        self.clock.sample = None
                        self.sync_clock(deadline_ns=deadline)
                        resynced = True
                        continue
                    if response.status != 200 or (type(code) is int and code < 0):
                        raise AdapterError("API_REJECTED")
                    return data, dict({"raw_body_sha256": hashlib.sha256(response.body).hexdigest()} if not signed else {},
                                      **(dict(audit_request_id=request_id,audit_attempt_id=attempt_id,
                                              audit_begin_event_id=audit_parent,audit_end_event_id=audit_end) if self.audit is not None else {}),
                                      source_endpoint=path, request_started_at_ms=request_wall,
                                      received_at_ms=received_wall, received_monotonic_ns=received_mono,
                                      request_started_monotonic_ns=request_mono)
            if attempt < 2:
                delay = (0.25, 0.5)[attempt]
                if self.clock.monotonic_ns() + round(delay * 1_000_000_000) >= deadline:
                    break
                self.sleep(delay)
        raise AdapterError("READ_UNAVAILABLE")

    def sync_clock(self, *, deadline_ns=None):
        self.clock.sample = None
        deadline = deadline_ns or self.clock.monotonic_ns() + 15_000_000_000
        for attempt in range(3):
            data, meta = self._request("/api/v3/time", {}, signed=False, weight=1, deadline_ns=deadline)
            if not isinstance(data, dict):
                raise AdapterError("SCHEMA_INVALID")
            try:
                server = _integer(data["serverTime"])
            except KeyError:
                raise AdapterError("SCHEMA_INVALID") from None
            try:
                self.clock.observe(server, meta["request_started_at_ms"], meta["received_at_ms"],
                                   meta["request_started_monotonic_ns"], meta["received_monotonic_ns"])
            except AdapterError as error:
                if error.reason != "CLOCK_QUALITY" or attempt == 2:
                    raise
                continue
            return {"server_time_ms": server, "clock_uncertainty_ms": self.clock.uncertainty_ms, **meta}

class BinanceDemoAdapter(DemoReadClient):
    def __init__(self, credentials, *, account_alias, credential_generation, episode_id,
                 scope_id, identity_key, transport=None, clock=None, enabled=True, sleep=None, audit=None):
        if not isinstance(credentials, DemoCredentials):
            raise AdapterError("INVALID_CREDENTIALS")
        for value in (credentials.api_key, credentials.api_secret):
            if (not isinstance(value, str) or not value or not value.isascii()
                    or any(c.isspace() or ord(c) < 33 or ord(c) == 127 for c in value)
                    or "PRIVATE KEY" in value or "-----BEGIN" in value):
                raise AdapterError("INVALID_CREDENTIALS")
        try:
            for value in (account_alias, credential_generation, episode_id, scope_id):
                _text(value)
            if not isinstance(identity_key, bytes) or len(identity_key) < 32 or type(enabled) is not bool:
                raise AdapterError("INVALID_IDENTITY")
        except AdapterError:
            raise AdapterError("INVALID_IDENTITY") from None
        self._credentials = credentials
        self._identity_key = identity_key
        self.identity = dict(application_id="binance_demo_bot", environment="BINANCE_SPOT_DEMO",
                             objective_contract_id="daily_net_equity_pnl_v1", account_alias=account_alias,
                             credential_generation=credential_generation, account_episode_id=episode_id,
                             portfolio_scope_id=scope_id)
        super().__init__(transport=transport, clock=clock, enabled=enabled, sleep=sleep,audit=audit)
        if audit is not None and any(audit.identity.get(k)!=self.identity[k]
                for k in ('account_alias','account_episode_id','portfolio_scope_id')):
            raise AdapterError('AUDIT_IDENTITY_MISMATCH')
        self._credentials = credentials
        self._cap_states = {}

    def __repr__(self):
        return "<BinanceDemoAdapter read_only=True>"

    def ping(self):
        data, meta = self._request("/api/v3/ping", {}, signed=False, weight=1)
        if data != {}:
            raise AdapterError("SCHEMA_INVALID")
        return {"reachable": True, **self.identity, **meta, "order_send_enabled": False}

    def _set_cap(self, name, state, source_hash=None):
        self._cap_states[name] = dict(state=state, checked_at_ms=self.clock.wall_ms(),
            checked_monotonic_ns=self.clock.monotonic_ns(), expires_at_ms=self.clock.wall_ms() + 30_000,
            source_sha256=source_hash, source="P1_A_REST", **self.identity)

    def capabilities(self):
        result = {}
        clock_state = None
        try:
            self.clock.timestamp_ms()
        except AdapterError as error:
            clock_state = "UNAVAILABLE" if error.reason == "CLOCK_JUMP" else "UNVERIFIED"
        for name in ("USER_DATA_READ", "ACCOUNT_SPOT_ALLOWED", "KEY_TRADE_VALIDATED", "PRIVATE_EVENTS", "ORDER_SEND"):
            entry = dict(self._cap_states.get(name, dict(state="UNVERIFIED", checked_at_ms=None,
                         expires_at_ms=None, source_sha256=None, source="P1_A_NOT_PROBED", **self.identity)))
            checked = entry.pop("checked_monotonic_ns", None)
            if checked is not None and not 0 <= self.clock.monotonic_ns() - checked <= 30_000_000_000:
                entry["state"] = "UNVERIFIED"
            if clock_state and entry["state"] == "SUPPORTED":
                entry["state"] = clock_state
            if name == "ORDER_SEND":
                entry["state"] = "DENIED"
                entry["source"] = "P1_READ_ONLY_CONTRACT"
            if not self.enabled or self.blocked or self.clock.monotonic_ns() < self.cooldown_until_ns:
                if name not in ("ORDER_SEND", "KEY_TRADE_VALIDATED", "PRIVATE_EVENTS"):
                    entry["state"] = "UNAVAILABLE"
            result[name] = entry
        return result

    def _observation(self, data, meta):
        return {**self.identity, **meta, "schema_version": 1, "source_method": "GET",
                "data": data, "source_sha256": _sha(data), "clock_uncertainty_ms": self.clock.uncertainty_ms,
                "state": "ACCOUNT_UNRECONCILED", "exchange_reconciled": False}

    def _event_time(self, value):
        timestamp = _integer(value)
        if timestamp > self.clock.timestamp_ms() + 250:
            raise AdapterError("SCHEMA_INVALID")
        return timestamp

    @_signed_read
    def read_account(self):
        try:
            raw, meta = self._request("/api/v3/account", {"omitZeroBalances": False}, signed=True, weight=20)
            if not isinstance(raw, dict) or raw.get("accountType") != "SPOT" or not isinstance(raw.get("balances"), list):
                raise AdapterError("SCHEMA_INVALID")
            balances, seen = [], set()
            for row in raw["balances"]:
                asset = _text(row["asset"])
                if asset in seen:
                    raise AdapterError("SCHEMA_INVALID")
                seen.add(asset)
                balances.append(dict(asset=asset, free=_money(row["free"]), locked=_money(row["locked"])))
            can_trade = _bool(raw["canTrade"]) if "canTrade" in raw else None
            permissions = raw.get("permissions")
            if permissions is not None:
                if not isinstance(permissions, list) or any(not isinstance(p, str) or not p for p in permissions):
                    raise AdapterError("SCHEMA_INVALID")
                permissions = sorted(set(permissions))
            uid = raw.get("uid")
            account_identity = None
            if uid is not None:
                _integer(uid)
                account_identity = hmac.new(self._identity_key, b"BINANCE_SPOT_DEMO:" + str(uid).encode(), hashlib.sha256).hexdigest()
            data = dict(balances=balances, returned_assets=sorted(seen), coverage="RETURNED_BALANCES_ONLY",
                        can_trade=can_trade, permissions=permissions, account_identity=account_identity,
                        update_time_ms=self._event_time(raw["updateTime"]))
            result = self._observation(data, meta)
            self._set_cap("USER_DATA_READ", "SUPPORTED", result["source_sha256"])
            allowed = "UNVERIFIED"
            if can_trade is False or (permissions is not None and "SPOT" not in permissions):
                allowed = "DENIED"
            elif can_trade is True and permissions is not None and "SPOT" in permissions:
                allowed = "SUPPORTED"
            self._set_cap("ACCOUNT_SPOT_ALLOWED", allowed, result["source_sha256"])
            return result
        except (KeyError, TypeError, ValueError):
            self._invalidate_reads()
            raise AdapterError("SCHEMA_INVALID") from None
        except AdapterError:
            self._invalidate_reads()
            raise

    def _order(self, row):
        if not isinstance(row, dict):
            raise AdapterError("SCHEMA_INVALID")
        statuses = {"NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", "PENDING_CANCEL", "REJECTED", "EXPIRED", "EXPIRED_IN_MATCH", "PENDING_NEW"}
        types = {"LIMIT", "MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT", "TAKE_PROFIT", "TAKE_PROFIT_LIMIT", "LIMIT_MAKER"}
        if row["status"] not in statuses or row["side"] not in ("BUY", "SELL") or row["type"] not in types:
            raise AdapterError("SCHEMA_INVALID")
        orig, executed = _money(row["origQty"]), _money(row["executedQty"])
        if orig > 0 and executed > orig:
            raise AdapterError("SCHEMA_INVALID")
        return dict(symbol=_text(row["symbol"]), order_id=_integer(row["orderId"]),
                    order_list_id=_integer(row["orderListId"], -1), client_order_id=_text(row["clientOrderId"]),
                    status=row["status"], side=row["side"], type=row["type"], price=_money(row["price"]),
                    original_quantity=orig, executed_quantity=executed,
                    cumulative_quote_quantity=_money(row["cummulativeQuoteQty"]),
                    time_ms=self._event_time(row["time"]), update_time_ms=self._event_time(row["updateTime"]),
                    ownership="UNATTRIBUTED")

    @_signed_read
    def read_open_orders(self, symbol=None):
        params = {} if symbol is None else {"symbol": self._symbol(symbol)}
        try:
            raw, meta = self._request("/api/v3/openOrders", params, signed=True, weight=80 if symbol is None else 6)
            if not isinstance(raw, list):
                raise AdapterError("SCHEMA_INVALID")
            data = [self._order(row) for row in raw]
            keys = [(r["symbol"], r["order_id"]) for r in data]
            if len(set(keys)) != len(keys) or (symbol is not None and any(r["symbol"] != symbol for r in data)):
                raise AdapterError("SCHEMA_INVALID")
            return self._observation(data, meta)
        except (KeyError, TypeError, ValueError):
            raise AdapterError("SCHEMA_INVALID") from None

    @_signed_read
    def read_order(self, symbol, order_id):
        symbol = self._symbol(symbol)
        _integer(order_id, reason="INVALID_PARAMETERS")
        try:
            raw, meta = self._request("/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True, weight=4)
            data = self._order(raw)
            if data["symbol"] != symbol or data["order_id"] != order_id:
                raise AdapterError("SCHEMA_INVALID")
            return self._observation(data, meta)
        except (KeyError, TypeError, ValueError):
            raise AdapterError("SCHEMA_INVALID") from None

    @staticmethod
    def _symbol(value):
        try:
            return _text(value)
        except AdapterError:
            raise AdapterError("INVALID_PARAMETERS") from None

    def _order_list(self, row):
        if (not isinstance(row, dict) or row["contingencyType"] not in {"OCO", "OTO", "OTOCO", "OPO", "OPOCO"}
                or row["listStatusType"] not in {"RESPONSE", "EXEC_STARTED", "ALL_DONE", "UPDATED"}
                or row["listOrderStatus"] not in {"EXECUTING", "ALL_DONE", "REJECT"}
                or not isinstance(row["orders"], list) or not row["orders"]):
            raise AdapterError("SCHEMA_INVALID")
        symbol = _text(row["symbol"])
        orders = [dict(symbol=_text(item["symbol"]), order_id=_integer(item["orderId"]),
                       client_order_id=_text(item["clientOrderId"])) for item in row["orders"]]
        if any(o["symbol"] != symbol for o in orders) or len({o["order_id"] for o in orders}) != len(orders):
            raise AdapterError("SCHEMA_INVALID")
        return dict(order_list_id=_integer(row["orderListId"]), symbol=symbol,
                    contingency_type=row["contingencyType"], list_status_type=row["listStatusType"],
                    list_order_status=row["listOrderStatus"], list_client_order_id=_text(row["listClientOrderId"]),
                    transaction_time_ms=self._event_time(row["transactionTime"]), orders=orders, ownership="UNATTRIBUTED")

    @_signed_read
    def read_open_order_lists(self):
        raw, meta = self._request("/api/v3/openOrderList", {}, signed=True, weight=6)
        if not isinstance(raw, list):
            raise AdapterError("SCHEMA_INVALID")
        data = [self._order_list(row) for row in raw]
        if len({row["order_list_id"] for row in data}) != len(data):
            raise AdapterError("SCHEMA_INVALID")
        return self._observation(data, meta)

    def _fill(self, row, symbol):
        if not isinstance(row, dict) or row.get("symbol") != symbol:
            raise AdapterError("SCHEMA_INVALID")
        return dict(symbol=symbol, trade_id=_integer(row["id"]), order_id=_integer(row["orderId"]),
                    order_list_id=_integer(row.get("orderListId", -1), -1), price=_money(row["price"], True),
                    quantity=_money(row["qty"], True), quote_quantity=_money(row["quoteQty"], True),
                    commission_quantity=_money(row["commission"]), commission_asset=_text(row["commissionAsset"]),
                    side="BUY" if _bool(row["isBuyer"]) else "SELL", is_maker=_bool(row["isMaker"]),
                    exchange_event_time_ms=self._event_time(row["time"]), ownership="UNATTRIBUTED")

    @_signed_read
    def recover_fills(self, symbol, *, from_id, page_size=1000, max_pages=100):
        symbol = self._symbol(symbol)
        _integer(from_id, reason="INVALID_PARAMETERS")
        if (type(page_size) is not int or not 1 <= page_size <= 1000 or type(max_pages) is not int
                or not 1 <= max_pages <= 100):
            raise AdapterError("INVALID_PARAMETERS")
        cursor, collected, observations = from_id, {}, []
        exhausted = False
        try:
            for _ in range(max_pages):
                raw, meta = self._request("/api/v3/myTrades", {"symbol": symbol, "fromId": cursor, "limit": page_size},
                                          signed=True, weight=20)
                if not isinstance(raw, list) or len(raw) > page_size:
                    raise AdapterError("SCHEMA_INVALID")
                data = [self._fill(row, symbol) for row in raw]
                for fill in data:
                    key = fill["trade_id"]
                    if key < cursor:
                        raise AdapterError("PAGINATION_INVALID")
                    if key in collected and collected[key] != fill:
                        raise AdapterError("FILL_CONFLICT")
                    collected[key] = fill
                observations.append(self._observation(data, meta))
                if data:
                    cursor = max(f["trade_id"] for f in data) + 1
                if len(raw) < page_size:
                    exhausted = True
                    break
            result = self._observation([collected[k] for k in sorted(collected)], observations[-1])
            result.update(symbol=symbol, tail_exhausted=exhausted, completeness="SYMBOL_TAIL_EXHAUSTED" if exhausted else "PARTIAL",
                          next_from_id=cursor, starting_from_id=from_id, coverage="QUERIED_SYMBOL_CURSOR_ONLY",
                          page_observations=observations)
            return result
        except (KeyError, TypeError, ValueError):
            raise AdapterError("SCHEMA_INVALID") from None
