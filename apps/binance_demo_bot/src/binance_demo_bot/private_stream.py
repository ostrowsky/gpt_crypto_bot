"""Owned HMAC private-event reader. Transport health never certifies money."""

from decimal import Decimal
import hashlib
import hmac
import json
import logging
import ssl
import threading
import time
import uuid

from websockets.sync.client import connect as ws_connect

from .adapter import AdapterError, BinanceDemoAdapter, _value, _integer, _text, _json, _money
from .ws_audit import AuditedWsTransport, bind_ws_audit, receipt_of, disposition
from .market_delivery import WirePacket
from .account_evidence import AccountEvidenceStore, INCIDENTS


WS_URL = "wss://demo-ws-api.binance.com/ws-api/v3"
WS_LIMIT = 1024 * 1024
METHODS = frozenset(("time", "ping", "session.subscriptions", "userDataStream.subscribe.signature"))


def sign_ws(parameters, secret):
    try:
        pairs = list(parameters.items()) if isinstance(parameters, dict) else list(parameters)
        names = [p[0] for p in pairs]
        if (any(not isinstance(n, str) or not n or n == "signature" for n in names)
                or len(names) != len(set(names)) or not isinstance(secret, str) or not secret):
            raise AdapterError("INVALID_PARAMETERS")
        text = "&".join(f"{name}={_value(value)}" for name, value in sorted(pairs))
        return hmac.new(secret.encode("utf-8"), text.encode("utf-8"), hashlib.sha256).hexdigest()
    except (TypeError, ValueError, UnicodeError):
        raise AdapterError("INVALID_PARAMETERS") from None


class DemoWsTransport:
    def __init__(self, url=WS_URL):
        if not isinstance(url, str) or url not in (WS_URL, "wss://demo-ws-api.binance.com:443/ws-api/v3"):
            raise AdapterError("FORBIDDEN_ENDPOINT")
        self._connection = None
        self._logger = logging.Logger("binance_demo_bot.private_transport")
        self._logger.disabled = True
        self._logger.propagate = False
        self._protocol_audit=None

    def set_protocol_audit(self,observer):
        if self._connection is not None:raise AdapterError('WS_ALREADY_CONNECTED')
        self._protocol_audit=observer

    def connect(self):
        if self._connection is not None:
            raise AdapterError("WS_ALREADY_CONNECTED")
        connection = None
        try:
            factory={} if self._protocol_audit is None else {'create_connection':self._protocol_audit.connection_factory()}
            connection = ws_connect(WS_URL, ssl=ssl.create_default_context(), proxy=None,
                open_timeout=5, close_timeout=2, compression=None, max_size=WS_LIMIT,
                max_queue=64, ping_interval=None, ping_timeout=5, logger=self._logger, legacy=True,**factory)
            request = connection.request
            if (request.path != "/ws-api/v3" or request.headers.get("Host") not in
                    ("demo-ws-api.binance.com", "demo-ws-api.binance.com:443")):
                raise AdapterError("HANDSHAKE_TARGET_MISMATCH")
            self._connection = connection
        except AdapterError:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            raise
        except Exception:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            raise AdapterError("WS_UNAVAILABLE") from None

    def send(self, value):
        if not isinstance(value, dict) or value.get("method") not in METHODS:
            raise AdapterError("FORBIDDEN_METHOD")
        if self._connection is None:
            raise AdapterError("WS_UNAVAILABLE")
        try:
            _text(value["id"])
            if set(value) - {"id", "method", "params"}:
                raise AdapterError("INVALID_PARAMETERS")
            serialized = json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
                default=lambda x: format(x, "f") if isinstance(x, Decimal) else None)
            if len(serialized.encode()) > WS_LIMIT:
                raise AdapterError("WS_BUFFER_OVERFLOW")
            self._connection.send(serialized)
        except AdapterError:
            raise
        except Exception:
            raise AdapterError("WS_UNAVAILABLE") from None

    def recv(self, timeout_s=1):
        if self._connection is None or type(timeout_s) not in (int, float) or not 0 < timeout_s <= 15:
            raise AdapterError("WS_UNAVAILABLE")
        try:
            value = self._connection.recv(timeout=timeout_s)
            if not isinstance(value, (str, bytes)) or len(value.encode() if isinstance(value, str) else value) > WS_LIMIT:
                raise AdapterError("WS_BUFFER_OVERFLOW")
            return value
        except TimeoutError:
            raise
        except AdapterError:
            raise
        except Exception:
            raise AdapterError("WS_UNAVAILABLE") from None

    def ping(self, timeout_s=5):
        if self._connection is None:
            raise AdapterError("WS_UNAVAILABLE")
        try:
            return self._connection.ping().wait(timeout_s)
        except Exception:
            raise AdapterError("WS_UNAVAILABLE") from None

    def close(self):
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


class PrivateEventReader:
    def __init__(self, adapter, store, *, transport=None, process_instance_id, background=True):
        if not isinstance(adapter, BinanceDemoAdapter) or not isinstance(store, AccountEvidenceStore):
            raise AdapterError("EVIDENCE_IDENTITY_MISMATCH")
        _text(process_instance_id)
        if type(background) is not bool:
            raise AdapterError("INVALID_PARAMETERS")
        self.adapter, self.store = adapter, store
        self.transport = transport or DemoWsTransport()
        self.transport = bind_ws_audit(self.transport,adapter.audit,adapter.clock,store.identity,
            private=True,credentials=lambda:adapter._credentials)
        self.process_id, self.background = process_instance_id, background
        self.session_id, self.subscription_id = None, None
        self._connected, self._last_pong_ns = False, None
        self._error = None
        self._stop = threading.Event()
        self._thread = None

    def __repr__(self):
        return "<PrivateEventReader read_only=True>"

    def _receive(self, timeout):
        value = self.transport.recv(timeout)
        receipt=receipt_of(self.transport,self.adapter.clock)
        wall, mono = receipt['wall_ms'],receipt['monotonic_ns']
        if isinstance(value, str):
            value = value.encode("utf-8")
        try:
            if not isinstance(value, bytes) or len(value) > WS_LIMIT:
                raise AdapterError("WS_BUFFER_OVERFLOW")
            payload=_json(value)
            if not isinstance(payload,dict):raise AdapterError('WS_PROTOCOL_ERROR')
        except AdapterError:
            disposition(self.transport,receipt['event_id'],'REJECTED',if_healthy=True)
            raise
        if isinstance(payload,dict):
            payload=WirePacket(payload,value)
            payload.audit_event_id=receipt['event_id']
        return payload, wall, mono

    def _fail(self, reason):
        safe_reason = reason if reason.startswith('AUDIT_') or reason in INCIDENTS else "WS_PROTOCOL_ERROR"
        self._error, self._connected = safe_reason, False
        self._stop.set()
        try:
            self.transport.close()
        except Exception:
            pass
        # Evidence store keeps its established incident vocabulary. The precise
        # audit failure remains in reader/transport health and journal gaps.
        self.store.record_incident('STREAM_GAP' if safe_reason.startswith('AUDIT_') else safe_reason)

    def _observe_limits(self, data):
        for limit in data.get("rateLimits", []):
            if (limit.get("rateLimitType") == "REQUEST_WEIGHT" and limit.get("interval") == "MINUTE"
                    and limit.get("intervalNum") == 1):
                used = _integer(limit["count"])
                self.adapter._shared_weight = max(self.adapter._shared_weight, used)
                self.adapter._shared_weight_expires_ns = self.adapter.clock.monotonic_ns()+60_000_000_000
        if data.get("status") in (429, 418):
            retry = data.get("retryAfter", data.get("error", {}).get("data", {}).get("retryAfter"))
            delay = 60000
            if type(retry) is int and retry > self.adapter.clock.wall_ms():
                delay = retry-self.adapter.clock.wall_ms()
            self.adapter.cooldown_until_ns = self.adapter.clock.monotonic_ns()+delay*1_000_000
            if data["status"] == 418:
                self.adapter.blocked = "IP_BANNED"

    def start(self):
        if self._connected or self._thread is not None and self._thread.is_alive():
            raise AdapterError("WS_ALREADY_CONNECTED")
        try:
            anchor = self.adapter.read_account()
            if anchor["data"]["account_identity"] != self.store.identity["account_identity"]:
                raise AdapterError("ACCOUNT_IDENTITY_MISMATCH")
            if any(anchor[k] != self.store.identity[k] for k in ("account_alias", "account_episode_id", "portfolio_scope_id")):
                raise AdapterError("EVIDENCE_NAMESPACE_MISMATCH")
            self.adapter._guard(4)  # Connect + signature subscription; own shared-IP limiter.
            self.transport.connect()
            request_id = uuid.uuid4().hex
            params = dict(apiKey=self.adapter._credentials.api_key,
                          timestamp=self.adapter.clock.timestamp_ms(), recvWindow=5000)
            params["signature"] = sign_ws(params, self.adapter._credentials.api_secret)
            self.transport.send(dict(id=request_id, method="userDataStream.subscribe.signature", params=params))
            deadline = self.adapter.clock.monotonic_ns()+15_000_000_000
            pending = []
            while True:
                remaining = (deadline-self.adapter.clock.monotonic_ns())/1_000_000_000
                if remaining <= 0:
                    raise AdapterError("WS_PROTOCOL_ERROR")
                data, wall, mono = self._receive(min(15, remaining))
                if isinstance(data, dict) and "event" in data:
                    if len(pending) >= 64:
                        raise AdapterError("WS_BUFFER_OVERFLOW")
                    pending.append((data, wall, mono))
                    continue
                self._accept_ack(data,request_id)
                break
            self._stop.clear()
            self._connected, self._error = True, None
            for value in pending:
                self._accept(*value)
            self.heartbeat()
            if self.background:
                self._thread = threading.Thread(target=self._run, name="binance-demo-private-reader", daemon=True)
                self._thread.start()
            return self.health()
        except AdapterError as error:
            self._fail(error.reason)
            safe = error.reason if error.reason.startswith('AUDIT_') or error.reason in INCIDENTS | {
                "ADAPTER_DISABLED", "AUTH_DENIED", "SIGNATURE_REJECTED", "CLOCK_UNSYNCED", "CLOCK_STALE",
                "CLOCK_QUALITY", "CLOCK_JUMP", "READ_UNAVAILABLE", "API_REJECTED", "TIMESTAMP_REJECTED",
                "LOCAL_RATE_BUDGET", "RATE_LIMITED", "IP_BANNED", "HANDSHAKE_TARGET_MISMATCH",
                "EVIDENCE_NAMESPACE_MISMATCH"} else "WS_PROTOCOL_ERROR"
            raise AdapterError(safe) from None
        except Exception:
            self._fail("WS_PROTOCOL_ERROR")
            raise AdapterError("WS_PROTOCOL_ERROR") from None

    def _accept_ack(self,data,request_id):
        try:
            if not isinstance(data,dict) or data.get('id')!=request_id:raise AdapterError('WS_PROTOCOL_ERROR')
            self._observe_limits(data)
            if data.get('status') in (401,403):raise AdapterError('WS_AUTH_DENIED')
            if data.get('status')==429:raise AdapterError('WS_RATE_LIMITED')
            if data.get('status')==418:raise AdapterError('WS_BANNED')
            if data.get('status')!=200 or not isinstance(data.get('result'),dict):raise AdapterError('WS_PROTOCOL_ERROR')
            self.subscription_id=_integer(data['result']['subscriptionId'])
            self.session_id=uuid.uuid4().hex
            self.store.bind_session(self.session_id,self.subscription_id,self.adapter.identity['credential_generation'])
            disposition(self.transport,getattr(data,'audit_event_id',None),'CONTROL_ACK')
        except Exception:
            disposition(self.transport,getattr(data,'audit_event_id',None),'REJECTED',if_healthy=True)
            raise

    def _normalize(self, raw):
        if not isinstance(raw, dict):
            raise AdapterError("SCHEMA_INVALID")
        kind = raw.get("e")
        exchange = self.adapter._event_time(raw["E"])
        if kind == "executionReport":
            execution = raw["x"]
            if execution not in {"NEW", "CANCELED", "REPLACED", "REJECTED", "TRADE", "EXPIRED", "TRADE_PREVENTION"}:
                raise AdapterError("SCHEMA_INVALID")
            order = self.adapter._order(dict(symbol=raw["s"], orderId=raw["i"], orderListId=raw["g"],
                clientOrderId=raw["c"], status=raw["X"], side=raw["S"], type=raw["o"], price=raw["p"],
                origQty=raw["q"], executedQty=raw["z"], cummulativeQuoteQty=raw["Z"],
                time=raw["O"], updateTime=raw["T"]))
            fill = None
            if execution == "TRADE":
                if _money(raw["n"]) != 0:
                    _text(raw["N"])
                fill = self.adapter._fill(dict(symbol=raw["s"], id=raw["t"], orderId=raw["i"], orderListId=raw["g"],
                    price=raw["L"], qty=raw["l"], quoteQty=raw["Y"], commission=raw["n"], commissionAsset=raw["N"] or "USDT",
                    time=raw["T"], isBuyer=raw["S"] == "BUY", isMaker=raw["m"]), raw["s"])
            elif _money(raw["l"]) != 0 or _money(raw["n"]) != 0:
                raise AdapterError("SCHEMA_INVALID")
            return dict(kind="ORDER_UPDATE", exchange_event_time_ms=exchange, transaction_time_ms=order["update_time_ms"],
                        execution_id=_integer(raw["I"]), order=order, fill=fill)
        if kind == "outboundAccountPosition":
            if not isinstance(raw["B"], list):
                raise AdapterError("SCHEMA_INVALID")
            return dict(kind="PARTIAL_BALANCES", exchange_event_time_ms=exchange, update_time_ms=self.adapter._event_time(raw["u"]),
                        balances=[dict(asset=_text(r["a"]), free=_money(r["f"]), locked=_money(r["l"])) for r in raw["B"]])
        if kind == "balanceUpdate":
            delta = raw["d"]
            magnitude = _money(delta[1:] if isinstance(delta, str) and delta.startswith("-") else delta)
            delta = Decimal("-"+format(magnitude, "f")) if delta.startswith("-") else magnitude
            return dict(kind="FLOW", exchange_event_time_ms=exchange, transaction_time_ms=self.adapter._event_time(raw["T"]),
                        asset=_text(raw["a"]), delta=delta)
        if kind == "listStatus":
            value = self.adapter._order_list(dict(orderListId=raw["g"], contingencyType=raw["c"],
                listStatusType=raw["l"], listOrderStatus=raw["L"], listClientOrderId=raw["C"],
                transactionTime=raw["T"], symbol=raw["s"],
                orders=[dict(symbol=o["s"], orderId=o["i"], clientOrderId=o["c"]) for o in raw["O"]]))
            return dict(kind="LIST_UPDATE", exchange_event_time_ms=exchange, list=value)
        if kind == "eventStreamTerminated":
            raise AdapterError("STREAM_TERMINATED")
        if kind == "externalLockUpdate":
            raise AdapterError("EXTERNAL_LOCK")
        raise AdapterError("UNKNOWN_EVENT")

    def _accept(self, data, wall, mono):
        try:return self._accept_record(data,wall,mono)
        except Exception:
            disposition(self.transport,getattr(data,'audit_event_id',None),'REJECTED',if_healthy=True)
            raise

    def _accept_record(self, data, wall, mono):
        if not isinstance(data, dict) or type(data.get("subscriptionId")) is not int or data["subscriptionId"] != self.subscription_id:
            raise AdapterError("UNMAPPED_SUBSCRIPTION")
        try:
            event = self._normalize(data["event"])
            seq = self.store.record_event(self.session_id, self.subscription_id, event,
                        received_at_ms=wall, received_monotonic_ns=mono, process_instance_id=self.process_id)
            receipt=getattr(data,'audit_event_id',None)
            if receipt and isinstance(self.transport,AuditedWsTransport):
                self.transport.processing([receipt],normalized_record_seq=seq,store_identity=self.store.identity,
                    database=self.store.path.name)
            return seq
        except (KeyError, TypeError, ValueError, ArithmeticError):
            raise AdapterError("SCHEMA_INVALID") from None

    def poll_once(self, timeout_s=1):
        if not self._connected:
            raise AdapterError("WS_UNAVAILABLE")
        payload=None
        try:
            payload,wall,mono=self._receive(timeout_s)
            return self._accept(payload,wall,mono)
        except TimeoutError:
            return None
        except AdapterError as error:
            self._fail(error.reason)
            safe = error.reason if error.reason.startswith('AUDIT_') or error.reason in INCIDENTS else "WS_PROTOCOL_ERROR"
            raise AdapterError(safe) from None
        except Exception:
            self._fail("WS_UNAVAILABLE")
            raise AdapterError("WS_UNAVAILABLE") from None

    def heartbeat(self):
        try:
            if not self._connected or not self.transport.ping(5):
                raise AdapterError("WS_HEARTBEAT_FAILED")
            self._last_pong_ns = self.adapter.clock.monotonic_ns()
        except Exception as error:
            reason=error.reason if isinstance(error,AdapterError) and error.reason.startswith('AUDIT_') else 'WS_HEARTBEAT_FAILED'
            self._fail(reason)
            raise AdapterError(reason) from None

    def _run(self):
        while not self._stop.is_set():
            try:
                result = self.poll_once(1)
                if self._last_pong_ns is None or self.adapter.clock.monotonic_ns()-self._last_pong_ns >= 15_000_000_000:
                    self.heartbeat()
                if result is None:
                    self._stop.wait(0.01)
            except Exception:
                return

    def health(self):
        age = None if self._last_pong_ns is None else self.adapter.clock.monotonic_ns()-self._last_pong_ns
        return dict(connected=self._connected, transport_healthy=self._connected and age is not None and 0 <= age <= 30_000_000_000,
                    private_events_certified=False, exchange_reconciled=False, order_send_enabled=False,
                    session_id=self.session_id, subscription_id=self.subscription_id, error_reason=self._error,
                    audit=self.transport.health() if isinstance(self.transport,AuditedWsTransport) else dict(state='UNCONFIGURED',full_audit_complete=False),
                    reader_alive=self._thread is not None and self._thread.is_alive())

    def close(self):
        was_connected = self._connected
        self._connected = False
        self._stop.set()
        self.transport.close()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                self.store.record_incident("READER_STOP_TIMEOUT")
        if was_connected:
            self.store.record_incident("STREAM_GAP")
