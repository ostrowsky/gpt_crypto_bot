"""Owned Demo market streams and bounded incremental books; no execution."""

from collections import deque
from copy import deepcopy
from decimal import Context, Decimal, localcontext
from functools import wraps
import time
import json
import logging
import re
import ssl
import threading
import uuid

from websockets.sync.client import connect as ws_connect

from .adapter import AdapterError, _integer, _money, _sha, _json
from .journal import _digest
from .market_data import MarketDataService, _symbol
from .market_delivery import DeliveryMetrics, WirePacket, lag_evidence

WS_URL = "wss://demo-stream.binance.com/stream"
WS_LIMIT = 1024*1024
TOPIC = re.compile(r"(\w{2,64})@(bookTicker|depth@100ms|kline_1m)\Z")


def _topics(value):
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 512:
        raise AdapterError("INVALID_PARAMETERS")
    if any(not isinstance(t, str) or not TOPIC.fullmatch(t) or t.split("@",1)[0] != t.split("@",1)[0].lower() for t in value) or len(set(value)) != len(value):
        raise AdapterError("INVALID_PARAMETERS")
    return list(value)


def stream_plan(universe, deep_symbols=(), *, include_candles=False):
    if type(include_candles) is not bool:
        raise AdapterError("INVALID_PARAMETERS")
    if not isinstance(deep_symbols, (list, tuple)) or len(set(deep_symbols)) > 32:
        raise AdapterError("INVALID_PARAMETERS")
    rows = universe["data"]["rows"]
    population = {r["symbol"] for r in rows if r["status"] == "TRADING"
                  and r["spot_allowed"] is True and r["permitted"] is not False}
    deep = sorted(set(_symbol(s) for s in deep_symbols))
    if not set(deep) <= population:
        raise AdapterError("UNMAPPED_SYMBOL")
    topics = sorted([s.lower()+"@bookTicker" for s in population] + [s.lower()+"@depth@100ms" for s in deep]
                    + ([s.lower()+"@kline_1m" for s in population] if include_candles else []))
    shards = [dict(shard_id=_digest(dict(universe=universe["snapshot_id"], topics=topics[i:i+512])),
                   topics=topics[i:i+512]) for i in range(0, len(topics), 512)]
    return dict(universe_snapshot_id=universe["snapshot_id"], shards=shards, deep_symbols=deep,
                discovered_pairs_n=universe["data"]["counts"]["discovered_pairs_n"],
                permitted_pairs_N=universe["data"]["counts"]["permitted_pairs_N"],
                subscribed_pairs_n=len(population), registry_version="combined-demo-v1")


def _universe_projection(value):
    result = {k:v for k,v in value.items() if k != "data"}
    result["data"] = dict(state=value["data"]["state"],counts=dict(value["data"]["counts"]),
        rows=[{k:r[k] for k in ("symbol","base","quote","status","spot_allowed","permitted")} for r in value["data"]["rows"]])
    return result


def _verified_market_tls(context=None):
    if context is None:
        try: context = ssl.create_default_context()
        except Exception: raise AdapterError("WS_UNAVAILABLE") from None
    if (not isinstance(context,ssl.SSLContext) or context.protocol != ssl.PROTOCOL_TLS_CLIENT
            or context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname
            or context.minimum_version < ssl.TLSVersion.TLSv1_2):
        raise AdapterError("TLS_UNSAFE")
    return context


class DemoMarketWsTransport:
    def __init__(self, url=WS_URL, *, ssl_context=None):
        if url not in (WS_URL, "wss://demo-stream.binance.com:443/stream"):
            raise AdapterError("FORBIDDEN_ENDPOINT")
        self._connection = None
        self._ssl_context = _verified_market_tls(ssl_context) if ssl_context is not None else None
        self._logger = logging.Logger("binance_demo_bot.market_transport")
        self._logger.disabled, self._logger.propagate = True, False
        self._controls = deque()

    def connect(self):
        if self._connection is not None:
            raise AdapterError("WS_ALREADY_CONNECTED")
        connection = None
        try:
            self._ssl_context = _verified_market_tls(self._ssl_context)
            connection = ws_connect(WS_URL, ssl=self._ssl_context, proxy=None,
                open_timeout=5, close_timeout=2, compression=None, max_size=WS_LIMIT,
                max_queue=16, ping_interval=None, ping_timeout=5, logger=self._logger, legacy=True)
            if (connection.request.path != "/stream" or connection.request.headers.get("Host")
                    not in ("demo-stream.binance.com", "demo-stream.binance.com:443")):
                raise AdapterError("HANDSHAKE_TARGET_MISMATCH")
            self._connection = connection
        except Exception as error:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            raise AdapterError(error.reason if isinstance(error, AdapterError) else "WS_UNAVAILABLE") from None

    def send(self, value):
        if not isinstance(value, dict) or value.get("method") != "SUBSCRIBE":
            raise AdapterError("FORBIDDEN_METHOD")
        if set(value) != {"id", "method", "params"} or not isinstance(value["id"], str) or not re.fullmatch(r"[a-zA-Z0-9]{1,36}", value["id"]):
            raise AdapterError("INVALID_PARAMETERS")
        _topics(value["params"])
        if self._connection is None:
            raise AdapterError("WS_UNAVAILABLE")
        import time
        now = time.monotonic_ns()
        while self._controls and now-self._controls[0] >= 1_000_000_000:
            self._controls.popleft()
        if len(self._controls) >= 2:
            raise AdapterError("WS_RATE_LIMITED")
        self._controls.append(now)
        try:
            self._connection.send(json.dumps(value, separators=(",", ":")))
        except Exception:
            raise AdapterError("WS_UNAVAILABLE") from None

    def recv(self, timeout_s=1):
        if self._connection is None or type(timeout_s) not in (int, float) or not 0 < timeout_s <= 15:
            raise AdapterError("WS_UNAVAILABLE")
        try:
            value = self._connection.recv(timeout=timeout_s)
            size = len(value.encode()) if isinstance(value, str) else len(value) if isinstance(value, bytes) else WS_LIMIT+1
            if size > WS_LIMIT:
                raise AdapterError("WS_BUFFER_OVERFLOW")
            return value
        except TimeoutError:
            raise
        except Exception as error:
            raise AdapterError(error.reason if isinstance(error, AdapterError) else "WS_UNAVAILABLE") from None

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


def _book_atomic(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class IncrementalBook:
    def __init__(self, symbol, clock, *, max_buffer=1024, max_levels=20000):
        self.symbol, self.clock = _symbol(symbol), clock
        if type(max_buffer) is not int or not 1 <= max_buffer <= 4096 or type(max_levels) is not int or not 2 <= max_levels <= 20000:
            raise AdapterError("INVALID_PARAMETERS")
        self.max_buffer, self.max_levels = max_buffer, max_levels
        self._lock = threading.RLock()
        self.state, self.reasons = "BUFFERING", []
        self.bids, self.asks, self.buffer, self.seen = {}, {}, deque(), {}
        self.last_id, self.event_ms, self.received_ms, self.received_ns = None, None, None, None

    @_book_atomic
    def invalidate(self, reason):
        self.state, self.reasons = "INVALID", [reason]
        self.bids.clear()
        self.asks.clear()
        self.buffer.clear()
        self.seen.clear()
        self.last_id = self.event_ms = self.received_ms = self.received_ns = None

    def _age(self, event, wall, mono):
        _integer(event)
        _integer(wall)
        _integer(mono)
        server = self.clock.timestamp_ms()
        with localcontext(Context(prec=50)):
            uncertainty = self.clock.uncertainty_ms
            if Decimal(event) > Decimal(server)+uncertainty:
                raise AdapterError("L2_FUTURE_EVENT")
            if Decimal(server-event)+uncertainty > 1000:
                raise AdapterError("L2_STALE_EVENT")
        if not 0 <= self.clock.wall_ms()-wall <= 1000 or not 0 <= self.clock.monotonic_ns()-mono <= 1_000_000_000:
            raise AdapterError("L2_STALE_RECEIPT")

    @staticmethod
    def _levels(values):
        if not isinstance(values, list):
            raise AdapterError("L2_SCHEMA_INVALID")
        result = {}
        for row in values:
            if not isinstance(row, list) or len(row) != 2:
                raise AdapterError("L2_SCHEMA_INVALID")
            price, qty = _money(row[0], True), _money(row[1])
            if price in result:
                raise AdapterError("L2_SCHEMA_INVALID")
            result[price] = qty
        return result

    @_book_atomic
    def on_diff(self, payload, *, received_at_ms, received_monotonic_ns):
        try:
            if not isinstance(payload, dict) or set(payload) != {"e", "E", "s", "U", "u", "b", "a"} or payload["e"] != "depthUpdate" or payload["s"] != self.symbol:
                raise AdapterError("L2_SCHEMA_INVALID")
            first, final = _integer(payload["U"]), _integer(payload["u"])
            if final < first:
                raise AdapterError("L2_SCHEMA_INVALID")
            bids, asks = self._levels(payload["b"]), self._levels(payload["a"])
            self._age(payload["E"], received_at_ms, received_monotonic_ns)
            event = dict(wire=deepcopy(payload), bids=bids, asks=asks, wall=received_at_ms, mono=received_monotonic_ns)
            if self.last_id is None:
                if len(self.buffer) >= self.max_buffer:
                    raise AdapterError("L2_BUFFER_OVERFLOW")
                self.buffer.append(event)
            else:
                self._apply(event)
        except (AdapterError, KeyError, TypeError, ValueError) as error:
            reason = error.reason if isinstance(error, AdapterError) and error.reason.startswith(("L2_", "CLOCK_")) else "L2_SCHEMA_INVALID"
            self.invalidate(reason)
        return self.view()

    def _apply(self, event):
        payload = event["wire"]
        first, final = payload["U"], payload["u"]
        fingerprint = _sha(payload)
        if final in self.seen and self.seen[final] != fingerprint:
            raise AdapterError("L2_CONFLICT")
        if final <= self.last_id:
            return
        if first > self.last_id+1:
            raise AdapterError("L2_GAP")
        if self.event_ms is not None and payload["E"] < self.event_ms:
            raise AdapterError("L2_EVENT_REGRESSION")
        self._age(payload["E"], event["wall"], event["mono"])
        for target, updates in ((self.bids,event["bids"]),(self.asks,event["asks"])):
            for price, qty in updates.items():
                if qty == 0:
                    target.pop(price, None)
                else:
                    target[price] = qty
        if not self.bids or not self.asks or max(self.bids) >= min(self.asks):
            raise AdapterError("L2_CROSSED_OR_EMPTY")
        if len(self.bids)+len(self.asks) > self.max_levels:
            raise AdapterError("L2_LEVEL_OVERFLOW")
        self.last_id, self.event_ms = final, payload["E"]
        self.received_ms, self.received_ns = event["wall"], event["mono"]
        self.state, self.reasons = "READY", []
        self.seen[final] = fingerprint
        if len(self.seen) > self.max_buffer:
            del self.seen[next(iter(self.seen))]

    @_book_atomic
    def install_snapshot(self, observation):
        try:
            params = observation["request_parameters"]
            raw = observation["data"]
            if (observation["environment"] != "BINANCE_SPOT_DEMO" or observation["source_endpoint"] != "/api/v3/depth"
                    or observation["source_method"] != "GET" or params != {"symbol":self.symbol,"limit":params.get("limit")}
                    or type(params["limit"]) is not int or not 1 <= params["limit"] <= 1000
                    or observation["source_sha256"] != _sha(raw)
                    or not 0 <= self.clock.wall_ms()-observation["received_at_ms"] <= 1000
                    or not 0 <= self.clock.monotonic_ns()-_integer(observation["received_monotonic_ns"]) <= 1_000_000_000
                    or observation["request_started_at_ms"] > observation["received_at_ms"]):
                raise AdapterError("L2_SNAPSHOT_INVALID")
            self.clock.timestamp_ms()
            bids, asks = self._levels(raw["bids"]), self._levels(raw["asks"])
            if len(bids) > params["limit"] or len(asks) > params["limit"] or len(bids)+len(asks) > self.max_levels:
                raise AdapterError("L2_SNAPSHOT_INVALID")
            if not bids or not asks or any(q <= 0 for q in list(bids.values())+list(asks.values())) or max(bids) >= min(asks):
                raise AdapterError("L2_SNAPSHOT_INVALID")
            pending = list(self.buffer)
            self.buffer.clear()
            self.seen.clear()
            self.bids, self.asks = bids, asks
            self.last_id, self.event_ms = _integer(raw["lastUpdateId"]), None
            self.received_ms = self.received_ns = None
            self.state, self.reasons = "BUFFERING", []
            for event in pending:
                self._apply(event)
        except (AdapterError, KeyError, TypeError, ValueError) as error:
            self.invalidate(error.reason if isinstance(error, AdapterError) else "L2_SNAPSHOT_INVALID")
        return self.view()

    @_book_atomic
    def view(self):
        if self.state == "READY":
            try:
                self._age(self.event_ms, self.received_ms, self.received_ns)
            except AdapterError as error:
                self.invalidate(error.reason)
        return dict(symbol=self.symbol, state=self.state, reasons=list(self.reasons), last_update_id=self.last_id,
                    event_time_ms=self.event_ms, received_at_ms=self.received_ms,
                    received_monotonic_ns=self.received_ns, clock_uncertainty_ms=self.clock.uncertainty_ms,
                    receipt_age_ms=self.clock.wall_ms()-self.received_ms if self.received_ms is not None else None,
                    bids=[[format(p,"f"),format(self.bids[p],"f")] for p in sorted(self.bids,reverse=True)],
                    asks=[[format(p,"f"),format(self.asks[p],"f")] for p in sorted(self.asks)],
                    buffer_n=len(self.buffer), full_book_certified=False)


class MarketStreamReader:
    def __init__(self, store, client, universe, topics, *, transport=None, background=True, latency_window=4096):
        if type(background) is not bool:
            raise AdapterError("INVALID_PARAMETERS")
        MarketDataService(store,client)._universe(universe,client.clock.wall_ms())
        self._initialize(store,client,_universe_projection(universe),topics,transport=transport,background=background,latency_window=latency_window)

    @classmethod
    def _from_verified_source(cls,store,client,projection,topics,*,transport=None,background=True):
        # Called only with supervisor-owned membership copied after full validation.
        store.validate_reference(projection["snapshot_id"],"UNIVERSE",client.clock.wall_ms())
        result=cls.__new__(cls)
        result._initialize(store,client,projection,topics,transport=transport,background=background)
        return result

    def _initialize(self,store,client,universe,topics,*,transport,background,latency_window=4096):
        self._delivery = DeliveryMetrics(latency_window)
        self._lag_evidence_buffer = None
        self.invalidation_state = "NOT_OBSERVED"
        selected = _topics(topics)
        symbol_map = {r["symbol"].lower():r["symbol"] for r in universe["data"]["rows"]}
        if len(symbol_map) != len(universe["data"]["rows"]):
            raise AdapterError("UNMAPPED_SYMBOL")
        if any(TOPIC.fullmatch(t).group(1) not in symbol_map for t in selected):
            raise AdapterError("UNMAPPED_SYMBOL")
        mapped = {t:symbol_map[TOPIC.fullmatch(t).group(1)] for t in selected}
        allowed = {t for shard in stream_plan(universe, [mapped[t] for t in selected if t.endswith("@depth@100ms")],include_candles=any(t.endswith("@kline_1m") for t in selected))["shards"] for t in shard["topics"]}
        if not set(selected) <= allowed:
            raise AdapterError("UNMAPPED_SYMBOL")
        self.store, self.client, self.clock = store,client,client.clock
        self.universe, self.topics = _universe_projection(universe),tuple(selected)
        self.topic_symbols = mapped
        self.members = {r["symbol"]:r for r in self.universe["data"]["rows"]}
        self.transport, self.background = transport or DemoMarketWsTransport(),background
        self.books = {mapped[t]:IncrementalBook(mapped[t],self.clock) for t in selected if t.endswith("@depth@100ms")}
        self.quote_symbols = {mapped[t] for t in selected if t.endswith("@bookTicker")}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self.connected, self.session_id, self.error = False,None,None
        self._quotes, self._depth_ids = {},{}
        self._blocked_quotes = set()
        self.last_processing_lag_ms = None
        self.processing_lags = deque(maxlen=4096)
        self.open_candle_packets_n = self.closed_candle_packets_n = 0
        self._failed_candle_symbol = None
        self._failed_candle_reason = "CANDLE_SCHEMA_INVALID"
        self._pending_store = None
        self.started_ns = self.last_ping_ns = None

    def _save(self, kind, data, wall=None, mono=None, extra=()):
        return (self._pending_store or self.store).append(kind, dict(data, session_id=self.session_id),
            received_at_ms=self.clock.wall_ms() if wall is None else wall,
            received_monotonic_ns=self.clock.monotonic_ns() if mono is None else mono,
            dependencies=[self.universe["snapshot_id"],*extra])

    def _invalidate_quotes(self, reason):
        rows = [dict(symbol=s,state="QUOTE_INVALID",quote=None,reasons=[reason]) for s in sorted(self.quote_symbols)]
        self._quotes.clear()
        for book in self.books.values():
            book.invalidate(reason)
        if rows:
            self._save("QUOTES", dict(rows=rows, source="DEMO_COMBINED_STREAM"))

    def _fail(self, reason, *, evidence=None):
        safe = reason if reason in {"WS_UNAVAILABLE","WS_PROTOCOL_ERROR","WS_BUFFER_OVERFLOW","WS_HEARTBEAT_FAILED","WS_PROCESSING_LAG","WS_STOPPED","WS_ROTATED","METADATA_STALE","CLOCK_STALE","CLOCK_UNSYNCED","CLOCK_JUMP","CLOCK_QUALITY","WS_RATE_LIMITED","CANDLE_SOURCE_CHANGED"} else "WS_PROTOCOL_ERROR"
        with self._lock:
            self.error, self.connected = safe,False
            self._quotes.clear()
            for book in self.books.values():
                book.invalidate(safe)
            previous_pending = self._pending_store
            try:
                with self.store.prepared_batch(monotonic_ns=self.clock.monotonic_ns) as failure_pending:
                    self._pending_store = failure_pending
                    try:
                        self._invalidate_quotes(safe)
                        if evidence is not None:
                            self._save("INCIDENT",evidence)
                        self._save("INCIDENT",dict(operation="MARKET_STREAM_FAILED",reason=safe,topics=list(self.topics)))
                        if self._failed_candle_symbol:
                            self._save("INCIDENT",dict(operation="CANDLE_COLLECTION_FAILED",symbol=self._failed_candle_symbol,reason=self._failed_candle_reason))
                    finally:
                        self._pending_store = previous_pending
                self._failed_candle_symbol = None
                self._failed_candle_reason = "CANDLE_SCHEMA_INVALID"
                self.invalidation_state = "PERSISTED"
                if evidence is not None:
                    self._delivery.evidence_state = "PERSISTED"
                    self._lag_evidence_buffer = None
            except Exception:
                self.invalidation_state = "PERSISTENCE_FAILED"
                if evidence is not None:
                    self._delivery.evidence_failures += 1
                    self._delivery.evidence_state = "PERSISTENCE_FAILED"
                raise
            finally:
                self._pending_store = previous_pending
                self.transport.close()

    def _receive(self, timeout=1):
        wire = self.transport.recv(timeout)
        wall, mono = self.clock.wall_ms(),self.clock.monotonic_ns()
        value = wire.encode() if isinstance(wire,str) else wire
        if not isinstance(value,bytes) or len(value) > WS_LIMIT:
            raise AdapterError("WS_BUFFER_OVERFLOW")
        payload = _json(value)
        return (WirePacket(payload, value) if isinstance(payload, dict) else payload),wall,mono

    def start(self):
        with self._lock:
            if self.connected or self._thread and self._thread.is_alive():
                raise AdapterError("WS_ALREADY_CONNECTED")
            self.store.validate_reference(self.universe["snapshot_id"],"UNIVERSE",self.clock.wall_ms())
            if self.clock.wall_ms()-self.universe["received_at_ms"] > 900000:
                raise AdapterError("METADATA_STALE")
            self.clock.timestamp_ms()
            self.session_id, self.error = uuid.uuid4().hex,None
            self._quotes.clear()
            self._blocked_quotes.clear()
            self._depth_ids.clear()
            self._invalidate_quotes("WAIT_FOR_STREAM")
            self._stop.clear()
            try:
                self.transport.connect()
                request_id = uuid.uuid4().hex
                self.transport.send(dict(id=request_id,method="SUBSCRIBE",params=list(self.topics)))
                deadline = self.clock.monotonic_ns()+5_000_000_000
                buffered = []
                # Also cap iterations for deterministic injected transports.
                for _ in range(65):
                    if self.clock.monotonic_ns() >= deadline:
                        raise AdapterError("WS_PROTOCOL_ERROR")
                    payload,wall,mono = self._receive(min(1,(deadline-self.clock.monotonic_ns())/1e9))
                    if isinstance(payload,dict) and "id" in payload:
                        if payload != {"result":None,"id":request_id}:
                            raise AdapterError("WS_PROTOCOL_ERROR")
                        break
                    buffered.append((payload,wall,mono))
                    if len(buffered) > 64:
                        raise AdapterError("WS_BUFFER_OVERFLOW")
                else:
                    raise AdapterError("WS_PROTOCOL_ERROR")
                self.connected = True
                self.started_ns = self.last_ping_ns = self.clock.monotonic_ns()
                for payload,wall,mono in buffered:
                    self._process(payload,wall,mono)
                if self.background:
                    self._thread = threading.Thread(target=self._run,name="binance-demo-market-reader",daemon=True)
                    self._thread.start()
            except Exception as error:
                self._fail(error.reason if isinstance(error,AdapterError) else "WS_PROTOCOL_ERROR")
                raise AdapterError(self.error) from None

    def _process(self, payload, wall, mono):
        if not isinstance(payload,dict) or set(payload) != {"stream","data"} or payload["stream"] not in self.topics:
            raise AdapterError("WS_PROTOCOL_ERROR")
        match = TOPIC.fullmatch(payload["stream"])
        symbol, kind = self.topic_symbols[payload["stream"]],match.group(2)
        wire = payload["data"]
        if not isinstance(wire,dict) or wire.get("s") != symbol:
            raise AdapterError("WS_PROTOCOL_ERROR")
        lag_ns = self.clock.monotonic_ns()-mono
        with localcontext(Context(prec=50)):
            self.last_processing_lag_ms = Decimal(lag_ns)/1_000_000
        self.processing_lags.append(self.last_processing_lag_ms)
        if not 0 <= lag_ns <= 250_000_000:
            raise AdapterError("WS_PROCESSING_LAG")
        if kind == "bookTicker":
            if symbol in self._blocked_quotes:
                return
            state,quote,reasons = "QUOTE_INVALID",None,[]
            fingerprint = _sha(wire)
            try:
                if set(wire) != {"s","u","b","B","a","A"}:
                    raise AdapterError("QUOTE_SCHEMA_INVALID")
                identifier = _integer(wire["u"])
                prior = self._quotes.get(symbol)
                if prior:
                    if identifier < prior["id"] or identifier == prior["id"] and fingerprint != prior["sha"]:
                        raise AdapterError("QUOTE_SEQUENCE_CONFLICT")
                    if identifier == prior["id"]:
                        return
                bid,ask = _money(wire["b"],True),_money(wire["a"],True)
                bqty,aqty = _money(wire["B"],True),_money(wire["A"],True)
                if bid > ask:
                    raise AdapterError("CROSSED_QUOTE")
                member = self.members[symbol]
                quote = dict(base=member["base"],quote=member["quote"],bid=bid,ask=ask,
                    bid_quantity=bqty,ask_quantity=aqty,venue="BINANCE_SPOT_DEMO",event_time_ms=None,
                    time_basis="OBSERVED_RESPONSE",convention_id="observed-stream-book-v1",
                    exchange_age_unknown=True,timestamp_state="TIMESTAMP_UNVERIFIED",source_healthy=True,
                    received_at_ms=wall,response_started_at_ms=wall)
                state = "QUOTE_READY"
                self._quotes[symbol] = dict(id=identifier,sha=fingerprint,wall=wall,mono=mono)
            except (AdapterError,KeyError,TypeError) as error:
                reasons = [error.reason if isinstance(error,AdapterError) else "QUOTE_SCHEMA_INVALID"]
                self._quotes.pop(symbol,None)
                self._blocked_quotes.add(symbol)
            self._save("QUOTES", dict(rows=[dict(symbol=symbol,state=state,quote=quote,reasons=reasons)],
                                      wire=wire,source="DEMO_COMBINED_STREAM"),wall,mono)
        elif kind == "kline_1m":
            try:
                result = MarketDataService(self._pending_store or self.store,self.client).record_closed_kline(wire,received_at_ms=wall,received_monotonic_ns=mono)
            except AdapterError:
                self._failed_candle_symbol = symbol
                raise
            if result is None:
                self.open_candle_packets_n += 1
            else:
                self.closed_candle_packets_n += 1
        else:
            book = self.books[symbol]
            view = book.on_diff(wire,received_at_ms=wall,received_monotonic_ns=mono)
            top = {k:v for k,v in view.items() if k not in {"bids","asks"}}
            top.update(best_bid=view["bids"][0] if view["bids"] else None,best_ask=view["asks"][0] if view["asks"] else None)
            extra = [self._depth_ids[symbol]] if symbol in self._depth_ids else []
            saved = self._save("DEPTH", dict(symbol=symbol,operation="L2_DIFF",wire=wire,view=top),wall,mono,extra)
            self._depth_ids[symbol] = saved["snapshot_id"]

    def install_depth(self, symbol, observation):
        with self._lock:
            if not self.connected or symbol not in self.books:
                raise AdapterError("UNMAPPED_SYMBOL")
            # Validate before making a receipt authoritative, then persist it.
            view = self.books[symbol].install_snapshot(observation)
            extra = [self._depth_ids[symbol]] if symbol in self._depth_ids else []
            saved = self._save("DEPTH",dict(symbol=symbol,operation="L2_SNAPSHOT",observation=observation,
                    state=view["state"],reasons=view["reasons"]),observation["received_at_ms"],observation["received_monotonic_ns"],extra)
            self._depth_ids[symbol] = saved["snapshot_id"]
            return view

    def read_once(self):
        if not self.connected:
            return False
        pending = None
        noted = False
        frames = []
        collected = acquired = None
        try:
            if self.clock.wall_ms()-self.universe["received_at_ms"] > 900000:
                raise AdapterError("METADATA_STALE")
            self.clock.timestamp_ms()
            if self.clock.monotonic_ns()-self.last_ping_ns >= 20_000_000_000:
                if not self.transport.ping(5):
                    raise AdapterError("WS_HEARTBEAT_FAILED")
                self.last_ping_ns = self.clock.monotonic_ns()
            payload,wall,mono = self._receive()
            frames = [(payload,wall,mono)]
            deadline = time.monotonic()+0.002
            while len(frames) < 8 and time.monotonic() < deadline:
                try:
                    frames.append(self._receive(0.001))
                except TimeoutError:
                    break
            collected = self.clock.monotonic_ns()
            with self._lock:
                acquired = self.clock.monotonic_ns()
                if self.connected:
                    pending = self.store.prepared_batch(monotonic_ns=self.clock.monotonic_ns)
                    try:
                        with pending:
                            self._pending_store = pending
                            for payload,wall,mono in frames:
                                self._process(payload,wall,mono)
                    finally:
                        self._pending_store = None
                    finished = self.clock.monotonic_ns()
                    elapsed = finished-min(frame[2] for frame in frames)
                    reason = None if 0 <= elapsed <= 250_000_000 else "WS_PROCESSING_LAG"
                    group = self._delivery.note(frames,pending,collected,acquired,finished,reason,
                        self.universe["snapshot_id"],self.session_id)
                    noted = True
                    if reason:
                        evidence = self._record_lag_evidence(frames,pending,group)
                        self._fail(reason,evidence=evidence)
                        return False
                else:
                    return False
            return True
        except TimeoutError:
            if self.clock.monotonic_ns()-self.last_ping_ns >= 20_000_000_000:
                try:
                    if not self.transport.ping(5):
                        raise AdapterError("WS_HEARTBEAT_FAILED")
                    self.last_ping_ns = self.clock.monotonic_ns()
                except Exception:
                    self._fail("WS_HEARTBEAT_FAILED")
            return False
        except Exception as error:
            if pending is not None and pending.conflict_symbol:
                self._failed_candle_symbol = pending.conflict_symbol
                self._failed_candle_reason = "CANDLE_SOURCE_CHANGED"
            reason = error.reason if isinstance(error,AdapterError) else "WS_PROTOCOL_ERROR"
            with self._lock:
                evidence = None
                if frames and not noted:
                    group = self._delivery.note(frames,pending,collected,acquired,self.clock.monotonic_ns(),reason,
                        self.universe["snapshot_id"],self.session_id)
                    if reason=="WS_PROCESSING_LAG": evidence = self._record_lag_evidence(frames,pending,group)
                if not noted or self.connected:
                    try: self._fail(reason,evidence=evidence)
                    except Exception: pass  # In-memory invalidation/close and explicit health failure already set.
            return False

    def _record_lag_evidence(self, frames, pending, group):
        try:
            self._lag_evidence_buffer = lag_evidence(frames,pending,group)
            self._delivery.evidence_state = "PENDING"
            return self._lag_evidence_buffer
        except Exception:
            self._delivery.evidence_failures += 1
            self._delivery.evidence_state = "PERSISTENCE_FAILED"
            return None

    def _run(self):
        try:
            while not self._stop.is_set() and self.connected:
                self.read_once()
        finally:
            self.store.release_thread_reader()

    def health(self):
        with self._lock:
            now,mono = self.clock.wall_ms(),self.clock.monotonic_ns()
            clock_state = "VALID"
            try:
                self.clock.timestamp_ms()
            except AdapterError as error:
                clock_state = error.reason
            fresh = sum(0 <= now-q["wall"] <= 5000 and 0 <= mono-q["mono"] <= 5_000_000_000 for q in self._quotes.values()) if self.connected and clock_state == "VALID" else 0
            views = {s:book.view() for s,book in self.books.items()}
            return dict(connected=self.connected,error=self.error,session_id=self.session_id,
                        thread_alive=bool(self._thread and self._thread.is_alive()),fresh_quotes_n=fresh,
                        subscribed_quote_pairs_N=len(self.quote_symbols),ready_depth_n=sum(v["state"]=="READY" for v in views.values()),
                        depth_pairs_N=len(views),depth_states={s:v["state"] for s,v in views.items()},
                        depth_diagnostics={s:{k:v[k] for k in ("event_time_ms","received_at_ms","receipt_age_ms","clock_uncertainty_ms","reasons")} for s,v in views.items()},
                        needs_reconnect=bool(self._blocked_quotes),last_processing_lag_ms=self.last_processing_lag_ms,
                        clock_state=clock_state,
                        open_candle_packets_n=self.open_candle_packets_n,closed_candle_packets_n=self.closed_candle_packets_n,
                        processing_lag_recent_n=len(self.processing_lags),processing_lag_window_limit_N=4096,
                        processing_lag_p50_ms=sorted(self.processing_lags)[max(0,(len(self.processing_lags)+1)//2-1)] if self.processing_lags else None,
                        processing_lag_p95_ms=sorted(self.processing_lags)[max(0,(len(self.processing_lags)*95+99)//100-1)] if self.processing_lags else None,
                        processing_lag_max_ms=max(self.processing_lags) if self.processing_lags else None,
                        latency=self._delivery.health(),
                        invalidation_state=self.invalidation_state,
                        order_send_enabled=False,account_certified=False)

    def stop(self):
        self._stop.set()
        if self.session_id is not None and self.connected:
            self._fail("WS_STOPPED")
        else:
            self.transport.close()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(3)
            if self._thread.is_alive():
                raise AdapterError("WS_STOP_TIMEOUT")


class MarketStreamSupervisor:
    def __init__(self, store, client, universe, deep_symbols=(), *, transport_factory=None,
                 background=False, enabled=True, include_candles=False):
        MarketDataService(store,client)._universe(universe,client.clock.wall_ms())
        if (type(background) is not bool or type(enabled) is not bool
                or transport_factory is not None and not callable(transport_factory)):
            raise AdapterError("INVALID_PARAMETERS")
        self.store,self.client,self.universe = store,client,_universe_projection(universe)
        self.plan = stream_plan(universe,deep_symbols,include_candles=include_candles)
        self._default_factory = transport_factory is None
        self._ssl_context = None
        self.transport_factory = self._default_transport if self._default_factory else transport_factory
        self.background,self.enabled = background,enabled
        self.readers, self.attempts, self.slots = {},deque(),{}
        self._lock,self._stop,self._thread = threading.RLock(),threading.Event(),None
        self.running = False

    def _prepare_tls(self):
        with self._lock:
            self._ssl_context = _verified_market_tls(self._ssl_context)
            return self._ssl_context

    def _default_transport(self):
        return DemoMarketWsTransport(ssl_context=self._prepare_tls())

    def start(self):
        if not self.enabled:
            raise AdapterError("MARKET_DATA_DISABLED")
        if self.running:
            raise AdapterError("WS_ALREADY_CONNECTED")
        try:
            self.client.clock.timestamp_ms()
        except AdapterError as error:
            if error.reason not in {"CLOCK_UNSYNCED","CLOCK_STALE"}:
                raise
            self.client.client.sync_clock()
        if self._default_factory: self._prepare_tls()
        self.running = True
        self._stop.clear()
        self.tick()
        if self.background:
            self._thread = threading.Thread(target=self._run,name="binance-demo-market-supervisor",daemon=True)
            self._thread.start()

    def tick(self):
        if not self.running or self._stop.is_set():
            return
        with self._lock:
            now = self.client.clock.monotonic_ns()
            if self.client.clock.wall_ms()-self.universe["received_at_ms"] > 900000:
                for reader in self.readers.values():
                    reader._fail("METADATA_STALE")
                return
            try:
                self.client.clock.timestamp_ms()
            except AdapterError as error:
                if error.reason in {"CLOCK_UNSYNCED","CLOCK_STALE"}:
                    try:
                        self.client.client.sync_clock()
                    except AdapterError:
                        for reader in self.readers.values():
                            reader._fail("CLOCK_UNSYNCED")
                        return
                else:
                    for reader in self.readers.values():
                        reader._fail(error.reason)
                    return
            while self.attempts and now-self.attempts[0] >= 300_000_000_000:
                self.attempts.popleft()
            for shard in self.plan["shards"]:
                if self._stop.is_set():
                    return
                key = shard["shard_id"]
                slot = self.slots.setdefault(key,dict(next_ns=0,failures=0,depth_next={},failed_session=None))
                reader = self.readers.get(key)
                if reader and reader.connected and reader.health()["needs_reconnect"]:
                    reader._fail("WS_PROTOCOL_ERROR")
                if reader and reader.connected and now-reader.started_ns >= 23*3600*1_000_000_000:
                    reader._fail("WS_ROTATED")
                if not reader or not reader.connected:
                    if reader and slot["failed_session"] != reader.session_id:
                        slot["failed_session"] = reader.session_id
                        slot["failures"] += 1
                        slot["next_ns"] = now+min(60,2**min(slot["failures"]-1,6))*1_000_000_000
                    if now < slot["next_ns"] or len(self.attempts) >= 20:
                        continue
                    if reader:
                        reader.stop()
                    self.attempts.append(now)
                    reader = MarketStreamReader._from_verified_source(self.store,self.client,self.universe,shard["topics"],
                                                transport=self.transport_factory(),background=True)
                    self.readers[key] = reader
                    try:
                        reader.start()
                        slot["depth_next"].clear()
                    except AdapterError:
                        slot["failed_session"] = reader.session_id
                        slot["failures"] += 1
                        slot["next_ns"] = now+min(60,2**min(slot["failures"]-1,6))*1_000_000_000
                        continue
                for symbol,book in reader.books.items():
                    if self._stop.is_set():
                        return
                    view = book.view()
                    if view["state"] == "READY" or now < slot["depth_next"].get(symbol,0):
                        continue
                    # Allow at least one buffered diff before fetching a snapshot.
                    if view["last_update_id"] is not None or not view["buffer_n"]:
                        continue
                    slot["depth_next"][symbol] = now+1_000_000_000
                    try:
                        observation = self.client.read_depth(symbol)
                        if self._stop.is_set():
                            return
                        reader.install_depth(symbol,observation)
                    except AdapterError as error:
                        self.store.append("INCIDENT",dict(operation="L2_SNAPSHOT_READ_FAILED",symbol=symbol,reason=error.reason),
                            received_at_ms=self.client.clock.wall_ms(),received_monotonic_ns=self.client.clock.monotonic_ns())

    def _run(self):
        while not self._stop.wait(0.25):
            try:
                self.tick()
            except Exception:
                for reader in self.readers.values():
                    reader._fail("WS_PROTOCOL_ERROR")
                self.running = False
                break

    def health(self):
        with self._lock:
            states = {key:reader.health() for key,reader in self.readers.items()}
            return dict(running=self.running,connected_shards_n=sum(s["connected"] for s in states.values()),
                        planned_shards_N=len(self.plan["shards"]),fresh_quotes_n=sum(s["fresh_quotes_n"] for s in states.values()),
                        permitted_pairs_N=self.plan["permitted_pairs_N"],subscribed_pairs_n=self.plan["subscribed_pairs_n"],
                        ready_depth_n=sum(s["ready_depth_n"] for s in states.values()),deep_pairs_N=len(self.plan["deep_symbols"]),
                        connection_attempts_n=len(self.attempts),connection_attempt_limit_N=20,shards=states,
                        order_send_enabled=False,account_certified=False)

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(20)
            if self._thread.is_alive():
                raise AdapterError("WS_STOP_TIMEOUT")
        with self._lock:
            self.running = False
            for reader in self.readers.values():
                reader.stop()
