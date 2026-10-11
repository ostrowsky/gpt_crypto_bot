"""Single-owner read-only market coordinator; public evidence never certifies money."""

from collections import deque
from decimal import Context, Decimal, localcontext
import threading
import time

from .adapter import AdapterError, DemoClock, _integer
from .market_data import MarketDataService, _quote_reasons
from .market_stream import MarketStreamSupervisor, _universe_projection
from .universe import UniverseService


def _history_ready(rows, cutoff):
    selected = sorted((r for r,_ in rows.values() if r["closed"] and r["close_time_ms"] < cutoff),key=lambda r:r["open_time_ms"])[-3:]
    return (len(selected)==3 and all(r["valid"] for r in selected)
        and all(selected[i]["open_time_ms"]-selected[i-1]["open_time_ms"]==60000 for i in (1,2))
        and 0 <= cutoff-selected[-1]["close_time_ms"] <= 90000)


class MarketDataRuntime:
    def __init__(self,store,client,*,account_provider=None,deep_symbols=(),transport_factory=None,
                 enabled=True,background=False):
        if type(enabled) is not bool or type(background) is not bool or account_provider is not None and not callable(account_provider):
            raise AdapterError("INVALID_PARAMETERS")
        self.store,self.client,self.clock = store,client,client.clock
        self.universe_service,self.service = UniverseService(store,client),MarketDataService(store,client)
        self.account_provider,self.deep_symbols,self.transport_factory = account_provider,tuple(deep_symbols),transport_factory
        self.enabled,self.background = enabled,background
        self.universe,self.supervisor = None,None
        self._projection = None
        self.pipeline_state,self.running = "NOT_STARTED",False
        self._stop,self._lock,self._thread = threading.Event(),threading.RLock(),None
        self._attempts = deque()
        self._metadata_due_ns,self._quote_due_ns = 0,0
        self._backfill_cursor,self._backfill_due = 0,{}
        self._started_ns = None
        self._tick_lags = deque(maxlen=4096)
        self._incidents_n = 0

    def _incident(self,operation,reason):
        self._incidents_n += 1
        self.store.append("INCIDENT",dict(operation=operation,reason=reason),
            received_at_ms=self.clock.wall_ms(),received_monotonic_ns=self.clock.monotonic_ns())

    def refresh_clock(self):
        candidate = DemoClock(wall_ms=self.clock.wall_ms,monotonic_ns=self.clock.monotonic_ns)
        deadline = self.clock.monotonic_ns()+15_000_000_000
        for attempt in range(3):
            observation = self.client._read("/api/v3/time",{},1)
            raw = observation["data"]
            try:
                candidate.observe(_integer(raw["serverTime"]),observation["request_started_at_ms"],observation["received_at_ms"],
                    observation["request_started_monotonic_ns"],observation["received_monotonic_ns"])
            except (KeyError,AdapterError) as error:
                if isinstance(error,AdapterError) and error.reason=="CLOCK_QUALITY" and attempt<2 and self.clock.monotonic_ns()<deadline:
                    continue
                raise AdapterError("CLOCK_QUALITY") from None
            self.clock.sample = candidate.sample
            return observation
        raise AdapterError("CLOCK_QUALITY")

    def _metadata(self):
        previous = self.supervisor
        account = self.account_provider() if self.account_provider else None
        source = self.universe_service.refresh(account)
        if source["data"]["state"] not in {"UNIVERSE_READY","PUBLIC_REFERENCE"}:
            raise AdapterError("METADATA_REFRESH_FAILED")
        if self._stop.is_set():
            return
        if previous:
            previous.stop()
            self._attempts = previous.attempts
        self.universe = source
        self._projection = _universe_projection(source)
        population = {r["symbol"] for r in source["data"]["rows"] if r["status"]=="TRADING" and r["spot_allowed"] is True and r["permitted"] is not False}
        deep = [s for s in self.deep_symbols if s in population]
        self.supervisor = MarketStreamSupervisor(self.store,self.client,source,deep,
            transport_factory=self.transport_factory,background=False,include_candles=True)
        self.supervisor.attempts = self._attempts
        self.supervisor.start()
        self._metadata_due_ns = self.clock.monotonic_ns()+600_000_000_000
        self.pipeline_state = "OBSERVING"

    def start(self):
        if not self.enabled:
            raise AdapterError("MARKET_DATA_DISABLED")
        if self.running:
            raise AdapterError("WS_ALREADY_CONNECTED")
        self._stop.clear()
        try:
            self.clock.timestamp_ms()
        except AdapterError as error:
            if error.reason not in {"CLOCK_STALE","CLOCK_UNSYNCED"}:
                raise
            self.refresh_clock()
        self._metadata()
        self.running = True
        self._started_ns = self.clock.monotonic_ns()
        if self.background:
            self._thread = threading.Thread(target=self._run,name="binance-demo-market-runtime",daemon=True)
            self._thread.start()

    def tick(self):
        if not self.running or self._stop.is_set():
            return
        started = time.monotonic_ns()
        with self._lock:
            now = self.clock.monotonic_ns()
            if now >= self._metadata_due_ns:
                try:
                    self._metadata()
                except AdapterError as error:
                    self.pipeline_state = "METADATA_REFRESH_FAILED"
                    self._metadata_due_ns = now+30_000_000_000
                    self._incident("MARKET_METADATA_REFRESH_FAILED",error.reason)
                    if self.supervisor:
                        self.supervisor.stop()
                    return
            if self._stop.is_set():
                return
            sample = self.clock.sample
            if sample is None or now-sample[1] >= 30_000_000_000:
                try:
                    self.refresh_clock()
                    if self.pipeline_state=="CLOCK_REFRESH_FAILED":
                        self.pipeline_state="OBSERVING"
                except AdapterError as error:
                    self.pipeline_state = "CLOCK_REFRESH_FAILED"
                    self._incident("MARKET_CLOCK_REFRESH_FAILED",error.reason)
                    return
            self.supervisor.tick()
            if self._stop.is_set():
                return
            if now >= self._quote_due_ns:
                if self.clock.wall_ms()-self._projection["received_at_ms"] > 900000:
                    raise AdapterError("METADATA_STALE")
                self.service._poll_quotes(self._projection)
                self._quote_due_ns = self.clock.monotonic_ns()+3_000_000_000
            population = sorted(r["symbol"] for r in self.universe["data"]["rows"]
                if r["status"]=="TRADING" and r["spot_allowed"] is True and r["permitted"] is not False)
            bars = self.store.candle_index(self.clock.wall_ms())
            attempted = 0
            for _ in range(len(population)):
                symbol = population[self._backfill_cursor % len(population)]
                self._backfill_cursor = (self._backfill_cursor+1)%len(population)
                if self._stop.is_set():
                    return
                if _history_ready(bars.get(symbol,{}),self.clock.wall_ms()) or now < self._backfill_due.get(symbol,0):
                    continue
                self._backfill_due[symbol] = now+30_000_000_000
                attempted += 1
                try:
                    self.service.collect_candles(symbol,limit=3)
                except AdapterError as error:
                    if error.reason in {"MARKET_RATE_BUDGET","LOCAL_RATE_BUDGET","RATE_LIMITED","IP_BANNED"}:
                        break
                if attempted >= 2:
                    break
            self._tick_lags.append(Decimal(time.monotonic_ns()-started)/1_000_000)

    def health(self):
        with self._lock:
            if self.universe is None:
                return dict(running=self.running,pipeline_state=self.pipeline_state,production_qualified=False,account_certified=False,order_send_enabled=False)
            now = self.clock.wall_ms()
            rows = self.universe["data"]["rows"]
            active = [r for r in rows if r["status"]=="TRADING" and r["spot_allowed"] is True and r["permitted"] is not False]
            quotes = self.store.quote_index(now)
            bars = self.store.candle_index(now)
            failed_candles = self.store.candle_failures(now)
            qn = sum(not _quote_reasons(quotes.get(r["symbol"],(None,None))[0],now) for r in active)
            cn = sum(r["symbol"] not in failed_candles and _history_ready(bars.get(r["symbol"],{}),now) for r in active)
            states = self.supervisor.health() if self.supervisor else {}
            pipeline_state = self.pipeline_state
            if not self.running:
                pipeline_state = "STOPPED"
            elif pipeline_state == "OBSERVING" and states.get("connected_shards_n",0) != states.get("planned_shards_N",0):
                pipeline_state = "STREAMS_DEGRADED"
            denominator = self.universe["data"]["counts"]["permitted_pairs_N"]
            with localcontext(Context(prec=50)):
                qr = Decimal(qn)/len(active) if active else None
                pr = Decimal(qn)/denominator if denominator and pipeline_state=="OBSERVING" else None
                elapsed = Decimal(self.clock.monotonic_ns()-self._started_ns)/1_000_000_000 if self._started_ns is not None else None
            return dict(running=self.running,pipeline_state=pipeline_state,source_snapshot_id=self.universe["snapshot_id"],
                discovered_pairs_n=self.universe["data"]["counts"]["discovered_pairs_n"],active_public_pairs_N=len(active),
                status_excluded_n=sum(r["status"]!="REMOVED" for r in rows)-len(active),
                removed_tombstones_n=sum(r["status"]=="REMOVED" for r in rows),quote_ready_n=qn,closed_history_ready_n=cn,
                public_quote_ready_ratio=qr,permitted_pairs_N=denominator,permitted_ready_ratio=pr,
                observation_window_s=elapsed,incidents_n=self._incidents_n,supervisor=states,
                processing_tick_recent_n=len(self._tick_lags),processing_tick_window_limit_N=4096,
                processing_tick_p50_ms=sorted(self._tick_lags)[max(0,(len(self._tick_lags)+1)//2-1)] if self._tick_lags else None,
                processing_tick_p95_ms=sorted(self._tick_lags)[max(0,(len(self._tick_lags)*95+99)//100-1)] if self._tick_lags else None,
                processing_tick_max_ms=max(self._tick_lags) if self._tick_lags else None,
                time_basis="OBSERVED_RESPONSE",exchange_quote_age_unknown=True,
                account_certified=False,financial_certified=False,production_qualified=False,order_send_enabled=False)

    def _run(self):
        while not self._stop.wait(0.25):
            try:
                self.tick()
            except Exception:
                self.pipeline_state="RUNTIME_FAILED"
                self._incident("MARKET_RUNTIME_FAILED","READ_UNAVAILABLE")
                break

    def stop(self):
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(20)
            if self._thread.is_alive():
                raise AdapterError("WS_STOP_TIMEOUT")
        with self._lock:
            self.running=False
            if self.supervisor:
                self.supervisor.stop()
