"""Read-only REST observations, causal features and whole-inventory midpoint marks.

No alpha, execution certification, supervisor, key discovery or financial anchors.
"""

from collections import defaultdict, deque
from decimal import Context, Decimal, localcontext
import json
import re
import time

from .adapter import AdapterError, DemoReadClient, _integer, _money, _sha
from .financial import decimal_amount, value_portfolio
from .journal import _digest
from .market_store import MarketSnapshotStore
from .universe import PERMISSIONS

FEATURE_DEFINITION = dict(version="closed-minute-v1", interval_ms=60_000,
                          close_return="last_close/first_close-1",
                          volume="sum_quote_volume", trades="sum_trade_count",
                          gap_policy="reject", open_policy="exclude")


def _symbol(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z0-9_]{2,64}", value):
        raise AdapterError("INVALID_PARAMETERS")
    return value


class DemoMarketClient:
    def __init__(self, client=None):
        self.client = client or DemoReadClient()
        self.clock = self.client.clock

    def _budget(self, weight):
        now = self.clock.monotonic_ns()
        local = sum(w for t, w in self.client._reservations if 0 <= now-t < 60_000_000_000)
        shared = self.client._shared_weight if now < self.client._shared_weight_expires_ns else 0
        if max(local, shared) + 3*weight > 1000:
            raise AdapterError("MARKET_RATE_BUDGET")

    def _read(self, path, params, weight):
        self._budget(weight)
        raw, meta = self.client.request_public(path, params, weight=weight)
        return dict(data=raw, environment="BINANCE_SPOT_DEMO", source_method="GET",
                    request_parameters=params, source_sha256=_sha(raw),
                    clock_uncertainty_ms=self.clock.uncertainty_ms, **meta)

    def read_exchange_info(self):
        return self._read("/api/v3/exchangeInfo", {"permissions": json.dumps(PERMISSIONS, separators=(",", ":")),
                                                  "showPermissionSets": True}, 20)

    def read_book_tickers(self, symbols=None):
        params = {}
        if symbols is not None:
            if not isinstance(symbols, (list, tuple)) or not symbols:
                raise AdapterError("INVALID_PARAMETERS")
            symbols = sorted(set(_symbol(s) for s in symbols))
            if len(symbols) > 128:
                raise AdapterError("INVALID_PARAMETERS")
            params["symbols"] = json.dumps(symbols, separators=(",", ":"))
        return self._read("/api/v3/ticker/bookTicker", params, 4)

    def read_klines(self, symbol, *, start_ms=None, end_ms=None, limit=1000):
        symbol = _symbol(symbol)
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise AdapterError("INVALID_PARAMETERS")
        params = dict(symbol=symbol, interval="1m", timeZone="0", limit=limit)
        for key, value in (("startTime", start_ms), ("endTime", end_ms)):
            if value is not None:
                params[key] = _integer(value, reason="INVALID_PARAMETERS")
        if start_ms is not None and end_ms is not None and start_ms > end_ms:
            raise AdapterError("INVALID_PARAMETERS")
        self._budget(3)  # Also cover a bounded clock probe before the market read.
        try:
            self.clock.timestamp_ms()
        except AdapterError as error:
            if error.reason not in ("CLOCK_UNSYNCED", "CLOCK_STALE"):
                raise
            self.client.sync_clock()
        return self._read("/api/v3/klines", params, 2)

    def close(self):
        self.client.close()


def _quote_index(store, cutoff):
    result = {}
    for snapshot in store.records("QUOTES", cutoff_ms=cutoff):
        for row in snapshot["data"].get("rows", []):
            result[row["symbol"]] = (row, snapshot)
    return result


def _candle_indices(store, cutoff):
    result = defaultdict(dict)
    for snapshot in store.records("CANDLES", cutoff_ms=cutoff):
        symbol = snapshot["data"].get("symbol")
        for row in snapshot["data"].get("rows", []):
            result[symbol][row["open_time_ms"]] = (row, snapshot)
    return result


def _bars_index(store, symbol, cutoff):
    return _candle_indices(store, cutoff).get(symbol, {})


def _quote_reasons(row, cutoff, age=5000):
    if row is None:
        return ["QUOTE_MISSING"]
    if row.get("state") != "QUOTE_READY":
        return row.get("reasons", ["QUOTE_INVALID"])
    quote = row["quote"]
    if not 0 <= cutoff-quote["received_at_ms"] <= age:
        return ["QUOTE_STALE"]
    if quote["response_started_at_ms"] > quote["received_at_ms"]:
        return ["QUOTE_INVALID_INTERVAL"]
    return []


class MarketDataService:
    def __init__(self, store, client, *, enabled=True):
        if type(enabled) is not bool:
            raise AdapterError("INVALID_PARAMETERS")
        self.store, self.client, self.enabled = store, client, enabled
        self._rotation = 0

    def _guard(self):
        if not self.enabled:
            raise AdapterError("MARKET_DATA_DISABLED")

    def _universe(self, value, cutoff):
        if not isinstance(value, dict) or not isinstance(value.get("snapshot_id"), str):
            raise AdapterError("UNIVERSE_SNAPSHOT_INVALID")
        saved = self.store.get(value["snapshot_id"])
        if saved is None or saved["kind"] != "UNIVERSE":
            raise AdapterError("UNIVERSE_SNAPSHOT_INVALID")
        expected = dict(saved)
        expected["data"] = dict(saved["data"])
        # A failed refresh returns a stale view of the immutable original row.
        if value.get("data", {}).get("state") == "UNIVERSE_REFRESH_FAILED":
            for key in ("state", "stale", "last_error"):
                if key in value["data"]:
                    expected["data"][key] = value["data"][key]
        if _digest(value) != _digest(expected):
            raise AdapterError("UNIVERSE_SNAPSHOT_INVALID")
        if saved["available_at_ms"] > cutoff:
            raise AdapterError("UNIVERSE_NOT_AVAILABLE")
        return saved

    def poll_quotes(self, universe, symbols=None):
        self._guard()
        self._universe(universe, self.client.clock.wall_ms())
        members = {r["symbol"]: r for r in universe["data"]["rows"] if r["status"] != "REMOVED"}
        if symbols is not None:
            if not isinstance(symbols, (tuple, list)) or not symbols:
                raise AdapterError("INVALID_PARAMETERS")
            requested = sorted(set(_symbol(s) for s in symbols))
            if any(s not in members for s in requested):
                raise AdapterError("UNMAPPED_SYMBOL")
            batches = [requested[i:i+128] for i in range(0, len(requested), 128)]
        else:
            requested, batches = list(members), [None]
        rows, observations, unexpected = [], [], []
        for batch in batches:
            expected = requested if batch is None else batch
            try:
                observation = self.client.read_book_tickers(batch)
                raw = observation["data"]
                if not isinstance(raw, list):
                    raise AdapterError("QUOTE_SCHEMA_INVALID")
                observations.append(observation)
                mapped, conflicts = {}, set()
                for item in raw:
                    if not isinstance(item, dict) or not isinstance(item.get("symbol"), str):
                        unexpected.append("MALFORMED_SYMBOL")
                        continue
                    symbol = item["symbol"]
                    if symbol not in expected:
                        unexpected.append(symbol)
                        continue
                    if symbol in mapped and mapped[symbol] != item:
                        conflicts.add(symbol)
                    mapped[symbol] = item
                for symbol in expected:
                    item = mapped.get(symbol)
                    row = dict(symbol=symbol, state="QUOTE_INVALID", quote=None, reasons=[])
                    try:
                        if item is None:
                            row["reasons"] = ["QUOTE_MISSING"]
                            rows.append(row)
                            continue
                        if symbol in conflicts:
                            raise AdapterError("QUOTE_CONFLICT")
                        bid, ask = _money(item["bidPrice"], True), _money(item["askPrice"], True)
                        bid_qty, ask_qty = _money(item["bidQty"], True), _money(item["askQty"], True)
                        if bid > ask:
                            raise AdapterError("CROSSED_QUOTE")
                        member = members[symbol]
                        row.update(state="QUOTE_READY", quote=dict(base=member["base"], quote=member["quote"],
                                   bid=bid, ask=ask, bid_quantity=bid_qty, ask_quantity=ask_qty,
                                   venue="BINANCE_SPOT_DEMO", event_time_ms=None,
                                   time_basis="OBSERVED_RESPONSE", convention_id="observed-book-v1",
                                   exchange_age_unknown=True, timestamp_state="TIMESTAMP_UNVERIFIED",
                                   received_at_ms=observation["received_at_ms"],
                                   response_started_at_ms=observation["request_started_at_ms"], source_healthy=True))
                    except (KeyError, TypeError, AdapterError) as error:
                        row["reasons"] = [error.reason if isinstance(error, AdapterError) else "QUOTE_SCHEMA_INVALID"]
                    rows.append(row)
            except AdapterError as error:
                rows.extend(dict(symbol=s, state="QUOTE_INVALID", quote=None, reasons=["QUOTE_READ_FAILED", error.reason]) for s in expected)
        now, mono = self.client.clock.wall_ms(), self.client.clock.monotonic_ns()
        return self.store.append("QUOTES", dict(rows=rows, observations=observations, unexpected_symbols=unexpected),
                                 received_at_ms=now, received_monotonic_ns=mono,
                                 dependencies=[universe["snapshot_id"]])

    def collect_candles(self, symbol, **parameters):
        self._guard()
        _symbol(symbol)
        try:
            return self._collect_candles(symbol, **parameters)
        except AdapterError as error:
            self.store.append("INCIDENT", dict(operation="CANDLE_COLLECTION_FAILED", symbol=symbol, reason=error.reason),
                              received_at_ms=self.client.clock.wall_ms(), received_monotonic_ns=self.client.clock.monotonic_ns())
            raise

    def _collect_candles(self, symbol, **parameters):
        self._guard()
        observation = self.client.read_klines(symbol, **parameters)
        raw = observation["data"]
        if not isinstance(raw, list) or len(raw) > parameters.get("limit", 1000):
            raise AdapterError("CANDLE_SCHEMA_INVALID")
        server = self.client.clock.timestamp_ms()
        with localcontext(Context(prec=50)):
            closed_cutoff = Decimal(server)-self.client.clock.uncertainty_ms
        previous = _bars_index(self.store, symbol, observation["received_at_ms"])
        rows = {}
        try:
            for wire in raw:
                if not isinstance(wire, list) or len(wire) != 12:
                    raise ValueError()
                opened, closed = _integer(wire[0]), _integer(wire[6])
                if opened % 60000 or closed != opened+59999 or opened > server:
                    raise ValueError()
                if parameters.get("start_ms") is not None and opened < parameters["start_ms"]:
                    raise ValueError()
                if parameters.get("end_ms") is not None and opened > parameters["end_ms"]:
                    raise ValueError()
                o, h, low, c = (_money(wire[i], True) for i in (1, 2, 3, 4))
                vol, qvol, tb, tq = (_money(wire[i]) for i in (5, 7, 9, 10))
                trades = _integer(wire[8])
                if not low <= min(o, c) <= max(o, c) <= h or tb > vol or tq > qvol:
                    raise ValueError()
                value = dict(open_time_ms=opened, close_time_ms=closed, open=o, high=h, low=low, close=c,
                             base_volume=vol, quote_volume=qvol, trades=trades,
                             taker_base_volume=tb, taker_quote_volume=tq,
                             closed=closed < closed_cutoff, valid=True, reasons=[], wire_sha256=_sha(wire))
                prior = previous.get(opened)
                conflict = (prior and (not prior[0]["valid"] or
                            (prior[0]["closed"] and prior[0]["wire_sha256"] != value["wire_sha256"])))
                if opened in rows and rows[opened]["wire_sha256"] != value["wire_sha256"]:
                    conflict = True
                if conflict:
                    value.update(valid=False, reasons=["CANDLE_CONFLICT"])
                rows[opened] = value
        except (AdapterError, ValueError, TypeError):
            raise AdapterError("CANDLE_SCHEMA_INVALID") from None
        return self.store.append("CANDLES", dict(symbol=symbol, interval="1m", rows=[rows[k] for k in sorted(rows)],
                                 observation=observation, validated_server_cutoff_ms=str(closed_cutoff)),
                                 received_at_ms=observation["received_at_ms"],
                                 received_monotonic_ns=observation["received_monotonic_ns"])

    def poll_candles(self, universe, *, max_symbols=50):
        self._guard()
        self._universe(universe, self.client.clock.wall_ms())
        if type(max_symbols) is not int or not 1 <= max_symbols <= 1000:
            raise AdapterError("INVALID_PARAMETERS")
        candidates = sorted(r["symbol"] for r in universe["data"]["rows"]
                            if r["permitted"] is True and r["status"] == "TRADING")
        scheduled, pending = [], []
        if candidates:
            for _ in range(min(max_symbols, len(candidates))):
                symbol = candidates[self._rotation % len(candidates)]
                self._rotation = (self._rotation+1) % len(candidates)
                try:
                    scheduled.append(self.collect_candles(symbol))
                except AdapterError as error:
                    pending.append(dict(symbol=symbol, reason=error.reason))
                    if error.reason in {"MARKET_RATE_BUDGET", "LOCAL_RATE_BUDGET", "RATE_LIMITED", "IP_BANNED"}:
                        break
        return dict(snapshots=scheduled, pending=pending, discovered_candidates_N=len(candidates),
                    scheduled_n=len(scheduled), unscheduled_N=len(candidates)-len(scheduled))

    def _feature(self, symbol, cutoff_ms, lookback=3, *, universe=None, quotes=None, bars=None):
        _symbol(symbol)
        _integer(cutoff_ms)
        if type(lookback) is not int or not 2 <= lookback <= 1000:
            raise AdapterError("INVALID_PARAMETERS")
        universe = universe or self.store.as_of("UNIVERSE", cutoff_ms)
        quotes = _quote_index(self.store, cutoff_ms) if quotes is None else quotes
        deps, reasons = set(), []
        member = None
        if not universe or not 0 <= cutoff_ms-universe["received_at_ms"] <= 900000:
            reasons.append("METADATA_STALE")
        if universe:
            deps.add(universe["snapshot_id"])
            member = next((r for r in universe["data"]["rows"] if r["symbol"] == symbol), None)
        if self.store.last_failure("UNIVERSE_REFRESH_FAILED", "UNIVERSE", cutoff_ms):
            reasons.append("METADATA_REFRESH_FAILED")
        if self.store.last_failure("CANDLE_COLLECTION_FAILED", "CANDLES", cutoff_ms, symbol=symbol):
            reasons.append("CANDLE_COLLECTION_FAILED")
        if member is None or member["status"] != "TRADING":
            reasons.append("SYMBOL_NOT_TRADING")
        quote, qsnap = quotes.get(symbol, (None, None))
        reasons.extend(_quote_reasons(quote, cutoff_ms))
        if qsnap:
            deps.add(qsnap["snapshot_id"])
        bars = _bars_index(self.store, symbol, cutoff_ms) if bars is None else bars
        selected = sorted(((r, s) for r, s in bars.values()
                          if r["closed"] and r["close_time_ms"] < cutoff_ms),
                          key=lambda item: item[0]["open_time_ms"])[-lookback:]
        for row, snapshot in selected:
            deps.add(snapshot["snapshot_id"])
            reasons.extend(row["reasons"])
        if len(selected) < lookback:
            reasons.append("CANDLE_HISTORY_SHORT")
        elif any(selected[i][0]["open_time_ms"]-selected[i-1][0]["open_time_ms"] != 60000 for i in range(1, len(selected))):
            reasons.append("CANDLE_GAP")
        if selected and not 0 <= cutoff_ms-selected[-1][0]["close_time_ms"] <= 90000:
            reasons.append("CANDLE_STALE")
        values = None
        if not reasons:
            with localcontext(Context(prec=50)):
                values = dict(close_return=Decimal(selected[-1][0]["close"])/Decimal(selected[0][0]["close"])-1,
                              quote_volume_sum=sum((Decimal(r["quote_volume"]) for r, _ in selected), Decimal(0)),
                              trades_sum=sum(r["trades"] for r, _ in selected))
        return dict(symbol=symbol, cutoff_ms=cutoff_ms, lookback=lookback,
                    state="FEATURE_UNAVAILABLE" if reasons else "FEATURE_READY", values=values,
                    reasons=sorted(set(reasons)), definition_sha256=_digest(dict(FEATURE_DEFINITION, lookback=lookback)),
                    dependency_snapshot_ids=sorted(deps)), sorted(deps)

    def feature(self, symbol, cutoff_ms, lookback=3):
        data, deps = self._feature(symbol, cutoff_ms, lookback)
        now, mono = self.client.clock.wall_ms(), self.client.clock.monotonic_ns()
        data.update(computed_at_ms=now, mode="REPLAY" if cutoff_ms < now else "CURRENT_OBSERVATION")
        return self.store.append("FEATURE", data, received_at_ms=now,
                                 received_monotonic_ns=mono, dependencies=deps)

    def availability(self, universe, cutoff_ms):
        _integer(cutoff_ms)
        self._universe(universe, cutoff_ms)
        quotes = _quote_index(self.store, cutoff_ms)
        bars = _candle_indices(self.store, cutoff_ms)
        rows, qn, fn = [], 0, 0
        for member in universe["data"]["rows"]:
            symbol = member["symbol"]
            quote = quotes.get(symbol, (None, None))[0]
            reasons = _quote_reasons(quote, cutoff_ms)
            if not 0 <= cutoff_ms-universe["received_at_ms"] <= 900000 or universe["available_at_ms"] > cutoff_ms:
                reasons.append("METADATA_STALE")
            if self.store.last_failure("UNIVERSE_REFRESH_FAILED", "UNIVERSE", cutoff_ms):
                reasons.append("METADATA_REFRESH_FAILED")
            if member["status"] != "TRADING":
                reasons.append("SYMBOL_NOT_TRADING")
            ready = not reasons
            feature, _ = self._feature(symbol, cutoff_ms, universe=universe, quotes=quotes, bars=bars.get(symbol, {}))
            feature_ready = feature["state"] == "FEATURE_READY" and ready
            qn += ready and member["permitted"] is True
            fn += feature_ready and member["permitted"] is True
            rows.append(dict(symbol=symbol, permitted=member["permitted"], quote_ready=ready,
                             feature_ready=feature_ready, reasons=sorted(set(reasons)), feature_reasons=feature["reasons"]))
        denominator = universe["data"]["counts"]["permitted_pairs_N"]
        with localcontext(Context(prec=50)):
            qr = Decimal(qn)/denominator if denominator else None
            fr = Decimal(fn)/denominator if denominator else None
        return dict(rows=rows, discovered_pairs_n=universe["data"]["counts"]["discovered_pairs_n"],
                    permitted_pairs_N=denominator, quote_ready_n=qn, feature_ready_n=fn,
                    quote_ready_ratio=qr, feature_ready_ratio=fr, cutoff_ms=cutoff_ms)


class QuoteConversionGraph:
    def __init__(self, store):
        self.store = store

    def value(self, balances, cutoff_ms, *, time_basis="OBSERVED_RESPONSE", max_age_ms=5000):
        _integer(cutoff_ms)
        if time_basis not in {"OBSERVED_RESPONSE", "EXCHANGE_EVENT"} or type(max_age_ms) is not int or max_age_ms < 0:
            raise AdapterError("INVALID_PARAMETERS")
        universe = self.store.as_of("UNIVERSE", cutoff_ms)
        index = _quote_index(self.store, cutoff_ms)
        adjacency, quotes, leg_ids = defaultdict(list), {}, {}
        if universe and 0 <= cutoff_ms-universe["received_at_ms"] <= 900000 and not self.store.last_failure("UNIVERSE_REFRESH_FAILED", "UNIVERSE", cutoff_ms):
            for row in universe["data"]["rows"]:
                symbol = row["symbol"]
                observed, snapshot = index.get(symbol, (None, None))
                if row["status"] != "TRADING" or _quote_reasons(observed, cutoff_ms, max_age_ms):
                    continue
                quote = observed["quote"]
                if time_basis == "EXCHANGE_EVENT" and quote["event_time_ms"] is None:
                    continue
                if quote["base"] != row["base"] or quote["quote"] != row["quote"]:
                    continue
                quotes[symbol] = quote
                leg_ids[symbol] = snapshot["snapshot_id"]
                adjacency[row["base"]].append((symbol, False, row["quote"]))
                adjacency[row["quote"]].append((symbol, True, row["base"]))
        routes, held = {}, []
        with localcontext(Context(prec=50)):
            for asset, balance in balances.items():
                quantity = decimal_amount(balance["free"])+decimal_amount(balance["locked"])
                if quantity < 0:
                    raise AdapterError("INVALID_BALANCES")
                if quantity == 0:
                    continue
                held.append(asset)
                if asset == "USDT":
                    routes[asset] = []
                    continue
                queue = deque([(asset, [], frozenset((asset,)))])
                while queue:
                    current, path, seen = queue.popleft()
                    if current == "USDT":
                        routes[asset] = path
                        break
                    if len(path) >= 3:
                        continue
                    for symbol, inverse, target in sorted(adjacency[current]):
                        if target not in seen:
                            queue.append((target, path+[(symbol, inverse)], seen | {target}))
            result = value_portfolio(balances, quotes, routes, boundary_ms=cutoff_ms,
                                     time_basis=time_basis, max_age_ms=max_age_ms)
            valued = sum(asset in result["marks"] for asset in held)
            ratio = Decimal(valued)/len(held) if held else None
        deps = {leg_ids[symbol] for path in routes.values() for symbol, _ in path}
        if universe:
            deps.add(universe["snapshot_id"])
        data = dict(result, balances=balances, cutoff_ms=cutoff_ms, routes=routes,
                    route_policy_id="fresh-shortest-lexical-v1", leg_snapshot_ids=leg_ids,
                    universe_snapshot_id=universe["snapshot_id"] if universe else None,
                    time_basis=time_basis, convention_id="observed-book-v1" if time_basis == "OBSERVED_RESPONSE" else None,
                    exchange_age_unknown=time_basis == "OBSERVED_RESPONSE", max_age_ms=max_age_ms,
                    held_assets_N=len(held), valued_assets_n=valued, valued_assets_ratio=ratio,
                    liquidation_certified=False, account_anchor_certified=False, exchange_reconciled=False,
                    input_balances_certified=False)
        return self.store.append("VALUATION", data, received_at_ms=time.time_ns()//1_000_000,
                                 received_monotonic_ns=time.monotonic_ns(), dependencies=sorted(deps))
