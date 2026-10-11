"""P2-B1 stream/L2 acceptance, written before the implementation."""

import importlib
import json
import time
import threading
import unittest
from unittest.mock import patch
from decimal import Decimal

from _p2_support import Fixture, NOW, pair
from binance_demo_bot.adapter import _sha, AdapterError


def diff(U=101, u=102, **changes):
    value = dict(e="depthUpdate", E=NOW-10, s="AAAUSDT", U=U, u=u, b=[["100","2"]], a=[])
    value.update(changes)
    return value


def snapshot(last=100, **changes):
    raw = dict(lastUpdateId=last, bids=[["99","5"]], asks=[["101","6"]])
    raw.update(changes)
    return dict(environment="BINANCE_SPOT_DEMO", source_endpoint="/api/v3/depth", source_method="GET",
                request_parameters={"symbol":"AAAUSDT", "limit":100}, data=raw, source_sha256=_sha(raw),
                received_at_ms=NOW, request_started_at_ms=NOW-1, received_monotonic_ns=1_000_000_000)


def tick(symbol="AAAUSDT", u=10, bid="99", ask="101"):
    return dict(stream=symbol.lower()+"@bookTicker", data=dict(s=symbol, u=u, b=bid, B="2", a=ask, A="3"))


class FakeStream:
    def __init__(self, frames=None):
        self.frames, self.sent, self.connected, self.closed = list(frames or []), [], False, False
    def connect(self):
        self.connected, self.closed = True, False
    def send(self, value):
        self.sent.append(value)
        self.frames.append({"result":None, "id":value["id"]})
    def recv(self, timeout_s=1):
        if self.closed:
            raise AdapterError("WS_UNAVAILABLE")
        if not self.frames:
            time.sleep(0.002)
            raise TimeoutError()
        value = self.frames.pop(0)
        if isinstance(value, BaseException):
            raise value
        return json.dumps(value)
    def ping(self, timeout_s=5):
        return True
    def close(self):
        self.closed = True


class StreamTests(Fixture, unittest.TestCase):
    def setUp(self):
        self.setup_market()
        self.api = importlib.import_module("binance_demo_bot.market_stream")

    def book(self, **options):
        return self.api.IncrementalBook("AAAUSDT", self.clock, **options)

    def push(self, book, payload=None):
        return book.on_diff(diff() if payload is None else payload,
                            received_at_ms=self.wall, received_monotonic_ns=self.mono)

    def ready(self, book=None):
        book = book or self.book()
        self.push(book)
        book.install_snapshot(snapshot())
        self.assertEqual(book.view()["state"], "READY")
        return book

    def reader(self, universe=None, topics=None, frames=None, background=False):
        universe = universe or self.refresh()
        transport = FakeStream(frames)
        reader = self.api.MarketStreamReader(self.store, self.client, universe,
            topics or ["aaausdt@bookTicker", "aaausdt@depth@100ms"], transport=transport, background=background)
        self.addCleanup(reader.stop)
        return reader, transport

    def test_P2B102_full_non_usdt_deterministic_plan_and_public_null_denominator(self):
        universe = self.refresh([pair(), pair("BBB","BTC"), pair("CCC", status="BREAK")], account=None)
        plan = self.api.stream_plan(universe, deep_symbols=["BBBBTC"])
        self.assertIsNone(plan["permitted_pairs_N"])
        topics = [t for shard in plan["shards"] for t in shard["topics"]]
        self.assertEqual(topics, ["aaausdt@bookTicker","bbbbtc@bookTicker","bbbbtc@depth@100ms"])
        self.assertEqual(plan, self.api.stream_plan(universe, deep_symbols=["BBBBTC"]))
        self.assertEqual(plan["discovered_pairs_n"], 3)

    def test_P2B102_shards_limit_512_and_denied_rows_not_erased(self):
        universe = self.refresh([pair(str(i)) for i in range(1100)])
        plan = self.api.stream_plan(universe)
        self.assertEqual(len(plan["shards"]), 3)
        self.assertEqual(sum(len(s["topics"]) for s in plan["shards"]), 1100)
        self.assertTrue(all(len(s["topics"]) <= 512 for s in plan["shards"]))
        universe = self.refresh([pair(permissionSets=[["TRD_GRP_002"]])])
        self.assertEqual(self.api.stream_plan(universe)["shards"], [])

    def test_P2B102_deep_outside_population_or_too_many_rejected(self):
        universe = self.refresh()
        self.assert_reason("UNMAPPED_SYMBOL", self.api.stream_plan, universe, ["OTHERUSDT"])
        universe = self.refresh([pair(str(i)) for i in range(33)])
        self.assert_reason("INVALID_PARAMETERS", self.api.stream_plan, universe, [str(i)+"USDT" for i in range(33)])

    def test_P2B102_mutated_future_source_rejected_before_network(self):
        universe = self.refresh()
        universe["data"]["rows"][0]["base"] = "OTHER"
        self.assert_reason("UNIVERSE_SNAPSHOT_INVALID", self.api.MarketStreamReader,
                           self.store, self.client, universe, ["aaausdt@bookTicker"])
        self.advance(1)
        universe = self.refresh()
        self.advance(-1)
        self.assert_reason("UNIVERSE_NOT_AVAILABLE", self.api.MarketStreamReader,
                           self.store, self.client, universe, ["aaausdt@bookTicker"])

    def test_P2B103_snapshot_alone_never_certifies_continuity(self):
        book = self.book()
        book.install_snapshot(snapshot())
        self.assertEqual(book.view()["state"], "BUFFERING")

    def test_P2B103_buffer_bridge_and_exact_replacement(self):
        book = self.ready()
        self.push(book, diff(103,104,b=[["100","7"]]))
        view = book.view()
        self.assertEqual(view["last_update_id"], 104)
        self.assertEqual(Decimal(view["bids"][0][1]), Decimal("7"))
        self.assertFalse(view["full_book_certified"])

    def test_P2B103_snapshot_discards_stale_buffer_then_waits_for_new_diff(self):
        book = self.book()
        self.push(book, diff(99,100))
        book.install_snapshot(snapshot())
        self.assertEqual(book.view()["state"], "BUFFERING")
        self.push(book)
        self.assertEqual(book.view()["state"], "READY")

    def test_P2B103_overlap_bridges_next_expected_id(self):
        book = self.ready()
        self.push(book, diff(102,104))
        self.assertEqual(book.view()["last_update_id"], 104)

    def test_P2B103_gap_invalid_until_new_snapshot_and_bridge(self):
        book = self.ready()
        self.push(book, diff(105,106))
        self.assertEqual(book.view()["state"], "INVALID")
        self.assertIn("L2_GAP", book.view()["reasons"])
        self.push(book, diff(107,108))
        self.assertNotEqual(book.view()["state"], "READY")
        book.install_snapshot(snapshot(106))
        self.assertEqual(book.view()["state"], "READY")

    def test_P2B103_zero_quantity_deletes_level(self):
        book = self.ready()
        self.push(book, diff(103,104,b=[["100","0"]]))
        self.assertEqual(Decimal(book.view()["bids"][0][0]), Decimal("99"))

    def test_P2B103_duplicate_not_refresh_clock_conflict_invalidates(self):
        book = self.ready()
        self.advance(100)
        self.push(book)
        self.assertEqual(book.view()["received_at_ms"], NOW)
        self.push(book, diff(b=[["100","8"]]))
        self.assertIn("L2_CONFLICT", book.view()["reasons"])

    def test_P2B103_buffer_overflow_and_partial_depth_rejected(self):
        book = self.book(max_buffer=1)
        self.push(book)
        self.push(book, diff(103,104))
        self.assertIn("L2_BUFFER_OVERFLOW", book.view()["reasons"])
        book = self.book()
        self.push(book, {"lastUpdateId":100,"bids":[],"asks":[]})
        self.assertIn("L2_SCHEMA_INVALID", book.view()["reasons"])

    def test_P2B103_crossed_empty_or_duplicate_levels_quarantined(self):
        for payload in (diff(b=[["102","1"]]), diff(b=[["99","0"]],a=[["101","0"]]), diff(b=[["100","2"],["100","3"]])):
            book = self.book()
            self.push(book, payload)
            book.install_snapshot(snapshot())
            self.assertNotEqual(book.view()["state"], "READY")

    def test_P2B103_bad_snapshot_hash_symbol_venue_rejected(self):
        for changes in ({"source_sha256":"bad"}, {"environment":"BINANCE_SPOT"},
                        {"request_parameters":{"symbol":"OTHERUSDT","limit":100}}):
            book = self.book()
            observation = snapshot()
            observation.update(changes)
            book.install_snapshot(observation)
            self.assertEqual(book.view()["state"], "INVALID")

    def test_P2B104_stale_future_clock_jump_and_reversed_event_time(self):
        for event_time in (NOW-1001, NOW+1):
            book = self.book()
            self.push(book, diff(E=event_time))
            self.assertEqual(book.view()["state"], "INVALID")
        book = self.ready()
        self.push(book, diff(103,104,E=NOW-11))
        self.assertIn("L2_EVENT_REGRESSION", book.view()["reasons"])
        book = self.ready()
        self.wall += 1000
        self.assertNotEqual(book.view()["state"], "READY")

    def test_P2B104_readiness_expires_without_new_update(self):
        book = self.ready()
        self.advance(1001)
        self.assertNotEqual(book.view()["state"], "READY")

    def test_P2B103_view_copies_cannot_mutate_live_book(self):
        book = self.ready()
        value = book.view()
        value["bids"].clear()
        self.assertTrue(book.view()["bids"])

    def test_P2B105_depth_kind_durable(self):
        result = self.store.append("DEPTH", {"symbol":"AAAUSDT"}, received_at_ms=NOW, received_monotonic_ns=self.mono)
        self.assertEqual(self.store.as_of("DEPTH", NOW)["snapshot_id"], result["snapshot_id"])

    def test_P2B105_ack_quote_null_event_time_and_preack_receipt(self):
        reader, transport = self.reader(frames=[tick()])
        reader.start()
        self.assertEqual(transport.sent[0]["method"], "SUBSCRIBE")
        row = self.store.as_of("QUOTES", NOW)["data"]["rows"][0]
        self.assertEqual(row["state"], "QUOTE_READY")
        self.assertIsNone(row["quote"]["event_time_ms"])
        self.assertEqual(row["quote"]["received_at_ms"], NOW)
        self.assertEqual(reader.health()["last_processing_lag_ms"], Decimal("0"))

    def test_P2B104_duplicate_quote_does_not_extend_freshness(self):
        reader, transport = self.reader()
        reader.start()
        transport.frames.append(tick())
        reader.read_once()
        self.advance(100)
        transport.frames.append(tick())
        reader.read_once()
        self.assertEqual(self.store.as_of("QUOTES", self.wall)["data"]["rows"][0]["quote"]["received_at_ms"], NOW)

    def test_P2B104_one_symbol_update_not_refresh_other_symbol(self):
        universe = self.refresh([pair(), pair("BBB")])
        reader, transport = self.reader(universe, ["aaausdt@bookTicker","bbbusdt@bookTicker"])
        reader.start()
        transport.frames.append(tick())
        reader.read_once()
        self.advance(5001)
        transport.frames.append(tick("BBBUSDT"))
        reader.read_once()
        coverage = self.service.availability(universe, self.wall)
        self.assertEqual(coverage["quote_ready_n"], 1)

    def test_P2B105_unmapped_stream_or_symbol_closes_reader(self):
        for value in (tick("OTHERUSDT"), {"stream":"aaausdt@bookTicker","data":tick("OTHERUSDT")["data"]}):
            reader, transport = self.reader()
            reader.start()
            transport.frames.append(value)
            reader.read_once()
            self.assertFalse(reader.health()["connected"])

    def test_P2B105_disconnect_and_stop_invalidate_previous_good_quote(self):
        reader, transport = self.reader()
        reader.start()
        transport.frames.append(tick())
        reader.read_once()
        self.assertEqual(reader.health()["fresh_quotes_n"], 1)
        reader.stop()
        universe = self.universe_service.current(self.wall)
        self.assertEqual(self.service.availability(universe, self.wall)["quote_ready_n"], 0)
        self.assertTrue(transport.closed)

    def test_P2B105_l2_snapshot_and_diff_preserve_dependency_and_receipts(self):
        reader, transport = self.reader()
        reader.start()
        transport.frames.append({"stream":"aaausdt@depth@100ms","data":diff()})
        reader.read_once()
        reader.install_depth("AAAUSDT", snapshot())
        result = self.store.as_of("DEPTH", NOW)
        self.assertTrue(result["dependencies"])
        self.assertEqual(reader.health()["ready_depth_n"], 1)
        diagnostics = reader.health()["depth_diagnostics"]["AAAUSDT"]
        self.assertEqual(diagnostics["event_time_ms"], NOW-10)
        self.assertEqual(diagnostics["received_at_ms"], NOW)

    def test_P2B105_real_owned_reader_thread_stops(self):
        reader, transport = self.reader(background=True)
        reader.start()
        transport.frames.append(tick())
        deadline = time.monotonic()+2
        while reader.health()["fresh_quotes_n"] == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(reader.health()["fresh_quotes_n"], 1)
        reader.stop()
        self.assertFalse(reader.health()["thread_alive"])

    def test_P2B106_disabled_supervisor_zero_network(self):
        universe = self.refresh()
        transports = []
        def factory():
            result = FakeStream()
            transports.append(result)
            return result
        supervisor = self.api.MarketStreamSupervisor(self.store,self.client,universe, transport_factory=factory, enabled=False)
        self.assert_reason("MARKET_DATA_DISABLED", supervisor.start)
        self.assertEqual(transports, [])

    def test_P2B106_supervisor_starts_shards_and_stops_only_owned(self):
        universe = self.refresh()
        supervisor = self.api.MarketStreamSupervisor(self.store,self.client,universe, transport_factory=FakeStream)
        self.addCleanup(supervisor.stop)
        supervisor.start()
        self.assertEqual(supervisor.health()["connected_shards_n"], 1)
        self.assertEqual(supervisor.health()["permitted_pairs_N"], 1)
        supervisor.stop()
        self.assertEqual(supervisor.health()["connected_shards_n"], 0)

    def test_P2B101_ws_demo_only_tls_proxy_logger_and_method_guards(self):
        for url in ("wss://stream.binance.com/stream", "wss://demo-stream.binance.com/ws", "wss://demo-stream.binance.com/stream?x=1"):
            self.assert_reason("FORBIDDEN_ENDPOINT", self.api.DemoMarketWsTransport, url)
        with patch.object(self.api, "ws_connect") as factory:
            connection = factory.return_value
            connection.request.path = "/stream"
            connection.request.headers = {"Host":"demo-stream.binance.com"}
            transport = self.api.DemoMarketWsTransport()
            transport.connect()
            kwargs = factory.call_args.kwargs
            self.assertIsNone(kwargs["proxy"])
            self.assertTrue(kwargs["logger"].disabled)
            self.assertEqual(kwargs["max_queue"], 16)
            self.assertEqual(kwargs["ssl"].verify_mode, 2)
            self.assert_reason("FORBIDDEN_METHOD", transport.send, {"id":"1","method":"order.place"})
            transport.close()

    def test_P2B101_invalid_topic_or_oversized_subscription_rejected(self):
        transport = self.api.DemoMarketWsTransport()
        for topics in (["aaausdt@trade"], ["AAAUSDT@bookTicker"], ["aaausdt@bookTicker"]*513):
            self.assert_reason("INVALID_PARAMETERS", transport.send, {"id":"1", "method":"SUBSCRIBE", "params":topics})

    def test_P2B104_quote_conflict_cannot_reset_watermark_and_recover_in_session(self):
        reader, transport = self.reader()
        reader.start()
        transport.frames.append(tick(u=10))
        reader.read_once()
        transport.frames.append(tick(u=9))
        reader.read_once()
        transport.frames.append(tick(u=11))
        reader.read_once()
        self.assertEqual(reader.health()["fresh_quotes_n"], 0)
        self.assertTrue(reader.health()["needs_reconnect"])

    def test_P2B104_preack_processing_delay_closes_readiness(self):
        reader, transport = self.reader(frames=[tick()])
        receive = transport.recv
        def delayed(timeout_s=1):
            value = receive(timeout_s)
            if '"result"' in value:
                self.advance(251)
            return value
        transport.recv = delayed
        self.assert_reason("WS_PROCESSING_LAG", reader.start)
        self.assertEqual(reader.health()["fresh_quotes_n"], 0)

    def test_P2B103_level_overflow_quarantined(self):
        book = self.book(max_levels=2)
        book.install_snapshot(snapshot())
        self.push(book)
        self.assertIn("L2_LEVEL_OVERFLOW", book.view()["reasons"])

    def test_P2B104_uncertainty_counts_toward_event_age(self):
        self.clock.observe(NOW,NOW-200,NOW,self.mono-200_000_000,self.mono)
        book = self.book()
        self.push(book, diff(E=NOW-900))
        self.assertNotEqual(book.view()["state"], "READY")
        self.assertIn("L2_STALE_EVENT", book.view()["reasons"])

    def test_P2B103_snapshot_future_monotonic_receipt_quarantined(self):
        book = self.book()
        self.push(book)
        value = snapshot()
        value["received_monotonic_ns"] = self.mono+1
        book.install_snapshot(value)
        self.assertEqual(book.view()["state"], "INVALID")

    def test_P2B105_bad_ack_id_closes_without_blind_subscribe_retry(self):
        reader, transport = self.reader(frames=[{"id":"wrong","result":None}])
        self.assert_reason("WS_PROTOCOL_ERROR", reader.start)
        self.assertEqual(len(transport.sent), 1)
        self.assertFalse(reader.health()["connected"])

    def test_P2B106_runtime_disconnect_has_backoff(self):
        supervisor = self.api.MarketStreamSupervisor(self.store,self.client,self.refresh(),transport_factory=FakeStream)
        self.addCleanup(supervisor.stop)
        supervisor.start()
        reader = next(iter(supervisor.readers.values()))
        reader._fail("WS_UNAVAILABLE")
        supervisor.tick()
        self.assertEqual(supervisor.health()["connection_attempts_n"], 1)
        self.advance(1000)
        supervisor.tick()
        self.assertEqual(supervisor.health()["connection_attempts_n"], 2)

    def test_P2B106_rolling_connection_quota_prevents_dispatch(self):
        transports = []
        def factory():
            value = FakeStream()
            transports.append(value)
            return value
        supervisor = self.api.MarketStreamSupervisor(self.store,self.client,self.refresh(),transport_factory=factory)
        self.addCleanup(supervisor.stop)
        supervisor.attempts.extend([self.mono]*20)
        supervisor.start()
        self.assertEqual(transports, [])

    def test_P2B105_restart_does_not_inherit_quote_or_depth_readiness(self):
        reader, transport = self.reader()
        reader.start()
        transport.frames.extend([tick(), {"stream":"aaausdt@depth@100ms","data":diff()}])
        reader.read_once()
        reader.read_once()
        reader.install_depth("AAAUSDT",snapshot())
        self.assertEqual(reader.health()["ready_depth_n"], 1)
        reader.stop()
        reader.start()
        self.assertEqual(reader.health()["fresh_quotes_n"], 0)
        self.assertEqual(reader.health()["ready_depth_n"], 0)

    def test_P2B104_ping_health_checked_under_continuous_messages(self):
        reader, transport = self.reader()
        reader.start()
        with patch.object(transport,"ping",return_value=True) as ping:
            self.advance(20000)
            transport.frames.append(tick())
            reader.read_once()
            ping.assert_called_once()

    def test_P2B103_concurrent_view_never_sees_half_applied_side_replacement(self):
        book = self.book()
        self.push(book, diff(b=[]))
        book.install_snapshot(snapshot())
        halfway, release, viewer_started = threading.Event(),threading.Event(),threading.Event()
        class GateDict(dict):
            def pop(inner, key, default=None):
                value = super().pop(key,default)
                halfway.set()
                release.wait(2)
                return value
        book.bids = GateDict(book.bids)
        results = []
        producer = threading.Thread(target=lambda:self.push(book,diff(103,104,b=[["99","0"],["100","2"]])))
        def read():
            viewer_started.set()
            results.append(book.view())
        viewer = threading.Thread(target=read)
        try:
            producer.start()
            self.assertTrue(halfway.wait(1))
            viewer.start()
            self.assertTrue(viewer_started.wait(1))
            time.sleep(0.03)
        finally:
            release.set()
            producer.join(2)
            if viewer.ident is not None:
                viewer.join(2)
        self.assertEqual(results[0]["bids"], [["100","2"]])

    def test_P2B102_utf8_symbol_and_exact_source_spelling(self):
        universe = self.refresh([pair("币安人生"),pair("iToken")],account=None)
        reader, transport = self.reader(universe,["币安人生usdt@bookTicker","itokenusdt@bookTicker"])
        reader.start()
        for symbol in ("币安人生USDT","iTokenUSDT"):
            transport.frames.append(tick(symbol))
            reader.read_once()
        self.assertEqual(reader.health()["fresh_quotes_n"], 2)

    def test_P2B104_clock_jump_closes_health_before_next_frame(self):
        reader, transport = self.reader()
        reader.start()
        transport.frames.append(tick())
        reader.read_once()
        self.wall += 1000
        self.assertEqual(reader.health()["fresh_quotes_n"], 0)

    def test_P2B106_stop_barrier_prevents_next_rest_snapshot(self):
        universe = self.refresh([pair(),pair("BBB"),pair("CCC")])
        supervisor = self.api.MarketStreamSupervisor(self.store,self.client,universe,
            ["AAAUSDT","BBBUSDT","CCCUSDT"],transport_factory=FakeStream)
        self.addCleanup(supervisor.stop)
        supervisor.start()
        reader = next(iter(supervisor.readers.values()))
        for symbol,book in reader.books.items():
            self.push(book,diff(s=symbol))
        calls = []
        def stopping(symbol):
            calls.append(symbol)
            supervisor._stop.set()  # Deterministic barrier from pending stop().
            return snapshot()
        with patch.object(self.client,"read_depth",side_effect=stopping):
            supervisor.tick()
        self.assertEqual(len(calls),1)


class PublicDepthTests(Fixture, unittest.TestCase):
    def setUp(self):
        self.setup_market()

    def test_P2B101_depth_request_is_public_exact_and_weighted(self):
        self.replies.append(snapshot()["data"])
        result = self.client.read_depth("AAAUSDT", limit=100)
        self.assertEqual(result["source_endpoint"], "/api/v3/depth")
        self.assertEqual(self.calls[0][3], {})
        self.assertEqual(sum(w for _, w in self.read_client._reservations), 5)

    def test_P2B101_depth_limits_and_reserve_before_network(self):
        self.assert_reason("INVALID_PARAMETERS", self.client.read_depth, "AAAUSDT", limit=1001)
        self.read_client._reservations.append((self.mono, 900))
        self.assert_reason("MARKET_RATE_BUDGET", self.client.read_depth, "AAAUSDT", limit=1000)
        self.assertEqual(self.calls, [])

    def test_P2B101_utf8_symbol_preserved_in_http_parameters(self):
        self.replies.append(snapshot()["data"])
        result = self.client.read_depth("币安人生USDT")
        self.assertEqual(result["request_parameters"]["symbol"], "币安人生USDT")
