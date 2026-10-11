"""P2-B2 contracts, written and run RED before implementation."""

import importlib
import json
import unittest
import threading
from decimal import Decimal,localcontext
from unittest.mock import patch

from _p2_support import Fixture, NOW, pair, metadata, book, candle
from test_market_stream import FakeStream
from binance_demo_bot.adapter import AdapterError, _sha
from binance_demo_bot.journal import _digest


def kline(closed=True, **changes):
    value = dict(e="kline",E=NOW,s="AAAUSDT",k=dict(t=NOW-60000,T=NOW-1,s="AAAUSDT",i="1m",
        f=1,L=10,o="100",h="120",l="90",c="102",v="10",n=10,x=closed,q="1000",V="5",Q="500",B="123456"))
    value["k"].update(changes)
    return value


class CompactAndStoreTests(Fixture,unittest.TestCase):
    def setUp(self): self.setup_market()

    def test_P2B201_compact_source_roundtrip_unknown_flags_and_order(self):
        rows = [pair("BBB", newFlag={"value":7}),pair(),pair("BBB",newFlag={"value":7})]
        result = self.refresh(rows)
        self.assertEqual(result["data"]["format_version"],"universe-v2")
        original = self.universe_api.universe_source(result)
        self.assertEqual(original,metadata(rows))
        self.assertEqual(_sha(original),result["data"]["observation"]["source_sha256"])
        self.assertNotIn("data",result["data"]["observation"])
        self.assertNotIn("raw",result["data"]["rows"][0])

    def test_P2B201_legacy_row_source_and_removed_tombstones(self):
        self.assertEqual(self.universe_api.symbol_source({"raw":pair()}),pair())
        self.refresh([pair(),pair("BBB")])
        self.advance(1)
        result = self.refresh([pair()])
        self.assertEqual(self.universe_api.universe_source(result),metadata([pair()]))
        removed=next(r for r in result["data"]["rows"] if r["symbol"]=="BBBUSDT")
        self.assertEqual(self.universe_api.symbol_source(removed)["status"],"TRADING")

    def test_P2B201_full_envelope_validation_tamper_future_and_kind(self):
        result = self.refresh()
        self.store.validate_snapshot(result,"UNIVERSE",NOW)
        self.assert_reason("UNIVERSE_NOT_AVAILABLE",self.store.validate_snapshot,result,"UNIVERSE",NOW-1)
        self.assert_reason("UNIVERSE_SNAPSHOT_INVALID",self.store.validate_snapshot,result,"QUOTES",NOW)
        result["data"]["rows"][0]["base"]="OTHER"
        self.assert_reason("UNIVERSE_SNAPSHOT_INVALID",self.store.validate_snapshot,result,"UNIVERSE",NOW)

    def test_P2B202_batch_rollback_and_atomic_receipt_dependencies(self):
        with self.assertRaises(RuntimeError):
            with self.store.batch():
                self.store.append("QUOTES",{},received_at_ms=NOW,received_monotonic_ns=self.mono)
                raise RuntimeError("synthetic rollback")
        self.assertEqual(self.store.records("QUOTES"),[])
        with self.store.batch():
            first = self.store.append("QUOTES",{},received_at_ms=NOW,received_monotonic_ns=self.mono)
            second = self.store.append("FEATURE",{},received_at_ms=NOW+1,received_monotonic_ns=self.mono+1,
                                      dependencies=[first["snapshot_id"]])
        self.assertEqual(len(self.store.records()),2)
        self.assertEqual(second["dependencies"],[first["snapshot_id"]])

    def test_P2B202_hashes_match_original_canonical_convention(self):
        result = self.store.append("FEATURE",{"amount":Decimal("1.0000")},received_at_ms=NOW,received_monotonic_ns=self.mono)
        self.assertEqual(result["source_sha256"],_digest({"amount":Decimal("1.0000")}))
        basis = {k:v for k,v in result.items() if k not in {"event_id","snapshot_id","recorded_at_ms"}}
        self.assertEqual(result["snapshot_id"],_digest(basis))

    def test_P2B202_standalone_serialization_does_not_block_receipt_lookup(self):
        universe=self.refresh()
        api=importlib.import_module("binance_demo_bot.market_store")
        original=api._json
        complete=threading.Event()
        workers=[]
        def serialize(value):
            if value=={"probe":True}:
                def lookup():
                    self.store.validate_reference(universe["snapshot_id"],"UNIVERSE",NOW)
                    complete.set()
                worker=threading.Thread(target=lookup)
                workers.append(worker);worker.start()
                self.assertTrue(complete.wait(1),"writer lock held during standalone serialization")
            return original(value)
        try:
            with patch.object(api,"_json",serialize):
                self.store.append("FEATURE",{"probe":True},received_at_ms=NOW,received_monotonic_ns=self.mono)
        finally:
            for worker in workers:worker.join(2)

    def test_P2B202_fifo_writer_cannot_overtake_waiting_owner(self):
        api=importlib.import_module("binance_demo_bot.market_store")
        lock=api._FairRLock()
        entered=threading.Event()
        order=[]
        lock.acquire()
        lock.acquire();lock.release()
        def worker():
            entered.set()
            with lock:order.append("waiting")
        thread=threading.Thread(target=worker);thread.start()
        self.assertTrue(entered.wait(1))
        with lock._condition:
            self.assertTrue(lock._condition.wait_for(lambda:lock._issued-lock._serving>=2,timeout=1))
        lock.release()
        with lock:order.append("previous")
        thread.join(2)
        self.assertEqual(order,["waiting","previous"])
        self.assertFalse(thread.is_alive())
        with self.assertRaises(RuntimeError):lock.release()

    def test_P2B202_committed_quote_index_preserves_past_and_invalid_latest(self):
        universe=self.refresh()
        self.quotes(universe=universe)
        first=self.store.quote_index(NOW)
        self.advance(1)
        self.quotes([book(bid="0",ask="0")],universe)
        self.assertEqual(self.store.quote_index(self.wall)["AAAUSDT"][0]["state"],"QUOTE_INVALID")
        self.assertEqual(first["AAAUSDT"][0]["state"],"QUOTE_READY")
        self.assertEqual(self.store.quote_index(NOW)["AAAUSDT"][0]["state"],"QUOTE_READY")

    def test_P2B202_candle_failure_bulk_current_and_past_prefix(self):
        self.bars()
        self.advance(1)
        self.store.append("INCIDENT",dict(operation="CANDLE_COLLECTION_FAILED",symbol="AAAUSDT",reason="READ_UNAVAILABLE"),
            received_at_ms=self.wall,received_monotonic_ns=self.mono)
        self.assertEqual(self.store.candle_failures(self.wall),{"AAAUSDT"})
        self.assertEqual(self.store.candle_failures(NOW),set())
        self.advance(1);self.bars()
        self.assertEqual(self.store.candle_failures(self.wall),set())
        self.assertEqual(self.store.candle_failures(NOW+1),{"AAAUSDT"})

    def test_P2B202_current_index_copy_does_not_hold_writer_lock(self):
        universe=self.refresh()
        self.quotes(universe=universe);self.bars()
        api=importlib.import_module("binance_demo_bot.market_store")
        original=api.deepcopy
        workers=[]
        def copying(value):
            done=threading.Event()
            def lookup():
                self.store.validate_reference(universe["snapshot_id"],"UNIVERSE",NOW)
                done.set()
            worker=threading.Thread(target=lookup);workers.append(worker);worker.start()
            self.assertTrue(done.wait(1),"index copy holds writer lock")
            return original(value)
        try:
            with patch.object(api,"deepcopy",copying):
                quotes=self.store.quote_index(NOW)
                bars=self.store.candle_index(NOW)
            quotes["AAAUSDT"][0]["state"]="CHANGED"
            bars["AAAUSDT"].clear()
            self.assertEqual(self.store.quote_index(NOW)["AAAUSDT"][0]["state"],"QUOTE_READY")
            self.assertEqual(len(self.store.candle_index(NOW)["AAAUSDT"]),3)
        finally:
            for worker in workers:worker.join(2)


class ClosedStreamTests(Fixture,unittest.TestCase):
    def setUp(self): self.setup_market()

    def record(self,payload):
        return self.service.record_closed_kline(payload,received_at_ms=self.wall,received_monotonic_ns=self.mono)

    def test_P2B203_closed_bar_receipt_and_event_provenance(self):
        result=self.record(kline())
        self.assertTrue(result["data"]["rows"][0]["closed"])
        self.assertEqual(result["received_at_ms"],NOW)
        self.assertEqual(result["data"]["observation"]["exchange_event_time_ms"],NOW)

    def test_P2B203_open_bar_does_not_refresh_feature_history(self):
        self.assertIsNone(self.record(kline(False)))
        self.assertEqual(self.store.records("CANDLES"),[])

    def test_P2B203_rest_ws_same_economics_not_conflicting_unused_wire_field(self):
        self.bars([candle(NOW-60000,"102")])
        result=self.record(kline(c="102.000"))
        self.assertTrue(result["data"]["rows"][0]["valid"])

    def test_P2B203_real_closed_revision_quarantined(self):
        self.record(kline())
        self.advance(1)
        changed=kline(c="103")
        changed["E"]=self.wall
        result=self.record(changed)
        self.assertFalse(result["data"]["rows"][0]["valid"])

    def test_P2B203_unknown_interval_shape_future_or_invalid_close_incident(self):
        values=[kline(i="3m"),kline(c="130"),kline(t=NOW-59999),kline(x="true")]
        future=kline();future["E"]=NOW+1000;values.append(future)
        for value in values:
            self.assert_reason("CANDLE_SCHEMA_INVALID",self.record,value)
        self.assertTrue(self.store.records("INCIDENT"))

    def test_P2B203_future_event_rejected_under_low_decimal_precision(self):
        payload=kline();payload["E"]=NOW+1000
        with localcontext() as context:
            context.prec=1
            self.assert_reason("CANDLE_SCHEMA_INVALID",self.record,payload)

    def test_P2B203_future_receipt_cannot_claim_current_closed_bar(self):
        self.assert_reason("CANDLE_SCHEMA_INVALID",self.service.record_closed_kline,kline(),
            received_at_ms=NOW+1,received_monotonic_ns=self.mono+1)
        self.assertTrue(self.store.last_failure("CANDLE_COLLECTION_FAILED","CANDLES",NOW,symbol="AAAUSDT"))

    def test_P2B203_failed_batch_persists_symbol_failure_after_rollback(self):
        api=importlib.import_module("binance_demo_bot.market_stream")
        transport=FakeStream()
        reader=api.MarketStreamReader(self.store,self.client,self.refresh(),["aaausdt@kline_1m"],transport=transport,background=False)
        self.addCleanup(reader.stop)
        reader.start()
        transport.frames.extend([{"stream":"aaausdt@kline_1m","data":kline()},
                                 {"stream":"aaausdt@kline_1m","data":kline(c="130")}])
        reader.read_once()
        self.assertEqual(self.store.records("CANDLES"),[])
        self.assertTrue(self.store.last_failure("CANDLE_COLLECTION_FAILED","CANDLES",NOW,symbol="AAAUSDT"))

    def test_P2B203_topics_every_pair_and_shard_limits(self):
        api=importlib.import_module("binance_demo_bot.market_stream")
        universe=self.refresh([pair(str(i),"BTC") for i in range(300)])
        plan=api.stream_plan(universe,include_candles=True)
        topics=[t for shard in plan["shards"] for t in shard["topics"]]
        self.assertEqual(len(topics),600)
        self.assertEqual(sum(t.endswith("@kline_1m") for t in topics),300)
        self.assertTrue(all(len(s["topics"])<=512 for s in plan["shards"]))

    def test_P2B203_reader_persists_closed_kline_and_ignores_open(self):
        api=importlib.import_module("binance_demo_bot.market_stream")
        transport=FakeStream()
        reader=api.MarketStreamReader(self.store,self.client,self.refresh(),["aaausdt@kline_1m"],transport=transport,background=False)
        self.addCleanup(reader.stop)
        reader.start()
        transport.frames.append({"stream":"aaausdt@kline_1m","data":kline(False)})
        reader.read_once()
        transport.frames.append({"stream":"aaausdt@kline_1m","data":kline()})
        reader.read_once()
        self.assertEqual(len(self.store.records("CANDLES")),1)

    def test_P2B201_reader_copies_membership_without_catalog_body(self):
        api=importlib.import_module("binance_demo_bot.market_stream")
        universe=self.refresh([pair(newLargeField=["x"]*10000)])
        reader=api.MarketStreamReader(self.store,self.client,universe,["aaausdt@bookTicker"],transport=FakeStream(),background=False)
        self.addCleanup(reader.stop)
        self.assertFalse("observation" in reader.universe["data"])
        universe["data"]["rows"][0]["base"]="OTHER"
        self.assertEqual(reader.members["AAAUSDT"]["base"],"AAA")

    def test_P2B201_supervisor_validates_full_body_once_for_all_readers(self):
        api=importlib.import_module("binance_demo_bot.market_stream")
        universe=self.refresh([pair(str(i)) for i in range(300)])
        original=api.MarketDataService._universe
        calls=[]
        def counting(service,value,cutoff):
            calls.append(value)
            return original(service,value,cutoff)
        with patch.object(api.MarketDataService,"_universe",counting):
            supervisor=api.MarketStreamSupervisor(self.store,self.client,universe,include_candles=True,transport_factory=FakeStream)
            self.addCleanup(supervisor.stop)
            universe["data"]["rows"][0]["base"]="CHANGED"
            supervisor.start()
            self.assertEqual(supervisor.health()["connected_shards_n"],2)
            self.assertEqual(len(calls),1)
            self.assertFalse("observation" in supervisor.universe["data"])


class FakeSupervisor:
    def __init__(self,store,client,universe,deep_symbols=(),**kwargs):
        self.store,self.client,self.universe=store,client,universe
        self.attempts=[]
        self.include_candles=kwargs.get("include_candles",False)
        self.running=False
    def start(self): self.running=True
    def tick(self): pass
    def stop(self): self.running=False
    def health(self):
        return dict(running=self.running,connected_shards_n=int(self.running),planned_shards_N=1,
                    fresh_quotes_n=0,ready_depth_n=0,deep_pairs_N=0,shards={})


class RuntimeTests(Fixture,unittest.TestCase):
    def setUp(self):
        self.setup_market()
        self.api=importlib.import_module("binance_demo_bot.market_runtime")
        patched=patch.object(self.api,"MarketStreamSupervisor",FakeSupervisor)
        patched.start();self.addCleanup(patched.stop)

    def runtime(self,**kwargs):
        result=self.api.MarketDataRuntime(self.store,self.client,**kwargs)
        self.addCleanup(result.stop)
        return result

    def start(self,rows=None):
        runtime=self.runtime()
        self.replies.append(metadata(rows))
        runtime.start()
        return runtime

    def test_P2B204_disabled_zero_network(self):
        runtime=self.runtime(enabled=False)
        self.assert_reason("MARKET_DATA_DISABLED",runtime.start)
        self.assertEqual(self.calls,[])

    def test_P2B204_public_start_candles_enabled_unknown_account_N(self):
        runtime=self.start()
        self.assertTrue(runtime.supervisor.include_candles)
        self.assertIsNone(runtime.health()["permitted_pairs_N"])
        self.assertFalse(runtime.health()["account_certified"])

    def test_P2B204_metadata_rotate_preserves_quota_and_past_membership(self):
        runtime=self.start()
        old=runtime.supervisor
        old.attempts.append(self.mono)
        self.advance(600000)
        self.clock.observe(self.wall,self.wall,self.wall,self.mono,self.mono)
        self.replies.extend([metadata([pair("BBB")]),[],[]])
        runtime.tick()
        self.assertFalse(old.running)
        self.assertIs(runtime.supervisor.attempts,old.attempts)
        self.assertEqual(runtime.universe["data"]["rows"][0]["symbol"],"BBBUSDT")
        self.assertEqual(self.universe_service.current(NOW)["data"]["rows"][0]["symbol"],"AAAUSDT")

    def test_P2B204_failed_refresh_is_stale_unknown_not_success(self):
        runtime=self.start()
        self.advance(600000)
        self.clock.observe(self.wall,self.wall,self.wall,self.mono,self.mono)
        self.replies.append(AdapterError("API_REJECTED"))
        runtime.tick()
        self.assertEqual(runtime.health()["pipeline_state"],"METADATA_REFRESH_FAILED")
        self.assertFalse(runtime.health()["account_certified"])

    def test_P2B205_fair_bounded_bootstrap_and_actual_rest_fallback(self):
        runtime=self.start([pair(),pair("BBB"),pair("CCC")])
        self.replies.extend([[book()],[],[]])
        runtime.tick()
        from urllib.parse import parse_qs
        queried=[parse_qs(c[2])["symbol"][0] for c in self.calls if c[1]=="/api/v3/klines"]
        self.assertEqual(queried,["AAAUSDT","BBBUSDT"])
        self.assertEqual(sum(c[1]=="/api/v3/ticker/bookTicker" for c in self.calls),1)
        self.advance(250)
        self.replies.extend([[],[]])
        runtime.tick()
        queried=[parse_qs(c[2])["symbol"][0] for c in self.calls if c[1]=="/api/v3/klines"]
        self.assertEqual(queried[:3],["AAAUSDT","BBBUSDT","CCCUSDT"])

    def test_P2B204_candidate_clock_swap_not_transient_unsynced(self):
        runtime=self.start()
        previous=self.clock.sample
        seen=[]
        original=self.client._read
        def observing(path,params,weight):
            if path=="/api/v3/time":seen.append(self.clock.sample)
            return original(path,params,weight)
        self.replies.append({"serverTime":NOW})
        with patch.object(self.client,"_read",side_effect=observing):
            runtime.refresh_clock()
        self.assertEqual(seen,[previous])
        self.assertIsNotNone(self.clock.sample)

    def test_P2B205_stop_prevents_more_work(self):
        runtime=self.start()
        runtime.stop()
        before=len(self.calls)
        runtime.tick()
        self.assertEqual(len(self.calls),before)

    def test_P2B205_evidence_discovered_active_missing_history_separate(self):
        runtime=self.start([pair(),pair("BBB",status="BREAK")])
        health=runtime.health()
        self.assertEqual(health["discovered_pairs_n"],2)
        self.assertEqual(health["active_public_pairs_N"],1)
        self.assertEqual(health["closed_history_ready_n"],0)
        self.assertIsNone(health["permitted_ready_ratio"])
        self.assertFalse(health["production_qualified"])

    def test_P2B205_removed_rows_do_not_inflate_current_exclusion_counts(self):
        runtime=self.start([pair(),pair("BBB")])
        self.advance(600000)
        self.clock.observe(self.wall,self.wall,self.wall,self.mono,self.mono)
        self.replies.extend([metadata([pair()]),[],[]])
        runtime.tick()
        health=runtime.health()
        self.assertEqual(health["discovered_pairs_n"],health["active_public_pairs_N"]+health["status_excluded_n"])
        self.assertEqual(health["removed_tombstones_n"],1)

    def test_P2B205_disconnected_streams_degrade_despite_public_rest(self):
        runtime=self.start()
        runtime.supervisor.running=False
        self.assertEqual(runtime.health()["pipeline_state"],"STREAMS_DEGRADED")
        runtime.supervisor.running=True
        self.assertEqual(runtime.health()["pipeline_state"],"OBSERVING")
        runtime.stop()
        self.assertEqual(runtime.health()["pipeline_state"],"STOPPED")

    def test_P2B201_runtime_quotes_do_not_hash_full_catalog_again(self):
        runtime=self.start()
        self.replies.extend([[book()],[],[]])
        with patch.object(self.service.__class__,"_universe",side_effect=AssertionError("full catalog on live hot path")):
            runtime.tick()
        self.assertEqual(runtime.health()["quote_ready_n"],1)

    def test_P2B205_failed_candle_closes_runtime_history_count(self):
        runtime=self.start()
        self.bars()
        self.assertEqual(runtime.health()["closed_history_ready_n"],1)
        self.store.append("INCIDENT",dict(operation="CANDLE_COLLECTION_FAILED",symbol="AAAUSDT",reason="READ_UNAVAILABLE"),
            received_at_ms=self.wall,received_monotonic_ns=self.mono)
        self.assertEqual(runtime.health()["closed_history_ready_n"],0)

    def test_P2B204_successful_clock_refresh_recovers_failed_clock_only(self):
        runtime=self.start()
        self.advance(30000)
        with patch.object(runtime,"refresh_clock",side_effect=AdapterError("CLOCK_QUALITY")):
            runtime.tick()
        self.assertEqual(runtime.health()["pipeline_state"],"CLOCK_REFRESH_FAILED")
        self.replies.extend([{"serverTime":self.wall},[],[],[]])
        runtime.tick()
        self.assertEqual(runtime.health()["pipeline_state"],"OBSERVING")

    def test_P2B205_latency_percentiles_are_recent_window_evidence(self):
        runtime=self.start()
        runtime._tick_lags.extend([Decimal(1),Decimal(2),Decimal(100)])
        health=runtime.health()
        self.assertEqual(health["processing_tick_p50_ms"],Decimal(2))
        self.assertEqual(health["processing_tick_p95_ms"],Decimal(100))
        self.assertEqual(health["processing_tick_recent_n"],3)
