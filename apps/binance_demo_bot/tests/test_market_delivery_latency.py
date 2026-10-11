"""P2B2-09: contract tests before durable latency/evidence implementation."""
import hashlib
import importlib
import json
import unittest
from decimal import Decimal, localcontext
from unittest.mock import patch
from _p2_support import Fixture, NOW
from test_market_stream import FakeStream, tick
from test_market_runtime import kline


class RawStream(FakeStream):
    def recv(self, timeout_s=1):
        if self.frames and isinstance(self.frames[0], (str, bytes)):
            return self.frames.pop(0)
        return super().recv(timeout_s)


class PreparedTimingTests(Fixture, unittest.TestCase):
    def setUp(self): self.setup_market()

    def pending(self): return self.store.prepared_batch(monotonic_ns=lambda: self.mono)

    def test_P2B209_actual_sql_phase_markers_and_detached_timing(self):
        original = self.store._con.execute
        def delayed(sql, *args):
            result = original(sql, *args)
            if sql.startswith("BEGIN"): self.advance(3)
            elif sql.startswith("INSERT"): self.advance(5)
            elif sql=="COMMIT": self.advance(7)
            return result
        with patch.object(self.store._con, "execute", delayed):
            with self.pending() as pending:
                self.advance(11)
                pending.append("FEATURE", {}, received_at_ms=NOW, received_monotonic_ns=1_000_000_000)
        timing = pending.timing
        self.assertEqual(timing["state"], "COMMITTED")
        self.assertEqual(timing["prepare_end_ns"]-timing["prepare_start_ns"], 11_000_000)
        self.assertEqual(timing["begin_finished_ns"]-timing["writer_acquired_ns"], 3_000_000)
        self.assertEqual(timing["insert_finished_ns"]-timing["begin_finished_ns"], 5_000_000)
        self.assertEqual(timing["commit_finished_ns"]-timing["insert_finished_ns"], 7_000_000)
        timing["state"] = "TAMPERED"
        self.assertEqual(pending.timing["state"], "COMMITTED")
        self.assertEqual(len(self.store.records("FEATURE")), 1)

    def test_P2B209_rollback_has_no_successful_commit_marker(self):
        pending = self.pending()
        self.store._con.execute("CREATE TEMP TRIGGER refuse BEFORE INSERT ON snapshots "
            "WHEN NEW.kind='FEATURE' BEGIN SELECT RAISE(ABORT,'test failure'); END")
        with self.assertRaises(Exception):
            with pending: pending.append("FEATURE", {}, received_at_ms=NOW, received_monotonic_ns=self.mono)
        self.assertEqual(pending.timing["state"], "FAILED")
        self.assertIsNone(pending.timing["commit_finished_ns"])
        self.assertIsNotNone(pending.timing["failure_at_ns"])
        self.assertEqual(self.store.records("FEATURE"), [])

    def test_P2B209_clock_option_rejects_noncallable(self):
        self.assert_reason("INVALID_PARAMETERS", self.store.prepared_batch, monotonic_ns=123)


class DeliveryTests(Fixture, unittest.TestCase):
    def setUp(self):
        self.setup_market()
        self.api = importlib.import_module("binance_demo_bot.market_stream")
        no_network = patch.object(self.api, "ws_connect", side_effect=AssertionError("network forbidden"))
        blocked = no_network.start(); self.addCleanup(no_network.stop)
        self.addCleanup(blocked.assert_not_called)

    def reader(self, topics=None, **options):
        transport = RawStream()
        reader = self.api.MarketStreamReader(self.store, self.client, self.refresh(),
            topics or ["aaausdt@bookTicker"], transport=transport, background=False, **options)
        self.addCleanup(reader.stop); reader.start()
        return reader, transport

    def evidence(self):
        return [row for row in self.store.records("INCIDENT")
            if row["data"].get("operation")=="MARKET_GROUP_LAG_EVIDENCE"]

    def test_P2B209_idle_is_unobserved_not_zero_latency(self):
        reader, _ = self.reader()
        self.assertFalse(reader.read_once())
        latency = reader.health()["latency"]
        self.assertEqual(latency["scope"], "READ_ONCE_STEADY_STATE")
        self.assertEqual(latency["attempted_groups_n"], 0)
        self.assertIsNone(latency["last_group"])
        self.assertFalse(latency["startup_observed"])

    def test_P2B209_stage_attribution_uses_oldest_receipt(self):
        reader, transport = self.reader()
        transport.frames.extend([tick(u=1), tick(u=2)])
        original = self.store._con.execute
        def delayed(sql, *args):
            result = original(sql, *args)
            if sql.startswith("BEGIN"): self.advance(3)
            elif sql.startswith("INSERT"): self.advance(5)
            elif sql=="COMMIT": self.advance(7)
            return result
        with patch.object(self.store._con, "execute", delayed): self.assertTrue(reader.read_once())
        latency = reader.health()["latency"]
        group = latency["last_group"]
        self.assertEqual(group["frames_N"], 2)
        self.assertEqual(group["begin_ms"], Decimal(3))
        self.assertEqual(group["insert_ms"], Decimal(5))
        self.assertEqual(group["commit_ms"], Decimal(7))
        self.assertEqual(group["end_to_end_ms"], Decimal(15))
        self.assertEqual(latency["durable_groups_n"], 1)
        self.assertEqual(latency["passed_groups_n"], 1)
        group["commit_ms"] = -1
        self.assertEqual(reader.health()["latency"]["last_group"]["commit_ms"], 7)

    def commit_delay(self, delay):
        original = self.store._con.commit
        def delayed():
            original(); self.advance(delay)
        return patch.object(self.store._con, "commit", delayed)

    def test_P2B209_exact_deadline_passes_without_relaxing_guard(self):
        reader, transport = self.reader(); transport.frames.append(tick())
        with self.commit_delay(250): self.assertTrue(reader.read_once())
        self.assertTrue(reader.connected)
        self.assertEqual(reader.health()["latency"]["late_groups_n"], 0)
        self.assertEqual(reader.health()["latency"]["last_group"]["end_to_end_ms"], 250)

    def test_P2B209_late_commit_fails_and_preserves_committed_raw_source(self):
        reader, transport = self.reader()
        wire = json.dumps(tick(), separators=(",", ":"))
        transport.frames.append(wire)
        with self.commit_delay(251): self.assertFalse(reader.read_once())
        self.assertFalse(reader.connected); self.assertEqual(reader.error, "WS_PROCESSING_LAG")
        self.assertEqual(reader.health()["fresh_quotes_n"], 0)
        latency = reader.health()["latency"]
        self.assertEqual((latency["durable_groups_n"], latency["late_groups_n"]), (1, 1))
        self.assertEqual(latency["last_group"]["end_to_end_ms"], 251)
        evidence = self.evidence()[-1]
        frame = evidence["data"]["frames"][0]
        self.assertEqual(frame["raw_utf8"], wire)
        self.assertEqual(frame["wire_sha256"], hashlib.sha256(wire.encode()).hexdigest())
        self.assertEqual(frame["received_at_ms"], NOW)
        self.assertEqual(frame["received_monotonic_ns"], 1_000_000_000)
        self.assertTrue(evidence["data"]["committed"])
        self.assertGreaterEqual(evidence["received_at_ms"], NOW+251)
        for identifier in evidence["data"]["committed_snapshot_ids"]:
            self.assertIsNotNone(self.store.get(identifier))
        self.assertEqual(self.store.quote_index(self.wall)["AAAUSDT"][0]["state"], "QUOTE_INVALID")

    def test_P2B209_preparation_lag_rolls_back_and_records_all_frames(self):
        reader, transport = self.reader()
        wires = [json.dumps(tick(u=value)) for value in (1, 2)]
        transport.frames.extend(wires)
        original = reader._process
        def late(*args): self.advance(251); return original(*args)
        with patch.object(reader, "_process", late): self.assertFalse(reader.read_once())
        self.assertEqual([row for row in self.store.records("QUOTES") if "wire" in row["data"]], [])
        evidence = self.evidence()[-1]["data"]
        self.assertFalse(evidence["committed"])
        self.assertEqual(evidence["committed_snapshot_ids"], [])
        self.assertEqual([frame["raw_utf8"] for frame in evidence["frames"]], wires)
        self.assertEqual(reader.health()["latency"]["durable_groups_n"], 0)

    def test_P2B209_open_bar_no_write_not_durable_and_window_bounded(self):
        reader, transport = self.reader(["aaausdt@kline_1m"], latency_window=3)
        for _ in range(4):
            transport.frames.append({"stream":"aaausdt@kline_1m", "data":kline(False)})
            self.assertTrue(reader.read_once())
        latency = reader.health()["latency"]
        self.assertEqual(latency["attempted_groups_n"], 4)
        self.assertEqual(latency["recent_groups_n"], 3)
        self.assertEqual(latency["window_limit_N"], 3)
        self.assertEqual(latency["durable_groups_n"], 0)
        self.assertEqual(latency["last_group"]["commit_state"], "NO_WRITE")
        self.assertIsNone(latency["last_group"]["commit_ms"])

    def test_P2B209_evidence_failure_is_visible(self):
        reader, transport = self.reader(); transport.frames.append(tick())
        self.store._con.execute("CREATE TEMP TRIGGER refuse_evidence BEFORE INSERT ON snapshots "
            "WHEN NEW.kind='INCIDENT' AND json_extract(NEW.payload_json,'$.data.operation')='MARKET_GROUP_LAG_EVIDENCE' "
            "BEGIN SELECT RAISE(ABORT,'test evidence failure'); END")
        with self.commit_delay(251):
            self.assertFalse(reader.read_once())
        latency = reader.health()["latency"]
        self.assertEqual(latency["evidence_failures_n"], 1)
        self.assertEqual(latency["last_evidence_state"], "PERSISTENCE_FAILED")
        self.assertNotIn("frames", latency)
        self.assertFalse(reader.connected)

    def test_P2B209_latency_window_validation(self):
        for value in (True, 0, 4097, "3"):
            self.assert_reason("INVALID_PARAMETERS", self.reader, latency_window=value)

    def test_P2B209_metrics_precision_independent_of_ambient_decimal(self):
        reader, transport = self.reader(); transport.frames.append(tick())
        original = self.store._con.commit
        def delayed(): original(); self.mono += 1234567; self.wall += 1
        with patch.object(self.store._con, "commit", delayed): self.assertTrue(reader.read_once())
        with localcontext() as context:
            context.prec = 2
            self.assertEqual(reader.health()["latency"]["last_group"]["end_to_end_ms"], Decimal("1.234567"))

    def test_P2B209_failed_invalidation_does_not_double_count_or_leave_transport_open(self):
        reader, transport = self.reader(); transport.frames.append(tick())
        self.store._con.execute("CREATE TEMP TRIGGER refuse_invalidation BEFORE INSERT ON snapshots "
            "WHEN NEW.kind='QUOTES' AND json_extract(NEW.payload_json,'$.data.rows[0].reasons[0]')='WS_PROCESSING_LAG' "
            "BEGIN SELECT RAISE(ABORT,'test invalidation failure'); END")
        with self.commit_delay(251):
            self.assertFalse(reader.read_once())
        health = reader.health()
        self.assertEqual(health["latency"]["attempted_groups_n"], 1)
        self.assertEqual(health["latency"]["durable_groups_n"], 1)
        self.assertEqual(health["latency"]["late_groups_n"], 1)
        self.assertEqual(health["invalidation_state"], "PERSISTENCE_FAILED")
        self.assertEqual(reader.error, "WS_PROCESSING_LAG")
        self.assertTrue(transport.closed)

    def test_P2B209_float_in_rejected_packet_has_lossless_wire_and_decoded_json(self):
        reader, transport = self.reader()
        payload = tick(); payload["data"]["u"] = 10.5
        wire = json.dumps(payload); transport.frames.append(wire)
        original = reader._process
        def delayed(*args): self.advance(251); return original(*args)
        with patch.object(reader, "_process", delayed): self.assertFalse(reader.read_once())
        frame = self.evidence()[-1]["data"]["frames"][0]
        self.assertEqual(json.loads(frame["decoded_json"]), payload)
        self.assertEqual(frame["raw_utf8"], wire)
        self.assertEqual(frame["wire_sha256"], hashlib.sha256(wire.encode()).hexdigest())

    def test_P2B209_future_receipt_produces_unknown_duration_not_zero(self):
        reader, _ = self.reader()
        with patch.object(reader, "_receive", side_effect=[(tick(), NOW, self.mono+1), TimeoutError()]):
            self.assertFalse(reader.read_once())
        group = reader.health()["latency"]["last_group"]
        self.assertIsNone(group["end_to_end_ms"])
        self.assertEqual(group["failure_reason"], "WS_PROCESSING_LAG")
        frame = self.evidence()[-1]["data"]["frames"][0]
        self.assertFalse(frame["exact_wire_observed"])
        self.assertIsNone(frame["wire_sha256"])

    def test_P2B209_late_failure_uses_one_atomic_failure_commit(self):
        reader, transport = self.reader(); transport.frames.append(tick())
        traced = []; self.store._con.set_trace_callback(traced.append)
        try:
            with self.commit_delay(251): self.assertFalse(reader.read_once())
        finally: self.store._con.set_trace_callback(None)
        self.assertEqual(sum(sql=="COMMIT" for sql in traced), 2)
        self.assertEqual(len(self.evidence()), 1)
        self.assertEqual(reader.health()["invalidation_state"], "PERSISTED")

    def test_P2B209_failure_rollback_removes_staged_evidence_and_invalidations(self):
        reader, transport = self.reader(); transport.frames.append(tick())
        self.store._con.execute("CREATE TEMP TRIGGER refuse_incident BEFORE INSERT ON snapshots "
            "WHEN NEW.kind='INCIDENT' AND json_extract(NEW.payload_json,'$.data.operation')='MARKET_STREAM_FAILED' "
            "BEGIN SELECT RAISE(ABORT,'test last failure row'); END")
        with self.commit_delay(251): self.assertFalse(reader.read_once())
        self.assertEqual(self.evidence(), [])
        invalidations = [row for row in self.store.records("QUOTES")
            if any("WS_PROCESSING_LAG" in entry.get("reasons", []) for entry in row["data"]["rows"])]
        self.assertEqual(invalidations, [])
        self.assertEqual(reader.health()["latency"]["last_evidence_state"], "PERSISTENCE_FAILED")
        self.assertEqual(reader.health()["invalidation_state"], "PERSISTENCE_FAILED")
        self.assertEqual(reader.health()["fresh_quotes_n"], 0)
        self.assertTrue(transport.closed)
