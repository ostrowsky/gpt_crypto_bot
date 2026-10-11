"""PB-01..03: spec-first private stream contracts; synthetic keys only."""

import importlib
import json
import ssl
import time
import unittest
from decimal import Decimal, localcontext
from types import SimpleNamespace
from unittest.mock import patch

from _p1_support import FixtureEnvironment, SyntheticWs, execution, NOW, TOKEN


class PrivateStreamTests(FixtureEnvironment, unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module("binance_demo_bot.private_stream")
        self.evidence = importlib.import_module("binance_demo_bot.account_evidence")
        self.set_fixture()
        self.ws = SyntheticWs()
        self.reader = self.make_reader()
        self.addCleanup(lambda: self.reader.close())

    def make_reader(self, **changes):
        values = dict(transport=self.ws, process_instance_id="fixture-process", background=False)
        values.update(changes)
        return self.api.PrivateEventReader(self.adapter, self.store, **values)

    def emit(self, event, subscription=7):
        self.ws.queue.append({"subscriptionId": subscription, "event": event})
        return self.reader.poll_once()

    def test_PB01_independent_ws_signing_differs_from_rest_encoding(self):
        self.assertEqual(self.api.sign_ws({"apiKey": "synthetic-key", "timestamp": 123456,
                         "symbol": "ΔUSDT", "quantity": Decimal("1.2300"), "flag": True}, "synthetic-secret"),
                         "0d89bb557dc9aa8352dbcfeadc727bf10fb46b2cb430b13151c7caf71969b308")

    def test_PB01_signing_rejects_duplicate_reserved_or_float_values(self):
        for parameters in ({"a": 0.5}, {"signature": "x"}, [("x", 1), ("x", 2)]):
            self.assert_reason("INVALID_PARAMETERS", self.api.sign_ws, parameters, "secret")

    def test_PB01_invalid_targets_and_mutation_rejected_before_connection(self):
        for url in ("wss://ws-api.binance.com/ws-api/v3", "ws://demo-ws-api.binance.com/ws-api/v3",
                    "wss://@demo-ws-api.binance.com/ws-api/v3", "wss://demo-ws-api.binance.com:9443/ws-api/v3",
                    "wss://demo-ws-api.binance.com/ws-api/v3#", "wss://demo-ws-api.binance.com/ws-api/v3\n"):
            self.assert_reason("FORBIDDEN_ENDPOINT", self.api.DemoWsTransport, url=url)
        transport = self.api.DemoWsTransport()
        self.assert_reason("FORBIDDEN_METHOD", transport.send, {"id": "x", "method": "order.place", "params": {}})

    def test_PB01_tls_proxy_limits_and_logger_config(self):
        connection = SimpleNamespace(request=SimpleNamespace(path="/ws-api/v3",
                                                             headers={"Host": "demo-ws-api.binance.com"}))
        connection.close = lambda: None
        with patch.object(self.api, "ws_connect", return_value=connection) as connect:
            transport = self.api.DemoWsTransport()
            transport.connect()
            kwargs = connect.call_args.kwargs
            self.assertIsNone(kwargs["proxy"])
            self.assertEqual(kwargs["ssl"].verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(kwargs["ssl"].check_hostname)
            self.assertEqual(kwargs["max_size"], 1024*1024)
            self.assertEqual(kwargs["max_queue"], 64)
            self.assertTrue(kwargs["logger"].disabled)
            transport.close()

    def test_PB01_alternate_handshake_host_rejected_before_private_send(self):
        with patch.object(self.api, "ws_connect") as connect:
            connection = connect.return_value
            connection.request.path = "/ws-api/v3"
            connection.request.headers = {"Host": "api.binance.com"}
            transport = self.api.DemoWsTransport()
            self.assert_reason("HANDSHAKE_TARGET_MISMATCH", transport.connect)
            connection.send.assert_not_called()
            connection.close.assert_called_once()

    def test_PB02_subscribe_identity_and_matching_ack(self):
        self.reader.start()
        self.assertEqual(len(self.ws.sent), 1)
        request = self.ws.sent[0]
        self.assertEqual(request["method"], "userDataStream.subscribe.signature")
        self.assertEqual(request["params"]["apiKey"], "synthetic-key")
        self.assertEqual(request["params"]["recvWindow"], 5000)
        self.assertEqual(len(request["params"]["signature"]), 64)
        self.assertTrue(self.reader.health()["connected"])
        self.assertFalse(self.reader.health()["exchange_reconciled"])
        self.assertFalse(self.reader.health()["order_send_enabled"])

    def test_PB02_events_before_ack_keep_original_receipt(self):
        self.ws.queue.append({"subscriptionId": 7, "event": execution()})
        self.reader.start()
        self.assertEqual(len(self.store.fills()), 1)
        self.assertEqual(self.store.events()[0]["received_at_ms"], NOW+1)

    def test_PB02_wrong_ack_status_or_missing_subscription_does_not_subscribe(self):
        self.ws.ack = {"status": 401, "error": {"msg": "synthetic-secret", "code": -2015}}
        error = self.assert_reason("WS_AUTH_DENIED", self.reader.start)
        self.assertNotIn("synthetic-secret", str(error))
        self.assertTrue(self.ws.closed)
        self.assertEqual(len(self.ws.sent), 1)
        self.assertFalse(self.reader.health()["connected"])

    def test_PB02_wrong_account_token_never_sends_subscription(self):
        self.store.close()
        self.store = None
        # Own separate database, not reuse with a mismatched frozen identity.
        self.store = self.evidence.AccountEvidenceStore(self.paths, account_alias="demo-one",
            account_identity="a"*64, episode_id="episode-one", scope_id="whole-account", database="other.sqlite")
        self.reader = self.make_reader()
        self.assert_reason("ACCOUNT_IDENTITY_MISMATCH", self.reader.start)
        self.assertFalse(self.ws.sent)

    def test_PB02_unknown_subscription_records_incident_and_closes(self):
        self.reader.start()
        self.assert_reason("UNMAPPED_SUBSCRIPTION", self.emit, execution(), 8)
        self.assertTrue(self.ws.closed)
        self.assertTrue(self.store.incidents())

    def test_PB02_preack_overflow_bounded(self):
        self.ws.queue = [{"subscriptionId": 7, "event": execution(I=i)} for i in range(65)]
        self.assert_reason("WS_BUFFER_OVERFLOW", self.reader.start)
        self.assertEqual(len(self.ws.sent), 1)
        self.assertTrue(self.ws.closed)

    def test_PB03_trade_normalization_matches_money_and_receipt(self):
        self.reader.start()
        self.emit(execution())
        value = self.store.fills()[0]["data"]
        self.assertEqual(Decimal(value["quantity"]), Decimal("1"))
        self.assertEqual(Decimal(value["commission_quantity"]), Decimal("0.001"))
        self.assertEqual(value["commission_asset"], "BNB")
        self.assertEqual(value["ownership"], "UNATTRIBUTED")
        self.assertEqual(self.store.cursor("BTCUSDT"), 0)

    def test_PB03_nontrade_order_update_is_not_fill(self):
        self.reader.start()
        self.emit(execution(x="NEW", X="NEW", l="0", L="0", Y="0", n="0", N=None, t=-1, z="0", Z="0"))
        self.assertFalse(self.store.fills())
        self.assertEqual(self.store.events()[0]["data"]["order"]["status"], "NEW")

    def test_PB03_partial_account_balance_is_not_full_snapshot(self):
        self.reader.start()
        self.emit({"e": "outboundAccountPosition", "E": NOW-10, "u": NOW-11,
                   "B": [{"a": "BTC", "f": "0.1", "l": "0.2"}]})
        value = self.store.events()[0]["data"]
        self.assertEqual(value["kind"], "PARTIAL_BALANCES")
        self.assertEqual(len(value["balances"]), 1)
        self.assertFalse(self.reader.health()["exchange_reconciled"])

    def test_PB03_flow_is_unclassified_and_blocking(self):
        self.reader.start()
        self.emit({"e": "balanceUpdate", "E": NOW-10, "T": NOW-11, "a": "USDT", "d": "-1.5"})
        self.assertEqual(self.store.events()[0]["data"]["delta"], "-1.5")
        self.assertIn("UNCLASSIFIED_FLOW", [r["data"]["reason"] for r in self.store.incidents()])
        self.assertFalse(self.store.fills())

    def test_PB03_unknown_termination_and_lock_are_quarantined(self):
        for event in ({"e": "future-private-type", "E": NOW, "secret": "synthetic-secret"},
                      {"e": "eventStreamTerminated", "E": NOW}, {"e": "externalLockUpdate", "E": NOW}):
            reader = self.make_reader(transport=SyntheticWs())
            reader.start()
            reader.transport.queue.append({"subscriptionId": 7, "event": event})
            with self.assertRaises(Exception):
                reader.poll_once()
            reader.close()
        self.assertNotIn("synthetic-secret", str(self.store.incidents()))
        self.assertFalse(self.store.fills())

    def test_PB03_malformed_future_or_unknown_order_fails_closed(self):
        self.reader.start()
        self.assert_reason("SCHEMA_INVALID", self.emit, execution(l="NaN"))
        self.assertFalse(self.store.fills())
        self.assertTrue(self.ws.closed)

    def test_PB03_list_status_preserves_member_ids(self):
        self.reader.start()
        self.emit({"e": "listStatus", "E": NOW-10, "s": "BTCUSDT", "g": 31, "c": "OCO",
                   "l": "EXEC_STARTED", "L": "EXECUTING", "C": "manual-list", "T": NOW-11,
                   "O": [{"s": "BTCUSDT", "i": 21, "c": "manual-order"}]})
        value = self.store.events()[0]["data"]["list"]
        self.assertEqual(value["order_list_id"], 31)
        self.assertEqual(value["orders"][0]["order_id"], 21)

    def test_PB03_silent_stream_not_certified_by_pong(self):
        self.reader.start()
        self.assertIsNone(self.reader.poll_once())
        self.reader.heartbeat()
        self.assertTrue(self.reader.health()["transport_healthy"])
        self.assertFalse(self.reader.health()["exchange_reconciled"])
        self.assertFalse(self.reader.health()["private_events_certified"])

    def test_PB03_pong_failure_records_gap(self):
        self.reader.start()
        self.ws.pong = False
        self.assert_reason("WS_HEARTBEAT_FAILED", self.reader.heartbeat)
        self.assertTrue(self.ws.closed)
        self.assertTrue(self.store.incidents())

    def test_PB03_reader_thread_persists_and_stops_only_itself(self):
        self.reader = self.make_reader(background=True)
        self.reader.start()
        self.ws.queue.append({"subscriptionId": 7, "event": execution()})
        deadline = time.monotonic()+2
        while not self.store.fills() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(self.store.fills()), 1)
        self.reader.close()
        self.assertFalse(self.reader.health()["connected"])
        self.assertFalse(self.reader.health()["reader_alive"])

    def test_PB03_positive_commission_without_asset_not_invented_usdt(self):
        self.reader.start()
        self.assert_reason("SCHEMA_INVALID", self.emit, execution(N=None))
        self.assertFalse(self.store.fills())

    def test_PB03_empty_positive_commission_asset_is_invalid(self):
        self.reader.start()
        self.assert_reason("SCHEMA_INVALID", self.emit, execution(N=""))
        self.assertFalse(self.store.fills())

    def test_PB03_future_exchange_event_rejected(self):
        self.reader.start()
        self.assert_reason("SCHEMA_INVALID", self.emit, execution(E=NOW+9999))
        self.assertFalse(self.store.events())

    def test_PB02_wrong_ack_id_or_missing_id_never_resends(self):
        self.ws.ack = dict(id="foreign-id", status=200, result={"subscriptionId": 7})
        self.assert_reason("WS_PROTOCOL_ERROR", self.reader.start)
        self.assertEqual(len(self.ws.sent), 1)
        self.assertFalse(self.reader.health()["connected"])

    def test_PB02_missing_ack_subscription_id_is_protocol_error(self):
        self.ws.ack = dict(status=200, result={})
        self.assert_reason("WS_PROTOCOL_ERROR", self.reader.start)
        self.assertEqual(len(self.ws.sent), 1)

    def test_PB03_unknown_dependency_reason_is_redacted(self):
        self.reader.start()
        self.ws.queue.append(self.api.AdapterError("synthetic-secret"))
        error = self.assert_reason("WS_PROTOCOL_ERROR", self.reader.poll_once)
        self.assertNotIn("synthetic-secret", str(error))

    def test_PB01_handshake_cleanup_exception_cannot_expose_secret(self):
        with patch.object(self.api, "ws_connect") as connect:
            connection = connect.return_value
            connection.request.path = "/ws-api/v3"
            connection.request.headers = {"Host": "api.binance.com"}
            connection.close.side_effect = RuntimeError("synthetic-secret")
            error = self.assert_reason("HANDSHAKE_TARGET_MISMATCH", self.api.DemoWsTransport().connect)
            self.assertNotIn("synthetic-secret", str(error))

    def test_PB03_negative_flow_exact_under_low_decimal_precision(self):
        self.reader.start()
        with localcontext() as context:
            context.prec = 2
            self.emit({"e": "balanceUpdate", "E": NOW-10, "T": NOW-11, "a": "USDT", "d": "-1234.56"})
        self.assertEqual(self.store.events()[0]["data"]["delta"], "-1234.56")

    def test_PB02_ws_rate_limit_blocks_shared_rest_budget(self):
        self.ws.ack = dict(status=429, retryAfter=NOW+2000)
        self.assert_reason("WS_RATE_LIMITED", self.reader.start)
        self.assert_reason("RATE_LIMITED", self.adapter.read_account)
        self.assertEqual(len(self.ws.sent), 1)

    def test_PB02_ws_minute_weight_is_seen_by_rest_limiter(self):
        self.ws.ack = dict(status=200, result={"subscriptionId": 7},
                          rateLimits=[dict(rateLimitType="REQUEST_WEIGHT", interval="MINUTE", intervalNum=1, count=1199, limit=6000)])
        self.reader.start()
        self.assert_reason("LOCAL_RATE_BUDGET", self.adapter.read_account)
