"""P1-A contract tests written before adapter implementation; synthetic only."""

import importlib
import json
import ssl
import unittest
from decimal import Decimal, localcontext
from unittest.mock import patch
from urllib.parse import parse_qs

import _support  # Own source path only; no legacy client imports.
from binance_demo_bot.isolation import DemoCredentials


class TimeSource:
    def __init__(self):
        self.wall = 1_800_000_000_000
        self.mono = 10_000_000_000

    def advance(self, milliseconds):
        self.wall += milliseconds
        self.mono += milliseconds * 1_000_000


def account(**changes):
    value = {"uid": 987654321, "accountType": "SPOT", "canTrade": True,
             "permissions": ["SPOT"], "updateTime": 1000,
             "balances": [{"asset": "USDT", "free": "1000.123456789012345678",
                           "locked": "20"}, {"asset": "BTC", "free": "0", "locked": "0"}]}
    value.update(changes)
    return value


def order(**changes):
    value = {"symbol": "BTCUSDT", "orderId": 21, "orderListId": -1,
             "clientOrderId": "foreign-manual", "status": "PARTIALLY_FILLED",
             "side": "BUY", "type": "LIMIT", "price": "100",
             "origQty": "2", "executedQty": "1", "cummulativeQuoteQty": "100",
             "time": 1000, "updateTime": 2000}
    value.update(changes)
    return value


def trade(identifier=1, **changes):
    value = {"symbol": "BTCUSDT", "id": identifier, "orderId": 21, "orderListId": -1,
             "price": "100", "qty": "1", "quoteQty": "100", "commission": "0.001",
             "commissionAsset": "BNB", "time": 2000, "isBuyer": True, "isMaker": False}
    value.update(changes)
    return value


class AdapterCase(unittest.TestCase):
    def setUp(self):
        # Delayed import exposes each intended missing-interface RED scenario.
        self.api = importlib.import_module("binance_demo_bot.adapter")
        self.time = TimeSource()
        self.clock = self.api.DemoClock(wall_ms=lambda: self.time.wall,
                                        monotonic_ns=lambda: self.time.mono)
        self.clock.observe(self.time.wall, self.time.wall, self.time.wall,
                           self.time.mono, self.time.mono)
        self.calls = []
        self.replies = []
        outer = self

        class ScriptedTransport:
            def request(self, method, path, query, headers, timeout_s):
                outer.calls.append((method, path, query, dict(headers), timeout_s))
                reply = outer.replies.pop(0)
                outer.time.advance(10)
                if isinstance(reply, BaseException):
                    raise reply
                status, data, response_headers = reply
                body = data if isinstance(data, bytes) else json.dumps(data).encode()
                return outer.api.HttpResponse(status, body, response_headers)

        self.transport = ScriptedTransport()
        self.credentials = DemoCredentials("synthetic-key", "synthetic-secret")
        self.adapter = self.make_adapter()

    def make_adapter(self, **changes):
        values = dict(account_alias="demo-one", credential_generation="g1",
                      episode_id="episode-one", scope_id="whole-account",
                      identity_key=b"i" * 32, transport=self.transport, clock=self.clock,
                      sleep=lambda seconds: self.time.advance(round(seconds * 1000)))
        values.update(changes)
        return self.api.BinanceDemoAdapter(self.credentials, **values)

    def reply(self, data, status=200, headers=None):
        self.replies.append((status, data, headers or {}))

    def assert_reason(self, reason, callable_, *args, **kwargs):
        with self.assertRaises(self.api.AdapterError) as caught:
            callable_(*args, **kwargs)
        self.assertEqual(caught.exception.reason, reason)
        return caught.exception

    def test_rest01_exact_independent_signature_vector(self):
        query = self.api.sign_rest({"timestamp": 123456, "symbol": "ΔUSDT",
                                   "quantity": Decimal("1.2300"), "flag": True},
                                  "synthetic-secret")
        self.assertEqual(query, "flag=true&quantity=1.2300&symbol=%CE%94USDT&timestamp=123456"
                         "&signature=803bf56f2112b5abbad04c97ced5a2ba4a2d144dea07d74395b5a866029842b5")

    def test_rest01_invalid_signature_values(self):
        for value in (1.1, None, [], Decimal("NaN"), Decimal("Infinity")):
            with self.subTest(value=value):
                self.assert_reason("INVALID_PARAMETERS", self.api.sign_rest, {"a": value}, "secret")
        self.assert_reason("INVALID_PARAMETERS", self.api.sign_rest, {"signature": "x"}, "secret")
        self.assert_reason("INVALID_PARAMETERS", self.api.sign_rest, [("a", 1), ("a", 2)], "secret")

    def test_rest01_explicit_time_units_without_truncation(self):
        self.assertEqual(self.api.time_ms(1234000, "us"), 1234)
        self.assertEqual(self.api.time_ms(1234, "ms"), 1234)
        for value, unit in ((True, "ms"), (-1, "ms"), (1234001, "us"), (1234, "seconds")):
            self.assert_reason("INVALID_TIME", self.api.time_ms, value, unit)

    def test_rest02_invalid_host_schemes_ports_and_userinfo(self):
        for url in ("https://api.binance.com/api/v3/", "https://testnet.binance.vision/api/v3/",
                    "http://demo-api.binance.com/api/v3/", "https://evil@demo-api.binance.com/api/v3/",
                    "https://demo-api.binance.com:9443/api/v3/", "https://demo-api.binance.com/api/v3/#x",
                    "https://demo-api.binance.com.evil/api/v3/", "https://demo-api.binance.com/api/v3/?x=1",
                    "https://@demo-api.binance.com/api/v3/", "https://demo-api.binance.com:/api/v3/",
                    "https://demo-api.binance.com/api/v3/#", "https://demo-api.binance.com/api/v3/?",
                    "https://demo-api.binance.com/api/v3/\n"):
            with self.subTest(url=url):
                self.assert_reason("FORBIDDEN_ENDPOINT", self.api.DemoHttpTransport, base_url=url)

    def test_rest02_dispatch_rejects_mutations_and_path_injection_before_socket(self):
        transport = self.api.DemoHttpTransport()
        with patch.object(self.api.http.client, "HTTPSConnection") as socket:
            for method, path in (("POST", "/api/v3/order"), ("DELETE", "/api/v3/order"),
                                 ("POST", "/api/v3/order/test"), ("GET", "/api/v3/order/../account"),
                                 ("GET", "https://api.binance.com/api/v3/account"),
                                 ("GET", "/api/v3/account?redirect=evil"), ("GET", "/sapi/v1/account"),
                                 ("GET", None), ("GET", [])):
                self.assert_reason("FORBIDDEN_ENDPOINT", transport.request, method, path, "", {}, 15)
            socket.assert_not_called()

    def test_rest02_transport_uses_verified_tls_and_exact_query(self):
        class Response:
            status = 200
            def read(self, maximum):
                return b"{}"
            def getheaders(self):
                return []
        with patch.object(self.api.http.client, "HTTPSConnection") as factory:
            factory.return_value.getresponse.return_value = Response()
            transport = self.api.DemoHttpTransport()
            result = transport.request("GET", "/api/v3/account", "a=%CE%94&signature=abc",
                                       {"X-MBX-APIKEY": "fake"}, 15)
            self.assertEqual(result.status, 200)
            self.assertEqual(factory.call_args.args[0], "demo-api.binance.com")
            context = factory.call_args.kwargs["context"]
            self.assertTrue(context.check_hostname)
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertEqual(factory.return_value.request.call_args.args[:2],
                             ("GET", "/api/v3/account?a=%CE%94&signature=abc"))
            transport.close()
            factory.return_value.close.assert_called_once()

    def test_rest02_redirect_is_not_followed(self):
        self.reply({}, 302, {"Location": "https://api.binance.com/api/v3/account"})
        self.assert_reason("REDIRECT_BLOCKED", self.adapter.read_account)
        self.assertEqual(len(self.calls), 1)

    def test_rest02_response_size_limit(self):
        with patch.object(self.api.http.client, "HTTPSConnection") as factory:
            response = factory.return_value.getresponse.return_value
            response.status = 200
            response.read.return_value = b"x" * (8 * 1024 * 1024 + 1)
            response.read1.side_effect = [b"x" * (8 * 1024 * 1024 + 1), b""]
            response.getheaders.return_value = []
            self.assert_reason("RESPONSE_TOO_LARGE", self.api.DemoHttpTransport().request,
                               "GET", "/api/v3/time", "", {}, 15)

    def test_rest03_clock_midpoint_offset_and_expiry(self):
        self.clock.observe(self.time.wall + 150, self.time.wall, self.time.wall + 100,
                           self.time.mono, self.time.mono + 100_000_000)
        self.time.advance(100)
        self.assertEqual(self.clock.timestamp_ms(), self.time.wall + 100)
        self.time.advance(60_001)
        self.assert_reason("CLOCK_STALE", self.clock.timestamp_ms)

    def test_rest03_clock_rejects_bad_sample_and_invalidates_old(self):
        for rtt, wall_delta in ((600, 600), (1100, 1100), (-1, -1), (10, 600)):
            self.clock.observe(self.time.wall, self.time.wall, self.time.wall, self.time.mono, self.time.mono)
            self.assert_reason("CLOCK_QUALITY", self.clock.observe, self.time.wall,
                               self.time.wall, self.time.wall + wall_delta,
                               self.time.mono, self.time.mono + rtt * 1_000_000)
            self.assert_reason("CLOCK_UNSYNCED", self.clock.timestamp_ms)

    def test_rest03_wall_jump_blocks_signed_request(self):
        self.time.wall += 1000
        self.assert_reason("CLOCK_JUMP", self.adapter.read_account)
        self.assertFalse(self.calls)

    def test_rest03_initial_clock_sync_before_signed_read(self):
        self.clock = self.api.DemoClock(wall_ms=lambda: self.time.wall, monotonic_ns=lambda: self.time.mono)
        adapter = self.make_adapter()
        self.reply({"serverTime": self.time.wall + 5})
        self.reply(account())
        adapter.read_account()
        self.assertEqual([c[1] for c in self.calls], ["/api/v3/time", "/api/v3/account"])
        self.assertFalse(self.calls[0][3])
        self.assertEqual(self.calls[1][3]["X-MBX-APIKEY"], "synthetic-key")

    def test_rest03_timestamp_error_resyncs_once_only(self):
        self.reply({"code": -1021, "msg": "skew"}, 400)
        self.reply({"serverTime": self.time.wall + 15})
        self.reply(account())
        self.adapter.read_account()
        self.assertEqual([c[1] for c in self.calls], ["/api/v3/account", "/api/v3/time", "/api/v3/account"])
        self.assertNotEqual(self.calls[0][2], self.calls[2][2])

    def test_rest03_second_timestamp_error_is_blocked(self):
        self.reply({"code": -1021}, 400)
        self.reply({"serverTime": self.time.wall + 15})
        self.reply({"code": -1021}, 400)
        self.assert_reason("TIMESTAMP_REJECTED", self.adapter.read_account)
        self.assertEqual(len(self.calls), 3)

    def test_rest04_account_precision_zeros_and_actual_receipt(self):
        self.reply(account())
        result = self.adapter.read_account()
        self.assertEqual(result["data"]["balances"][0]["free"], Decimal("1000.123456789012345678"))
        self.assertEqual(result["data"]["balances"][1]["free"], Decimal("0"))
        self.assertEqual(result["received_at_ms"], self.time.wall)
        self.assertEqual(result["data"]["update_time_ms"], 1000)
        self.assertFalse(result["exchange_reconciled"])
        self.assertEqual(result["state"], "ACCOUNT_UNRECONCILED")
        self.assertEqual(parse_qs(self.calls[0][2])["omitZeroBalances"], ["false"])
        self.assertEqual(result["credential_generation"], "g1")
        self.assertEqual(len(result["source_sha256"]), 64)

    def test_rest04_missing_permissions_and_assets_are_unknown(self):
        data = account(balances=[])
        del data["canTrade"]
        del data["permissions"]
        del data["uid"]
        self.reply(data)
        result = self.adapter.read_account()
        self.assertEqual(result["data"]["balances"], [])
        self.assertIsNone(result["data"]["can_trade"])
        self.assertIsNone(result["data"]["account_identity"])
        caps = self.adapter.capabilities()
        self.assertEqual(caps["ACCOUNT_SPOT_ALLOWED"]["state"], "UNVERIFIED")
        self.assertEqual(caps["USER_DATA_READ"]["state"], "SUPPORTED")

    def test_rest04_cantrade_never_grants_key_trade_or_send(self):
        for can_trade, state in ((False, "DENIED"), (True, "SUPPORTED")):
            self.reply(account(canTrade=can_trade))
            self.adapter.read_account()
            caps = self.adapter.capabilities()
            self.assertEqual(caps["ACCOUNT_SPOT_ALLOWED"]["state"], state)
            self.assertEqual(caps["KEY_TRADE_VALIDATED"]["state"], "UNVERIFIED")
            self.assertEqual(caps["ORDER_SEND"]["state"], "DENIED")
            self.assertEqual(caps["PRIVATE_EVENTS"]["state"], "UNVERIFIED")

    def test_rest04_uid_token_same_account_across_key_generation(self):
        self.reply(account())
        first = self.adapter.read_account()["data"]["account_identity"]
        self.reply(account())
        new = self.make_adapter(credential_generation="g2")
        self.assertEqual(first, new.read_account()["data"]["account_identity"])
        self.assertEqual(len(first), 64)
        self.assertNotIn("987654321", first)
        self.assertEqual(new.capabilities()["USER_DATA_READ"]["credential_generation"], "g2")

    def test_rest04_duplicate_negative_float_and_future_balances_fail_closed(self):
        cases = [account(balances=[{"asset": "BTC", "free": "1", "locked": "0"}] * 2),
                 account(balances=[{"asset": "BTC", "free": "-1", "locked": "0"}]),
                 account(balances=[{"asset": "BTC", "free": 1.0, "locked": "0"}]),
                 account(canTrade="true"), account(accountType="MARGIN"),
                 account(updateTime=self.time.wall + 9999)]
        for data in cases:
            with self.subTest(data=data):
                self.reply(data)
                self.assert_reason("SCHEMA_INVALID", self.adapter.read_account)
                self.assertNotEqual(self.adapter.capabilities()["USER_DATA_READ"]["state"], "SUPPORTED")

    def test_rest05_open_order_preserves_ids_and_quantities(self):
        self.reply([order()])
        result = self.adapter.read_open_orders()
        actual = result["data"][0]
        self.assertEqual(actual["client_order_id"], "foreign-manual")
        self.assertEqual(actual["order_list_id"], -1)
        self.assertEqual(actual["executed_quantity"], Decimal("1"))
        self.assertEqual(actual["cumulative_quote_quantity"], Decimal("100"))
        self.assertEqual(actual["ownership"], "UNATTRIBUTED")
        self.assertNotIn("symbol", parse_qs(self.calls[0][2]))

    def test_rest05_order_lookup_identity_mismatch_rejected(self):
        self.reply(order(orderId=22))
        self.assert_reason("SCHEMA_INVALID", self.adapter.read_order, "BTCUSDT", 21)

    def test_rest05_unknown_order_enum_not_invented_fill(self):
        for changes in ({"status": "FUTURE_ENUM"}, {"type": "FUTURE_TYPE"}, {"side": "LONG"},
                        {"executedQty": "3"}, {"price": "NaN"}):
            self.reply([order(**changes)])
            self.assert_reason("SCHEMA_INVALID", self.adapter.read_open_orders)

    def test_rest05_fill_commission_and_ownership(self):
        self.reply([trade()])
        result = self.adapter.recover_fills("BTCUSDT", from_id=0)
        value = result["data"][0]
        self.assertEqual(value["commission_quantity"], Decimal("0.001"))
        self.assertEqual(value["commission_asset"], "BNB")
        self.assertEqual(value["side"], "BUY")
        self.assertEqual(value["ownership"], "UNATTRIBUTED")
        self.assertFalse(result["exchange_reconciled"])

    def test_rest05_fill_malformed_bool_money_foreign_symbol_rejected(self):
        for changes in ({"isBuyer": 1}, {"commission": "-1"}, {"symbol": "ETHUSDT"},
                        {"qty": "0"}, {"quoteQty": 100.0}, {"time": self.time.wall + 9999}):
            self.reply([trade(**changes)])
            self.assert_reason("SCHEMA_INVALID", self.adapter.recover_fills, "BTCUSDT", from_id=0)

    def test_rest06_paginated_sorted_duplicate_economic_ids(self):
        self.reply([trade(2), trade(1), trade(1)])
        self.reply([trade(3)])
        result = self.adapter.recover_fills("BTCUSDT", from_id=0, page_size=3)
        self.assertEqual([r["trade_id"] for r in result["data"]], [1, 2, 3])
        self.assertTrue(result["tail_exhausted"])
        self.assertEqual(result["next_from_id"], 4)
        self.assertEqual(parse_qs(self.calls[1][2])["fromId"], ["3"])
        self.assertEqual(result["coverage"], "QUERIED_SYMBOL_CURSOR_ONLY")

    def test_rest06_conflicting_trade_identity_rejected(self):
        self.reply([trade(1), trade(1, commission="0.1")])
        self.assert_reason("FILL_CONFLICT", self.adapter.recover_fills, "BTCUSDT", from_id=0)

    def test_rest06_nonprogressing_or_ignored_cursor_rejected(self):
        self.reply([trade(1), trade(2)])
        self.reply([trade(1), trade(2)])
        self.assert_reason("PAGINATION_INVALID", self.adapter.recover_fills, "BTCUSDT", from_id=0, page_size=2)

    def test_rest06_max_pages_explicit_partial_and_resume(self):
        self.reply([trade(1), trade(2)])
        result = self.adapter.recover_fills("BTCUSDT", from_id=0, page_size=2, max_pages=1)
        self.assertFalse(result["tail_exhausted"])
        self.assertEqual(result["completeness"], "PARTIAL")
        self.assertEqual(result["next_from_id"], 3)

    def test_rest06_no_start_id_or_invalid_page_size(self):
        for changes in ({"from_id": -1}, {"from_id": True}, {"from_id": 0, "page_size": 1001},
                        {"from_id": 0, "max_pages": 0}):
            self.assert_reason("INVALID_PARAMETERS", self.adapter.recover_fills, "BTCUSDT", **changes)
        self.assertFalse(self.calls)

    def test_rest07_transient_retry_fresh_signature_three_attempts(self):
        self.replies.append(TimeoutError("synthetic-secret leaked-url"))
        self.reply({}, 503)
        self.reply(account())
        self.adapter.read_account()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(set(c[2] for c in self.calls)), 3)
        self.assertTrue(all(c[4] <= 15 for c in self.calls))

    def test_rest07_attempt_exhaustion_not_empty_account(self):
        for _ in range(3):
            self.reply({}, 503)
        self.assert_reason("READ_UNAVAILABLE", self.adapter.read_account)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.adapter.capabilities()["USER_DATA_READ"]["state"], "UNAVAILABLE")

    def test_rest07_rate_limit_respects_retry_after_no_immediate_retry(self):
        self.reply({"code": -1003}, 429, {"Retry-After": "2"})
        self.assert_reason("RATE_LIMITED", self.adapter.read_account)
        self.assert_reason("RATE_LIMITED", self.adapter.read_account)
        self.assertEqual(len(self.calls), 1)
        self.time.advance(2001)
        self.reply(account())
        self.adapter.read_account()
        self.assertEqual(len(self.calls), 2)

    def test_rest07_missing_retry_after_conservative_block(self):
        self.reply({}, 429)
        self.assert_reason("RATE_LIMITED", self.adapter.read_account)
        self.time.advance(59_000)
        self.assert_reason("RATE_LIMITED", self.adapter.read_account)
        self.assertEqual(len(self.calls), 1)

    def test_rest07_ban_and_auth_error_require_new_generation(self):
        for status, code, reason in ((418, -1003, "IP_BANNED"), (401, -2015, "AUTH_DENIED"),
                                     (400, -1022, "SIGNATURE_REJECTED")):
            adapter = self.make_adapter()
            self.reply({"code": code, "msg": "synthetic-key synthetic-secret"}, status)
            self.assert_reason(reason, adapter.read_account)
            count = len(self.calls)
            self.assert_reason(reason, adapter.read_account)
            self.assertEqual(len(self.calls), count)

    def test_rest07_shared_weight_and_local_budget_prevent_more_requests(self):
        self.reply(account(), headers={"X-MBX-USED-WEIGHT-1M": "1199"})
        self.adapter.read_account()
        self.assert_reason("LOCAL_RATE_BUDGET", self.adapter.read_account)
        self.assertEqual(len(self.calls), 1)

    def test_rest07_disabled_adapter_zero_calls(self):
        adapter = self.make_adapter(enabled=False)
        for method in (adapter.ping, adapter.sync_clock, adapter.read_account, adapter.read_open_orders):
            self.assert_reason("ADAPTER_DISABLED", method)
        self.assertFalse(self.calls)

    def test_rest07_capabilities_expire_on_monotonic_age(self):
        self.reply(account())
        self.adapter.read_account()
        self.time.advance(30_001)
        self.assertEqual(self.adapter.capabilities()["USER_DATA_READ"]["state"], "UNVERIFIED")
        self.assertEqual(self.adapter.capabilities()["ORDER_SEND"]["state"], "DENIED")

    def test_rest07_invalid_json_nan_duplicate_fields_fail_closed(self):
        for data in (b'{"balances":NaN}', b'{"balances":[],"balances":[]}', b'not-json'):
            self.reply(data)
            self.assert_reason("SCHEMA_INVALID", self.adapter.read_account)

    def test_rest08_private_diagnostics_and_repr_redacted(self):
        self.assertNotIn("synthetic-key", repr(self.adapter))
        self.assertNotIn("synthetic-secret", repr(self.adapter))
        response = self.api.HttpResponse(400, b"synthetic-secret", {"X": "synthetic-key"})
        self.assertNotIn("synthetic-secret", repr(response))
        self.assertNotIn("synthetic-key", repr(response))
        self.replies.append(RuntimeError("synthetic-key synthetic-secret signature=BAD uid=987654321"))
        error = self.assert_reason("TRANSPORT_FAILURE", self.adapter.read_account)
        self.assertIsNone(error.__cause__)
        for value in ("synthetic-key", "synthetic-secret", "987654321", "BAD"):
            self.assertNotIn(value, str(error))

    def test_rest08_uid_and_arbitrary_payload_fields_not_in_evidence(self):
        self.reply(account(debug="synthetic-secret", apiKey="synthetic-key"))
        result = self.adapter.read_account()
        for value in ("synthetic-key", "synthetic-secret", "987654321"):
            self.assertNotIn(value, str(result))
        self.assertNotIn("uid", result["data"])

    def test_rest08_whitespace_pem_and_missing_identity_rejected(self):
        for key, secret in (("", "secret"), ("key ", "secret"), ("key", "secret\n"),
                            ("key", "-----BEGIN PRIVATE KEY-----"), ("key", "é")):
            self.credentials = DemoCredentials(key, secret)
            self.assert_reason("INVALID_CREDENTIALS", self.make_adapter)
        self.credentials = DemoCredentials("synthetic-key", "synthetic-secret")
        self.assert_reason("INVALID_IDENTITY", self.make_adapter, identity_key=b"short")
        self.assert_reason("INVALID_IDENTITY", self.make_adapter, episode_id="")
        self.assertFalse(self.calls)

    def test_rest04_noncanonical_money_strings_rejected(self):
        for amount in (" 1", "1 ", "1_000", "1e2", "+1", "١"):
            self.reply(account(balances=[{"asset": "BTC", "free": amount, "locked": "0"}]))
            self.assert_reason("SCHEMA_INVALID", self.adapter.read_account)

    def test_rest07_order_and_fill_failure_invalidate_account_evidence(self):
        for operation, data in ((self.adapter.read_open_orders, [order(status="UNKNOWN")]),
                                (lambda: self.adapter.recover_fills("BTCUSDT", from_id=0), [trade(qty="NaN")])):
            self.reply(account())
            self.adapter.read_account()
            self.reply(data)
            self.assert_reason("SCHEMA_INVALID", operation)
            self.assertEqual(self.adapter.capabilities()["USER_DATA_READ"]["state"], "UNAVAILABLE")

    def test_rest07_order_transport_failure_invalidate_account_evidence(self):
        self.reply(account())
        self.adapter.read_account()
        self.reply({}, 400)
        self.assert_reason("API_REJECTED", self.adapter.read_order, "BTCUSDT", 21)
        self.assertEqual(self.adapter.capabilities()["USER_DATA_READ"]["state"], "UNAVAILABLE")

    def test_rest03_failed_sync_invalidates_previous_sample(self):
        self.reply({"serverTime": "bad"})
        self.assert_reason("SCHEMA_INVALID", self.adapter.sync_clock)
        self.assert_reason("CLOCK_UNSYNCED", self.clock.timestamp_ms)

    def test_rest03_clock_jump_invalidates_capability(self):
        self.reply(account())
        self.adapter.read_account()
        self.time.wall += 1000
        self.assertEqual(self.adapter.capabilities()["USER_DATA_READ"]["state"], "UNAVAILABLE")

    def test_rest07_rolling_budget_retains_recent_reservations(self):
        self.time.advance(59_000)
        self.reply(account())
        self.adapter.read_account()
        for _ in range(14):
            self.reply([])
            self.adapter.read_open_orders()  # 20 + 14 * 80 = 1140.
        self.time.advance(1100)
        self.reply([])
        self.assert_reason("LOCAL_RATE_BUDGET", self.adapter.read_open_orders)
        self.assertEqual(len(self.calls), 15)

    def test_rest07_shared_weight_floor_lives_for_full_receipt_minute(self):
        self.time.advance(59_000)
        self.reply(account(), headers={"X-MBX-USED-WEIGHT-1M": "1199"})
        self.adapter.read_account()
        self.time.advance(1100)
        self.assert_reason("LOCAL_RATE_BUDGET", self.adapter.read_account)
        self.assertEqual(len(self.calls), 1)

    def test_rest07_deadline_response_cannot_establish_success(self):
        class LateTransport:
            def request(inner, *args):
                self.time.advance(15_001)
                return self.api.HttpResponse(200, json.dumps(account()).encode())
        self.assert_reason("READ_UNAVAILABLE", self.make_adapter(transport=LateTransport()).read_account)

    def test_rest08_unknown_dependency_exception_is_redacted(self):
        self.replies.append(self.api.AdapterError("synthetic-secret"))
        error = self.assert_reason("TRANSPORT_FAILURE", self.adapter.read_account)
        self.assertNotIn("synthetic-secret", str(error))

    def test_rest08_malformed_response_headers_are_schema_error(self):
        self.replies.append((200, account(), {"Retry-After": 123}))
        self.assert_reason("SCHEMA_INVALID", self.adapter.read_account)

    def test_rest03_uncertainty_preserves_precision_with_ambient_decimal_context(self):
        self.clock.observe(self.time.wall, self.time.wall, self.time.wall,
                           self.time.mono, self.time.mono + 123_456)
        with localcontext() as context:
            context.prec = 2
            self.assertEqual(self.clock.uncertainty_ms, Decimal("0.061728"))

    def test_rest02_persistent_connection_does_not_reuse_private_headers(self):
        class Response:
            status = 200
            def read(self, maximum):
                return b"{}"
            def getheaders(self):
                return []
        with patch.object(self.api.http.client, "HTTPSConnection") as factory:
            factory.return_value.getresponse.return_value = Response()
            transport = self.api.DemoHttpTransport()
            transport.request("GET", "/api/v3/account", "signature=abc", {"X-MBX-APIKEY": "fake"}, 15)
            transport.request("GET", "/api/v3/ping", "", {}, 15)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(factory.return_value.request.call_args.kwargs["headers"], {})
            factory.return_value.close.assert_not_called()
            transport.close()
            transport.close()
            factory.return_value.close.assert_called_once()

    def test_rest02_transport_error_discards_connection_and_redacts_cause(self):
        with patch.object(self.api.http.client, "HTTPSConnection") as factory:
            factory.return_value.connect.side_effect = OSError("synthetic-secret")
            transport = self.api.DemoHttpTransport()
            error = self.assert_reason("TRANSPORT_UNAVAILABLE", transport.request,
                                       "GET", "/api/v3/time", "", {}, 15)
            self.assertNotIn("synthetic-secret", str(error))
            factory.return_value.close.assert_called_once()
            factory.return_value.connect.side_effect = None
            response = factory.return_value.getresponse.return_value
            response.status = 302
            self.assert_reason("REDIRECT_BLOCKED", transport.request, "GET", "/api/v3/time", "", {}, 15)
            self.assertEqual(factory.call_count, 2)

    def test_rest03_slow_tls_sample_discarded_then_warm_sample_passes(self):
        durations = iter([600, 10])
        outer = self
        class TimeTransport:
            def request(inner, *args):
                duration = next(durations)
                server = outer.time.wall + duration // 2
                outer.time.advance(duration)
                return outer.api.HttpResponse(200, json.dumps({"serverTime": server}).encode())
        adapter = self.make_adapter(transport=TimeTransport())
        result = adapter.sync_clock()
        self.assertEqual(result["clock_uncertainty_ms"], Decimal("5"))
        self.assertEqual(self.clock.timestamp_ms(), self.time.wall)

    def test_rest03_three_bad_samples_cannot_establish_clock(self):
        outer = self
        class SlowTransport:
            calls = 0
            def request(inner, *args):
                inner.calls += 1
                server = outer.time.wall + 300
                outer.time.advance(600)
                return outer.api.HttpResponse(200, json.dumps({"serverTime": server}).encode())
        transport = SlowTransport()
        self.assert_reason("CLOCK_QUALITY", self.make_adapter(transport=transport).sync_clock)
        self.assertEqual(transport.calls, 3)
        self.assert_reason("CLOCK_UNSYNCED", self.clock.timestamp_ms)


if __name__ == "__main__":
    unittest.main()
