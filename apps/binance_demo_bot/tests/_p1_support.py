"""Synthetic P1-B protocol fixtures; no real keys/account/network."""

import hashlib
import hmac
import json
import tempfile
from decimal import Decimal
from pathlib import Path

import _support
from binance_demo_bot.adapter import BinanceDemoAdapter, DemoClock, HttpResponse
from binance_demo_bot.isolation import AppPaths, DemoCredentials
from test_adapter import account, order, trade, TimeSource

NOW = 1_700_000_000_000
TOKEN = hmac.new(b"i" * 32, b"BINANCE_SPOT_DEMO:987654321", "sha256").hexdigest()
IDENTITY = dict(application_id="binance_demo_bot", environment="BINANCE_SPOT_DEMO",
                objective_contract_id="daily_net_equity_pnl_v1", account_alias="demo-one",
                credential_generation="g1", account_episode_id="episode-one", portfolio_scope_id="whole-account")


def source_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                    default=lambda x: format(x, "f")).encode()).hexdigest()


def observation(data, endpoint, **changes):
    result = dict(IDENTITY, schema_version=1, data=data, source_endpoint=endpoint, source_method="GET",
                  source_sha256=source_hash(data), received_at_ms=NOW,
                  request_started_at_ms=NOW-10, received_monotonic_ns=10_000_000_000,
                  request_started_monotonic_ns=9_990_000_000, state="ACCOUNT_UNRECONCILED",
                  exchange_reconciled=False, clock_uncertainty_ms=Decimal("5"))
    result.update(changes)
    return result


def normalized_fill(identifier=1, **changes):
    result = dict(symbol="BTCUSDT", trade_id=identifier, order_id=21, order_list_id=-1,
                  price=Decimal("100"), quantity=Decimal("1"), quote_quantity=Decimal("100"),
                  commission_quantity=Decimal("0.001"), commission_asset="BNB",
                  side="BUY", is_maker=False, exchange_event_time_ms=NOW-1000, ownership="UNATTRIBUTED")
    result.update(changes)
    return result


def recovery(fills=None, *, start=0, next_id=None, exhausted=True, **changes):
    fills = [normalized_fill()] if fills is None else fills
    page = observation(fills, "/api/v3/myTrades")
    result = observation(fills, "/api/v3/myTrades")
    result.update(tail_exhausted=exhausted, completeness="SYMBOL_TAIL_EXHAUSTED" if exhausted else "PARTIAL",
                  starting_from_id=start, next_from_id=next_id if next_id is not None else
                  max([start-1] + [r["trade_id"] for r in fills])+1,
                  coverage="QUERIED_SYMBOL_CURSOR_ONLY", page_observations=[page])
    # The explicit symbol is present in the request, even for an empty tail.
    result["symbol"] = fills[0]["symbol"] if fills else "BTCUSDT"
    result.update(changes)
    return result


def execution(**changes):
    result = dict(e="executionReport", E=NOW-900, s="BTCUSDT", c="manual-order",
                  S="BUY", o="LIMIT", q="2", p="100", g=-1, x="TRADE",
                  X="PARTIALLY_FILLED", i=21, l="1", z="1", L="100", n="0.001", N="BNB",
                  T=NOW-1000, t=1, I=10, m=False, O=NOW-2000, Z="100", Y="100")
    result.update(changes)
    return result


def normalized_order(**changes):
    result = dict(symbol="BTCUSDT", order_id=21, order_list_id=-1, client_order_id="manual-order",
                  status="PARTIALLY_FILLED", side="BUY", type="LIMIT", price="100",
                  original_quantity="2", executed_quantity="1", cumulative_quote_quantity="100",
                  time_ms=NOW-2000, update_time_ms=NOW-1000, ownership="UNATTRIBUTED")
    result.update(changes)
    return result


def normalized_event(fill=..., **changes):
    result = dict(kind="ORDER_UPDATE", exchange_event_time_ms=NOW-900,
                  transaction_time_ms=NOW-1000, execution_id=10,
                  order=normalized_order(), fill=normalized_fill() if fill is ... else fill)
    result.update(changes)
    return result


def list_data(**changes):
    result = dict(orderListId=31, contingencyType="OCO", listStatusType="EXEC_STARTED",
                  listOrderStatus="EXECUTING", listClientOrderId="manual-list", transactionTime=NOW-1000,
                  symbol="BTCUSDT", orders=[dict(symbol="BTCUSDT", orderId=21, clientOrderId="manual-order")])
    result.update(changes)
    return result


class SyntheticWs:
    def __init__(self):
        self.sent, self.queue = [], []
        self.connected = False
        self.closed = False
        self.ack = dict(status=200, result={"subscriptionId": 7})
        self.pong = True

    def connect(self):
        self.connected = True

    def send(self, data):
        self.sent.append(data)
        self.queue.append({"id": data["id"], **self.ack})

    def recv(self, timeout_s):
        if not self.queue:
            raise TimeoutError()
        value = self.queue.pop(0)
        if isinstance(value, BaseException):
            raise value
        return json.dumps(value)

    def ping(self, timeout_s):
        return self.pong

    def close(self):
        self.closed = True


class FixtureEnvironment:
    def set_fixture(self):
        fixture_root = Path(__file__).resolve().parents[1] / ".runtime" / "unit-fixtures"
        fixture_root.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=fixture_root)
        self.root = Path(temp.name).resolve()
        self.addCleanup(temp.cleanup)
        self.paths = AppPaths(self.root)
        self.time = TimeSource()
        self.time.wall = NOW
        self.clock = DemoClock(wall_ms=lambda: self.time.wall, monotonic_ns=lambda: self.time.mono)
        self.clock.observe(NOW, NOW, NOW, self.time.mono, self.time.mono)
        self.rest_calls, self.rest_replies = [], []
        outer = self

        class Rest:
            def request(inner, method, path, query, headers, timeout_s):
                outer.rest_calls.append((method, path, query))
                outer.time.advance(1)
                reply = outer.rest_replies.pop(0) if outer.rest_replies else account()
                if isinstance(reply, BaseException):
                    raise reply
                return HttpResponse(200, json.dumps(reply).encode())

        self.adapter = BinanceDemoAdapter(DemoCredentials("synthetic-key", "synthetic-secret"),
            account_alias="demo-one", credential_generation="g1", episode_id="episode-one",
            scope_id="whole-account", identity_key=b"i"*32, transport=Rest(), clock=self.clock)
        self.store = self.evidence.AccountEvidenceStore(self.paths, account_alias="demo-one",
            account_identity=TOKEN, episode_id="episode-one", scope_id="whole-account")
        self.addCleanup(self.close_store)

    def close_store(self):
        if self.store:
            self.store.close()
            self.store = None

    def reopen_store(self, **changes):
        values = dict(account_alias="demo-one", account_identity=TOKEN,
                      episode_id="episode-one", scope_id="whole-account")
        values.update(changes)
        self.store = self.evidence.AccountEvidenceStore(self.paths, **values)
        return self.store

    def assert_reason(self, reason, method, *args, **kwargs):
        with self.assertRaises(Exception) as caught:
            method(*args, **kwargs)
        self.assertEqual(getattr(caught.exception, "reason", None), reason)
        return caught.exception
