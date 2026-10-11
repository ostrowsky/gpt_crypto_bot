"""Synthetic P2 fixtures; public protocol, no private key discovery/network."""

import importlib
import json
import tempfile
from pathlib import Path
from decimal import Decimal

import _support
from _p1_support import IDENTITY, TOKEN, source_hash
from binance_demo_bot.adapter import DemoClock, HttpResponse
from binance_demo_bot.isolation import AppPaths

NOW = 1_700_000_040_000  # Exact UTC minute boundary.


def pair(base="AAA", quote="USDT", **changes):
    result = dict(symbol=base+quote, baseAsset=base, quoteAsset=quote, status="TRADING",
        isSpotTradingAllowed=True, permissionSets=[["SPOT"]], permissions=[],
        orderTypes=["LIMIT", "MARKET"], ocoAllowed=True, otoAllowed=True,
        filters=[dict(filterType="PRICE_FILTER", minPrice="0", maxPrice="0", tickSize="0.01"),
                 dict(filterType="LOT_SIZE", minQty="0.001", maxQty="10000", stepSize="0.001")])
    result.update(changes)
    return result


def metadata(rows=None):
    return dict(timezone="UTC", serverTime=NOW, rateLimits=[], exchangeFilters=[],
                symbols=[pair()] if rows is None else rows)


def book(symbol="AAAUSDT", bid="99", ask="101", **changes):
    value = dict(symbol=symbol, bidPrice=bid, askPrice=ask, bidQty="10", askQty="10")
    value.update(changes)
    return value


def candle(open_ms, close="100", **changes):
    values = [open_ms, "100", "120", "90", close, "10", open_ms+59999,
              "1000", 10, "5", "500", "0"]
    for position, value in changes.items():
        values[int(position)] = value
    return values


def account_snapshot(permissions=None, **changes):
    data = dict(permissions=["SPOT"] if permissions is None else permissions,
                can_trade=True, account_identity=TOKEN, balances=[], update_time_ms=NOW-1)
    result = dict(IDENTITY, schema_version=1, data=data, source_endpoint="/api/v3/account",
                  source_method="GET", source_sha256=source_hash(data), received_at_ms=NOW,
                  request_started_at_ms=NOW-1, received_monotonic_ns=1_000_000_000,
                  clock_uncertainty_ms=Decimal("0"), exchange_reconciled=False)
    result.update(changes)
    return result


class Fixture:
    def setup_market(self):
        self.market = importlib.import_module("binance_demo_bot.market_data")
        self.universe_api = importlib.import_module("binance_demo_bot.universe")
        fixture_root = Path(__file__).resolve().parents[1] / ".runtime" / "unit-fixtures"
        fixture_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=fixture_root)
        self.addCleanup(temporary.cleanup)
        self.paths = AppPaths(Path(temporary.name).resolve())
        self.wall, self.mono = NOW, 1_000_000_000
        self.clock = DemoClock(wall_ms=lambda: self.wall, monotonic_ns=lambda: self.mono)
        self.clock.observe(NOW, NOW, NOW, self.mono, self.mono)
        self.replies, self.calls = [], []
        outer = self

        class Transport:
            def request(inner, method, path, query, headers, timeout_s):
                outer.calls.append((method, path, query, dict(headers), timeout_s))
                value = outer.replies.pop(0)
                if isinstance(value, BaseException):
                    raise value
                return HttpResponse(200, json.dumps(value).encode())

        adapter = importlib.import_module("binance_demo_bot.adapter")
        self.read_client = adapter.DemoReadClient(transport=Transport(), clock=self.clock,
                                                 sleep=lambda seconds: self.advance(round(seconds*1000)))
        self.client = self.market.DemoMarketClient(self.read_client)
        self.store = self.market.MarketSnapshotStore(self.paths, account_alias="demo-one",
                                                    episode_id="episode-one", scope_id="whole-account")
        self.addCleanup(lambda: self.store.close())
        self.universe_service = self.universe_api.UniverseService(self.store, self.client)
        self.service = self.market.MarketDataService(self.store, self.client)

    def advance(self, milliseconds):
        self.wall += milliseconds
        self.mono += milliseconds*1_000_000

    def refresh(self, rows=None, account=...):
        self.replies.append(metadata(rows))
        return self.universe_service.refresh(account_snapshot() if account is ... else account)

    def quotes(self, rows=None, universe=None):
        self.replies.append([book()] if rows is None else rows)
        return self.service.poll_quotes(universe or self.universe_service.current(self.wall))

    def bars(self, rows=None, symbol="AAAUSDT"):
        # Real candle collection needs a fresh validated server clock, including
        # when a synthetic fixture advances past the P1 60-second clock lease.
        from binance_demo_bot.adapter import AdapterError
        try:
            self.clock.timestamp_ms()
        except AdapterError as error:
            if error.reason not in ("CLOCK_UNSYNCED", "CLOCK_STALE"):
                raise
            self.replies.append({"serverTime": self.wall})
        self.replies.append(rows if rows is not None else
            [candle(NOW-180000, "100"), candle(NOW-120000, "101"), candle(NOW-60000, "102")])
        return self.service.collect_candles(symbol)

    def assert_reason(self, reason, method, *args, **kwargs):
        with self.assertRaises(Exception) as caught:
            method(*args, **kwargs)
        self.assertEqual(getattr(caught.exception, "reason", None), reason)
