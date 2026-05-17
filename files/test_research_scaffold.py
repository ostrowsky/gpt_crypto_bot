import asyncio
import unittest

from features.microstructure import book_slope, microprice, orderbook_imbalance
from paper.execution import BookWalkSimulator
from runtime.price_tracker import PriceTracker
from ws.binance_stream import BinanceStream


class TestMicrostructureFeatures(unittest.TestCase):
    def test_orderbook_imbalance_microprice_and_slope(self) -> None:
        bids = [(100.0, 3.0), (99.5, 2.0)]
        asks = [(100.5, 1.0), (101.0, 1.0)]

        self.assertAlmostEqual(orderbook_imbalance(bids, asks), 3 / 7)
        self.assertAlmostEqual(microprice(bids, asks), 100.375)
        self.assertAlmostEqual(book_slope(bids, side="bid"), 0.005)
        self.assertAlmostEqual(book_slope(asks, side="ask"), 0.004975124378109453)


class TestBookWalkSimulator(unittest.TestCase):
    def test_market_buy_and_sell_walk_levels(self) -> None:
        sim = BookWalkSimulator(fee_bps=10.0)

        buy = sim.fill_market_buy_quote([(100.0, 1.0), (101.0, 1.0)], quote_amount=150.0)
        self.assertIsNotNone(buy)
        assert buy is not None
        self.assertAlmostEqual(buy.filled_qty, 1.495049504950495)
        self.assertAlmostEqual(buy.avg_price, 100.33112582781457)
        self.assertGreater(buy.slippage_bps, 0)
        self.assertAlmostEqual(buy.fee_quote, 0.15)

        sell = sim.fill_market_sell_qty([(99.0, 1.0), (98.0, 1.0)], qty_amount=1.5)
        self.assertIsNotNone(sell)
        assert sell is not None
        self.assertAlmostEqual(sell.quote_spent, 148.0)
        self.assertAlmostEqual(sell.avg_price, 98.66666666666667)
        self.assertGreater(sell.slippage_bps, 0)
        self.assertAlmostEqual(sell.fee_quote, 0.148)


class TestPriceTracker(unittest.TestCase):
    def test_impulse_detects_short_window_move(self) -> None:
        tracker = PriceTracker(lookback_sec=180)
        tracker.record(1_000, 100.0)
        tracker.record(60_000, 100.5)
        tracker.record(120_000, 101.2)

        impulse = tracker.impulse(min_ret=0.01)
        self.assertIsNotNone(impulse)
        assert impulse is not None
        self.assertAlmostEqual(impulse[0], 0.012)
        self.assertEqual(impulse[1], 119_000)


class TestBinanceStreamDispatch(unittest.IsolatedAsyncioTestCase):
    async def test_dispatch_routes_closed_klines_and_depth(self) -> None:
        seen = {"klines": [], "books": []}

        async def on_kline(symbol, interval, payload):
            seen["klines"].append((symbol, interval, payload["x"]))

        async def on_book(symbol, payload):
            seen["books"].append((symbol, payload["lastUpdateId"]))

        stream = BinanceStream(
            ["BTCUSDT"],
            kline_intervals=["15m"],
            on_kline=on_kline,
            on_book=on_book,
        )
        await stream._dispatch(
            {
                "stream": "btcusdt@kline_15m",
                "data": {"s": "BTCUSDT", "k": {"i": "15m", "x": True}},
            }
        )
        await stream._dispatch(
            {
                "stream": "btcusdt@depth5@100ms",
                "data": {"s": "BTCUSDT", "lastUpdateId": 42},
            }
        )

        self.assertEqual(seen["klines"], [("BTCUSDT", "15m", True)])
        self.assertEqual(seen["books"], [("BTCUSDT", 42)])


if __name__ == "__main__":
    unittest.main()
