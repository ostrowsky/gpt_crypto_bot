"""Synthetic fixtures for P0-A; no credentials, API calls or legacy imports."""

import importlib
import sys
import unittest
from decimal import Decimal
from pathlib import Path


# Only the independent app's prospective source package is made importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
D = Decimal
BOUNDARY = 10_000


class FinancialContractCase(unittest.TestCase):
    def setUp(self):
        # Intentionally fails until the SPEC -> TESTS -> CODE implementation step.
        # Delayed import keeps every scenario discoverable in the RED run.
        self.api = importlib.import_module("binance_demo_bot.financial")


def quote(base="AAA", quote_asset="USDT", bid="99", ask="101", **changes):
    value = {
        "base": base, "quote": quote_asset, "bid": bid, "ask": ask,
        "venue": "BINANCE_SPOT_DEMO", "event_time_ms": BOUNDARY - 1_000,
        "received_at_ms": BOUNDARY - 900,
        "response_started_at_ms": BOUNDARY - 950, "source_healthy": True,
    }
    value.update(changes)
    return value


def snapshot(equity="1000", *, boundary=0, status="COMPLETE", **changes):
    value = {
        "boundary_ms": boundary, "equity_usdt": equity, "status": status,
        "episode_id": "synthetic-episode", "scope_id": "synthetic-account",
    }
    value.update(changes)
    return value


def lot(asset, quantity, basis, ownership="SEED"):
    return {"asset": asset, "quantity": quantity, "basis_usdt": basis,
            "ownership": ownership}


def fill(trade="t1", *, side="BUY", quantity="2", price="100",
         quote_quantity="200", fee="0", fee_asset="USDT", **changes):
    value = {
        "application_id": "binance_demo_bot", "environment": "BINANCE_SPOT_DEMO",
        "episode_id": "synthetic-episode", "symbol": "AAAUSDT",
        "trade_id": trade, "order_id": "synthetic-order",
        "client_order_id": "synthetic-client", "base_asset": "AAA",
        "quote_asset": "USDT", "side": side, "quantity": quantity,
        "quote_quantity": quote_quantity, "price": price,
        "commission_asset": fee_asset, "commission_quantity": fee,
        "exchange_event_time_ms": 1_000, "ownership": "APPLICATION",
    }
    value.update(changes)
    return value
