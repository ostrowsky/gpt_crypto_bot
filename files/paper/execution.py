from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class FillResult:
    avg_price: float
    filled_qty: float
    quote_spent: float
    slippage_bps: float
    fee_quote: float

class BookWalkSimulator:
    """Simple market-fill simulator consuming L2 levels."""

    def __init__(self, fee_bps: float = 7.5) -> None:
        self.fee_bps = float(fee_bps)

    def fill_market_buy_quote(self, asks: List[Tuple[float,float]], quote_amount: float, ref_price: Optional[float]=None) -> Optional[FillResult]:
        if not asks or quote_amount <= 0:
            return None
        remaining = quote_amount
        qty = 0.0
        spent = 0.0
        for price, avail in asks:
            if price <= 0 or avail <= 0:
                continue
            max_quote = price * avail
            take_quote = min(remaining, max_quote)
            take_qty = take_quote / price
            qty += take_qty
            spent += take_quote
            remaining -= take_quote
            if remaining <= 1e-12:
                break
        if qty <= 0:
            return None
        avg = spent / qty
        base = ref_price if ref_price and ref_price>0 else asks[0][0]
        slip_bps = ((avg/base) - 1.0) * 10000.0
        fee = spent * (self.fee_bps/10000.0)
        return FillResult(avg_price=avg, filled_qty=qty, quote_spent=spent, slippage_bps=slip_bps, fee_quote=fee)

    def fill_market_sell_qty(self, bids: List[Tuple[float,float]], qty_amount: float, ref_price: Optional[float]=None) -> Optional[FillResult]:
        if not bids or qty_amount <= 0:
            return None
        remaining = qty_amount
        got_quote = 0.0
        sold = 0.0
        for price, avail in bids:
            if price <= 0 or avail <= 0:
                continue
            take_qty = min(remaining, avail)
            got_quote += take_qty * price
            sold += take_qty
            remaining -= take_qty
            if remaining <= 1e-12:
                break
        if sold <= 0:
            return None
        avg = got_quote / sold
        base = ref_price if ref_price and ref_price>0 else bids[0][0]
        slip_bps = (1.0 - (avg/base)) * 10000.0
        fee = got_quote * (self.fee_bps/10000.0)
        return FillResult(avg_price=avg, filled_qty=sold, quote_spent=got_quote, slippage_bps=slip_bps, fee_quote=fee)
