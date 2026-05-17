from __future__ import annotations
from typing import List, Tuple, Optional
import math

# book side: list of (price, qty) sorted best->worse

def _safe_float(x) -> Optional[float]:
    try:
        v=float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None

def orderbook_imbalance(bids: List[Tuple[float,float]], asks: List[Tuple[float,float]], levels: int = 5) -> Optional[float]:
    """(sum_bid_qty - sum_ask_qty)/(sum_bid_qty+sum_ask_qty) in [-1,1]."""
    if not bids or not asks:
        return None
    b=sum(q for _,q in bids[:levels] if q is not None)
    a=sum(q for _,q in asks[:levels] if q is not None)
    denom=b+a
    if denom<=0:
        return None
    return (b-a)/denom

def microprice(bids: List[Tuple[float,float]], asks: List[Tuple[float,float]]) -> Optional[float]:
    """Weighted by opposite side size: (ask*bid_qty + bid*ask_qty)/(bid_qty+ask_qty)."""
    if not bids or not asks:
        return None
    bp,bq=bids[0]
    ap,aq=asks[0]
    denom=bq+aq
    if denom<=0:
        return None
    return (ap*bq + bp*aq)/denom

def book_slope(levels: List[Tuple[float,float]], side: str = "bid") -> Optional[float]:
    """Simple slope proxy: (p0 - plast)/p0 for bids, (plast - p0)/p0 for asks."""
    if not levels or len(levels)<2:
        return None
    p0=levels[0][0]
    plast=levels[-1][0]
    if not p0 or p0<=0:
        return None
    if side.lower().startswith("bid"):
        return (p0 - plast)/p0
    return (plast - p0)/p0
