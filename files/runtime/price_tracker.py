from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple
import time

@dataclass
class Tick:
    ts_ms: int
    price: float
    vol: float

class PriceTracker:
    """Lightweight tracker for short-window impulse detection."""
    def __init__(self, lookback_sec: int = 180, maxlen: int = 2000) -> None:
        self.lookback_ms = int(lookback_sec * 1000)
        self.buf: Deque[Tick] = deque(maxlen=maxlen)

    def record(self, ts_ms: int, price: float, vol: float=0.0) -> None:
        self.buf.append(Tick(ts_ms=ts_ms, price=float(price), vol=float(vol)))

    def impulse(self, min_ret: float = 0.01) -> Optional[Tuple[float,int]]:
        """Return (ret, ms) if price moved up by min_ret within lookback."""
        if len(self.buf) < 3:
            return None
        now = self.buf[-1]
        # find oldest within window
        cutoff = now.ts_ms - self.lookback_ms
        old = None
        for t in self.buf:
            if t.ts_ms >= cutoff:
                old = t
                break
        if old is None or old.price <= 0:
            return None
        ret = (now.price/old.price) - 1.0
        if ret >= min_ret:
            return (ret, now.ts_ms - old.ts_ms)
        return None
