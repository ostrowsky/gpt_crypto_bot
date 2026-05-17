from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Any
import aiohttp

OnKline = Callable[[str, str, Dict[str, Any]], Awaitable[None]]   # (symbol, interval, kline)
OnBook  = Callable[[str, Dict[str, Any]], Awaitable[None]]        # (symbol, depth)

@dataclass
class StreamConfig:
    base_url: str = "wss://stream.binance.com:9443/stream"
    reconnect_base_sec: float = 1.0
    reconnect_max_sec: float = 30.0

class BinanceStream:
    """Multiplexed Binance WS stream: klines + partial orderbook."""

    def __init__(
        self,
        symbols: List[str],
        *,
        kline_intervals: List[str],
        depth_levels: int = 5,
        depth_speed: str = "100ms",
        on_kline: Optional[OnKline] = None,
        on_book: Optional[OnBook] = None,
        cfg: Optional[StreamConfig] = None,
    ) -> None:
        self.symbols = [s.lower() for s in symbols]
        self.kline_intervals = kline_intervals
        self.depth_levels = depth_levels
        self.depth_speed = depth_speed
        self.on_kline = on_kline
        self.on_book = on_book
        self.cfg = cfg or StreamConfig()
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except Exception:
                pass

    def _streams(self) -> List[str]:
        streams: List[str] = []
        for sym in self.symbols:
            for iv in self.kline_intervals:
                streams.append(f"{sym}@kline_{iv}")
            streams.append(f"{sym}@depth{self.depth_levels}@{self.depth_speed}")
        return streams

    async def _run(self) -> None:
        backoff = self.cfg.reconnect_base_sec
        while not self._stop.is_set():
            streams = "/".join(self._streams())
            url = f"{self.cfg.base_url}?streams={streams}"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, heartbeat=30) as ws:
                        backoff = self.cfg.reconnect_base_sec
                        async for msg in ws:
                            if self._stop.is_set():
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    payload = json.loads(msg.data)
                                except Exception:
                                    continue
                                await self._dispatch(payload)
                            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                                break
            except Exception:
                pass
            await asyncio.sleep(backoff)
            backoff = min(self.cfg.reconnect_max_sec, backoff * 1.6)

    async def _dispatch(self, payload: Dict[str, Any]) -> None:
        data = payload.get("data") or {}
        stream = payload.get("stream", "")
        if not stream:
            return
        # stream looks like "btcusdt@kline_15m" or "btcusdt@depth5@100ms"
        if "@kline_" in stream:
            if not self.on_kline:
                return
            sym = data.get("s", "")
            k = data.get("k", {})
            interval = k.get("i", "")
            # only closed klines
            if k.get("x") is True:
                await self.on_kline(sym, interval, k)
        elif "@depth" in stream:
            if not self.on_book:
                return
            sym = data.get("s", "")
            await self.on_book(sym, data)
