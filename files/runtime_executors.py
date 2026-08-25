from __future__ import annotations

import asyncio
import atexit
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


CPU_EXECUTOR = ThreadPoolExecutor(
    max_workers=_env_int("CRYPTO_BOT_CPU_WORKERS", max(2, min(6, (os.cpu_count() or 4) // 2))),
    thread_name_prefix="crypto-cpu",
)
TELEGRAM_EXECUTOR = ThreadPoolExecutor(
    max_workers=_env_int("CRYPTO_BOT_TELEGRAM_WORKERS", 8),
    thread_name_prefix="telegram-control",
)
IO_EXECUTOR = ThreadPoolExecutor(
    max_workers=_env_int("CRYPTO_BOT_ASYNCIO_IO_WORKERS", 48),
    thread_name_prefix="asyncio-io",
)
EVIDENCE_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="evidence-io",
)


async def run_cpu(func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(CPU_EXECUTOR, partial(func, *args, **kwargs))


async def run_telegram_io(func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(TELEGRAM_EXECUTOR, partial(func, *args, **kwargs))


async def run_evidence_io(func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run ordered dataset persistence without blocking Telegram's event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(EVIDENCE_EXECUTOR, partial(func, *args, **kwargs))


def install_default_io_executor() -> None:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.set_default_executor(IO_EXECUTOR)


def _shutdown() -> None:
    for executor in (CPU_EXECUTOR, TELEGRAM_EXECUTOR, IO_EXECUTOR):
        executor.shutdown(wait=False, cancel_futures=True)
    EVIDENCE_EXECUTOR.shutdown(wait=True, cancel_futures=False)


atexit.register(_shutdown)
