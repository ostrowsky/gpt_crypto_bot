from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any

@dataclass
class StrategySignal:
    name: str
    stage: str   # 'warn' | 'confirm' | 'exit'
    score: float
    reason: str
    meta: Dict[str, Any]

REGISTRY: Dict[str, Callable[[Dict[str, Any]], Optional[StrategySignal]]] = {}

def register(name: str):
    def deco(fn: Callable[[Dict[str, Any]], Optional[StrategySignal]]):
        REGISTRY[name] = fn
        return fn
    return deco
