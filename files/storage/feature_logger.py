from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import threading
import time

class FeatureLogger:
    """Append-only JSONL logger for market snapshots and decisions."""

    def __init__(self, path: str = "feature_events.jsonl") -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        rec = {
            "ts_ms": int(time.time()*1000),
            "event_type": event_type,
            **payload,
        }
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
