from __future__ import annotations

import json
import hashlib
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

import config
from replay_research_early_trend_discriminator import FEATURE_NAMES, feature_vector


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / ".runtime" / "models" / "research_early_trend_discriminator.cbm"
DEFAULT_METADATA = DEFAULT_MODEL.with_suffix(".json")
log = logging.getLogger("research_early_trend_shadow")


class ShadowScorer:
    def __init__(self, model_file: Path = DEFAULT_MODEL, metadata_file: Path = DEFAULT_METADATA) -> None:
        self.model_file = model_file
        self.metadata_file = metadata_file
        self._model: Any | None = None
        self._metadata: dict[str, Any] | None = None
        self._model_mtime_ns = -1
        self._lock = threading.RLock()
        self.last_error = ""

    def annotate(self, record: dict[str, Any]) -> dict[str, Any] | None:
        if not bool(getattr(config, "RESEARCH_EARLY_TREND_SHADOW_ENABLED", True)):
            return None
        if (
            str(record.get("tf") or "") != "15m"
            or not bool(record.get("in_trade_watchlist"))
            or str(record.get("rule_signal") or "none") != "none"
        ):
            return None
        with self._lock:
            try:
                self._reload_if_needed()
                if self._model is None or self._metadata is None:
                    return None
                vector = np.asarray([feature_vector(record)], dtype=np.float32)
                score = float(self._model.predict_proba(vector)[0][1])
                threshold = float(self._metadata["threshold"])
                annotation = {
                    "profile": str(self._metadata.get("profile") or "research_early_trend_catboost_v1"),
                    "score": round(score, 8),
                    "threshold": round(threshold, 8),
                    "candidate": bool(score >= threshold),
                    "model_created_at_utc": self._metadata.get("created_at_utc"),
                    "holdout_end": self._metadata.get("holdout_end"),
                }
                record["early_trend_shadow"] = annotation
                self.last_error = ""
                return annotation
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("early-trend shadow scoring failed: %s", self.last_error)
                return None

    def _reload_if_needed(self) -> None:
        if not self.model_file.exists() or not self.metadata_file.exists():
            self._model = None
            self._metadata = None
            self.last_error = "model_missing"
            return
        mtime_ns = self.model_file.stat().st_mtime_ns
        if self._model is not None and mtime_ns == self._model_mtime_ns:
            return
        from catboost import CatBoostClassifier

        metadata = json.loads(self.metadata_file.read_text(encoding="utf-8"))
        if tuple(metadata.get("feature_names") or ()) != FEATURE_NAMES:
            raise ValueError("shadow model feature schema mismatch")
        expected_hash = str(metadata.get("model_sha256") or "")
        actual_hash = hashlib.sha256(self.model_file.read_bytes()).hexdigest()
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError("shadow model hash mismatch")
        model = CatBoostClassifier()
        model.load_model(str(self.model_file))
        self._model = model
        self._metadata = metadata
        self._model_mtime_ns = mtime_ns


SCORER = ShadowScorer()


def annotate_record(record: dict[str, Any]) -> dict[str, Any] | None:
    return SCORER.annotate(record)
