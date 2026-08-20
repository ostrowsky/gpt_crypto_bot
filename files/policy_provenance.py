from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import config


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
POLICY_EPOCH_VERSION = "decision-policy-v1"
SENSITIVE_PARTS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CHAT_ID")
POLICY_SOURCE_FILES = (
    "config.py",
    "data_collector.py",
    "indicators.py",
    "ml_signal_model.py",
    "critic_dataset.py",
    "ml_candidate_ranker.py",
    "strategy.py",
    "monitor.py",
    "market_signal_agent.py",
    "unified_portfolio.py",
    "signal_quality_feedback.py",
    "policy_provenance.py",
)
TARGET_LABEL_DEFINITIONS = {
    "ret_3": "percent return from decision-bar close to close of bar T+3",
    "label_3": "ret_3 > 0",
    "ret_5": "percent return from decision-bar close to close of bar T+5",
    "label_5": "ret_5 > 0",
    "ret_10": "percent return from decision-bar close to close of bar T+10",
    "label_10": "ret_10 > 0",
    "trade_exit_pnl": "realized paper-trade percent return at the bot exit decision",
    "trade_exit_reason": "bot exit reason recorded at the exit decision",
    "trade_bars_held": "closed timeframe bars held at the bot exit decision",
    "exit_pnl": "realized paper-trade percent return at the bot exit decision",
    "exit_reason": "bot exit reason recorded at the exit decision",
    "bars_held": "closed timeframe bars held at the bot exit decision",
}


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        rows = [_normalize(item) for item in value]
        return sorted(rows, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def stable_hash(value: Any) -> str:
    raw = json.dumps(_normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_config_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in dir(config):
        if not name.isupper() or any(part in name for part in SENSITIVE_PARTS):
            continue
        value = getattr(config, name)
        if callable(value):
            continue
        out[name] = _normalize(value)
    return dict(sorted(out.items()))


def _watchlist_snapshot() -> list[str]:
    configured = Path(getattr(config, "WATCHLIST_FILE", "watchlist.json"))
    path = configured if configured.is_absolute() else ROOT / configured
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = []
    else:
        raw = list(getattr(config, "DEFAULT_WATCHLIST", ()))
    out: list[str] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else ():
        symbol = str(item or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return sorted(out)


def _source_hashes(paths: Iterable[str] = POLICY_SOURCE_FILES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in paths:
        path = ROOT / name
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return out


@lru_cache(maxsize=1)
def current_policy_manifest() -> dict[str, Any]:
    config_snapshot = _safe_config_snapshot()
    source_hashes = _source_hashes()
    watchlist = _watchlist_snapshot()
    policy_payload = {
        "epoch_version": POLICY_EPOCH_VERSION,
        "config": config_snapshot,
        "source_hashes": source_hashes,
        "watchlist": watchlist,
    }
    policy_hash = stable_hash(policy_payload)
    return {
        "epoch_version": POLICY_EPOCH_VERSION,
        "policy_epoch": f"pe1-{policy_hash[:16]}",
        "policy_hash": policy_hash,
        "config_hash": stable_hash(config_snapshot),
        "watchlist_hash": stable_hash(watchlist),
        "watchlist_count": len(watchlist),
        "source_hashes": source_hashes,
        "config_count": len(config_snapshot),
    }


def utc_iso(value: datetime | None = None) -> str:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def timeframe_delta(tf: str) -> timedelta:
    value = str(tf or "").strip().lower()
    units = {"m": "minutes", "h": "hours", "d": "days"}
    if len(value) < 2 or value[-1] not in units:
        raise ValueError(f"Unsupported timeframe for provenance: {tf!r}")
    count = int(value[:-1])
    return timedelta(**{units[value[-1]]: count})


def feature_cutoff(bar_ts: int, tf: str) -> datetime:
    bar_open = datetime.fromtimestamp(int(bar_ts) / 1000, tz=timezone.utc)
    return bar_open + timeframe_delta(tf)


def build_observation_provenance(
    *,
    bar_ts: int,
    tf: str,
    source: str,
    recorded_at: datetime | None = None,
) -> dict[str, Any]:
    manifest = current_policy_manifest()
    return {
        "schema_version": SCHEMA_VERSION,
        **manifest,
        "source": str(source),
        "feature_time": utc_iso(feature_cutoff(bar_ts, tf)),
        "decision_time": utc_iso(recorded_at),
        "feature_contract": "closed-bar features only; no label or future-bar fields",
    }


def build_label_provenance(
    *,
    label_keys: Iterable[str],
    definition: str | dict[str, str] | None = None,
    label_time: datetime | None = None,
    recorded_at: datetime | None = None,
    source: str,
) -> dict[str, dict[str, Any]]:
    recorded = recorded_at or datetime.now(timezone.utc)
    available = label_time or recorded
    definitions = definition if isinstance(definition, dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for raw_key in label_keys:
        key = str(raw_key)
        out[key] = {
            "schema_version": SCHEMA_VERSION,
            "definition": definitions.get(key) or (definition if isinstance(definition, str) else None) or TARGET_LABEL_DEFINITIONS.get(key) or f"learning label {key}",
            "label_time": utc_iso(available),
            "recorded_at": utc_iso(recorded),
            "source": str(source),
        }
    return out


def forward_label_time(*, bar_ts: int, tf: str, horizon: int) -> datetime:
    # T+N uses the close of the bar whose open is N intervals after the
    # decision bar, so the value is available after N+1 bar intervals.
    return datetime.fromtimestamp(int(bar_ts) / 1000, tz=timezone.utc) + timeframe_delta(tf) * (int(horizon) + 1)


def closed_target_index(t_arr: Any, *, bar_ts: int, bar_ms: int, horizon: int) -> int | None:
    """Return the exact T+N bar only when a later bar proves it is closed."""
    target_open = int(bar_ts) + int(horizon) * int(bar_ms)
    for idx, raw in enumerate(t_arr):
        if int(raw) != target_open:
            continue
        return idx if idx + 1 < len(t_arr) else None
    return None


def attach_label_provenance(
    record: dict[str, Any],
    *,
    label_keys: Iterable[str],
    definition: str | dict[str, str] | None = None,
    label_time: datetime | None = None,
    source: str,
) -> bool:
    current = record.setdefault("label_provenance", {})
    changed = False
    for key, value in build_label_provenance(
        label_keys=label_keys,
        definition=definition,
        label_time=label_time,
        source=source,
    ).items():
        # Provenance is immutable for a label once recorded. Re-running a
        # rewrite cannot make a historical label look newer or differently
        # defined.
        if key not in current:
            current[key] = value
            changed = True
    return changed


def update_decision_provenance(record: dict[str, Any], provenance: dict[str, Any]) -> bool:
    previous = record.get("decision_provenance")
    if previous == provenance:
        return False
    root_observation = record.get("provenance")
    if not isinstance(previous, dict) or not previous:
        # Never retrofit a current decision epoch onto a legacy observation.
        # A rescan is not evidence of when or under which policy the original
        # features were observed.
        if not isinstance(root_observation, dict) or not root_observation:
            return False
        record["decision_provenance"] = provenance
        return True
    if isinstance(previous, dict) and previous:
        identity = ("policy_epoch", "policy_hash", "source")
        if all(previous.get(key) == provenance.get(key) for key in identity):
            # Repeated scans under the same decision policy/source keep the
            # first observation time. This avoids rewriting history on every
            # poll and preserves the causal first-decision boundary.
            return False
        history = record.setdefault("decision_provenance_history", [])
        if previous not in history:
            history.append(previous)
    record["decision_provenance"] = provenance
    return True


def provenance_required() -> bool:
    return bool(getattr(config, "POLICY_PROVENANCE_REQUIRED_FOR_RANKER", True))


def label_provenance_valid(record: dict[str, Any], key: str) -> bool:
    row = (record.get("label_provenance") or {}).get(key) or {}
    definition = str(row.get("definition") or "").strip()
    label_time = parse_utc(row.get("label_time"))
    recorded_at = parse_utc(row.get("recorded_at"))
    return bool(definition and label_time and recorded_at and recorded_at >= label_time)


def observation_provenance_valid(record: dict[str, Any]) -> bool:
    provenance = record.get("provenance") or {}
    decision = record.get("decision_provenance") or {}
    feature_time = parse_utc(provenance.get("feature_time"))
    decision_time = parse_utc(decision.get("decision_time"))
    return bool(
        provenance.get("policy_epoch")
        and provenance.get("policy_hash")
        and provenance.get("feature_contract")
        and decision.get("policy_epoch")
        and decision.get("policy_hash")
        and feature_time
        and decision_time
        and decision_time >= feature_time
    )
