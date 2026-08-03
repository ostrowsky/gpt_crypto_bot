from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import config


SCHEMA_VERSION = 1
POLICY_VERSION = "live-config-v1"
POLICY_PREFIXES = (
    "AGENT_",
    "COOLDOWN_",
    "OPEN_SIGNAL_",
    "REGIME_START_",
    "SIGNAL_QUALITY_",
    "TOP_GAINER_",
    "UNIFIED_PORTFOLIO_",
)
POLICY_NAMES = {"MAX_OPEN_POSITIONS"}
SENSITIVE_PARTS = {"API", "CHAT", "KEY", "SECRET", "TOKEN"}


def stable_hash(value: Any) -> str:
    raw = json.dumps(_normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def current_policy_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in dir(config):
        if not (name in POLICY_NAMES or name.startswith(POLICY_PREFIXES)):
            continue
        if any(part in name for part in SENSITIVE_PARTS):
            continue
        value = getattr(config, name)
        normalized = _normalize(value)
        if normalized is not None or value is None:
            out[name] = normalized
    return dict(sorted(out.items()))


def build_provenance(
    *,
    builder: str,
    research_config: Any,
    input_paths: Iterable[Path] = (),
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(timezone.utc)
    policy = current_policy_snapshot()
    return {
        "schema_version": SCHEMA_VERSION,
        "builder": builder,
        "generated_at_utc": generated_at.isoformat().replace("+00:00", "Z"),
        "source_policy_version": POLICY_VERSION,
        "source_policy_hash": stable_hash(policy),
        "source_policy_config_count": len(policy),
        "research_config_hash": stable_hash(research_config),
        "input_watermarks": _input_watermarks(input_paths),
    }


def artifact_freshness(
    payload: dict[str, Any],
    *,
    expected_builder: str,
    expected_research_config: Any,
    max_age_hours: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    reasons: list[str] = []
    if not provenance:
        reasons.append("missing_provenance")
    if str(provenance.get("builder") or "") != expected_builder:
        reasons.append("builder_mismatch")
    expected_policy_hash = stable_hash(current_policy_snapshot())
    if str(provenance.get("source_policy_version") or "") != POLICY_VERSION:
        reasons.append("policy_version_mismatch")
    if str(provenance.get("source_policy_hash") or "") != expected_policy_hash:
        reasons.append("source_policy_hash_mismatch")
    expected_config_hash = stable_hash(expected_research_config)
    if str(provenance.get("research_config_hash") or "") != expected_config_hash:
        reasons.append("research_config_hash_mismatch")
    generated = _parse_utc(provenance.get("generated_at_utc") or payload.get("generated_at_utc"))
    age_hours: float | None = None
    if generated is None:
        reasons.append("missing_generated_at")
    else:
        age_hours = (now - generated).total_seconds() / 3600.0
        if age_hours < 0:
            reasons.append("generated_in_future")
        elif age_hours > max_age_hours:
            reasons.append("age_budget_exceeded")
    return {
        "status": "fresh" if not reasons else "stale",
        "reasons": reasons,
        "age_hours": round(age_hours, 3) if age_hours is not None else None,
        "max_age_hours": max_age_hours,
        "expected_builder": expected_builder,
        "expected_source_policy_hash": expected_policy_hash,
        "expected_research_config_hash": expected_config_hash,
    }


def latest_path(directory: Path, pattern: str) -> Path | None:
    paths = sorted(directory.glob(pattern))
    return paths[-1] if paths else None


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        rows = [_normalize(item) for item in value]
        return sorted(rows, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True)) if isinstance(value, (set, frozenset)) else rows
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _input_watermarks(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out = []
    for raw in paths:
        path = Path(raw)
        row: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            stat = path.stat()
            row.update({
                "size": stat.st_size,
                "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            })
        out.append(row)
    return out


def _parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
