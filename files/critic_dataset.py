from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows fallback
    msvcrt = None

from ml_signal_model import build_runtime_record
import policy_provenance


ROOT = Path(__file__).resolve().parent
LEGACY_CRITIC_FILE = ROOT / "critic_dataset.jsonl"
CRITIC_FILE = ROOT / "critic_dataset_v2.jsonl"
SEQ_FEATURE_NAMES = [
    "close_norm",
    "high_norm",
    "low_norm",
    "open_norm",
    "vol_x",
    "slope",
    "adx",
    "rsi",
    "macd_hist_norm",
    "atr_pct",
]
_FILE_LOCK = threading.RLock()
_pylog = logging.getLogger("critic_dataset")
_logged_candidates: OrderedDict[str, bool] = OrderedDict()
_disk_id_cache: Dict[str, tuple] = {}
_MAX_LOGGED = 100_000
_CROSS_PROCESS_LOCK_TIMEOUT_SEC = max(
    10.0,
    float(os.getenv("GPT_BOT_CRITIC_LOCK_TIMEOUT_SEC", "300")),
)
_CROSS_PROCESS_LOCK_POLL_SEC = 0.05
_REPLACE_RETRY_SEC = 0.10
_REPLACE_TIMEOUT_SEC = max(
    10.0,
    float(os.getenv("GPT_BOT_CRITIC_REPLACE_TIMEOUT_SEC", "120")),
)
if "GPT_BOT_CRITIC_LOCK_TIMEOUT_SEC" not in os.environ:
    _CROSS_PROCESS_LOCK_TIMEOUT_SEC = max(_CROSS_PROCESS_LOCK_TIMEOUT_SEC, 2*_REPLACE_TIMEOUT_SEC+60)


class DatasetIntegrityError(RuntimeError):
    """A required evidence write did not complete and must not be counted."""


class _Enc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _safe(v: object) -> float:
    try:
        f = float(v)
        return 0.0 if f != f else f
    except Exception:
        return 0.0


def _safe_int(v: object) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except Exception:
        return None


def _safe_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _candidate_id(sym: str, tf: str, bar_ts: int) -> str:
    ts_str = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{sym}_{tf}_{ts_str.replace(':', '').replace('-', '')}"


def _parse_utc_iso(raw: object) -> Optional[datetime]:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _mark_logged(record_id: str) -> None:
    if record_id in _logged_candidates:
        return
    if len(_logged_candidates) >= _MAX_LOGGED:
        evict = max(1, _MAX_LOGGED // 10)
        for _ in range(evict):
            _logged_candidates.popitem(last=False)
    _logged_candidates[record_id] = True


def _is_logged(record_id: str) -> bool:
    return record_id in _logged_candidates


def _lock_file_path() -> Path:
    return CRITIC_FILE.with_name(CRITIC_FILE.name + ".lock")


@contextmanager
def _dataset_io_lock():
    with _FILE_LOCK:
        if msvcrt is None:
            yield
            return

        lock_path = _lock_file_path()
        lock_handle = None
        start = time.monotonic()
        while True:
            try:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                lock_handle = lock_path.open("a+b")
                lock_handle.seek(0, os.SEEK_END)
                if lock_handle.tell() == 0:
                    lock_handle.write(b"0")
                    lock_handle.flush()
                lock_handle.seek(0)
                msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if lock_handle is not None:
                    try:
                        lock_handle.close()
                    except Exception:
                        pass
                    lock_handle = None
                if time.monotonic() - start >= _CROSS_PROCESS_LOCK_TIMEOUT_SEC:
                    raise TimeoutError(f"timeout acquiring critic_dataset lock: {lock_path}")
                time.sleep(_CROSS_PROCESS_LOCK_POLL_SEC)

        try:
            yield
        finally:
            try:
                if lock_handle is not None:
                    lock_handle.seek(0)
                    msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                if lock_handle is not None:
                    lock_handle.close()


def _atomic_replace_with_retry(tmp: Path, target: Path) -> None:
    """Replace a dataset snapshot after transient Windows reader handles close."""
    deadline = time.monotonic() + _REPLACE_TIMEOUT_SEC
    while True:
        try:
            tmp.replace(target)
            cached = _disk_id_cache.pop(str(tmp.resolve()), None)
            if cached:
                stat = target.stat()
                _disk_id_cache[str(target.resolve())] = (
                    (stat.st_dev, stat.st_ino), stat.st_size, stat.st_mtime_ns, cached[3],
                )
            else:
                _disk_id_cache.pop(str(target.resolve()), None)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            # Python readers on Windows can temporarily deny rename/delete
            # sharing. Retry through bounded evidence scans while the writer
            # lock prevents another mutation from overtaking this commit.
            time.sleep(_REPLACE_RETRY_SEC)


def _scan_mutations(mutator) -> tuple[bool, bool]:
    changed = False
    had_bad_rows = False
    with CRITIC_FILE.open("r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                had_bad_rows = True
                continue
            if not isinstance(rec, dict):
                had_bad_rows = True
                continue
            changed = bool(mutator(rec)) or changed
    return changed, had_bad_rows


def _write_mutated_stream(mutator, tmp: Path) -> tuple[bool, bool]:
    changed = False
    had_bad_rows = False
    ids = set()
    try:
        with CRITIC_FILE.open("r", encoding="utf-8", errors="ignore") as source, tmp.open(
            "w", encoding="utf-8"
        ) as destination:
            for line in source:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    had_bad_rows = True
                    continue
                if not isinstance(rec, dict):
                    had_bad_rows = True
                    continue
                changed = bool(mutator(rec)) or changed
                if rec.get("id"):
                    ids.add(rec["id"])
                destination.write(json.dumps(rec, ensure_ascii=False, cls=_Enc) + "\n")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    if changed or had_bad_rows:
        stat = tmp.stat()
        _disk_id_cache[str(tmp.resolve())] = ((stat.st_dev, stat.st_ino), stat.st_size, stat.st_mtime_ns, ids)
    return changed, had_bad_rows


def _append(record: Dict[str, Any]) -> bool:
    """Cross-process uniqueness check and append share the same IO barrier."""
    with _dataset_io_lock():
        CRITIC_FILE.parent.mkdir(parents=True, exist_ok=True)
        record_id = record.get("id")
        cache_key = str(CRITIC_FILE.resolve())
        ids = set()
        if record_id and CRITIC_FILE.exists():
            stat = CRITIC_FILE.stat()
            identity = (stat.st_dev, stat.st_ino)
            cached = _disk_id_cache.get(cache_key)
            offset = 0
            if cached and cached[0] == identity and stat.st_size >= cached[1]:
                # Cooperative writers append or atomically replace, never edit
                # a prefix in place. Replacements force a full ID rescan.
                if stat.st_size > cached[1] or stat.st_mtime_ns == cached[2]:
                    offset, ids = cached[1], cached[3].copy()
            with CRITIC_FILE.open("rb") as source:
                source.seek(offset)
                for line in source:
                    if not line.strip():
                        continue
                    # Malformed history is an integrity failure, not permission
                    # to append an ID whose uniqueness cannot be established.
                    existing = json.loads(line)
                    if not isinstance(existing, dict):
                        raise DatasetIntegrityError("non-object dataset row")
                    if existing.get("id"):
                        ids.add(existing["id"])
            _disk_id_cache[cache_key] = (identity, stat.st_size, stat.st_mtime_ns, ids)
            if record_id in ids:
                return False
        with CRITIC_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, cls=_Enc) + "\n")
        if record_id:
            ids.add(record_id)
            stat = CRITIC_FILE.stat()
            _disk_id_cache[cache_key] = ((stat.st_dev, stat.st_ino), stat.st_size, stat.st_mtime_ns, ids)
        else:
            _disk_id_cache.pop(cache_key, None)
        return True


def get_records(record_ids: set[str]) -> Dict[str, Dict[str, Any]]:
    """Resolve multiple critic records with one ordered dataset scan."""
    pending = {str(record_id) for record_id in record_ids if str(record_id)}
    if not pending or not CRITIC_FILE.exists():
        return {}
    found: Dict[str, Dict[str, Any]] = {}
    try:
        with _dataset_io_lock():
            with CRITIC_FILE.open("r", encoding="utf-8", errors="ignore") as source:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    record_id = str(rec.get("id") or "")
                    if record_id in pending:
                        found[record_id] = rec
                        pending.remove(record_id)
                        if not pending:
                            break
    except Exception as e:
        _pylog.warning("critic_dataset get_records error for %d id(s): %s", len(record_ids), e)
    return found


def get_record(record_id: str) -> Optional[Dict[str, Any]]:
    return get_records({record_id}).get(record_id) if record_id else None


def _rewrite_records(mutator, *, strict: bool = False) -> bool:
    if not CRITIC_FILE.exists():
        return False
    try:
        maybe_changed, maybe_bad_rows = _scan_mutations(mutator)
        if not (maybe_changed or maybe_bad_rows):
            return False

        with _dataset_io_lock():
            CRITIC_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = CRITIC_FILE.with_name(
                f"{CRITIC_FILE.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            changed, had_bad_rows = _write_mutated_stream(mutator, tmp)
            if changed or had_bad_rows:
                _atomic_replace_with_retry(tmp, CRITIC_FILE)
            else:
                tmp.unlink(missing_ok=True)
        return bool(changed or had_bad_rows)
    except Exception as e:
        _pylog.warning("critic_dataset rewrite error: %s", e)
        if strict:
            raise DatasetIntegrityError(str(e)) from e
        return False


def _decision_priority(action: str, stage: str) -> int:
    a = str(action or "").lower()
    s = str(stage or "").lower()
    if a == "take":
        return 30
    if a == "blocked":
        return 20
    if a in {"candidate", "snapshot"} or s == "collector":
        return 10
    return 0


def _update_existing_candidate(
    *,
    record_id: str,
    action: str,
    reason_code: str,
    reason: str,
    stage: str,
    candidate_score: float,
    base_score: float,
    score_floor: float,
    forecast_return_pct: float,
    today_change_pct: float,
    ml_proba: Optional[float],
    mtf_soft_penalty: float,
    fresh_priority: bool,
    catchup: bool,
    continuation_profile: bool,
    signal_flags: Optional[Dict[str, bool]],
    near_miss: bool,
    decision_provenance: Dict[str, Any],
    strict: bool = False,
) -> bool:
    changed = False

    def _mutate(rec: Dict[str, Any]) -> bool:
        nonlocal changed
        if rec.get("id") != record_id:
            return False
        rec.setdefault("decision", {})
        old_action = str(rec["decision"].get("action", ""))
        old_stage = str(rec["decision"].get("stage", ""))
        if _decision_priority(action, stage) < _decision_priority(old_action, old_stage):
            return False

        next_decision = {
            "action": action,
            "reason_code": reason_code,
            "reason": reason,
            "stage": stage,
            "candidate_score": round(_safe(candidate_score), 4),
            "base_score": round(_safe(base_score), 4),
            "score_floor": round(_safe(score_floor), 4),
            "forecast_return_pct": round(_safe(forecast_return_pct), 4),
            "today_change_pct": round(_safe(today_change_pct), 4),
            "ml_proba": None if ml_proba is None else round(_safe(ml_proba), 6),
            "mtf_soft_penalty": round(_safe(mtf_soft_penalty), 4),
            "fresh_priority": bool(fresh_priority),
            "catchup": bool(catchup),
            "continuation_profile": bool(continuation_profile),
            "near_miss": bool(near_miss),
            "signal_flags": signal_flags or {},
        }
        decision_changed = rec.get("decision") != next_decision
        if decision_changed:
            # A changed action/score is a new decision, not the first collector
            # observation. Preserve its prior boundary instead of backdating it.
            rec.setdefault("decision_history", []).append({
                "decision": rec.get("decision"),
                "decision_provenance": rec.get("decision_provenance"),
            })
            rec["decision"] = next_decision
        provenance_changed = policy_provenance.update_decision_provenance(
            rec, decision_provenance
        )
        if decision_changed and rec.get("decision_provenance") != decision_provenance:
            rec["decision_provenance"] = dict(decision_provenance)
            provenance_changed = True
        rec.setdefault("labels", {})
        trade_taken_changed = action == "take" and rec["labels"].get("trade_taken") is not True
        if trade_taken_changed:
            rec["labels"]["trade_taken"] = True
        changed = decision_changed or provenance_changed or trade_taken_changed
        return changed

    _rewrite_records(_mutate, strict=strict)
    return changed


def log_candidate(
    *,
    sym: str,
    tf: str,
    bar_ts: int,
    signal_type: str,
    is_bull_day: bool,
    feat: Dict[str, Any],
    i: int,
    data: Any,
    action: str,
    reason_code: str = "",
    reason: str = "",
    stage: str = "",
    candidate_score: float = 0.0,
    base_score: float = 0.0,
    score_floor: float = 0.0,
    forecast_return_pct: float = 0.0,
    today_change_pct: float = 0.0,
    ml_proba: Optional[float] = None,
    mtf_soft_penalty: float = 0.0,
    fresh_priority: bool = False,
    catchup: bool = False,
    continuation_profile: bool = False,
    signal_flags: Optional[Dict[str, bool]] = None,
    near_miss: bool = False,
    btc_vs_ema50: float = 0.0,
    btc_momentum_4h: float = 0.0,
    market_vol_24h: float = 0.0,
    strict: bool = False,
    record_buffer: Optional[list] = None,
) -> str:
    record_id = _candidate_id(sym, tf, bar_ts)
    source = "data_collector" if stage == "collector" else "main_monitor"
    decision_provenance = policy_provenance.build_observation_provenance(
        bar_ts=bar_ts,
        tf=tf,
        source=source,
    )
    if record_buffer is None and _is_logged(record_id):
        _update_existing_candidate(
            record_id=record_id,
            action=action,
            reason_code=reason_code,
            reason=reason,
            stage=stage,
            candidate_score=candidate_score,
            base_score=base_score,
            score_floor=score_floor,
            forecast_return_pct=forecast_return_pct,
            today_change_pct=today_change_pct,
            ml_proba=ml_proba,
            mtf_soft_penalty=mtf_soft_penalty,
            fresh_priority=fresh_priority,
            catchup=catchup,
            continuation_profile=continuation_profile,
            signal_flags=signal_flags,
            near_miss=near_miss,
            decision_provenance=decision_provenance,
            strict=strict,
        )
        return record_id

    rec = build_runtime_record(
        sym=sym,
        tf=tf,
        signal_type=signal_type,
        is_bull_day=is_bull_day,
        bar_ts=bar_ts,
        feat=feat,
        data=data,
        i=i,
        btc_vs_ema50=btc_vs_ema50,
        btc_momentum_4h=btc_momentum_4h,
        market_vol_24h=market_vol_24h,
    )
    rec_id = record_id
    rec["id"] = rec_id
    rec["ts_signal"] = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec["bar_ts"] = bar_ts
    rec["seq_feature_names"] = SEQ_FEATURE_NAMES
    rec["provenance"] = dict(decision_provenance)
    rec["decision_provenance"] = dict(decision_provenance)
    rec["decision"] = {
        "action": action,
        "reason_code": reason_code,
        "reason": reason,
        "stage": stage,
        "candidate_score": round(_safe(candidate_score), 4),
        "base_score": round(_safe(base_score), 4),
        "score_floor": round(_safe(score_floor), 4),
        "forecast_return_pct": round(_safe(forecast_return_pct), 4),
        "today_change_pct": round(_safe(today_change_pct), 4),
        "ml_proba": None if ml_proba is None else round(_safe(ml_proba), 6),
        "mtf_soft_penalty": round(_safe(mtf_soft_penalty), 4),
        "fresh_priority": bool(fresh_priority),
        "catchup": bool(catchup),
        "continuation_profile": bool(continuation_profile),
        "near_miss": bool(near_miss),
        "signal_flags": signal_flags or {},
    }
    rec["labels"] = {
        "ret_3": None,
        "ret_5": None,
        "ret_10": None,
        "label_3": None,
        "label_5": None,
        "label_10": None,
        "trade_taken": action == "take",
        "trade_exit_pnl": None,
        "trade_exit_reason": None,
        "trade_bars_held": None,
        "linked_ml_record_id": "",
    }
    if record_buffer is not None:
        if stage != "collector" or action != "candidate":
            raise ValueError("only collector snapshots may be buffered")
        record_buffer.append(rec)
        return rec_id
    try:
        if _append(rec) is False:
            _update_existing_candidate(
                record_id=record_id, decision_provenance=decision_provenance,
                strict=strict, **rec["decision"],
            )
        _mark_logged(record_id)
    except Exception as e:
        _pylog.warning("critic_dataset write error: %s", e)
        if strict:
            raise DatasetIntegrityError(str(e)) from e
        return ""
    return rec_id


def append_collector_batch(records: list[dict]) -> dict:
    """One uniqueness barrier; preserve existing, possibly higher-priority evidence."""
    prepared = {}
    for rec in records:
        if not rec.get("id") or rec.get("decision", {}).get("stage") != "collector" or rec["decision"].get("action") != "candidate":
            raise DatasetIntegrityError("invalid collector batch record")
        prepared.setdefault(rec["id"], rec)
    if not prepared:
        return {"new_ids": 0, "existing_ids": 0}
    try:
        with _dataset_io_lock():
            ids = set()
            if CRITIC_FILE.exists():
                with CRITIC_FILE.open("r", encoding="utf-8") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise DatasetIntegrityError("non-object dataset row")
                        if row.get("id"):
                            ids.add(row["id"])
            new = [r for k, r in prepared.items() if k not in ids]
            # Serialize all rows before any write; serialization failures cannot
            # leave a prefix falsely reported as a complete successful batch.
            payload = "".join(json.dumps(r, ensure_ascii=False, cls=_Enc) + "\n" for r in new)
            CRITIC_FILE.parent.mkdir(parents=True, exist_ok=True)
            with CRITIC_FILE.open("a", encoding="utf-8") as destination:
                destination.write(payload)
                destination.flush()
                os.fsync(destination.fileno())
            for rec_id in prepared:
                _mark_logged(rec_id)
            _disk_id_cache.pop(str(CRITIC_FILE.resolve()), None)
        return {"new_ids": len(new), "existing_ids": len(prepared)-len(new)}
    except Exception as exc:
        raise DatasetIntegrityError(str(exc)) from exc


def mark_trade_taken(record_id: str, linked_ml_record_id: str = "") -> None:
    if not CRITIC_FILE.exists():
        return

    def _mutate(rec: Dict[str, Any]) -> bool:
        if rec.get("id") != record_id:
            return False
        rec.setdefault("labels", {})
        changed = False
        if rec["labels"].get("trade_taken") is not True:
            rec["labels"]["trade_taken"] = True
            changed = True
        if linked_ml_record_id and rec["labels"].get("linked_ml_record_id") != linked_ml_record_id:
            rec["labels"]["linked_ml_record_id"] = linked_ml_record_id
            changed = True
        if not linked_ml_record_id and not changed:
            return False
        return changed

    _rewrite_records(_mutate)


def fill_trade_outcome(record_id: str, exit_pnl: float, exit_reason: str, bars_held: int) -> None:
    if not CRITIC_FILE.exists():
        return

    def _mutate(rec: Dict[str, Any]) -> bool:
        if rec.get("id") != record_id:
            return False
        rec.setdefault("labels", {})
        new_exit_pnl = round(_safe(exit_pnl), 4)
        new_exit_reason = exit_reason
        new_bars_held = int(bars_held)
        if (
            rec["labels"].get("trade_exit_pnl") == new_exit_pnl
            and rec["labels"].get("trade_exit_reason") == new_exit_reason
            and rec["labels"].get("trade_bars_held") == new_bars_held
        ):
            return False
        rec["labels"]["trade_exit_pnl"] = new_exit_pnl
        rec["labels"]["trade_exit_reason"] = new_exit_reason
        rec["labels"]["trade_bars_held"] = new_bars_held
        policy_provenance.attach_label_provenance(
            rec,
            label_keys=("trade_exit_pnl", "trade_exit_reason", "trade_bars_held"),
            source="main_monitor_trade_exit",
        )
        return True

    _rewrite_records(_mutate)


def fill_learning_labels(
    record_id: str,
    labels: Dict[str, Any],
    *,
    label_definition: str | Dict[str, str] | None = None,
    label_time: Optional[datetime] = None,
    source: str = "learning_labeler",
) -> None:
    if not record_id or not labels or not CRITIC_FILE.exists():
        return

    def _mutate(rec: Dict[str, Any]) -> bool:
        if rec.get("id") != record_id:
            return False
        rec.setdefault("labels", {})
        changed = False
        for key, value in labels.items():
            if rec["labels"].get(key) != value:
                rec["labels"][key] = value
                changed = True
        if policy_provenance.attach_label_provenance(
            rec,
            label_keys=labels,
            definition=label_definition,
            label_time=label_time,
            source=source,
        ):
            changed = True
        return changed

    _rewrite_records(_mutate)


def fill_forward_label(record_id: str, horizon: int, ret_pct: float) -> None:
    if not CRITIC_FILE.exists():
        return
    key_ret = f"ret_{horizon}"
    key_label = f"label_{horizon}"

    def _mutate(rec: Dict[str, Any]) -> bool:
        if rec.get("id") != record_id:
            return False
        rec.setdefault("labels", {})
        new_ret = round(_safe(ret_pct), 4)
        new_label = _safe(ret_pct) > 0
        if rec["labels"].get(key_ret) == new_ret and rec["labels"].get(key_label) == new_label:
            return False
        rec["labels"][key_ret] = new_ret
        rec["labels"][key_label] = new_label
        try:
            label_time = policy_provenance.forward_label_time(
                bar_ts=int(rec.get("bar_ts") or 0),
                tf=str(rec.get("tf") or ""),
                horizon=horizon,
            )
        except (TypeError, ValueError):
            label_time = None
        policy_provenance.attach_label_provenance(
            rec,
            label_keys=(key_ret, key_label),
            label_time=label_time,
            source="main_monitor_forward_label",
        )
        return True

    _rewrite_records(_mutate)


def _fill_pending_record(
    rec: Dict[str, Any],
    *,
    t_arr: Any,
    c_arr: Any,
    bar_ms: int,
    source: str = "collector_forward_label",
    market_evidence: Optional[dict] = None,
) -> bool:
    if (source == "historical_candidate_label_recovery"
            and not policy_provenance.observation_provenance_valid(rec)):
        return False
    lab = rec.get("labels", {})
    rec_bar_ts = rec.get("bar_ts", 0)
    idx_arr = np.where(t_arr == rec_bar_ts)[0]
    if len(idx_arr) == 0:
        return False
    entry_close = float(c_arr[idx_arr[0]])
    if not np.isfinite(entry_close) or entry_close <= 0:
        return False
    changed = False
    for h in (3, 5, 10):
        key_ret = f"ret_{h}"
        key_label = f"label_{h}"
        if lab.get(key_ret) is not None:
            continue
        target_idx = policy_provenance.closed_target_index(
            t_arr,
            bar_ts=int(rec_bar_ts),
            bar_ms=bar_ms,
            horizon=h,
        )
        if target_idx is None:
            continue
        # Exact endpoints alone do not prove a complete closed-bar horizon.
        start_idx = int(idx_arr[0])
        window = np.asarray(t_arr[start_idx:target_idx + 2], dtype=np.int64)
        if len(window) != h + 2 or not np.all(np.diff(window) == bar_ms):
            continue
        future_close = float(c_arr[target_idx])
        if not np.isfinite(future_close) or future_close <= 0:
            continue
        ret_pct = (future_close / entry_close - 1) * 100
        rec["labels"][key_ret] = round(ret_pct, 4)
        rec["labels"][key_label] = ret_pct > 0
        policy_provenance.attach_label_provenance(
            rec,
            label_keys=(key_ret, key_label),
            label_time=policy_provenance.forward_label_time(
                bar_ts=int(rec_bar_ts),
                tf=str(rec.get("tf") or ""),
                horizon=h,
            ),
            source=source,
        )
        if market_evidence:
            for key in (key_ret, key_label):
                rec["label_provenance"][key].setdefault("market_evidence", market_evidence)
        changed = True
    return changed


def fill_pending_from_data(sym: str, tf: str, t_arr: Any, c_arr: Any, bar_ms: int) -> None:
    if not CRITIC_FILE.exists():
        return

    def _mutate(rec: Dict[str, Any]) -> bool:
        if rec.get("sym") != sym or rec.get("tf") != tf:
            return False
        return _fill_pending_record(rec, t_arr=t_arr, c_arr=c_arr, bar_ms=bar_ms)

    _rewrite_records(_mutate)


def fill_pending_batch(series: Any, *, strict: bool = False) -> None:
    """Mature all supplied symbol/timeframe series with one dataset rewrite."""
    if not CRITIC_FILE.exists():
        return
    by_pair: Dict[tuple[str, str], Dict[str, Any]] = {}
    for item in series or ():
        if not isinstance(item, dict):
            continue
        sym = str(item.get("sym") or "")
        tf = str(item.get("tf") or "")
        t_arr = item.get("t_arr")
        c_arr = item.get("c_arr")
        bar_ms = _safe_int(item.get("bar_ms"))
        if not sym or not tf or t_arr is None or c_arr is None or not bar_ms:
            continue
        by_pair[(sym, tf)] = {
            "t_arr": t_arr,
            "c_arr": c_arr,
            "bar_ms": bar_ms,
            "source": str(item.get("source") or "collector_forward_label"),
            "market_evidence": item.get("market_evidence"),
        }
    if not by_pair:
        return

    def _mutate(rec: Dict[str, Any]) -> bool:
        payload = by_pair.get((str(rec.get("sym") or ""), str(rec.get("tf") or "")))
        if payload is None:
            return False
        return _fill_pending_record(rec, **payload)

    _rewrite_records(_mutate, strict=strict)


def _teacher_local_window(target_day: date, phase: str, tz: ZoneInfo) -> tuple[datetime, datetime]:
    start_local = datetime.combine(target_day, dt_time.min, tzinfo=tz)
    if phase == "midday":
        end_local = datetime.combine(target_day, dt_time(hour=12), tzinfo=tz)
    else:
        end_local = datetime.combine(target_day, dt_time.max, tzinfo=tz)
    return start_local, end_local


def _teacher_label_available_at(target_day: date, phase: str, tz: ZoneInfo) -> datetime:
    local_midnight = datetime.combine(target_day, dt_time.min, tzinfo=tz)
    available_local = (
        local_midnight + timedelta(hours=12)
        if phase == "midday"
        else local_midnight + timedelta(days=1)
    )
    return available_local.astimezone(timezone.utc)


def _record_local_dt(rec: Dict[str, Any], tz: ZoneInfo) -> Optional[datetime]:
    ts = _parse_utc_iso(rec.get("ts_signal"))
    if ts is None:
        bar_ts = _safe_int(rec.get("bar_ts"))
        if bar_ts is None:
            return None
        ts = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc)
    return ts.astimezone(tz)


def _phase_teacher_payload(
    *,
    sym: str,
    phase: str,
    target_day_local: str,
    timezone_name: str,
    early_capture_ratio_min: float,
    exchange_summary: Optional[Dict[str, Any]],
    exchange_rank: Optional[int],
    watchlist_summary: Optional[Dict[str, Any]],
    watchlist_rank: Optional[int],
    false_positive_buy: bool,
) -> Dict[str, Any]:
    chosen = watchlist_summary or exchange_summary or {}
    capture_ratio = chosen.get("capture_ratio")
    capture_ratio_f = None if capture_ratio is None else round(_safe(capture_ratio), 4)
    status = None
    reason = None
    if watchlist_summary:
        status = watchlist_summary.get("status")
        reason = watchlist_summary.get("reason")
    elif exchange_summary:
        status = exchange_summary.get("status")
        reason = exchange_summary.get("reason")

    return {
        "phase": phase,
        "target_day_local": target_day_local,
        "timezone": timezone_name,
        "exchange_top_gainer": exchange_summary is not None,
        "exchange_top_rank": exchange_rank,
        "watchlist_top_gainer": watchlist_summary is not None,
        "watchlist_top_rank": watchlist_rank,
        "status": status,
        "reason": reason,
        "capture_ratio": capture_ratio_f,
        "early_capture": bool(
            watchlist_summary is not None
            and capture_ratio_f is not None
            and capture_ratio_f >= float(early_capture_ratio_min)
            and str(status or "") == "bought"
        ),
        "bot_false_positive_buy": bool(false_positive_buy),
        "day_change_pct": None if chosen.get("day_change_pct") is None else round(_safe(chosen.get("day_change_pct")), 4),
        "quote_volume_24h": None if chosen.get("quote_volume_24h") is None else round(_safe(chosen.get("quote_volume_24h")), 2),
        "entries_count": _safe_int(chosen.get("entries_count")),
        "blocked_count": _safe_int(chosen.get("blocked_count")),
        "first_entry_time": chosen.get("first_entry_time"),
        "first_entry_mode": chosen.get("first_entry_mode"),
        "first_entry_price": None if chosen.get("first_entry_price") is None else round(_safe(chosen.get("first_entry_price")), 8),
        "capture_ratio_at_entry": None if chosen.get("capture_ratio_at_entry") is None else round(_safe(chosen.get("capture_ratio_at_entry")), 4),
        "lead_time_to_final_top_min": _safe_int(chosen.get("lead_time_to_final_top_min")),
        "opportunity_from_entry_pct": None if chosen.get("opportunity_from_entry_pct") is None else round(_safe(chosen.get("opportunity_from_entry_pct")), 4),
        "latest_exit_time": chosen.get("latest_exit_time"),
        "latest_exit_pnl_pct": None if chosen.get("latest_exit_pnl_pct") is None else round(_safe(chosen.get("latest_exit_pnl_pct")), 4),
        "exit_efficiency": None if chosen.get("exit_efficiency") is None else round(_safe(chosen.get("exit_efficiency")), 4),
        "giveback_pct": None if chosen.get("giveback_pct") is None else round(_safe(chosen.get("giveback_pct")), 4),
        "cooldown_harm_pct": None if chosen.get("cooldown_harm_pct") is None else round(_safe(chosen.get("cooldown_harm_pct")), 4),
    }


def annotate_top_gainer_teacher(report: Dict[str, Any]) -> Dict[str, Any]:
    if not CRITIC_FILE.exists():
        return {"rows_scanned": 0, "rows_annotated": 0, "symbols_tagged": 0, "phase": str(report.get("phase", ""))}

    phase = str(report.get("phase", "") or "").strip().lower()
    if phase not in {"midday", "final"}:
        raise ValueError(f"Unsupported top-gainer phase: {phase!r}")

    target_day_text = str(report.get("target_day_local", "") or "").strip()
    if not target_day_text:
        raise ValueError("Top-gainer report missing target_day_local")
    target_day = date.fromisoformat(target_day_text)

    settings = report.get("settings") or {}
    timezone_name = str(settings.get("timezone", "Europe/Budapest") or "Europe/Budapest")
    tz = ZoneInfo(timezone_name)
    start_local, end_local = _teacher_local_window(target_day, phase, tz)
    early_capture_ratio_min = float(settings.get("early_capture_ratio_min", 0.35) or 0.35)

    exchange_map: Dict[str, Dict[str, Any]] = {}
    exchange_rank_map: Dict[str, int] = {}
    for idx, item in enumerate(report.get("exchange_top_gainers") or [], start=1):
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).strip()
        if not sym:
            continue
        exchange_map[sym] = item
        exchange_rank_map[sym] = idx

    watchlist_map: Dict[str, Dict[str, Any]] = {}
    watchlist_rank_map: Dict[str, int] = {}
    for idx, item in enumerate(report.get("watchlist_top_gainers") or [], start=1):
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).strip()
        if not sym:
            continue
        watchlist_map[sym] = item
        watchlist_rank_map[sym] = idx

    false_positive_symbols = {
        str(sym).strip()
        for sym in (report.get("bot_false_positive_symbols") or [])
        if str(sym).strip()
    }
    tagged_symbols = set(exchange_map) | set(watchlist_map) | false_positive_symbols
    rows_scanned = 0
    label_time = _teacher_label_available_at(target_day, phase, tz)

    def _annotate(rec: Dict[str, Any], *, count_scan: bool) -> bool:
        nonlocal rows_scanned
        if count_scan:
            rows_scanned += 1
        local_dt = _record_local_dt(rec, tz)
        if local_dt is None or not (start_local <= local_dt <= end_local):
            return False

        sym = str(rec.get("sym", "")).strip()
        phase_payload = _phase_teacher_payload(
            sym=sym,
            phase=phase,
            target_day_local=target_day_text,
            timezone_name=timezone_name,
            early_capture_ratio_min=early_capture_ratio_min,
            exchange_summary=exchange_map.get(sym),
            exchange_rank=exchange_rank_map.get(sym),
            watchlist_summary=watchlist_map.get(sym),
            watchlist_rank=watchlist_rank_map.get(sym),
            false_positive_buy=sym in false_positive_symbols,
        )
        teacher = rec.setdefault("teacher", {})
        changed = False
        if teacher.get(phase) != phase_payload:
            teacher[phase] = phase_payload
            changed = True
        if policy_provenance.attach_label_provenance(
            rec,
            label_keys=(f"teacher.{phase}",),
            definition=(
                f"immutable {phase} top-gainer critic membership and capture outcome "
                f"for local day {target_day_text}"
            ),
            label_time=label_time,
            source=f"top_gainer_critic_{phase}",
        ):
            changed = True
        return changed

    def _mutate_preview(rec: Dict[str, Any]) -> bool:
        nonlocal rows_scanned
        rows_scanned += 1
        return _annotate(rec, count_scan=False)

    maybe_changed, maybe_bad_rows = _scan_mutations(_mutate_preview)
    rows_annotated = 0
    if maybe_changed or maybe_bad_rows:
        def _mutate_commit(rec: Dict[str, Any]) -> bool:
            nonlocal rows_annotated
            changed = _annotate(rec, count_scan=False)
            if changed:
                rows_annotated += 1
            return changed

        try:
            with _dataset_io_lock():
                CRITIC_FILE.parent.mkdir(parents=True, exist_ok=True)
                tmp = CRITIC_FILE.with_name(
                    f"{CRITIC_FILE.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                changed, had_bad_rows = _write_mutated_stream(_mutate_commit, tmp)
                if changed or had_bad_rows:
                    _atomic_replace_with_retry(tmp, CRITIC_FILE)
                else:
                    tmp.unlink(missing_ok=True)
        except Exception as e:
            _pylog.warning("critic_dataset rewrite error: %s", e)
            rows_annotated = 0
    return {
        "phase": phase,
        "target_day_local": target_day_text,
        "rows_scanned": rows_scanned,
        "rows_annotated": rows_annotated,
        "symbols_tagged": len(tagged_symbols),
        "timezone": timezone_name,
    }
