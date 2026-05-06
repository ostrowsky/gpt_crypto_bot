from __future__ import annotations

import json
import os
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple


Mutator = Callable[[Dict[str, Any]], bool]


def _module_dataset_path(ml_dataset_module: Any, dataset_path: Path | None = None) -> Path:
    raw = dataset_path if dataset_path is not None else getattr(ml_dataset_module, "ML_FILE")
    return Path(raw)


def dataset_lock(ml_dataset_module: Any):
    lock_factory = getattr(ml_dataset_module, "_dataset_io_lock", None)
    if callable(lock_factory):
        return lock_factory()
    return nullcontext()


def collect_mutated_lines(
    ml_dataset_module: Any,
    mutator: Mutator,
    *,
    dataset_path: Path | None = None,
) -> Tuple[List[str], bool, bool]:
    target_path = _module_dataset_path(ml_dataset_module, dataset_path)
    collector = getattr(ml_dataset_module, "_collect_mutated_lines", None)
    module_path = _module_dataset_path(ml_dataset_module)
    if callable(collector):
        if target_path == module_path:
            return collector(mutator)
        original_path = getattr(ml_dataset_module, "ML_FILE")
        setattr(ml_dataset_module, "ML_FILE", target_path)
        try:
            return collector(mutator)
        finally:
            setattr(ml_dataset_module, "ML_FILE", original_path)

    if not target_path.exists():
        return [], False, False

    updated: List[str] = []
    changed = False
    had_bad_rows = False
    for line in target_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            had_bad_rows = True
            updated.append(line)
            continue
        if not isinstance(rec, dict):
            updated.append(line)
            continue
        row_changed = bool(mutator(rec))
        changed = changed or row_changed
        updated.append(json.dumps(rec, ensure_ascii=False))
    return updated, changed, had_bad_rows


def write_dataset_lines(
    ml_dataset_module: Any,
    updated_lines: Iterable[str],
    *,
    dataset_path: Path | None = None,
) -> None:
    target_path = _module_dataset_path(ml_dataset_module, dataset_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_name(f"{target_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    payload = "\n".join(list(updated_lines))
    if payload:
        payload += "\n"
    tmp.write_text(payload, encoding="utf-8")

    atomic_replace = getattr(ml_dataset_module, "_atomic_replace_with_retry", None)
    if callable(atomic_replace):
        atomic_replace(tmp, target_path)
    else:
        tmp.replace(target_path)
