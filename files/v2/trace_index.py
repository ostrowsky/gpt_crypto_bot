"""Rebuildable, bounded-memory deduplication for an append-only JSONL trace.

The JSONL file is authoritative. A transaction serializes participating writers;
if a write succeeds but the index commit fails, the next sync replays that suffix.
"""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Mapping

from evidence_snapshot_io import open_snapshot


def index_path_for(path: Path) -> Path:
    digest = hashlib.sha256(os.path.normcase(str(path.resolve())).encode("utf-8")).hexdigest()[:24]
    return path.parent / ".runtime" / "trace_indexes" / f"{digest}.sqlite3"


def _key(event: Mapping[str, object]) -> str:
    return f"{event.get('sym')}|{event.get('tf')}|{event.get('bar_ts')}"


def _identity(stat) -> str:
    return f"{stat.st_dev}:{stat.st_ino}"


def _save_position(db: sqlite3.Connection, path: Path, unterminated: bool) -> None:
    stat = path.stat()
    db.execute(
        "INSERT OR REPLACE INTO position VALUES (1, ?, ?, ?, ?)",
        (_identity(stat), stat.st_size, stat.st_mtime_ns, int(unterminated)),
    )


def _sync(db: sqlite3.Connection, path: Path) -> bool:
    previous = db.execute("SELECT identity, size, mtime, unterminated FROM position WHERE id=1").fetchone()
    if not path.exists():
        db.execute("DELETE FROM keys")
        db.execute("DELETE FROM position")
        return False
    stat = path.stat()
    identity = _identity(stat)
    if previous and previous[:3] == (identity, stat.st_size, stat.st_mtime_ns):
        return bool(previous[3])
    incremental = (
        previous and previous[0] == identity and stat.st_size > previous[1]
        and not previous[3]
    )
    offset = previous[1] if incremental else 0
    if not incremental:
        db.execute("DELETE FROM keys")
    unterminated = False
    with open_snapshot(path, "rb") as source:
        source.seek(offset)
        for line in source:
            unterminated = not line.endswith(b"\n")
            try:
                row = json.loads(line.decode("utf-8", errors="ignore"))
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict):
                db.execute("INSERT OR IGNORE INTO keys VALUES (?)", (_key(row),))
    _save_position(db, path, unterminated)
    return unterminated


def append_unique_trace(path: Path, event: Mapping[str, object]) -> None:
    # Validate required fields before creating an index or modifying history.
    event["sym"], event["tf"], event["bar_ts"]
    index_path = index_path_for(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(index_path, timeout=60)) as db:
        db.execute("PRAGMA cache_size=-2048")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("CREATE TABLE IF NOT EXISTS keys (key TEXT PRIMARY KEY) WITHOUT ROWID")
        db.execute("CREATE TABLE IF NOT EXISTS position (id INTEGER PRIMARY KEY, identity TEXT, size INTEGER, mtime INTEGER, unterminated INTEGER)")
        with db:
            db.execute("BEGIN IMMEDIATE")
            unterminated = _sync(db, path)
            key = _key(event)
            if db.execute("SELECT 1 FROM keys WHERE key=?", (key,)).fetchone():
                return
            payload = json.dumps(dict(event), ensure_ascii=False) + "\n"
            with path.open("a", encoding="utf-8") as output:
                if unterminated:
                    output.write("\n")
                output.write(payload)
            db.execute("INSERT INTO keys VALUES (?)", (key,))
            _save_position(db, path, False)
