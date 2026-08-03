from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from blocking import normalize_blocked_reason


ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
DEFAULT_DB = ROOT / ".runtime" / "research_event_cohorts.sqlite3"
TZ = ZoneInfo("Europe/Budapest")
SOURCE_NAMES = ("bot_events.jsonl", "agent_events.jsonl")
FLUSH_LINES = 50_000
LOCK_TIMEOUT_SECONDS = 120.0
SCHEMA_VERSION = 2
BLOCK_INTERVAL_MS = 15 * 60 * 1000


def sync_event_cohorts(
    files_dir: Path = FILES,
    db_path: Path = DEFAULT_DB,
) -> dict[str, Any]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _single_writer_lock(db_path.with_suffix(db_path.suffix + ".lock")):
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            _ensure_schema(conn)
            sources = []
            for name in SOURCE_NAMES:
                sources.append(_sync_source(conn, files_dir / name))
            return {
                "status": "complete",
                "db_path": str(db_path),
                "sources": sources,
                "bytes_processed": sum(int(row.get("bytes_processed") or 0) for row in sources),
                "lines_processed": sum(int(row.get("lines_processed") or 0) for row in sources),
                "relevant_events": sum(int(row.get("relevant_events") or 0) for row in sources),
            }
        finally:
            conn.close()


def load_replay_inputs(
    *,
    files_dir: Path,
    allowed_days: Iterable[str],
    db_path: Path,
    sync: bool = True,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    sync_summary = sync_event_cohorts(files_dir, db_path) if sync else {"status": "not_requested", "db_path": str(db_path)}
    days = sorted({str(day) for day in allowed_days if str(day)})
    if not days:
        return [], {}, sync_summary
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in days)
        blocked_rows = conn.execute(
            f"SELECT day, symbol, reason_code, block_count, first_ts, first_hour, "
            f"first_price, first_source, first_tf, first_signal_type "
            f"FROM blocked_cohorts WHERE day IN ({placeholders}) "
            "ORDER BY day, symbol, reason_code, first_ts",
            days,
        ).fetchall()
        merged_blocked: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in blocked_rows:
            key = (row["day"], row["symbol"], row["reason_code"])
            current = merged_blocked.get(key)
            if current is not None:
                current["block_count"] += int(row["block_count"] or 0)
                continue
            merged_blocked[key] = {
                "day": row["day"],
                "symbol": row["symbol"],
                "reason_code": row["reason_code"],
                "block_count": int(row["block_count"] or 0),
                "ts": row["first_ts"],
                "hour": row["first_hour"],
                "price": row["first_price"],
                "source": row["first_source"],
                "tf": row["first_tf"],
                "signal_type": row["first_signal_type"],
            }

        entry_rows = conn.execute(
            f"SELECT day, symbol, ts, hour, price, mode FROM trade_events "
            f"WHERE event_type='entry' AND day IN ({placeholders}) ORDER BY ts",
            days,
        ).fetchall()
        entries: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in entry_rows:
            entries.setdefault((row["day"], row["symbol"]), []).append({
                "ts": row["ts"],
                "hour": row["hour"],
                "price": row["price"],
                "mode": row["mode"],
            })
        return list(merged_blocked.values()), entries, sync_summary
    finally:
        conn.close()


def load_trade_events(
    *,
    files_dir: Path,
    db_path: Path,
    sync: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sync_summary = sync_event_cohorts(files_dir, db_path) if sync else {"status": "not_requested", "db_path": str(db_path)}
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT event_type, ts, symbol, tf, mode, price, pnl_pct, reason, source "
            "FROM trade_events ORDER BY ts, source_file, byte_offset"
        ).fetchall()
        return [
            {
                "event": row["event_type"],
                "ts": row["ts"],
                "sym": row["symbol"],
                "tf": row["tf"],
                "mode": row["mode"],
                "price": row["price"],
                "pnl_pct": row["pnl_pct"],
                "reason": row["reason"],
                "source": row["source"],
            }
            for row in rows
        ], sync_summary
    finally:
        conn.close()


def load_blocked_intervals(
    *,
    files_dir: Path,
    allowed_days: Iterable[str],
    db_path: Path,
    sync: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load compact 15-minute blocker intervals for causal episode joins."""
    sync_summary = sync_event_cohorts(files_dir, db_path) if sync else {"status": "not_requested", "db_path": str(db_path)}
    days = sorted({str(day) for day in allowed_days if str(day)})
    if not days:
        return [], sync_summary
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in days)
        rows = conn.execute(
            f"SELECT day, symbol, reason_code, bucket_ms, block_count, first_ts, last_ts, "
            f"first_price, first_source, first_tf, first_signal_type "
            f"FROM blocked_intervals WHERE day IN ({placeholders}) "
            "ORDER BY day, symbol, bucket_ms, reason_code",
            days,
        ).fetchall()
        return [dict(row) for row in rows], sync_summary
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    previous_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS source_state (
            source_file TEXT PRIMARY KEY,
            byte_offset INTEGER NOT NULL,
            source_size INTEGER NOT NULL,
            source_mtime_ns INTEGER NOT NULL,
            synced_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS blocked_cohorts (
            source_file TEXT NOT NULL,
            day TEXT NOT NULL,
            symbol TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            block_count INTEGER NOT NULL,
            first_ts TEXT NOT NULL,
            first_hour INTEGER,
            first_price REAL,
            first_source TEXT,
            first_tf TEXT,
            first_signal_type TEXT,
            PRIMARY KEY (source_file, day, symbol, reason_code)
        );
        CREATE INDEX IF NOT EXISTS idx_blocked_day ON blocked_cohorts(day);
        CREATE TABLE IF NOT EXISTS blocked_intervals (
            source_file TEXT NOT NULL,
            day TEXT NOT NULL,
            symbol TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            bucket_ms INTEGER NOT NULL,
            block_count INTEGER NOT NULL,
            first_ts TEXT NOT NULL,
            last_ts TEXT NOT NULL,
            first_price REAL,
            first_source TEXT,
            first_tf TEXT,
            first_signal_type TEXT,
            PRIMARY KEY (source_file, symbol, reason_code, bucket_ms)
        );
        CREATE INDEX IF NOT EXISTS idx_blocked_interval_day_symbol
            ON blocked_intervals(day, symbol, bucket_ms);
        CREATE TABLE IF NOT EXISTS trade_events (
            source_file TEXT NOT NULL,
            byte_offset INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            ts TEXT NOT NULL,
            day TEXT NOT NULL,
            hour INTEGER,
            symbol TEXT NOT NULL,
            tf TEXT,
            mode TEXT,
            price REAL,
            pnl_pct REAL,
            reason TEXT,
            source TEXT,
            PRIMARY KEY (source_file, byte_offset)
        );
        CREATE INDEX IF NOT EXISTS idx_trade_day_event ON trade_events(day, event_type);
        CREATE INDEX IF NOT EXISTS idx_trade_symbol_ts ON trade_events(symbol, ts);
        """
    )
    existing_sources = int(conn.execute("SELECT COUNT(*) FROM source_state").fetchone()[0])
    if previous_version < SCHEMA_VERSION and existing_sources:
        conn.execute("DELETE FROM blocked_cohorts")
        conn.execute("DELETE FROM blocked_intervals")
        conn.execute("DELETE FROM trade_events")
        conn.execute("DELETE FROM source_state")
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()


@contextlib.contextmanager
def _single_writer_lock(path: Path, timeout_seconds: float = LOCK_TIMEOUT_SECONDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    deadline = time.monotonic() + timeout_seconds
    locked = False
    try:
        while not locked:
            try:
                handle.seek(0)
                if __import__("os").name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"research cohort sync lock timeout: {path}")
                time.sleep(0.1)
        yield
    finally:
        if locked:
            handle.seek(0)
            if __import__("os").name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _sync_source(conn: sqlite3.Connection, path: Path) -> dict[str, Any]:
    source_file = str(path.resolve())
    if not path.exists():
        return {"source_file": source_file, "status": "missing", "bytes_processed": 0, "lines_processed": 0, "relevant_events": 0}
    stat = path.stat()
    state = conn.execute(
        "SELECT byte_offset, source_size FROM source_state WHERE source_file=?",
        (source_file,),
    ).fetchone()
    offset = int(state[0]) if state else 0
    reset = bool(state and (offset > stat.st_size or int(state[1]) > stat.st_size))
    if reset:
        with conn:
            conn.execute("DELETE FROM blocked_cohorts WHERE source_file=?", (source_file,))
            conn.execute("DELETE FROM blocked_intervals WHERE source_file=?", (source_file,))
            conn.execute("DELETE FROM trade_events WHERE source_file=?", (source_file,))
            conn.execute("DELETE FROM source_state WHERE source_file=?", (source_file,))
        offset = 0
    if offset == stat.st_size:
        return {
            "source_file": source_file,
            "status": "current",
            "reset": reset,
            "byte_offset": offset,
            "source_size": stat.st_size,
            "bytes_processed": 0,
            "lines_processed": 0,
            "relevant_events": 0,
        }

    blocked: dict[tuple[str, str, str], dict[str, Any]] = {}
    blocked_intervals: dict[tuple[str, str, int], dict[str, Any]] = {}
    trades: list[tuple[Any, ...]] = []
    start_offset = offset
    lines_processed = 0
    relevant_events = 0
    with path.open("rb") as handle:
        handle.seek(offset)
        while True:
            line_offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            offset = handle.tell()
            lines_processed += 1
            event_type = _event_type(raw)
            if event_type:
                row = _decode_event(raw)
                if row:
                    normalized = _normalize_event(row, event_type, source_file, line_offset)
                    if normalized:
                        relevant_events += 1
                        if event_type == "blocked":
                            _accumulate_blocked(blocked, normalized)
                            _accumulate_blocked_interval(blocked_intervals, normalized)
                        else:
                            trades.append(normalized)
            if lines_processed % FLUSH_LINES == 0:
                _flush(conn, source_file, blocked, blocked_intervals, trades, offset, stat)
                blocked.clear()
                blocked_intervals.clear()
                trades.clear()
        _flush(conn, source_file, blocked, blocked_intervals, trades, offset, stat)
    return {
        "source_file": source_file,
        "status": "rebuilt" if start_offset == 0 else "appended",
        "reset": reset,
        "byte_offset": offset,
        "source_size": stat.st_size,
        "bytes_processed": offset - start_offset,
        "lines_processed": lines_processed,
        "relevant_events": relevant_events,
    }


def _event_type(raw: bytes) -> str:
    for event_type in ("blocked", "entry", "exit"):
        if f'"event": "{event_type}"'.encode() in raw or f'"event":"{event_type}"'.encode() in raw:
            return event_type
    return ""


def _decode_event(raw: bytes) -> dict[str, Any]:
    try:
        row = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return row if isinstance(row, dict) else {}


def _normalize_event(
    row: dict[str, Any],
    event_type: str,
    source_file: str,
    byte_offset: int,
) -> dict[str, Any] | tuple[Any, ...] | None:
    ts = str(row.get("ts") or "")
    day, hour = _local_day_hour(ts)
    symbol = str(row.get("sym") or row.get("symbol") or "").upper()
    if not day or hour is None or not symbol:
        return None
    source = str(row.get("source") or ("market_agent" if Path(source_file).name.startswith("agent") else "bot"))
    if event_type == "blocked":
        return {
            "day": day,
            "hour": hour,
            "ts": ts,
            "symbol": symbol,
            "reason_code": normalize_blocked_reason(str(row.get("signal_type") or ""), str(row.get("reason") or "")),
            "price": _num(row.get("price")),
            "source": source,
            "tf": str(row.get("tf") or ""),
            "signal_type": str(row.get("signal_type") or ""),
            "ts_ms": _ts_ms(ts),
        }
    price = row.get("price") if row.get("price") is not None else row.get("exit_price")
    return (
        source_file,
        byte_offset,
        event_type,
        ts,
        day,
        hour,
        symbol,
        str(row.get("tf") or ""),
        str(row.get("mode") or ""),
        _num(price),
        _num(row.get("pnl_pct")),
        str(row.get("reason") or ""),
        source,
    )


def _accumulate_blocked(target: dict[tuple[str, str, str], dict[str, Any]], row: dict[str, Any]) -> None:
    key = (row["day"], row["symbol"], row["reason_code"])
    current = target.get(key)
    if current is None:
        target[key] = {**row, "block_count": 1}
        return
    current["block_count"] += 1
    if row["ts"] < current["ts"]:
        count = current["block_count"]
        target[key] = {**row, "block_count": count}


def _accumulate_blocked_interval(
    target: dict[tuple[str, str, int], dict[str, Any]],
    row: dict[str, Any],
) -> None:
    ts_ms = int(row.get("ts_ms") or 0)
    if ts_ms <= 0:
        return
    bucket_ms = ts_ms - (ts_ms % BLOCK_INTERVAL_MS)
    key = (row["symbol"], row["reason_code"], bucket_ms)
    current = target.get(key)
    if current is None:
        target[key] = {**row, "bucket_ms": bucket_ms, "block_count": 1, "last_ts": row["ts"]}
        return
    current["block_count"] += 1
    if row["ts"] < current["ts"]:
        current.update({
            "ts": row["ts"],
            "price": row["price"],
            "source": row["source"],
            "tf": row["tf"],
            "signal_type": row["signal_type"],
        })
    if row["ts"] > current["last_ts"]:
        current["last_ts"] = row["ts"]


def _flush(
    conn: sqlite3.Connection,
    source_file: str,
    blocked: dict[tuple[str, str, str], dict[str, Any]],
    blocked_intervals: dict[tuple[str, str, int], dict[str, Any]],
    trades: list[tuple[Any, ...]],
    offset: int,
    stat: Any,
) -> None:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    blocked_values = [
        (
            source_file,
            row["day"],
            row["symbol"],
            row["reason_code"],
            row["block_count"],
            row["ts"],
            row["hour"],
            row["price"],
            row["source"],
            row["tf"],
            row["signal_type"],
        )
        for row in blocked.values()
    ]
    interval_values = [
        (
            source_file,
            row["day"],
            row["symbol"],
            row["reason_code"],
            row["bucket_ms"],
            row["block_count"],
            row["ts"],
            row["last_ts"],
            row["price"],
            row["source"],
            row["tf"],
            row["signal_type"],
        )
        for row in blocked_intervals.values()
    ]
    with conn:
        if blocked_values:
            conn.executemany(
                """
                INSERT INTO blocked_cohorts (
                    source_file, day, symbol, reason_code, block_count,
                    first_ts, first_hour, first_price, first_source, first_tf, first_signal_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_file, day, symbol, reason_code) DO UPDATE SET
                    block_count = blocked_cohorts.block_count + excluded.block_count,
                    first_hour = CASE WHEN excluded.first_ts < blocked_cohorts.first_ts THEN excluded.first_hour ELSE blocked_cohorts.first_hour END,
                    first_price = CASE WHEN excluded.first_ts < blocked_cohorts.first_ts THEN excluded.first_price ELSE blocked_cohorts.first_price END,
                    first_source = CASE WHEN excluded.first_ts < blocked_cohorts.first_ts THEN excluded.first_source ELSE blocked_cohorts.first_source END,
                    first_tf = CASE WHEN excluded.first_ts < blocked_cohorts.first_ts THEN excluded.first_tf ELSE blocked_cohorts.first_tf END,
                    first_signal_type = CASE WHEN excluded.first_ts < blocked_cohorts.first_ts THEN excluded.first_signal_type ELSE blocked_cohorts.first_signal_type END,
                    first_ts = MIN(blocked_cohorts.first_ts, excluded.first_ts)
                """,
                blocked_values,
            )
        if interval_values:
            conn.executemany(
                """
                INSERT INTO blocked_intervals (
                    source_file, day, symbol, reason_code, bucket_ms, block_count,
                    first_ts, last_ts, first_price, first_source, first_tf, first_signal_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_file, symbol, reason_code, bucket_ms) DO UPDATE SET
                    block_count = blocked_intervals.block_count + excluded.block_count,
                    first_price = CASE WHEN excluded.first_ts < blocked_intervals.first_ts THEN excluded.first_price ELSE blocked_intervals.first_price END,
                    first_source = CASE WHEN excluded.first_ts < blocked_intervals.first_ts THEN excluded.first_source ELSE blocked_intervals.first_source END,
                    first_tf = CASE WHEN excluded.first_ts < blocked_intervals.first_ts THEN excluded.first_tf ELSE blocked_intervals.first_tf END,
                    first_signal_type = CASE WHEN excluded.first_ts < blocked_intervals.first_ts THEN excluded.first_signal_type ELSE blocked_intervals.first_signal_type END,
                    first_ts = MIN(blocked_intervals.first_ts, excluded.first_ts),
                    last_ts = MAX(blocked_intervals.last_ts, excluded.last_ts)
                """,
                interval_values,
            )
        if trades:
            conn.executemany(
                "INSERT OR IGNORE INTO trade_events "
                "(source_file, byte_offset, event_type, ts, day, hour, symbol, tf, mode, price, pnl_pct, reason, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                trades,
            )
        conn.execute(
            "INSERT INTO source_state (source_file, byte_offset, source_size, source_mtime_ns, synced_at_utc) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(source_file) DO UPDATE SET "
            "byte_offset=excluded.byte_offset, source_size=excluded.source_size, "
            "source_mtime_ns=excluded.source_mtime_ns, synced_at_utc=excluded.synced_at_utc",
            (source_file, offset, stat.st_size, stat.st_mtime_ns, now),
        )


def _local_day_hour(value: str) -> tuple[str, int | None]:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "", None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    local = parsed.astimezone(TZ)
    return local.date().isoformat(), local.hour


def _ts_ms(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _num(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally index research-relevant event cohorts.")
    parser.add_argument("--files-dir", type=Path, default=FILES)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = sync_event_cohorts(args.files_dir, args.db)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
