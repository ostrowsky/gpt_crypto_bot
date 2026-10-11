"""App-local immutable public snapshots; receipt-time, never event-time replay."""

import json
import sqlite3
import threading
import time
import uuid

from .adapter import AdapterError, _integer, _text
from .isolation import AppPaths
from .journal import _WriterLease, _json, _digest

FIXED = dict(application_id="binance_demo_bot", environment="BINANCE_SPOT_DEMO",
             objective_contract_id="daily_net_equity_pnl_v1", schema_version=1)
KINDS = frozenset(("UNIVERSE", "QUOTES", "CANDLES", "FEATURE", "VALUATION", "INCIDENT"))
PROCESS_ID = uuid.uuid4().hex


class MarketSnapshotStore:
    def __init__(self, paths, *, account_alias, episode_id, scope_id, database="market.sqlite"):
        if not isinstance(paths, AppPaths):
            raise AdapterError("MARKET_IDENTITY_MISMATCH")
        for value in (account_alias, episode_id, scope_id):
            _text(value)
        self.identity = dict(FIXED, account_alias=account_alias,
                             account_episode_id=episode_id, portfolio_scope_id=scope_id)
        self.path = paths.runtime_path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._lease = _WriterLease(paths.runtime_path(database + ".writer.lock"))
        self._con = None
        try:
            self._con = sqlite3.connect(self.path, timeout=0, isolation_level=None, check_same_thread=False)
            self._con.row_factory = sqlite3.Row
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA synchronous=FULL")
            self._con.execute("PRAGMA foreign_keys=ON")
            tables = {r[0] for r in self._con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not tables:
                self._con.execute("BEGIN IMMEDIATE")
                self._con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
                self._con.execute("CREATE TABLE snapshots (seq INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,kind TEXT NOT NULL,available_ms INTEGER NOT NULL,payload_json TEXT NOT NULL,content_sha256 TEXT NOT NULL)")
                self._con.execute("CREATE INDEX snapshot_cutoff ON snapshots(kind,available_ms,seq)")
                self._con.execute("INSERT INTO meta VALUES ('identity',?)", (_json(self.identity),))
                for table in ("meta", "snapshots"):
                    for operation in ("UPDATE", "DELETE"):
                        self._con.execute(f"CREATE TRIGGER immutable_{table}_{operation} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT,'immutable market evidence'); END")
                self._con.commit()
            else:
                row = self._con.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
                if not row or json.loads(row[0]) != self.identity:
                    raise AdapterError("MARKET_IDENTITY_MISMATCH")
            self._verify()
        except BaseException:
            self.close()
            raise

    def _verify(self):
        try:
            triggers = {r[0] for r in self._con.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
            if not {f"immutable_{t}_{op}" for t in ("meta", "snapshots") for op in ("UPDATE", "DELETE")} <= triggers:
                raise ValueError()
            seen = {}
            for row in self._con.execute("SELECT * FROM snapshots ORDER BY seq"):
                value = json.loads(row["payload_json"])
                basis = {k: v for k, v in value.items() if k not in {"snapshot_id", "event_id", "recorded_at_ms"}}
                if (any(value.get(k) != v for k, v in self.identity.items())
                        or _digest(value) != row["content_sha256"] or _digest(basis) != row["id"]
                        or value["snapshot_id"] != row["id"] or value["event_id"] != row["id"]
                        or value["kind"] != row["kind"] or value["kind"] not in KINDS
                        or value["available_at_ms"] != row["available_ms"]
                        or value["source_sha256"] != _digest(value["data"])):
                    raise ValueError()
                minimum = value["received_at_ms"]
                for dep in value["dependencies"]:
                    minimum = max(minimum, seen[dep])
                if value["available_at_ms"] != minimum:
                    raise ValueError()
                seen[row["id"]] = minimum
        except (KeyError, TypeError, ValueError, sqlite3.Error):
            raise AdapterError("MARKET_INTEGRITY_FAILED") from None

    def append(self, kind, data, *, received_at_ms, received_monotonic_ns, dependencies=()):
        if kind not in KINDS or not isinstance(data, dict):
            raise AdapterError("SCHEMA_INVALID")
        _integer(received_at_ms)
        _integer(received_monotonic_ns)
        if not isinstance(dependencies, (tuple, list)) or any(not isinstance(d, str) for d in dependencies):
            raise AdapterError("DEPENDENCY_MISSING")
        with self._lock:
            dependencies = sorted(set(dependencies))
            available = received_at_ms
            for dep in dependencies:
                row = self._con.execute("SELECT available_ms FROM snapshots WHERE id=?", (dep,)).fetchone()
                if row is None:
                    raise AdapterError("DEPENDENCY_MISSING")
                available = max(available, row[0])
            basis = dict(self.identity, kind=kind, data=data, dependencies=dependencies,
                         event_type="MarketSnapshotRecorded", exchange_event_time_ms=None,
                         received_at_ms=received_at_ms, received_monotonic_ns=received_monotonic_ns,
                         available_at_ms=available, process_instance_id=PROCESS_ID,
                         source_sha256=_digest(data), causation_id=None, model_id=None,
                         policy_version=None, arm_id=None, risk_contract_id=None)
            identifier = _digest(basis)
            value = dict(basis, event_id=identifier, snapshot_id=identifier, recorded_at_ms=time.time_ns()//1_000_000)
            self._con.execute("BEGIN IMMEDIATE")
            try:
                self._con.execute("INSERT OR IGNORE INTO snapshots(id,kind,available_ms,payload_json,content_sha256) VALUES (?,?,?,?,?)",
                                  (identifier, kind, available, _json(value), _digest(value)))
                self._con.commit()
            except BaseException:
                self._con.rollback()
                raise
            return self.get(identifier)

    def get(self, identifier):
        with self._lock:
            row = self._con.execute("SELECT payload_json FROM snapshots WHERE id=?", (identifier,)).fetchone()
            return json.loads(row[0]) if row else None

    def records(self, kind=None, *, cutoff_ms=None):
        clauses, args = [], []
        if kind is not None:
            clauses.append("kind=?")
            args.append(kind)
        if cutoff_ms is not None:
            _integer(cutoff_ms)
            clauses.append("available_ms<=?")
            args.append(cutoff_ms)
        sql = "SELECT payload_json FROM snapshots" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY seq"
        with self._lock:
            return [json.loads(r[0]) for r in self._con.execute(sql, args)]

    def as_of(self, kind, cutoff_ms):
        rows = self.records(kind, cutoff_ms=cutoff_ms)
        return rows[-1] if rows else None

    def sqlite_settings(self):
        return {key: self._con.execute("PRAGMA " + key).fetchone()[0]
                for key in ("journal_mode", "synchronous", "foreign_keys")}

    def last_failure(self, operation, success_kind, cutoff_ms, *, symbol=None):
        """Sequence ordering resolves equal clocks; future incidents are excluded."""
        with self._lock:
            # Do not deserialize a 50MB universe for every member's readiness.
            sql = "SELECT kind,CASE WHEN kind='INCIDENT' THEN json_extract(payload_json,'$.data.operation') END FROM snapshots WHERE available_ms<=? AND kind IN (?, 'INCIDENT')"
            args = [cutoff_ms, success_kind]
            if symbol is not None:
                sql += " AND json_extract(payload_json,'$.data.symbol')=?"
                args.append(symbol)
            sql += " ORDER BY seq DESC"
            for kind, recorded_operation in self._con.execute(sql, args):
                if kind == success_kind:
                    return False
                if recorded_operation == operation:
                    return True
        return False

    def close(self):
        with self._lock:
            if self._con is not None:
                self._con.close()
                self._con = None
            self._lease.close()
