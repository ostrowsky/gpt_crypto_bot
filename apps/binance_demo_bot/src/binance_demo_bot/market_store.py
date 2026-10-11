"""App-local immutable public snapshots; receipt-time, never event-time replay."""

import json
import sqlite3
import threading
import time
import uuid
import hashlib
from contextlib import contextmanager
from copy import deepcopy

from .adapter import AdapterError, _integer, _text
from .isolation import AppPaths
from .journal import _WriterLease, _json, _digest

FIXED = dict(application_id="binance_demo_bot", environment="BINANCE_SPOT_DEMO",
             objective_contract_id="daily_net_equity_pnl_v1", schema_version=1)
KINDS = frozenset(("UNIVERSE", "QUOTES", "CANDLES", "FEATURE", "VALUATION", "INCIDENT", "DEPTH"))
PROCESS_ID = uuid.uuid4().hex


class _FairRLock:
    """FIFO contenders; reentrancy keeps a bounded group indivisible."""
    def __init__(self):
        self._condition=threading.Condition(threading.Lock())
        self._owner,self._depth=None,0
        self._issued=self._serving=0

    def acquire(self):
        owner=threading.get_ident()
        with self._condition:
            if self._owner==owner:
                self._depth+=1
                return
            ticket=self._issued
            self._issued+=1
            self._condition.notify_all()
            self._condition.wait_for(lambda:self._owner is None and ticket==self._serving)
            self._owner,self._depth=owner,1

    def release(self):
        with self._condition:
            if self._owner!=threading.get_ident():
                raise RuntimeError("market lock is not owned")
            self._depth-=1
            if not self._depth:
                self._owner=None
                self._serving+=1
                self._condition.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self,*unused):
        self.release()


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
        self._lock = _FairRLock()
        self._lease = _WriterLease(paths.runtime_path(database + ".writer.lock"))
        self._con = None
        self._batch_depth = 0
        self._quote_seq, self._quote_max_ms, self._quote_cache = 0, 0, {}
        self._candle_seq, self._candle_max_ms, self._candle_cache = 0, 0, {}
        self._failure_seq, self._failure_max_ms, self._failure_cache = 0,0,set()
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
        # Large REST snapshots must not exclude writers during pure CPU work.
        data_json = _json(data)
        normalized = json.loads(data_json)
        with self._lock:
            dependencies = sorted(set(dependencies))
            available = received_at_ms
            for dep in dependencies:
                row = self._con.execute("SELECT available_ms FROM snapshots WHERE id=?", (dep,)).fetchone()
                if row is None:
                    raise AdapterError("DEPENDENCY_MISSING")
                available = max(available, row[0])
        basis = dict(self.identity, kind=kind, data=normalized, dependencies=dependencies,
                     event_type="MarketSnapshotRecorded", exchange_event_time_ms=None,
                     received_at_ms=received_at_ms, received_monotonic_ns=received_monotonic_ns,
                     available_at_ms=available, process_instance_id=PROCESS_ID,
                     source_sha256=hashlib.sha256(data_json.encode()).hexdigest(), causation_id=None, model_id=None,
                     policy_version=None, arm_id=None, risk_contract_id=None)
        serialize = lambda value: json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
        identifier = hashlib.sha256(serialize(basis).encode()).hexdigest()
        value = dict(basis, event_id=identifier, snapshot_id=identifier, recorded_at_ms=time.time_ns()//1_000_000)
        payload = serialize(value)
        content_sha = hashlib.sha256(payload.encode()).hexdigest()
        with self._lock:
            standalone = self._batch_depth == 0
            if standalone:
                self._con.execute("BEGIN IMMEDIATE")
            try:
                self._con.execute("INSERT OR IGNORE INTO snapshots(id,kind,available_ms,payload_json,content_sha256) VALUES (?,?,?,?,?)",
                                  (identifier, kind, available, payload, content_sha))
                if standalone:
                    self._con.commit()
            except BaseException:
                if standalone:
                    self._con.rollback()
                raise
            stored = self._con.execute("SELECT payload_json FROM snapshots WHERE id=?",(identifier,)).fetchone()[0]
        return json.loads(stored)

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
        _integer(cutoff_ms)
        with self._lock:
            row = self._con.execute("SELECT payload_json FROM snapshots WHERE kind=? AND available_ms<=? ORDER BY seq DESC LIMIT 1",(kind,cutoff_ms)).fetchone()
            return json.loads(row[0]) if row else None

    @contextmanager
    def batch(self):
        with self._lock:
            if self._batch_depth:
                raise AdapterError("NESTED_MARKET_BATCH")
            self._con.execute("BEGIN IMMEDIATE")
            self._batch_depth = 1
            try:
                yield
                self._con.commit()
            except BaseException:
                self._con.rollback()
                self._quote_seq,self._quote_max_ms,self._quote_cache = 0,0,{}
                self._candle_seq,self._candle_max_ms,self._candle_cache = 0,0,{}
                self._failure_seq,self._failure_max_ms,self._failure_cache = 0,0,set()
                raise
            finally:
                self._batch_depth = 0

    def validate_snapshot(self, value, kind, cutoff_ms):
        try:
            identifier = value["snapshot_id"]
            with self._lock:
                row = self._con.execute("SELECT kind,available_ms,content_sha256 FROM snapshots WHERE id=?",(identifier,)).fetchone()
                payload = json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
                if row is None or row[0] != kind or hashlib.sha256(payload.encode()).hexdigest() != row[2]:
                    raise AdapterError("UNIVERSE_SNAPSHOT_INVALID")
                if row[1] > cutoff_ms:
                    raise AdapterError("UNIVERSE_NOT_AVAILABLE")
        except (KeyError,TypeError,ValueError,RecursionError):
            raise AdapterError("UNIVERSE_SNAPSHOT_INVALID") from None

    def validate_reference(self, identifier, kind, cutoff_ms):
        with self._lock:
            row = self._con.execute("SELECT kind,available_ms FROM snapshots WHERE id=?",(identifier,)).fetchone()
            if row is None or row[0] != kind:
                raise AdapterError("UNIVERSE_SNAPSHOT_INVALID")
            if row[1] > cutoff_ms:
                raise AdapterError("UNIVERSE_NOT_AVAILABLE")

    def quote_index(self, cutoff_ms):
        _integer(cutoff_ms)
        with self._lock:
            if self._batch_depth:
                # Do not publish/cache uncommitted groups.
                raise AdapterError("MARKET_BATCH_NOT_PUBLISHED")
            for row in self._con.execute("SELECT seq,available_ms,payload_json FROM snapshots WHERE kind='QUOTES' AND seq>? ORDER BY seq",(self._quote_seq,)):
                snapshot = json.loads(row[2])
                reference = {k:v for k,v in snapshot.items() if k != "data"}
                for item in snapshot["data"].get("rows",[]):
                    self._quote_cache[item["symbol"]] = (item,reference)
                self._quote_seq = row[0]
                self._quote_max_ms = max(self._quote_max_ms,row[1])
            if self._quote_max_ms <= cutoff_ms:
                result = dict(self._quote_cache)
            else:
                result = {}
                for snapshot in self.records("QUOTES",cutoff_ms=cutoff_ms):
                    reference = {k:v for k,v in snapshot.items() if k != "data"}
                    for item in snapshot["data"].get("rows",[]):
                        result[item["symbol"]] = (item,reference)
        return deepcopy(result)

    def candle_index(self, cutoff_ms, symbol=None):
        _integer(cutoff_ms)
        current = False
        with self._lock:
            if not self._batch_depth:
                for row in self._con.execute("SELECT seq,available_ms,payload_json FROM snapshots WHERE kind='CANDLES' AND seq>? ORDER BY seq",(self._candle_seq,)):
                    snapshot = json.loads(row[2])
                    reference = {k:v for k,v in snapshot.items() if k != "data"}
                    target = self._candle_cache.setdefault(snapshot["data"]["symbol"],{})
                    for item in snapshot["data"]["rows"]:
                        target[item["open_time_ms"]] = (item,reference)
                    self._candle_seq = row[0]
                    self._candle_max_ms = max(self._candle_max_ms,row[1])
                if self._candle_max_ms <= cutoff_ms:
                    result = ({symbol:dict(self._candle_cache.get(symbol,{}))} if symbol is not None
                        else {name:dict(rows) for name,rows in self._candle_cache.items()})
                    current = True
            if not current:
                result = {}
                for snapshot in self.records("CANDLES",cutoff_ms=cutoff_ms):
                    name = snapshot["data"]["symbol"]
                    if symbol is not None and name != symbol:
                        continue
                    reference = {k:v for k,v in snapshot.items() if k != "data"}
                    target = result.setdefault(name,{})
                    for item in snapshot["data"]["rows"]:
                        target[item["open_time_ms"]] = (item,reference)
        return deepcopy(result.get(symbol,{}) if symbol is not None else result)

    def sqlite_settings(self):
        return {key: self._con.execute("PRAGMA " + key).fetchone()[0]
                for key in ("journal_mode", "synchronous", "foreign_keys")}

    def candle_failures(self,cutoff_ms):
        _integer(cutoff_ms)
        with self._lock:
            if self._batch_depth:
                raise AdapterError("MARKET_BATCH_NOT_PUBLISHED")
            sql="SELECT seq,available_ms,kind,json_extract(payload_json,'$.data.symbol'),json_extract(payload_json,'$.data.operation') FROM snapshots WHERE kind IN ('CANDLES','INCIDENT')"
            def apply(rows,target):
                for seq,available,kind,symbol,operation in rows:
                    if kind=="CANDLES":target.discard(symbol)
                    elif operation=="CANDLE_COLLECTION_FAILED":target.add(symbol)
                    yield seq,available
            for seq,available in apply(self._con.execute(sql+" AND seq>? ORDER BY seq",(self._failure_seq,)),self._failure_cache):
                self._failure_seq=seq
                self._failure_max_ms=max(self._failure_max_ms,available)
            if self._failure_max_ms<=cutoff_ms:
                return set(self._failure_cache)
            target=set()
            list(apply(self._con.execute(sql+" AND available_ms<=? ORDER BY seq",(cutoff_ms,)),target))
            return target

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
