"""App-local immutable public snapshots; receipt-time, never event-time replay."""

import json
from . import market_sqlite as sqlite3
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


def _check_append(kind, data, wall, mono, dependencies):
    if kind not in KINDS or not isinstance(data, dict):
        raise AdapterError("SCHEMA_INVALID")
    _integer(wall); _integer(mono)
    if not isinstance(dependencies, (tuple, list)) or any(not isinstance(d, str) for d in dependencies):
        raise AdapterError("DEPENDENCY_MISSING")


def _prepared_row(identity, kind, data, wall, mono, dependencies, available):
    data_json = _json(data)
    basis = dict(identity, kind=kind, data=json.loads(data_json), dependencies=dependencies,
        event_type="MarketSnapshotRecorded", exchange_event_time_ms=None,
        received_at_ms=wall, received_monotonic_ns=mono, available_at_ms=available,
        process_instance_id=PROCESS_ID, source_sha256=hashlib.sha256(data_json.encode()).hexdigest(),
        causation_id=None, model_id=None, policy_version=None, arm_id=None, risk_contract_id=None)
    serialize = lambda value: json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
    identifier = hashlib.sha256(serialize(basis).encode()).hexdigest()
    value = dict(basis,event_id=identifier,snapshot_id=identifier,recorded_at_ms=time.time_ns()//1_000_000)
    payload = serialize(value)
    return value, (identifier,kind,available,payload,hashlib.sha256(payload.encode()).hexdigest())


class _PreparedMarketBatch:
    """Reader-private preparation; one atomic primary insert after validation."""
    def __init__(self, store, monotonic_ns):
        self.store = store
        self._now = monotonic_ns
        self._timing = dict(state="NOT_STARTED", prepare_start_ns=None, prepare_end_ns=None,
            writer_wait_started_ns=None, writer_acquired_ns=None, begin_finished_ns=None,
            insert_finished_ns=None, commit_finished_ns=None, failure_at_ns=None)
        self.rows, self.values, self.guards = [], {}, {}
        self._dependency_times = {}
        self._read_context = None
        self.active = False
        self.conflict_symbol = None

    def __enter__(self):
        if self.store._own_batch() or self.active:
            raise AdapterError("NESTED_MARKET_BATCH")
        self._timing.update(state="PREPARING", prepare_start_ns=self._now())
        self._read_context = self.store._reading()
        try:
            self._read_context.__enter__()
        except BaseException:
            self._timing.update(state="FAILED", failure_at_ns=self._now())
            raise
        self.active = True
        return self

    @property
    def timing(self): return dict(self._timing)

    def append(self, kind, data, *, received_at_ms, received_monotonic_ns, dependencies=()):
        if not self.active:
            raise AdapterError("INVALID_PARAMETERS")
        _check_append(kind,data,received_at_ms,received_monotonic_ns,dependencies)
        dependencies = sorted(set(dependencies))
        unknown = [dependency for dependency in dependencies
                   if dependency not in self.values and dependency not in self._dependency_times]
        if unknown:
            with self.store._reading() as connection:
                for dependency in unknown:
                    row = connection.execute("SELECT available_ms FROM snapshots WHERE id=?",(dependency,)).fetchone()
                    if row is None:
                        raise AdapterError("DEPENDENCY_MISSING")
                    self._dependency_times[dependency] = row[0]
        available = received_at_ms
        for dependency in dependencies:
            timestamp = (self.values[dependency]["available_at_ms"] if dependency in self.values
                         else self._dependency_times[dependency])
            available = max(available,timestamp)
        value,row = _prepared_row(self.store.identity,kind,data,received_at_ms,received_monotonic_ns,dependencies,available)
        identifier = value["snapshot_id"]
        existing = self.values.get(identifier) or self.store.get(identifier)
        if existing is not None:
            return deepcopy(existing)
        self.rows.append(row)
        self.values[identifier] = value
        return deepcopy(value)

    def candle_index(self, cutoff_ms, symbol=None):
        _integer(cutoff_ms)
        if symbol is None:
            raise AdapterError("INVALID_PARAMETERS")
        result = {}
        with self.store._reading() as connection:
            sequence = connection.execute("SELECT COALESCE(MAX(seq),0) FROM snapshots INDEXED BY candle_symbol_seq WHERE kind='CANDLES' AND json_extract(payload_json,'$.data.symbol')=?",(symbol,)).fetchone()[0]
            self.guards.setdefault(symbol,sequence)
            query = "SELECT payload_json FROM snapshots INDEXED BY candle_symbol_seq WHERE kind='CANDLES' AND json_extract(payload_json,'$.data.symbol')=? AND available_ms<=? ORDER BY seq"
            sources = [json.loads(row[0]) for row in connection.execute(query,(symbol,cutoff_ms))]
        sources.extend(value for value in self.values.values() if value["kind"]=="CANDLES"
            and value["data"]["symbol"]==symbol and value["available_at_ms"]<=cutoff_ms)
        for value in sources:
            reference = {key:item for key,item in value.items() if key!="data"}
            for item in value["data"]["rows"]:
                result[item["open_time_ms"]] = (item,reference)
        return deepcopy(result)

    def __exit__(self, error_type, error, traceback):
        self.active = False
        try:
            self._read_context.__exit__(error_type,error,traceback)
        except BaseException:
            self._timing.update(state="FAILED", prepare_end_ns=self._now(), failure_at_ns=self._now())
            raise
        self._timing["prepare_end_ns"] = self._now()
        if error_type is not None:
            self._timing.update(state="FAILED", failure_at_ns=self._now())
            return
        if not self.rows:
            self._timing["state"] = "NO_WRITE"
            return
        self._timing["writer_wait_started_ns"] = self._now()
        try:
            if self.store._group_writer is not None:
                self.store._group_writer.submit(self)
                return
            with self.store._lock:
                self._timing["writer_acquired_ns"] = self._now()
                with self.store.batch():
                    self._timing["begin_finished_ns"] = self._now()
                    for symbol,expected in self.guards.items():
                        actual = self.store._con.execute("SELECT COALESCE(MAX(seq),0) FROM snapshots INDEXED BY candle_symbol_seq WHERE kind='CANDLES' AND json_extract(payload_json,'$.data.symbol')=?",(symbol,)).fetchone()[0]
                        if actual != expected:
                            self.conflict_symbol = symbol
                            raise AdapterError("CANDLE_SOURCE_CHANGED")
                    for offset in range(0, len(self.rows), 64):
                        chunk = self.rows[offset:offset+64]
                        sql = "INSERT OR IGNORE INTO snapshots(id,kind,available_ms,payload_json,content_sha256) VALUES "
                        sql += ",".join("(?,?,?,?,?)" for _ in chunk)
                        self.store._con.execute(sql, tuple(value for row in chunk for value in row))
                    self._timing["insert_finished_ns"] = self._now()
                self._timing.update(state="COMMITTED", commit_finished_ns=self._now())
        except BaseException:
            self._timing.update(state="FAILED", failure_at_ns=self._now())
            raise


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
        self._checkpoint = None
        self._group_writer = None
        self._read_con = None
        self._reader_local = threading.local()
        self._reader_registry_lock = threading.Lock()
        self._readers = []
        self._index_lock = threading.RLock()
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
            self._con.execute("CREATE INDEX IF NOT EXISTS snapshot_kind_seq ON snapshots(kind,seq)")
            self._con.execute("CREATE INDEX IF NOT EXISTS candle_symbol_seq ON snapshots(kind,json_extract(payload_json,'$.data.symbol'),seq) WHERE kind='CANDLES'")
            self._read_con = self._thread_reader()["connection"]
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
        self._storage_guard()
        _check_append(kind,data,received_at_ms,received_monotonic_ns,dependencies)
        with self._lock:
            dependencies = sorted(set(dependencies))
            available = received_at_ms
            for dep in dependencies:
                row = self._con.execute("SELECT available_ms FROM snapshots WHERE id=?", (dep,)).fetchone()
                if row is None:
                    raise AdapterError("DEPENDENCY_MISSING")
                available = max(available, row[0])
        value,row = _prepared_row(self.identity,kind,data,received_at_ms,received_monotonic_ns,dependencies,available)
        identifier = value["snapshot_id"]
        if self._group_writer is not None and not self._own_batch():
            pending = _PreparedMarketBatch(self,time.monotonic_ns)
            pending.rows = [row]
            pending.values = {identifier:value}
            self._group_writer.submit(pending)
            standalone = True
        else:
            self._append_direct(row)
            standalone = not self._own_batch()
        return self._append_result(identifier,standalone)

    def _append_direct(self,row):
        with self._lock:
            self._storage_guard()
            standalone = self._batch_depth == 0
            if standalone:
                self._con.execute("BEGIN IMMEDIATE")
            try:
                self._con.execute("INSERT OR IGNORE INTO snapshots(id,kind,available_ms,payload_json,content_sha256) VALUES (?,?,?,?,?)",
                                  row)
                if standalone:
                    self._con.commit()
            except BaseException:
                if standalone:
                    self._con.rollback()
                raise

    def _append_result(self,identifier,standalone):
        # The committed row is immutable. Large result copies need no writer lock.
        # An explicit owner batch still routes get() through its staged connection.
        state = getattr(self._reader_local, "state", None)
        if standalone and state and state["depth"]:
            # Preserve the caller's older pinned prefix; this result is a fresh cut.
            connection = sqlite3.connect(self.path.as_uri()+"?mode=ro", uri=True)
            try:
                stored = connection.execute("SELECT payload_json FROM snapshots WHERE id=?", (identifier,)).fetchone()[0]
            finally:
                connection.close()
            return json.loads(stored)
        return self.get(identifier)

    def get(self, identifier):
        with self._reading() as connection:
            row = connection.execute("SELECT payload_json FROM snapshots WHERE id=?", (identifier,)).fetchone()
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
        with self._reading() as connection:
            return [json.loads(r[0]) for r in connection.execute(sql, args)]

    def as_of(self, kind, cutoff_ms):
        _integer(cutoff_ms)
        with self._reading() as connection:
            row = connection.execute("SELECT payload_json FROM snapshots WHERE kind=? AND available_ms<=? ORDER BY seq DESC LIMIT 1",(kind,cutoff_ms)).fetchone()
            return json.loads(row[0]) if row else None

    def _own_batch(self):
        return self._batch_depth and self._lock._owner == threading.get_ident()

    def prepared_batch(self, *, monotonic_ns=None):
        if monotonic_ns is not None and not callable(monotonic_ns):
            raise AdapterError("INVALID_PARAMETERS")
        return _PreparedMarketBatch(self, time.monotonic_ns if monotonic_ns is None else monotonic_ns)

    def start_group_writer(self,**options):
        from .market_group_writer import MarketGroupWriter
        if self._group_writer is None:self._group_writer=MarketGroupWriter(self,**options)
        elif options:raise AdapterError('INVALID_PARAMETERS')
        self._group_writer.start()
        return self._group_writer

    def stop_group_writer(self):
        if self._group_writer is not None:self._group_writer.stop()

    def group_writer_health(self):
        return self._group_writer.health() if self._group_writer is not None else dict(state='NOT_STARTED',running=False)

    def start_checkpointing(self):
        from .market_checkpoint import MarketCheckpointer
        if self._checkpoint is None: self._checkpoint = MarketCheckpointer(self)
        self._checkpoint.start()

    def stop_checkpointing(self):
        if self._checkpoint is not None: self._checkpoint.stop()

    def _storage_guard(self):
        if self._checkpoint and self._checkpoint.running:
            try:
                size = self._checkpoint.wal_bytes()
            except OSError:
                raise AdapterError("MARKET_STORAGE_UNAVAILABLE") from None
            if size>=self._checkpoint.HARD_BYTES:
                raise AdapterError("MARKET_STORAGE_BACKPRESSURE")

    def _thread_reader(self):
        state = getattr(self._reader_local,"state",None)
        if state is None:
            with self._reader_registry_lock:
                if self._con is None:
                    raise AdapterError("MARKET_INTEGRITY_FAILED")
                connection = sqlite3.connect(self.path.as_uri()+"?mode=ro",uri=True,
                    timeout=0,isolation_level=None,check_same_thread=False)
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA query_only=ON")
                state = dict(connection=connection,lock=threading.RLock(),depth=0)
                self._readers.append(state)
                self._reader_local.state = state
        return state

    def release_thread_reader(self):
        state = getattr(self._reader_local,"state",None)
        if state is None:
            return
        if state["depth"]:
            raise AdapterError("INVALID_PARAMETERS")
        with self._reader_registry_lock:
            if state in self._readers:
                with state["lock"]:
                    state["connection"].close()
                self._readers.remove(state)
            del self._reader_local.state

    @contextmanager
    def _indexed_reading(self):
        if self._own_batch():
            yield self._con
            return
        with self._index_lock:
            with self._reading() as connection:
                yield connection

    @contextmanager
    def _reading(self):
        if self._own_batch():
            # Validator sees its staged predecessors without a writer→reader lock.
            yield self._con
            return
        state = self._thread_reader()
        connection = state["connection"]
        with state["lock"]:
            if state["depth"] == 0:
                connection.execute("BEGIN")
                state["failed"] = False
            state["depth"] += 1
            try:
                yield connection
            except BaseException:
                state["failed"] = True
                raise
            finally:
                state["depth"] -= 1
                # Overlapping prepared contexts can release out of LIFO order.
                # The last readonly lease, rather than the first opener, closes.
                if state["depth"] == 0:
                    try:
                        if state["failed"]: connection.rollback()
                        else: connection.commit()
                    except BaseException:
                        connection.rollback()
                        raise

    @contextmanager
    def batch(self):
        self._storage_guard()
        with self._lock:
            self._storage_guard()
            if self._batch_depth:
                raise AdapterError("NESTED_MARKET_BATCH")
            self._con.execute("BEGIN IMMEDIATE")
            self._batch_depth = 1
            try:
                yield
                self._con.commit()
            except BaseException:
                self._con.rollback()
                # Diagnostic caches only read committed WAL rows; staged owner
                # reads never publish into them, so rollback cannot contaminate.
                raise
            finally:
                self._batch_depth = 0

    def validate_snapshot(self, value, kind, cutoff_ms):
        try:
            identifier = value["snapshot_id"]
            with self._reading() as connection:
                row = connection.execute("SELECT kind,available_ms,content_sha256 FROM snapshots WHERE id=?",(identifier,)).fetchone()
                payload = json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
                if row is None or row[0] != kind or hashlib.sha256(payload.encode()).hexdigest() != row[2]:
                    raise AdapterError("UNIVERSE_SNAPSHOT_INVALID")
                if row[1] > cutoff_ms:
                    raise AdapterError("UNIVERSE_NOT_AVAILABLE")
        except (KeyError,TypeError,ValueError,RecursionError):
            raise AdapterError("UNIVERSE_SNAPSHOT_INVALID") from None

    def validate_reference(self, identifier, kind, cutoff_ms):
        with self._reading() as connection:
            row = connection.execute("SELECT kind,available_ms FROM snapshots WHERE id=?",(identifier,)).fetchone()
            if row is None or row[0] != kind:
                raise AdapterError("UNIVERSE_SNAPSHOT_INVALID")
            if row[1] > cutoff_ms:
                raise AdapterError("UNIVERSE_NOT_AVAILABLE")

    def quote_index(self, cutoff_ms):
        _integer(cutoff_ms)
        with self._indexed_reading() as connection:
            if self._own_batch():
                # Do not publish/cache uncommitted groups.
                raise AdapterError("MARKET_BATCH_NOT_PUBLISHED")
            for row in connection.execute("SELECT seq,available_ms,payload_json FROM snapshots WHERE kind='QUOTES' AND seq>? ORDER BY seq",(self._quote_seq,)):
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
        with self._indexed_reading() as connection:
            if not self._own_batch():
                for row in connection.execute("SELECT seq,available_ms,payload_json FROM snapshots WHERE kind='CANDLES' AND seq>? ORDER BY seq",(self._candle_seq,)):
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
                table = "snapshots INDEXED BY candle_symbol_seq" if symbol is not None else "snapshots"
                sql = "SELECT payload_json FROM "+table+" WHERE kind='CANDLES' AND available_ms<=?"
                parameters = [cutoff_ms]
                if symbol is not None:
                    sql += " AND json_extract(payload_json,'$.data.symbol')=?"
                    parameters.append(symbol)
                for stored in connection.execute(sql+" ORDER BY seq",parameters):
                    snapshot = json.loads(stored[0])
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
        with self._indexed_reading() as connection:
            if self._own_batch():
                raise AdapterError("MARKET_BATCH_NOT_PUBLISHED")
            sql="SELECT seq,available_ms,kind,json_extract(payload_json,'$.data.symbol'),json_extract(payload_json,'$.data.operation') FROM snapshots WHERE kind IN ('CANDLES','INCIDENT')"
            def apply(rows,target):
                for seq,available,kind,symbol,operation in rows:
                    if kind=="CANDLES":target.discard(symbol)
                    elif operation=="CANDLE_COLLECTION_FAILED":target.add(symbol)
                    yield seq,available
            for seq,available in apply(connection.execute(sql+" AND seq>? ORDER BY seq",(self._failure_seq,)),self._failure_cache):
                self._failure_seq=seq
                self._failure_max_ms=max(self._failure_max_ms,available)
            if self._failure_max_ms<=cutoff_ms:
                return set(self._failure_cache)
            target=set()
            list(apply(connection.execute(sql+" AND available_ms<=? ORDER BY seq",(cutoff_ms,)),target))
            return target

    def last_failure(self, operation, success_kind, cutoff_ms, *, symbol=None):
        """Sequence ordering resolves equal clocks; future incidents are excluded."""
        with self._reading() as connection:
            # Do not deserialize a 50MB universe for every member's readiness.
            sql = "SELECT kind,CASE WHEN kind='INCIDENT' THEN json_extract(payload_json,'$.data.operation') END FROM snapshots WHERE available_ms<=? AND kind IN (?, 'INCIDENT')"
            args = [cutoff_ms, success_kind]
            if symbol is not None:
                sql += " AND json_extract(payload_json,'$.data.symbol')=?"
                args.append(symbol)
            sql += " ORDER BY seq DESC"
            for kind, recorded_operation in connection.execute(sql, args):
                if kind == success_kind:
                    return False
                if recorded_operation == operation:
                    return True
        return False

    def close(self):
        self.stop_group_writer()
        self.stop_checkpointing()
        with self._lock:
            with self._reader_registry_lock:
                for state in self._readers:
                    with state["lock"]:
                        state["connection"].close()
                self._readers.clear()
                self._read_con = None
            if self._con is not None:
                self._con.close()
                self._con = None
            self._lease.close()
