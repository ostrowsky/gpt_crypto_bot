"""Pinned SQLite for market storage only; financial journal keeps its engine."""
import sqlite3 as _stdlib
import apsw

Error = _stdlib.Error
DatabaseError = _stdlib.DatabaseError
OperationalError = _stdlib.OperationalError
IntegrityError = _stdlib.IntegrityError
ProgrammingError = _stdlib.ProgrammingError


class BusyError(OperationalError):
    """Keep contention distinct without breaking existing sqlite3 catches."""


sqlite_version = apsw.sqlitelibversion()
sqlite_version_info = tuple(map(int, sqlite_version.split(".")))
backend_version = apsw.apswversion()
SQLITE_OK, SQLITE_DENY, SQLITE_PRAGMA = apsw.SQLITE_OK, apsw.SQLITE_DENY, apsw.SQLITE_PRAGMA
SQLITE_READ, SQLITE_SELECT, SQLITE_FUNCTION = apsw.SQLITE_READ, apsw.SQLITE_SELECT, apsw.SQLITE_FUNCTION


def _call(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except apsw.ConstraintError as error:
        raise IntegrityError(str(error)) from None
    except apsw.BusyError as error:
        raise BusyError(str(error)) from None
    except apsw.Error as error:
        raise OperationalError(str(error)) from None


class Row:
    def __init__(self, cursor, values):
        self._values = tuple(values)
        self._names = [column[0] for column in cursor.description]
    def __getitem__(self, key):
        return self._values[self._names.index(key) if isinstance(key, str) else key]
    def __iter__(self): return iter(self._values)
    def __len__(self): return len(self._values)
    def keys(self): return list(self._names)


class Cursor:
    def __init__(self, connection, native):
        self.connection, self.native = connection, native
        try: self.description = native.get_description()
        except apsw.ExecutionCompleteError: self.description = ()
    def fetchone(self):
        self.connection._check()
        values = _call(self.native.fetchone)
        if values is None: return None
        factory = self.connection.row_factory
        return factory(self, values) if factory else values
    def fetchall(self): return list(self)
    def __iter__(self): return self
    def __next__(self):
        row = self.fetchone()
        if row is None: raise StopIteration
        return row


class Connection:
    def __init__(self, filename, *, timeout=0, isolation_level=None, check_same_thread=False, uri=False):
        if isolation_level is not None:
            raise ProgrammingError("Market adapter requires explicit transactions")
        flags = apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE
        if uri: flags |= apsw.SQLITE_OPEN_URI
        self.native = _call(apsw.Connection, str(filename), flags=flags)
        _call(self.native.set_busy_timeout, round(timeout*1000))
        self.row_factory, self.closed = None, False
    def _check(self):
        if self.closed: raise ProgrammingError("Connection is closed")
    @property
    def in_transaction(self):
        self._check()
        return not self.native.get_autocommit()
    def execute(self, statement, parameters=None):
        self._check()
        return Cursor(self, _call(self.native.execute, statement, parameters))
    def executemany(self, statement, parameters):
        self._check()
        return Cursor(self, _call(self.native.executemany, statement, parameters))
    def commit(self):
        if self.in_transaction: self.execute("COMMIT")
    def rollback(self):
        if self.in_transaction: self.execute("ROLLBACK")
    def close(self):
        if not self.closed:
            _call(self.native.close)
            self.closed = True
    def set_trace_callback(self, callback):
        self._check()
        def tracing(cursor, statement, bindings):
            callback(cursor.expanded_sql or statement)
            return True
        _call(self.native.set_exec_trace, tracing if callback else None)
    def set_authorizer(self, callback):
        self._check()
        _call(self.native.set_authorizer, callback)
    def passive_checkpoint(self):
        self._check()
        return _call(self.native.wal_checkpoint, "main", apsw.SQLITE_CHECKPOINT_PASSIVE)

    def cache_stats(self):
        """Connection-local diagnostics; caller owns writer, counters never reset."""
        self._check()
        codes = dict(hits_n=apsw.SQLITE_DBSTATUS_CACHE_HIT,
                     misses_n=apsw.SQLITE_DBSTATUS_CACHE_MISS,
                     writes_n=apsw.SQLITE_DBSTATUS_CACHE_WRITE,
                     spills_n=apsw.SQLITE_DBSTATUS_CACHE_SPILL,
                     used_bytes_n=apsw.SQLITE_DBSTATUS_CACHE_USED)
        return {name: _call(self.native.status, code, False)[0] for name, code in codes.items()}


def connect(filename, **options): return Connection(filename, **options)
