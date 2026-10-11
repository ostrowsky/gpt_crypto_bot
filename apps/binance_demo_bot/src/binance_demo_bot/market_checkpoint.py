"""Explicit owned PASSIVE maintenance, independent of the economic writer."""
import threading
import time
from collections import deque
from decimal import Decimal
from . import market_sqlite
from .adapter import AdapterError


class MarketCheckpointer:
    SOFT_BYTES = 4*1024*1024
    HARD_BYTES = 256*1024*1024

    def __init__(self, store):
        self.store, self.api = store, market_sqlite
        self._thread, self._connection, self._original = None, None, None
        self._stop, self._operation_lock, self._health_lock = threading.Event(), threading.Lock(), threading.RLock()
        self.state, self.error, self.running = "NOT_STARTED", None, False
        self.attempts, self.partial, self.failures, self.busy = 0, 0, 0, 0
        self.frames = (None,None)
        self.lags = deque(maxlen=4096)

    def wal_bytes(self):
        try: return self.store.path.with_name(self.store.path.name+"-wal").stat().st_size
        except FileNotFoundError: return 0

    def start(self):
        version = self.api.sqlite_version_info
        patched = version >= (3,51,3) or (3,50,7) <= version < (3,51,0) or (3,44,6) <= version < (3,45,0)
        if not patched: raise AdapterError("SQLITE_VERSION_UNSAFE")
        if self.store._own_batch(): raise AdapterError("NESTED_MARKET_BATCH")
        if self.running: raise AdapterError("WS_ALREADY_CONNECTED")
        with self.store._lock:
            settings = self.store.sqlite_settings()
            if settings["journal_mode"]!="wal" or settings["synchronous"]!=2:
                raise AdapterError("MARKET_INTEGRITY_FAILED")
            self._original = self.store._con.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
            self._connection = self.api.connect(self.store.path.as_uri()+"?mode=rw",uri=True)
            try:
                self._connection.execute("PRAGMA synchronous=FULL")
                self._connection.execute("PRAGMA wal_autocheckpoint=0")
                allowed = {"journal_mode","synchronous","wal_autocheckpoint","busy_timeout"}
                def authorize(action, name, argument, database, source):
                    permitted = (action==self.api.SQLITE_PRAGMA and
                        (name in allowed and argument is None or
                         name=="wal_checkpoint" and (argument is None or argument.upper()=="PASSIVE")))
                    return self.api.SQLITE_OK if permitted else self.api.SQLITE_DENY
                self._connection.set_authorizer(authorize)
                self.store._con.execute("PRAGMA wal_autocheckpoint=0")
                self.store._checkpoint = self
                self._stop.clear()
                self.running, self.state, self.error = True, "READY", None
                self._thread = threading.Thread(target=self._run,name="binance-demo-market-checkpoint",daemon=True)
                self._thread.start()
            except BaseException:
                self.running, self.state, self.error = False, "CHECKPOINT_FAILED", "CHECKPOINT_START_FAILED"
                self._thread = None
                try:
                    self.store._con.execute("PRAGMA wal_autocheckpoint="+str(self._original))
                finally:
                    self._connection.close();self._connection=None
                raise

    def _passive(self): return self._connection.passive_checkpoint()

    def run_once(self):
        with self._operation_lock:
            if not self.running: return
            started = time.monotonic_ns()
            try:
                frames = self._passive()
                with self._health_lock:
                    self.frames = frames
                    self.state = "READY" if frames[0]>=0 and frames[0]==frames[1] else "PARTIAL"
                    self.error = None
                    self.partial += self.state=="PARTIAL"
            except self.api.BusyError:
                with self._health_lock: self.busy += 1
                self._failure("CHECKPOINT_UNAVAILABLE")
            except Exception:
                self._failure("CHECKPOINT_UNAVAILABLE")
            finally:
                with self._health_lock:
                    self.attempts += 1
                    self.lags.append(Decimal(time.monotonic_ns()-started)/1_000_000)

    def _run(self):
        while not self._stop.wait(1):
            try:
                if self.wal_bytes()>=self.SOFT_BYTES: self.run_once()
            except Exception:
                self._failure("WAL_SIZE_UNAVAILABLE")

    def _failure(self, reason):
        with self._health_lock:
            self.state, self.error = "CHECKPOINT_FAILED", reason
            self.failures += 1

    def health(self):
        with self._health_lock:
            try:
                size = self.wal_bytes()
                state = "STORAGE_BACKPRESSURE" if size>=self.HARD_BYTES else self.state
                error = self.error
            except OSError:
                size, state, error = None, "CHECKPOINT_FAILED", "WAL_SIZE_UNAVAILABLE"
            return dict(running=self.running,state=state,
                error=error,sqlite_version=self.api.sqlite_version,backend_version=self.api.backend_version,
                wal_physical_bytes_n=size,wal_hard_limit_bytes_N=self.HARD_BYTES,wal_soft_limit_bytes_N=self.SOFT_BYTES,
                attempts_n=self.attempts,partial_n=self.partial,failures_n=self.failures,busy_n=self.busy,
                log_frames_n=self.frames[0],checkpointed_frames_n=self.frames[1],
                lag_recent_n=len(self.lags),lag_window_limit_N=4096,max_ms=max(self.lags) if self.lags else None,
                maintenance_only=True,order_send_enabled=False)

    def stop(self):
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(10)
            if self._thread.is_alive(): raise AdapterError("CHECKPOINT_STOP_TIMEOUT")
        with self._operation_lock:
            if self._connection is not None:
                with self.store._lock:
                    self.store._con.execute("PRAGMA wal_autocheckpoint="+str(self._original))
                self._connection.close();self._connection=None
            self.running=False
            if self.state!="NOT_STARTED": self.state="STOPPED"
