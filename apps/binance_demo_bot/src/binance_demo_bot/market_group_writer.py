"""Bounded store-owned market coalescer; ACK follows real FULL commit."""
from collections import deque
import threading
import time
from .adapter import AdapterError
from . import market_sqlite as sqlite3


class _Request:
    def __init__(self,pending):
        self.pending=pending
        self.rows=tuple(tuple(row) for row in pending.rows)
        self.guards=dict(pending.guards)
        self.bytes=sum(len(row[3].encode('utf-8')) for row in self.rows)
        self.done,self.error=threading.Event(),None


class MarketGroupWriter:
    def __init__(self,store,*,capacity=8,byte_limit=64*1024*1024,collect_ms=2):
        if (type(capacity) is not int or not 1<=capacity<=8
                or type(byte_limit) is not int or not 1024<=byte_limit<=64*1024*1024
                or type(collect_ms) is not int or not 0<=collect_ms<=2):
            raise AdapterError('INVALID_PARAMETERS')
        self.store,self.capacity,self.byte_limit,self.collect_ms=store,capacity,byte_limit,collect_ms
        self._condition=threading.Condition(threading.RLock())
        self._queue,self._pending=deque(),set()
        self._bytes=0
        self._thread=None
        self.accepting,self.running=False,False
        self.state='NOT_STARTED'
        self.submitted=self.accepted=self.rejected=self.committed=self.failed=0
        self.commit_acks=self.fallbacks=self.unconfirmed=0

    def start(self):
        with self._condition:
            if self.state=='FAILED':raise AdapterError('MARKET_WRITER_UNAVAILABLE')
            if self.running:return
            self.accepting,self.running,self.state=True,True,'READY'
            self._thread=threading.Thread(target=self._run,name='binance-demo-market-writer',daemon=True)
            try:self._thread.start()
            except BaseException:
                self.accepting,self.running,self.state=False,False,'FAILED'
                self._thread=None
                raise AdapterError('MARKET_WRITER_UNAVAILABLE') from None

    def submit(self,pending):
        request=_Request(pending)
        pending._timing.update(writer_wait_started_ns=pending._now(),writer_attempts_n=0,
            coalesced_groups_N=0,unconfirmed=False)
        with self._condition:
            self.submitted+=1
            if not self.accepting:
                self.rejected+=1
                raise AdapterError('MARKET_WRITER_UNAVAILABLE')
            if (len(self._pending)>=self.capacity or len(request.rows)>256
                    or self._bytes+request.bytes>self.byte_limit):
                self.rejected+=1
                raise AdapterError('MARKET_WRITER_BACKPRESSURE')
            self.accepted+=1
            self._pending.add(request);self._bytes+=request.bytes
            self._queue.append(request);self._condition.notify_all()
        request.done.wait()
        if request.error is not None:raise request.error

    def _take_batch(self):
        # Called after FIFO writer acquisition: collect arrivals already waiting
        # behind that writer, rather than fixing a batch before a long lock wait.
        deadline=time.monotonic_ns()+self.collect_ms*1_000_000
        groups=[];rows=size=0
        with self._condition:
            while len(groups)<8:
                if self._queue:
                    head=self._queue[0]
                    if groups and (rows+len(head.rows)>64 or size+head.bytes>4*1024*1024):break
                    groups.append(self._queue.popleft());rows+=len(head.rows);size+=head.bytes
                    if rows>=64 or size>=4*1024*1024:break
                elif not self.accepting:break
                remaining=(deadline-time.monotonic_ns())/1e9
                if remaining<=0:break
                if not self._queue:self._condition.wait(remaining)
        return groups

    def _insert(self,rows):
        for offset in range(0,len(rows),64):
            chunk=rows[offset:offset+64]
            sql='INSERT OR IGNORE INTO snapshots(id,kind,available_ms,payload_json,content_sha256) VALUES '
            sql+=','.join('(?,?,?,?,?)' for _ in chunk)
            self.store._con.execute(sql,tuple(value for row in chunk for value in row))

    def _guards(self,request):
        for symbol,expected in request.guards.items():
            actual=self.store._con.execute("SELECT COALESCE(MAX(seq),0) FROM snapshots INDEXED BY candle_symbol_seq WHERE kind='CANDLES' AND json_extract(payload_json,'$.data.symbol')=?",(symbol,)).fetchone()[0]
            if actual!=expected:
                request.pending.conflict_symbol=symbol
                raise AdapterError('CANDLE_SOURCE_CHANGED')

    def _perform(self,groups):
        errors={};phase='BEGIN'
        for request in groups:
            pending=request.pending
            pending._timing.update(writer_acquired_ns=pending._now(),
                writer_attempts_n=pending._timing['writer_attempts_n']+1,coalesced_groups_N=len(groups))
        try:
            with self.store.batch():
                for request in groups:request.pending._timing['begin_finished_ns']=request.pending._now()
                phase='INSERT'
                if any(request.guards for request in groups):
                    for request in groups:
                        self.store._con.execute('SAVEPOINT market_group')
                        try:
                            self._guards(request);self._insert(request.rows)
                        except (AdapterError,sqlite3.IntegrityError) as error:
                            self.store._con.execute('ROLLBACK TO market_group')
                            errors[request]=(error if isinstance(error,AdapterError) else AdapterError('MARKET_WRITER_FAILED'))
                        finally:self.store._con.execute('RELEASE market_group')
                else:self._insert(tuple(row for request in groups for row in request.rows))
                for request in groups:
                    if request not in errors:request.pending._timing['insert_finished_ns']=request.pending._now()
                phase='COMMIT'
            with self._condition:self.commit_acks+=1
            for request in groups:
                if request not in errors:
                    request.pending._timing.update(state='COMMITTED',commit_finished_ns=request.pending._now())
            return errors
        except BaseException as error:
            if phase=='COMMIT':
                for request in groups:
                    if request not in errors:request.pending._timing['unconfirmed']=True
                raise AdapterError('MARKET_WRITER_COMMIT_UNKNOWN') from None
            if isinstance(error,sqlite3.IntegrityError) and phase=='INSERT' and not self.store._con.in_transaction:
                raise
            raise AdapterError(error.reason if isinstance(error,AdapterError) else 'MARKET_WRITER_FAILED') from None

    def _execute(self,groups):
        try:return self._perform(groups)
        except sqlite3.IntegrityError:
            # Only a confirmed pre-commit constraint rollback may be isolated.
            with self._condition:self.fallbacks+=1
            errors={}
            for request in groups:
                try:errors.update(self._perform([request]))
                except sqlite3.IntegrityError:errors[request]=AdapterError('MARKET_WRITER_FAILED')
                self._finish(request,errors.get(request))
            return errors

    def _finish(self,request,error=None):
        pending=request.pending
        with self._condition:
            if request not in self._pending:return
            if error is not None:
                try:failed_at=pending._now()
                except BaseException:failed_at=None
                pending._timing.update(state='FAILED',failure_at_ns=failed_at)
                self.failed+=1
                self.unconfirmed+=bool(pending._timing.get('unconfirmed'))
            else:self.committed+=1
            request.error=error
            self._pending.remove(request);self._bytes-=request.bytes
            self._condition.notify_all()
        request.done.set()

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda:self._queue or not self.accepting)
                    if not self._queue:break
                with self.store._lock:
                    groups=self._take_batch()
                    errors=self._execute(groups)
                for request in groups:self._finish(request,errors.get(request))
        except BaseException as error:
            with self._condition:
                self.accepting=False;self.state='FAILED'
                requests=list(self._pending);self._queue.clear()
            for request in requests:
                reason=('MARKET_WRITER_COMMIT_UNKNOWN' if request.pending._timing.get('unconfirmed') else 'MARKET_WRITER_UNAVAILABLE')
                self._finish(request,AdapterError(reason))
        finally:
            with self._condition:
                self.running=False;self.accepting=False
                if self.state!='FAILED':self.state='STOPPED'
                self._condition.notify_all()

    def stop(self):
        with self._condition:
            self.accepting=False
            if self.running and self.state!='FAILED':self.state='DRAINING'
            self._condition.notify_all()
            worker=self._thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(20)
            if worker.is_alive():raise AdapterError('MARKET_WRITER_STOP_TIMEOUT')

    def health(self):
        with self._condition:
            return dict(scope='COALESCER_SUBMISSIONS',state=self.state,running=self.running,
                accepting=self.accepting,pending_groups_n=len(self._pending),pending_groups_limit_N=self.capacity,
                pending_bytes_n=self._bytes,pending_bytes_limit_N=self.byte_limit,
                request_rows_limit_N=256,batch_rows_limit_N=64,batch_bytes_limit_N=4*1024*1024,
                collect_ms_limit_N=self.collect_ms,submitted_groups_n=self.submitted,accepted_groups_n=self.accepted,
                rejected_groups_n=self.rejected,committed_groups_n=self.committed,failed_groups_n=self.failed,
                commit_acks_n=self.commit_acks,constraint_fallbacks_n=self.fallbacks,unconfirmed_groups_n=self.unconfirmed)
