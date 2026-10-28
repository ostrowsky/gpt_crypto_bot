"""Resumable maximum-available native15m archive; technical reference only."""
import argparse,hashlib,json,sqlite3,threading,time,zlib
from contextlib import closing
from pathlib import Path
from .adapter import _sha,DemoReadClient,AdapterError
from .execution_pilot import fail
from .isolation import AppPaths
from .journal import _WriterLease
from .interaction_audit import AuditJournal


class HistoryArchive:
    def __init__(self,paths,*,cutoff_ms):
        self.path=paths.runtime_path('all-usdt/history.sqlite');self.path.parent.mkdir(parents=True,exist_ok=True)
        self._lease=_WriterLease(paths.runtime_path('all-usdt/history.sqlite.writer.lock'))
        self._con=None;self._lock=threading.RLock()
        try:
            self._con=sqlite3.connect(self.path,timeout=5,isolation_level=None,check_same_thread=False)
            self._con.execute('PRAGMA journal_mode=WAL');self._con.execute('PRAGMA synchronous=FULL')
            self._con.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            self._con.execute('CREATE TABLE IF NOT EXISTS pages (symbol TEXT NOT NULL,start_ms INTEGER NOT NULL,rows_n INTEGER NOT NULL,first_ms INTEGER,last_ms INTEGER,source_sha TEXT NOT NULL,blob_sha TEXT NOT NULL,body BLOB NOT NULL,received_ms INTEGER NOT NULL,PRIMARY KEY(symbol,start_ms))')
            self._con.execute('CREATE TABLE IF NOT EXISTS refs (symbol TEXT PRIMARY KEY,manifest_sha TEXT NOT NULL,payload TEXT NOT NULL)')
            row=self._con.execute("SELECT value FROM meta WHERE key='cutoff_ms'").fetchone()
            if row and int(row[0])!=cutoff_ms:fail('USDT_HISTORY_BINDING')
            if type(cutoff_ms) is not int or cutoff_ms<=0 or cutoff_ms%900000:fail('USDT_HISTORY_BINDING')
            if row is None:self._con.execute("INSERT INTO meta VALUES ('cutoff_ms',?)",(str(cutoff_ms),))
            self.cutoff_ms=cutoff_ms
        except BaseException:self.close();raise
    def close(self):
        if self._con is not None:self._con.close();self._con=None
        self._lease.close()
    def _manifest(self,symbol):
        rows=self._con.execute('SELECT start_ms,rows_n,first_ms,last_ms,source_sha,blob_sha FROM pages WHERE symbol=? ORDER BY start_ms',(symbol,)).fetchall()
        count=sum(r[1] for r in rows);last=max((r[3] for r in rows if r[1]),default=None)
        end=any(r[1]==0 for r in rows) or (last is not None and last+900000>=self.cutoff_ms)
        state='MAXIMUM_AVAILABLE_FETCH_COMPLETE' if end and count else 'NO_HISTORY' if end else 'COLLECTING'
        value=dict(symbol=symbol,cutoff_ms=self.cutoff_ms,rows_n=count,pages_n=len(rows),
            first_open_ms=min((r[2] for r in rows if r[1]),default=None),last_open_ms=last,state=state,
            source_pages=[list(r) for r in rows])
        value['manifest_sha256']=_sha(value);return value
    def summary(self,symbol):
        with self._lock:return self._manifest(symbol)
    def next_start(self,symbol):
        with self._lock:
            last=self._manifest(symbol)['last_open_ms'];return last+900000 if last is not None else 0
    def append_page(self,symbol,rows,*,requested_start_ms,received_at_ms,source_sha256):
        with self._lock:
            if not isinstance(rows,list) or len(rows)>1000 or _sha(rows)!=source_sha256:fail('USDT_HISTORY_SOURCE')
            old=self._con.execute('SELECT source_sha FROM pages WHERE symbol=? AND start_ms=?',(symbol,requested_start_ms)).fetchone()
            if old:
                if old[0]!=source_sha256:fail('USDT_HISTORY_CONFLICT')
                return self._manifest(symbol)
            if requested_start_ms!=self.next_start(symbol) or self._manifest(symbol)['state']!='COLLECTING':fail('USDT_HISTORY_CURSOR')
            previous=None
            for row in rows:
                if (not isinstance(row,list) or len(row)<12 or type(row[0]) is not int or type(row[6]) is not int
                    or row[0]<requested_start_ms or not row[0]<=row[6]<=row[0]+899999 or row[6]>=self.cutoff_ms
                    or received_at_ms<row[6] or (previous is not None and row[0]<=previous)):fail('USDT_HISTORY_PAGE')
                previous=row[0]
            body=zlib.compress(json.dumps(rows,separators=(',',':')).encode(),6)
            self._con.execute('INSERT INTO pages VALUES (?,?,?,?,?,?,?,?,?)',(symbol,requested_start_ms,len(rows),
                rows[0][0] if rows else None,rows[-1][0] if rows else None,source_sha256,hashlib.sha256(body).hexdigest(),body,received_at_ms))
            return self._manifest(symbol)
    def bars(self,symbol):
        with self._lock:
            result=[]
            for digest,compressed_sha,blob in self._con.execute('SELECT source_sha,blob_sha,body FROM pages WHERE symbol=? ORDER BY start_ms',(symbol,)):
                if hashlib.sha256(blob).hexdigest()!=compressed_sha:fail('USDT_HISTORY_INTEGRITY')
                try:rows=json.loads(zlib.decompress(blob))
                except (ValueError,zlib.error):fail('USDT_HISTORY_INTEGRITY')
                if _sha(rows)!=digest:fail('USDT_HISTORY_INTEGRITY')
                result.extend(rows)
            return result
    def record_reference(self,symbol,result):
        with self._lock:
            manifest=self._manifest(symbol)
            if (manifest['state']!='MAXIMUM_AVAILABLE_FETCH_COMPLETE' or result.get('evidence_kind')!='SIMULATION'
                or result.get('alpha_qualified') is not False or result.get('verdict')!='NOT_ALPHA_EVIDENCE'):fail('USDT_HISTORY_NOT_READY')
            self.bars(symbol)
            payload=json.dumps(result,sort_keys=True,separators=(',',':'))
            old=self._con.execute('SELECT manifest_sha,payload FROM refs WHERE symbol=?',(symbol,)).fetchone()
            if old:
                if old!=(manifest['manifest_sha256'],payload):fail('USDT_REFERENCE_CONFLICT')
            else:self._con.execute('INSERT INTO refs VALUES (?,?,?)',(symbol,manifest['manifest_sha256'],payload))
    def ready_symbols(self):
        with self._lock:
            return sorted(s for s,h in self._con.execute('SELECT symbol,manifest_sha FROM refs').fetchall()
                if self._manifest(s)['manifest_sha256']==h)


def read_ready(paths):
    path=paths.runtime_path('all-usdt/history.sqlite')
    if not path.exists():return []
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5)) as con:
        result=[]
        for symbol,manifest_sha,payload in con.execute('SELECT symbol,manifest_sha,payload FROM refs ORDER BY symbol'):
            reference=json.loads(payload)
            if reference.get('evidence_kind')=='SIMULATION' and reference.get('alpha_qualified') is False and reference.get('verdict')=='NOT_ALPHA_EVIDENCE':
                result.append(symbol)
        return result


def verify_reference(paths,symbol):
    """Read-only full selected-history verification before any expanded BUY."""
    path=paths.runtime_path('all-usdt/history.sqlite')
    if not path.exists():fail('USDT_HISTORY_NOT_READY')
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5)) as con:
        ref=con.execute('SELECT manifest_sha,payload FROM refs WHERE symbol=?',(symbol,)).fetchone()
        if ref is None:fail('USDT_HISTORY_NOT_READY')
        cutoff=int(con.execute("SELECT value FROM meta WHERE key='cutoff_ms'").fetchone()[0])
        rows=con.execute('SELECT start_ms,rows_n,first_ms,last_ms,source_sha,blob_sha,body FROM pages WHERE symbol=? ORDER BY start_ms',(symbol,)).fetchall()
        for r in rows:
            if hashlib.sha256(r[6]).hexdigest()!=r[5]:fail('USDT_HISTORY_INTEGRITY')
            try:page=json.loads(zlib.decompress(r[6]))
            except (ValueError,zlib.error):fail('USDT_HISTORY_INTEGRITY')
            if _sha(page)!=r[4] or len(page)!=r[1]:fail('USDT_HISTORY_INTEGRITY')
        count=sum(r[1] for r in rows);last=max((r[3] for r in rows if r[1]),default=None)
        end=any(r[1]==0 for r in rows) or (last is not None and last+900000>=cutoff)
        manifest=dict(symbol=symbol,cutoff_ms=cutoff,rows_n=count,pages_n=len(rows),
            first_open_ms=min((r[2] for r in rows if r[1]),default=None),last_open_ms=last,
            state='MAXIMUM_AVAILABLE_FETCH_COMPLETE' if end and count else 'NO_HISTORY' if end else 'COLLECTING',
            source_pages=[list(r[:6]) for r in rows])
        manifest['manifest_sha256']=_sha(manifest);reference=json.loads(ref[1])
        if (manifest['manifest_sha256']!=ref[0] or manifest['state']!='MAXIMUM_AVAILABLE_FETCH_COMPLETE'
            or count<3 or reference.get('evidence_kind')!='SIMULATION' or reference.get('alpha_qualified') is not False
            or reference.get('verdict')!='NOT_ALPHA_EVIDENCE'):fail('USDT_HISTORY_NOT_READY')
        return manifest


def main():
    from concurrent.futures import ThreadPoolExecutor
    from .pilot_research import reference_replay
    parser=argparse.ArgumentParser();parser.add_argument('--collect',action='store_true');parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if not args.collect or not 1<=args.workers<=4:raise SystemExit('EXPLICIT_BOUNDED_COLLECTION_REQUIRED')
    paths=AppPaths(Path(__file__).resolve().parents[2]);folder=paths.runtime_path('all-usdt')
    catalog=json.loads((folder/'catalog.json').read_text());symbols=sorted(r['symbol'] for r in catalog if r['monitored'])
    registration=folder/'history-registration.json'
    if registration.exists():registered=json.loads(registration.read_text())
    else:
        registered=dict(symbols=symbols,cutoff_ms=(time.time_ns()//1000000)//900000*900000,
            policy='technical-round-robin-v1',cohort_sha256=_sha(symbols),ordinary_strategy_enabled=False)
        registration.write_text(json.dumps(registered,indent=2))
    if registered['symbols']!=symbols:raise SystemExit('HISTORY_COHORT_CHANGED')
    archive=HistoryArchive(paths,cutoff_ms=registered['cutoff_ms']);rate_lock=threading.Lock();next_request=[0.]
    status_lock=threading.Lock();completed=set(archive.ready_symbols());failures={}
    def status():
        with status_lock:
            result=dict(state='COMPLETE' if len(completed)==len(symbols) else 'COLLECTING',
                symbols_N=len(symbols),validated_n=len(completed),failed_n=len(failures),failures=failures,
                checked_at_ms=time.time_ns()//1000000,cohort_sha256=registered['cohort_sha256'],alpha_qualified=False)
            temp=folder/'history-status.next';temp.write_text(json.dumps(result,indent=2));temp.replace(folder/'history-status.json')
    def collect(symbol):
        if symbol in completed:return
        ns=dict(account_alias='demo-primary',episode_id='opening-candidate-20261027',scope_id='whole-account-opening')
        audit=AuditJournal(paths,**ns,database='all-usdt/history-audit-'+symbol+'.sqlite',max_bytes=1024*1024*1024)
        client=DemoReadClient(audit=audit)
        try:
            while archive.summary(symbol)['state']=='COLLECTING':
                with rate_lock:
                    delay=max(0,next_request[0]-time.monotonic());next_request[0]=max(next_request[0],time.monotonic())+.3
                if delay:time.sleep(delay)
                start=archive.next_start(symbol)
                try:
                    rows,meta=client.request_public('/api/v3/klines',dict(symbol=symbol,interval='15m',startTime=start,
                        endTime=registered['cutoff_ms']-1,limit=1000),weight=2)
                except AdapterError as error:
                    if error.reason in ('LOCAL_RATE_BUDGET','RATE_LIMITED','READ_UNAVAILABLE','CLOCK_STALE'):
                        time.sleep(60);continue
                    raise
                archive.append_page(symbol,rows,requested_start_ms=start,received_at_ms=meta['received_at_ms'],source_sha256=_sha(rows))
            if archive.summary(symbol)['state']=='NO_HISTORY':fail('USDT_NO_HISTORY')
            bars=archive.bars(symbol);clean=[r for r in bars if r[6]==r[0]+899999]
            reference=reference_replay(clean);reference.pop('rows',None)
            reference.update(baseline='USDT_CASH',cash_baseline_net_usdt='0',capital_convention='INDEPENDENT_REFERENCE_CYCLES',
                historical_membership='CURRENT_UNIVERSE_SURVIVOR_REFERENCE',promotion_allowed=False,
                raw_history_rows_n=len(bars),quarantined_irregular_bars_n=len(bars)-len(clean))
            archive.record_reference(symbol,reference)
            with status_lock:completed.add(symbol)
        except Exception as error:
            with status_lock:failures[symbol]=error.reason if isinstance(error,AdapterError) else type(error).__name__
        finally:client.close();audit.close();status()
    try:
        status()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:list(executor.map(collect,symbols))
        status()
    finally:archive.close()


if __name__=='__main__':main()
