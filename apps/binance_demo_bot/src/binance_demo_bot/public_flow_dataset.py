"""Finite, credential-free Demo public flow capture. No trading dependencies."""
import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import ssl
import time
import uuid
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

ORIGIN = 'wss://demo-stream.binance.com/stream'
SYMBOLS = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')
STREAMS = tuple(sorted(s.lower()+suffix for s in SYMBOLS
                       for suffix in ('@aggTrade', '@depth20@100ms')))
URL = ORIGIN + '?streams=' + '/'.join(STREAMS)
DOC_URL = 'https://github.com/binance/binance-spot-api-docs/blob/master/demo-mode/general-info.md'
RAW_DOC_URL = 'https://raw.githubusercontent.com/binance/binance-spot-api-docs/master/demo-mode/general-info.md'
MAX_COMPRESSED_BYTES = 8 * 1024**3
MINIMUM_FREE_BYTES = 20 * 1024**3
CHUNK_BYTES = 8 * 1024**2
CHUNK_NS = 60 * 1_000_000_000
MAX_FRAME_BYTES = 1024**2
STALE_RECEIPT_MS = 5000


class FlowError(Exception):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _positive(value):
    try:
        number = Decimal(value) if isinstance(value, str) else None
        return number if number is not None and number.is_finite() and number > 0 else None
    except (InvalidOperation, ValueError):
        return None


def verify_official_host(*, fetch=None):
    """Read the primary endpoint table before opening a public market connection."""
    if fetch is None:
        from urllib.request import Request, urlopen
        def fetch():
            request = Request(RAW_DOC_URL, headers={'User-Agent': 'public-demo-flow/1'})
            with urlopen(request, timeout=10, context=ssl.create_default_context()) as response:
                if urlsplit(response.geturl()).hostname != 'raw.githubusercontent.com':
                    raise FlowError('OFFICIAL_HOST_NOT_VERIFIED')
                return response.read(256*1024+1)
    try:
        body = fetch()
        if not isinstance(body, bytes) or len(body) > 256*1024 or ORIGIN.encode() not in body:
            raise FlowError('OFFICIAL_HOST_NOT_VERIFIED')
    except (OSError, ValueError):
        raise FlowError('OFFICIAL_HOST_NOT_VERIFIED') from None
    return dict(origin=ORIGIN, url=RAW_DOC_URL, sha256=hashlib.sha256(body).hexdigest(),
                fetched_at_ms=time.time_ns()//1_000_000)


def make_registration(started_at_ms, *, documentation_sha256=None):
    if not _integer(started_at_ms, 1):
        raise FlowError('INVALID_START')
    zone = ZoneInfo('Europe/Budapest')
    local = datetime.fromtimestamp(started_at_ms / 1000, timezone.utc).astimezone(zone)
    first = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), zone)
    last = first + timedelta(days=7)
    return dict(schema_version=1, dataset='public-demo-flow-v1', origin=ORIGIN,
                streams=list(STREAMS), symbols=list(SYMBOLS), started_at_ms=started_at_ms,
                full_days_start_ms=int(first.timestamp()*1000), planned_end_ms=int(last.timestamp()*1000),
                full_calendar_days=7, timezone='Europe/Budapest',
                max_compressed_bytes=MAX_COMPRESSED_BYTES, minimum_free_bytes=MINIMUM_FREE_BYTES,
                max_chunk_uncompressed_bytes=CHUNK_BYTES, max_chunk_seconds=60,
                stale_receipt_ms=STALE_RECEIPT_MS, official_demo_documentation=DOC_URL,
                host_verified_date='2026-10-29',
                official_documentation_sha256=documentation_sha256,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                evidence_kind='PUBLIC_DEMO_OBSERVATION', credentials=False, orders=False,
                promotion_allowed=False)


def _validate_registration(registration):
    try:
        expected = make_registration(registration['started_at_ms'],
            documentation_sha256=registration.get('official_documentation_sha256'))
        # Reading historical evidence does not require the current source version.
        digest = registration['source_sha256']
        expected['source_sha256'] = digest
        if (registration != expected or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise FlowError('REGISTRATION_INVALID') from None


def stop_reason(registration, *, now_ms, compressed_bytes, free_bytes, stop_requested):
    if stop_requested:
        return 'OWN_STOP'
    if now_ms >= registration['planned_end_ms']:
        return 'PLANNED_END'
    if compressed_bytes >= registration['max_compressed_bytes']:
        return 'COMPRESSED_BUDGET'
    if free_bytes < registration['minimum_free_bytes']:
        return 'DISK_RESERVE'
    return None


def own_runtime_path(app_root, output=None):
    base = (Path(app_root) / '.runtime').resolve()
    target = Path(output).resolve() if output is not None else base / 'public-flow'
    if target == base or not target.is_relative_to(base):
        raise FlowError('OUTPUT_OUTSIDE_OWN_RUNTIME')
    return target


class FlowValidator:
    """Validate observed frames; a gap never establishes absence of trades."""
    def __init__(self):
        self.session_id = None
        self.sessions_n = 0
        self.last_wall = None
        self.last_mono = None
        self.by_topic = {}
        self.counts = {s: 0 for s in STREAMS}
        self.first_receipt = None
        self.last_receipt = None
        self.gaps_n = self.conflicts_n = self.invalid_n = self.duplicates_n = 0

    def begin_session(self, session_id, *, wall_ms, mono_ns):
        if not isinstance(session_id, str) or not session_id or session_id == self.session_id:
            raise FlowError('INVALID_SESSION')
        if not _integer(wall_ms) or not _integer(mono_ns):
            raise FlowError('INVALID_RECEIPT')
        if ((self.last_wall is not None and wall_ms < self.last_wall)
                or (self.last_mono is not None and mono_ns < self.last_mono)):
            raise FlowError('RECEIPT_CLOCK_REGRESSION')
        self.last_wall, self.last_mono = wall_ms, mono_ns
        if self.sessions_n:
            self.gaps_n += 1
        self.session_id = session_id
        self.sessions_n += 1
        self.by_topic = {}
        return dict(kind='SESSION_START', session_id=session_id, wall_ms=wall_ms,
                    mono_ns=mono_ns, continuity='UNANCHORED', raw=None)

    def ingest(self, raw, *, wall_ms, mono_ns, session_id):
        result = dict(kind='MARKET', raw=raw, wall_ms=wall_ms, mono_ns=mono_ns,
                      session_id=session_id, stream=None)
        v = result['validation'] = dict(valid=False, fatal=True, reasons=[],
                                       continuity='INVALID', reconnect=False)
        try:
            if (session_id != self.session_id or not _integer(wall_ms)
                    or not _integer(mono_ns)):
                raise FlowError('INVALID_SESSION_OR_RECEIPT')
            if ((self.last_wall is not None and wall_ms < self.last_wall)
                    or (self.last_mono is not None and mono_ns < self.last_mono)):
                raise FlowError('RECEIPT_CLOCK_REGRESSION')
            self.last_wall, self.last_mono = wall_ms, mono_ns
            if not isinstance(raw, str) or len(raw.encode('utf-8')) > MAX_FRAME_BYTES:
                raise FlowError('INVALID_FRAME')
            try:
                def invalid_constant(value):
                    raise FlowError('NONFINITE_JSON')
                frame = json.loads(raw, parse_constant=invalid_constant)
            except (ValueError, TypeError):
                raise FlowError('MALFORMED_JSON') from None
            if not isinstance(frame, dict) or not isinstance(frame.get('data'), dict):
                raise FlowError('INVALID_COMBINED_FRAME')
            topic, data = frame.get('stream'), frame['data']
            result['stream'] = topic
            if topic == '!serverShutdown' and data.get('e') == 'serverShutdown':
                v.update(valid=True, fatal=False, reconnect=True, continuity='CONTROL')
                return result
            if topic not in STREAMS:
                raise FlowError('UNEXPECTED_TOPIC')
            self.counts[topic] += 1
            old = self.by_topic.get(topic)
            digest = _sha(data)
            if topic.endswith('@aggTrade'):
                if (data.get('e') != 'aggTrade' or data.get('s') != topic.split('@')[0].upper()
                        or not all(_integer(data.get(k)) for k in ('a', 'f', 'l'))
                        or not all(_integer(data.get(k), 1) for k in ('E', 'T'))
                        or data['f'] > data['l'] or data['T'] > data['E']
                        or type(data.get('m')) is not bool
                        or _positive(data.get('p')) is None or _positive(data.get('q')) is None):
                    raise FlowError('INVALID_AGGREGATE_TRADE')
                identity = data['a']
                v.update(first_trade_id=data['f'], last_trade_id=data['l'],
                         aggregate_id=identity, event_time_ms=data['E'], trade_time_ms=data['T'],
                         clock_quality='UNCALIBRATED_LOCAL_RECEIPT')
            else:
                if not _integer(data.get('lastUpdateId')):
                    raise FlowError('INVALID_DEPTH_ID')
                sides = []
                for name, descending in (('bids', True), ('asks', False)):
                    levels = data.get(name)
                    if not isinstance(levels, list) or not 1 <= len(levels) <= 20:
                        raise FlowError('INVALID_DEPTH_LEVELS')
                    prices = []
                    for level in levels:
                        if (not isinstance(level, list) or len(level) != 2
                                or _positive(level[0]) is None or _positive(level[1]) is None):
                            raise FlowError('INVALID_DEPTH_LEVELS')
                        prices.append(_positive(level[0]))
                    if any((a <= b if descending else a >= b) for a, b in zip(prices, prices[1:])):
                        raise FlowError('UNSORTED_DEPTH')
                    sides.append(prices)
                if sides[0][0] >= sides[1][0]:
                    raise FlowError('CROSSED_DEPTH')
                identity = data['lastUpdateId']
                v.update(last_update_id=identity, book_semantics='STANDING_TOP20_SNAPSHOT',
                         exchange_age='UNKNOWN')
            continuity = 'UNANCHORED' if old is None else 'CONTIGUOUS'
            if old:
                if identity < old['id']:
                    raise FlowError('BACKWARD_ID')
                if identity == old['id']:
                    if digest != old['digest']:
                        self.conflicts_n += 1
                        raise FlowError('ID_CONFLICT')
                    continuity = 'DUPLICATE'; self.duplicates_n += 1
                elif topic.endswith('@aggTrade'):
                    if data['f'] <= old['last_trade_id']:
                        raise FlowError('BACKWARD_TRADE_ID')
                    if data['E'] < old['event_time_ms'] or data['T'] < old['trade_time_ms']:
                        raise FlowError('SOURCE_TIME_REGRESSION')
                    if identity != old['id'] + 1:
                        v['reasons'].append('AGGREGATE_ID_GAP')
                    if data['f'] != old['last_trade_id'] + 1:
                        v['reasons'].append('TRADE_ID_GAP')
                if wall_ms - old['wall_ms'] > STALE_RECEIPT_MS:
                    v['reasons'].append('STALE_RECEIPT_GAP')
                if v['reasons']:
                    continuity = 'GAP'; self.gaps_n += 1
            self.by_topic[topic] = dict(id=identity, digest=digest, wall_ms=wall_ms,
                last_trade_id=data.get('l'), event_time_ms=data.get('E'), trade_time_ms=data.get('T'))
            self.first_receipt = wall_ms if self.first_receipt is None else self.first_receipt
            self.last_receipt = wall_ms
            v.update(valid=True, fatal=False, continuity=continuity)
        except FlowError as error:
            v['reasons'].append(str(error)); self.invalid_n += 1
        return result

    def summary(self, now_ms=None):
        stale = [] if now_ms is None else [s for s in STREAMS if s not in self.by_topic
            or now_ms-self.by_topic[s]['wall_ms'] > STALE_RECEIPT_MS]
        return dict(messages_per_topic=dict(self.counts), sessions_n=self.sessions_n,
                    reconnections_n=max(0, self.sessions_n-1), gaps_n=self.gaps_n,
                    conflicts_n=self.conflicts_n, invalid_n=self.invalid_n, duplicates_n=self.duplicates_n,
                    first_receipt_ms=self.first_receipt, last_receipt_ms=self.last_receipt,
                    stale_topics=stale, exchange_age='UNKNOWN')


def _atomic_json(path, value):
    temporary = path.with_name(path.name + '.next')
    with temporary.open('wb') as stream:
        stream.write(_json(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class ChunkStore:
    def __init__(self, folder, registration, *, free_bytes=None, chunk_bytes=CHUNK_BYTES):
        _validate_registration(registration)
        if not _integer(chunk_bytes, 1) or chunk_bytes > CHUNK_BYTES:
            raise FlowError('INVALID_CHUNK_LIMIT')
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = self.folder / 'collector.lock'
        if (self.folder / 'registration.json').exists():
            raise FlowError('DATASET_EXISTS')
        try:
            with self.lock.open('x') as stream:
                stream.write(str(os.getpid()))
        except FileExistsError:
            raise FlowError('COLLECTOR_BUSY') from None
        self.registration = deepcopy_json(registration)
        self.free_bytes = free_bytes or (lambda: shutil.disk_usage(self.folder).free)
        self.chunk_bytes = chunk_bytes
        self.buffer = bytearray(); self.buffer_records = []
        self.first_mono = None; self.compressed_bytes = 0
        self.chunks = []; self.failed = False; self.closed = False
        try:
            if self.free_bytes() < MINIMUM_FREE_BYTES:
                raise FlowError('DISK_RESERVE')
            _atomic_json(self.folder / 'registration.json', registration)
            self._manifest()
        except BaseException:
            self.lock.unlink(missing_ok=True)
            raise

    def _manifest(self):
        _atomic_json(self.folder / 'manifest.json', dict(schema_version=1,
            registration_sha256=_sha(self.registration), chunks=self.chunks,
            compressed_bytes=self.compressed_bytes,
            records_n=sum(c['records_n'] for c in self.chunks)))

    def append(self, record, *, monotonic_ns):
        if self.failed or self.closed:
            raise FlowError('STORE_CLOSED_OR_FAILED')
        line = _json(record) + b'\n'
        if len(line) > self.chunk_bytes:
            raise FlowError('RECORD_TOO_LARGE')
        if self.buffer and (len(self.buffer)+len(line) > self.chunk_bytes
                            or monotonic_ns-self.first_mono >= CHUNK_NS):
            self.flush()
        if not self.buffer:
            self.first_mono = monotonic_ns
        self.buffer.extend(line)
        self.buffer_records.append((record.get('wall_ms'), record.get('session_id')))

    def flush_due(self, monotonic_ns):
        if self.buffer and monotonic_ns-self.first_mono >= CHUNK_NS:
            self.flush(); return True
        return False

    def flush(self):
        if not self.buffer:
            return
        if self.failed or self.closed:
            raise FlowError('STORE_CLOSED_OR_FAILED')
        try:
            body = gzip.compress(bytes(self.buffer), compresslevel=6, mtime=0)
            if self.compressed_bytes+len(body) > MAX_COMPRESSED_BYTES:
                raise FlowError('COMPRESSED_BUDGET')
            if self.free_bytes()-len(body) < MINIMUM_FREE_BYTES:
                raise FlowError('DISK_RESERVE')
            name = 'chunk-%08d.jsonl.gz' % len(self.chunks)
            target = self.folder / name; temporary = target.with_suffix(target.suffix + '.next')
            with temporary.open('xb') as stream:
                stream.write(body); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, target)
            times = [t for t, s in self.buffer_records if _integer(t)]
            self.chunks.append(dict(file=name, sha256=hashlib.sha256(body).hexdigest(),
                uncompressed_sha256=hashlib.sha256(self.buffer).hexdigest(), compressed_bytes=len(body),
                uncompressed_bytes=len(self.buffer), records_n=len(self.buffer_records),
                first_receipt_ms=min(times) if times else None, last_receipt_ms=max(times) if times else None,
                session_ids=sorted({s for t, s in self.buffer_records if isinstance(s, str)})))
            self.compressed_bytes += len(body)
            self._manifest()
            self.buffer.clear(); self.buffer_records.clear(); self.first_mono = None
        except FlowError:
            self.failed = True; raise
        except (OSError, ValueError):
            self.failed = True
            raise FlowError('PERSISTENCE_FAILED') from None

    def close(self):
        if self.closed:
            return
        try:
            if not self.failed:
                self.flush()
        finally:
            self.closed = True
            self.lock.unlink(missing_ok=True)


def deepcopy_json(value):
    return json.loads(_json(value))


def verify_dataset(folder):
    folder = Path(folder)
    try:
        registration = json.loads((folder / 'registration.json').read_text(encoding='utf-8'))
        _validate_registration(registration)
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['registration_sha256'] != _sha(registration):
            raise FlowError('REGISTRATION_INTEGRITY')
        n = compressed = 0
        for i, item in enumerate(manifest['chunks']):
            if item['file'] != 'chunk-%08d.jsonl.gz' % i:
                raise FlowError('CHUNK_INTEGRITY')
            path = folder / item['file']
            if path.stat().st_size > CHUNK_BYTES+65536:
                raise FlowError('CHUNK_INTEGRITY')
            body = path.read_bytes()
            if hashlib.sha256(body).hexdigest() != item['sha256'] or len(body) != item['compressed_bytes']:
                raise FlowError('CHUNK_INTEGRITY')
            with gzip.open(path, 'rb') as stream:
                plain = stream.read(CHUNK_BYTES+1)
            if (len(plain) != item['uncompressed_bytes'] or len(plain) > CHUNK_BYTES
                    or hashlib.sha256(plain).hexdigest() != item['uncompressed_sha256']):
                raise FlowError('CHUNK_INTEGRITY')
            rows = plain.splitlines()
            if len(rows) != item['records_n']:
                raise FlowError('CHUNK_INTEGRITY')
            for row in rows:
                json.loads(row)
            n += len(rows); compressed += len(body)
        if n != manifest['records_n'] or compressed != manifest['compressed_bytes']:
            raise FlowError('MANIFEST_INTEGRITY')
        if len(list(folder.glob('chunk-*.jsonl.gz'))) != len(manifest['chunks']):
            raise FlowError('ORPHAN_CHUNK')
        return dict(registration=registration, records_n=n, chunks_n=len(manifest['chunks']),
                    compressed_bytes=compressed, integrity='PASS')
    except (OSError, KeyError, TypeError, ValueError, gzip.BadGzipFile, EOFError):
        raise FlowError('CHUNK_INTEGRITY') from None


def _below_normal():
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise FlowError('PRIORITY_NOT_APPLIED')


def run_capture(folder):
    """One finite public-only connection, with explicit coverage gaps on reconnect."""
    from websockets.sync.client import connect
    from websockets.exceptions import ConnectionClosed, InvalidHandshake, WebSocketException
    _below_normal()
    host_verification = verify_official_host()
    start_ms = time.time_ns() // 1_000_000
    registration = make_registration(start_ms, documentation_sha256=host_verification['sha256'])
    store = ChunkStore(folder, registration)
    validator = FlowValidator()
    reason = None; error_class = None; backoff = 1; last_status = 0
    logger = logging.Logger('public_demo_flow_transport'); logger.disabled = True

    def status(force=False):
        nonlocal last_status
        now = time.time_ns() // 1_000_000
        if force or now-last_status >= 5000:
            summary = validator.summary(now)
            summary.update(state='COLLECTING' if reason is None else
                ('CAPTURE_FINISHED' if reason == 'PLANNED_END' else 'INCOMPLETE'),
                stop_reason=reason, error_class=error_class, checked_at_ms=now,
                registration_sha256=_sha(registration), planned_end_ms=registration['planned_end_ms'],
                full_days_start_ms=registration['full_days_start_ms'],
                compressed_bytes=store.compressed_bytes, free_bytes=store.free_bytes(),
                calendar_schedule_finished=reason == 'PLANNED_END',
                complete_receipt_coverage=False if validator.gaps_n or validator.invalid_n
                    or reason and reason != 'PLANNED_END' else None,
                profitability_evidence=False, model_training='NOT_TRAINED', promotion_allowed=False)
            _atomic_json(store.folder / 'status.json', summary); last_status = now

    def stopping():
        return stop_reason(registration, now_ms=time.time_ns()//1_000_000,
            compressed_bytes=store.compressed_bytes, free_bytes=store.free_bytes(),
            stop_requested=(store.folder / 'stop').exists())

    try:
        _atomic_json(store.folder / 'host-verification.json', host_verification)
        status(True)
        while reason is None:
            reason = stopping()
            if reason:
                break
            session_id = str(uuid.uuid4())
            record = validator.begin_session(session_id, wall_ms=time.time_ns()//1_000_000,
                                              mono_ns=time.monotonic_ns())
            store.append(record, monotonic_ns=record['mono_ns']); store.flush()
            session_reason = 'TRANSPORT_CLOSED'
            try:
                with connect(URL, ssl=ssl.create_default_context(), proxy=None,
                        open_timeout=10, close_timeout=2, compression=None,
                        max_size=MAX_FRAME_BYTES, max_queue=16, ping_interval=None,
                        logger=logger, legacy=True) as connection:
                    if (connection.request.path != '/stream?streams='+'/'.join(STREAMS)
                            or connection.request.headers.get('Host') != 'demo-stream.binance.com'):
                        raise FlowError('HANDSHAKE_TARGET_MISMATCH')
                    last_message = time.monotonic_ns()
                    while reason is None:
                        reason = stopping()
                        if reason:
                            break
                        try:
                            raw = connection.recv(timeout=1)
                        except TimeoutError:
                            store.flush_due(time.monotonic_ns()); status()
                            if time.monotonic_ns()-last_message > 30_000_000_000:
                                session_reason = 'NO_MARKET_MESSAGES'; break
                            continue
                        wall, mono = time.time_ns()//1_000_000, time.monotonic_ns()
                        # Binary frames are invalid, but their exact bytes remain available.
                        if isinstance(raw, bytes):
                            record = dict(kind='INVALID_BINARY', raw_hex=raw.hex(),
                                wall_ms=wall, mono_ns=mono, session_id=session_id,
                                validation=dict(valid=False, fatal=True, reasons=['BINARY_FRAME']))
                        else:
                            record = validator.ingest(raw, wall_ms=wall, mono_ns=mono,
                                                      session_id=session_id)
                        store.append(record, monotonic_ns=mono)
                        last_message = mono; backoff = 1
                        if record['validation']['fatal']:
                            store.flush(); reason = 'PROTOCOL_INCIDENT'; break
                        if record['validation'].get('reconnect'):
                            store.flush(); session_reason = 'SERVER_SHUTDOWN'; break
                        status()
            except (ConnectionClosed, OSError, TimeoutError) as error:
                session_reason = type(error).__name__
            except InvalidHandshake:
                raise FlowError('HANDSHAKE_FAILED') from None
            except WebSocketException:
                raise FlowError('WEBSOCKET_PROTOCOL_FAILED') from None
            store.append(dict(kind='SESSION_END', session_id=session_id,
                wall_ms=time.time_ns()//1_000_000, mono_ns=time.monotonic_ns(),
                reason=session_reason, raw=None), monotonic_ns=time.monotonic_ns())
            store.flush(); status(True)
            if reason is None:
                until = time.monotonic()+backoff
                while time.monotonic() < until and reason is None:
                    reason = stopping(); time.sleep(.25)
                backoff = min(60, backoff*2)
    except KeyboardInterrupt:
        reason = 'OWN_INTERRUPTION'
    except (FlowError, OSError) as error:
        reason = str(error) if isinstance(error, FlowError) else 'PERSISTENCE_FAILED'
        error_class = type(error).__name__
    except Exception as error:
        reason = 'UNEXPECTED_COLLECTOR_FAILURE'
        error_class = type(error).__name__
    finally:
        try:
            store.close()
        except FlowError as error:
            reason = str(error)
        status(True)
    return json.loads((store.folder / 'status.json').read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    app = Path(__file__).resolve().parents[2]
    if args.verify:
        print(json.dumps(verify_dataset(own_runtime_path(app, args.verify)), sort_keys=True)); return
    if not args.collect:
        raise SystemExit('EXPLICIT_COLLECTION_REQUIRED')
    report = run_capture(own_runtime_path(app, args.output))
    print(json.dumps(report, sort_keys=True))
    if report['stop_reason'] != 'PLANNED_END':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
