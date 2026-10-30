"""R2A05 read-only post-close verification of recorded public probe attempts.

No journal writer, credentials, transport or source compatibility decision is
constructed here. Verified network errors remain errors, not cached candles.
"""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


ENDPOINT = 'https://demo-api.binance.com/api/v3/klines'
KINDS = {'RUN_BEGIN', 'RUN_END', 'REST_BEGIN', 'REST_END'}


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def verify_probe_audit(folder, audit_path):
    """Verify every native record and link accepted caches to exact receipts.

    UNKNOWN describes absent/incomplete provenance; FAIL describes an actual
    contradiction or broken chain. Neither state can certify source matching.
    """
    folder, audit_path = Path(folder), Path(audit_path)
    failures, unknown = [], []
    counts = Counter()
    runids, runs, begins, ends, raw_bodies = [], {}, {}, {}, {}
    attempts = Counter()
    tip = '0' * 64
    linked = 0
    seen_cache_ends = set()

    def issue(reason, *, fail=False):
        target = failures if fail else unknown
        if reason not in target:
            target.append(reason)

    def result():
        return dict(state='FAIL' if failures else 'UNKNOWN' if unknown else 'VERIFIED_RECORDED_PUBLIC_ATTEMPTS',
            reasons=failures + unknown, failures=failures, unknowns=unknown,
            counts=dict(counts), records_n=sum(counts.values()), attempts_n=counts['REST_BEGIN'],
            responses_n=sum(r['data'].get('outcome') == 'RESPONSE' for r in ends.values()),
            errors_n=sum(r['data'].get('outcome') != 'RESPONSE' or r['data'].get('status') != 200
                         for r in ends.values()), linked_cached_responses_n=linked,
            tip=tip, runids=runids, run_ids=runids, runs_n=len(runids),
            closed_runs_n=sum(v.get('end') is not None for v in runs.values()),
            scope='RECORDED_PUBLIC_ATTEMPTS_AND_CACHE_LINKS_ONLY', promotion_allowed=False)

    if not audit_path.is_file():
        issue('NATIVE_AUDIT_MISSING')
        return result()
    # A normal mode=ro SQLite connection may create WAL/SHM sidecars. This
    # verifier runs after writer close: first refuse uncheckpointed WAL, then
    # inspect only the immutable main file without SQLite bookkeeping writes.
    wal = audit_path.with_name(audit_path.name + '-wal')
    if wal.is_file() and wal.stat().st_size:
        issue('NATIVE_AUDIT_WAL_NOT_FINALIZED')
        return result()
    try:
        connection = sqlite3.connect(audit_path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True, timeout=5)
    except (sqlite3.Error, OSError):
        issue('NATIVE_AUDIT_UNREADABLE')
        return result()
    records = []
    try:
        try:
            identity_row = connection.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
            identity = json.loads(identity_row[0]) if identity_row else None
            rows = connection.execute('SELECT seq,id,kind,payload_json,content_sha256 FROM records ORDER BY seq')
            seen_ids = set()
            for number, (seq, identifier, kind, payload, claimed) in enumerate(rows, 1):
                value = json.loads(payload)
                if (type(seq) is not int or seq != number or value.get('seq') != seq
                        or not isinstance(identifier, str) or not identifier or identifier in seen_ids
                        or value.get('event_id') != identifier or value.get('kind') != kind
                        or kind not in KINDS or value.get('previous_sha256') != tip or _sha(value) != claimed):
                    issue('NATIVE_AUDIT_CHAIN_INVALID', fail=True)
                    return result()
                if (value.get('application_id') != 'binance_demo_bot'
                        or value.get('environment') != 'BINANCE_SPOT_DEMO'
                        or identity is not None and any(value.get(k) != v for k, v in identity.items())):
                    issue('NATIVE_AUDIT_IDENTITY_CONTRADICTION', fail=True)
                if not isinstance(value.get('data'), dict):
                    issue('NATIVE_AUDIT_DATA_INVALID', fail=True)
                    return result()
                seen_ids.add(identifier)
                tip = claimed
                counts[kind] += 1
                records.append(value)
            if identity is None:
                issue('NATIVE_AUDIT_IDENTITY_MISSING')
        except (sqlite3.Error, ValueError, TypeError, KeyError, AttributeError):
            issue('NATIVE_AUDIT_SCHEMA_OR_CHAIN_INVALID', fail=True)
            return result()
    finally:
        connection.close()
    if not records:
        issue('NATIVE_AUDIT_EMPTY')
    for value in records:
        kind, data, rid = value['kind'], value['data'], value.get('run_id')
        if not isinstance(rid, str) or not rid:
            issue('NATIVE_RUN_PROVENANCE_MISSING')
            continue
        if kind == 'RUN_BEGIN':
            if rid in runs:
                issue('NATIVE_DUPLICATE_RUN_BEGIN', fail=True)
            else:
                runs[rid] = dict(begin=value['seq'], end=None)
                runids.append(rid)
            continue
        if rid not in runs:
            issue('NATIVE_RUN_BEGIN_MISSING', fail=True)
            continue
        run = runs[rid]
        if kind == 'RUN_END':
            if run['end'] is not None:
                issue('NATIVE_DUPLICATE_RUN_END', fail=True)
            run['end'] = value['seq']
            continue
        if run['end'] is not None:
            issue('NATIVE_EVENT_AFTER_RUN_END', fail=True)
        if not all(k in data for k in ('method', 'endpoint', 'private', 'request_id', 'attempt_id')):
            issue('NATIVE_ATTEMPT_PROVENANCE_MISSING')
            continue
        if (data['method'] != 'GET' or data['endpoint'] != '/api/v3/klines'
                or data['private'] is not False):
            issue('NATIVE_NONPUBLIC_OR_UNEXPECTED_ENDPOINT', fail=True)
        if not all(isinstance(data[k], str) and data[k] for k in ('request_id', 'attempt_id')):
            issue('NATIVE_ATTEMPT_IDS_MISSING')
            continue
        if kind == 'REST_BEGIN':
            if not isinstance(data.get('parameters'), dict):
                issue('NATIVE_REQUEST_PARAMETERS_MISSING')
            attempts[data['attempt_id']] += 1
            if attempts[data['attempt_id']] > 1:
                issue('NATIVE_DUPLICATE_ATTEMPT_ID', fail=True)
            begins[value['event_id']] = value
        elif kind == 'REST_END':
            parent = data.get('parent_event_id')
            if not isinstance(parent, str) or not parent:
                issue('NATIVE_PARENT_ID_MISSING')
                continue
            if parent not in begins:
                issue('NATIVE_ORPHAN_TERMINAL', fail=True)
                continue
            begin = begins[parent]
            if any(data.get(k) != begin['data'].get(k) for k in
                   ('attempt_id', 'request_id', 'method', 'endpoint', 'private')) or begin['run_id'] != rid:
                issue('NATIVE_BEGIN_END_CONTRADICTION', fail=True)
            if any(r['data'].get('parent_event_id') == parent for r in ends.values()):
                issue('NATIVE_DUPLICATE_TERMINAL', fail=True)
            ends[value['event_id']] = value
            if type(data.get('received_at_ms')) is not int:
                issue('NATIVE_RECEIPT_MISSING')
            outcome = data.get('outcome')
            if outcome not in ('RESPONSE', 'TIMEOUT', 'TRANSPORT_ERROR', 'CANCELLED'):
                issue('NATIVE_OUTCOME_MISSING')
            wrapper = data.get('payload')
            if wrapper is not None:
                if not isinstance(wrapper, dict) or wrapper.get('representation') != 'EXACT_PUBLIC_BYTES':
                    issue('NATIVE_EXACT_PUBLIC_BYTES_UNAVAILABLE')
                else:
                    try:
                        raw = base64.b64decode(wrapper['body_base64'], validate=True)
                        if (type(wrapper['bytes_n']) is not int or len(raw) != wrapper['bytes_n']
                                or hashlib.sha256(raw).hexdigest() != wrapper['content_sha256']):
                            raise ValueError('RAW_BYTES')
                        raw_bodies[value['event_id']] = raw
                    except (ValueError, TypeError, KeyError):
                        issue('NATIVE_RAW_BODY_CONTRADICTION', fail=True)
            elif outcome == 'RESPONSE':
                issue('NATIVE_RESPONSE_BODY_MISSING')
    if any(r['end'] is None for r in runs.values()):
        issue('NATIVE_RUN_OPEN')
    paired = {r['data'].get('parent_event_id') for r in ends.values()}
    if set(begins) - paired:
        issue('NATIVE_UNPAIRED_ATTEMPTS')
    if not counts['REST_BEGIN']:
        issue('NO_RECORDED_PUBLIC_ATTEMPTS')

    for path in sorted((folder / 'responses').glob('*.json')):
        name = path.name
        try:
            cached = json.loads(path.read_text(encoding='utf-8'))
            meta = cached['metadata']
            required = ('audit_request_id', 'audit_attempt_id', 'audit_begin_event_id',
                        'audit_end_event_id', 'raw_body_sha256', 'received_at_ms')
            if (not isinstance(meta, dict) or any(meta.get(k) is None for k in required)
                    or any(k not in cached for k in ('parameters', 'endpoint', 'rows'))):
                issue('CACHE_PROVENANCE_MISSING:' + name)
                continue
        except (OSError, ValueError, KeyError, TypeError):
            issue('CACHE_PROVENANCE_UNREADABLE:' + name)
            continue
        begin = begins.get(meta['audit_begin_event_id'])
        end = ends.get(meta['audit_end_event_id'])
        if begin is None or end is None:
            issue('CACHE_NATIVE_RECORD_NOT_FOUND:' + name, fail=True)
            continue
        bd, ed = begin['data'], end['data']
        if (cached['endpoint'] != ENDPOINT or cached['parameters'] != bd.get('parameters')
                or meta['audit_request_id'] != bd['request_id'] or meta['audit_attempt_id'] != bd['attempt_id']
                or ed.get('parent_event_id') != begin['event_id']
                or ed.get('attempt_id') != meta['audit_attempt_id']
                or ed.get('request_id') != meta['audit_request_id']
                or type(meta['received_at_ms']) is not int or meta['received_at_ms'] != ed.get('received_at_ms')
                or ed.get('outcome') != 'RESPONSE' or ed.get('status') != 200):
            issue('CACHE_RECEIPT_OR_REQUEST_CONTRADICTION:' + name, fail=True)
            continue
        raw = raw_bodies.get(end['event_id'])
        if raw is None:
            issue('CACHE_EXACT_NATIVE_BODY_UNAVAILABLE:' + name)
            continue
        try:
            if (hashlib.sha256(raw).hexdigest() != meta['raw_body_sha256']
                    or json.loads(raw) != cached['rows']
                    or 'source_sha256' in cached and _sha(cached['rows']) != cached['source_sha256']):
                raise ValueError('CONTENT')
        except (ValueError, TypeError, UnicodeError):
            issue('CACHE_NATIVE_BODY_CONTRADICTION:' + name, fail=True)
            continue
        if end['event_id'] in seen_cache_ends:
            issue('CACHE_DUPLICATE_NATIVE_RESPONSE:' + name, fail=True)
            continue
        seen_cache_ends.add(end['event_id'])
        linked += 1
    return result()
