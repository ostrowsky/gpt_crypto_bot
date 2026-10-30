"""RDE-2 native-candle compatibility; Decimal arithmetic and read-only lookup.

These source checks cannot certify fills, profitability or unrequested history.
Volume compatibility uses the preregistered ASSUMED_NEAREST_ROUNDING bound;
response formatting never selects or widens the permitted quantum.
"""
from decimal import Context, Decimal, MAX_EMAX, MIN_EMIN, localcontext
import hashlib
import json
from pathlib import Path
import re
import zlib


PARENT_MS = 900_000
PRICE_FIELDS = {1: 'open', 2: 'high', 3: 'low', 4: 'close'}
VOLUME_FIELDS = {5: 'base_volume', 7: 'quote_volume',
                 9: 'taker_base_volume', 10: 'taker_quote_volume'}
NUMBER_FIELDS = (1, 2, 3, 4, 5, 7, 9, 10, 11)
NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')


def canonical_sha256(value):
    """The same sorted canonical UTF-8 JSON hash as the read-only adapter."""
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _decimal(value):
    if not isinstance(value, str) or not NUMBER.fullmatch(value):
        raise ValueError('CANDLE_DECIMAL_SCHEMA')
    number = Decimal(value)
    if not number.is_finite():
        raise ValueError('CANDLE_NONFINITE')
    return number


def _row(row, interval_ms, expected_ms=None):
    if (not isinstance(row, list) or len(row) != 12
            or type(row[0]) is not int or row[0] < 0 or row[0] % interval_ms
            or type(row[6]) is not int or row[6] != row[0] + interval_ms - 1
            or type(row[8]) is not int or row[8] < 0
            or expected_ms is not None and row[0] != expected_ms):
        raise ValueError('CANDLE_GRID_OR_SCHEMA')
    value = list(row)
    for field in NUMBER_FIELDS:
        value[field] = _decimal(row[field])
    if (min(value[i] for i in PRICE_FIELDS) <= 0
            or value[2] < max(value[1], value[4])
            or value[3] > min(value[1], value[4]) or value[2] < value[3]
            or any(value[i] < 0 for i in VOLUME_FIELDS)
            or value[9] > value[5] or value[10] > value[7]):
        raise ValueError('CANDLE_VALUES_INVALID')
    return value


def _context(original, fine):
    values = [row[field] for row in [original] + fine for field in NUMBER_FIELDS]
    high = max(v.adjusted() for v in values)
    low = min(v.as_tuple().exponent for v in values)
    # Exact addition/subtraction, including small discrepancies against large
    # totals, must not inherit Decimal's process-global default precision28.
    precision = max(28, high - low + len(str(len(fine) + 1)) + 8)
    return Context(prec=precision, Emax=MAX_EMAX, Emin=MIN_EMIN)


def _result(state, reasons, interval_ms, aggregate=None, differences=None):
    return dict(state=state, reasons=reasons, aggregate=aggregate,
                differences={} if differences is None else differences, interval_ms=interval_ms)


def validate_window(original, fine, *, interval_ms=60_000, received_at_ms,
                    parent_interval_ms=PARENT_MS):
    """Validate one exact parent interval against ordered native subcandles.

    Original errors are caller/source-contract errors. Bad, missing or unclosed
    fine responses return INCOMPLETE; valid but incompatible values MISMATCH.
    ``parent_interval_ms=60000`` is used only for second-to-minute checks.
    """
    if (type(interval_ms) is not int or interval_ms not in (60_000, 1_000)
            or type(parent_interval_ms) is not int or parent_interval_ms not in (PARENT_MS, 60_000)
            or parent_interval_ms % interval_ms):
        raise ValueError('INVALID_CANDLE_INTERVAL')
    reference = _row(original, parent_interval_ms)
    end = reference[0] + parent_interval_ms
    if type(received_at_ms) is not int or received_at_ms < end:
        return _result('INCOMPLETE', ['RECEIPT_BEFORE_WINDOW_END'], interval_ms)
    expected = parent_interval_ms // interval_ms
    if not isinstance(fine, list) or len(fine) != expected:
        return _result('INCOMPLETE', ['FINE_COUNT_MISMATCH'], interval_ms)
    parsed = []
    for i, row in enumerate(fine):
        try:
            parsed.append(_row(row, interval_ms, reference[0] + i * interval_ms))
        except (ValueError, TypeError, ArithmeticError):
            return _result('INCOMPLETE', [f'FINE_CANDLE_INVALID:{i}'], interval_ms)
    with localcontext(_context(reference, parsed)):
        aggregate = [reference[0], parsed[0][1], max(r[2] for r in parsed),
            min(r[3] for r in parsed), parsed[-1][4], sum((r[5] for r in parsed), Decimal(0)),
            end - 1, sum((r[7] for r in parsed), Decimal(0)), sum(r[8] for r in parsed),
            sum((r[9] for r in parsed), Decimal(0)), sum((r[10] for r in parsed), Decimal(0)), Decimal(0)]
        differences = {}
        reasons = []
        for field, name in PRICE_FIELDS.items():
            actual, target = aggregate[field], reference[field]
            difference = abs(actual - target)
            differences[name] = dict(original=format(target, 'f'), aggregate=format(actual, 'f'),
                difference=format(difference, 'f'), bound='0', within_bound=difference == 0,
                exact_equal=difference == 0)
            if difference:
                reasons.append('FIELD_MISMATCH:' + name)
        for field, name in VOLUME_FIELDS.items():
            actual, target = aggregate[field], reference[field]
            # Freeze from the ORIGINAL before requesting fine data. Coarser
            # textual precision in a response cannot relax this threshold.
            quantum = min(Decimal('1e-8'), Decimal((0, (1,), target.as_tuple().exponent)))
            bound = Decimal(expected + 1) / 2 * quantum
            difference = abs(actual - target)
            differences[name] = dict(original=format(target, 'f'), aggregate=format(actual, 'f'),
                difference=format(difference, 'f'), signed_difference=format(actual-target, 'f'),
                bound=format(bound, 'f'), quantum=format(quantum, 'f'),
                within_bound=difference <= bound, exact_equal=difference == 0,
                rounding_assumption='ASSUMED_NEAREST_ROUNDING')
            if difference > bound:
                reasons.append('FIELD_MISMATCH:' + name)
        difference = abs(aggregate[8] - reference[8])
        differences['trade_count'] = dict(original=str(reference[8]), aggregate=str(aggregate[8]),
            difference=str(difference), bound='0', within_bound=difference == 0, exact_equal=difference == 0)
        if difference:
            reasons.append('FIELD_MISMATCH:trade_count')
        output = list(aggregate)
        for field in NUMBER_FIELDS:
            output[field] = format(output[field], 'f')
    return _result('MISMATCH' if reasons else 'MATCH', reasons, interval_ms, output, differences)


def validate_seconds(original, minutes, seconds, *, received_at_ms):
    """Check900 seconds against BOTH15m source and every accepted minute."""
    minute_reference = validate_window(original, minutes, interval_ms=60_000,
                                       received_at_ms=received_at_ms)
    whole = validate_window(original, seconds, interval_ms=1_000, received_at_ms=received_at_ms)
    checks = []
    reasons = list(whole['reasons'])
    if minute_reference['state'] != 'MATCH':
        reasons.append('MINUTE_REFERENCE_NOT_MATCHED')
    if minute_reference['state'] == 'MATCH' and whole['aggregate'] is not None:
        for i, minute in enumerate(minutes):
            checked = validate_window(minute, seconds[i*60:(i+1)*60], interval_ms=1_000,
                parent_interval_ms=60_000, received_at_ms=received_at_ms)
            checked['origin_ms'] = minute[0]
            checks.append(checked)
            if checked['state'] != 'MATCH':
                reasons.append(f'MINUTE_BUCKET_{checked["state"]}:{i}')
    states = [minute_reference['state'], whole['state']] + [r['state'] for r in checks]
    state = 'INCOMPLETE' if 'INCOMPLETE' in states else 'MISMATCH' if 'MISMATCH' in states else 'MATCH'
    return dict(whole, state=state, reasons=reasons, minute_reference=minute_reference,
                minute_checks=checks)


def original_bar(archive_path, symbol, origin_ms):
    """Read an exact regular15m candle without opening an archive writer.

    Only page candidates containing the requested origin are decompressed.
    Each candidate's metadata, compressed SHA, canonical UTF-8 SHA and row
    order are verified before the requested candle can be accepted.
    """
    if (not isinstance(symbol, str) or not symbol or type(origin_ms) is not int
            or origin_ms < 0 or origin_ms % PARENT_MS):
        raise ValueError('ORIGINAL_LOOKUP_ARGUMENTS')
    path = Path(archive_path).resolve()
    if not path.is_file():
        raise ValueError('ORIGINAL_ARCHIVE_UNAVAILABLE')
    try:
        import sqlite3
    except ImportError:
        raise ValueError('ORIGINAL_ARCHIVE_UNAVAILABLE') from None
    try:
        connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=5)
    except (sqlite3.Error, OSError, ValueError):
        raise ValueError('ORIGINAL_ARCHIVE_UNAVAILABLE') from None
    matches = []
    try:
        try:
            pages = connection.execute('SELECT start_ms,rows_n,first_ms,last_ms,source_sha,blob_sha,body,received_ms '
                'FROM pages WHERE symbol=? AND first_ms<=? AND last_ms>=? ORDER BY start_ms',
                (symbol, origin_ms, origin_ms)).fetchall()
        except sqlite3.Error:
            raise ValueError('ORIGINAL_ARCHIVE_SCHEMA') from None
        for start, count, first, last, source_sha, blob_sha, blob, received in pages:
            if not isinstance(blob, bytes) or hashlib.sha256(blob).hexdigest() != blob_sha:
                raise ValueError('ORIGINAL_COMPRESSED_HASH')
            try:
                rows = json.loads(zlib.decompress(blob))
                digest = canonical_sha256(rows)
            except (ValueError, TypeError, zlib.error, UnicodeError):
                raise ValueError('ORIGINAL_SOURCE_HASH') from None
            if digest != source_sha:
                raise ValueError('ORIGINAL_SOURCE_HASH')
            if (type(start) is not int or type(count) is not int or type(received) is not int
                    or not isinstance(rows, list) or len(rows) != count or not 1 <= count <= 1000
                    or any(not isinstance(r, list) or len(r) != 12 or type(r[0]) is not int
                           or type(r[6]) is not int for r in rows)
                    or first != rows[0][0] or last != rows[-1][0] or start > first
                    or any(a[0] >= b[0] for a, b in zip(rows, rows[1:]))
                    or received < max(r[6] for r in rows)):
                raise ValueError('ORIGINAL_PAGE_SCHEMA')
            for row in rows:
                if row[0] == origin_ms:
                    _row(row, PARENT_MS, origin_ms)
                    matches.append(dict(row=list(row), page_source_sha256=source_sha,
                                        page_blob_sha256=blob_sha))
    finally:
        connection.close()
    if len(matches) != 1:
        raise ValueError('ORIGINAL_NOT_FOUND' if not matches else 'ORIGINAL_DUPLICATE')
    return matches[0]
