"""RDE-1 read-only archive materialization and causal, bounded data primitives.

NumPy is imported only by numerical entry points in the isolated research
runtime. An origin index i observes bars[:i], never the execution bar bars[i].
"""
import hashlib
import json
import math
import os
from pathlib import Path
import zlib

STEP = 900_000
FIELDS = ('open', 'high', 'low', 'close', 'quote_volume', 'taker_quote', 'trade_count')
OPEN, HIGH, LOW, CLOSE, QUOTE_VOLUME, TAKER_QUOTE, TRADE_COUNT = range(7)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_frozen(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('DATA_MANIFEST_CHANGED')
        return
    temporary = path.with_suffix(path.suffix + '.next')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _readonly(archive):
    try:
        import sqlite3
    except ImportError:
        import apsw
        return apsw.Connection(str(archive), flags=apsw.SQLITE_OPEN_READONLY)
    return sqlite3.connect(archive.as_uri() + '?mode=ro', uri=True, timeout=10)


def _valid_rows(bars):
    import numpy as np
    values = np.asarray(bars)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError('BAR_SHAPE')
    return (np.isfinite(values).all(axis=1) & (values[:, :4] > 0).all(axis=1)
            & (values[:, HIGH] >= np.maximum(values[:, OPEN], values[:, CLOSE]))
            & (values[:, LOW] <= np.minimum(values[:, OPEN], values[:, CLOSE]))
            & (values[:, HIGH] >= values[:, LOW]) & (values[:, QUOTE_VOLUME] >= 0)
            & (values[:, TAKER_QUOTE] >= 0) & (values[:, TAKER_QUOTE] <= values[:, QUOTE_VOLUME])
            & (values[:, TRADE_COUNT] >= 0) & (values[:, TRADE_COUNT] == np.floor(values[:, TRADE_COUNT])))


def _row_values(row):
    try:
        if any(isinstance(row[i], bool) for i in (1, 2, 3, 4, 5, 7, 8, 9, 10)):
            return None
        values = [float(row[i]) for i in (1, 2, 3, 4, 7, 10, 8)]
        base, buy_base = float(row[5]), float(row[9])
    except (ValueError, TypeError, OverflowError):
        return None
    if (not all(math.isfinite(v) for v in values + [base, buy_base])
            or min(values[:4]) <= 0 or values[HIGH] < max(values[OPEN], values[CLOSE])
            or values[LOW] > min(values[OPEN], values[CLOSE]) or values[HIGH] < values[LOW]
            or values[QUOTE_VOLUME] < 0 or not 0 <= values[TAKER_QUOTE] <= values[QUOTE_VOLUME]
            or values[TRADE_COUNT] < 0 or values[TRADE_COUNT] != int(values[TRADE_COUNT])
            or base < 0 or not 0 <= buy_base <= base):
        return None
    return values


def _page_vectors(rows,*,start,first_ms,cutoff_ms,previous):
    """Vectorized equivalent of strict scalar validation, including bad rows."""
    import numpy as np
    if any(not isinstance(r,list) or len(r)!=12 or type(r[0]) is not int or type(r[6]) is not int for r in rows):
        raise ValueError('ARCHIVE_SCHEMA')
    opens=np.asarray([r[0] for r in rows],dtype=np.int64)
    closes=np.asarray([r[6] for r in rows],dtype=np.int64)
    if ((opens<max(start,first_ms)).any() or (closes<opens).any()
            or (closes>opens+STEP-1).any() or (closes>=cutoff_ms).any()):
        raise ValueError('ARCHIVE_SCHEMA')
    if (len(opens)>1 and (np.diff(opens)<=0).any()) or (previous is not None and opens[0]<=previous):
        raise ValueError('ARCHIVE_ORDER')
    regular=(opens%STEP==0)&(closes==opens+STEP-1)
    try:
        numeric=np.asarray(rows,dtype=np.float64)
        values=numeric[:,[1,2,3,4,7,10,8]]
        valid=_valid_rows(values)&np.isfinite(numeric[:,[5,9]]).all(axis=1)
        valid&=(numeric[:,5]>=0)&(numeric[:,9]>=0)&(numeric[:,9]<=numeric[:,5])
        valid&=np.asarray([not any(type(r[j]) is bool for j in (1,2,3,4,5,7,8,9,10)) for r in rows])
    except (ValueError,TypeError,OverflowError):
        values=np.full((len(rows),7),np.nan)
        for i,row in enumerate(rows):
            value=_row_values(row)
            if value is not None:values[i]=value
        valid=_valid_rows(values)
    good=regular&valid
    return dict(slots=((opens[good]-first_ms)//STEP),values=values[good],
        observed_slots=(opens-first_ms)//STEP,short_n=int((~regular).sum()),
        invalid_n=int((regular&~valid).sum()),previous=int(opens[-1]))


def stream_archive(archive, folder, *, cutoff_ms):
    """Validate every source page and write a separate regular-grid .npy per symbol.

    Shortened/invalid rows remain unavailable. Missing slots are NaN, never
    forward filled. Source receipts are retained but do not certify historical
    availability. The output manifest and existing symbol arrays are immutable.
    """
    import numpy as np
    if type(cutoff_ms) is not int or cutoff_ms <= 0 or cutoff_ms % STEP:
        raise ValueError('ARCHIVE_CUTOFF')
    archive = Path(archive).resolve()
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    con = _readonly(archive)
    entries = {}
    try:
        bound = con.execute("SELECT value FROM meta WHERE key='cutoff_ms'").fetchone()
        if bound is None or int(bound[0]) != cutoff_ms:
            raise ValueError('ARCHIVE_CUTOFF')
        symbols = [row[0] for row in con.execute('SELECT DISTINCT symbol FROM pages ORDER BY symbol')]
        references = {s: (h, p) for s, h, p in con.execute('SELECT symbol,manifest_sha,payload FROM refs ORDER BY symbol')}
        if not symbols or set(symbols) != set(references):
            raise ValueError('ARCHIVE_POPULATION')
        for ordinal, symbol in enumerate(symbols):
            if not isinstance(symbol, str) or not symbol:
                raise ValueError('ARCHIVE_SCHEMA')
            first = con.execute('SELECT first_ms FROM pages WHERE symbol=? AND first_ms IS NOT NULL ORDER BY start_ms LIMIT 1', (symbol,)).fetchone()
            first_ms = (int(first[0]) // STEP) * STEP if first else cutoff_ms
            if first_ms < 0 or first_ms > cutoff_ms:
                raise ValueError('ARCHIVE_SCHEMA')
            count = (cutoff_ms - first_ms) // STEP
            path = folder / ('symbol-' + format(ordinal, '04d') + '.npy')
            temporary = path.with_suffix('.next.npy')
            grid = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float64, shape=(count, 7))
            grid[:] = np.nan
            observed = np.zeros(count, dtype=bool)
            raw_n = short_n = invalid_n = regular_n = 0
            previous = None
            expected_start = 0
            terminal_seen = False
            vectors, receipts = [], []
            try:
                for page in con.execute('SELECT start_ms,rows_n,first_ms,last_ms,source_sha,blob_sha,body,received_ms FROM pages WHERE symbol=? ORDER BY start_ms', (symbol,)):
                    start, rows_n, first_row, last_row, source_sha, blob_sha, blob, received = page
                    if hashlib.sha256(blob).hexdigest() != blob_sha:
                        raise ValueError('ARCHIVE_HASH')
                    try:
                        rows = json.loads(zlib.decompress(blob))
                    except (ValueError, zlib.error, TypeError):
                        raise ValueError('ARCHIVE_HASH') from None
                    if _sha(rows) != source_sha:
                        raise ValueError('ARCHIVE_HASH')
                    if (type(start) is not int or type(rows_n) is not int or type(received) is not int
                            or not isinstance(rows, list) or rows_n != len(rows) or not 0 <= rows_n <= 1000):
                        raise ValueError('ARCHIVE_SCHEMA')
                    if terminal_seen or start != expected_start:
                        raise ValueError('ARCHIVE_ORDER')
                    if not rows:
                        if first_row is not None or last_row is not None:
                            raise ValueError('ARCHIVE_SCHEMA')
                        terminal_seen = True
                    else:
                        if (not isinstance(rows[0], list) or not isinstance(rows[-1], list)
                                or not rows[0] or not rows[-1] or first_row != rows[0][0] or last_row != rows[-1][0]):
                            raise ValueError('ARCHIVE_SCHEMA')
                        page_values=_page_vectors(rows,start=start,first_ms=first_ms,cutoff_ms=cutoff_ms,previous=previous)
                        if received < max(r[6] for r in rows):raise ValueError('ARCHIVE_RECEIPT')
                        previous=page_values['previous'];raw_n+=len(rows)
                        observed[page_values['observed_slots']]=True
                        short_n+=page_values['short_n'];invalid_n+=page_values['invalid_n']
                        slots=page_values['slots']
                        if np.isfinite(grid[slots]).any():raise ValueError('ARCHIVE_ORDER')
                        grid[slots]=page_values['values'];regular_n+=len(slots)
                        expected_start = rows[-1][0] + STEP
                    vectors.append([start, rows_n, first_row, last_row, source_sha, blob_sha])
                    receipts.append(received)
                complete = terminal_seen or (previous is not None and previous + STEP >= cutoff_ms)
                source_manifest = dict(symbol=symbol, cutoff_ms=cutoff_ms, rows_n=raw_n, pages_n=len(vectors),
                    first_open_ms=int(first[0]) if first else None, last_open_ms=previous,
                    state='MAXIMUM_AVAILABLE_FETCH_COMPLETE' if complete and raw_n else 'NO_HISTORY' if complete else 'COLLECTING',
                    source_pages=vectors)
                try:
                    reference = json.loads(references[symbol][1])
                except (ValueError, TypeError):
                    raise ValueError('ARCHIVE_REFERENCE') from None
                if (_sha(source_manifest) != references[symbol][0] or not complete
                        or reference.get('evidence_kind') != 'SIMULATION'
                        or reference.get('alpha_qualified') is not False or reference.get('verdict') != 'NOT_ALPHA_EVIDENCE'):
                    raise ValueError('ARCHIVE_REFERENCE')
                eligible_n = len(eligible_origins(grid))
                grid.flush()
            except BaseException:
                del grid
                temporary.unlink(missing_ok=True)
                raise
            del grid
            file_sha = _file_sha(temporary)
            if path.exists():
                if _file_sha(path) != file_sha:
                    temporary.unlink(missing_ok=True)
                    raise ValueError('DATA_ARRAY_CHANGED')
                temporary.unlink()
            else:
                os.replace(temporary, path)
            entries[symbol] = dict(path=str(path), first_ms=first_ms, count=count, sha256=file_sha,
                raw_rows_n=raw_n, regular_rows_n=regular_n, shortened_rows_n=short_n,
                invalid_rows_n=invalid_n, gap_rows_n=count-int(observed.sum()), eligible_origins_n=eligible_n,
                pages_n=len(vectors), source_manifest_sha256=references[symbol][0],
                source_pages_sha256=_sha(vectors), received_min_ms=min(receipts), received_max_ms=max(receipts))
    finally:
        con.close()
    manifest = dict(schema_version=1, symbols=symbols, entries=entries, fields=list(FIELDS), step_ms=STEP,
        archive=str(archive), global_first_ms=min(v['first_ms'] for v in entries.values()), cutoff_ms=cutoff_ms,
        evidence_kind='SIMULATION', holdout_status='RETROSPECTIVE_REUSED_HOLDOUT',
        historical_receipts_certified=False, historical_membership='CURRENT_UNIVERSE_SURVIVOR_REFERENCE', promotion_allowed=False)
    manifest['data_sha256'] = _sha(manifest)
    _write_frozen(folder / 'manifest.json', manifest)
    return manifest


def primitives(bars):
    """Five channels for each closed bar; first return/volume difference are NaN."""
    import numpy as np
    source = np.asarray(bars, dtype=np.float64)
    valid = _valid_rows(source)
    values = np.full((len(source), 5), np.nan, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        values[:, 1] = np.log(source[:, HIGH] / source[:, LOW])
        values[:, 2] = np.log(source[:, CLOSE] / source[:, OPEN])
        values[:, 4] = np.divide(2*source[:, TAKER_QUOTE], source[:, QUOTE_VOLUME],
            out=np.ones(len(source)), where=source[:, QUOTE_VOLUME] != 0) - 1
        values[1:, 0] = np.log(source[1:, CLOSE] / source[:-1, CLOSE])
        values[1:, 3] = np.log1p(source[1:, QUOTE_VOLUME]) - np.log1p(source[:-1, QUOTE_VOLUME])
    values[~valid] = np.nan
    pair_valid = valid[1:] & valid[:-1]
    values[1:, 0][~pair_valid] = np.nan
    values[1:, 3][~pair_valid] = np.nan
    return values.astype(np.float32)


def eligible_origins(bars, *, lookback=32):
    """Origin i requires lookback+1 valid past bars; no access to bars[i:]."""
    import numpy as np
    if type(lookback) is not int or lookback < 1:
        raise ValueError('FEATURE_WINDOW')
    valid = _valid_rows(bars)
    length = lookback + 1
    if len(valid) < length:
        return np.empty(0, dtype=np.int64)
    cumulative = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    ends = np.arange(length, len(valid)+1, dtype=np.int64)
    return ends[(cumulative[ends]-cumulative[ends-length]) == length]


def _indices(indices):
    import numpy as np
    result = np.asarray(indices)
    if result.ndim != 1 or (result.size and result.dtype.kind not in 'iu'):
        raise ValueError('ORIGIN_INDEX')
    return result.astype(np.int64)


def sequences(bars, indices, *, lookback=32):
    import numpy as np
    indices = _indices(indices)
    if not np.isin(indices, eligible_origins(bars, lookback=lookback)).all():
        raise ValueError('FEATURE_WINDOW')
    features = primitives(bars)
    return features[indices[:, None] + np.arange(-lookback, 0)[None, :]]


def targets(bars, indices, *, horizons=(1, 4), fit_end_index=None):
    """Direct labels, with maturity/missing masks separate from causal origins."""
    import numpy as np
    indices = _indices(indices)
    source = np.asarray(bars, dtype=np.float64)
    end = len(source) if fit_end_index is None else fit_end_index
    if type(end) is not int or end < 0 or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError('TARGET_BOUNDARY')
    if ((indices < 1) | (indices > len(source))).any():
        raise ValueError('ORIGIN_INDEX')
    row_valid = _valid_rows(source)
    cumulative = np.concatenate(([0], np.cumsum(row_valid, dtype=np.int64)))
    values = np.full((len(indices), len(horizons)), np.nan, dtype=np.float64)
    valid = np.zeros(values.shape, dtype=bool)
    for column, horizon in enumerate(horizons):
        possible = (indices+horizon <= end) & (indices+horizon <= len(source))
        locations = np.flatnonzero(possible)
        starts = indices[locations]-1
        stops = indices[locations]+horizon
        contiguous = cumulative[stops]-cumulative[starts] == horizon+1
        locations = locations[contiguous]
        valid[locations, column] = True
        values[locations, column] = np.log(source[indices[locations]+horizon-1, CLOSE] / source[indices[locations]-1, CLOSE])
    labels = np.where(valid, (values > 0).astype(np.float32), np.nan)
    return dict(returns=values, labels=labels, valid=valid, horizons=list(horizons))


def training_indices(bars, start_index, end_index, *, limit=256, purge=16):
    """Deterministic uniform time-stratified mature examples from train only."""
    import numpy as np
    if any(type(v) is not int for v in (start_index, end_index, limit, purge)) or limit < 0 or purge < 0:
        raise ValueError('TRAIN_BOUNDARY')
    origins = eligible_origins(bars)
    origins = origins[(origins >= max(0, start_index)+purge) & (origins < end_index-purge)]
    if not len(origins) or limit == 0:
        return np.empty(0, dtype=np.int64)
    mature = targets(bars, origins, fit_end_index=end_index)['valid'].all(axis=1)
    origins = origins[mature]
    if len(origins) <= limit:
        return origins
    return origins[np.linspace(0, len(origins)-1, limit, dtype=np.int64)]


def candidate_screen(manifest):
    """All-symbol causal top1; array size depends on calendar, not population."""
    import numpy as np
    symbols = sorted(manifest['symbols'])
    first, cutoff = manifest['global_first_ms'], manifest['cutoff_ms']
    times = np.arange(first, cutoff+STEP, STEP, dtype=np.int64)
    selected = np.full(len(times), -1, dtype=np.int32)
    scores = np.full(len(times), -np.inf, dtype=np.float64)
    counts = {}
    for symbol_index, symbol in enumerate(symbols):
        entry = manifest['entries'][symbol]
        source = np.load(entry['path'], mmap_mode='r')
        origins = eligible_origins(source)
        origins = origins[source[origins-1, QUOTE_VOLUME] > 0]
        positions = (entry['first_ms']-first)//STEP + origins
        inside = (positions >= 0) & (positions < len(times))
        origins, positions = origins[inside], positions[inside]
        counts[symbol] = len(origins)
        if len(origins):
            returns = np.zeros(len(source), dtype=np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                returns[1:] = np.log(source[1:, CLOSE] / source[:-1, CLOSE])
            returns[~np.isfinite(returns)] = 0
            cumulative = np.concatenate(([0.], np.cumsum(returns)))
            squares = np.concatenate(([0.], np.cumsum(returns*returns)))
            mean = (cumulative[origins]-cumulative[origins-32])/32
            variance = (squares[origins]-squares[origins-32])/32-mean*mean
            scale = np.maximum(np.sqrt(np.maximum(variance, 0)), .0001)
            strength = np.abs(np.log(source[origins-1, CLOSE]/source[origins-5, CLOSE]))/scale
            improve = strength > scores[positions]
            # Sorted symbols and strict improvement implement the lexical tie.
            selected[positions[improve]] = symbol_index
            scores[positions[improve]] = strength[improve]
        del source
    return dict(times_ms=times, symbol_indices=selected, scores=scores, symbols=symbols,
                eligible_counts=counts, total_selected_n=int((selected >= 0).sum()))
