"""RDE01..03: real archive fixtures and causal numeric contracts."""
import hashlib
import importlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np

STEP = 900_000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def raw_bar(index, price=100., quote=100., buy=50.):
    at = index * STEP
    return [at, str(price), str(price * 1.01), str(price * .99), str(price),
            str(quote / price), at + STEP - 1, str(quote), 10, str(buy / price), str(buy), '0']


def bars(count=96):
    close = np.exp(np.arange(count, dtype=float) * .002) * 100
    return np.column_stack((close, close * 1.01, close * .99, close,
                            np.full(count, 100.), np.full(count, 60.), np.full(count, 10.)))


def archive(path, rows_by_symbol, *, cutoff=96 * STEP, mutate=None):
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    con.execute('INSERT INTO meta VALUES (?,?)', ('cutoff_ms', str(cutoff)))
    con.execute('CREATE TABLE pages (symbol TEXT NOT NULL,start_ms INTEGER NOT NULL,rows_n INTEGER NOT NULL,first_ms INTEGER,last_ms INTEGER,source_sha TEXT NOT NULL,blob_sha TEXT NOT NULL,body BLOB NOT NULL,received_ms INTEGER NOT NULL,PRIMARY KEY(symbol,start_ms))')
    con.execute('CREATE TABLE refs (symbol TEXT PRIMARY KEY,manifest_sha TEXT NOT NULL,payload TEXT NOT NULL)')
    for symbol, rows in rows_by_symbol.items():
        pages = []
        cursor = 0
        for offset in range(0, len(rows), 25):
            body = rows[offset:offset + 25]
            compressed = zlib.compress(json.dumps(body, separators=(',', ':')).encode())
            item = [symbol, cursor, len(body), body[0][0], body[-1][0], digest(body),
                    hashlib.sha256(compressed).hexdigest(), compressed, cutoff + 100]
            if mutate:
                mutate(item, body)
            con.execute('INSERT INTO pages VALUES (?,?,?,?,?,?,?,?,?)', item)
            pages.append([cursor, len(body), body[0][0], body[-1][0], item[5], item[6]])
            cursor = body[-1][0] + STEP
        source = dict(symbol=symbol, cutoff_ms=cutoff, rows_n=len(rows), pages_n=len(pages),
                      first_open_ms=rows[0][0], last_open_ms=rows[-1][0],
                      state='MAXIMUM_AVAILABLE_FETCH_COMPLETE', source_pages=pages)
        con.execute('INSERT INTO refs VALUES (?,?,?)', (symbol, digest(source), json.dumps({
            'evidence_kind': 'SIMULATION', 'alpha_qualified': False, 'verdict': 'NOT_ALPHA_EVIDENCE'})))
    con.commit()
    con.close()


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('binance_demo_bot.rl_direction_data')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def load(self, rows, **kwargs):
        archive(self.root / 'source.sqlite', rows, **kwargs)
        return self.api.stream_archive(self.root / 'source.sqlite', self.root / 'data', cutoff_ms=96 * STEP)

    def test_RDE01_utf8_symbols_match_collector_manifest_hash_without_escaping(self):
        symbol='\u5e01\u5b89\u4eba\u751fUSDT'
        manifest=self.load({symbol:[raw_bar(i) for i in range(96)]})
        self.assertEqual(manifest['symbols'],[symbol])
        self.assertEqual(manifest['entries'][symbol]['regular_rows_n'],96)
        self.assertEqual(self.api._sha({'symbol':symbol}),digest({'symbol':symbol}))

    def test_RDE01_every_symbol_kept_and_gap_short_row_not_fabricated(self):
        source = [raw_bar(i) for i in range(96) if i != 40]
        source[50][6] -= 1
        manifest = self.load({'AAAUSDT': source, 'SHORTUSDT': [raw_bar(i) for i in range(90, 96)]})
        self.assertEqual(manifest['symbols'], ['AAAUSDT', 'SHORTUSDT'])
        entry = manifest['entries']['AAAUSDT']
        grid = np.load(entry['path'], mmap_mode='r')
        self.assertEqual(grid.shape, (96, 7))
        self.assertTrue(np.isnan(grid[40]).all())
        self.assertTrue(np.isnan(grid[51]).all())
        self.assertEqual(entry['raw_rows_n'], 95)
        self.assertEqual(entry['shortened_rows_n'], 1)
        self.assertEqual(entry['gap_rows_n'], 1)
        self.assertEqual(manifest['entries']['SHORTUSDT']['eligible_origins_n'], 0)
        self.assertFalse(manifest['historical_receipts_certified'])
        self.assertEqual(manifest['holdout_status'], 'RETROSPECTIVE_REUSED_HOLDOUT')
        self.assertFalse(manifest['promotion_allowed'])

    def test_RDE01_source_unchanged_and_output_frozen(self):
        archive(self.root / 'source.sqlite', {'AAAUSDT': [raw_bar(i) for i in range(96)]})
        before = (self.root / 'source.sqlite').read_bytes()
        result = self.api.stream_archive(self.root / 'source.sqlite', self.root / 'data', cutoff_ms=96 * STEP)
        again = self.api.stream_archive(self.root / 'source.sqlite', self.root / 'data', cutoff_ms=96 * STEP)
        self.assertEqual(result, again)
        self.assertEqual(before, (self.root / 'source.sqlite').read_bytes())
        entry = result['entries']['AAAUSDT']
        self.assertEqual(entry['sha256'], hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest())
        con = sqlite3.connect(self.root / 'source.sqlite')
        con.execute("UPDATE pages SET blob_sha='wrong' WHERE start_ms=0")
        con.commit()
        con.close()
        with self.assertRaisesRegex(ValueError, 'ARCHIVE_HASH'):
            self.api.stream_archive(self.root / 'source.sqlite', self.root / 'data', cutoff_ms=96 * STEP)

    def test_RDE01_compressed_and_decoded_hashes_both_checked(self):
        for altered in ('blob', 'source'):
            with self.subTest(altered=altered), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                def mutate(item, body):
                    item[6 if altered == 'blob' else 5] = 'wrong'
                archive(path / 'source.sqlite', {'AAAUSDT': [raw_bar(i) for i in range(96)]}, mutate=mutate)
                with self.assertRaisesRegex(ValueError, 'ARCHIVE_HASH'):
                    self.api.stream_archive(path / 'source.sqlite', path / 'data', cutoff_ms=96 * STEP)

    def test_RDE01_schema_and_order_fail_closed_even_with_valid_hash(self):
        for kind in ('shape', 'duplicate', 'future_close', 'receipt', 'row_count'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                source = [raw_bar(i) for i in range(96)]
                if kind == 'shape': source[10] = source[10][:5]
                if kind == 'duplicate': source[10] = raw_bar(9)
                if kind == 'future_close': source[-1][6] = 96 * STEP
                def mutate(item, body):
                    if kind == 'receipt': item[8] = 0
                    if kind == 'row_count': item[2] += 1
                archive(path / 'source.sqlite', {'AAAUSDT': source}, mutate=mutate)
                with self.assertRaisesRegex(ValueError, 'ARCHIVE_SCHEMA|ARCHIVE_ORDER|ARCHIVE_RECEIPT'):
                    self.api.stream_archive(path / 'source.sqlite', path / 'data', cutoff_ms=96 * STEP)

    def test_RDE01_invalid_numbers_are_explicit_unavailable_slots(self):
        source = [raw_bar(i) for i in range(96)]
        source[45][10] = '101'
        manifest = self.load({'AAAUSDT': source})
        entry = manifest['entries']['AAAUSDT']
        grid = np.load(entry['path'], mmap_mode='r')
        self.assertTrue(np.isnan(grid[45]).all())
        self.assertEqual(entry['invalid_rows_n'], 1)
        self.assertNotIn(46, self.api.eligible_origins(grid))

    def test_RDE01_frozen_cutoff_must_match_source(self):
        archive(self.root / 'source.sqlite', {'AAAUSDT': [raw_bar(i) for i in range(96)]})
        with self.assertRaisesRegex(ValueError, 'ARCHIVE_CUTOFF'):
            self.api.stream_archive(self.root / 'source.sqlite', self.root / 'data', cutoff_ms=95 * STEP)


class NumericTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('binance_demo_bot.rl_direction_data')

    def test_RDE02_exact_five_channels_zero_volume(self):
        source = bars(40)
        source[20, 4:6] = 0
        values = self.api.primitives(source)
        self.assertEqual(values.shape, (40, 5))
        self.assertAlmostEqual(values[5, 0], .002, places=6)
        self.assertAlmostEqual(values[5, 1], math.log(1.01 / .99), places=6)
        self.assertAlmostEqual(values[5, 2], 0.)
        self.assertAlmostEqual(values[5, 3], 0.)
        self.assertAlmostEqual(values[5, 4], .2, places=6)
        self.assertEqual(values[20, 4], 0.)

    def test_RDE02_sequence_suffix_mutation_cannot_change_past(self):
        source = bars(80)
        expected = self.api.sequences(source, np.array([40]))
        source[40:] *= 1000
        actual = self.api.sequences(source, np.array([40]))
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual(actual.shape, (1, 32, 5))
        self.assertAlmostEqual(actual[0, -1, 0], .002, places=6)

    def test_RDE01_eligibility_never_checks_future_or_label(self):
        source = bars(40)
        source[35:] = np.nan
        eligible = self.api.eligible_origins(source)
        self.assertIn(35, eligible)
        self.assertNotIn(36, eligible)
        self.assertIn(40, self.api.eligible_origins(bars(40)))
        self.assertNotIn(32, self.api.eligible_origins(bars(40)))

    def test_RDE02_gap_requires_full_causal_warmup(self):
        source = bars(96)
        source[40] = np.nan
        eligible = self.api.eligible_origins(source)
        self.assertIn(40, eligible)
        self.assertNotIn(41, eligible)
        self.assertNotIn(73, eligible)
        self.assertIn(74, eligible)
        with self.assertRaisesRegex(ValueError, 'FEATURE_WINDOW'):
            self.api.sequences(source, np.array([41]))

    def test_RDE03_targets_direct_contiguous_and_fit_mature(self):
        source = bars(50)
        result = self.api.targets(source, np.array([40, 47]), horizons=(1, 4), fit_end_index=50)
        np.testing.assert_allclose(result['returns'][0], [.002, .008], atol=1e-10)
        np.testing.assert_array_equal(result['labels'][0], [1., 1.])
        np.testing.assert_array_equal(result['valid'], [[True, True], [True, False]])
        self.assertTrue(np.isnan(result['returns'][1, 1]))
        source[42] = np.nan
        result = self.api.targets(source, np.array([40]), horizons=(1, 4), fit_end_index=50)
        np.testing.assert_array_equal(result['valid'], [[True, False]])

    def test_RDE03_fit_cutoff_exclusive_and_zero_return_is_down_class(self):
        source = bars(50)
        source[:, :4] = [100, 101, 99, 100]
        result = self.api.targets(source, np.array([40]), horizons=(1, 4), fit_end_index=43)
        np.testing.assert_array_equal(result['valid'], [[True, False]])
        self.assertEqual(result['labels'][0, 0], 0.)
        source[43:] *= 100
        again = self.api.targets(source, np.array([40]), horizons=(1, 4), fit_end_index=43)
        np.testing.assert_array_equal(result['returns'], again['returns'])

    def test_RDE03_training_sample_bounded_deterministic_mature_purged(self):
        source = bars(3000)
        first = self.api.training_indices(source, 0, 2500, limit=256, purge=16)
        np.testing.assert_array_equal(first, self.api.training_indices(source, 0, 2500, limit=256, purge=16))
        self.assertEqual(len(first), 256)
        self.assertGreaterEqual(first.min(), 33)
        self.assertLess(first.max(), 2500 - 16)
        self.assertLess(first.min(), 50)
        self.assertGreater(first.max(), 2450)
        changed = source.copy()
        changed[2500:] = np.nan
        np.testing.assert_array_equal(first, self.api.training_indices(changed, 0, 2500, limit=256, purge=16))


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('binance_demo_bot.rl_direction_data')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def manifest(self, values):
        entries = {}
        for i, (symbol, data) in enumerate(sorted(values.items())):
            path = self.root / (str(i) + '.npy')
            np.save(path, data)
            entries[symbol] = dict(path=str(path), first_ms=0, count=len(data))
        return dict(symbols=sorted(values), entries=entries, global_first_ms=0, cutoff_ms=80 * STEP)

    def test_RDE01_screen_all_symbols_past_only_lexical_tie(self):
        values = {'ZZZUSDT': bars(80), 'AAAUSDT': bars(80)}
        manifest = self.manifest(values)
        screen = self.api.candidate_screen(manifest)
        at = int(np.flatnonzero(screen['times_ms'] == 40 * STEP)[0])
        self.assertEqual(screen['symbols'][int(screen['symbol_indices'][at])], 'AAAUSDT')
        self.assertAlmostEqual(screen['scores'][at], .008 / .0001, places=5)
        values['AAAUSDT'][40:] = np.nan
        values['ZZZUSDT'][40:] *= 1000
        changed = self.api.candidate_screen(self.manifest(values))
        np.testing.assert_array_equal(screen['symbol_indices'][:at + 1], changed['symbol_indices'][:at + 1])
        np.testing.assert_allclose(screen['scores'][:at + 1], changed['scores'][:at + 1])

    def test_RDE01_zero_last_volume_not_selected_but_past_zero_valid(self):
        low = bars(80)
        low[39, 4:6] = 0
        high = bars(80)
        high[20, 4:6] = 0
        result = self.api.candidate_screen(self.manifest({'AAAUSDT': low, 'ZZZUSDT': high}))
        at = int(np.flatnonzero(result['times_ms'] == 40 * STEP)[0])
        self.assertEqual(result['symbols'][int(result['symbol_indices'][at])], 'ZZZUSDT')

    def test_RDE01_missing_future_preserves_candidate_without_substitute(self):
        source = bars(80)
        source[40:] = np.nan
        result = self.api.candidate_screen(self.manifest({'AAAUSDT': source}))
        at = int(np.flatnonzero(result['times_ms'] == 40 * STEP)[0])
        self.assertEqual(int(result['symbol_indices'][at]), 0)
        self.assertEqual(int(result['symbol_indices'][at + 1]), -1)


class BulkPageTests(unittest.TestCase):
    def test_bulk_values_match_scalar_and_keep_short_invalid_counts(self):
        import numpy as np
        from binance_demo_bot import rl_direction_data as api
        rows=[]
        for i in range(5):
            rows.append([i*STEP,'100','101','99','100','10',i*STEP+STEP-1,'1000',10,'5','500','0'])
        rows[1][6]-=1;rows[2][1]=True;rows[3][2]='bad'
        value=api._page_vectors(rows,start=0,first_ms=0,cutoff_ms=5*STEP,previous=None)
        self.assertEqual(value['short_n'],1);self.assertEqual(value['invalid_n'],2)
        self.assertEqual(value['slots'].tolist(),[0,4])
        np.testing.assert_allclose(value['values'],np.asarray([api._row_values(rows[0]),api._row_values(rows[4])]))

    def test_bulk_rejects_schema_timestamp_or_order_without_imputation(self):
        from binance_demo_bot import rl_direction_data as api
        row=[0,'100','101','99','100','10',STEP-1,'1000',10,'5','500','0']
        for changed in ([row,row],[[str(row[0])]+row[1:]],[row[:-1]]):
            with self.assertRaises(ValueError):
                api._page_vectors(changed,start=0,first_ms=0,cutoff_ms=2*STEP,previous=None)


if __name__ == '__main__':
    unittest.main()
