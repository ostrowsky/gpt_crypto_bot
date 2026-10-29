"""PFD01..05: credential-free public flow capture and durable evidence."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binance_demo_bot import public_flow_dataset as flow


def ms(text):
    return int(datetime.fromisoformat(text).timestamp() * 1000)


def book(update=10, bids=None, asks=None):
    return json.dumps(dict(stream='btcusdt@depth20@100ms', data=dict(
        lastUpdateId=update,
        bids=bids or [['100', '2'], ['99', '3']],
        asks=asks or [['101', '4'], ['102', '5']])), separators=(',', ':'))


def trade(aggregate=1, first=10, last=11, price='100'):
    return json.dumps(dict(stream='btcusdt@aggTrade', data=dict(
        e='aggTrade', E=1000, s='BTCUSDT', a=aggregate, p=price, q='2',
        f=first, l=last, T=999, m=False, M=True)), separators=(',', ':'))


class PublicFlowContract(unittest.TestCase):
    def setUp(self):
        self.validator = flow.FlowValidator()
        self.validator.begin_session('s1', wall_ms=1000, mono_ns=1_000_000)

    def ingest(self, raw, wall=1001, mono=2_000_000, session='s1'):
        return self.validator.ingest(raw, wall_ms=wall, mono_ns=mono,
                                     session_id=session)

    def test_PFD01_fixed_public_demo_origin_and_no_private_dependencies(self):
        registration = flow.make_registration(ms('2026-10-29T00:00:00+00:00'))
        self.assertEqual(registration['origin'], 'wss://demo-stream.binance.com/stream')
        self.assertEqual(len(registration['streams']), 6)
        self.assertTrue(all(s.endswith(('@aggTrade', '@depth20@100ms'))
                            for s in registration['streams']))
        text = Path(flow.__file__).read_text(encoding='utf-8')
        tree = ast.parse(text)
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any(n.level for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)))
        self.assertFalse(set(imports) & {'dotenv', 'binance_demo_bot.adapter',
                                        'binance_demo_bot.isolation'})
        self.assertNotIn('BINANCE_DEMO_API_KEY', text)
        self.assertNotIn('BINANCE_DEMO_API_SECRET', text)

    def test_PFD01_bad_json_topic_symbol_and_values_preserve_raw(self):
        bads = ['{', json.dumps(dict(stream='xrpusdt@aggTrade', data={})),
                trade(price='NaN'), trade(price='0')]
        mismatch = json.loads(trade()); mismatch['data']['s'] = 'ETHUSDT'
        bads.append(json.dumps(mismatch))
        for raw in bads:
            with self.subTest(raw=raw):
                result = self.ingest(raw)
                self.assertFalse(result['validation']['valid'])
                self.assertTrue(result['validation']['fatal'])
                self.assertEqual(result['raw'], raw)

    def test_PFD02_raw_receipts_session_and_trade_sequence(self):
        raw = trade(); result = self.ingest(raw)
        self.assertTrue(result['validation']['valid'])
        self.assertEqual(result['raw'], raw)
        self.assertEqual((result['wall_ms'], result['mono_ns'], result['session_id']),
                         (1001, 2_000_000, 's1'))
        self.assertEqual(result['validation']['continuity'], 'UNANCHORED')
        result = self.ingest(trade(2, 12, 14), wall=1002, mono=3_000_000)
        self.assertEqual(result['validation']['continuity'], 'CONTIGUOUS')
        self.assertEqual(result['validation']['first_trade_id'], 12)
        self.assertEqual(result['validation']['last_trade_id'], 14)

    def test_PFD02_both_aggregate_and_underlying_trade_gaps_recorded(self):
        self.ingest(trade())
        result = self.ingest(trade(4, 15, 16), wall=1002, mono=3_000_000)
        self.assertTrue(result['validation']['valid'])
        self.assertFalse(result['validation']['fatal'])
        self.assertEqual(result['validation']['continuity'], 'GAP')
        self.assertIn('AGGREGATE_ID_GAP', result['validation']['reasons'])
        self.assertIn('TRADE_ID_GAP', result['validation']['reasons'])

    def test_PFD02_session_reset_does_not_assert_cross_session_continuity(self):
        self.ingest(trade(20, 100, 102))
        self.validator.begin_session('s2', wall_ms=2000, mono_ns=4_000_000)
        result = self.ingest(trade(1, 1, 2), wall=2001, mono=5_000_000, session='s2')
        self.assertEqual(result['validation']['continuity'], 'UNANCHORED')
        self.assertEqual(self.validator.summary()['reconnections_n'], 1)

    def test_PFD02_receipt_gap_and_regression_never_become_zero_flow(self):
        self.ingest(trade())
        result = self.ingest(trade(2, 12, 13), wall=7002, mono=6_003_000_000)
        self.assertIn('STALE_RECEIPT_GAP', result['validation']['reasons'])
        self.assertEqual(result['validation']['continuity'], 'GAP')
        result = self.ingest(trade(3, 14, 15), wall=7001, mono=6_004_000_000)
        self.assertFalse(result['validation']['valid'])
        self.assertIn('RECEIPT_CLOCK_REGRESSION', result['validation']['reasons'])

    def test_PFD03_partial_depth_is_snapshot_and_exchange_age_unknown(self):
        result = self.ingest(book())
        self.assertTrue(result['validation']['valid'])
        self.assertEqual(result['validation']['book_semantics'], 'STANDING_TOP20_SNAPSHOT')
        self.assertEqual(result['validation']['exchange_age'], 'UNKNOWN')
        # Partial depth skips update IDs legitimately; it is not a diff sequence.
        result = self.ingest(book(999), wall=1002, mono=3_000_000)
        self.assertTrue(result['validation']['valid'])
        self.assertNotIn('BOOK_UPDATE_GAP', result['validation']['reasons'])

    def test_PFD03_duplicate_conflict_and_backwards_ids(self):
        self.ingest(book())
        duplicate = self.ingest(book(), wall=1002, mono=3_000_000)
        self.assertEqual(duplicate['validation']['continuity'], 'DUPLICATE')
        conflict = self.ingest(book(bids=[['100', '9']]), wall=1003, mono=4_000_000)
        self.assertFalse(conflict['validation']['valid'])
        self.assertIn('ID_CONFLICT', conflict['validation']['reasons'])
        backward = self.ingest(book(9), wall=1004, mono=5_000_000)
        self.assertFalse(backward['validation']['valid'])
        self.assertIn('BACKWARD_ID', backward['validation']['reasons'])

    def test_PFD03_crossed_unsorted_or_zero_levels_are_invalid(self):
        bads = [book(asks=[['100', '2']]), book(bids=[['99', '2'], ['100', '3']]),
                book(asks=[['102', '1'], ['101', '2']]), book(bids=[['100', '0']])]
        for raw in bads:
            with self.subTest(raw=raw):
                self.assertFalse(self.ingest(raw)['validation']['valid'])

    def test_PFD01_server_shutdown_is_preserved_reconnect_control(self):
        raw = json.dumps(dict(stream='!serverShutdown', data=dict(e='serverShutdown', E=1000)))
        result = self.ingest(raw)
        self.assertTrue(result['validation']['reconnect'])
        self.assertFalse(result['validation']['fatal'])
        self.assertEqual(result['raw'], raw)

    def test_PFD01_official_host_document_is_verified_before_collection(self):
        result = flow.verify_official_host(fetch=lambda: b'official table: wss://demo-stream.binance.com/stream')
        self.assertEqual(len(result['sha256']), 64)
        self.assertEqual(result['origin'], flow.ORIGIN)
        with self.assertRaisesRegex(flow.FlowError, 'OFFICIAL_HOST_NOT_VERIFIED'):
            flow.verify_official_host(fetch=lambda: b'wss://unrelated.example/ws')

    def test_PFD01_nonstandard_nan_json_stays_raw_and_fails_closed(self):
        raw = trade().replace('"q":"2"', '"q":NaN')
        result = self.ingest(raw)
        self.assertFalse(result['validation']['valid'])
        self.assertTrue(result['validation']['fatal'])
        self.assertEqual(result['raw'], raw)

    def test_PFD02_invalid_timestamp_and_maker_types_are_rejected(self):
        for key, value in [('E', True), ('T', 0), ('m', 1), ('T', 1001)]:
            frame = json.loads(trade()); frame['data'][key] = value
            with self.subTest(key=key, value=value):
                self.assertFalse(self.ingest(json.dumps(frame))['validation']['valid'])

    def test_PFD02_underlying_trade_id_backwards_or_overlap_is_fatal(self):
        self.ingest(trade())
        for first in (1, 10, 11):
            result = self.ingest(trade(2, first, 13), wall=1002, mono=3_000_000)
            self.assertFalse(result['validation']['valid'])
            self.assertTrue(result['validation']['fatal'])
            self.assertIn('BACKWARD_TRADE_ID', result['validation']['reasons'])

    def test_PFD02_source_timestamp_regression_is_not_sorted_away(self):
        self.ingest(trade())
        frame = json.loads(trade(2, 12, 13))
        frame['data'].update(E=999, T=998)
        result = self.ingest(json.dumps(frame), wall=1002, mono=3_000_000)
        self.assertFalse(result['validation']['valid'])
        self.assertIn('SOURCE_TIME_REGRESSION', result['validation']['reasons'])

    def test_PFD02_clock_regression_before_first_frame_is_terminal(self):
        with self.assertRaisesRegex(flow.FlowError, 'RECEIPT_CLOCK_REGRESSION'):
            self.validator.begin_session('s2', wall_ms=999, mono_ns=2_000_000)


class PublicFlowPersistence(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / '.runtime' / 'unit-fixtures'
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='public-flow-', dir=root)
        self.folder = Path(self.temp.name) / 'dataset'
        self.reg = flow.make_registration(ms('2026-10-29T12:00:00+00:00'))
        self.free = lambda: 100 * 1024**3

    def tearDown(self):
        self.temp.cleanup()

    def record(self, raw='exact raw string'):
        return dict(raw=raw, wall_ms=1000, mono_ns=1, session_id='s1',
                    stream='btcusdt@aggTrade', validation=dict(valid=True))

    def test_PFD04_local_calendar_schedule_handles_both_DST_transitions(self):
        for start, hours in [('2026-03-27T12:00:00+00:00', 167),
                             ('2026-10-23T12:00:00+00:00', 169)]:
            reg = flow.make_registration(ms(start))
            self.assertEqual((reg['planned_end_ms']-reg['full_days_start_ms']) // 3_600_000, hours)
            self.assertEqual(reg['full_calendar_days'], 7)
        self.assertEqual(self.reg['max_compressed_bytes'], 8 * 1024**3)
        self.assertEqual(self.reg['minimum_free_bytes'], 20 * 1024**3)

    def test_PFD04_schedule_stop_quota_and_disk_are_explicit(self):
        common = dict(compressed_bytes=0, free_bytes=self.free(), stop_requested=False)
        self.assertIsNone(flow.stop_reason(self.reg, now_ms=self.reg['started_at_ms'], **common))
        self.assertEqual(flow.stop_reason(self.reg, now_ms=self.reg['planned_end_ms'], **common), 'PLANNED_END')
        self.assertEqual(flow.stop_reason(self.reg, now_ms=self.reg['started_at_ms'], **dict(common, stop_requested=True)), 'OWN_STOP')
        self.assertEqual(flow.stop_reason(self.reg, now_ms=self.reg['started_at_ms'], **dict(common, free_bytes=20*1024**3-1)), 'DISK_RESERVE')
        self.assertEqual(flow.stop_reason(self.reg, now_ms=self.reg['started_at_ms'], **dict(common, compressed_bytes=8*1024**3)), 'COMPRESSED_BUDGET')

    def test_PFD04_output_cannot_escape_own_runtime(self):
        app = Path(__file__).resolve().parents[1]
        self.assertTrue(flow.own_runtime_path(app).is_relative_to(app / '.runtime'))
        with self.assertRaises(flow.FlowError):
            flow.own_runtime_path(app, app.parent / 'legacy')

    def test_PFD05_gzip_roundtrip_hash_and_exact_raw(self):
        store = flow.ChunkStore(self.folder, self.reg, free_bytes=self.free)
        raw = '{"malformed but retained":\n"raw"}'
        store.append(self.record(raw), monotonic_ns=1)
        store.flush()
        store.close()
        result = flow.verify_dataset(self.folder)
        self.assertEqual(result['records_n'], 1)
        self.assertEqual(result['chunks_n'], 1)
        manifest = json.loads((self.folder / 'manifest.json').read_text())
        payload = gzip.decompress((self.folder / manifest['chunks'][0]['file']).read_bytes())
        self.assertEqual(json.loads(payload)['raw'], raw)

    def test_PFD05_tampering_and_registration_overwrite_rejected(self):
        store = flow.ChunkStore(self.folder, self.reg, free_bytes=self.free)
        store.append(self.record(), monotonic_ns=1); store.close()
        with self.assertRaisesRegex(flow.FlowError, 'DATASET_EXISTS'):
            flow.ChunkStore(self.folder, self.reg, free_bytes=self.free)
        p = next(self.folder.glob('chunk-*.jsonl.gz'))
        p.write_bytes(p.read_bytes() + b'tamper')
        with self.assertRaisesRegex(flow.FlowError, 'CHUNK_INTEGRITY'):
            flow.verify_dataset(self.folder)

    def test_PFD05_time_and_size_rollover_are_bounded(self):
        store = flow.ChunkStore(self.folder, self.reg, free_bytes=self.free, chunk_bytes=256)
        store.append(self.record(), monotonic_ns=1)
        self.assertTrue(store.flush_due(60_000_000_001))
        for i in range(3): store.append(self.record(str(i)), monotonic_ns=61_000_000_000+i)
        store.close()
        manifest = json.loads((self.folder / 'manifest.json').read_text())
        self.assertGreaterEqual(len(manifest['chunks']), 3)
        self.assertTrue(all(c['uncompressed_bytes'] <= 256 for c in manifest['chunks']))

    def test_PFD05_durable_manifest_failure_aborts_and_preserves_chunk(self):
        store = flow.ChunkStore(self.folder, self.reg, free_bytes=self.free)
        store.append(self.record(), monotonic_ns=1)
        with patch.object(flow, '_atomic_json', side_effect=OSError('disk write failure')):
            with self.assertRaisesRegex(flow.FlowError, 'PERSISTENCE_FAILED'): store.flush()
        self.assertTrue(store.failed)
        self.assertEqual(len(list(self.folder.glob('chunk-*.jsonl.gz'))), 1)
        store.close()

    def test_PFD05_low_disk_or_byte_cap_cannot_publish_a_chunk(self):
        store = flow.ChunkStore(self.folder, self.reg, free_bytes=self.free)
        store.append(self.record(), monotonic_ns=1)
        store.free_bytes = lambda: 20 * 1024**3
        with self.assertRaisesRegex(flow.FlowError, 'DISK_RESERVE'): store.flush()
        self.assertFalse(list(self.folder.glob('chunk-*.jsonl.gz')))
        store.close()

    def test_PFD04_fake_transport_own_stop_persists_raw_and_final_status(self):
        folder = self.folder
        class Connection:
            request = type('Request', (), dict(path='/stream?streams='+'/'.join(flow.STREAMS),
                                               headers={'Host': 'demo-stream.binance.com'}))()
            def __init__(self): self.frames = iter([book(), trade()])
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def recv(self, timeout):
                try: return next(self.frames)
                except StopIteration:
                    (folder/'stop').write_text('test stop')
                    raise TimeoutError
        with patch('websockets.sync.client.connect', return_value=Connection()) as connect, \
                patch.object(flow, '_below_normal'), \
                patch.object(flow, 'verify_official_host', return_value=dict(sha256='a'*64, origin=flow.ORIGIN)):
            result = flow.run_capture(folder)
        self.assertEqual(result['state'], 'INCOMPLETE')
        self.assertEqual(result['stop_reason'], 'OWN_STOP')
        self.assertEqual(connect.call_args.args[0], flow.URL)
        self.assertEqual(result['messages_per_topic']['btcusdt@aggTrade'], 1)
        self.assertGreaterEqual(flow.verify_dataset(folder)['records_n'], 4)

    def test_PFD04_failed_handshake_never_leaves_collecting_status(self):
        from websockets.exceptions import InvalidHandshake
        with patch('websockets.sync.client.connect', side_effect=InvalidHandshake('test failure')), \
                patch.object(flow, '_below_normal'), \
                patch.object(flow, 'verify_official_host', return_value=dict(sha256='a'*64, origin=flow.ORIGIN)):
            result = flow.run_capture(self.folder)
        self.assertEqual(result['state'], 'INCOMPLETE')
        self.assertEqual(result['stop_reason'], 'HANDSHAKE_FAILED')

    def test_PFD01_redirected_host_cannot_supply_market_frames(self):
        class Connection:
            request = type('Request', (), dict(path='/stream?streams='+'/'.join(flow.STREAMS),
                                               headers={'Host': 'other.example'}))()
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def recv(self, timeout): raise AssertionError('wrong venue read')
        with patch('websockets.sync.client.connect', return_value=Connection()), \
                patch.object(flow, '_below_normal'), \
                patch.object(flow, 'verify_official_host', return_value=dict(sha256='a'*64, origin=flow.ORIGIN)):
            result = flow.run_capture(self.folder)
        self.assertEqual(result['state'], 'INCOMPLETE')
        self.assertEqual(result['stop_reason'], 'HANDSHAKE_TARGET_MISMATCH')


if __name__ == '__main__':
    unittest.main()
