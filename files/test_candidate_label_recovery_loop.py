import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import backfill_candidate_outcomes_v2 as recovery
import critic_dataset
import rl_headless_worker as worker
from test_candidate_dataset_quality import _row


STEP = 900000
START = 1785542400000


def candle(i):
    t = START + i * STEP
    return [t, '100', '102', '99', '101', '20', t + STEP - 1]


class Response:
    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    def raise_for_status(self):
        pass

    async def json(self):
        return self.payload


class Session:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(kwargs['params'])
        return Response(next(self.pages))


class RecoveryTests(unittest.TestCase):
    def fetch(self, payload):
        return asyncio.run(recovery.fetch_kline_range(
            Session([payload]), sym='XUSDT', tf='15m',
            start_ms=START, end_ms=START + 12 * STEP))

    def test_rejects_invalid_exchange_candles(self):
        for field, value in [(4, 'nan'), (4, '-1'), (2, '50'),
                             (5, '-1'), (6, START), (0, START + 1)]:
            payload = [candle(i) for i in range(12)]
            payload[0][field] = value
            with self.subTest(field=field, value=value):
                self.assertIsNone(self.fetch(payload))

    def test_pagination_uses_exact_next_open(self):
        session = Session([[candle(i) for i in range(3)],
                           [candle(i) for i in range(3, 5)]])
        with patch.object(recovery, 'PAGE_LIMIT', 3):
            data = asyncio.run(recovery.fetch_kline_range(
                session, sym='XUSDT', tf='15m', start_ms=START,
                end_ms=START + 4 * STEP))
        self.assertEqual(len(data), 5)
        self.assertEqual(session.calls[1]['startTime'], START + 3 * STEP)

    def test_missing_bar_or_invalid_price_stays_unknown(self):
        for defect in ('gap', 'nan', 'negative'):
            row = {'bar_ts': START, 'tf': '15m', 'labels': {}}
            timestamps = np.arange(12) * STEP + START
            prices = np.full(12, 100.0)
            if defect == 'gap':
                timestamps = np.delete(timestamps, 2)
                prices = np.delete(prices, 2)
            else:
                prices[5] = float('nan') if defect == 'nan' else -1
            critic_dataset._fill_pending_record(row, t_arr=timestamps,
                                                c_arr=prices, bar_ms=STEP)
            self.assertNotIn('ret_5', row['labels'])

    def test_idempotent_labels_and_recovery_evidence(self):
        row = _row(1, action='blocked')
        row.update({'bar_ts': START, 'tf': '15m', 'labels': {'ret_3': 7},
                    'label_provenance': {}})
        kwargs = dict(t_arr=np.arange(12) * STEP + START,
                      c_arr=np.arange(12) + 100., bar_ms=STEP,
                      source='historical_candidate_label_recovery',
                      market_evidence={'closed_series_sha256': 'digest'})
        self.assertTrue(critic_dataset._fill_pending_record(row, **kwargs))
        before = json.dumps(row, sort_keys=True)
        self.assertFalse(critic_dataset._fill_pending_record(row, **kwargs))
        self.assertEqual(before, json.dumps(row, sort_keys=True))
        self.assertEqual(row['labels']['ret_3'], 7)
        self.assertEqual(row['label_provenance']['ret_5']['market_evidence'],
                         kwargs['market_evidence'])

    def test_invalid_provenance_not_recertified(self):
        row = _row(1, action='blocked')
        row['labels'] = {}
        row['provenance']['feature_time'] = 'invalid'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.jsonl'
            path.write_text(json.dumps(row), encoding='utf-8')
            self.assertEqual(recovery.pending_requirements(path), {})

    def test_worker_recovery_independent_of_disabled_collector(self):
        state = worker.WorkerState(3600, 300, 120, 20, False)
        process = AsyncMock()
        process.returncode = 0
        process.communicate.return_value = (json.dumps({
            'remaining_rows': 0, 'evidence_status': 'complete'}).encode(), b'')
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(worker, 'REPORT_DIR', Path(directory)), \
                patch.object(worker.asyncio, 'create_subprocess_exec', return_value=process):
            asyncio.run(worker._recover_candidate_labels(state))
        self.assertEqual(state.label_recovery['evidence_status'], 'complete')
        self.assertFalse(state.collector_enabled)

    def test_recovery_runs_before_training_gate(self):
        state = worker.WorkerState(3600, 300, 120, 20, False)
        with patch.object(worker, '_recover_candidate_labels', new_callable=AsyncMock) as repair, \
                patch.object(worker.config, 'EXIT_FAILURE_ONLINE_LEARNING_ENABLED', False), \
                patch.object(worker, '_restore_training_state'), \
                patch.object(worker, '_count_ranker_rows', return_value=0), \
                patch.object(worker, '_file_mtime', return_value=0), \
                patch.object(worker.asyncio, 'sleep', side_effect=[None, asyncio.CancelledError()]):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(worker._training_loop(state))
        repair.assert_awaited_once_with(state)

    def test_collector_recovers_only_known_permission_failure(self):
        state = worker.WorkerState(3600, 300, 120, 20, True)
        error = critic_dataset.DatasetIntegrityError('replace denied')
        error.__cause__ = PermissionError('Windows file sharing')
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(worker, 'COLLECTOR_STOP_FILE', Path(directory) / 'stop'), \
                patch.object(worker, '_write_status_now', new_callable=AsyncMock), \
                patch.object(worker.data_collector, '_get_btc_context', new=AsyncMock(return_value={})), \
                patch.object(worker.data_collector, '_collect_once',
                             side_effect=[error, {'ok': 1, 'total': 1}]) as collect, \
                patch.object(worker.data_collector, '_seconds_until_next_bar', return_value=900), \
                patch.object(worker.asyncio, 'sleep', side_effect=[None, asyncio.CancelledError()]) as sleep:
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(worker._collector_supervisor(state))
            self.assertFalse(worker.COLLECTOR_STOP_FILE.exists())
        self.assertEqual(collect.call_count, 2)
        self.assertEqual(sleep.call_args_list[0].args, (300,))
        self.assertTrue(state.collector_enabled)

    def test_worker_timeout_and_cancel_terminate_child(self):
        for error in (TimeoutError(), asyncio.CancelledError()):
            state = worker.WorkerState(3600, 300, 120, 20, False)
            process = AsyncMock()
            process.returncode = None
            process.kill = unittest.mock.Mock()
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(worker, 'REPORT_DIR', Path(directory)), \
                    patch.object(worker.asyncio, 'create_subprocess_exec', return_value=process), \
                    patch.object(worker.asyncio, 'wait_for', side_effect=error):
                # Avoid creating an unawaited mock coroutine when wait_for is mocked.
                process.communicate = unittest.mock.Mock(return_value=None)
                if isinstance(error, asyncio.CancelledError):
                    with self.assertRaises(asyncio.CancelledError):
                        asyncio.run(worker._recover_candidate_labels(state))
                else:
                    asyncio.run(worker._recover_candidate_labels(state))
                    self.assertEqual(state.label_recovery['evidence_status'], 'blocked_recovery_error')
                process.kill.assert_called_once()


if __name__ == '__main__':
    unittest.main()
