import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import closed_grid_policy_replay as runner
from portfolio_alpha import evaluate_portfolio_alpha


class ClosedGridTests(unittest.TestCase):
    def alpha(self, btc, prices=None, trades=None, days=30):
        return evaluate_portfolio_alpha(trades or [], price_series_by_symbol=prices or {},
            benchmark_series=btc, window_start_ms=900000, window_end_ms=days*86400000+900000,
            requested_days=days, universe=['BTCUSDT', 'A'], variant='test')

    def test_endpoint_span_cannot_certify_coverage(self):
        r = self.alpha([(900000, 100), (30*86400000+900000, 100)])
        self.assertFalse(r['decision_grade'])
        self.assertEqual(r['coverage']['benchmark_grid']['observed'], 2)
        self.assertGreater(r['coverage']['benchmark_grid']['missing'], 2800)
        self.assertIsNone(r['portfolio']['max_drawdown_after_costs_pct'])

    def test_complete_grid_and_closed_trade(self):
        btc = [(t, 100) for t in range(900000, 30*86400000+900001, 900000)]
        trade = {'sym': 'A', 'entry_ts': 900000, 'exit_ts': 1800000,
                 'entry_price': 100, 'exit_price': 101}
        r = self.alpha(btc, {'A': btc}, [trade])
        self.assertTrue(r['decision_grade'])
        self.assertEqual(r['coverage']['benchmark_grid']['missing'], 0)
        self.assertIsNotNone(r['portfolio']['max_drawdown_after_costs_pct'])

    def test_stale_holding_mark_is_missing(self):
        btc = [(t, 100) for t in range(900000, 30*86400000+900001, 900000)]
        trade = {'sym': 'A', 'entry_ts': 900000, 'exit_ts': 2700000,
                 'entry_price': 100, 'exit_price': 101}
        r = self.alpha(btc, {'A': [(900000, 100)]}, [trade])
        self.assertFalse(r['decision_grade'])
        self.assertIn('incomplete_holding_price_grid', r['coverage']['contract_violations'])
        self.assertIsNone(r['portfolio']['max_drawdown_after_costs_pct'])

    def test_bad_closed_grid(self):
        row = {'t': 0, 'o': 1, 'h': 1, 'l': 1, 'c': 1, 'v': 0}
        runner.validate_series([row], 900000, 0, 900000)
        for rows in ([], [row, row], [{**row, 'c': float('nan')}], [{**row, 't': 1}]):
            with self.assertRaises(ValueError):
                runner.validate_series(rows, 900000, 0, 900000)

    def test_clock_excludes_warmup_and_keeps_final_close(self):
        raw, times, count = runner.event_clock({1: ['warmup'], 2: ['valid'], 3: ['end']}, {1, 2}, 2, 3)
        self.assertEqual(raw, {2: ['valid']})
        self.assertEqual(times, {2, 3})
        self.assertEqual(count, 1)

    def test_manifest_hash_mismatch_and_existing_output_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, output = Path(tmp)/'archive', Path(tmp)/'out'
            archive.mkdir()
            start, end = 0, 11*86400000
            item = {'symbol': 'BTCUSDT', 'tf': '15m', 'start': start, 'end': end,
                    'status': 'complete', 'sha256': '0'*64}
            (archive/'recovery_summary.json').write_text(json.dumps({
                'start': start, 'end': end, 'series': [item]}))
            stem = f'BTCUSDT_15m_{start}_{end}'
            (archive/(stem+'.json')).write_text('[]')
            (archive/(stem+'.manifest.json')).write_text(json.dumps(item))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                runner.snapshot_archive(archive, output)
            with self.assertRaises(FileExistsError):
                runner.snapshot_archive(archive, output)


if __name__ == '__main__':
    unittest.main()
