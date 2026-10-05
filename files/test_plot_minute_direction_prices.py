"""Frozen visualization must not acquire information from realized futures."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from minute_direction_data import sha
from plot_minute_direction_prices import (
    METHODS, build_view, category, load_inputs, render, select_origins,
)


def fixture():
    origin = 1_200_000
    time = np.arange(200) * 10_000
    book = np.zeros((len(time), 40))
    book[:, 0] = 100 + np.arange(len(time)) / 1000 + .01
    book[:, 2] = book[:, 0] - .02
    books = {'BTCUSDT': dict(time=time, book=book, segment=np.zeros(len(time), dtype=int))}
    pred = dict(time=np.array([origin, origin + 60_000]), symbol=np.array(['BTCUSDT'] * 2))
    p = np.array([[.7, .2, .1], [.2, .6, .2], [.1, .3, .6]])
    pred.update({m: np.stack([p, p]) for m in METHODS})
    return pred, books, dict(test=origin, end=origin + 60_000)


class PricePlotTests(unittest.TestCase):
    def test_example_selection_is_shared_and_reads_no_future_labels(self):
        pred, books, cuts = fixture()
        extra = copy.deepcopy(books['BTCUSDT']); books['ETHUSDT'] = extra
        pred['time'] = np.r_[pred['time'], pred['time']]
        pred['symbol'] = np.array(['BTCUSDT', 'BTCUSDT', 'ETHUSDT', 'ETHUSDT'])
        chosen = select_origins(pred, books, cuts)
        self.assertEqual(chosen.tolist(), [cuts['test']])
        # Complete future outage cannot remove the already issued example.
        for b in books.values():
            b['segment'][b['time'] > cuts['test']] = -1
        np.testing.assert_array_equal(select_origins(pred, books, cuts), chosen)
        # An inference row without the other asset is never an asymmetric view.
        for b in books.values():
            b['segment'][:] = 0
        pred['symbol'][2] = 'NOT_ETH'
        self.assertEqual(select_origins(pred, books, cuts).tolist(), [cuts['test'] + 60_000])

    def test_future_mutation_changes_only_fact_not_forecast_or_history(self):
        pred, books, cuts = fixture(); book = books['BTCUSDT']
        before = build_view(pred, 'BTCUSDT', cuts['test'], book)
        book['book'][book['time'] > cuts['test'], 0] *= 2
        book['book'][book['time'] > cuts['test'], 2] *= 2
        after = build_view(pred, 'BTCUSDT', cuts['test'], book)
        self.assertEqual(before['forecasts'], after['forecasts'])
        self.assertEqual(before['neutral_price_bounds'], after['neutral_price_bounds'])
        self.assertEqual(before['price'][:121], after['price'][:121])
        self.assertNotEqual(before['facts'], after['facts'])
        for m in METHODS:
            self.assertEqual([f['target_ms'] for f in after['forecasts'][m]],
                [cuts['test'] + h * 60_000 for h in (1, 3, 5)])

    def test_gap_and_unobserved_future_are_null_without_backfill(self):
        pred, books, cuts = fixture(); book = books['BTCUSDT']
        book['segment'][126] = -1  # exact +1-minute target
        view = build_view(pred, 'BTCUSDT', cuts['test'], book)
        self.assertIsNone(view['price'][126]); self.assertEqual(view['facts'][0]['actual_class'], 'UNKNOWN')
        self.assertIsNone(view['facts'][0]['price']); self.assertIsNotNone(view['facts'][1]['price'])
        shortened = {k: v[:130] for k, v in book.items()}
        future = build_view(pred, 'BTCUSDT', cuts['test'], shortened)
        self.assertIsNone(future['facts'][1]['price']); self.assertIsNone(future['facts'][2]['price'])
        self.assertEqual(view['forecasts'], future['forecasts'])

    def test_exact_registered_class_band_not_raw_sign_or_fee_threshold(self):
        self.assertEqual(category(-.00021), 0)
        self.assertEqual(category(-.0002), 1)
        self.assertEqual(category(.0002), 1)
        self.assertEqual(category(.00021), 2)

    def write_fixture(self, folder):
        pred, books, cuts = fixture()
        path = folder / 'BTCUSDT.npz'; np.savez_compressed(path, **books['BTCUSDT'])
        np.savez_compressed(folder / 'test_predictions.npz', **pred)
        result = dict(status='COMPLETED_RETROSPECTIVE_L2_DIAGNOSTIC', cuts=cuts,
            coverage={'assets': {'BTCUSDT': {'sha256': sha(path)}}})
        (folder / 'result.json').write_text(json.dumps(result), encoding='utf-8')
        receipt = dict(status='PASS', result_sha256=sha(folder / 'result.json'),
            predictions_sha256=sha(folder / 'test_predictions.npz'))
        (folder / 'verification.json').write_text(json.dumps(receipt), encoding='utf-8')

    def test_rejects_prediction_and_book_drift_against_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); self.write_fixture(folder)
            load_inputs(folder, folder)
            path = folder / 'test_predictions.npz'; original = path.read_bytes()
            path.write_bytes(original + b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): load_inputs(folder, folder)
            path.write_bytes(original)
            with (folder / 'BTCUSDT.npz').open('ab') as f: f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'source drift'): load_inputs(folder, folder)

    def test_renders_each_method_with_shared_clock_offline_and_static(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); self.write_fixture(folder); render(folder, folder)
            document = (folder / 'price_forecasts.html').read_text(encoding='utf-8')
            self.assertIn('Факт после выдачи', document); self.assertIn('connectgaps:false', document)
            self.assertIn('Plotly.react', document); self.assertNotIn('<script src=', document)
            payload = json.loads((folder / 'price_forecasts.json').read_text(encoding='utf-8'))
            view = payload['views']['BTCUSDT'][str(payload['origins_ms'][0])]
            self.assertEqual(set(view['forecasts']), set(METHODS))
            self.assertEqual(len({tuple(f['target_ms'] for f in v) for v in view['forecasts'].values()}), 1)
            for m in METHODS:
                self.assertGreater((folder / 'price_forecast_plots' / f'BTCUSDT_{m}.png').stat().st_size, 10000)


if __name__ == '__main__':
    unittest.main()
