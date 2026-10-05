"""Single-origin predicted price points never follow subsequently observed prices."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from minute_direction_data import sha
from minute_price_paths import METHODS, score
from plot_minute_direction_prices import build_view
from render_minute_price_paths import build_price_view, render
from test_plot_minute_direction_prices import fixture


class PricePathRenderTests(unittest.TestCase):
    def data(self):
        oldpred, books, cuts = fixture()
        base = build_view(oldpred, 'BTCUSDT', cuts['test'], books['BTCUSDT'])
        r = np.array([.001, -.002, .003, -.004, .005])
        pred = dict(time=oldpred['time'], symbol=oldpred['symbol'], origin_price=np.array([base['origin_price'], 100.126]),
            **{m: np.stack([r, r]) for m in METHODS})
        widths = {m: [.0002] * 5 for m in METHODS}
        return base, pred, widths, oldpred, books, cuts

    def test_points_are_model_displacements_anchored_to_one_known_price(self):
        base, pred, widths, *_ = self.data()
        view = build_price_view(base, pred, widths)
        for m in METHODS:
            np.testing.assert_allclose(view['forecasts'][m]['price'], np.r_[base['origin_price'], base['origin_price'] * np.exp(pred[m][0])])
        self.assertEqual(view['target_ms'], [base['origin_ms'] + h * 60000 for h in range(6)])
        self.assertEqual(view['forecasts']['Ridge']['price'][0], base['origin_price'])

    def test_future_price_mutation_cannot_reanchor_model_forecast(self):
        base, pred, widths, *_ = self.data()
        first = build_price_view(base, pred, widths)
        altered = copy.deepcopy(base)
        altered['price'] = [p * 2 if t > base['origin_ms'] else p for t, p in zip(base['clock_ms'], base['price'])]
        second = build_price_view(altered, pred, widths)
        self.assertEqual(first['forecasts'], second['forecasts'])
        self.assertNotEqual(first['actual_endpoints'], second['actual_endpoints'])

    def test_unknown_future_stays_unknown_with_forecast_preserved(self):
        base, pred, widths, *_ = self.data()
        base['price'] = [p if t <= base['origin_ms'] else None for t, p in zip(base['clock_ms'], base['price'])]
        view = build_price_view(base, pred, widths)
        self.assertEqual(view['actual_endpoints'][1:], [None] * 5)
        self.assertTrue(all(np.isfinite(view['forecasts'][m]['price']).all() for m in METHODS))
        bad = copy.deepcopy(pred); bad['origin_price'][0] *= 2
        with self.assertRaisesRegex(ValueError, 'Origin price mismatch'): build_price_view(base, bad, widths)

    def test_complete_price_report_has_real_model_points_pngs_and_metrics(self):
        base, pred, widths, oldpred, books, cuts = self.data()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); old = root / 'classifier'; output = root / 'regression'; old.mkdir(); output.mkdir()
            np.savez_compressed(old / 'BTCUSDT.npz', **books['BTCUSDT'])
            np.savez_compressed(old / 'test_predictions.npz', **oldpred)
            oldresult = dict(status='COMPLETED_RETROSPECTIVE_L2_DIAGNOSTIC', cuts=cuts,
                coverage={'assets': {'BTCUSDT': {'sha256': sha(old / 'BTCUSDT.npz')}}})
            (old / 'result.json').write_text(json.dumps(oldresult), encoding='utf-8')
            oldreceipt = dict(status='PASS', result_sha256=sha(old / 'result.json'), predictions_sha256=sha(old / 'test_predictions.npz'))
            (old / 'verification.json').write_text(json.dumps(oldreceipt), encoding='utf-8')
            actual = np.ones((2, 5)) * .001
            np.savez_compressed(output / 'predictions.npz', **pred, actual_returns=actual, scored=np.ones(2, dtype=bool))
            scores = {m: dict(pooled=score(actual, pred[m], widths[m]), assets={'BTCUSDT': score(actual, pred[m], widths[m], pred['origin_price'])}) for m in METHODS}
            result = dict(status='COMPLETED_RETROSPECTIVE_PRICE_PATHS',
                registration={'classifier_hashes': dict(result=oldreceipt['result_sha256'], predictions=oldreceipt['predictions_sha256'])},
                coverage=oldresult['coverage'], calibration_widths=widths, metrics=scores)
            (output / 'result.json').write_text(json.dumps(result), encoding='utf-8')
            receipt = dict(status='PASS', result_sha256=sha(output / 'result.json'), predictions_sha256=sha(output / 'predictions.npz'))
            (output / 'verification.json').write_text(json.dumps(receipt), encoding='utf-8')
            render(output, old, old)
            text = (output / 'price_paths.html').read_text(encoding='utf-8')
            self.assertIn('ПРОГНОЗ цены +1..+5 мин', text); self.assertIn('Факт после выдачи', text)
            self.assertIn('Увеличить прогнозный участок', text)
            self.assertNotIn('<script src=', text); self.assertIn('connectgaps:false', text)
            self.assertTrue((output / 'daily_metrics.csv').exists())
            payload = json.loads((output / 'price_paths.json').read_text(encoding='utf-8'))
            self.assertEqual(set(payload['views']['BTCUSDT'][str(base['origin_ms'])]['forecasts']), set(METHODS))
            for m in METHODS: self.assertGreater((output / 'plots' / f'BTCUSDT_{m}.png').stat().st_size, 10000)


if __name__ == '__main__': unittest.main()
