"""Price magnitude targets, native heads and independent calibration contracts."""
import unittest
import tempfile
from pathlib import Path

import numpy as np

from minute_price_paths import targets, train_scale, residual_quantiles, score, build_deeplob_regression
import minute_direction_models as m


class PricePathTests(unittest.TestCase):
    def test_targets_are_exact_displacements_from_one_origin(self):
        mid = np.exp(np.arange(80) * .001)
        actual = targets(mid, np.zeros(80, dtype=int))
        np.testing.assert_allclose(actual[0], np.arange(1, 6) * .006)
        np.testing.assert_allclose(actual[20], actual[0])
        self.assertTrue(np.isnan(actual[-1]).all())

    def test_gap_invalidates_crossing_targets_not_observed_short_targets(self):
        segment = np.zeros(80, dtype=int); segment[10] = -1; segment[11:] = 1
        actual = targets(np.linspace(100, 101, 80), segment)
        self.assertTrue(np.isfinite(actual[0, 0])); self.assertTrue(np.isnan(actual[0, 1:]).all())
        self.assertTrue(np.isnan(actual[10]).all()); self.assertTrue(np.isfinite(actual[11]).all())

    def test_target_scaling_is_frozen_train_only_and_finite_for_constant_column(self):
        train = np.arange(30).reshape(6, 5).astype(float); train[:, 0] = 0
        mean, scale = train_scale(train)
        np.testing.assert_allclose(mean, train.mean(axis=0)); self.assertEqual(scale[0], 1e-8)
        validation = train.copy() * 100
        validation[:] = -999
        np.testing.assert_allclose(train_scale(train)[0], mean)
        np.testing.assert_allclose(train_scale(train)[1], scale)

    def test_residual_quantiles_use_calibration_not_test(self):
        actual = np.tile(np.arange(10)[:, None], (1, 5)).astype(float)
        predicted = np.zeros_like(actual)
        q = residual_quantiles(actual, predicted)
        np.testing.assert_array_equal(q, [9] * 5)
        test = np.ones((100, 5)) * 1000
        metrics = score(test, np.zeros_like(test), q)
        self.assertEqual(metrics[0]['interval_covered'], 0)
        np.testing.assert_array_equal(q, [9] * 5)

    def test_metrics_show_displacement_baseline_and_exact_direction_denominator(self):
        actual = np.tile(np.array([-.01, 0, .01])[:, None], (1, 5))
        prediction = actual * .5
        values = score(actual, prediction, np.ones(5) * .005, np.array([100, 200, 300]))
        for s in values:
            self.assertEqual(s['n'], 3); self.assertEqual(s['direction_n'], 2)
            self.assertEqual(s['direction_correct'], 2); self.assertEqual(s['observed_majority_correct'], 1)
            self.assertAlmostEqual(s['mae_bp'], 100 / 3)
            self.assertAlmostEqual(s['zero_return_mae_bp'], 200 / 3)
            self.assertEqual(s['interval_covered'], 3); self.assertGreater(s['mae_USDT'], 0)
        with self.assertRaises(ValueError): score(actual * np.nan, prediction, np.ones(5))

    def test_reused_purge_remains_strict_for_five_minute_path(self):
        time = np.array([0, 60_000, 300_000, 660_000])
        cuts = dict(start=0, validation=610_000, calibration=1_220_000, test=1_830_000, end=2_440_000)
        mask = m.split_masks(time, np.ones(4, bool), np.zeros((4, 3), dtype=int), cuts)
        self.assertTrue(mask['train'][0]); self.assertFalse(mask['train'][2])

    def test_regression_encoder_has_five_scalar_heads_and_native_roundtrip(self):
        import torch
        from safetensors.torch import save_file, load_file
        torch.set_num_threads(2)
        model = build_deeplob_regression().eval()
        self.assertEqual(len(model.encoder.heads), 5)
        self.assertTrue(all(head.out_features == 1 for head in model.encoder.heads))
        self.assertIsInstance(model.encoder.lstm, torch.nn.LSTM)
        x = torch.randn(2, 100, 40)
        with torch.no_grad(): first = model(x)
        self.assertEqual(first.shape, (2, 5))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'native.safetensors'
            save_file(model.state_dict(), str(path))
            restored = build_deeplob_regression().eval(); restored.load_state_dict(load_file(str(path)))
            with torch.no_grad(): torch.testing.assert_close(restored(x), first)


if __name__ == '__main__': unittest.main()
