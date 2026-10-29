"""RDE11: paired calendar-block effects, absolute earnings and honest N."""
from datetime import date, timedelta
import importlib
import unittest

import _support


def daily(values, start='2026-01-01'):
    first = date.fromisoformat(start)
    return {(first + timedelta(days=i)).isoformat(): value for i, value in enumerate(values)}


class CalendarBlockTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('binance_demo_bot.rl_direction_statistics')

    def compare(self, candidate, reference, **kwargs):
        return self.api.paired_block_intervals(candidate, reference, **kwargs)

    def test_RDE11_constant_paired_and_absolute_effects_are_exact(self):
        result = self.compare(daily([.03] * 35), daily([.01] * 35), draws=101)
        self.assertEqual(result['paired_n'], 35)
        self.assertAlmostEqual(result['delta_mean'], .02)
        self.assertAlmostEqual(result['absolute_mean'], .03)
        for key in ('delta_lower', 'delta_upper'):
            self.assertAlmostEqual(result[key], .02)
        for key in ('absolute_lower', 'absolute_upper'):
            self.assertAlmostEqual(result[key], .03)
        self.assertAlmostEqual(result['confidence'], 1 - .05 / 6)
        self.assertFalse(result['promotion_allowed'])

    def test_RDE11_less_than30_days_has_means_but_null_bounds(self):
        for n in (0, 1, 29):
            result = self.compare(daily([.02] * n), daily([.01] * n), draws=31)
            self.assertEqual(result['paired_n'], n)
            for key in ('delta_lower', 'delta_upper', 'absolute_lower', 'absolute_upper'):
                self.assertIsNone(result[key])
            self.assertEqual(result['state'], 'INCONCLUSIVE')
            if n:
                self.assertAlmostEqual(result['delta_mean'], .01)
            else:
                self.assertIsNone(result['delta_mean'])

    def test_RDE11_pair_must_have_identical_calendar_keys(self):
        with self.assertRaisesRegex(ValueError, 'UNPAIRED_CALENDAR_DAYS'):
            self.compare(daily([0.] * 35), daily([0.] * 35, start='2026-01-02'))

    def test_RDE11_internal_missing_calendar_day_is_rejected_not_compressed(self):
        candidate = daily([0.] * 36)
        del candidate['2026-01-15']
        with self.assertRaisesRegex(ValueError, 'NON_CONTIGUOUS_CALENDAR_DAYS'):
            self.compare(candidate, dict(candidate))

    def test_RDE11_invalid_or_unknown_values_cannot_be_zero_imputed(self):
        for invalid in (None, float('nan'), float('inf'), True, 'missing'):
            candidate = daily([0.] * 35)
            candidate['2026-01-15'] = invalid
            with self.assertRaises(ValueError):
                self.compare(candidate, daily([0.] * 35), draws=31)

    def test_RDE11_canonical_calendar_dates_required(self):
        for key in ('2026-1-01', '20260101', '2026-02-30', 'not-a-day'):
            with self.assertRaises(ValueError):
                self.compare({key: 0.}, {key: 0.})

    def test_RDE11_block_resamples_are_seeded_and_independent_of_mapping_order(self):
        candidate = daily([(i % 9 - 4) / 100 for i in range(65)])
        reference = daily([(i % 5 - 2) / 100 for i in range(65)])
        first = self.compare(candidate, reference, draws=211, seed=42)
        reversed_order = self.compare(dict(reversed(list(candidate.items()))),
                                      dict(reversed(list(reference.items()))), draws=211, seed=42)
        self.assertEqual(first, reversed_order)
        self.assertEqual(first, self.compare(candidate, reference, draws=211, seed=42))
        self.assertNotEqual(first, self.compare(candidate, reference, draws=211, seed=71))

    def test_RDE11_common_reference_translation_preserves_paired_effect(self):
        values = [(i % 9 - 4) / 100 for i in range(65)]
        candidate = daily(values)
        reference = daily([0.] * len(values))
        first = self.compare(candidate, reference, draws=311, confidence=.95)
        shifted = self.compare(daily([v + 1 for v in values]), daily([1.] * len(values)),
                               draws=311, confidence=.95)
        for key in ('delta_mean', 'delta_lower', 'delta_upper'):
            self.assertAlmostEqual(first[key], shifted[key], places=12)
        for key in ('absolute_mean', 'absolute_lower', 'absolute_upper'):
            self.assertAlmostEqual(first[key] + 1, shifted[key], places=12)

    def test_RDE11_daily_market_dependence_is_not_iid_day_sampling(self):
        # Long regimes make dependence materially larger than Monte-Carlo
        # quantile noise; alternating exactly7day regimes are a fragile fixture.
        values = [(.1 if i // 28 % 2 else -.1) for i in range(280)]
        reference = daily([0.] * len(values))
        blocks = self.compare(daily(values), reference, draws=1001, confidence=.95, block_days=7)
        iid = self.compare(daily(values), reference, draws=1001, confidence=.95, block_days=1)
        self.assertGreater(blocks['delta_upper'] - blocks['delta_lower'],
                           1.5 * (iid['delta_upper'] - iid['delta_lower']))

    def test_RDE11_candidate_can_beat_losing_reference_without_positive_earnings(self):
        result = self.compare(daily([-.01] * 35), daily([-.03] * 35), draws=101)
        self.assertGreater(result['delta_lower'], .01)
        self.assertLess(result['absolute_upper'], 0.)

    def test_RDE11_zero_trade_days_are_real_denominator_members(self):
        result = self.compare(daily([0.] * 35), daily([0.] * 35), draws=101)
        self.assertEqual(result['paired_n'], 35)
        self.assertEqual(result['delta_lower'], 0.)
        self.assertEqual(result['absolute_lower'], 0.)

    def test_RDE11_invalid_sampling_protocol_is_rejected(self):
        source = daily([0.] * 35)
        for kwargs in (dict(draws=0), dict(draws=True), dict(confidence=1),
                       dict(confidence=float('nan')), dict(block_days=0),
                       dict(block_days=36), dict(seed=True)):
            with self.assertRaises(ValueError):
                self.compare(source, source, **kwargs)


if __name__ == '__main__':
    unittest.main()
