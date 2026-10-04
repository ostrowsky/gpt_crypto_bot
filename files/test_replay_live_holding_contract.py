"""Base picker contract only; deliberately not a full live BUY parity claim."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import config
import replay_backtest as rb
from audit_live_buy_replay_contract import live_hourly_hold


class HoldingContractTests(unittest.TestCase):
    def pick(self, branch, tf, *, trend_mode='trend', early=False):
        detectors = ('entry', 'breakout', 'retest', 'trend_surge', 'impulse', 'alignment')
        with ExitStack() as stack:
            for detector in detectors:
                stack.enter_context(patch.object(rb, 'check_' + detector + '_conditions',
                    return_value=(branch == detector, 'fixture')))
            stack.enter_context(patch.object(rb, 'get_effective_entry_mode',
                return_value=(trend_mode, early)))
            stack.enter_context(patch.object(config, 'ALIGNMENT_BUY_ENABLED', True))
            return rb._entry_candidate({}, 30, np.ones(64), tf)

    def test_all_base_branches_follow_timeframe_not_hardcoded_48(self):
        with patch.object(config, 'MAX_HOLD_BARS_15M', 37, create=True), \
             patch.object(config, 'MAX_HOLD_BARS', 11):
            for tf, expected in (('15m', 37), ('1h', 11), ('4h', 11)):
                for branch in ('entry', 'trend_surge', 'impulse', 'alignment'):
                    with self.subTest(tf=tf, branch=branch):
                        self.assertEqual(self.pick(branch, tf)[2], expected)

    def test_breakout_and_retest_keep_independent_limits(self):
        with patch.object(config, 'MAX_HOLD_BARS_BREAKOUT', 5), \
             patch.object(config, 'MAX_HOLD_BARS_RETEST', 9):
            for tf in ('15m', '1h'):
                self.assertEqual(self.pick('breakout', tf)[2], 5)
                self.assertEqual(self.pick('retest', tf)[2], 9)

    def test_default_15m_limit_when_override_is_absent(self):
        with patch.dict(config.__dict__):
            config.__dict__.pop('MAX_HOLD_BARS_15M', None)
            for branch in ('entry', 'trend_surge', 'impulse', 'alignment'):
                self.assertEqual(self.pick(branch, '15m')[2], 48)

    def test_mode_stop_and_continuation_preserved(self):
        with patch.object(config, 'ATR_TRAIL_K_STRONG', 3.25):
            for mode in ('strong_trend', 'impulse_speed'):
                row = self.pick('entry', '1h', trend_mode=mode, early=True)
                self.assertEqual(row, (mode, 3.25, config.MAX_HOLD_BARS, True))
            self.assertEqual(self.pick('trend_surge', '1h')[1], 3.25)
        self.assertIsNone(self.pick('none', '1h'))

    def test_real_live_source_assignment_matches_custom_hourly_limit(self):
        with patch.object(config, 'MAX_HOLD_BARS', 13):
            live, _ = live_hourly_hold()
            self.assertEqual(self.pick('entry', '1h')[2], live)
            self.assertEqual(live, 13)


if __name__ == '__main__':
    unittest.main()
