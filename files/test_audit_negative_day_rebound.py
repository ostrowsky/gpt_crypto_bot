import unittest
from audit_negative_day_rebound import adjusted, trade_summary, windows, DAY, bounded_cache, finalize_at_boundary
from replay_backtest import ReplayCandidate, ReplayTrade
import numpy as np


def candidate(**kwargs):
    values = dict(sym='TRXUSDT', tf='15m', mode='breakout', ts_ms=1, i=1,
                  price=1, trail_k=2.5, max_hold_bars=48, score=118.44,
                  top_gainer_score=8.79, intraday_change_pct=-1.0059)
    values.update(kwargs)
    return ReplayCandidate(**values)


class ReboundAuditTests(unittest.TestCase):
    def test_trx_still_blocked(self):
        c = candidate()
        self.assertEqual(adjusted(c, 'extra_penalty_off').top_gainer_score, 16.79)
        self.assertEqual(adjusted(c, 'negative_terms_off').top_gainer_score, 22.8254)
        self.assertLess(adjusted(c, 'negative_terms_off').top_gainer_score, 34)
        self.assertEqual(c.top_gainer_score, 8.79)

    def test_boundaries(self):
        c = candidate(intraday_change_pct=-.25)
        self.assertEqual(adjusted(c, 'extra_penalty_off').top_gainer_score, 8.79)
        self.assertEqual(adjusted(c, 'negative_terms_off').top_gainer_score, 10.29)
        self.assertEqual(adjusted(candidate(intraday_change_pct=-5), 'negative_terms_off').top_gainer_score, 24.79)

    def test_unchanged_scope(self):
        for changes in ({'tf': '1h'}, {'mode': 'strong_trend'}, {'intraday_change_pct': 0},
                        {'intraday_change_pct': 1}):
            c = candidate(**changes)
            for arm in ('baseline', 'extra_penalty_off', 'negative_terms_off'):
                self.assertEqual(adjusted(c, arm).top_gainer_score, c.top_gainer_score)
                self.assertIsNot(adjusted(c, arm), c)

    def test_invalid_arm(self):
        with self.assertRaises(ValueError):
            adjusted(candidate(), 'off')

    def test_partition(self):
        full, a, b, c = windows(0, 181*DAY)
        self.assertEqual(a[1], full[1])
        self.assertEqual(a[2], b[1])
        self.assertEqual(b[2], c[1])
        self.assertEqual(c[2], full[2])
        self.assertEqual(sum(x[2]-x[1] for x in (a,b,c)), 181*DAY)
        with self.assertRaises(ValueError):
            windows(0, 5*DAY+1)

    def test_missing_denominator(self):
        self.assertIsNone(trade_summary([], [])['added_net_losing_fraction'])

    def test_closed_boundary_and_original_indices(self):
        data = np.array([(0, 1.), (900000, 2.), (1800000, 99.)], dtype=[('t','i8'),('c','f8')])
        cache = {('TRXUSDT','15m'): (data, {'rsi': np.array([40,50,90])})}
        local = bounded_cache(cache, 1800000)
        self.assertEqual(local['TRXUSDT','15m'][0]['c'].tolist(), [1,2])
        self.assertEqual(local['TRXUSDT','15m'][1]['rsi'].tolist(), [40,50])
        trade = ReplayTrade('TRXUSDT','15m','breakout',900000,1,0,2.5,48,0,
                            exit_reason='open_at_end')
        finalize_at_boundary([trade], local, 1800000)
        self.assertEqual(trade.exit_price, 2)
        self.assertEqual(trade.exit_ts, 1800000)
        trade.exit_reason = 'RSI'; trade.exit_ts = 1800001
        with self.assertRaises(ValueError):
            finalize_at_boundary([trade], local, 1800000)


if __name__ == '__main__':
    unittest.main()
