import unittest
from report_capacity_experiment import trade_diagnostics


class CostDiagnosticsTests(unittest.TestCase):
    def test_unknown_and_partial_do_not_become_full_exit_successes(self):
        row={'entry_ts':1000,'exit_ts':61000,'entry_price':100,'exit_price':101}
        r=trade_diagnostics([row,{**row,'partial_exit_taken':True},{**row,'exit_price':0},
                             {**row,'exit_price':100.1}])
        self.assertEqual(r['trades_total'],4)
        self.assertEqual(r['closed_known'],3);self.assertEqual(r['unknown'],1)
        self.assertEqual(r['partial_exits_excluded_from_hurdle'],1)
        self.assertEqual(r['simple_full_exits'],2)
        self.assertEqual(r['gross_positive_count'],2)
        self.assertEqual(r['gross_move_exceeds_cost_count'],1)
        self.assertAlmostEqual(r['roundtrip_cost_hurdle_pct'],.25)

    def test_empty_or_unclosed_cannot_claim_zero_quality(self):
        r=trade_diagnostics([{'entry_ts':1000,'exit_ts':0,'entry_price':1,'exit_price':2}])
        self.assertEqual(r['unknown'],1)
        self.assertIsNone(r['median_hold_minutes'])
        self.assertEqual(r['simple_full_exits'],0)


if __name__=='__main__':unittest.main()
