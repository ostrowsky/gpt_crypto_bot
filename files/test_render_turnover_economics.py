import unittest
from render_turnover_economics import summary_rows,exit_diagnostics
from turnover_economics import ARMS


class ReportTests(unittest.TestCase):
    def test_missing_exit_labels_are_not_zero_quality(self):
        r=exit_diagnostics([{'entry_price':100,'exit_price':101,'exit_efficiency':None}])
        self.assertEqual(r['winning_retention_n'],0)
        self.assertIsNone(r['winning_retention_mean']);self.assertIsNone(r['giveback_mean_pct'])

    def test_favorable_return_does_not_hide_failed_goal_gate(self):
        account=dict(net_return_pct=5.,alpha_pp=3.,test_return_pct=2.,max_drawdown_pct=10.,
            average_gross_exposure_pct=20.,trades=100,replacements=0)
        mission=dict(early_pair_count=1,captured_pair_count=2,label_pair_count=10,
                     objective_trade_count=2,eligible_trade_count=10)
        r=dict(accounts={k:account for k in ARMS},missions={k:mission for k in ARMS},
               comparisons={k:{'numerical_gate':'REJECTED'} for k in ARMS[1:]})
        rows=summary_rows(r)
        self.assertEqual(len(rows),4)
        self.assertTrue(all(x['numerical_gate']=='REJECTED' for x in rows[1:]))
        self.assertTrue(all(x['runtime_eligible'] is False for x in rows))
        self.assertEqual(rows[0]['leader_pairs'],10)


if __name__=='__main__':unittest.main()
