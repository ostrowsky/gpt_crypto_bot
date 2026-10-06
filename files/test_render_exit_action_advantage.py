import unittest
from render_exit_action_advantage import summary,ARMS


class SummaryTests(unittest.TestCase):
    def test_counts_cash_and_no_promotion_preserved(self):
        a=dict(net_return_pct=-20,alpha_pp=-25,test_return_pct=-5,max_drawdown_pct=30,
               trades=12,average_gross_exposure_pct=70,totals=dict(fees=20,slippage=10))
        m=dict(early_pair_count=3,captured_pair_count=5,label_pair_count=15,
               objective_trade_count=7,eligible_trade_count=12)
        r=dict(accounts={n:a for n in ARMS},missions={n:m for n in ARMS},test_missions={n:m for n in ARMS})
        rows=summary(r);self.assertEqual(rows[1]['leader_pairs'],15);self.assertEqual(rows[1]['precision_N'],12)
        self.assertFalse(rows[1]['runtime_eligible'])


if __name__=='__main__':unittest.main()
