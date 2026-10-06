import unittest
from render_joint_direction_amplitude import summary,ARMS


class SummaryTests(unittest.TestCase):
    def test_prior_reference_and_common_denominators_preserved(self):
        a=dict(net_return_pct=-20,alpha_pp=-25,test_return_pct=-5,max_drawdown_pct=30,
               trades=12,average_gross_exposure_pct=70,totals=dict(fees=20,slippage=10))
        m=dict(early_pair_count=3,captured_pair_count=5,label_pair_count=15,
               objective_trade_count=7,eligible_trade_count=12)
        r=dict(accounts={n:a for n in ARMS},missions={n:m for n in ARMS},test_missions={n:m for n in ARMS})
        rows=summary(r);self.assertEqual([v['arm'] for v in rows],list(ARMS))
        self.assertEqual(rows[2]['leader_pairs'],15);self.assertEqual(rows[2]['precision_N'],12)
        self.assertEqual(rows[2]['cost'],30);self.assertFalse(rows[2]['runtime_eligible'])


if __name__=='__main__':unittest.main()
