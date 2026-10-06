import unittest
from render_impulse_entry_catboost import summary


class SummaryTests(unittest.TestCase):
    def test_common_account_and_mission_denominators_preserved(self):
        a=dict(net_return_pct=-20,alpha_pp=-25,test_return_pct=-5,max_drawdown_pct=30,
               trades=12,average_gross_exposure_pct=70,totals=dict(fees=20,slippage=10))
        m=dict(early_pair_count=3,captured_pair_count=5,label_pair_count=15,
               objective_trade_count=7,eligible_trade_count=12)
        r=dict(accounts={n:a for n in ('control','catboost')},missions={n:m for n in ('control','catboost')},
               test_missions={n:m for n in ('control','catboost')})
        rows=summary(r)
        self.assertEqual(rows[0]['net_return_pct'],-20);self.assertEqual(rows[1]['leader_pairs'],15)
        self.assertEqual(rows[1]['precision_n'],7);self.assertEqual(rows[1]['precision_N'],12)
        self.assertFalse(rows[1]['runtime_eligible'])


if __name__=='__main__':unittest.main()
