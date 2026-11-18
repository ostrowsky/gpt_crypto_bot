"""TA02/10/11 orchestration interface contracts, first RED."""
import unittest
import _support
from binance_demo_bot.trade_minute_study import complete_gate,verify_golden_reports,minute_plan,reuse_source

class TradeStudyTests(unittest.TestCase):
    def test_TA01_exact_scope_and_no_target_invariant_reuse(self):
        p=dict(seeds=[11,23,37,53,71],splits=[['train',0,900000],['tune',900000,1800000],['test',1800000,2700000]])
        plan=minute_plan(p,dict(CatBoost=0,GRU=0))
        self.assertEqual(len(plan),168);self.assertEqual(sum(reuse_source(t) is not None for t in plan),72)
    def test_TA11_missing_data_cannot_be_financial_pass(self):
        rows=[dict(state='COMPLETE',financial_check=dict(state='PASS'))]*168
        with self.assertRaises(ValueError):complete_gate(83,rows,dict(state='VERIFIED_RECORDED_TRADE_MINUTE_DATA'))
        with self.assertRaises(ValueError):complete_gate(84,rows,dict(state='FAIL'))
        self.assertEqual(complete_gate(84,rows,dict(state='VERIFIED_RECORDED_TRADE_MINUTE_DATA')),'COMPLETE_CONDITIONAL_SIMULATION')
    def test_TA02_missing_prior_golden_not_synthesized(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises((ValueError,FileNotFoundError)):verify_golden_reports(Path(root),Path(root)/'parent')

if __name__=='__main__':unittest.main()
