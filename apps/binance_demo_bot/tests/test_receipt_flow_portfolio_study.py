"""RFP08/09 orchestration contracts, no credentials or model fits."""
import unittest
from binance_demo_bot import receipt_flow_portfolio_study as study


class PortfolioStudyContract(unittest.TestCase):
    def test_RFP07_paired_days_exclude_unknown_and_never_zip_unmatched(self):
        left=[dict(day='a',start_ms=0,complete=True,pnl_usdt='2'),
              dict(day='b',start_ms=5,complete=False,pnl_usdt=None),
              dict(day='c',start_ms=10,complete=True,pnl_usdt='9')]
        right=[dict(day='a',start_ms=0,complete=True,pnl_usdt='1'),
               dict(day='b',start_ms=5,complete=True,pnl_usdt='8')]
        out=study.paired_daily(left,right,test_start_ms=0)
        self.assertEqual(out['common_days'],['a'])
        self.assertEqual(out['paired_days_n'],1)
        self.assertEqual(out['mean_delta_usdt'],'1')

    def test_RFP08_all_fixed_cases_include_ablation_and_both_controls(self):
        cases=study.case_definitions()
        self.assertEqual(len(cases),61)
        self.assertEqual(len({x['case_id'] for x in cases}),61)
        arms={x['arm'] for x in cases};self.assertEqual(arms,{'CASH','M0','M1','P','B','F'})
        self.assertEqual(sum(x['arm']=='F' for x in cases),12)

    def test_RFP08_labels_are_not_an_entry_filter(self):
        rows=[dict(at_ms=0,symbol='BTCUSDT',source_row=1,features_sha256='a',
                   micro_bps=2.,spread_bps=1.,predictions={'F_5':3.},targets=None),
              dict(at_ms=5000,symbol='ETHUSDT',source_row=2,features_sha256='b',
                   micro_bps=-2.,spread_bps=1.,predictions={'F_5':4.},targets={5:999})]
        a=study.signal_rows(rows,arm='F',horizon=5,coefficients={})
        rows[0]['targets']={5:-999};rows[1]['targets']=None
        self.assertEqual(a,study.signal_rows(rows,arm='F',horizon=5,coefficients={}))
        self.assertEqual(len(a),2)

    def test_RFP08_micro_linear_mapping_is_frozen_not_calibration_interval(self):
        rows=[dict(at_ms=0,symbol='BTCUSDT',source_row=1,features_sha256='a',
                   micro_bps=2.,spread_bps=1.,predictions={},targets=None)]
        out=study.signal_rows(rows,arm='M1',horizon=10,coefficients={'10':{'intercept':1.,'slope':3.}})
        self.assertEqual(out[0]['score_bps'],7.)


if __name__=='__main__':unittest.main()
