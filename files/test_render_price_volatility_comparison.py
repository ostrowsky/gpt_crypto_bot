"""Evidence-bound reports must reject incomplete or leaking comparisons."""
import copy
import unittest
import json,hashlib,tempfile
from pathlib import Path
import render_price_volatility_comparison as r


def fixture():
    data=dict(status='COMPLETED_RETROSPECTIVE_DIAGNOSTIC',runtime_eligible=False,
              achievement_claimed=False,start_ms=0,end_ms=1800000,
              benchmark=dict(status='complete',net_return_after_costs_pct=2),
              accounts={},captures={},cost_stress={},folds=[dict(start=0,last_training_label=-1)])
    for name in r.NAMES:
        data['accounts'][name]=dict(curve=[[0,1],[900000,1.01],[1800000,1.02]],
            max_positions=2,net_return_pct=2,max_drawdown_pct=0,
            average_gross_exposure_pct=10,trades=3)
        data['captures'][name]=dict(captured_pair_count=1,label_pair_count=20,
            eligible_trade_count=3,objective_trade_count=1,early_pair_count=1)
        data['cost_stress'][name]=dict(net_return_pct=1)
    return data


class EvidenceTests(unittest.TestCase):
    def test_derived_intervals_recomputed_and_frozen_accounts_protected(self):
        import compare_price_volatility_bot as c
        raw=fixture();grid=list(range(1774994400000,1774994400000+45*c.DAY+c.STEP,c.STEP))
        for name,row in raw['accounts'].items():row['curve']=[[at,.99 if name=='baseline' else 1.0] for at in grid]
        raw['paired_intervals']={}
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);source=folder/'result.json';source.write_text(json.dumps(raw),encoding='utf-8')
            derived=copy.deepcopy(raw);derived['paired_intervals']=c.paired_intervals(raw['accounts'])
            derived.update(derivation_kind='starting-capital-statistics-repair',source_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                derivation_source_hashes={name:hashlib.sha256(Path(r.__file__).with_name(name).read_bytes()).hexdigest() for name in
                    ('compare_price_volatility_bot.py','render_price_volatility_comparison.py')})
            path=folder/'analysis.json';path.write_text(json.dumps(derived),encoding='utf-8')
            self.assertEqual(r.load_evidence(folder)['accounts'],raw['accounts'])
            derived['paired_intervals']['direction']['blocks']['1']['mean_daily_log_gain_bps']=1000
            path.write_text(json.dumps(derived),encoding='utf-8')
            with self.assertRaises(ValueError):r.load_evidence(folder)
            derived['accounts']['direction']['net_return_pct']=100
            path.write_text(json.dumps(derived),encoding='utf-8')
            with self.assertRaises(ValueError):r.load_evidence(folder)
    def test_rows_keep_activity_and_capture_denominators(self):
        rows=r.comparison_rows(fixture())
        self.assertEqual(len(rows),5)
        self.assertEqual(rows[0]['captured'],'1/20')
        self.assertEqual(rows[0]['false_BUY'],'2/3')
        self.assertEqual(rows[0]['btc_alpha_pp'],0)

    def test_cash_arm_preserves_zero_activity(self):
        data=fixture();obj=data['captures']['direction']
        obj.update(eligible_trade_count=0,objective_trade_count=0,captured_pair_count=0,early_pair_count=0)
        data['accounts']['direction']['trades']=0
        row=r.comparison_rows(data)[2]
        self.assertEqual(row['false_BUY'],'0/0')
        self.assertEqual(row['captured'],'0/20')

    def test_rejects_missing_curve_points_and_future_training_label(self):
        data=fixture();data['accounts']['combined']['curve'].pop(1)
        with self.assertRaises(ValueError):r.validate(data)
        data=fixture();data['folds'][0]['last_training_label']=0
        with self.assertRaises(ValueError):r.validate(data)

    def test_rejects_missing_arm_incomplete_benchmark_and_claims(self):
        for key,value in [('runtime_eligible',True),('achievement_claimed',True),('status','PARTIAL')]:
            data=fixture();data[key]=value
            with self.assertRaises(ValueError):r.validate(data)
        data=fixture();del data['accounts']['ewma_control']
        with self.assertRaises(ValueError):r.validate(data)
        data=fixture();data['benchmark']['status']='missing'
        with self.assertRaises(ValueError):r.validate(data)


if __name__=='__main__':unittest.main()
