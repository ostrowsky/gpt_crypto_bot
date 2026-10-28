"""PIL-07..08 causal simulation and honest learning integration."""
import importlib
import unittest
import _support


def bars(n=400):
    return [[i*900000,'100','101','99','100','10',i*900000+899999,'1000',10,'1','100','0'] for i in range(n)]


class ResearchTests(unittest.TestCase):
    def setUp(self):self.api=importlib.import_module('binance_demo_bot.pilot_research')
    def test_pil07_ambiguous_bar_stop_first_with_costs(self):
        result=self.api.reference_replay(bars(30))
        self.assertGreater(result['trades_n'],0);self.assertLess(float(result['net_usdt']),0)
        self.assertEqual(result['evidence_kind'],'SIMULATION');self.assertFalse(result['alpha_qualified'])
    def test_pil07_no_trade_before_prior_closed_features(self):
        result=self.api.reference_replay(bars(30))
        for row in result['rows']:
            self.assertLess(row['feature_end_ms'],row['origin_ms']);self.assertGreater(row['label_end_ms'],row['origin_ms'])
    def test_pil07_gap_disclosed_not_interpolated(self):
        data=bars(40);data.pop(10)
        result=self.api.reference_replay(data);self.assertGreater(result['gaps_n'],0)
    def test_pil08_future_mutation_leaves_prior_training_unchanged(self):
        data=bars(500);first=self.api.fit_diagnostic(data,train_fraction=0.8)
        for row in data[450:]:row[4]='110'
        second=self.api.fit_diagnostic(data,train_fraction=0.8)
        self.assertEqual(first['weights'],second['weights']);self.assertEqual(first['scaler'],second['scaler'])
    def test_pil08_actual_insufficient_rows_does_not_fake_fit(self):
        result=self.api.learning_handoff([],fit_cutoff_ms=10000)
        self.assertEqual(result['state'],'NOT_TRAINED');self.assertFalse(result['promotion_allowed'])
    def test_pil08_future_or_open_outcome_excluded(self):
        rows=[dict(state='OPEN',label_available_at_ms=1),dict(state='CLOSED',label_available_at_ms=20000)]
        result=self.api.learning_handoff(rows,fit_cutoff_ms=10000)
        self.assertEqual(result['eligible_n'],0);self.assertEqual(result['excluded_n'],2)
    def test_pil08_unverified_delivery_or_leaked_feature_excluded(self):
        row=dict(state='CLOSED',source_kind='ACTUAL_DEMO',features=[0]*4,origin_ms=100,
            feature_end_ms=99,label_end_ms=200,label_available_at_ms=201,private_delivery_matched=True)
        rows=[dict(row,private_delivery_matched=False),dict(row,feature_end_ms=101),row]
        result=self.api.learning_handoff(rows,fit_cutoff_ms=1000)
        self.assertEqual(result['eligible_n'],1);self.assertEqual(result['excluded_n'],2)
    def test_usd06_feature_received_after_origin_cannot_enter_learning(self):
        row=dict(state='CLOSED',source_kind='ACTUAL_DEMO',features=[0]*4,origin_ms=100,
            feature_end_ms=99,feature_available_at_ms=101,label_end_ms=200,label_available_at_ms=201,private_delivery_matched=True)
        self.assertEqual(self.api.learning_handoff([row],fit_cutoff_ms=1000)['eligible_n'],0)
    def test_pil08_classifier_has_named_constant_baseline(self):
        result=self.api.fit_diagnostic(bars(500))
        self.assertIn('constant_baseline_brier',result)
        self.assertEqual(result['test_positive_n'],0)


if __name__=='__main__':unittest.main()
