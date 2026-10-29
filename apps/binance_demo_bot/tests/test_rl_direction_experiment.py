"""RDE10..12: immutable research scope and honest evaluation; no ML/network."""
import importlib,json,tempfile,unittest
from pathlib import Path
import _support


class ExperimentContracts(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.rl_direction_experiment')
        root=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures';root.mkdir(exist_ok=True,parents=True)
        self.temp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)

    def registration(self):
        return dict(attempt='RDE-1-TEST',source_sha256={'core':'a'*64},data_sha256='b'*64,
            evidence_kind='SIMULATION',promotion_allowed=False,
            holdout_kind='RETROSPECTIVE_REUSED_HOLDOUT',seeds=[11,23])

    def test_rde12_registration_immutable_and_exact_resume(self):
        value=self.registration();digest=self.api.register_run(self.folder,value)
        self.assertEqual(digest,self.api.register_run(self.folder,dict(value)))
        with self.assertRaisesRegex(ValueError,'REGISTRATION_CHANGED'):
            self.api.register_run(self.folder,dict(value,seeds=[11]))
        self.assertEqual(json.loads((self.folder/'registration.json').read_text()),value)

    def test_rde12_actual_or_promotion_claim_cannot_register(self):
        for changes in (dict(evidence_kind='ACTUAL_DEMO'),dict(promotion_allowed=True),dict(holdout_kind='UNTOUCHED')):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError,'RESEARCH_SCOPE_INVALID'):
                    self.api.register_run(self.folder,dict(self.registration(),**changes))

    def test_rde03_probability_metrics_have_counts_and_base_rate(self):
        result=self.api.direction_metrics([0,1,1,0],[.1,.8,.6,.4],[-.1,.2,.1,-.1],[-.05,.1,.05,-.02])
        self.assertEqual(result['rows_N'],4);self.assertEqual(result['positive_n'],2)
        self.assertEqual(result['correct_n'],4);self.assertAlmostEqual(result['brier'],.0925)
        self.assertEqual(result['train_or_test_prevalence'],.5)
        self.assertEqual(result['balanced_accuracy'],1.)

    def test_rde03_empty_or_one_class_not_fabricated_balance(self):
        result=self.api.direction_metrics([],[],[],[])
        self.assertEqual(result['rows_N'],0);self.assertIsNone(result['brier'])
        result=self.api.direction_metrics([1,1],[.7,.9],[.1,.2],[0.,.1])
        self.assertIsNone(result['balanced_accuracy']);self.assertEqual(result['positive_n'],2)

    def test_rde03_invalid_probabilities_are_not_silently_clipped(self):
        for prob in (float('nan'),-1.,1.01):
            with self.assertRaisesRegex(ValueError,'PREDICTION_INVALID'):
                self.api.direction_metrics([1],[prob],[.1],[.1])

    def test_rde12_action_probability_never_becomes_direction(self):
        raw=dict(cat_proba=[[.2,.3]],cat_return=[[.01,.02]],gru_proba=[[.4,.5]],gru_return=[[.03,.04]],latent=[[0.]*16])
        result=self.api.forecast_records(raw,0,origin_ms=100,feature_end_ms=99,fit_cutoff_ms=50)
        self.assertEqual(result['PPO']['p_up'],[.4,.5])
        self.assertEqual(result['PPO']['forecast_source'],'GRU_FROZEN')
        self.assertEqual(result['CatBoost']['p_up'],[.2,.3])
        self.assertNotIn('action_probabilities',result['PPO'])

    def test_rde12_forecast_rejects_future_features_or_fit(self):
        for feature,fit in ((100,50),(99,101)):
            with self.assertRaisesRegex(ValueError,'FORECAST_TIME_INVALID'):
                self.api.forecast_records({},0,origin_ms=100,feature_end_ms=feature,fit_cutoff_ms=fit)

    def test_rde11_seed_mean_requires_all_registered_seeds_and_days(self):
        rows=[dict(seed=11,daily_pnl={'d1':1.,'d2':-2.}),dict(seed=23,daily_pnl={'d1':3.,'d2':0.})]
        result=self.api.seed_daily_mean(rows,[11,23]);self.assertEqual(result,{'d1':2.,'d2':-1.})
        for changed in (rows[:1],[rows[0],dict(seed=23,daily_pnl={'d1':3.})],rows+[rows[0]]):
            with self.assertRaisesRegex(ValueError,'SEED_OR_DAY_COVERAGE'):
                self.api.seed_daily_mean(changed,[11,23])

    def test_rde11_small_or_failed_bound_is_not_positive_evidence(self):
        self.assertEqual(self.api.comparison_verdict(paired_n=29,delta_lower=1,absolute_lower=1,stress_net=2,max_dd=0,reference_dd=0),'INCONCLUSIVE')
        self.assertEqual(self.api.comparison_verdict(paired_n=100,delta_lower=.02,absolute_lower=-.01,stress_net=2,max_dd=0,reference_dd=0),'NOT_SUPPORTED')
        self.assertEqual(self.api.comparison_verdict(paired_n=100,delta_lower=None,absolute_lower=1,stress_net=2,max_dd=0,reference_dd=0),'INCONCLUSIVE')

    def test_rde11_support_is_exploratory_never_pass_or_promotion(self):
        verdict=self.api.comparison_verdict(paired_n=100,delta_lower=.02,absolute_lower=.01,stress_net=2,max_dd=.1,reference_dd=0)
        self.assertEqual(verdict,'EXPLORATORY_PATTERN_POWER_UNCERTIFIED')

    def test_rde12_threshold_search_is_tune_only_and_tie_prefers_abstention(self):
        rows=[dict(threshold=t,segment='tune',mean_daily_net_pnl_usdt=0.) for t in (0.,.0005,.001,.002)]
        self.assertEqual(self.api.select_threshold(rows),.002)
        with self.assertRaisesRegex(ValueError,'TUNE_ONLY'):
            self.api.select_threshold([dict(rows[0],segment='test')])

    def test_rde12_frozen_predictions_cannot_be_overwritten_with_actual(self):
        record=[dict(origin_ms=100,p_up=.6,source_kind='RETROSPECTIVE_ASOF')]
        path=self.folder/'predictions.json';digest=self.api.freeze_predictions(path,record)
        self.assertEqual(digest,self.api.freeze_predictions(path,record))
        with self.assertRaisesRegex(ValueError,'FROZEN_PREDICTIONS_CHANGED'):
            self.api.freeze_predictions(path,[dict(record[0],p_up=1.)])


if __name__=='__main__':unittest.main()
