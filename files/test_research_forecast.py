"""Focused temporal integrity and service-contract tests, with no Binance access."""
import json
import tempfile
import unittest
import io
import zipfile
import hashlib
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import research_forecast as rf


def raw_rows(cfg):
    end = rf.utc(cfg.end_utc)
    start = end-pd.Timedelta(days=cfg.history_days)
    rows = []
    for i in range(cfg.history_days*1440):
        t = (start+pd.Timedelta(minutes=i)).value//1_000_000
        close = 100*np.exp(0.001*np.sin(i/20)+0.00001*i)
        rows.append([int(t), str(close), str(close+1), str(close-1), str(close),
                     "10", int(t+59999), "1000", 10, "5", "500", "0"])
    return rows


class TemporalIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = replace(rf.ForecastConfig(), symbols=("BTCUSDT",), history_days=5,
                          models=("Persistence",), bootstrap_draws=100,min_test_days=10)
        cls.rows = raw_rows(cls.cfg)
        cls.market = rf.clean_klines(cls.rows, cls.cfg)
        cls.full = rf.prepare(cls.market, cls.cfg)

    def test_reject_gaps_duplicates_schema_and_unclosed(self):
        cases = [self.rows[:100]+self.rows[101:], self.rows+[self.rows[-1][:-1]]]
        conflict = [r[:] for r in self.rows]
        changed = conflict[-1][:]
        changed[4] = "10000"
        cases.append(conflict+[changed])
        unclosed = [r[:] for r in self.rows]
        unclosed[-1][6] += 1
        cases.append(unclosed)
        nonfinite = [r[:] for r in self.rows]
        nonfinite[100][4] = "inf"
        cases.append(nonfinite)
        for rows in cases:
            with self.assertRaises(ValueError):
                rf.clean_klines(rows, self.cfg)
        self.assertEqual(len(rf.clean_klines(self.rows+[self.rows[-1]], self.cfg)), len(self.rows))

    def test_partial_download_is_not_cached_as_success(self):
        calls=[]
        def request(url):
            calls.append(url)
            if len(calls)==1:return self.rows[:1000]
            raise OSError('network unavailable')
        with tempfile.TemporaryDirectory() as root, patch.object(rf.time,'sleep'):
            with self.assertRaises(OSError):rf.fetch_history('BTCUSDT',self.cfg,root,request=request)
            self.assertEqual(list(Path(root).iterdir()),[])

    def test_features_and_sequences_do_not_change_when_future_changes(self):
        changed = self.market.copy()
        changed.loc[1001:, rf.OHLCV] *= 9
        future = rf.prepare(changed, self.cfg)
        pd.testing.assert_frame_equal(self.full.loc[:1000,rf.FEATURES], future.loc[:1000,rf.FEATURES])
        origins = self.full.iloc[[500,1000]]
        np.testing.assert_array_equal(rf.sequence_inputs(self.full, origins, self.cfg),
                                      rf.sequence_inputs(future, origins, self.cfg))
        prefix = rf.prepare(self.market.iloc[:1001], self.cfg)
        pd.testing.assert_frame_equal(prefix[rf.FEATURES], self.full.loc[:1000,rf.FEATURES])

    def test_purge_uses_label_availability_and_utc_cutoffs(self):
        splits, bounds = rf.split_frames(self.full, self.cfg)
        for i, name in enumerate(("train","tune","calibration","test")):
            self.assertTrue((splits[name].available_at >= bounds[i]).all())
            self.assertTrue((splits[name].label_available_at < bounds[i+1]).all())
        for left,right in zip(("train","tune","calibration"),("tune","calibration","test")):
            self.assertLess(splits[left].label_available_at.max(), splits[right].available_at.min())

    def test_training_transform_ignores_later_labels(self):
        splits,_ = rf.split_frames(self.full,self.cfg)
        model = rf.FixedRegressor("Ridge",splits["train"],splits["tune"],self.full,self.cfg)
        scaler = model.model.steps[0][1]
        np.testing.assert_allclose(scaler.mean_,rf.evaluation_grid(splits["train"],self.cfg.train_stride)[rf.FEATURES].mean().to_numpy())
        changed = self.full.copy()
        changed.loc[splits["test"].index,rf.target_columns(self.cfg)] = 100
        model2 = rf.FixedRegressor("Ridge",splits["train"],splits["tune"],changed,self.cfg)
        np.testing.assert_allclose(model.predict(self.full,splits["test"]),
                                   model2.predict(changed,splits["test"]))

    def test_classical_fit_is_conditioned_and_causal(self):
        raw=self.market.copy()
        rng=np.random.default_rng(123)
        close=100*np.exp(np.cumsum(rng.normal(0,0.0005,len(raw))))
        raw['close'],raw['open'],raw['high'],raw['low']=close,close,close+1,close-1
        full=rf.prepare(raw,self.cfg)
        rows=full.iloc[[1000]]
        policy=rf.ClassicalPolicy('ARIMA',None,None,full,self.cfg)
        first=policy.predict(full,rows)
        changed=full.copy()
        changed.loc[1001:,'log_close']+=10
        second=policy.predict(changed,rows)
        np.testing.assert_allclose(first,second)
        self.assertTrue(np.isfinite(first).all())

    def test_seasonal_parameters_train_only_and_forecast_prefix_invariant(self):
        splits,_=rf.split_frames(self.full,self.cfg)
        rows=splits['test'].iloc[[10]]
        idx=int(rows.time_idx.iloc[0])
        changed=self.full.copy()
        changed.loc[idx+1:,'log_close']+=100
        changed.loc[idx+1:,rf.STATE_EXOG]=999999
        for name in ('SARIMA','SARIMAX'):
            policy=rf.build_policy(name,splits['train'],splits['tune'],self.full,self.cfg)
            before=policy.predict(self.full,rows)
            after=policy.predict(changed,rows)
            np.testing.assert_allclose(before,after)
            altered=self.full.copy()
            altered.loc[splits['tune'].index,rf.CALENDAR]=999
            second=rf.build_policy(name,splits['train'],altered.loc[splits['tune'].index],altered,self.cfg)
            np.testing.assert_allclose(policy.beta,second.beta)
            self.assertAlmostEqual(policy.phi,second.phi)
            self.assertAlmostEqual(policy.seasonal,second.seasonal)
            self.assertTrue(policy.fit_diagnostics['success'])

    def test_archive_microseconds_checksum_and_complete_grid(self):
        cfg=replace(self.cfg,end_utc='2026-09-01T00:00:00Z',history_days=1)
        rows=raw_rows(cfg)
        micro=[r[:] for r in rows]
        for row in micro:
            row[0]*=1000
            row[6]=row[6]*1000+999
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w') as archive:
            archive.writestr('BTCUSDT-1m-2026-08.csv','\n'.join(','.join(map(str,r)) for r in micro))
        data=buffer.getvalue()
        checksum=hashlib.sha256(data).hexdigest()+'  BTCUSDT-1m-2026-08.zip'
        def opener(url,timeout):return io.BytesIO(checksum.encode() if url.endswith('CHECKSUM') else data)
        with tempfile.TemporaryDirectory() as root:
            actual,manifest=rf.fetch_archive_history('BTCUSDT',cfg,root,opener=opener)
            self.assertEqual(len(actual),1440)
            self.assertEqual(manifest['archives'][0]['sha256'],hashlib.sha256(data).hexdigest())
            path=Path(root)/'BTCUSDT-1m-2026-08.zip'
            path.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):rf.fetch_archive_history('BTCUSDT',cfg,root,opener=opener)

    def test_direction_baseline_is_train_only_and_abstentions_are_counted(self):
        train=self.full.iloc[100:200].copy()
        train['target_h15']=-1
        rows=self.full.iloc[300:320].copy()
        rows['target_h15']=1
        report=rf.direction_report(rows,np.ones((20,15)),train,self.cfg)
        self.assertEqual(report['direction_baseline_correct'],0)
        self.assertEqual(report['direction_train_majority_sign'],-1)
        self.assertEqual(report['direction_population_correct'],20)
        report=rf.direction_report(rows,np.zeros((20,15)),train,self.cfg)
        self.assertEqual(report['direction_abstentions'],20)
        self.assertEqual(report['direction_hit_rate'],0)
        self.assertIsNone(report['direction_balanced_accuracy'])

    def test_selection_includes_persistence_and_sees_only_tuning(self):
        actual = np.ones((10,self.cfg.horizon))*0.001
        pred = {"Persistence":np.zeros_like(actual),"XGBoost":np.ones_like(actual)*100}
        name,losses = rf.choose_model(pred,actual)
        self.assertEqual(name,"Persistence")
        self.assertEqual(rf.choose_model({"XGBoost":np.zeros_like(actual),"Persistence":np.zeros_like(actual)},actual)[0],"Persistence")

    def test_finite_sample_quantiles_separate_every_horizon(self):
        errors = np.arange(1,20)[:,None]*np.arange(1,16)[None,:]
        q = rf.finite_sample_widths(errors,np.zeros_like(errors))
        np.testing.assert_equal(q,18*np.arange(1,16))
        with self.assertRaises(ValueError):
            rf.finite_sample_widths(np.ones((2,15)),np.zeros((2,15)))

    def test_zero_denominator_and_partial_predictions_are_unknown(self):
        rows = self.full.iloc[100:110].copy()
        rows[rf.target_columns(self.cfg)] = 0.0
        row = rf.score("BTCUSDT","Persistence",rows,np.zeros((10,15)),np.ones(15),self.cfg)
        self.assertIsNone(row["improvement_pct"])
        self.assertIsNone(row["direction_accuracy"])
        self.assertEqual(row["verdict"],"UNKNOWN")
        with self.assertRaises(ValueError):
            rf.score("BTCUSDT","broken",rows,np.zeros((9,15)),None,self.cfg)

    def test_bootstrap_requires_days_and_counts_coverage(self):
        splits,_=rf.split_frames(self.full,self.cfg)
        rows=rf.evaluation_grid(splits["test"],60)
        pred=np.zeros((len(rows),15))
        result=rf.score("BTCUSDT","Persistence",rows,pred,np.ones(15),self.cfg)
        self.assertEqual(result["PI90_covered_by_horizon"],[len(rows)]*15)
        self.assertEqual(result["verdict"],"UNKNOWN")
        self.assertEqual(result["PI90_denominator"],len(rows))

    def test_paired_bootstrap_positive_and_negative_evidence(self):
        rows=self.full.iloc[100:110].copy()
        rows['open_time']=pd.date_range('2026-09-01',periods=10,freq='D',tz='UTC')
        actual=np.ones((10,15))*0.1
        supported=rf.paired_day_bootstrap(rows,actual,actual/2,self.cfg)
        rejected=rf.paired_day_bootstrap(rows,actual,-actual,self.cfg)
        self.assertEqual(supported['verdict'],'SUPPORTED_DIAGNOSTIC')
        np.testing.assert_allclose(supported['ci95_improvement_pct'],[50,50])
        self.assertEqual(rejected['verdict'],'NO_IMPROVEMENT')
        np.testing.assert_allclose(rejected['ci95_improvement_pct'],[-100,-100])

    def test_experiment_grid_and_calibration_cannot_change_selection(self):
        with patch.object(rf,"build_policy",wraps=rf.build_policy) as build:
            cfg=replace(self.cfg,models=("Persistence","Ridge"),complete_test_days_only=False)
            first=rf.run_experiment({"BTCUSDT":self.market},cfg)
            changed=self.market.copy()
            # Shift calibration and test prices without changing pre-calibration history.
            start=int(7200*0.75)
            changed.loc[start:, ["open","close","high","low"]]*=2
            second=rf.run_experiment({"BTCUSDT":changed},cfg)
            self.assertEqual(first["choices"],second["choices"])
            self.assertEqual(build.call_count,2)
            self.assertEqual(len(first["grids"]["BTCUSDT"]["test"]),first["results"][0]["n_origins"])


class ServingContract(unittest.TestCase):
    def setUp(self):
        self.cfg=replace(rf.ForecastConfig(),symbols=("BTCUSDT",))
        self.now=rf.utc(self.cfg.end_utc)+pd.Timedelta(seconds=10)
        self.experiment=dict(choices={"BTCUSDT":"XGBoost"},results=[dict(symbol="BTCUSDT",model="XGBoost",verdict="UNKNOWN")],
                            policies={"BTCUSDT":{"Persistence":None}},widths={"BTCUSDT":{"Persistence":np.ones(15)*0.01}})
        self.token="forecast-test-key-long-enough"
        self.fail=False
        def fetcher(symbol,cfg,cache):
            if self.fail: raise ValueError("network_failure")
            return rf.clean_klines(raw_rows(cfg),cfg),dict(sha256="test-input-digest")
        self.fetcher=fetcher
        self.service=rf.ForecastService(self.experiment,self.cfg,self.token,fetcher,lambda:self.now)

    def test_unproven_model_falls_back_and_stale_data_is_rejected(self):
        self.service.refresh("unused")
        row=self.service.forecast("BTCUSDT")
        self.assertTrue(row["fallback"])
        self.assertEqual(row["model"],"Persistence")
        self.assertGreater(rf.utc(row["target_close_at"][0]),rf.utc(row["issued_at"]))
        self.now+=pd.Timedelta(minutes=2)
        with self.assertRaises(ValueError):self.service.forecast("BTCUSDT")

    def test_live_history_matches_declared_classical_window_and_context(self):
        seen=[]
        original=self.service.fetcher
        def capture(symbol,cfg,cache):
            seen.append(cfg.history_days)
            return original(symbol,cfg,cache)
        self.service.fetcher=capture
        self.service.refresh('unused')
        self.assertEqual(seen,[3])
        self.assertGreaterEqual(seen[0]*1440,self.cfg.classical_window+self.cfg.context)

    def test_failed_refresh_does_not_publish_partial_values(self):
        self.service.refresh("unused")
        before=self.service.snapshots.copy()
        self.fail=True
        with self.assertRaises(ValueError):self.service.refresh("unused")
        self.assertEqual(before,self.service.snapshots)

    def test_expired_release_and_short_api_token_are_rejected(self):
        with self.assertRaises(ValueError):rf.ForecastService(self.experiment,self.cfg,"short")
        self.now+=pd.Timedelta(days=8)
        with self.assertRaises(ValueError):self.service.refresh("unused")

    def test_quality_is_scored_only_after_future_label_matures(self):
        self.service.refresh("unused")
        self.assertEqual(self.service.quality,[])
        self.now+=pd.Timedelta(minutes=10)
        self.service.refresh("unused")
        self.assertEqual(self.service.quality,[])
        self.now+=pd.Timedelta(minutes=6)
        self.service.refresh("unused")
        self.assertEqual(len(self.service.quality),1)
        self.assertEqual(self.service.quality[0]['status'],'MATURED')

    def test_inference_never_publishes_already_closed_future_target(self):
        original=self.fetcher
        def slow(symbol,cfg,cache):
            response=original(symbol,cfg,cache)
            self.now+=pd.Timedelta(minutes=2)
            return response
        self.service.fetcher=slow
        with self.assertRaises(ValueError):self.service.refresh('unused')
        self.assertFalse(self.service.snapshots)

    def test_closed_first_horizon_is_not_served_during_api_outage(self):
        self.service.refresh('unused')
        self.now+=pd.Timedelta(seconds=55)
        with self.assertRaises(ValueError):self.service.forecast('BTCUSDT')

    def test_http_auth_health_and_error_contract(self):
        from fastapi.testclient import TestClient
        app=rf.create_app(self.experiment,self.cfg,token=self.token,fetcher=self.fetcher,clock=lambda:self.now)
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code,200)
            self.assertEqual(client.get("/forecast/BTCUSDT").status_code,401)
            headers={"x-api-key":self.token}
            self.assertEqual(client.get("/forecast/NOPE",headers=headers).status_code,404)
            self.assertEqual(client.get("/forecast/BTCUSDT",headers=headers).status_code,200)
            self.now+=pd.Timedelta(minutes=2)
            self.assertEqual(client.get("/forecast/BTCUSDT",headers=headers).status_code,503)
            self.assertEqual(client.get("/health").status_code,503)


class NotebookContract(unittest.TestCase):
    def test_standalone_notebook_cells_compile_and_outputs_are_empty(self):
        from build_research_forecast_notebook import build,ROOT
        notebook=build()
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),f'notebook-cell-{i}','exec')
                self.assertEqual(cell['outputs'],[])
                self.assertIsNone(cell['execution_count'])
        checked_in=ROOT/'research'/'Binance_BTC_ETH_SOL_ML_Researcher_Demo_v3.ipynb'
        if checked_in.exists():
            self.assertEqual(json.loads(checked_in.read_text(encoding='utf-8')),notebook)

    def test_notebook_export_creates_a_runnable_standalone_module(self):
        import os
        from build_research_forecast_notebook import build
        notebook=build()
        export=''.join(notebook['cells'][-2]['source']).replace('EXPORT_SERVICE = False','EXPORT_SERVICE = True')
        cwd=Path.cwd()
        with tempfile.TemporaryDirectory() as root:
            try:
                os.chdir(root)
                Path('Binance_BTC_ETH_SOL_ML_Researcher_Demo_v3.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
                exec(export,{'Path':Path,'json':json})
                source=Path('research_forecast.py').read_text(encoding='utf-8')
                compile(source,'exported-research-forecast','exec')
                self.assertTrue(source.endswith('    main()\n'))
            finally:
                os.chdir(cwd)

    def test_notebook_has_every_method_graph_and_interactive_controls(self):
        from build_research_forecast_notebook import build
        source='\n'.join(''.join(c['source']) for c in build()['cells'])
        self.assertIn('plot_method_evidence(experiment,cv_traces,CFG)',source)
        self.assertIn('plot_comparison(experiment,CFG)',source)
        self.assertIn('interactive_comparison(experiment,CFG)',source)
        self.assertIn('direction_ci95_familywise_gain_pp',source)


class ProspectiveContract(unittest.TestCase):
    def setUp(self):
        self.cfg=replace(rf.ForecastConfig(),symbols=('BTCUSDT',),models=('Persistence',))
        self.now=rf.utc('2026-10-04T12:00:10Z')
        self.experiment=dict(predictions={'BTCUSDT':{'Persistence':np.zeros((1,15))}},
                             policies={'BTCUSDT':{'Persistence':None}},
                             widths={'BTCUSDT':{'Persistence':np.ones(15)*.01}})
        def fetcher(symbol,cfg,cache):
            return rf.clean_klines(raw_rows(cfg),cfg),dict(sha256='test-input')
        self.controller=rf.ForecastComparison(self.experiment,self.cfg,fetcher=fetcher,clock=lambda:self.now)

    def test_refresh_only_mature_facts_and_never_rewrites_forecast(self):
        key=self.controller.create('BTCUSDT')
        frozen=json.dumps(self.controller.snapshots[key],sort_keys=True)
        initial=self.controller.metrics(key).iloc[0]
        self.assertEqual(initial.mature_path_points,0)
        self.assertIsNone(initial.MAE_USDT)
        self.now+=pd.Timedelta(minutes=5)
        actual=self.controller.refresh(key).iloc[0]
        self.assertEqual(actual.mature_path_points,5)
        self.assertEqual(actual.state,'PARTIAL')
        self.now+=pd.Timedelta(minutes=10)
        actual=self.controller.refresh(key).iloc[0]
        self.assertEqual(actual.mature_path_points,15)
        self.assertEqual(actual.state,'COMPLETE')
        self.assertEqual(frozen,json.dumps(self.controller.snapshots[key],sort_keys=True))

    def test_deadline_and_missing_mature_data_do_not_turn_into_success(self):
        fetcher=self.controller.fetcher
        def delayed(symbol,cfg,cache):
            data=fetcher(symbol,cfg,cache)
            self.now+=pd.Timedelta(minutes=2)
            return data
        self.controller.fetcher=delayed
        with self.assertRaises(ValueError):self.controller.create('BTCUSDT')
        self.assertFalse(self.controller.snapshots)
        self.controller.fetcher=fetcher
        key=self.controller.create('BTCUSDT')
        self.now+=pd.Timedelta(days=4)
        result=self.controller.refresh(key).iloc[0]
        self.assertEqual(result.state,'MISSING_OBSERVATIONS')
        self.assertEqual(result.missing_mature_points,15)
        self.assertIsNone(result.MAE_USDT)

    def test_snapshot_json_preserves_original_paths_and_timestamp(self):
        with tempfile.TemporaryDirectory() as root:
            self.controller.output_dir=Path(root)
            key=self.controller.create('BTCUSDT')
            payload=json.loads((Path(root)/(key+'.json')).read_text(encoding='utf-8'))
            self.assertEqual(payload,self.controller.snapshots[key])
            self.assertGreater(rf.utc(payload['target_close_at'][0]),rf.utc(payload['issued_at']))


if __name__ == "__main__":
    unittest.main()
