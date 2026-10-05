"""Report denominators stay faithful to saved frozen results."""
import unittest
import tempfile,json
from pathlib import Path
import numpy as np
from minute_direction_models import metrics
from render_minute_direction import tables,daily_coverage,render,METHODS
from minute_direction_data import sha
from verify_minute_direction import verify

class ReportTests(unittest.TestCase):
    def test_cash_is_not_converted_to_directional_profit_and_asset_counts_stay_separate(self):
        y=np.array([0,1,2,0]);p=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.1,.1,.8]])
        s=metrics(y,p,np.array([-.01,0,.01,-.01]))
        account=dict(return_pct=0,double_cost_return_pct=0,trades=0,mean_exposure_pct=0,max_drawdown_pct=0,
            drawdown_fully_observed=True,missing_holding_marks=0,funding_paid_initial_capital=0)
        result=dict(metrics={'1':{'CatBoost':dict(pooled=s,assets={'BTCUSDT':s})}},
            accounts={'1':{'CatBoost':account}},paired_losses={'1':{'CatBoost vs Logistic':dict(n_days=7,
            mean_loss_gain=.001,familywise95=[-.01,.01],robust_claim_allowed=False)}})
        pooled,assets,paired=tables(result)
        self.assertEqual(pooled.iloc[0].raw_n,3);self.assertEqual(pooled.iloc[0].raw_correct,2)
        self.assertAlmostEqual(pooled.iloc[0].raw_direction_pct,200/3)
        self.assertEqual(pooled.iloc[0].trades,0);self.assertEqual(pooled.iloc[0].net_return_pct,0)
        self.assertEqual(assets.iloc[0].asset,'BTCUSDT');self.assertNotIn('net_return_pct',assets.columns)
        self.assertFalse(paired.iloc[0].robust_claim_allowed)
    def test_verifier_rejects_changed_frozen_source_before_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'source_snapshot').mkdir()
            (folder/'source_snapshot'/'minute_direction_models.py').write_text('changed')
            result={'metadata':{'registration':{'source_hashes':{'minute_direction_models.py':'bad-hash'}}}}
            (folder/'result.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError):verify(folder,folder/'absent_books')
    def test_missing_states_are_coverage_not_forecast_misses(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);path=folder/'BTCUSDT.npz'
            np.savez_compressed(path,time=np.array([10000,20000,30000]),segment=np.array([1,-1,1]))
            expected={'assets':{'BTCUSDT':{'sha256':sha(path)}}}
            row=daily_coverage(folder,expected).iloc[0]
            self.assertEqual(row.clock_samples,3);self.assertEqual(row.missing_states,1)
            self.assertEqual(row.valid_states,2);self.assertAlmostEqual(row.valid_pct,200/3)
    def test_complete_report_keeps_coverage_table_after_daily_direction_plot(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);book=folder/'BTCUSDT.npz'
            np.savez_compressed(book,time=np.array([10000,20000,30000]),segment=np.array([1,-1,1]))
            y=np.array([0,1,2,0]);p=np.tile([.2,.3,.5],(4,1));actual=np.array([-.01,0,.01,-.01])
            score=metrics(y,p,actual)
            account=dict(return_pct=0,double_cost_return_pct=0,trades=0,mean_exposure_pct=0,max_drawdown_pct=0,
                drawdown_fully_observed=True,missing_holding_marks=0,funding_paid_initial_capital=0)
            meta=dict(start=10000,end=30000,samples=3,valid=2,invalid=1,trade_gap_bins=0,
                state_audit=dict(clock_gaps=1,backward_rows=0),sha256=sha(book))
            result=dict(status='COMPLETED_RETROSPECTIVE_L2_DIAGNOSTIC',coverage={'assets':{'BTCUSDT':meta}},
                cuts={'test':10000,'end':40000},cohort_counts={k:4 for k in ['train','validation','calibration','test','inference']},
                metrics={str(h):{name:dict(pooled=score,assets={'BTCUSDT':score}) for name in METHODS} for h in [1,3,5]},
                accounts={str(h):{name:account for name in METHODS} for h in [1,3,5]},paired_losses={str(h):{} for h in [1,3,5]})
            (folder/'result.json').write_text(json.dumps(result),encoding='utf-8')
            curve=np.array([[10000,1],[20000,1],[30000,1]])
            for h in [1,3,5]:
                for name in METHODS:np.savez_compressed(folder/f'curve_{name}_{h}.npz',curve=curve)
            np.savez_compressed(folder/'test_predictions.npz',time=np.array([10000,20000,30000,40000]),
                labels=np.tile(y[:,None],(1,3)),returns=np.tile(actual[:,None],(1,3)),scored=np.ones(4,dtype=bool),
                **{name:np.tile(p[:,None,:],(1,3,1)) for name in METHODS})
            (folder/'price_forecasts.html').write_text('existing frozen chart view',encoding='utf-8')
            (folder/'price_paths.html').write_text('existing regression chart view',encoding='utf-8')
            render(folder,folder,folder/'price_paths.html')
            document=(folder/'comparison.html').read_text(encoding='utf-8')
            self.assertIn('Daily source coverage',document)
            self.assertIn('href="price_forecasts.html"',document)
            self.assertIn('href="price_paths.html"',document)
            self.assertTrue((folder/'comparison.png').exists());self.assertTrue((folder/'coverage_daily.csv').exists())

if __name__=='__main__':unittest.main()
