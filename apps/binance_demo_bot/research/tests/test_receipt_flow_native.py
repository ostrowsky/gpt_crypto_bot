"""RFD04/06/07: real isolated NumPy/CatBoost research contracts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from binance_demo_bot import receipt_flow_study as study


class NativeFlowContracts(unittest.TestCase):
    def test_RFD07_existing_output_is_never_mutated_by_main_error(self):
        root=Path(__file__).resolve().parents[2]/'.runtime'
        with tempfile.TemporaryDirectory(dir=root) as folder:
            base=Path(folder);capture=base/'capture';capture.mkdir()
            output=base/'existing';output.mkdir()
            with patch('sys.argv',['study','--capture',str(capture),'--output',str(output)]):
                with self.assertRaisesRegex(ValueError,'OUTPUT_EXISTS'):study.main()
            self.assertEqual(list(output.iterdir()),[])

    def test_RFD04_intermediate_gap_excludes_label_keeps_issuance(self):
        grid=np.ones((100,6),dtype='float64');grid[:,0]=100.;grid[:,1]=1.
        grid[45,2]=0.
        issued,scored=study.masks(grid,0,100)
        self.assertIn(30,issued)
        self.assertNotIn(30,scored)
        self.assertIn(50,scored)
        self.assertNotIn(70,scored)

    def test_RFD04_epoch_boundary_and_future_feature_flag_not_label_input(self):
        grid=np.ones((100,6),dtype='float64');grid[:,0]=100.;grid[:,1]=1.
        grid[40:,1]=2.;grid[40:,3]=0.
        issued,scored=study.masks(grid,0,100)
        self.assertIn(30,issued);self.assertNotIn(30,scored)
        self.assertIn(5,scored)

    def test_RFD06_real_fit_native_reload_and_fixed_params(self):
        root=Path(__file__).resolve().parents[2]/'.runtime'
        with tempfile.TemporaryDirectory(dir=root) as folder:
            x=np.arange(240,dtype='float64').reshape(120,2)/100
            y=np.sin(x[:,0])
            model,proof=study.fit_model(x,y,['x','z'],Path(folder)/'native.cbm')
            self.assertEqual(proof['reload_rows_n'],32)
            self.assertLessEqual(proof['max_reload_delta'],1e-12)
            self.assertEqual(model.get_params()['iterations'],300)
            self.assertEqual(model.get_params()['thread_count'],2)

    def test_RFD07_registration_contains_all_fixed_timing_and_no_promotion(self):
        from datetime import datetime
        ms=lambda s:int(datetime.fromisoformat(s).timestamp()*1000)
        capture={'full_days_start_ms':ms('2026-11-06T00:00:00+01:00'),
                 'planned_end_ms':ms('2026-11-13T00:00:00+01:00'),
                 'timezone':'Europe/Budapest','symbols':['BTCUSDT','ETHUSDT','SOLUSDT']}
        plan=study.make_plan(capture,{'manifest':'abc'})
        self.assertEqual(plan['horizons_seconds'],[5,10,30])
        self.assertEqual(plan['arms'],['P','B','F'])
        self.assertEqual(plan['inputs'],{'manifest':'abc'})
        self.assertFalse(plan['promotion_allowed'])
        self.assertIsNone(plan['net_daily_pnl'])
        self.assertEqual(plan['tune_use'],'DIAGNOSTIC_ONLY_NO_SELECTION')


if __name__=='__main__':unittest.main()
