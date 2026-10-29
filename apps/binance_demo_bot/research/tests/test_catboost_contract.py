"""Real pinned-library integration; run with own research Python, not trading venv."""
import math
from pathlib import Path
import tempfile
import unittest
from catboost import CatBoostRegressor
from binance_demo_bot.feature_ablation_runner import fit_model

class CatBoostContract(unittest.TestCase):
    def test_fit_with_names_then_save_reload(self):
        rows=[{'a':i/40,'b':(i%3)/3} for i in range(40)]
        labels=[(i%7-3)/100 for i in range(40)]
        base=Path(__file__).resolve().parents[2]/'.runtime'/'unit-fixtures';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            path=Path(temporary)/'model.cbm'
            model=fit_model(rows,labels,['a','b'],path)
            self.assertEqual(model.feature_names_,['a','b'])
            self.assertEqual(model.tree_count_,300)
            result=model.predict([[.1,.2]])
            self.assertTrue(math.isfinite(float(result[0])))
            restored=CatBoostRegressor();restored.load_model(str(path))
            self.assertEqual(float(result[0]),float(restored.predict([[.1,.2]])[0]))

if __name__=='__main__':unittest.main()
