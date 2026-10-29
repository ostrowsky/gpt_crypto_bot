"""Explicit real ML integration: intentionally outside minimal app discovery."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


class RealCPUModelIntegration(unittest.TestCase):
    def test_RDE08_real_cat_gru_ppo_train_save_reload(self):
        path = Path(__file__).resolve().parents[1]/'test_rl_direction_models.py'
        spec = importlib.util.spec_from_file_location('rde_model_contracts', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix='rde-real-cpu-') as folder:
            evidence = module.run_real_integration(folder)
        self.assertEqual(evidence['state'],'REAL_CPU_TRAIN_SAVE_LOAD_PASS')
        self.assertEqual(evidence['ppo_steps'],512)


if __name__ == '__main__':
    unittest.main()
