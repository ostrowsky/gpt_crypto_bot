import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import config
import certified_rule_score_policy as policy
import ml_candidate_ranker as ranker
import paired_full_policy_replay as paired


class CertifiedRulePolicyTests(unittest.TestCase):
    def test_default_disabled_does_not_load_model_or_enable_legacy(self):
        with patch.object(config,'CERTIFIED_RULE_SCORE_POLICY_ENABLED',False), \
             patch.object(policy.release,'select') as select:
            self.assertEqual(policy.bonus(),0)
        select.assert_not_called()

    def test_probability_formula_and_bad_predictions(self):
        self.assertEqual(policy.probability_bonus(0),-1)
        self.assertEqual(policy.probability_bonus(0.5),0)
        self.assertEqual(policy.probability_bonus(1),1)
        for value in (float('nan'),float('inf'),-0.01,1.01):
            with self.assertRaises(ValueError): policy.probability_bonus(value)

    def test_signed_selection_path_works_with_legacy_ranker_off(self):
        captured = {}
        def build(**kw): captured.update(kw); return {'f':{}}
        with patch.object(config,'CERTIFIED_RULE_SCORE_POLICY_ENABLED',True), \
             patch.object(config,'ML_CANDIDATE_RANKER_RUNTIME_ENABLED',False,create=True), \
             patch.object(policy,'champion_bytes',return_value=b'bound-rule'), \
             patch.object(policy.release,'select',return_value=({'weights':[]},{})) as select, \
             patch.object(ranker,'build_runtime_candidate_record',side_effect=build), \
             patch.object(ranker,'predict_components_from_candidate_payload',return_value={'quality_proba':0.9}):
            value = policy.bonus(sym='BTCUSDT',tf='15m',data={'t':[900000]},i=0)
        self.assertAlmostEqual(value,0.8)
        self.assertEqual(captured['bar_ts'],900000)
        self.assertFalse(captured['near_miss'])
        self.assertEqual(select.call_args.args[1],b'bound-rule')

    def test_missing_ticket_or_model_failure_preserves_champion_score(self):
        with patch.object(config,'CERTIFIED_RULE_SCORE_POLICY_ENABLED',True), \
             patch.object(policy,'champion_bytes',return_value=b'bound-rule'), \
             patch.object(policy.release,'select',return_value=None):
            self.assertEqual(policy.bonus(sym='BTCUSDT'),0)
        with patch.object(config,'CERTIFIED_RULE_SCORE_POLICY_ENABLED',True), \
             patch.object(policy,'champion_bytes',side_effect=RuntimeError('drift')):
            self.assertEqual(policy.bonus(sym='BTCUSDT'),0)

    def test_champion_binds_sources_models_and_runtime_config_but_not_kill_switch(self):
        original = policy.champion_bytes()
        with patch.object(config,'CERTIFIED_RULE_SCORE_POLICY_ENABLED',True):
            self.assertEqual(policy.champion_bytes(),original)
        with patch.object(config,'TOP_GAINER_SCORE_GATE_MIN_SCORE',999):
            self.assertNotEqual(policy.champion_bytes(),original)
        body = json.loads(original)
        self.assertEqual(body['contract'],policy.CONTRACT)
        self.assertIn('monitor.py',body['sources'])
        self.assertIn('ml_candidate_ranker.json',body['models'])

    def test_monitor_and_replay_use_same_kwargs_without_fake_ranker_components(self):
        for name in ('monitor.py','replay_backtest.py'):
            source=Path(policy.__file__).with_name(name).read_text(encoding='utf-8')
            ast.parse(source)
            self.assertIn('candidate_score += certified_score_bonus(**ranker_kwargs)',source)
            self.assertIn('ranker_info = _ml_candidate_ranker_components(**ranker_kwargs)',source)

    def test_rule_preflight_does_not_need_invalid_legacy_champion(self):
        body=json.loads(policy.champion_bytes())
        self.assertEqual(paired.preflight(body),[])
        body['config']['TOP_GAINER_SCORE_GATE_MIN_SCORE']=999
        self.assertTrue(paired.preflight(body))


if __name__ == '__main__': unittest.main()
