import unittest
from unittest.mock import Mock
import replay_backtest as rb
from mission_rule_policy import rules_only

class PolicyTests(unittest.TestCase):
    def test_baseline_values_and_exit_are_unchanged_without_call_recording(self):
        original=rb.check_exit_conditions;score=rb._ml_general_score_replay
        with rules_only():
            self.assertIs(rb.check_exit_conditions,original)
            self.assertNotIsInstance(rb._ml_general_score_replay,Mock)
            for _ in range(1000):
                self.assertIsNone(rb._ml_general_score_replay('symbol'))
                self.assertEqual(rb._ml_candidate_ranker_components(),{})
                self.assertEqual(rb._ml_candidate_ranker_runtime_bonus(),0.)
            self.assertEqual(rb._load_temporal_scout_events(),({},{}))
        self.assertIs(rb._ml_general_score_replay,score)

if __name__=='__main__':unittest.main()
