import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import local_learning_runtime as local
import forward_evidence_service as service
import learning_loop_health as health


class LocalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with patch.object(local, 'current_sid', return_value='test-sid'):
            self.path = local.initialize(self.root)
        self.deployment = json.loads(self.path.read_bytes())

    def test_initialization_creates_local_keys_but_no_approval(self):
        self.assertFalse(self.deployment['os_access_isolation'])
        self.assertEqual(self.deployment['isolation_mode'], 'logical_same_user')
        self.assertFalse(Path(self.deployment['release_root']).exists())
        self.assertFalse(Path(self.deployment['portfolio_inputs']).exists())
        self.assertFalse((self.root/'.runtime/learning_roles').exists())
        self.assertEqual((self.path.parent/'authority/coverage.key').stat().st_size, 48)

    def test_initialize_is_idempotent_but_refuses_other_sid(self):
        with patch.object(local, 'current_sid', return_value='test-sid'):
            self.assertEqual(local.initialize(self.root), self.path)
        with patch.object(local, 'current_sid', return_value='other'):
            with self.assertRaises(ValueError):
                local.initialize(self.root)

    def test_secret_allowlist_for_each_child(self):
        source = {'PATH': 'p', 'TEMP': 't', 'TELEGRAM_TOKEN': 'secret',
                  'RANKER_EVALUATOR_KEY': 'key', 'COVERAGE_AUTHORITY_KEY': 'secret',
                  'BINANCE_SECRET': 'secret', 'PYTHONPATH': 'injected'}
        for role in local.INTERVALS:
            env = local.role_environment(role, source)
            self.assertEqual(set(env), {'PATH', 'TEMP', 'OPENBLAS_NUM_THREADS',
                                       'OMP_NUM_THREADS', 'MKL_NUM_THREADS'} | (
                {'RANKER_EVALUATOR_KEY'} if role == 'controller' else set()))

    def test_identity_adoption_requires_stop_and_preserves_backup(self):
        base = self.path.parent
        service.atomic(base/'supervisor.json', {'state': 'RUNNING'})
        (base/'stop.request').touch()
        with self.assertRaises(ValueError): local.adopt_current_user(self.root)
        service.atomic(base/'supervisor.json', {'state': 'STOPPED'})
        old = self.path.read_bytes()
        with patch.object(local, 'current_sid', return_value='actual-user'):
            local.adopt_current_user(self.root)
        value = json.loads(self.path.read_bytes())
        self.assertEqual(value['trainer_sid'], 'actual-user')
        self.assertEqual(value['evaluator_sid'], 'actual-user')
        self.assertFalse(value['os_access_isolation'])
        self.assertEqual(next(base.glob('deployment.before_identity_*.json')).read_bytes(), old)

    def test_unaccepted_identity_adoption_is_rejected(self):
        base = self.path.parent
        (base/'stop.request').touch()
        service.atomic(base/'supervisor.json', {'state': 'STOPPED'})
        service.atomic(self.path, dict(self.deployment, logical_isolation_accepted=False))
        with self.assertRaises(ValueError): local.adopt_current_user(self.root)

    def test_role_requires_explicit_acceptance_and_matching_sid(self):
        for key, value in (('logical_isolation_accepted', False), ('trainer_sid', 'other')):
            deployment = dict(self.deployment, **{key: value})
            with patch.object(service, 'current_sid', return_value='test-sid'):
                with self.assertRaises(PermissionError):
                    service.run_tick(deployment, 'evaluator')

    def test_local_trainer_uses_snapshot_not_raw_and_never_os_probe(self):
        import training_snapshot
        import ml_candidate_ranker as ranker
        with patch.object(service, 'current_sid', return_value='test-sid'), \
             patch.object(service, 'verify_trainer_isolation') as probe, \
             patch.object(training_snapshot, 'resolve', return_value=(self.root/'eligible', {'sha256':'hash'})), \
             patch.object(training_snapshot, 'digest', return_value='hash'), \
             patch.object(ranker, 'train_and_evaluate', return_value={}) as train, \
             patch.object(ranker, 'build_live_model_payload', return_value={'candidate':True}):
            result = service.run_tick(self.deployment, 'trainer')
        probe.assert_not_called()
        train.assert_called_once_with(self.root/'eligible', optimize_prediction_error=True)
        self.assertEqual(result['state'], 'CANDIDATE_ONLY')
        self.assertFalse(result['runtime_eligible'])
        self.assertFalse(result['os_access_isolation'])

    def test_collector_does_not_run_controller_or_export(self):
        with patch.object(service, 'current_sid', return_value='test-sid'), \
             patch.object(service.evaluator, 'register', side_effect=ValueError('missing candidate')), \
             patch.object(service, 'controller_tick') as controller, \
             patch.object(service, 'export_training') as export:
            result = service.run_tick(self.deployment, 'evaluator')
        controller.assert_not_called()
        export.assert_not_called()
        self.assertFalse(result['closed_loop'])
        self.assertEqual(result['controller_schedule'], 'separate_process')

    def test_missing_portfolios_block_separate_controller(self):
        with patch.object(service, 'current_sid', return_value='test-sid'):
            result = service.run_tick(self.deployment, 'controller')
        self.assertEqual(result['controller']['state'], 'BLOCKED')
        self.assertFalse(result['controller']['runtime_eligible'])
        self.assertFalse(result['closed_loop'])

    def test_reporting_selects_explicit_local_runtime_without_success_inference(self):
        self.assertEqual(health.runtime_root(self.root/'.runtime'), Path(self.deployment['registry']))
        value = health.summarize(Path(self.deployment['registry']))
        self.assertEqual(value['state'], 'NOT_CLOSED')
        self.assertEqual(value['improvement_verdict'], 'UNKNOWN')

    def test_supervisor_stop_touches_only_owned_children(self):
        # Stop immediately after first status publication; no long sleeps.
        from unittest.mock import MagicMock
        children = []
        def spawn(*args, **kwargs):
            child = MagicMock()
            child.pid = 123 + len(children)
            child.poll.return_value = None
            children.append(child)
            return child
        original_atomic = local.atomic
        def publish(path, value):
            original_atomic(path, value)
            if value.get('state') == 'RUNNING':
                (self.path.parent/'stop.request').touch()
        with patch.object(local.subprocess, 'Popen', side_effect=spawn), \
             patch.object(local, 'atomic', side_effect=publish), patch.object(local.time, 'sleep'):
            local.supervise(self.path)
        self.assertEqual(len(children), 5)
        for child in children:
            child.terminate.assert_called_once()
            child.wait.assert_called_once()
        self.assertEqual(json.loads((self.path.parent/'supervisor.json').read_bytes())['state'], 'STOPPED')


if __name__ == '__main__':
    unittest.main()
