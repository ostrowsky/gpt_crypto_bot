import copy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import forward_evidence_service as service
import independent_signal_evaluator as evaluator
from test_independent_signal_evaluator import NOW, model, row, prediction


class ForwardServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.registry = self.root/'registry'
        candidate = self.root/'model.json'
        candidate.write_text(json.dumps(model()))
        evaluator.register(candidate, self.registry, now=NOW)
        self.dataset = self.root/'rows.jsonl'
        self.feature = NOW+timedelta(days=2)

    def write(self, rows):
        self.dataset.write_text('\n'.join(json.dumps(r) for r in rows))

    def observe(self, rows):
        pending = copy.deepcopy(rows)
        for r in pending:
            r['labels']['ret_5'] = None
            r['label_provenance']['ret_5'] = {}
        self.write(pending)
        return service.collect(self.registry, self.dataset, self.feature+timedelta(seconds=10))

    def test_arrival_maturity_restart_and_independent_evaluation(self):
        rows = [row(0), row(1)]
        first = self.observe(rows)
        self.assertEqual(first['new_observations'], 2)
        self.assertEqual(self.observe(rows)['new_observations'], 0)
        self.write(rows)
        mature = service.collect(self.registry, self.dataset, self.feature+timedelta(hours=2))
        self.assertEqual(mature['new_outcomes'], 2)
        self.assertEqual(service.collect(self.registry, self.dataset, self.feature+timedelta(hours=2))['new_outcomes'], 0)
        report = evaluator.evaluate(self.registry, self.registry/'forward_snapshot.jsonl', prediction, self.feature+timedelta(hours=2))
        self.assertFalse(report['runtime_eligible'])

    def test_retrospective_labeled_rows_never_admitted(self):
        self.write([row(0)])
        result = service.collect(self.registry, self.dataset, self.feature+timedelta(seconds=10))
        self.assertEqual(result['observations'], 0)

    def test_orphan_legacy_lock_does_not_reset_or_block_forward_journal(self):
        (self.registry/'collector.lock').touch()
        self.assertEqual(self.observe([row(0)])['new_observations'],1)
        before = (self.registry/'forward_journal.jsonl').read_bytes()
        self.assertEqual(self.observe([row(0)])['new_observations'],0)
        self.assertEqual((self.registry/'forward_journal.jsonl').read_bytes(),before)

    def test_late_observation_rejected(self):
        r = row(0)
        r['labels']['ret_5'] = None
        self.write([r])
        self.assertEqual(service.collect(self.registry, self.dataset, self.feature+timedelta(minutes=3))['observations'], 0)

    def test_future_label_not_collected(self):
        rows = [row(0)]
        self.observe(rows)
        self.write(rows)
        self.assertEqual(service.collect(self.registry, self.dataset, self.feature+timedelta(minutes=1))['outcomes'], 0)

    def test_conflicting_feature_blocks(self):
        rows = [row(0)]
        self.observe(rows)
        rows[0]['f']['predict'] = 9
        self.write(rows)
        with self.assertRaisesRegex(ValueError, 'identity conflict'):
            service.collect(self.registry, self.dataset, self.feature+timedelta(hours=2))

    def test_tampered_chain_blocks(self):
        self.observe([row(0)])
        journal = self.registry/'forward_journal.jsonl'
        record = json.loads(journal.read_text())
        record['body']['row']['f']['predict'] = 7
        journal.write_text(json.dumps(record)+'\n')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            service.collect(self.registry, self.dataset, self.feature+timedelta(hours=2))

    def test_training_export_excludes_forward_exposure_and_unknown_provenance(self):
        self.write([row(0, -5), row(1), {'id':'legacy'}])
        output = self.root/'training.jsonl'
        count = service.export_training(self.dataset, output, self.feature)
        self.assertEqual(count, 1)
        self.assertEqual(json.loads(output.read_text())['id'], row(0, -5)['id'])

    def test_windows_read_lock_preserves_previous_training_snapshot(self):
        self.write([row(0, -5)])
        output = self.root/'training.jsonl'
        output.write_bytes(b'previous-complete-snapshot\n')
        locked = OSError('sharing violation')
        locked.winerror = 32
        with patch.object(Path, 'replace', side_effect=locked):
            self.assertIsNone(service.export_training(self.dataset, output, self.feature))
        self.assertEqual(output.read_bytes(), b'previous-complete-snapshot\n')
        self.assertFalse(output.with_suffix('.tmp').exists())

    def test_wrong_sid_cannot_start_either_role(self):
        with patch.object(service, 'current_sid', return_value='wrong'):
            for role in ('trainer', 'evaluator', 'exporter'):
                with self.assertRaises(PermissionError):
                    service.run_tick({'trainer_sid':'expected','evaluator_sid':'expected'}, role)

    def test_same_user_readable_holdout_blocks_training(self):
        with self.assertRaisesRegex(PermissionError, 'protected registry'):
            service.verify_trainer_isolation({'registry':str(self.registry)})

    def test_missing_portfolio_request_blocks_and_rolls_back(self):
        release = self.root/'release'
        service.atomic(release/'active.json', {'stage':'CANARY'})
        result = service.controller_tick(self.root/'missing.json', release)
        self.assertEqual(result['state'], 'BLOCKED')
        self.assertEqual(result['rollback'], 'applied')
        self.assertFalse(result['runtime_eligible'])

    def test_provisioning_intent_not_an_acl_proof(self):
        script = Path(__file__).parents[1]/'install_learning_roles.ps1'
        text = script.read_text()
        for phrase in ('Administrator PowerShell is required', 'GptBotTrainer', 'GptBotEvaluator',
                       'LogonType = 2', 'MultipleInstances = 2', "'Deny'", 'ConvertFrom-SecureString',
                       '$ResumeTrainerSid', '$user.SID.Value -ne $expectedSid', 'installation_public.json'):
            self.assertIn(phrase, text)
        self.assertNotIn("Get-ChildItem -LiteralPath (Join-Path $ProjectRoot '.runtime')", text)
        self.assertIn('$registered.Definition.Principal.LogonType -ne 2', text)
        self.assertIn('$folder.RegisterTaskDefinition', text)
        self.assertIn('Refusing to replace a task belonging to another principal.', text)
        self.assertIn('$credential.GetNetworkCredential().Password', text)
        self.assertIn('$trigger.Enabled = $true', text)
        self.assertIn('$definition.Settings.DisallowStartIfOnBatteries = $false', text)
        self.assertIn('[GptBotBatchRights]::Grant($trainerSid)', text)
        self.assertIn('[GptBotBatchRights]::Grant($evaluatorSid)', text)
        self.assertIn('LsaAddAccountRights', text)
        self.assertNotIn('LsaRemoveAccountRights', text)

    def test_copied_gate_checks_live_project_harness(self):
        import independent_portfolio_gate as gate
        with patch.object(gate.subprocess, 'run') as run:
            run.return_value.returncode = 1
            self.assertFalse(gate.harness_passes(self.root))
            self.assertEqual(run.call_args.args[0][1], str(self.root/'files'/'truth_harness.py'))

    def test_embedded_interpreter_resolves_role_source_first(self):
        import sys
        self.assertEqual(sys.path[0], str(Path(service.__file__).resolve().parent))

    def deployment(self):
        return {'evaluator_sid':'expected', 'registry':str(self.root/'new'),
                'dataset':str(self.dataset), 'training_input':str(self.root/'train'),
                'candidate_input':str(self.root/'pending'), 'portfolio_request':str(self.root/'request'),
                'release_root':str(self.root/'release'), 'status':str(self.root/'status')}

    def test_minute_bootstrap_never_performs_full_export(self):
        deployment = self.deployment()
        with patch.object(service, 'current_sid', return_value='expected'), \
             patch.object(service, 'export_training', return_value=1) as export, \
             patch.object(evaluator, 'register', side_effect=FileNotFoundError('pending candidate')):
            result = service.run_tick(deployment, 'evaluator')
        export.assert_not_called()
        self.assertEqual(result['controller']['state'], 'BLOCKED')

    def test_exporter_bootstraps_and_does_not_write_collector_status(self):
        deployment = self.deployment()
        with patch.object(service,'current_sid',return_value='expected'), \
             patch.object(service,'export_training',return_value={'rows':1}) as export:
            result = service.run_tick(deployment,'exporter')
        self.assertGreater(export.call_args.args[2],NOW)
        self.assertTrue(export.call_args.kwargs['immutable'])
        self.assertEqual(result['state'],'EXPORTED_NOT_APPROVED')
        self.assertFalse(Path(deployment['status']).exists())

    def test_exporter_pins_cutoff_and_releases_registration_barrier_before_scan(self):
        deployment = self.deployment()
        deployment['registry'] = str(self.registry)
        cutoff = json.loads((self.registry/'manifest.json').read_bytes())['holdout_start']
        def scan(*args, **kwargs):
            with service.process_lock(self.registry/'registration.lock'):
                pass
            with self.assertRaises(BlockingIOError):
                with service.process_lock(self.registry/'training_export.lock'):
                    pass
            return {'rows':1}
        with patch.object(service,'current_sid',return_value='expected'), \
             patch.object(service,'export_training',side_effect=scan) as export:
            result = service.run_tick(deployment,'exporter')
        self.assertEqual(export.call_args.args[2],service.provenance.parse_utc(cutoff))
        self.assertEqual(result['state'],'EXPORTED_NOT_APPROVED')

    def test_invalid_cutoff_blocks_without_export(self):
        deployment = self.deployment()
        deployment['registry'] = str(self.registry)
        service.atomic(self.registry/'manifest.json',{'holdout_start':'not-a-date'})
        with patch.object(service,'current_sid',return_value='expected'), \
             patch.object(service,'export_training') as export:
            result = service.run_tick(deployment,'exporter')
        export.assert_not_called()
        self.assertEqual(result['state'],'BLOCKED')

    def test_minute_success_is_proxy_only_even_when_controller_authorizes(self):
        deployment = self.deployment()
        deployment['registry'] = str(self.registry)
        import independent_portfolio_confirmation as confirmation
        with patch.object(service,'current_sid',return_value='expected'), \
             patch.object(evaluator,'register'), patch.object(service,'collect',return_value={'observations':1}), \
             patch.object(evaluator,'evaluate',return_value={'learning_quality':'UNKNOWN'}), \
             patch.object(service,'export_training') as export, \
             patch.object(confirmation,'confirm',return_value={'state':'READY'}), \
             patch.object(service,'controller_tick',return_value={'state':'CANARY','runtime_eligible':True}):
            result = service.run_tick(deployment,'evaluator')
        export.assert_not_called()
        self.assertEqual(result['state'],'COLLECTED_PROXY_ONLY')
        self.assertTrue(result['activation_authorized'])
        self.assertFalse(result['closed_loop'])
        self.assertEqual(result['production_effect'],'UNKNOWN')

    def test_unknown_role_rejected(self):
        with self.assertRaises(ValueError):
            service.run_tick({},'administrator')


if __name__ == '__main__':
    unittest.main()
