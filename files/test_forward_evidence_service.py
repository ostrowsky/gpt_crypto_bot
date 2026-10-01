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

    def test_wrong_sid_cannot_start_either_role(self):
        with patch.object(service, 'current_sid', return_value='wrong'):
            for role in ('trainer', 'evaluator'):
                with self.assertRaises(PermissionError):
                    service.run_tick({role+'_sid':'expected'}, role)

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
                       "-LogonType S4U", '-MultipleInstances IgnoreNew', "'Deny'", 'ConvertFrom-SecureString',
                       '$ResumeTrainerSid', '$user.SID.Value -ne $expectedSid', 'installation_public.json'):
            self.assertIn(phrase, text)
        self.assertNotIn("Get-ChildItem -LiteralPath (Join-Path $ProjectRoot '.runtime')", text)
        self.assertIn("$registered.Principal.LogonType -ne 'S4U'", text)
        self.assertIn('$credential.GetNetworkCredential().Password', text)

    def test_copied_gate_checks_live_project_harness(self):
        import independent_portfolio_gate as gate
        with patch.object(gate.subprocess, 'run') as run:
            run.return_value.returncode = 1
            self.assertFalse(gate.harness_passes(self.root))
            self.assertEqual(run.call_args.args[0][1], str(self.root/'files'/'truth_harness.py'))


if __name__ == '__main__':
    unittest.main()
