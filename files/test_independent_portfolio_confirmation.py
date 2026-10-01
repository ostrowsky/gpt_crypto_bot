import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import independent_portfolio_confirmation as intake
import forward_evidence_service as service
from test_validated_ranker_rollout import AUTHORITY, KEY, NOW, CANDIDATE, CHAMPION, bundle, evidence


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root/'portfolio_inputs.json'
        self.request = self.root/'portfolio_request.json'
        def save(name, raw):
            (self.root/name).write_bytes(raw)
            return {'path': name, 'sha256': intake.sha(raw)}
        self.inputs = {'contract': intake.CONTRACT, 'stage': 'CANARY',
                       'candidate': save('candidate.json', CANDIDATE),
                       'champion': save('champion.json', CHAMPION), 'evidence': {}}
        for n, phase in enumerate(('historical', 'sealed', 'shadow')):
            raw, cert = evidence(bundle(phase, n*40))
            self.inputs['evidence'][phase] = {'bundle': save(phase+'.json', raw),
                'certification': save(phase+'.cert.json', intake.canonical(cert))}
        self.write()

    def write(self):
        self.manifest.write_bytes(intake.canonical(self.inputs))

    def check(self):
        with patch.object(intake.gate, 'harness_passes', return_value=True):
            return intake.confirm(self.manifest, self.request, AUTHORITY, KEY, now=NOW)

    def test_complete_signed_cohorts_recompute_and_publish(self):
        r = self.check()
        self.assertEqual(r['state'], 'READY')
        self.assertFalse(r['runtime_eligible'])
        self.assertEqual({p['state'] for p in r['phases'].values()}, {'PASS'})
        self.assertEqual(r['phases']['shadow']['paired_days'], 30)
        self.assertIn('reports', r['phases']['shadow'])
        self.assertEqual(json.loads(self.request.read_bytes())['stage'], 'CANARY')

    def test_missing_manifest_never_synthesizes_authority(self):
        self.manifest.unlink()
        self.assertEqual(self.check()['state'], 'BLOCKED')
        self.assertFalse(self.request.exists())

    def test_partial_phases_preserve_unknown(self):
        del self.inputs['evidence']['shadow']
        self.write()
        r = self.check()
        self.assertEqual(r['phases']['shadow']['state'], 'UNKNOWN')
        self.assertEqual(r['phases']['historical']['state'], 'PASS')
        self.assertFalse(self.request.exists())

    def test_changed_model_bundle_certificate_bytes(self):
        for name in ('candidate.json', 'shadow.json', 'shadow.cert.json'):
            old = (self.root/name).read_bytes()
            (self.root/name).write_bytes(old+b' ')
            self.assertEqual(self.check()['state'], 'BLOCKED')
            (self.root/name).write_bytes(old)

    def test_phase_identity_mismatch(self):
        self.inputs['evidence']['shadow'] = self.inputs['evidence']['sealed']
        self.write()
        self.assertIn('phase identity', ' '.join(self.check()['blockers']))

    def test_stale_certificates_and_unexpected_phase_block(self):
        r = intake.confirm(self.manifest, self.request, AUTHORITY, KEY, now=NOW+7200)
        self.assertEqual(r['state'], 'BLOCKED')
        self.inputs['evidence']['unregistered'] = self.inputs['evidence']['sealed']
        self.write()
        self.assertIn('unexpected evidence', ' '.join(self.check()['blockers']))

    def test_source_binding_includes_confirmation_implementation(self):
        import inspect
        self.assertIn('independent_portfolio_confirmation.py', inspect.getsource(intake.gate.source_hash))

    def test_escape_and_absolute_paths(self):
        for path in ('../outside.json', str(self.root/'candidate.json')):
            self.inputs['candidate']['path'] = path
            self.write()
            self.assertEqual(self.check()['state'], 'BLOCKED')
        self.assertFalse(self.request.exists())

    def test_contract_stage_and_missing_canary(self):
        for field, value in (('contract', 'fixed-T5-proxy'), ('stage', 'AUTO')):
            old = self.inputs[field]
            self.inputs[field] = value
            self.write()
            self.assertEqual(self.check()['state'], 'BLOCKED')
            self.inputs[field] = old
        self.inputs['stage'] = 'PROMOTED'
        self.write()
        self.assertEqual(self.check()['phases']['canary']['state'], 'UNKNOWN')

    def test_missing_keys_and_harness_failure(self):
        r = intake.confirm(self.manifest, self.request, b'', KEY, now=NOW)
        self.assertEqual(r['state'], 'BLOCKED')
        with patch.object(intake.gate, 'harness_passes', return_value=False):
            r = intake.confirm(self.manifest, self.request, AUTHORITY, KEY, now=NOW)
        self.assertIn('Harness', ' '.join(r['blockers']))
        self.assertFalse(self.request.exists())

    def test_losing_candidate_rejected(self):
        b = bundle('shadow', 80)
        b['candidate_trades'] = b['champion_trades']
        raw, cert = evidence(b)
        for key, name, contents in (('bundle', 'shadow.json', raw),
                                   ('certification', 'shadow.cert.json', intake.canonical(cert))):
            (self.root/name).write_bytes(contents)
            self.inputs['evidence']['shadow'][key]['sha256'] = intake.sha(contents)
        self.write()
        self.assertEqual(self.check()['phases']['shadow']['state'], 'REJECTED')
        self.assertFalse(self.request.exists())

    def test_manifest_mutation_during_evaluation(self):
        def mutate(*a, **kw):
            self.manifest.write_bytes(b'{}')
        with patch.object(intake.gate, 'authorize', side_effect=mutate):
            r = intake.confirm(self.manifest, self.request, AUTHORITY, KEY, now=NOW)
        self.assertIn('changed during', ' '.join(r['blockers']))
        self.assertFalse(self.request.exists())

    def test_failed_current_confirmation_blocks_old_request_and_rolls_back(self):
        self.check()
        release = self.root/'release'
        service.atomic(release/'active.json', {'stage': 'CANARY'})
        with patch.object(intake.gate, 'authorize') as authorize:
            r = service.controller_tick(self.request, release,
                confirmation={'state':'BLOCKED', 'blockers':['gap']})
        authorize.assert_not_called()
        self.assertEqual(r['rollback'], 'applied')
        self.assertFalse(r['runtime_eligible'])

    def test_report_io_failure_still_runs_rollback(self):
        registry = self.root/'registry'
        registry.mkdir()
        release = self.root/'release'
        service.atomic(release/'active.json', {'stage':'CANARY'})
        deployment = {'evaluator_sid':'expected', 'registry':str(registry),
                      'dataset':str(self.root/'missing'), 'training_input':str(self.root/'train'),
                      'candidate_input':str(self.root/'pending'), 'portfolio_request':str(self.request),
                      'release_root':str(release), 'status':str(self.root/'status.json')}
        original = service.atomic
        def fail_report(path, value):
            if path.name == 'portfolio_confirmation_latest.json':
                raise PermissionError('report locked')
            return original(path, value)
        with patch.object(service, 'current_sid', return_value='expected'), \
             patch.object(service, 'export_training', side_effect=FileNotFoundError('input missing')), \
             patch.object(service, 'atomic', side_effect=fail_report):
            r = service.run_tick(deployment, 'evaluator')
        self.assertEqual(r['controller']['rollback'], 'applied')
        self.assertIn('publication failed', r['controller']['reason'])


if __name__ == '__main__':
    unittest.main()
