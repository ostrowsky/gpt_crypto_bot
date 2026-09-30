import copy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import independent_signal_evaluator as evaluator
import historical_signal_evaluation as historical
import safe_signal_release as controller
from test_independent_signal_evaluator import NOW, model, row, prediction


class HistoricalReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        predictor_patch = patch.object(controller, 'predict_score', side_effect=prediction)
        predictor_patch.start()
        self.addCleanup(predictor_patch.stop)
        self.root = Path(self.tmp.name)
        self.registry = self.root/'registry'
        self.model = self.root/'model.json'
        self.data = self.root/'data.jsonl'
        self.run = self.root/'audit'
        self.model.write_text(json.dumps(model()), encoding='utf-8')
        evaluator.register(self.model, self.registry, NOW)
        self.now = NOW+timedelta(days=40)
        self.rows = [row(0), row(1), row(2, -1)]
        self.write_data()

    def write_data(self):
        self.data.write_text('\n'.join(json.dumps(r) for r in self.rows), encoding='utf-8')

    def audit(self):
        return historical.audit(self.data, self.registry, self.run, self.now, prediction)

    def forward(self):
        result = evaluator.evaluate(self.registry, self.data, prediction, self.now)
        path = self.registry/'evaluation_latest.json'
        path.write_text(json.dumps(result), encoding='utf-8')
        return path

    def test_entire_history_is_snapshotted_and_exposure_used(self):
        result = self.audit()
        self.assertEqual(result['available_history']['rows'], 3)
        self.assertEqual(result['evaluation_start'], '2026-08-04T00:00:00Z')
        self.assertFalse(result['sealed_holdout'])
        self.assertFalse(result['runtime_eligible'])
        self.assertEqual((self.run/'snapshot.jsonl').read_bytes(), self.data.read_bytes())

    def test_maximum_exposure_includes_validation_test_recording(self):
        payload = model()
        payload['evaluation_provenance']['split_scopes']['test']['last_label_recorded_at'] = '2026-08-31T12:00:00Z'
        other = self.root/'other.json'
        other.write_text(json.dumps(payload), encoding='utf-8')
        registry = self.root/'other'
        evaluator.register(other, registry, NOW)
        result = historical.audit(self.data, registry, self.run, self.now, prediction)
        self.assertEqual(result['evaluation_start'], '2026-09-02T12:00:00Z')
        self.assertEqual(result['available_history']['excluded_before_cutoff_rows'], 1)

    def test_restart_reuses_frozen_inputs_not_new_dataset(self):
        before = self.audit()
        self.data.write_text('', encoding='utf-8')
        self.assertEqual(before, self.audit())

    def test_snapshot_or_result_tampering_rejected(self):
        self.audit()
        (self.run/'snapshot.jsonl').write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.audit()
        with self.assertRaises(ValueError):
            controller.verify_history(self.run)

    def test_verifier_reads_frozen_trace(self):
        result = self.audit()
        self.assertEqual(controller.verify_history(self.run), result)
        self.assertEqual(controller.reconstruct(result), 'UNKNOWN')

    def test_controller_cannot_promote_even_with_matching_evidence(self):
        self.audit()
        decision = controller.decide(self.run, self.forward(), self.registry, self.now)
        self.assertEqual(decision['state'], 'BLOCKED')
        self.assertFalse(decision['runtime_eligible'])
        self.assertIn('maximum_period_after_cost_ten_slot_replay_missing', decision['reasons'])

    def test_wrong_candidate_stale_and_other_scope_fail_closed(self):
        self.audit()
        path = self.forward()
        original = json.loads(path.read_bytes())
        for kind in ('candidate', 'stale', 'future', 'scope'):
            value = copy.deepcopy(original)
            if kind == 'candidate':
                value['candidate_sha256'] = 'other'
            elif kind == 'stale':
                value['generated_at'] = evaluator.provenance.utc_iso(self.now-timedelta(days=2))
            elif kind == 'future':
                value['generated_at'] = evaluator.provenance.utc_iso(self.now+timedelta(hours=1))
            else:
                value['evaluation_scope'] = 'trainer_test'
            path.write_text(json.dumps(value), encoding='utf-8')
            decision = controller.decide(self.run, path, self.registry, self.now)
            self.assertEqual(decision['state'], 'BLOCKED')
            self.assertTrue(any('evidence_verification_failed' in r for r in decision['reasons']))

    def test_summary_and_verdict_tampering_rejected(self):
        result = self.audit()
        for key, value in [('paired_groups', 999), ('verdict', 'PASS_PROXY'),
                           ('paired_daily_95ci', [1, 2]), ('positive_rate_lift', 99)]:
            bad = copy.deepcopy(result)
            bad['learning_quality'][key] = value
            with self.assertRaises(ValueError):
                controller.reconstruct(bad)

    def test_forged_selected_candidate_fails_independent_reconstruction(self):
        result = self.audit()
        pair = result['pairs'][0]
        pair['candidate_id'] = pair['baseline_id']
        pair['candidate_ret5'] = pair['baseline_ret5']
        pair['delta'] = 0
        path = self.run/'result.json'
        path.write_text(json.dumps(result), encoding='utf-8')
        (self.run/'receipt.json').write_text(json.dumps({'result_sha256': historical.sha(path)}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'decision reconstruction'):
            controller.verify_history(self.run)

    def test_proxy_pass_can_only_reach_shadow(self):
        self.rows = []
        for day in range(30):
            for group in range(4):
                pair = [row(day*8+group*2, day), row(day*8+group*2+1, day)]
                shift = timedelta(hours=group*3)
                for r in pair:
                    r['bar_ts'] += int(shift.total_seconds()*1000)
                    for container, key in [(r['provenance'], 'feature_time'),
                                           (r['decision_provenance'], 'decision_time'),
                                           (r['label_provenance']['ret_5'], 'label_time'),
                                           (r['label_provenance']['ret_5'], 'recorded_at')]:
                        container[key] = evaluator.provenance.utc_iso(evaluator.provenance.parse_utc(container[key])+shift)
                self.rows.extend(pair)
        self.write_data()
        self.audit()
        decision = controller.decide(self.run, self.forward(), self.registry, self.now)
        self.assertEqual(decision['state'], 'SHADOW')
        self.assertFalse(decision['runtime_eligible'])

    def test_proxy_failure_rejects_candidate(self):
        self.rows = []
        for day in range(30):
            for group in range(4):
                pair = [row(day*8+group*2, day), row(day*8+group*2+1, day)]
                shift = timedelta(hours=group*3)
                for r in pair:
                    r['bar_ts'] += int(shift.total_seconds()*1000)
                    for container, key in [(r['provenance'], 'feature_time'),
                                           (r['decision_provenance'], 'decision_time'),
                                           (r['label_provenance']['ret_5'], 'label_time'),
                                           (r['label_provenance']['ret_5'], 'recorded_at')]:
                        container[key] = evaluator.provenance.utc_iso(evaluator.provenance.parse_utc(container[key])+shift)
                    r['labels']['ret_5'] *= -1
                self.rows.extend(pair)
        self.write_data()
        self.audit()
        decision = controller.decide(self.run, self.forward(), self.registry, self.now)
        self.assertEqual(decision['state'], 'REJECTED')
        self.assertFalse(decision['achievement_claimed'])

    def test_ledger_restart_idempotence_and_hash_chain(self):
        self.audit()
        decision = controller.decide(self.run, self.forward(), self.registry, self.now)
        state = self.root/'state'
        controller.persist(state, decision)
        later = dict(decision, generated_at=evaluator.provenance.utc_iso(self.now+timedelta(minutes=1)))
        controller.persist(state, later)
        ledger = state/'transitions.jsonl'
        self.assertEqual(len(ledger.read_text().splitlines()), 1)
        rejected = dict(later, state='REJECTED')
        controller.persist(state, rejected)
        self.assertEqual(len(ledger.read_text().splitlines()), 2)
        with self.assertRaises(ValueError):
            controller.persist(state, later)

    def test_ledger_tampering_or_concurrent_run_blocks(self):
        decision = {'generated_at': 'now', 'candidate_sha256': 'x', 'state': 'BLOCKED',
                    'runtime_eligible': False, 'achievement_claimed': False}
        state = self.root/'state'
        controller.persist(state, decision)
        ledger = state/'transitions.jsonl'
        entry = json.loads(ledger.read_text())
        entry['body']['state'] = 'SHADOW'
        ledger.write_text(json.dumps(entry)+'\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            controller.persist(state, decision)
        (state/'controller.lock').touch()
        with self.assertRaises(FileExistsError):
            controller.persist(state, decision)

    def test_candidate_rotation_requires_new_registry(self):
        state = self.root/'state'
        decision = {'generated_at': 'now', 'candidate_sha256': 'x', 'state': 'BLOCKED',
                    'runtime_eligible': False, 'achievement_claimed': False}
        controller.persist(state, decision)
        with self.assertRaises(ValueError):
            controller.persist(state, dict(decision, candidate_sha256='y'))

    def test_no_direct_promoted_state_or_runtime_permission(self):
        decision = {'state': 'PROMOTED', 'runtime_eligible': False, 'achievement_claimed': False}
        with self.assertRaises(ValueError):
            controller.persist(self.root/'state', decision)
        with self.assertRaises(ValueError):
            controller.persist(self.root/'state', dict(decision, state='SHADOW', runtime_eligible=True))


if __name__ == '__main__':
    unittest.main()
