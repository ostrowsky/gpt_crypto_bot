import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

import independent_signal_evaluator as evaluator
import policy_provenance as provenance
from test_candidate_dataset_quality import _row

NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def model():
    scope = {'last_feature_time': '2026-08-01T00:00:00Z',
             'last_label_time': '2026-08-02T00:00:00Z',
             'last_label_recorded_at': '2026-08-02T00:00:00Z',
             'policy_epoch_counts': {'pe-quality-v2': 1}}
    return {'feature_names': ['candidate_score'], 'evaluation_provenance': {
        'split_scopes': {k: copy.deepcopy(scope) for k in ('train', 'validation', 'test')}}}


def row(index, day=0):
    result = _row(index, action='blocked')
    feature = NOW + timedelta(days=2+day)
    bar = feature - timedelta(minutes=15)
    result['bar_ts'] = int(bar.timestamp()*1000)
    result['provenance']['feature_time'] = provenance.utc_iso(feature)
    result['decision_provenance']['decision_time'] = provenance.utc_iso(feature+timedelta(seconds=5))
    result['decision']['candidate_score'] = 10 if index % 2 == 0 else 1
    result['f']['predict'] = 0 if index % 2 == 0 else 1
    result['labels']['ret_5'] = -1 if index % 2 == 0 else 2
    due = provenance.forward_label_time(bar_ts=result['bar_ts'], tf='15m', horizon=5)
    result['label_provenance']['ret_5'].update(label_time=provenance.utc_iso(due),
                                            recorded_at=provenance.utc_iso(due))
    return result


def prediction(payload, causal):
    assert 'labels' not in causal and 'teacher' not in causal and 'label_provenance' not in causal
    return causal['f']['predict']


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.registry = self.root / 'registry'
        self.model = self.root / 'model.json'
        self.data = self.root / 'rows.jsonl'
        self.model.write_text(json.dumps(model()), encoding='utf-8')
        evaluator.register(self.model, self.registry, now=NOW)

    def run_rows(self, rows, now=None):
        self.data.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
        return evaluator.evaluate(self.registry, self.data, prediction,
                                  now=now or NOW+timedelta(days=3))

    def test_registration_does_not_replace_frozen_model(self):
        before = (self.registry/'candidate.json').read_bytes()
        self.model.write_text('{}', encoding='utf-8')
        evaluator.register(self.model, self.registry, now=NOW+timedelta(days=1))
        self.assertEqual(before, (self.registry/'candidate.json').read_bytes())

    def test_tampered_model_or_criteria_rejected(self):
        manifest_path = self.registry/'manifest.json'
        manifest = json.loads(manifest_path.read_bytes())
        manifest['min_groups'] = 1
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        self.data.write_text('', encoding='utf-8')
        with self.assertRaises(ValueError):
            evaluator.evaluate(self.registry, self.data)
        manifest['min_groups'] = 100
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        (self.registry/'candidate.json').write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError):
            evaluator.evaluate(self.registry, self.data)

    def test_unknown_training_exposure_rejected(self):
        other = self.root / 'other.json'
        other.write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError):
            evaluator.register(other, self.root/'other', now=NOW)

    def test_future_exposure_rejected(self):
        value = model()
        value['evaluation_provenance']['split_scopes']['test']['last_label_time'] = '2026-09-05T00:00:00Z'
        other = self.root/'future.json'
        other.write_text(json.dumps(value), encoding='utf-8')
        with self.assertRaises(ValueError):
            evaluator.register(other, self.root/'future', now=NOW)

    def test_pairing_and_small_sample_unknown(self):
        result = self.run_rows([row(0), row(1)])
        quality = result['learning_quality']
        self.assertEqual(quality['verdict'], 'UNKNOWN')
        self.assertEqual(quality['mean_daily_ret5_delta_pp'], 3)
        self.assertEqual(quality['candidate_positive'], {'numerator': 1, 'denominator': 1})
        self.assertIsNone(quality['positive_rate_lift'])
        self.assertEqual(result['signal_quality']['verdict'], 'UNKNOWN')
        self.assertFalse(result['runtime_eligible'])

    def test_training_era_data_not_reused(self):
        result = self.run_rows([row(0, -1), row(1, -1)])
        self.assertEqual(result['learning_quality']['eligible_rows'], 0)

    def test_same_bar_processing_timestamps_still_pair(self):
        rows = [row(0), row(1)]
        for container, key in [(rows[1]['provenance'], 'feature_time'),
                               (rows[1]['decision_provenance'], 'decision_time')]:
            container[key] = provenance.utc_iso(provenance.parse_utc(container[key])+timedelta(seconds=10))
        self.assertEqual(self.run_rows(rows)['learning_quality']['paired_groups'], 1)

    def test_duplicate_or_wrong_label_time_blocks(self):
        for kind in ('duplicate', 'wrong_time', 'future_recorded', 'policy'):
            rows = [row(0), row(1)]
            if kind == 'duplicate':
                rows.append(copy.deepcopy(rows[0]))
            elif kind == 'wrong_time':
                rows[0]['label_provenance']['ret_5']['label_time'] = '2026-09-03T00:00:00Z'
            elif kind == 'policy':
                rows[0]['provenance']['policy_epoch'] = 'different'
            else:
                rows[0]['label_provenance']['ret_5']['recorded_at'] = '2026-09-10T00:00:00Z'
            with self.subTest(kind=kind):
                self.assertEqual(self.run_rows(rows)['learning_quality']['verdict'], 'BLOCKED')

    def test_partial_group_is_not_scored(self):
        rows = [row(0), row(1), row(2)]
        rows[2]['labels']['ret_5'] = None
        due = provenance.forward_label_time(bar_ts=rows[0]['bar_ts'], tf='15m', horizon=5)
        result = self.run_rows(rows, due+timedelta(minutes=1))
        self.assertEqual(result['learning_quality']['paired_groups'], 0)
        self.assertEqual(result['learning_quality']['excluded_incomplete_groups'], 1)
        self.assertEqual(result['learning_quality']['verdict'], 'UNKNOWN')
        self.assertEqual(self.run_rows(rows)['learning_quality']['verdict'], 'BLOCKED')

    def test_bootstrap_positive_is_proxy_only(self):
        # Four groups per day, not four candidates in one group.
        rows = []
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
                        container[key] = provenance.utc_iso(provenance.parse_utc(container[key])+shift)
                rows.extend(pair)
        result = self.run_rows(rows, NOW+timedelta(days=40))
        self.assertEqual(result['learning_quality']['verdict'], 'PASS_PROXY')
        self.assertEqual(result['learning_quality']['paired_groups'], 120)
        self.assertFalse(result['achievement_claimed'])


if __name__ == '__main__':
    unittest.main()
