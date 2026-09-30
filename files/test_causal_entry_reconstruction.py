import copy
import asyncio
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import causal_entry_reconstruction as repair
import critic_dataset
import independent_signal_evaluator as evaluator
from test_independent_signal_evaluator import NOW, row, model, prediction


def bar(ts, step, o=100, c=100):
    return [ts, str(o), str(max(o, c)+1), str(min(o, c)-1), str(c), '1', ts+step-1]


def reconstructed(index=0):
    r = row(index)
    feature = evaluator.provenance.parse_utc(r['provenance']['feature_time'])
    r['decision_provenance']['decision_time'] = evaluator.provenance.utc_iso(feature+timedelta(minutes=8, seconds=15))
    nxt = int((feature+timedelta(minutes=9)).timestamp()*1000)
    target = 99 if index % 2 == 0 else 102
    window = [bar(r['bar_ts']+i*900_000, 900_000, c=target if i == 5 else 100) for i in range(7)]
    r['execution_reconstruction'] = {
        'contract': repair.CONTRACT,
        'minute_bars': [bar(nxt-120_000, 60_000), bar(nxt, 60_000, o=101), bar(nxt+60_000, 60_000)],
        'target_bars': window, 'minute_response_sha256': 'a'*64, 'target_response_sha256': 'b'*64,
        'retrieved_at': evaluator.provenance.utc_iso(feature+timedelta(hours=3)),
    }
    return r


class CausalRepairTests(unittest.TestCase):
    def test_identical_duplicates_collapse_without_selection(self):
        original = row(0)
        result, counts = repair.canonical([original, copy.deepcopy(original)])
        self.assertEqual(result, [original])
        self.assertEqual(counts['removed_duplicate_rows'], 1)
        self.assertEqual(counts['quarantined_ids'], 0)

    def test_any_conflicting_variant_is_unknown_not_best_return(self):
        for key in ('f', 'seq', 'decision', 'provenance', 'labels'):
            a = row(0)
            b = copy.deepcopy(a)
            b[key] = {'different': 999}
            result, counts = repair.canonical([a, b])
            self.assertEqual(result[0]['labels']['ret_5'], a['labels']['ret_5'])
            self.assertEqual(result[0]['reconstruction_integrity'], 'conflicting_duplicate')
            self.assertEqual(counts['quarantined_ids'], 1)

    def test_next_minute_not_original_close_or_future_low(self):
        r = reconstructed()
        self.assertAlmostEqual(repair.execution_return(r, NOW+timedelta(days=3)), (99/101-1)*100)
        self.assertEqual(r['labels']['ret_5'], -1)
        r['execution_reconstruction']['minute_bars'][1][3] = '1'
        self.assertAlmostEqual(repair.execution_return(r, NOW+timedelta(days=3)), (99/101-1)*100)

    def test_missing_proof_wrong_boundary_and_bad_prices_rejected(self):
        for kind in ('missing', 'boundary', 'nan', 'zero', 'gap', 'target', 'future', 'source'):
            r = reconstructed()
            e = r['execution_reconstruction']
            if kind == 'missing': e['minute_bars'].pop()
            elif kind == 'boundary': e['minute_bars'][1][0] -= 60_000
            elif kind == 'nan': e['minute_bars'][1][1] = 'nan'
            elif kind == 'zero': e['minute_bars'][1][1] = '0'
            elif kind == 'gap': e['target_bars'].pop(2)
            elif kind == 'target': r['labels']['ret_5'] = 40
            elif kind == 'future': e['retrieved_at'] = evaluator.provenance.utc_iso(NOW+timedelta(days=4))
            else: e['minute_response_sha256'] = ''
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                repair.execution_return(r, NOW+timedelta(days=3))

    def test_entire_group_quarantined_and_predictor_cannot_see_outcomes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            m = root/'model.json'
            m.write_text(json.dumps(model()), encoding='utf-8')
            evaluator.register(m, root/'registry', NOW)
            def pred(payload, causal):
                self.assertNotIn('execution_reconstruction', causal)
                return prediction(payload, causal)
            data = root/'data.jsonl'
            rows = [reconstructed(0), reconstructed(1)]
            data.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            report = evaluator.evaluate(root/'registry', data, pred, NOW+timedelta(days=3), historical=True, execution=True)
            self.assertEqual(report['learning_quality']['paired_groups'], 1)
            self.assertEqual(report['contract'], 'independent-retrospective-causal-entry-v1')
            self.assertFalse(report['runtime_eligible'])
            rows.append(reconstructed(2))
            rows[-1]['reconstruction_integrity'] = 'conflicting_duplicate'
            data.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            report = evaluator.evaluate(root/'registry', data, pred, NOW+timedelta(days=3), historical=True, execution=True)
            self.assertEqual(report['learning_quality']['paired_groups'], 0)
            self.assertEqual(report['learning_quality']['verdict'], 'BLOCKED')

    def test_prospective_contract_cannot_be_retroactively_changed(self):
        with self.assertRaises(ValueError):
            evaluator.evaluate(Path('.'), Path('.'), execution=True)

    def test_restart_cannot_append_same_id(self):
        with tempfile.TemporaryDirectory() as td, patch.object(critic_dataset, 'CRITIC_FILE', Path(td)/'data.jsonl'):
            self.assertTrue(critic_dataset._append({'id': 'a', 'decision': 1}))
            critic_dataset._logged_candidates.clear()
            self.assertFalse(critic_dataset._append({'id': 'a', 'decision': 2}))
            self.assertEqual(len(critic_dataset.CRITIC_FILE.read_text().splitlines()), 1)

    def test_multiple_processes_share_uniqueness_barrier(self):
        if critic_dataset.msvcrt is None:
            self.skipTest('Windows cross-process lock required')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'data.jsonl'
            code = "import critic_dataset as d;from pathlib import Path;d.CRITIC_FILE=Path(%r);d._append({'id':'shared'})" % str(path)
            children = [subprocess.Popen([sys.executable, '-c', code], cwd=Path(__file__).parent,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(3)]
            for child in children:
                out, err = child.communicate(timeout=60)
                self.assertEqual(child.returncode, 0, err.decode(errors='replace'))
            self.assertEqual(len(path.read_text().splitlines()), 1)

    def test_corrupt_dataset_never_accepts_new_append(self):
        with tempfile.TemporaryDirectory() as td, patch.object(critic_dataset, 'CRITIC_FILE', Path(td)/'data.jsonl'):
            critic_dataset.CRITIC_FILE.write_text('bad json\n')
            with self.assertRaises(ValueError):
                critic_dataset._append({'id': 'new'})

    def test_cache_detects_external_append_and_atomic_replace(self):
        with tempfile.TemporaryDirectory() as td, patch.object(critic_dataset, 'CRITIC_FILE', Path(td)/'data.jsonl'):
            path = critic_dataset.CRITIC_FILE
            self.assertTrue(critic_dataset._append({'id': 'a'}))
            with path.open('a') as f:
                f.write(json.dumps({'id': 'b'})+'\n')
            self.assertFalse(critic_dataset._append({'id': 'b'}))
            replacement = path.with_suffix('.new')
            replacement.write_text(json.dumps({'id': 'c'})+'\n')
            replacement.replace(path)
            self.assertFalse(critic_dataset._append({'id': 'c'}))
            self.assertTrue(critic_dataset._append({'id': 'a'}))

    def test_own_rewrite_reuses_verified_ids_without_extra_full_scan(self):
        with tempfile.TemporaryDirectory() as td, patch.object(critic_dataset, 'CRITIC_FILE', Path(td)/'data.jsonl'):
            critic_dataset._append({'id': 'a', 'value': 0})
            def mutate(r):
                r['value'] = 1
                return True
            critic_dataset._rewrite_records(mutate, strict=True)
            with patch.object(critic_dataset.json, 'loads', side_effect=AssertionError('no repeated full scan')):
                self.assertFalse(critic_dataset._append({'id': 'a'}))

    def test_live_cleanup_archives_all_variants_and_excludes_conflicts(self):
        with tempfile.TemporaryDirectory() as td, patch.object(critic_dataset, 'CRITIC_FILE', Path(td)/'data.jsonl'):
            path = critic_dataset.CRITIC_FILE
            rows = [{'id': 'same', 'f': 1}, {'id': 'same', 'f': 1},
                    {'id': 'conflict', 'f': 1}, {'id': 'conflict', 'f': 2}, {'id': 'good'}]
            path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            original = path.read_bytes()
            report = repair.deduplicate_live(path, Path(td)/'archive')
            self.assertEqual((Path(td)/'archive'/'original.jsonl').read_bytes(), original)
            saved = [json.loads(l) for l in path.read_bytes().splitlines()]
            self.assertEqual([r['id'] for r in saved], ['same', 'good'])
            self.assertEqual(report['retained_rows'], 2)
            self.assertEqual(report['quarantined_ids'], ['conflict'])
            self.assertFalse(critic_dataset._append({'id': 'good'}))

    def test_log_restart_updates_priority_and_changed_decision_time(self):
        with tempfile.TemporaryDirectory() as td, patch.object(critic_dataset, 'CRITIC_FILE', Path(td)/'data.jsonl'):
            data = row(0)
            first = data['decision_provenance'].copy()
            second = {**first, 'decision_time': evaluator.provenance.utc_iso(NOW+timedelta(days=2, minutes=9))}
            kwargs = dict(sym=data['sym'], tf='15m', bar_ts=data['bar_ts'],
                          signal_type='trend', is_bull_day=True, feat={}, i=0, data=None)
            critic_dataset._logged_candidates.clear()
            with patch.object(critic_dataset, 'build_runtime_record', return_value={}), patch.object(
                    critic_dataset.policy_provenance, 'build_observation_provenance', side_effect=[first, second]):
                record_id = critic_dataset.log_candidate(**kwargs, action='candidate', stage='collector')
                critic_dataset._logged_candidates.clear()
                self.assertEqual(record_id, critic_dataset.log_candidate(**kwargs, action='blocked',
                                                                          stage='quality_floor', candidate_score=20))
            saved = [json.loads(l) for l in critic_dataset.CRITIC_FILE.read_text().splitlines()]
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]['decision']['candidate_score'], 20)
            self.assertEqual(saved[0]['decision_provenance']['decision_time'], second['decision_time'])
            self.assertEqual(saved[0]['decision_history'][0]['decision_provenance'], first)

    def test_immutable_reconstruction_response_binding_and_cache_reuse(self):
        rows = [reconstructed(0), reconstructed(1)]
        markets = {}
        for r in rows:
            for tf, bars in (('1m', r['execution_reconstruction']['minute_bars']),
                             ('15m', r['execution_reconstruction']['target_bars'])):
                markets[(r['sym'], tf)] = bars
            r.pop('execution_reconstruction')
        class Response:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def raise_for_status(self): pass
            async def read(self): return self.raw
        class Session:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def get(self, url, params, **kwargs):
                response = Response()
                response.raw = json.dumps(markets[(params['symbol'], params['interval'])]).encode()
                return response
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidate = root/'model.json'
            candidate.write_text(json.dumps(model()), encoding='utf-8')
            evaluator.register(candidate, root/'registry', NOW)
            dataset = root/'data.jsonl'
            dataset.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            before = dataset.read_bytes()
            original_evaluate = evaluator.evaluate
            def evaluate(*args, **kwargs):
                kwargs['predictor'] = prediction
                return original_evaluate(*args, **kwargs)
            with patch.object(repair.aiohttp, 'ClientSession', Session), patch.object(repair.evaluator, 'evaluate', side_effect=evaluate):
                report = asyncio.run(repair.reconstruct(dataset, root/'registry', root/'run', NOW+timedelta(days=3)))
                self.assertEqual(report['repair']['restored_rows'], 2)
                self.assertEqual(repair.verify(root/'run'), report)
                self.assertEqual(dataset.read_bytes(), before)
                with patch.object(Session, 'get', side_effect=AssertionError('cache must avoid network')):
                    other = asyncio.run(repair.reconstruct(dataset, root/'registry', root/'reused',
                                                           NOW+timedelta(days=3), root/'run'))
                self.assertEqual(other['repair']['restored_rows'], 2)
            first = next((root/'run'/'responses').iterdir())
            first.write_text('[]')
            with self.assertRaises(ValueError):
                repair.verify(root/'run')


if __name__ == '__main__':
    unittest.main()
