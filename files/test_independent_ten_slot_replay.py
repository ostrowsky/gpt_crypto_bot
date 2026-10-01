import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import independent_ten_slot_replay as replay
from test_causal_entry_reconstruction import reconstructed
from test_independent_signal_evaluator import NOW
from datetime import timedelta


def event(i=0, **kw):
    return {'id': str(i), 'sym': 'S'+str(i), 'entry_ms': 100,
            'exit_ms': 200, 'entry_price': 100, 'exit_price': 110,
            'baseline_score': i, 'candidate_score': -i, **kw}


class PortfolioTests(unittest.TestCase):
    def test_cash_and_full_liquidation(self):
        r = replay.simulate([event()], 'baseline_score', 0, 0)
        self.assertAlmostEqual(r['final_equity'], 1.01)
        self.assertAlmostEqual(r['fills'][0]['net_pnl'], .01)

    def test_capacity_and_ranking(self):
        r = replay.simulate([event(i) for i in range(12)], 'baseline_score', 0, 0)
        self.assertEqual(r['trades'], 10)
        self.assertEqual({f['id'] for f in r['fills']}, set(map(str, range(2, 12))))
        self.assertEqual(r['capacity_competition_batches'], {'numerator': 1, 'denominator': 1})

    def test_late_better_score_cannot_displace_earlier(self):
        events = [event(i) for i in range(10)]+[event(11, entry_ms=101, baseline_score=999)]
        r = replay.simulate(events, 'baseline_score', 0, 0)
        self.assertNotIn('11', [f['id'] for f in r['fills']])

    def test_one_symbol_across_timeframes(self):
        r = replay.simulate([event(0, sym='A'), event(1, sym='A')], 'baseline_score')
        self.assertEqual(r['trades'], 1)
        self.assertEqual(r['skips'][0]['reason'], 'symbol_already_open')

    def test_exit_before_same_time_entry_and_cash_guard(self):
        rows = [event(i, exit_price=50) for i in range(10)]
        rows += [event(i+20, entry_ms=200, exit_ms=300) for i in range(10)]
        r = replay.simulate(rows, 'baseline_score', 0, 0)
        self.assertEqual(r['trades'], 15)
        self.assertEqual(sum(s['reason'] == 'cash' for s in r['skips']), 5)

    def test_adverse_costs(self):
        r = replay.simulate([event(exit_price=100)], 'baseline_score', 10, 5)
        self.assertLess(r['net_return_pct'], 0)
        self.assertGreater(r['fees_initial_capital'], 0)
        self.assertIsNone(r['mark_to_market_drawdown_pct'])

    def test_invalid_numbers_timing_ids(self):
        for kw in ({'entry_price': 0}, {'exit_price': float('nan')},
                   {'entry_ms': 200}, {'entry_ms': 100.5}, {'baseline_score': float('inf')}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                replay.simulate([event(**kw)], 'baseline_score')
        with self.assertRaises(ValueError):
            replay.simulate([event(), event()], 'baseline_score')
        for cost in (-1, float('nan'), 10000):
            with self.assertRaises(ValueError):
                replay.simulate([], 'baseline_score', cost)

    def test_order_independent_and_empty_denominator(self):
        rows = [event(i, baseline_score=1) for i in range(12)]
        a = replay.simulate(rows, 'baseline_score')
        self.assertEqual(a, replay.simulate(list(reversed(rows)), 'baseline_score'))
        self.assertEqual(replay.simulate([], 'baseline_score')['positive_trades'],
                         {'numerator': 0, 'denominator': 0})

    def test_immutable_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            report = {'runtime_eligible': False,
                      'evaluator_source_sha256': replay.sha(Path(replay.__file__))}
            replay.publish(out, report)
            original = (out/'result.json').read_bytes()
            with self.assertRaises(FileExistsError):
                replay.publish(out, report)
            self.assertEqual(original, (out/'result.json').read_bytes())

    def test_build_filters_incomplete_groups_and_removes_future_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            rows = [reconstructed(0), reconstructed(1)]
            group = rows[0]['provenance']['feature_time']+'|'+rows[0]['tf']
            (run/'repaired.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (run/'candidate.json').write_text('{}')
            (run/'manifest.json').write_text('{}')
            (run/'receipt.json').write_text('{}')
            (run/'repair_manifest.json').write_text('{"sources":{}}')
            checked = {'pairs': [{'group': group}], 'available_history': {'rows': 2},
                       'evaluation_start': 'exposure+embargo', 'learning_quality': {}}
            def predict(payload, row):
                for key in ('teacher', 'labels', 'label_provenance', 'execution_reconstruction'):
                    self.assertNotIn(key, row)
                return 1
            with patch.object(replay.repair, 'verify', return_value={
                    'generated_at': replay.evaluator.provenance.utc_iso(NOW+timedelta(days=3))}), \
                 patch.object(replay.evaluator, 'evaluate', return_value=checked) as evaluate, \
                 patch.object(replay, 'predict_final_score_from_candidate_payload', side_effect=predict):
                report = replay.build(run)
            self.assertEqual(report['eligible_events'], 2)
            self.assertEqual(report['state'], 'BLOCKED')
            self.assertFalse(report['runtime_eligible'])
            self.assertFalse(report['achievement_claimed'])
            self.assertTrue(evaluate.call_args.kwargs['historical'])
            self.assertTrue(evaluate.call_args.kwargs['execution'])
            checked['pairs'] = []
            with patch.object(replay.repair, 'verify', return_value={
                    'generated_at': replay.evaluator.provenance.utc_iso(NOW)}), \
                 patch.object(replay.evaluator, 'evaluate', return_value=checked):
                self.assertIsNone(replay.build(run)['net_return_delta_pp'])

    def test_source_mismatch_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for name in ('receipt.json', 'candidate.json', 'manifest.json', 'repaired.jsonl'):
                (run/name).write_text('{}')
            (run/'repair_manifest.json').write_text(json.dumps({
                'sources': {'independent_ten_slot_replay.py': '0'*64}}))
            with patch.object(replay.repair, 'verify', return_value={}), self.assertRaises(ValueError):
                replay.build(run)

    def test_publish_rejects_changed_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            replay.publish(Path(tmp)/'run', {'evaluator_source_sha256': '0'*64})


if __name__ == '__main__':
    unittest.main()
