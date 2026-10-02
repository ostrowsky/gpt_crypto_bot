import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import independent_portfolio_gate as gate
import validated_ranker_rollout as release

KEY = b'e'*32
AUTHORITY = b'a'*32
CANDIDATE = b'{"version":2}'
CHAMPION = b'{"version":1}'
NOW = 1800000000


def ticket(stage='CANARY'):
    return release.seal({'contract': release.CONTRACT, 'stage': stage,
        'issued_at': NOW-1, 'expires_at': NOW+3600, 'candidate_sha256': release.sha(CANDIDATE),
        'champion_sha256': release.sha(CHAMPION), 'evaluator_sha256': gate.source_hash(),
        'max_bonus': 1.0, 'fraction': .05 if stage == 'CANARY' else 1.0}, KEY)


def bundle(phase='sealed', offset=0):
    start = NOW*1000-110*gate.DAY+offset*gate.DAY
    end = start+30*gate.DAY
    times = list(range(start, end+1, gate.STEP))
    prices = [[t, 101 if (t-start)//gate.STEP % 3 == 1 else 100] for t in times]
    left, right = [], []
    for i in range(100):
        entry = start+((i*(len(times)-3)//100)//3)*3*gate.STEP
        left.append({'sym': 'A', 'entry_ts': entry, 'exit_ts': entry,
                     'entry_price': 100, 'exit_price': 100})
        right.append({'sym': 'A', 'entry_ts': entry, 'exit_ts': entry+gate.STEP,
                      'entry_price': 100, 'exit_price': 101})
    return {'phase': phase, 'start_ms': start, 'end_ms': end,
            'maximum_available_bounds': [start, end], 'last_model_exposure_ms': start-3*gate.DAY,
            'universe': ['BTCUSDT', 'A'], 'prices': {'BTCUSDT': [[t, 100] for t in times], 'A': prices},
            'fee_bps': 7.5, 'slippage_bps': 5, 'champion_trades': left, 'candidate_trades': right}


def evidence(value):
    raw = release.canonical(value)
    cert = {'contract': 'independent-market-policy-certification-v1',
            'bundle_sha256': release.sha(raw), 'candidate_sha256': release.sha(CANDIDATE),
            'champion_sha256': release.sha(CHAMPION), 'evaluator_sha256': gate.source_hash(),
            'issued_at': NOW-1, 'expires_at': NOW+3600,
            'point_in_time_universe': True, 'raw_closed_provenance': True,
            'live_policy_parity': True, 'no_trainer_holdout_access': True,
            'operator_experiment_approved': True, 'actual_assignment_verified': True}
    return raw, release.seal(cert, AUTHORITY)


class RolloutTests(unittest.TestCase):
    def test_orphan_pointer_lock_keeps_compare_and_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'pointer.lock').touch()
            release.atomic_pointer(root, {'state':'initial'}, None)
            raw = (root/'active.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'concurrently'):
                release.atomic_pointer(root, {'state':'wrong'}, None)
            self.assertEqual((root/'active.json').read_bytes(),raw)
            release.rollback(root, release.sha(raw))
            self.assertEqual(json.loads((root/'active.json').read_bytes())['state'],'ROLLED_BACK')

    def test_runtime_bonus_disabled_and_clipped(self):
        import monitor
        info = {'payload_version': 2, 'final_score': 0, 'validated_overlay_bonus': 100}
        with patch.object(monitor.config, 'ML_CANDIDATE_RANKER_RUNTIME_ENABLED', True, create=True), \
             patch.object(monitor.config, 'ML_CANDIDATE_RANKER_SCORE_WEIGHT', 1, create=True), \
             patch.object(monitor.config, 'ML_CANDIDATE_RANKER_USE_FINAL_SCORE', True, create=True):
            with patch.object(monitor.config, 'VALIDATED_RANKER_ROLLOUT_ENABLED', False):
                self.assertEqual(monitor._ml_candidate_ranker_runtime_bonus(info), 0)
            with patch.object(monitor.config, 'VALIDATED_RANKER_ROLLOUT_ENABLED', True):
                self.assertEqual(monitor._ml_candidate_ranker_runtime_bonus(info), 1)

    def test_invalid_candidate_preserves_champion_components(self):
        import inspect
        import monitor
        import ml_candidate_ranker
        import numpy as np
        kwargs = {k:None for k in inspect.signature(monitor._ml_candidate_ranker_components).parameters}
        kwargs.update(sym='A', tf='15m', feat={}, data=np.array([(0,)], dtype=[('t','i8')]), i=0)
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)/'champion.json'
            model.write_bytes(CHAMPION)
            with patch.object(monitor.config, 'ML_CANDIDATE_RANKER_RUNTIME_ENABLED', True, create=True), \
                 patch.object(monitor.config, 'VALIDATED_RANKER_ROLLOUT_ENABLED', True), \
                 patch.object(monitor, '_load_ranker_payload', return_value=json.loads(CHAMPION)), \
                 patch.object(monitor, '_RANKER_MODEL_FILE', model), \
                 patch.object(ml_candidate_ranker, 'build_runtime_candidate_record', return_value={}), \
                 patch.object(ml_candidate_ranker, 'predict_components_from_candidate_payload',
                              side_effect=[{'final_score': .5}, ValueError('invalid candidate')]), \
                 patch.object(release, 'select', return_value=({}, ticket()['body'])):
                self.assertEqual(monitor._ml_candidate_ranker_components(**kwargs)['final_score'], .5)

    def test_bad_signature(self):
        t = ticket()
        t['body']['stage'] = 'PROMOTED'
        with self.assertRaises(ValueError):
            release.validate_ticket(t, KEY, CHAMPION, NOW)

    def test_ticket_rejects_expiry_source_champion_and_bonus(self):
        for key, value in [('expires_at', NOW), ('issued_at', NOW+1),
                           ('evaluator_sha256', 'changed'), ('max_bonus', 2), ('fraction', 1)]:
            t = ticket()['body']
            t[key] = value
            with self.assertRaises(ValueError):
                release.validate_ticket(release.seal(t, KEY), KEY, CHAMPION, NOW)
        with self.assertRaises(ValueError):
            release.validate_ticket(ticket(), KEY, b'changed', NOW)

    def test_atomic_activation_disabled_fallback_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release.rollout(root, CANDIDATE, ticket('PROMOTED'), KEY, CHAMPION, now=NOW)
            self.assertIsNone(release.select(root, CHAMPION, 'A', KEY, now=NOW))
            self.assertEqual(release.select(root, CHAMPION, 'A', KEY, True, NOW)[0], json.loads(CANDIDATE))
            with self.assertRaises(ValueError):
                release.atomic_pointer(root, {}, None)
            release.rollback(root, release.sha((root/'active.json').read_bytes()))
            self.assertIsNone(release.select(root, CHAMPION, 'A', KEY, True, NOW))

    def test_model_tamper_and_champion_change_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release.rollout(root, CANDIDATE, ticket('PROMOTED'), KEY, CHAMPION, now=NOW)
            self.assertIsNone(release.select(root, b'other', 'A', KEY, True, NOW))
            (root/(release.sha(CANDIDATE)+'.json')).write_bytes(b'{}')
            self.assertIsNone(release.select(root, CHAMPION, 'A', KEY, True, NOW))

    def test_deterministic_canary_and_finite_bound(self):
        body = ticket()['body']
        assigned = [release.assigned(str(i), body) for i in range(10000)]
        self.assertTrue(350 < sum(assigned) < 650)
        self.assertEqual(assigned, [release.assigned(str(i), body) for i in range(10000)])
        self.assertEqual(release.bounded_bonus(100), 1)
        self.assertEqual(release.bounded_bonus(-100), -1)
        self.assertEqual(release.bounded_bonus(float('nan')), 0)

    def test_independent_account_recomputation(self):
        raw, cert = evidence(bundle())
        r = gate.evaluate_bundle(raw, cert, AUTHORITY, release.sha(CANDIDATE), release.sha(CHAMPION), NOW)
        self.assertTrue(r['passed'])
        self.assertGreater(r['portfolio_delta_pp'], 0)

    def test_unsigned_or_changed_bundle(self):
        raw, cert = evidence(bundle())
        for changed, certificate in [(raw+b' ', cert), (raw, release.seal(cert['body'], KEY))]:
            with self.assertRaises(ValueError):
                gate.evaluate_bundle(changed, certificate, AUTHORITY, release.sha(CANDIDATE), release.sha(CHAMPION), NOW)

    def test_gaps_duplicates_nonfinite_partial_and_costs(self):
        for case in ('gap', 'duplicate', 'nonfinite', 'cost', 'partial', 'exposure', 'shortened'):
            b = bundle('historical' if case == 'shortened' else 'sealed')
            if case == 'gap': b['prices']['A'].pop()
            if case == 'duplicate': b['prices']['A'][1][0] = b['prices']['A'][0][0]
            if case == 'nonfinite': b['prices']['A'][0][1] = float('inf')
            if case == 'cost': b['slippage_bps'] = 0
            if case == 'partial': b['candidate_trades'][0]['partial_exit_taken'] = True
            if case == 'exposure': b['last_model_exposure_ms'] = b['start_ms']
            if case == 'shortened': b['maximum_available_bounds'][0] -= gate.DAY
            with self.assertRaises(ValueError, msg=case):
                raw, cert = evidence(b)
                gate.evaluate_bundle(raw, cert, AUTHORITY, release.sha(CANDIDATE), release.sha(CHAMPION), NOW)

    def test_harness_failure_and_missing_forward_block(self):
        ev = {p:evidence(bundle(p, n*40)) for n,p in enumerate(('historical', 'sealed', 'shadow'))}
        with patch.object(gate, 'harness_passes', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Harness'):
                gate.authorize(ev, AUTHORITY, KEY, CANDIDATE, CHAMPION, now=NOW)
        with self.assertRaises(ValueError):
            gate.authorize({}, AUTHORITY, KEY, CANDIDATE, CHAMPION, now=NOW)

    def test_authorization_needs_distinct_cohorts(self):
        ev = {p:evidence(bundle(p, n*40)) for n,p in enumerate(('historical', 'sealed', 'shadow'))}
        with patch.object(gate, 'harness_passes', return_value=True):
            auth = gate.authorize(ev, AUTHORITY, KEY, CANDIDATE, CHAMPION, now=NOW)
            self.assertEqual(release.validate_ticket(auth, KEY, CHAMPION, NOW)['stage'], 'CANARY')
            ev['shadow'] = evidence(bundle('shadow', 40))
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                gate.authorize(ev, AUTHORITY, KEY, CANDIDATE, CHAMPION, now=NOW)

    def test_stale_forward_and_missing_canary_block_promotion(self):
        ev = {p:evidence(bundle(p, n*35)) for n,p in enumerate(('historical', 'sealed', 'shadow'))}
        with patch.object(gate, 'harness_passes', return_value=True):
            with self.assertRaisesRegex(ValueError, 'stale'):
                gate.authorize(ev, AUTHORITY, KEY, CANDIDATE, CHAMPION, now=NOW)
            with self.assertRaises(ValueError):
                gate.authorize(ev, AUTHORITY, KEY, CANDIDATE, CHAMPION, stage='PROMOTED', now=NOW)

    def test_full_synthetic_evidence_promotes_then_rolls_back(self):
        phases = ('historical', 'sealed', 'shadow', 'canary')
        ev = {p:evidence(bundle(p, -40+n*40)) for n,p in enumerate(phases)}
        with patch.object(gate, 'harness_passes', return_value=True), tempfile.TemporaryDirectory() as tmp:
            auth = gate.authorize(ev, AUTHORITY, KEY, CANDIDATE, CHAMPION, stage='PROMOTED', now=NOW)
            root = Path(tmp)
            release.rollout(root, CANDIDATE, auth, KEY, CHAMPION, now=NOW)
            self.assertEqual(release.select(root, CHAMPION, 'A', KEY, True, NOW)[1]['stage'], 'PROMOTED')
            release.rollback(root, release.sha((root/'active.json').read_bytes()))
            self.assertIsNone(release.select(root, CHAMPION, 'A', KEY, True, NOW))

    def test_losses_are_rejected_not_approved(self):
        b = bundle()
        b['candidate_trades'] = b['champion_trades']
        raw, cert = evidence(b)
        r = gate.evaluate_bundle(raw, cert, AUTHORITY, release.sha(CANDIDATE), release.sha(CHAMPION), NOW)
        self.assertFalse(r['passed'])
        self.assertEqual(r['portfolio_delta_pp'], 0)


if __name__ == '__main__':
    unittest.main()
