import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import portfolio_evidence_intake as intake
from forward_evidence_service import atomic
from logical_learning_authority import provision
from validated_ranker_rollout import canonical, sha
import verify_policy_stream_parity as parity


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'evaluator'
        self.portfolio = self.root/'portfolios'
        self.portfolio.mkdir(parents=True)
        provision(self.root.parent/'authority')
        for name in ('champion','candidate'): atomic(self.portfolio/(name+'.json'),{'model':name})
        self.deployment = {'registry':str(self.root), 'portfolio_inputs':str(self.root/'portfolio_inputs.json'),
                           'isolation_mode':'logical_same_user', 'logical_isolation_accepted':True}

    def phase(self, name):
        atomic(self.portfolio/(name+'_unsigned.json'),{'phase':name})
        atomic(self.portfolio/(name+'_certification.json'),{'not':'a valid signature'})

    def test_missing_certificates_publish_blocked_not_ready(self):
        atomic(self.portfolio/'sealed_unsigned.json',{'phase':'sealed'})
        result = intake.tick(self.deployment,1000)
        self.assertEqual(result['state'],'BLOCKED')
        self.assertEqual(result['phases']['sealed']['state'],'UNKNOWN')
        self.assertFalse(result['runtime_eligible'])
        self.assertEqual(json.loads((self.root/'portfolio_inputs.json').read_bytes())['evidence'],{})

    def test_invalid_signature_never_enters_manifest(self):
        self.phase('sealed')
        result = intake.tick(self.deployment,1000)
        self.assertEqual(result['phases']['sealed']['state'],'UNKNOWN')
        self.assertNotIn('sealed',json.loads((self.root/'portfolio_inputs.json').read_bytes())['evidence'])

    def test_automatic_discovery_and_stage_selection_not_activation(self):
        def evaluate(raw,*args): return dict(json.loads(raw),passed=True)
        for name in ('historical','sealed','shadow'): self.phase(name)
        with patch.object(intake,'evaluate_bundle',side_effect=evaluate):
            result = intake.tick(self.deployment,1000)
        self.assertEqual(result['state'],'PHASES_READY')
        self.assertEqual(result['stage'],'CANARY')
        self.assertFalse(result['runtime_eligible'])
        self.phase('canary')
        with patch.object(intake,'evaluate_bundle',side_effect=evaluate):
            result = intake.tick(self.deployment,1000)
        self.assertEqual(result['stage'],'PROMOTED')

    def test_failed_numeric_phase_retained_as_rejected(self):
        self.phase('historical')
        with patch.object(intake,'evaluate_bundle',return_value={'phase':'historical','passed':False,'portfolio_delta_pp':-2}):
            result = intake.tick(self.deployment,1000)
        self.assertEqual(result['state'],'REJECTED')
        self.assertEqual(result['phases']['historical']['portfolio_delta_pp'],-2)

    def test_model_drift_clears_old_ready_manifest(self):
        intake.tick(self.deployment,1000)
        atomic(self.portfolio/'candidate.json',{'changed':True})
        result = intake.tick(self.deployment,1000)
        self.assertEqual(result['state'],'BLOCKED')
        self.assertIn('hash mismatch',result['blockers'][0])
        self.assertEqual(json.loads((self.root/'portfolio_inputs.json').read_bytes())['evidence'],{})

    def test_failed_canary_cannot_downgrade_to_canary_authorization(self):
        for name in ('historical','sealed','shadow','canary'): self.phase(name)
        def evaluate(raw,*args):
            phase=json.loads(raw)['phase']
            return {'phase':phase,'passed':phase!='canary'}
        with patch.object(intake,'evaluate_bundle',side_effect=evaluate):
            result=intake.tick(self.deployment,1000)
        self.assertEqual(result['stage'],'PROMOTED')
        self.assertEqual(result['state'],'REJECTED')

    def test_registration_traversal_is_rejected(self):
        registration = intake.discover(self.root)
        registration['candidate'] = {'path':'../outside.json','sha256':'bad'}
        atomic(self.root/'portfolio_intake_registration.json',registration)
        self.assertEqual(intake.tick(self.deployment,1000)['state'],'BLOCKED')

    def test_mutation_during_evaluation_clears_verified_phase_inputs(self):
        for name in ('historical','sealed','shadow'): self.phase(name)
        def evaluate(raw,*args):
            value=json.loads(raw)
            if value['phase']=='shadow':
                registration=self.root/'portfolio_intake_registration.json'
                atomic(registration, {'changed':True})
            return dict(value,passed=True)
        with patch.object(intake,'evaluate_bundle',side_effect=evaluate):
            result=intake.tick(self.deployment,1000)
        self.assertEqual(result['state'],'BLOCKED')
        self.assertEqual(json.loads((self.root/'portfolio_inputs.json').read_bytes())['evidence'],{})

    def test_execution_parity_checks_restart_state_not_just_trade_count(self):
        async def execute(*args,**kw):
            state = kw['stream_state']
            frames = sorted(kw['candidate_snapshot'][1])
            state.update(open_positions=[],last_closed_by_symbol={},last_ts=frames[-1])
            state['count'] = state.get('count',0)+len(frames)
            return [],None
        with patch.object(parity.rb,'simulate_portfolio',side_effect=execute):
            value = asyncio.run(parity.compare([],{},None,({}, {1,2,3},0)))
        self.assertEqual(value['state'],'PASS')
        self.assertIn('not_candidate_generation',value['scope'])

    def test_different_incremental_state_fails(self):
        async def execute(*args,**kw):
            state=kw['stream_state']
            state.update(open_positions=[],last_closed_by_symbol={},calls=state.get('calls',0)+1)
            return [],None
        with patch.object(parity.rb,'simulate_portfolio',side_effect=execute):
            self.assertEqual(asyncio.run(parity.compare([],{},None,({}, {1,2},0)))['state'],'FAIL')


if __name__=='__main__': unittest.main()
