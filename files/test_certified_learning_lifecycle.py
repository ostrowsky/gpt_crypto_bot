import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import learning_certificate_issuer as issuer
import learning_cohort_controller as cohorts
import policy_runtime_receipts as receipts
import forward_evidence_service as service
import independent_portfolio_gate as gate
import validated_ranker_rollout as release
from logical_learning_authority import LogicalAuthority
from test_validated_ranker_rollout import KEY, AUTHORITY, CANDIDATE, CHAMPION, NOW, ticket, bundle


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.portfolios=self.root/'evaluator'/'portfolios'
        self.portfolios.mkdir(parents=True)
        self.deployment={'registry':str(self.root/'evaluator'),'release_root':str(self.root/'release')}
        self.authority=LogicalAuthority(AUTHORITY)
        for name,raw in [('candidate',CANDIDATE),('champion',CHAMPION)]:
            (self.portfolios/(name+'.json')).write_bytes(raw)
        self.raw=release.canonical(bundle())
        self.proof=dict(contract=issuer.PROOF,phase='sealed',bundle_sha256=release.sha(self.raw),
            candidate_sha256=release.sha(CANDIDATE),champion_sha256=release.sha(CHAMPION),
            evaluator_sha256=gate.source_hash(),scope='full_candidate_population_and_live_policy',
            issued_at=NOW-1,expires_at=NOW+3600,bounds=[bundle()['start_ms'],bundle()['end_ms']],
            **{name:True for name in issuer.CHECKS})

    def issue(self,proof=None,raw=None,key=KEY):
        return issuer.issue(raw or self.raw,release.seal(proof or self.proof,key),
            CANDIDATE,CHAMPION,self.authority,KEY,NOW)

    def test_issuer_independent_recomputation(self):
        cert,numeric=self.issue()
        self.assertTrue(numeric['passed'])
        body=release.unseal(cert,AUTHORITY)
        self.assertFalse(body['os_access_isolation'])
        self.assertEqual(body['bundle_sha256'],release.sha(self.raw))

    def test_execution_only_scope_is_not_upgraded(self):
        proof=dict(self.proof,scope='execution_state_only')
        with self.assertRaises(ValueError): self.issue(proof)

    def test_rejects_signature_source_bounds_expiry_and_false_checks(self):
        with self.assertRaises(ValueError): self.issue(key=b'wrong')
        bad=[dict(self.proof,expires_at=NOW),dict(self.proof,issued_at=NOW+1),
             dict(self.proof,evaluator_sha256='0'*64),dict(self.proof,bounds=[1,2]),
             dict(self.proof,bundle_sha256='0'*64)]
        bad.extend(dict(self.proof,**{name:False}) for name in issuer.CHECKS)
        for proof in bad:
            with self.subTest(proof=proof),self.assertRaises(ValueError): self.issue(proof)

    def test_canary_requires_actual_assignment(self):
        value=bundle('canary')
        raw=release.canonical(value)
        proof=dict(self.proof,phase='canary',bundle_sha256=release.sha(raw))
        with self.assertRaisesRegex(ValueError,'assignment'): self.issue(proof,raw)

    def test_numerical_failure_can_be_certified_not_approved(self):
        value=bundle()
        value['candidate_trades']=copy.deepcopy(value['champion_trades'])
        raw=release.canonical(value)
        cert,numeric=self.issue(dict(self.proof,bundle_sha256=release.sha(raw)),raw)
        self.assertFalse(numeric['passed'])
        self.assertTrue(release.unseal(cert,AUTHORITY)['raw_closed_provenance'])

    def test_invalid_proof_invalidates_existing_certificate(self):
        (self.portfolios/'sealed_unsigned.json').write_bytes(self.raw)
        service.atomic(self.portfolios/'sealed_certification.json',{'previous':'valid'})
        with patch.object(issuer,'material',return_value=(self.authority,KEY)):
            result=issuer.tick(self.deployment,NOW)
        self.assertEqual(result['phases']['sealed']['state'],'BLOCKED')
        self.assertEqual(json.loads((self.portfolios/'sealed_certification.json').read_bytes())['state'],'BLOCKED')

    def prepare_cohort(self,phase='sealed'):
        start=bundle()['start_ms']
        service.atomic(self.portfolios/'registration.json',{'observation_start_ms':start})
        service.atomic(self.portfolios/'cohort.json',{'contract':cohorts.CONTRACT,
            'phase':phase,'start_ms':start,'state':'COLLECTING'})
        for name in ('historical',phase):
            (self.portfolios/(name+'_unsigned.json')).write_bytes(release.canonical({'phase':name}))
            service.atomic(self.portfolios/(name+'_certification.json'),{})
        return start

    def cohort_tick(self,verifier):
        with patch.object(cohorts,'material',return_value=(self.authority,KEY)), \
             patch.object(cohorts,'evaluate_bundle',side_effect=verifier):
            return cohorts.tick(self.deployment,NOW)

    def test_disjoint_transition_and_restart(self):
        start=self.prepare_cohort()
        def verify(raw,*args):
            return {'phase':json.loads(raw)['phase'],'passed':True,'start_ms':start,'end_ms':start+30*gate.DAY}
        result=self.cohort_tick(verify)
        self.assertEqual(result['phase'],'shadow')
        self.assertGreater(result['start_ms'],start+30*gate.DAY)
        pointer=(self.portfolios/'cohort.json').read_bytes()
        self.assertEqual(len(list((self.portfolios/'cohorts').glob('*.json'))),1)
        self.assertEqual(self.cohort_tick(verify)['state'],'BLOCKED')
        self.assertEqual((self.portfolios/'cohort.json').read_bytes(),pointer)

    def test_failed_cohort_is_durable_rejection(self):
        start=self.prepare_cohort('canary')
        def verify(raw,*args):
            return {'phase':json.loads(raw)['phase'],'passed':False,'start_ms':start,'end_ms':start+30*gate.DAY}
        self.assertEqual(self.cohort_tick(verify)['state'],'REJECTED')
        with patch.object(cohorts,'evaluate_bundle',side_effect=AssertionError('must not re-evaluate')):
            with patch.object(cohorts,'material',return_value=(self.authority,KEY)):
                self.assertEqual(cohorts.tick(self.deployment,NOW)['state'],'REJECTED')
        self.assertEqual(json.loads((self.root/'evaluator'/'cohort_controller_latest.json').read_bytes())['state'],'REJECTED')

    def test_shadow_cannot_enter_canary_without_release(self):
        start=self.prepare_cohort('shadow')
        def verify(raw,*args):
            return {'phase':json.loads(raw)['phase'],'passed':True,'start_ms':start,'end_ms':start+30*gate.DAY}
        before=(self.portfolios/'cohort.json').read_bytes()
        self.assertEqual(self.cohort_tick(verify)['state'],'BLOCKED')
        self.assertEqual((self.portfolios/'cohort.json').read_bytes(),before)
        service.atomic(self.root/'release'/'active.json',ticket())
        self.assertEqual(self.cohort_tick(verify)['phase'],'canary')

    def test_start_mutation_rejected(self):
        start=self.prepare_cohort()
        def verify(raw,*args):
            return {'phase':json.loads(raw)['phase'],'passed':True,'start_ms':start+gate.STEP,'end_ms':NOW*1000}
        self.assertEqual(self.cohort_tick(verify)['state'],'BLOCKED')

    def prepare_admission(self):
        root=self.root/'release'
        authorization=ticket('PROMOTED')
        score={'sym':'AAAUSDT','tf':'15m','bar_ts':NOW*1000,'bonus':.5,
            'candidate_sha256':release.sha(CANDIDATE),'ticket_sha256':release.sha(release.canonical(authorization['body']))}
        service.atomic(receipts.context_path(root,'AAAUSDT','15m',NOW*1000),
            {'score':score,'authorization':authorization})
        position={'tf':'15m','entry_ts':NOW*1000,'entry_price':100}
        positions=self.root/'positions.json'
        service.atomic(positions,{'AAAUSDT':position})
        service.atomic(root/'active.json',authorization)
        return root,positions

    def test_actual_persisted_admission_not_exchange_fill(self):
        root,positions=self.prepare_admission()
        body=receipts.admission(root,KEY,CHAMPION,positions,'AAAUSDT','15m',NOW*1000,NOW)
        self.assertEqual(receipts.verify_admission(release.seal(body,KEY),KEY,CHAMPION),body)
        self.assertFalse(body['closed_loop'])
        self.assertEqual(receipts.summarize(root,KEY,CHAMPION,NOW)['application'],'CURRENT_PAPER_ADMISSION_VERIFIED')
        self.assertEqual(receipts.summarize(root,KEY,CHAMPION,NOW+601)['application'],'UNKNOWN')

    def test_score_alone_or_wrong_position_not_application(self):
        root,positions=self.prepare_admission()
        self.assertEqual(receipts.summarize(root,KEY,CHAMPION,NOW)['application'],'UNKNOWN')
        service.atomic(positions,{'AAAUSDT':{'tf':'15m','entry_ts':1,'entry_price':100}})
        with self.assertRaises(ValueError): receipts.admission(root,KEY,CHAMPION,positions,'AAAUSDT','15m',NOW*1000,NOW)

    def test_admission_rejects_forged_and_inconsistent_score(self):
        root,positions=self.prepare_admission()
        body=receipts.admission(root,KEY,CHAMPION,positions,'AAAUSDT','15m',NOW*1000,NOW)
        bad=release.seal(body,KEY)
        bad['body']['bar_ts']=1
        with self.assertRaises(ValueError): receipts.verify_admission(bad,KEY,CHAMPION)
        body['score']['candidate_sha256']='0'*64
        with self.assertRaises(ValueError): receipts.verify_admission(release.seal(body,KEY),KEY,CHAMPION)

    def prepare_rollback(self):
        root=self.root/'release'
        service.atomic(root/'active.json',{'state':'ROLLED_BACK'})
        service.atomic(root/'rollback_request.json',{'requested_at':NOW,
            'previous_authorization':ticket('PROMOTED'),
            'rollback_pointer_sha256':release.sha((root/'active.json').read_bytes())})
        return root

    def test_rollback_requires_later_runtime_observation(self):
        root=self.prepare_rollback()
        self.assertEqual(receipts.verify_rollback(root,KEY,NOW)['state'],'POINTER_ONLY_WAITING_RUNTIME')
        with self.assertRaises(ValueError): receipts.fallback(root,KEY,'AAAUSDT','15m',NOW*1000,NOW-1)
        receipts.fallback(root,KEY,'AAAUSDT','15m',NOW*1000,NOW+1)
        self.assertEqual(receipts.verify_rollback(root,KEY,NOW)['state'],'POINTER_ONLY_WAITING_RUNTIME')
        self.assertEqual(receipts.verify_rollback(root,KEY,NOW+1)['state'],'RUNTIME_FALLBACK_VERIFIED')
        service.atomic(root/'active.json',{'state':'OTHER'})
        with self.assertRaises(ValueError): receipts.verify_rollback(root,KEY,NOW+1)

    def test_repeated_controller_failure_preserves_rollback_request(self):
        root=self.prepare_rollback()
        before=(root/'rollback_request.json').read_bytes()
        result=service.controller_tick(self.root/'missing.json',root,evaluator_key=KEY,now=NOW+1)
        self.assertEqual(result['rollback'],'applied')
        self.assertEqual((root/'rollback_request.json').read_bytes(),before)


if __name__=='__main__': unittest.main()
