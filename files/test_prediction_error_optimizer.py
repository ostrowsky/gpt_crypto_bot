import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import json
import numpy as np

import prediction_error_optimizer as optimizer
import forward_evidence_service as service
import training_snapshot
import local_learning_runtime as local


class Model:
    def __init__(self, value): self.value=value
    def fit(self, X, y, X_eval, y_eval):
        assert len(X_eval) == len(y_eval) == 0
        self.seen=X.copy()
        return self
    def predict_proba(self, X): return np.full(len(X), self.value)


class PredictionOptimizationTests(unittest.TestCase):
    def test_validation_selection_bounded_no_test_input_or_early_stop(self):
        made=[]
        def factory(params):
            made.append(params)
            return Model([.9,.1,.5][len(made)-1])
        selected, reference, report=optimizer.search(
            factory,np.zeros((5,1)),np.zeros(5),np.ones((3,1)),np.zeros(3))
        self.assertEqual(len(made),3)
        self.assertEqual(report['selected_index'],1)
        self.assertEqual(selected.value,.1)
        self.assertEqual(reference.value,.9)
        self.assertFalse(report['runtime_eligible'])
        np.testing.assert_array_equal(selected.seen,np.zeros((5,1)))

    def test_tie_retains_reference(self):
        _,_,report=optimizer.search(lambda p:Model(.5),np.zeros((4,1)),
                                   np.zeros(4),np.zeros((2,1)),np.zeros(2))
        self.assertEqual(report['selected_index'],0)

    def test_independent_error_does_not_claim_profit_or_select(self):
        days=[str(i) for i in range(12)]
        report=optimizer.holdout(np.zeros(12),np.full(12,.1),np.full(12,.9),days,.5)
        self.assertEqual(report['state'],'SUPPORTED')
        self.assertGreater(report['daily_95ci'][0],0)
        self.assertFalse(report['runtime_eligible'])
        self.assertNotIn('selected_index',report)
        self.assertEqual(optimizer.holdout(np.zeros(12),np.full(12,.9),
                         np.full(12,.1),days,.5)['state'],'NOT_PROVEN')
        self.assertEqual(optimizer.holdout([0],[.1],[.9],['day'],.5)['state'],'UNKNOWN')

    def test_invalid_evidence_and_same_prediction_are_not_improvement(self):
        for y,p in (([],[]),([0],[float('nan')]),([2],[.5]),([0],[-.1])):
            with self.assertRaises(ValueError): optimizer.errors(y,p)
        report=optimizer.holdout(np.zeros(12),np.full(12,.1),np.full(12,.1),
                                 [str(i) for i in range(12)],.5)
        self.assertEqual(report['state'],'NOT_PROVEN')
        with self.assertRaises(ValueError): optimizer.holdout([0],[.1],[.2],[],.5)

    def test_scheduled_trainer_wiring_snapshot_dedup_and_no_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dataset=root/'training.jsonl'
            dataset.write_bytes(b'[]')
            snapshot={'sha256':training_snapshot.digest(dataset),'rows':500}
            deployment={'isolation_mode':'logical_same_user','logical_isolation_accepted':True,
                'trainer_sid':'user','evaluator_sid':'user','registry':str(root/'registry'),
                'training_input':str(dataset),'candidate_output':str(root/'candidate.json'),
                'trainer_status':str(root/'status.json')}
            report={'model_payload':{'runtime_eligible':False},
                    'prediction_error_optimization':{'holdout':{'state':'NOT_PROVEN'}}}
            with patch.object(service,'current_sid',return_value='user'), \
                 patch.object(training_snapshot,'resolve',return_value=(dataset,snapshot)), \
                 patch('ml_candidate_ranker.train_and_evaluate',return_value=report) as fit:
                result=service.run_tick(deployment,'trainer')
                self.assertEqual(result['state'],'CANDIDATE_ONLY')
                self.assertFalse(result['runtime_eligible'])
                self.assertEqual(fit.call_args.kwargs,{'optimize_prediction_error':True})
                second=service.run_tick(deployment,'trainer')
                self.assertTrue(second['unchanged_snapshot'])
                self.assertEqual(fit.call_count,1)
                self.assertEqual(json.loads((root/'candidate.json').read_bytes()),
                                 {'runtime_eligible':False})
                (root/'candidate.json').write_bytes(b'{}')
                service.run_tick(deployment,'trainer')
                self.assertEqual(fit.call_count,2)

    def test_snapshot_drift_blocks_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); dataset=root/'training'; dataset.write_bytes(b'x')
            deployment={'isolation_mode':'logical_same_user','logical_isolation_accepted':True,
                'trainer_sid':'u','evaluator_sid':'u','registry':str(root),
                'training_input':str(dataset),'candidate_output':str(root/'candidate.json'),
                'trainer_status':str(root/'status.json')}
            with patch.object(service,'current_sid',return_value='u'), \
                 patch.object(training_snapshot,'resolve',return_value=(dataset,{'sha256':'bad'})), \
                 patch('ml_candidate_ranker.train_and_evaluate',return_value={}):
                result=service.run_tick(deployment,'trainer')
                self.assertEqual(result['state'],'BLOCKED')
                self.assertFalse((root/'candidate.json').exists())

    def test_serial_scheduler_order_cadence_and_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'deployment.json'
            path.write_text('{}')
            due={'portfolio':float('inf'),'controller':float('inf')}
            roles=[]
            def run(deployment,role):
                roles.append(role)
                return {'state':'CANDIDATE_ONLY' if role=='trainer' else 'BLOCKED'}
            with patch.object(local,'run_tick',side_effect=run):
                local.serial_cycle(path,due)
                self.assertEqual(roles,['exporter','trainer','evaluator'])
                local.serial_cycle(path,due)
                self.assertEqual(len(roles),3)
                (path.parent/'stop.request').touch()
                self.assertEqual(local.serial_cycle(path,{}),{})

    def test_optimizer_fingerprint_changes_with_source(self):
        with patch.object(Path,'read_bytes',return_value=b'one'):
            first=optimizer.fingerprint()
        with patch.object(Path,'read_bytes',return_value=b'two'):
            self.assertNotEqual(first,optimizer.fingerprint())


if __name__ == '__main__': unittest.main()
