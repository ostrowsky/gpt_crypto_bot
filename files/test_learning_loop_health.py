from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

import learning_loop_health as health


class LoopHealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.now = datetime(2026, 10, 2, tzinfo=timezone.utc)

    def write(self, name, value):
        (self.root/name).write_text(json.dumps(value))

    def test_missing_status_is_not_zero_or_learning_success(self):
        value = health.summarize(self.root, self.now)
        self.assertEqual(value['state'], 'NOT_CLOSED')
        self.assertEqual(value['improvement_verdict'], 'UNKNOWN')
        self.assertIsNone(value['outcomes'])

    def test_proxy_success_and_export_are_not_closed_loop(self):
        self.write('status.json', {'run_time': self.now.isoformat(),
            'collection': {'observations': 10, 'outcomes': 8}})
        self.write('training_export_latest.json', {'run_time': self.now.isoformat(),
            'state': 'EXPORTED_NOT_APPROVED'})
        value = health.summarize(self.root, self.now)
        self.assertEqual(value['collection_state'], 'PROXY_ONLY')
        self.assertEqual(value['observations'], 10)
        self.assertFalse(value['closed_loop'])

    def test_authorized_controller_is_not_verified_application(self):
        self.write('status.json', {'run_time': self.now.isoformat(),
            'portfolio_confirmation': {'state':'READY'},
            'controller': {'state':'CANARY','runtime_eligible':True}, 'closed_loop':True})
        value = health.summarize(self.root, self.now)
        self.assertTrue(value['activation_authorized'])
        self.assertEqual(value['production_effect'], 'UNKNOWN')
        self.assertFalse(value['closed_loop'])

    def test_future_stale_or_corrupt_status_never_authorizes(self):
        for stamp in (self.now+timedelta(seconds=1), self.now-timedelta(minutes=5)):
            self.write('status.json', {'run_time':stamp.isoformat(),
                'controller':{'state':'PROMOTED','runtime_eligible':True},
                'portfolio_confirmation':{'state':'READY'}, 'collection':{'outcomes':999}})
            value = health.summarize(self.root, self.now)
            self.assertFalse(value['activation_authorized'])
            self.assertIsNone(value['outcomes'])
        (self.root/'status.json').write_text('broken')
        self.assertEqual(health.summarize(self.root, self.now)['collection_state'],'UNKNOWN')

    def test_collection_failure_does_not_reuse_positive_counts(self):
        self.write('status.json', {'run_time':self.now.isoformat(),
            'collection':{'observations':100}, 'collection_blocker':'corrupt journal'})
        self.assertIsNone(health.summarize(self.root,self.now)['observations'])

    def test_report_renders_actual_loop_status(self):
        import learning_progress_report as report
        text = report.render_text({'automatic_learning_loop':health.summarize(self.root,self.now)})
        self.assertIn('NOT_CLOSED',text)
        self.assertIn('развитие/деградация: UNKNOWN',text)
        self.assertIn('применение в live=UNKNOWN',text)


if __name__ == '__main__':
    unittest.main()
