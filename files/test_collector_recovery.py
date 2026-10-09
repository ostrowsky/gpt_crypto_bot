import asyncio,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,AsyncMock
import rl_headless_worker as worker
from collector_recovery import retryable,RETRY_LIMIT,CollectorNetworkError


def nested_timeout():
    root=TimeoutError('timeout acquiring critic_dataset lock: example.jsonl.lock')
    middle=RuntimeError('wrapped');middle.__cause__=root
    outer=worker.critic_dataset.DatasetIntegrityError('wrapped twice');outer.__cause__=middle
    return outer


class RecoveryTests(unittest.TestCase):
    def test_nested_known_timeout_and_only_windows_sharing_retry(self):
        self.assertTrue(retryable(nested_timeout()))
        sharing=PermissionError('sharing');sharing.winerror=32;self.assertTrue(retryable(sharing))
        self.assertFalse(retryable(PermissionError('ACL denied')))
        self.assertFalse(retryable(TimeoutError('network timeout')))
        self.assertTrue(retryable(CollectorNetworkError('public API temporarily unavailable')))
        self.assertFalse(retryable(worker.critic_dataset.DatasetIntegrityError('malformed JSON')))
        cyclic=RuntimeError('cycle');cyclic.__cause__=cyclic;self.assertFalse(retryable(cyclic))

    def run_supervisor(self,outcomes,state):
        with tempfile.TemporaryDirectory() as td,patch.object(worker,'COLLECTOR_STOP_FILE',Path(td)/'incident.stop'),\
             patch.object(worker.data_collector,'_get_btc_context',new=AsyncMock(return_value={})),\
             patch.object(worker.data_collector,'_collect_once',new=AsyncMock(side_effect=outcomes)),\
             patch.object(worker.data_collector,'_seconds_until_next_bar',return_value=1),\
             patch.object(worker,'_write_status_now',new=AsyncMock()),patch.object(worker.asyncio,'sleep',new=AsyncMock()):
            try:asyncio.run(worker._collector_supervisor(state))
            except asyncio.CancelledError:pass
            return worker.COLLECTOR_STOP_FILE.exists()

    def test_known_transient_recovers_real_success_clears_marker_and_counter(self):
        s=worker.WorkerState(3600,300,120,20,True)
        marker=self.run_supervisor([nested_timeout(),{'ok':2,'fail':0,'total':2},asyncio.CancelledError()],s)
        self.assertFalse(marker);self.assertTrue(s.collector_enabled);self.assertFalse(s.collector_running)
        self.assertEqual(s.collector_recovery_state,'healthy');self.assertEqual(s.collector_recovery_attempt,0)
        self.assertEqual(s.collector_last_error,'');self.assertEqual(s.collector_last_cycle_stats['ok'],2)

    def test_exhausted_retry_stays_blocked_not_success(self):
        s=worker.WorkerState(3600,300,120,20,True)
        marker=self.run_supervisor([nested_timeout() for _ in range(RETRY_LIMIT+1)],s)
        self.assertTrue(marker);self.assertFalse(s.collector_enabled);self.assertFalse(s.collector_running)
        self.assertEqual(s.collector_recovery_state,'blocked');self.assertEqual(s.collector_recovery_attempt,RETRY_LIMIT)
        self.assertFalse(s.collector_last_cycle_stats)

    def test_malformed_failure_never_retries_or_deletes_incident(self):
        s=worker.WorkerState(3600,300,120,20,True)
        marker=self.run_supervisor([worker.critic_dataset.DatasetIntegrityError('malformed')],s)
        self.assertTrue(marker);self.assertEqual(s.collector_recovery_attempt,0);self.assertFalse(s.collector_enabled)

    def test_status_failure_cannot_swallow_integrity_incident(self):
        s=worker.WorkerState(3600,300,120,20,True)
        with tempfile.TemporaryDirectory() as td,patch.object(worker,'COLLECTOR_STOP_FILE',Path(td)/'incident'),\
             patch.object(worker.data_collector,'_get_btc_context',new=AsyncMock(return_value={})),\
             patch.object(worker.data_collector,'_collect_once',new=AsyncMock(side_effect=worker.critic_dataset.DatasetIntegrityError('malformed'))),\
             patch.object(worker,'_write_status_now',new=AsyncMock(side_effect=PermissionError('status unavailable'))):
            asyncio.run(worker._collector_supervisor(s))
            self.assertTrue(worker.COLLECTOR_STOP_FILE.exists());self.assertEqual(s.collector_recovery_state,'blocked')


if __name__=='__main__':unittest.main()
