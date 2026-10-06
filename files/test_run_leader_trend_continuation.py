import unittest
from types import SimpleNamespace
from run_leader_trend_continuation import skip_record


class RunnerTests(unittest.TestCase):
    def test_skip_records_asof_origin_without_future_return(self):
        candidate=SimpleNamespace(sym='A',tf='15m',mode='trend',ts_ms=9000000,price=10.,top_gainer_score=40.)
        t=SimpleNamespace(entry_ts=100,exit_ts=8500000,exit_reason='WEAK: x',mode='trend',tf='15m')
        r=skip_record(t,candidate)
        self.assertEqual(r['at'],9000000);self.assertGreater(r['cooldown_until'],r['at']);self.assertNotIn('forward_return',r)
        self.assertIsNone(skip_record(None,candidate)['previous_exit'])


if __name__=='__main__':unittest.main()
