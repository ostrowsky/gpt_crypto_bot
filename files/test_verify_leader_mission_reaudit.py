import unittest
import numpy as np
from test_leader_mission_metrics import episode_fixture
from leader_mission_metrics import episode
from verify_leader_mission_reaudit import scalar_episode,assert_values


class VerificationTests(unittest.TestCase):
    def test_independent_scalar_matches_all_marker_fields(self):
        r,d,f=episode_fixture();raw=[dict(t=int(t-900000),o=float(c),h=float(c+.5),l=float(c-.5),c=float(c)) for t,c in zip(d['time'],d['close'])]
        expected=scalar_episode(r,raw,f['ema'],f['atr'])
        assert_values(expected,episode(r,d,f))
        corrupt=dict(episode(r,d,f),marker_ts=1)
        with self.assertRaises(ValueError):assert_values(expected,corrupt)

    def test_independent_partial_and_reentry_coverage(self):
        r,d,f=episode_fixture();a=r['trade'];b=dict(a,entry_ts=27*900000,exit_ts=29*900000,partial_exit_taken=False)
        raw=[dict(t=int(t-900000),o=float(c),h=float(c+.5),l=float(c-.5),c=float(c)) for t,c in zip(d['time'],d['close'])]
        assert_values(scalar_episode(r,raw,f['ema'],f['atr'],[a,b]),episode(r,d,f,[a,b]))


if __name__=='__main__':unittest.main()
