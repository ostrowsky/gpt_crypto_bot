import unittest
import numpy as np
from impulse_entry_catboost import features,net_target,BAR
from verify_impulse_entry_catboost import raw_oracle
from test_impulse_entry_catboost import data


class OracleTests(unittest.TestCase):
    def test_independent_json_oracle_matches_and_tail_does_not_hide_features(self):
        a=data();raw={int(v['t'])+BAR:{k:float(v[k]) for k in ('o','h','l','c','v')} for v in a}
        for tf in ('15m','1h'):
            x,y=raw_oracle(raw,20*BAR,tf,7.5,5)
            np.testing.assert_allclose(x,features(a,20*BAR,tf),rtol=1e-12)
            self.assertAlmostEqual(y,net_target(a,20*BAR,7.5,5))
        tail={k:v for k,v in raw.items() if k<=22*BAR}
        x,y=raw_oracle(tail,20*BAR,'1h',7.5,5)
        self.assertIsNotNone(x);self.assertIsNone(y)

    def test_gap_and_corrupt_raw_invalidated(self):
        a=data();raw={int(v['t'])+BAR:{k:float(v[k]) for k in ('o','h','l','c','v')} for v in a}
        del raw[18*BAR]
        self.assertIsNone(raw_oracle(raw,20*BAR,'15m',7.5,5)[0])
        raw[18*BAR]=dict(o=100,c=100,l=90,h=80,v=1)
        self.assertIsNone(raw_oracle(raw,20*BAR,'15m',7.5,5)[0])


if __name__=='__main__':unittest.main()
