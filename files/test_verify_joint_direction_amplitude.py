import unittest
import numpy as np
from joint_direction_amplitude import probability_metrics
from verify_joint_direction_amplitude import probability_oracle,assert_nested


class MetricOracleTests(unittest.TestCase):
    def test_bin_edges_and_counts_match_independent_oracle(self):
        p=np.r_[np.linspace(0,1,11),[.15,.75]];truth=p>.6
        assert_nested(probability_oracle(p,truth),probability_metrics(p,truth))
        result=probability_oracle(p,truth)
        self.assertEqual(sum(v['n'] for v in result['bins']),len(p))
        self.assertEqual(result['up_n'],int(truth.sum()))

    def test_metric_tampering_rejected(self):
        with self.assertRaises(ValueError):assert_nested({'n':2},{'n':3})
        with self.assertRaises(ValueError):assert_nested({'p':.1},{'p':.2})
        assert_nested({'unknown':None},{'unknown':None})


if __name__=='__main__':unittest.main()
