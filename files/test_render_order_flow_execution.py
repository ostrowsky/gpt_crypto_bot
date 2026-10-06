import unittest
from render_order_flow_execution import rows


class RenderTest(unittest.TestCase):
    def test_unknown_not_formatted_as_zero(self):
        fields=['issued','deferred','action_known','action_unknown','matched','matched_deferred',
            'mean_gain_bp','selected_mean_gain_bp','harmed','benefited','mean_shortfall_bp','p95_shortfall_bp','p99_shortfall_bp']
        r={k:None for k in fields};r.update(issued=10,action_unknown=10,matched=0)
        out=rows(dict(scores={'5':dict(methods={'OFI':r})}))
        self.assertIsNone(out[0]['mean_gain_bp']);self.assertEqual(out[0]['action_unknown'],10)


if __name__=='__main__':unittest.main()
