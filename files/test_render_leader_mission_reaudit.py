import unittest
from render_leader_mission_reaudit import table_rows


class RenderTests(unittest.TestCase):
    def test_report_uses_mission_fields_and_keeps_unknown(self):
        s=dict(early=1,leader_pairs=15,captured=2,precision_n=3,precision_N=6,unique_precision_n=2,unique_precision_N=4,
            stored_early=4,corrected_early_disagreements=3)
        e=dict(confirmed_n=0,first_early_exit_n=0,retention_mean=None,accompaniment_mean=None,any_remaining_n=0,rebound_n=0,rebound_N=0)
        r=dict(arms={'control':dict(full=s,test=s,exits_full=e,exits_test=e)})
        out=table_rows(r);self.assertEqual(out[0]['early'],1);self.assertEqual(out[0]['old_early'],4)
        self.assertIsNone(out[0]['retention']);self.assertNotIn('net_return_pct',out[0])


if __name__=='__main__':unittest.main()
