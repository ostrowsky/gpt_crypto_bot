import unittest
from mission_calendar_evaluation import interval,evaluate

class CalendarTests(unittest.TestCase):
    def test_missing_day_is_a_mask_not_a_consecutive_row(self):
        a=[dict(day=d,early=1) for d in ('2026-04-01','2026-04-04','2026-04-05')]
        b=[dict(r,early=2) for r in a];ci=interval(a,b)
        self.assertEqual((ci['known_days'],ci['calendar_days'],ci['unknown_calendar_days']), (3,5,2))
        self.assertEqual(ci['early_count_delta95'],[3.,3.]);self.assertEqual(ci['known_early_delta'],3)

    def test_incompatible_windows_and_unknown_exit_gate_reject(self):
        with self.assertRaises(ValueError):interval([dict(day='2026-04-01',early=1)],[])
        base=dict(early=1,captured=2,unique_precision_n=2,unique_precision_N=4)
        target=dict(early=2,captured=2,unique_precision_n=2,unique_precision_N=4)
        r=evaluate(base,target,dict(early_count_delta95=[1,2]));self.assertEqual(r['state'],'RETROSPECTIVE_NOT_ACCEPTED')

if __name__=='__main__':unittest.main()
