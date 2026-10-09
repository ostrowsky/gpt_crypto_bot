import unittest
from mission_forward_shadow import validate_issue,hourly_day_row
from mission_contract import window

class IssueTests(unittest.TestCase):
    def test_known_receipt_and_frozen_fit_required(self):
        validate_issue(1000000,1000100,1000200,999999)
        for args in ((1000000,999999,1000200,1),(1000000,1000200,1000100,1),(1000000,1000100,1400000,1),(1000000,1000100,1000200,1000000)):
            with self.assertRaises(ValueError):validate_issue(*args)

    def test_dst_hourly_aggregation_preserves_actual_local_day(self):
        for day,n in (('2026-03-29',23),('2026-10-25',25)):
            lo,hi=window(day);rows=[[t,100,101,99,100,1,t+3599999,1000] for t in range(lo,hi,3600000)]
            row=hourly_day_row(day,rows);self.assertEqual((row[0],row[6],row[7]),(lo,hi-1,n*1000))
            with self.assertRaises(ValueError):hourly_day_row(day,rows[:-1])

if __name__=='__main__':unittest.main()
