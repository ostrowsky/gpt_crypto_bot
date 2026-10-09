import unittest
from fetch_mission_history import daily_request_range,completed_rows
from mission_contract import window


class FetchTests(unittest.TestCase):
    def test_constant_offset_requests_closed_local_days(self):
        lo=window('2026-04-01')[0];hi=window('2026-10-09')[0]
        r=daily_request_range(lo,hi);self.assertEqual(r['timeZone'],'2:00');self.assertEqual(r['endTime'],hi-1)

    def test_dst_change_is_not_silently_24h(self):
        with self.assertRaisesRegex(ValueError,'DST'):daily_request_range(window('2026-10-24')[0],window('2026-10-27')[0])

    def test_delisting_last_partial_bar_does_not_erase_prior_complete_days(self):
        lo,hi=window('2026-04-01');lo2,hi2=window('2026-04-02')
        full=[lo,100,100,100,100,0,hi-1,2000000]
        short=[lo2,100,100,100,100,0,lo2+5*3600000-1,100000]
        valid,partial=completed_rows([full,short],lo,hi2)
        self.assertEqual(valid,[full]);self.assertEqual(partial[0]['day'],'2026-04-02')


if __name__=='__main__':unittest.main()
