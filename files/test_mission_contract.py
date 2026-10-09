import unittest
from mission_contract import window,daily_label,candidate_target,remaining,CONTRACT,target_symbol


class ContractTests(unittest.TestCase):
    def test_complete_local_day_dst_boundaries(self):
        self.assertEqual(window('2026-03-29')[1]-window('2026-03-29')[0],23*3600000)
        self.assertEqual(window('2026-10-25')[1]-window('2026-10-25')[0],25*3600000)
        self.assertEqual(window('2026-10-08')[1]-window('2026-10-08')[0],24*3600000)

    def test_exchange_top_before_watchlist_filter_and_historical_scope(self):
        lo,hi=window('2026-10-08')
        bars={f'A{i}USDT':[lo,100+i,200,50,200-i,0,hi-1,2000000] for i in range(25)}
        watch={'A0USDT','A24USDT'};coverage=dict(missing=[],historical_PIT_certified=False)
        label=daily_label('2026-10-08',bars,watch,hi,coverage)
        self.assertEqual(len(label['exchange_top']),20);self.assertEqual(label['leaders'],['A0USDT'])
        self.assertFalse(label['historical_PIT_certified']);self.assertEqual(label['contract'],CONTRACT['id'])
        self.assertIsNone(candidate_target('A0USDT',lo+1000,110,label,hi))
        self.assertEqual(candidate_target('A0USDT',lo+1000,110,label,hi+1)['early_leader'],1)

    def test_incomplete_future_unknown_and_invalid_boundary(self):
        lo,hi=window('2026-10-08');coverage=dict(missing=['B'],historical_PIT_certified=False)
        self.assertIsNone(daily_label('2026-10-08',{},set(),hi-1,coverage)['leaders'])
        row=[lo,100,101,99,101,0,hi-2,2000000]
        with self.assertRaisesRegex(ValueError,'boundary'):daily_label('2026-10-08',{'AUSDT':row},set(),hi,coverage)
        self.assertIsNone(remaining(100,99,100));self.assertFalse(target_symbol('BTCUPUSDT'))

    def test_known_illiquid_candidate_is_negative_not_missing(self):
        lo,hi=window('2026-10-08');coverage=dict(missing=[],historical_PIT_certified=False)
        row=[lo,100,200,99,200,0,hi-1,1]
        label=daily_label('2026-10-08',{'AUSDT':row},{'AUSDT'},hi,coverage)
        self.assertEqual(label['leaders'],[])
        self.assertEqual(candidate_target('AUSDT',lo+1000,110,label,hi+1)['leader'],0)


if __name__=='__main__':unittest.main()
