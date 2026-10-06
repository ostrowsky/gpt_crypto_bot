import unittest
from render_leader_trend_continuation import table_rows,verify_entry_identity,ENTRY_FIELDS


class RenderTests(unittest.TestCase):
    def test_table_preserves_denominators_and_corrected_early(self):
        m=dict(early=247,leader_pairs=570,captured=480,precision_n=1250,precision_N=2470)
        e=dict(confirmed_n=430,retention_mean=.08,accompaniment_mean=.53,first_early_exit_n=407)
        arm=dict(full=m,test=m,exits_full=e,exits_test=e)
        rows=table_rows(dict(arms=dict(control=arm,trend=arm)))
        self.assertEqual(len(rows),4);self.assertEqual(rows[0]['early'],247);self.assertEqual(rows[0]['all_BUY'],2470)
        self.assertEqual(rows[0]['confirmed'],430)

    def test_same_clock_price_requires_same_initial_mode_but_not_future_extrema(self):
        t={k:1 for k in ENTRY_FIELDS};t.update(sym='A',tf='15m',mode='trend',entry_ts=100,entry_price=10)
        a=dict(day='2026-09-01',symbol='A',trade=dict(t,max_favorable_pct=1))
        b=dict(a,trade=dict(t,max_favorable_pct=999))
        self.assertEqual(verify_entry_identity([a],[b],0),1)
        b['trade']['tf']='1h'
        with self.assertRaisesRegex(ValueError,'immutable'):verify_entry_identity([a],[b],0)


if __name__=='__main__':unittest.main()
