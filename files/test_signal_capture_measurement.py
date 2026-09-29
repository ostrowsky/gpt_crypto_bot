import unittest
from signal_capture_measurement import measure
from top_gainer_critic import _exit_quality_metrics


def e(t, **kw):
    return dict(ts='2026-09-24T'+t+':00Z', sym='QNTUSDT', tf='15m', _log_source='bot', **kw)


class MeasurementTests(unittest.TestCase):
    def test_repeated_trades(self):
        result = measure(dict(entries=[e('10:00', price=100), e('12:00', price=120)],
                              exits=[e('11:00', entry_price=100, exit_price=110),
                                     e('13:00', entry_price=120, exit_price=115)]), 100, 125)
        self.assertEqual(result['realized_capture_ratio'], .2)
        self.assertEqual(result['matched_trades'], 2)
        self.assertIsNone(result['exit_efficiency'])

    def test_fail_closed(self):
        entry = e('10:00', price=100)
        for exit in [e('11:00', entry_price=90, exit_price=110),
                     e('11:00', entry_price=100),
                     e('11:00', entry_price=100, exit_price=float('nan')),
                     e('10:00', entry_price=100, exit_price=110),
                     e('11:00', entry_price=100, exit_price=110, position_id='different')]:
            with self.subTest(exit=exit):
                self.assertIsNone(measure(dict(entries=[entry], exits=[exit]),100,125)['realized_capture_ratio'])

    def test_duplicates_and_open_positions(self):
        entry, exit = e('10:00',price=100), e('11:00',entry_price=100,exit_price=110)
        self.assertEqual(measure(dict(entries=[entry,entry],exits=[exit,exit]),100,125)['matched_trades'],1)
        self.assertIsNone(measure(dict(entries=[entry]),100,125)['realized_capture_ratio'])

    def test_old_day_high_metric_is_unknown(self):
        self.assertEqual(_exit_quality_metrics(entry_price=100,day_high=150,
                         last_exit={'pnl_pct':5}), (None,None))

    def test_source_and_overlapping_chain_are_unknown(self):
        entries=[e('10:00',price=100),e('10:30',price=101)]
        exits=[e('11:00',entry_price=101,exit_price=110)]
        self.assertIsNone(measure(dict(entries=entries,exits=exits),100,125)['realized_capture_ratio'])
        exits[0]['_log_source']='agent'
        self.assertIsNone(measure(dict(entries=entries[:1],exits=exits),100,125)['realized_capture_ratio'])

    def test_summary_exposes_new_definition(self):
        from top_gainer_critic import DayPerformance, summarize_top_gainer
        perf=DayPerformance('QNTUSDT',100,125,160,90,25,1000000,True)
        result=summarize_top_gainer(perf,dict(entries=[e('10:00',price=100)],
            exits=[e('11:00',entry_price=100,exit_price=110,pnl_pct=10)]))
        self.assertEqual(result['realized_capture_ratio'],.4)
        self.assertIsNone(result['exit_efficiency'])
        self.assertEqual(result['exit_quality_denominator'],0)


if __name__ == '__main__': unittest.main()
