from __future__ import annotations
import unittest
import replay_observable_tail_selector as obs

class ObservableTailSelectorReplayTests(unittest.TestCase):
    def test_candidate_weak_signal_mfe_threshold(self):
        rows=[{'exit_reason_bucket':'weak_signal','max_favorable_pct':2.0},{'exit_reason_bucket':'weak_signal','max_favorable_pct':0.5},{'exit_reason_bucket':'ema_break','max_favorable_pct':3.0}]
        fn=dict((name,fn) for name,_,fn in obs._candidate_selectors())['weak_signal_mfe150']
        self.assertEqual([fn(r) for r in rows],[True,False,False])
    def test_score_blocks_unselected_rows_at_baseline(self):
        rows=[{'bucket':'early_exits','pnl_pct':1.0,'tail50_h10_ema20_cap150_pnl_pct':3.0,'exit_reason_bucket':'weak_signal'}, {'bucket':'false_positive_buys','pnl_pct':-1.0,'tail50_h10_ema20_cap150_pnl_pct':-3.0,'exit_reason_bucket':'ema_break'}]
        fn=lambda r: r['exit_reason_bucket']=='weak_signal'
        s=obs._score(rows,'x',fn,'tail50_h10_ema20_cap150')
        self.assertEqual(s['allowed_total'],1)
        self.assertEqual(s['false_positive_allowed_rate_pct'],0.0)
        self.assertEqual(s['avg_delta_pct'],1.0)
    def test_decision_requires_test_gate(self):
        ranked=[{'name':'bad','test':{'n':10,'avg_delta_pct':0.2,'median_delta_pct':-0.1,'worse_rate_pct':10,'allowed_rate_pct':10,'false_positive_allowed_rate_pct':0}}]
        self.assertEqual(obs._decision(ranked),'no_observable_selector_passed_test_gate')

    def test_decision_accepts_zero_false_positive_rate(self):
        ranked=[{'name':'good','test':{'n':10,'avg_delta_pct':0.2,'median_delta_pct':0.0,'worse_rate_pct':10,'allowed_rate_pct':10,'false_positive_allowed_rate_pct':0.0}}]
        self.assertEqual(obs._decision(ranked),'advance_good_to_shadow_observable_tail_selector')

if __name__ == '__main__':
    unittest.main()
