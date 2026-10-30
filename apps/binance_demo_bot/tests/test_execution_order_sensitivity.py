"""EOS04..09 causal account and absorbing-risk equivalence contracts."""
import copy
import unittest

import _support
from binance_demo_bot.rl_direction_core import ResearchAccount, STEP
from binance_demo_bot.rl_direction_runner import run_portfolio
from binance_demo_bot.execution_order_sensitivity import (ScenarioAccount, run_scenario,
    make_signal_schedule, compare_reference, financial_check)


def candle(o=100,h=101,l=99,c=100): return dict(open=o,high=h,low=l,close=c)


class Tape:
    def __init__(self,rows): self.rows=rows; self.bar_calls=[]; self.feature_calls=[]
    def candidate(self,at): return 'AAAUSDT'
    def bar(self,symbol,at): self.bar_calls.append((symbol,at)); return self.rows[at//STEP]
    def forecast(self,symbol,at,kind): self.feature_calls.append(at); return (.9,.02)
    def vector(self,symbol,at): self.feature_calls.append(at); return [0.]*20


class Policy:
    def __init__(self,action=2,tape=None): self.calls=0; self.action=action; self.tape=tape
    def predict(self,observation,deterministic=True):
        if self.tape is not None: assert len(self.tape.bar_calls)<=self.calls
        self.calls+=1; return self.action


class ExecutionOrderTests(unittest.TestCase):
    def test_EOS05_both_touch_resolves_only_declared_convention(self):
        for mode,reason in (('STOP_FIRST','SL'),('TARGET_FIRST','TP')):
            account=ScenarioAccount(intrabar_order=mode)
            account.step(0,'AAAUSDT',2,lambda *_:candle())
            self.assertEqual(account.trades[0]['reason'],reason)
            self.assertEqual(len(account.ambiguities),1)
            self.assertEqual(account.ambiguities[0]['chosen_reason'],reason)
            self.assertIsNone(account.position)
            self.assertEqual(financial_check(account.report(0,STEP))['state'],'PASS')

    def test_EOS05_stop_first_exact_reference_on_diverse_transitions(self):
        sequences=[[(2,candle()),(2,candle()),(3,candle())],
            [(2,candle(h=100.1,l=99.9)),(3,candle(o=99,h=102,l=98))],
            [(2,candle(h=100.1,l=99.9)),(0,candle(o=102,h=103,l=98))],
            [(2,candle(h=100.1,l=99.9))]+[(0,candle(h=100.1,l=99.9))]*16,
            [(2,candle()),(2,None)]]
        for mode in ('fixed','trailing','learned'):
            for sequence in sequences:
                with self.subTest(mode=mode,sequence=sequence):
                    original=ResearchAccount(exit_mode=mode); scenario=ScenarioAccount(exit_mode=mode)
                    for i,(action,row) in enumerate(sequence):
                        fn=lambda *_,row=row:row
                        a=original.step(i*STEP,'AAAUSDT',action,fn)
                        b=scenario.step(i*STEP,'AAAUSDT',action,fn)
                        self.assertEqual(a,b)
                    self.assertEqual(compare_reference(original.report(0,len(sequence)*STEP),
                        scenario.report(0,len(sequence)*STEP))['state'],'MATCH')

    def test_EOS05_open_stop_precedes_later_target_even_in_target_first(self):
        a=ScenarioAccount(intrabar_order='TARGET_FIRST')
        a.step(0,'AAAUSDT',2,lambda *_:candle(h=100.1,l=99.9))
        a.step(STEP,'AAAUSDT',3,lambda *_:candle(o=99,h=103,l=98))
        self.assertEqual(a.trades[0]['reason'],'SL'); self.assertEqual(a.ambiguities,[])

    def test_EOS05_discretionary_open_exit_precedes_intrabar_protection(self):
        a=ScenarioAccount(intrabar_order='TARGET_FIRST')
        a.step(0,'AAAUSDT',2,lambda *_:candle(h=100.1,l=99.9))
        a.step(STEP,'AAAUSDT',3,lambda *_:candle(h=103,l=98))
        self.assertEqual(a.trades[0]['reason'],'EXIT'); self.assertEqual(a.ambiguities,[])

    def test_EOS05_favourable_open_gap_target_never_fills_above_target(self):
        a=ScenarioAccount(intrabar_order='TARGET_FIRST')
        a.step(0,'AAAUSDT',2,lambda *_:candle(h=100.1,l=99.9)); entry=a.position['entry_price']
        a.step(STEP,'AAAUSDT',0,lambda *_:candle(o=102,h=104,l=98))
        self.assertEqual(a.trades[0]['exit_price'],entry*1.005)
        self.assertEqual(a.ambiguities,[])

    def test_EOS04_nonfixed_arms_invariant_and_no_invented_target(self):
        for mode in ('trailing','learned'):
            a=ScenarioAccount(intrabar_order='STOP_FIRST',exit_mode=mode)
            b=ScenarioAccount(intrabar_order='TARGET_FIRST',exit_mode=mode)
            for i,row in enumerate((candle(h=102,l=99.9,c=101),candle(o=101,h=104,l=98,c=102))):
                self.assertEqual(a.step(i*STEP,'AAAUSDT',2,lambda *_:row),b.step(i*STEP,'AAAUSDT',2,lambda *_:row))
            self.assertEqual(a.ambiguities,b.ambiguities)

    def test_EOS05_nonambiguous_fixed_fills_invariant(self):
        for row in (candle(h=100.1,l=99),candle(h=102,l=99.9),candle(h=100.1,l=99.9)):
            a=ScenarioAccount(intrabar_order='STOP_FIRST'); b=ScenarioAccount(intrabar_order='TARGET_FIRST')
            self.assertEqual(a.step(0,'AAAUSDT',2,lambda *_:row),b.step(0,'AAAUSDT',2,lambda *_:row))

    def test_EOS05_missing_selected_or_held_data_stays_incomplete(self):
        a=ScenarioAccount(intrabar_order='TARGET_FIRST')
        a.step(0,'AAAUSDT',2,lambda *_:None)
        self.assertEqual(a.report(0,STEP)['state'],'INCOMPLETE')
        self.assertIsNone(a.report(0,STEP)['net_pnl_usdt'])

    def test_EOS07_shortcut_equivalent_but_intents_are_explicitly_different(self):
        rows=[candle(h=100.1,l=98)]*15
        p=Policy(); reference=run_portfolio(Tape(rows),0,15*STEP,kind='PPO',policy=p)
        q=Policy(); result=run_scenario(Tape(rows),0,15*STEP,kind='PPO',policy=q)
        self.assertEqual(compare_reference(reference,result)['state'],'MATCH')
        self.assertLess(q.calls,p.calls); self.assertGreater(result['risk_locked_steps_n'],0)
        self.assertTrue(any(a['source']=='SYSTEM_RISK_LOCK' and a['action'] is None for a in result['actions']))
        self.assertEqual(result['observed_steps_n'],15)

    def test_EOS07_shortcut_never_skips_while_holding(self):
        tape=Tape([candle(h=100.1,l=99.9)]*6); p=Policy()
        result=run_scenario(tape,0,6*STEP,kind='PPO',policy=p)
        self.assertEqual(p.calls,6); self.assertEqual(result['risk_locked_steps_n'],0)
        self.assertGreater(result['open_exposure_usdt'],0)

    def test_EOS05_policy_decision_precedes_selected_execution_lookup(self):
        tape=Tape([candle(h=100.1,l=99.9)]*3); p=Policy(tape=tape)
        run_scenario(tape,0,3*STEP,kind='PPO',policy=p)
        self.assertEqual(p.calls,3)

    def test_EOS06_signal_stream_full_coverage_and_same_as_frozen_runner(self):
        rows=[candle()]*20; tape=Tape(rows)
        reference=run_portfolio(Tape(rows),0,20*STEP,kind='SIGNAL',threshold=.002)
        signals=make_signal_schedule(tape,0,20*STEP,.002)
        result=run_scenario(Tape(rows),0,20*STEP,kind='SIGNAL',threshold=.002,signals=signals)
        self.assertEqual(result['signal_sha256'],reference['signal_sha256'])
        self.assertEqual(result['entry_signals_n'],20)
        self.assertEqual(compare_reference(reference,result)['state'],'MATCH')

    def test_EOS06_new_state_changes_future_admissions_not_old_trade_patch(self):
        # Repeated ambiguous entries: stop-first reaches the risk latch first.
        rows=[candle()]*30
        signals=make_signal_schedule(Tape(rows),0,30*STEP,.002)
        stop=run_scenario(Tape(rows),0,30*STEP,kind='SIGNAL',threshold=.002,signals=signals)
        target=run_scenario(Tape(rows),0,30*STEP,kind='SIGNAL',threshold=.002,signals=signals,intrabar_order='TARGET_FIRST')
        self.assertLess(stop['trades_n'],target['trades_n'])
        self.assertEqual(stop['signal_sha256'],target['signal_sha256'])
        self.assertEqual(target['trades_n'],30)

    def test_EOS09_financial_drift_and_missing_fields_cannot_match(self):
        a=ScenarioAccount(); a.step(0,'AAAUSDT',2,lambda *_:candle())
        reference=a.report(0,STEP); changed=copy.deepcopy(reference); changed['cash_usdt']+=.01
        self.assertEqual(compare_reference(reference,changed)['state'],'MISMATCH')
        self.assertEqual(financial_check(changed)['state'],'FAIL')
        changed=copy.deepcopy(reference); del changed['fees_usdt']
        self.assertEqual(compare_reference(reference,changed)['state'],'MISMATCH')

    def test_EOS09_open_mtm_and_complete_partial_days_reconcile(self):
        a=ScenarioAccount(exit_mode='learned'); a.step(0,'AAAUSDT',2,lambda *_:candle(h=100.1,l=99.9))
        r=a.report(0,STEP)
        self.assertEqual(financial_check(r)['state'],'PASS')
        self.assertEqual(r['complete_days_n'],0); self.assertEqual(r['partial_days_n'],1)

    def test_EOS09_reported_fees_must_match_execution_ledger(self):
        a=ScenarioAccount(); a.step(0,'AAAUSDT',2,lambda *_:candle())
        r=a.report(0,STEP); r['fees_usdt']+=.01
        self.assertEqual(financial_check(r)['state'],'FAIL')

    def test_EOS04_invalid_convention_rejected(self):
        for value in ('UNKNOWN','TP',None):
            with self.assertRaises(ValueError): ScenarioAccount(intrabar_order=value)


if __name__=='__main__': unittest.main()
