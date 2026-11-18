"""TA03..07 named contracts, written before implementation."""
from decimal import Decimal as D
import copy
import unittest
import _support
from binance_demo_bot.trade_minute_core import TradeMinuteAccount, validate_trade_window, run_trade_minute
from binance_demo_bot.execution_order_sensitivity import financial_check
from binance_demo_bot.rl_direction_core import STEP


def activity_frame(at=0,active=None,carry=100):
    active={0:(100,100.1,99.9,100)} if active is None else active
    rows=[];previous=str(carry)
    for i in range(15):
        values=active.get(i);o,h,l,c=map(str,values) if values else (previous,)*4
        rows.append([at+i*60000,o,h,l,c,'1' if values else '0',at+(i+1)*60000-1,
                     '100' if values else '0',1 if values else 0,'0.5' if values else '0','50' if values else '0','0'])
        previous=c
    traded=[r for r in rows if r[8]>0]
    ohlc=(traded[0][1],str(max(D(r[2]) for r in traded)),str(min(D(r[3]) for r in traded)),traded[-1][4]) if traded else (str(carry),)*4
    n=len(traded)
    original=[at,*ohlc,str(n),at+STEP-1,str(100*n),n,str(D('.5')*n),str(50*n),'0']
    return dict(source_kind='BINANCE_SPOT_DEMO_1m',original=original,rows=rows,received_at_ms=at+STEP,source_ref='synthetic-sealed',source_sha256='a'*64)

def parent(f):return dict(zip(('open','high','low','close'),map(float,f['original'][1:5])))
def step(a,f,action=2):return a.step(f['original'][0],'AAAUSDT',action,lambda *_:parent(f),lambda *_:f)


class TradeValidationTests(unittest.TestCase):
    def test_TA04_padding_matches_new_rule_but_old_mismatch_retained(self):
        f=activity_frame(active={3:(100,100.1,99.9,100)},carry=98)
        v=validate_trade_window(f['original'],f['rows'],received_at_ms=f['received_at_ms'])
        self.assertEqual(v['state'],'MATCH_TRADE_AWARE');self.assertEqual(v['strict']['state'],'MISMATCH')
        self.assertEqual(v['empty_minutes_n'],14);self.assertEqual(v['first_trade_ms'],180000)

    def test_TA03_empty_activity_and_discontinuous_carry_rejected(self):
        for mutate in (lambda f:f['rows'][1].__setitem__(5,'1'),lambda f:f['rows'][1].__setitem__(2,'101'),
                       lambda f:f['rows'][1].__setitem__(1,'99')):
            f=activity_frame();mutate(f)
            self.assertIn(validate_trade_window(f['original'],f['rows'],received_at_ms=STEP)['state'],('MISMATCH','INCOMPLETE'))

    def test_TA04_price_or_volume_true_mismatch_not_excused(self):
        for field,value in ((1,'101'),(5,'999'),(8,99)):
            f=activity_frame();f['original'][field]=value
            if field==1:f['original'][2]='102'
            self.assertNotIn(validate_trade_window(f['original'],f['rows'],received_at_ms=STEP)['state'],('MATCH_TRADE_AWARE','MATCH_NO_TRADES'))

    def test_TA04_missing_minute_is_not_a_no_trade_minute(self):
        f=activity_frame();f['rows'].pop(3)
        self.assertEqual(validate_trade_window(f['original'],f['rows'],received_at_ms=STEP)['state'],'INCOMPLETE')

    def test_TA04_entire_empty_bar_has_separate_state(self):
        f=activity_frame(active={});v=validate_trade_window(f['original'],f['rows'],received_at_ms=STEP)
        self.assertEqual(v['state'],'MATCH_NO_TRADES');self.assertIsNone(v['first_trade_ms'])


class TradeAccountTests(unittest.TestCase):
    def test_TA05_entry_waits_for_first_traded_minute_no_phantom_stop(self):
        a=TradeMinuteAccount();step(a,activity_frame(active={3:(100,100.1,99.9,100)},carry=98))
        self.assertEqual(a.events[0]['at_ms'],180000);self.assertIsNotNone(a.position)
        self.assertEqual(a.position['entry_decision_ms'],0);self.assertEqual(a.position['entry_ms'],180000)
        self.assertEqual(a.trades,[]);self.assertEqual(financial_check(a.report(0,STEP))['state'],'PASS')
        obs=a.observation(STEP,'AAAUSDT',lambda *_:[0.]*20)
        self.assertEqual(obs[23],1/16)

    def test_TA05_empty_buy_does_not_debit_and_expired_intent_is_counted(self):
        a=TradeMinuteAccount();step(a,activity_frame(active={}))
        self.assertEqual(a.cash,100);self.assertEqual(a.events,[]);self.assertEqual(a.unfilled_intents_n,1)
        self.assertEqual(a.report(0,STEP)['state'],'COMPLETE')

    def test_TA05_empty_held_exit_retains_exposure_and_carry_mark(self):
        a=TradeMinuteAccount();step(a,activity_frame());cash=a.cash
        step(a,activity_frame(STEP,active={}),3)
        self.assertEqual(a.cash,cash);self.assertIsNotNone(a.position);self.assertEqual(a.unfilled_intents_n,1)
        self.assertEqual(a.carry_exposure_steps_n,1)

    def test_TA06_resting_gap_before_delayed_discretionary_exit(self):
        a=TradeMinuteAccount();step(a,activity_frame())
        step(a,activity_frame(STEP,active={5:(98,102,97,100)},carry=100),3)
        self.assertEqual(a.trades[0]['reason'],'SL');self.assertEqual(a.trades[0]['exit_ms'],STEP+300000)
        self.assertEqual(a.trades[0]['exit_price'],98*(1-a.slip))

    def test_TA06_same_traded_minute_remains_two_scenarios(self):
        for order,reason in (('MINUTE_STOP_FIRST','SL'),('MINUTE_TARGET_FIRST','TP')):
            a=TradeMinuteAccount(intrabar_order=order);step(a,activity_frame(active={4:(100,101,99,100)}))
            self.assertEqual(a.trades[0]['reason'],reason);self.assertEqual(len(a.ambiguities),1)

    def test_TA06_different_minutes_resolve_before_convention(self):
        a=TradeMinuteAccount();step(a,activity_frame(active={2:(100,101,99.9,100),7:(100,100.1,99,100)}))
        self.assertEqual(a.trades[0]['reason'],'TP');self.assertEqual(a.ambiguities,[])

    def test_TA07_deadline_not_backdated_to_last_trade_in_sixteenth_bar(self):
        a=TradeMinuteAccount(exit_mode='learned')
        for i in range(16):step(a,activity_frame(i*STEP,active={2:(100,100.1,99.9,100)}),2 if i==0 else 0)
        self.assertIsNotNone(a.position);self.assertEqual(a.trades,[])
        step(a,activity_frame(16*STEP,active={}),0)
        self.assertIsNotNone(a.position);self.assertEqual(a.overdue_empty_steps_n,1)
        step(a,activity_frame(17*STEP,active={7:(100,100.1,99.9,100)}),0)
        self.assertEqual(a.trades[0]['reason'],'HOLDING_LIMIT');self.assertEqual(a.trades[0]['exit_ms'],17*STEP+420000)
        self.assertEqual(financial_check(a.report(0,18*STEP))['state'],'PASS')

    def test_TA06_trailing_and_learned_invariant_across_unknown_target_conventions(self):
        for mode in ('trailing','learned'):
            reports=[]
            for order in ('MINUTE_STOP_FIRST','MINUTE_TARGET_FIRST'):
                a=TradeMinuteAccount(exit_mode=mode,intrabar_order=order)
                step(a,activity_frame(active={2:(100,102,99.9,101)}));step(a,activity_frame(STEP,active={3:(101,103,98,100)}),0)
                reports.append(a.report(0,2*STEP))
            self.assertEqual(*reports)

    def test_TA05_action_precedes_native_lookup(self):
        calls=[]
        class Tape:
            candidate=lambda s,at:'AAAUSDT'
            vector=lambda s,*_:[0.]*20
            def bar(s,*_):calls.append('bar');return parent(activity_frame())
        class Policy:
            def predict(s,obs,deterministic=True):calls.append('decision');return 2
        def lookup(symbol,at,decision):
            self.assertEqual(calls,['decision','bar']);self.assertEqual(decision['action'],2);return activity_frame()
        r=run_trade_minute(Tape(),0,STEP,kind='PPO',policy=Policy(),minute_lookup=lookup)
        self.assertEqual(r['state'],'COMPLETE')

    def test_TA04_missing_native_data_nulls_objective(self):
        a=TradeMinuteAccount();f=activity_frame();f['rows'].pop()
        step(a,f);self.assertIsNone(a.report(0,STEP)['net_pnl_usdt']);self.assertEqual(a.cash,100)

if __name__=='__main__':unittest.main()
