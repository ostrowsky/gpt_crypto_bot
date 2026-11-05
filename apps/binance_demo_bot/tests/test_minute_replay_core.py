"""MRP-1 acceptance contracts, written before implementation."""
import copy
import unittest
import _support
from binance_demo_bot.minute_replay_core import MinuteAccount, CandidateTape, MemoPolicy, run_minute
from binance_demo_bot.execution_order_sensitivity import financial_check
from binance_demo_bot.rl_direction_core import STEP


def frame(at=0, touches=()):
    rows=[[at+i*60000,'100','100.1','99.9','100','1',at+(i+1)*60000-1,'100',1,'0.5','50','0'] for i in range(15)]
    for i,high,low in touches: rows[i][2]=str(high); rows[i][3]=str(low)
    original=[at,'100',str(max(float(r[2]) for r in rows)),str(min(float(r[3]) for r in rows)),'100','15',at+STEP-1,'1500',15,'7.5','750','0']
    return dict(source_kind='BINANCE_SPOT_DEMO_1m',original=original,rows=rows,received_at_ms=at+STEP,source_ref='sealed-test')

def parent(f): return dict(zip(('open','high','low','close'),map(float,f['original'][1:5])))

def step(a,f,action=2): return a.step(f['original'][0],'AAAUSDT',action,lambda *_:parent(f),lambda *_:f)


class MinuteAccountTests(unittest.TestCase):
    def test_MR02_chronological_target_overrides_coarse_stop_first(self):
        for order in ('MINUTE_STOP_FIRST','MINUTE_TARGET_FIRST'):
            a=MinuteAccount(intrabar_order=order)
            step(a,frame(touches=((2,101,99.9),(7,100.1,99))))
            self.assertEqual(a.trades[0]['reason'],'TP')
            self.assertEqual(a.trades[0]['exit_ms'],120000)
            self.assertEqual(a.ambiguities,[])
            self.assertEqual(financial_check(a.report(0,STEP))['state'],'PASS')

    def test_MR02_stop_before_target_is_not_overridden(self):
        a=MinuteAccount(intrabar_order='MINUTE_TARGET_FIRST')
        step(a,frame(touches=((1,100.1,99),(3,101,99.9))))
        self.assertEqual(a.trades[0]['reason'],'SL'); self.assertEqual(a.ambiguities,[])

    def test_MR02_only_same_minute_uses_convention(self):
        for order,reason in (('MINUTE_STOP_FIRST','SL'),('MINUTE_TARGET_FIRST','TP')):
            a=MinuteAccount(intrabar_order=order); step(a,frame(touches=((4,101,99),)))
            self.assertEqual(a.trades[0]['reason'],reason)
            self.assertEqual(a.ambiguities[0]['bucket_origin_ms'],240000)
            self.assertEqual(a.ambiguities[0]['resolution_ms'],60000)

    def test_MR04_missing_wrong_source_and_mismatch_invalidate_without_cash_mutation(self):
        for mutate in (lambda f:f['rows'].pop(),lambda f:f.update(source_kind='BINANCE_SPOT_1s'),
                       lambda f:f['rows'][1].__setitem__(2,'102')):
            f=frame(); mutate(f); a=MinuteAccount(); step(a,f)
            self.assertEqual(a.cash,100); self.assertEqual(a.events,[])
            r=a.report(0,STEP); self.assertEqual(r['state'],'INCOMPLETE'); self.assertIsNone(r['net_pnl_usdt'])

    def test_MR05_gap_stop_and_open_exit_priority(self):
        for gap in (99,102,100):
            a=MinuteAccount(); step(a,frame()); entry=a.position['entry_price']
            f=frame(STEP,touches=((2,104,98),)); f['rows'][0][1]=str(gap)
            f['rows'][0][2]=str(max(100.1,gap)); f['rows'][0][3]=str(min(99.9,gap)); f['original'][1]=str(gap)
            step(a,f,3)
            trade=a.trades[0]
            self.assertEqual(trade['reason'],'SL' if gap==99 else 'TP' if gap==102 else 'EXIT')
            self.assertAlmostEqual(trade['exit_price'],entry*1.005 if gap==102 else gap*(1-a.slip))
            self.assertEqual(a.ambiguities,[])

    def test_MR05_within_bar_open_gap_uses_worse_price(self):
        a=MinuteAccount(); f=frame(touches=((3,100.1,98),))
        f['rows'][3][1]='98'; step(a,f)
        self.assertEqual(a.trades[0]['exit_price'],98*(1-a.slip))

    def test_MR06_trailing_uses_prior_parent_peak_and_nonfixed_invariant(self):
        reports=[]
        for order in ('MINUTE_STOP_FIRST','MINUTE_TARGET_FIRST'):
            a=MinuteAccount(intrabar_order=order,exit_mode='trailing')
            step(a,frame(touches=((2,102,99.9),)))
            self.assertIsNotNone(a.position)  # no retroactive trail within this parent bar
            f=frame(STEP,touches=((3,103,99),)); step(a,f,0)
            self.assertEqual(a.trades[0]['reason'],'TRAIL')
            self.assertEqual(a.ambiguities,[]); reports.append(a.report(0,2*STEP))
        self.assertEqual(reports[0],reports[1])

    def test_MR04_flat_wait_does_not_fetch_execution(self):
        def forbidden(*_): self.fail('flat wait looked at execution')
        a=MinuteAccount(); a.step(0,'AAAUSDT',0,forbidden,forbidden)
        self.assertTrue(a.valid); self.assertEqual(a.cash,100)

    def test_MR03_changes_future_admissions_not_trade_patch(self):
        class Tape:
            candidate=lambda s,at:'AAAUSDT'
            forecast=lambda s,*args:(.9,.02)
            bar=lambda s,symbol,at:parent(frame(at,touches=((4,101,99),)))
        reports=[run_minute(Tape(),0,30*STEP,kind='SIGNAL',threshold=0,
            intrabar_order=o,minute_lookup=lambda symbol,at,decision:frame(at,touches=((4,101,99),)))
            for o in ('MINUTE_STOP_FIRST','MINUTE_TARGET_FIRST')]
        self.assertLess(reports[0]['trades_n'],reports[1]['trades_n'])
        self.assertEqual(reports[1]['trades_n'],30)
        self.assertTrue(all(financial_check(r)['state']=='PASS' for r in reports))

    def test_MR04_action_frozen_before_lookup(self):
        calls=[]
        class Tape:
            candidate=lambda s,at:'AAAUSDT'
            vector=lambda s,*args:[0.]*20
            def bar(s,symbol,at): calls.append('bar'); return parent(frame(at))
        class Policy:
            def predict(s,obs,deterministic=True): calls.append('decision'); return 2
        def lookup(symbol,at,decision):
            self.assertEqual(decision['action'],2); self.assertEqual(calls,['decision','bar'])
            calls.append('minute'); return frame(at)
        run_minute(Tape(),0,STEP,kind='PPO',policy=Policy(),minute_lookup=lookup)
        self.assertEqual(calls,['decision','bar','minute'])


class AccelerationTests(unittest.TestCase):
    def test_MR11_candidate_cache_exact_and_held_fallback_without_bar_lookup(self):
        class Tape:
            def __init__(s): s.calls=[]
            def candidate(s,at): return None if at==STEP else 'AAAUSDT'
            def prediction(s,symbol,at):
                s.calls.append((symbol,at)); return None if symbol is None else [float(i)/7 for i in range(24)]
            def bar(s,*_): raise AssertionError('execution during feature preparation')
        raw=Tape(); cached=CandidateTape(raw,0,3*STEP)
        for at in range(0,3*STEP,STEP):
            symbol=raw.candidate(at)
            self.assertEqual(cached.prediction(symbol,at),raw.prediction(symbol,at))
        self.assertIsNone(cached.forecast(None,STEP,'GRU'))
        self.assertEqual(cached.vector('HELDUSDT',STEP),[float(i)/7 for i in range(20)])
        self.assertIn(('HELDUSDT',STEP),raw.calls)
        self.assertEqual(cached.forecast('AAAUSDT',0,'CatBoost'),(21/7,23/7))

    def test_MR11_memo_uses_exact_observation_and_model_identity(self):
        class Policy:
            def __init__(s): s.calls=0
            def predict(s,obs,deterministic=True): s.calls+=1; return 2
        p=Policy(); memo=MemoPolicy(p,'a'*64,max_entries=2); obs=[0.]*27
        self.assertEqual(memo.predict(obs),2); self.assertEqual(memo.predict(obs),2); self.assertEqual(p.calls,1)
        obs[0]=1e-15; memo.predict(obs); self.assertEqual(p.calls,2)
        with self.assertRaises(ValueError): memo.predict(obs,deterministic=False)
        other=MemoPolicy(p,'b'*64,max_entries=2); other.predict(obs); self.assertEqual(p.calls,3)
        memo.predict([1.]*27); self.assertLessEqual(memo.entries_n,2)


class CandidateIdentityTests(unittest.TestCase):
    def test_MR11_feature_identity_includes_dtype_mask_and_exact_values(self):
        class Tape:
            candidate=lambda s,at:'AAAUSDT'
            def prediction(s,symbol,at):return None if at==900000 else [s.value]*24
            value=.125
        a=CandidateTape(Tape(),0,1800000);b=CandidateTape(Tape(),0,1800000)
        self.assertEqual(a.identity()['sha256'],b.identity()['sha256'])
        raw=Tape();raw.value=.12500000000001;c=CandidateTape(raw,0,1800000)
        self.assertNotEqual(a.identity()['sha256'],c.identity()['sha256'])
        self.assertEqual(a.identity()['missing_candidate_features_n'],1)

if __name__=='__main__':unittest.main()
