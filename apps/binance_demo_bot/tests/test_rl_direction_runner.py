"""RDE04..09: end-to-end portfolio runner causal action and exit contracts."""
import importlib
import unittest
import _support

STEP=900000


class FakeTape:
    def __init__(self): self.calls=[];self.signal=True
    def candidate(self,at):return 'AAAUSDT'
    def vector(self,symbol,at):return [0.]*16+[.6,.7,.01,.02]
    def forecast(self,symbol,at,kind):
        self.calls.append(('forecast',at))
        return (.7,.02 if self.signal else -.01)
    def bar(self,symbol,at):
        self.calls.append(('bar',at))
        return dict(open=100.,high=100.,low=100.,close=100.)


class RunnerContracts(unittest.TestCase):
    def setUp(self):self.api=importlib.import_module('binance_demo_bot.rl_direction_runner')
    def test_RDE09_forecast_entry_cost_threshold_and_exit(self):
        from binance_demo_bot.rl_direction_core import ResearchAccount
        tape=FakeTape();account=ResearchAccount()
        self.assertEqual(self.api.threshold_action(account,tape,0,'GRU',.002),2)
        account.step(0,'AAAUSDT',2,tape.bar);tape.signal=False
        self.assertEqual(self.api.threshold_action(account,tape,STEP,'GRU',.002),3)
    def test_RDE05_action_uses_forecast_before_future_execution(self):
        tape=FakeTape()
        report=self.api.run_portfolio(tape,0,2*STEP,kind='GRU',threshold=.002)
        self.assertEqual(tape.calls[0],('forecast',0))
        self.assertEqual(tape.calls[1],('bar',0))
        self.assertEqual(report['state'],'COMPLETE');self.assertIsNotNone(report['open_position'])
        self.assertLess(report['net_pnl_usdt'],0)
    def test_RDE09_exit_transfer_cannot_create_non_signal_entry(self):
        class AlwaysBuy:
            def predict(self,obs,deterministic=True):return 2
        tape=FakeTape();tape.signal=False
        report=self.api.run_portfolio(tape,0,4*STEP,kind='EXIT',policy=AlwaysBuy(),threshold=.002)
        self.assertEqual(report['trades_n'],0);self.assertEqual(report['counters']['buys_n'],0)
        self.assertEqual(report['net_pnl_usdt'],0.)
    def test_RDE09_common_signals_immutable_across_exit_admissions(self):
        tape=FakeTape()
        fixed=self.api.run_portfolio(tape,0,20*STEP,kind='SIGNAL',threshold=.002,exit_mode='fixed')
        trail=self.api.run_portfolio(tape,0,20*STEP,kind='SIGNAL',threshold=.002,exit_mode='trailing')
        self.assertEqual(fixed['signal_sha256'],trail['signal_sha256'])
        self.assertEqual(fixed['entry_signals_n'],20)
        self.assertLess(fixed['counters']['buys_n'],20)
    def test_RDE05_missing_execution_is_incomplete_not_fallback(self):
        tape=FakeTape();tape.bar=lambda *a:None
        report=self.api.run_portfolio(tape,0,4*STEP,kind='GRU',threshold=.002)
        self.assertEqual(report['state'],'INCOMPLETE');self.assertIsNone(report['net_pnl_usdt'])
    def test_RDE06_policies_share_exact_observation(self):
        seen=[]
        class Capture:
            def predict(self,obs,deterministic=True):seen.append(list(obs));return 0
        for name in ('MYOPIC','PPO'):
            self.api.run_portfolio(FakeTape(),0,STEP,kind=name,policy=Capture())
        self.assertEqual(seen[0],seen[1]);self.assertEqual(len(seen[0]),27)
    def test_RDE10_cash_and_unmanaged_buy_hold_include_all_days(self):
        from binance_demo_bot.rl_direction_core import day_bounds
        start,end=day_bounds('2026-10-25')
        cash=self.api.run_portfolio(FakeTape(),start,end,kind='CASH')
        hold=self.api.buy_hold(FakeTape(),start,end)
        self.assertEqual(cash['daily_pnl'],{'2026-10-25':0.})
        self.assertEqual(hold['complete_days_n'],1);self.assertLess(hold['net_pnl_usdt'],0)
        self.assertAlmostEqual(hold['open_exposure_usdt']+hold['cash_usdt'],100+hold['net_pnl_usdt'])

    def test_RDE12_study_registration_binds_population_split_sources(self):
        manifest=dict(symbols=['AAAUSDT'],data_sha256='b'*64,global_first_ms=0,cutoff_ms=100*STEP)
        row=self.api.study_registration(manifest,{'x':'a'*64},{'versions':{}},attempt='TEST')
        self.assertFalse(row['promotion_allowed']);self.assertEqual(row['source_sha256'],{'x':'a'*64})
        self.assertEqual(row['holdout_kind'],'RETROSPECTIVE_REUSED_HOLDOUT')
        self.assertEqual(row['seeds'],[11,23,37,53,71]);self.assertEqual(row['symbols'],['AAAUSDT'])
        self.assertEqual(row['policy_steps_per_seed'],131072)
        self.assertEqual(row['splits'][1][0],'tune')


if __name__=='__main__':unittest.main()
