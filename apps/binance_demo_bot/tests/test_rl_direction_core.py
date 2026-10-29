"""RDE04..06/RDE10: independent causal portfolio simulator contracts."""
import importlib
import unittest

import _support  # Own application source path; no credentials or legacy runtime.


STEP = 900_000


def candle(open=100., high=None, low=None, close=None):
    return dict(open=open, high=open if high is None else high,
                low=open if low is None else low, close=open if close is None else close)


class ResearchAccountTests(unittest.TestCase):
    def test_RDE04_resting_tp_gap_precedes_discretionary_exit(self):
        account=self.account(fee_bps=0,slippage_bps=0)
        self.step(account,0,2)
        result,_=self.step(account,STEP,3,candle(open=102,high=102,low=100,close=101))
        self.assertEqual(result['events'][-1]['reason'],'TP')
        self.assertAlmostEqual(result['events'][-1]['price'],100.5)

    def test_RDE04_resting_tp_prevents_unexecutable_gap_mark_and_phantom_risk_kill(self):
        account=self.account(fee_bps=0,slippage_bps=0)
        self.step(account,0,2)
        self.step(account,STEP,0,candle(open=150))
        self.assertAlmostEqual(account.peak_equity,100.15)
        self.assertAlmostEqual(account.max_drawdown,0.)
        self.assertFalse(account.entry_killed)

    def test_RDE04_gap_stop_precedes_exit_but_later_low_does_not(self):
        for bar,reason,price in ((candle(open=98),'SL',98.),
                                (candle(open=100,low=98),'EXIT',100.)):
            account=self.account(fee_bps=0,slippage_bps=0)
            self.step(account,0,2)
            result,_=self.step(account,STEP,3,bar)
            self.assertEqual(result['events'][-1]['reason'],reason)
            self.assertEqual(result['events'][-1]['price'],price)

    def setUp(self):
        self.api = importlib.import_module('binance_demo_bot.rl_direction_core')

    def account(self, **kwargs):
        return self.api.ResearchAccount(**kwargs)

    def step(self, account, at, action, bar=None, candidate='AAAUSDT', **kwargs):
        calls = []
        def lookup(symbol, origin):
            calls.append((symbol, origin))
            return candle() if bar is None else bar
        result = account.step(at, candidate, action, lookup, **kwargs)
        return result, calls

    def test_RDE04_flat_round_trip_has_exact_two_fees_and_adverse_slip(self):
        account = self.account()
        bought, _ = self.step(account, 0, 2)
        sold, _ = self.step(account, STEP, 3)
        quantity = 30 / 1.001 / 100.05
        expected = 70 + quantity * 99.95 * .999
        self.assertAlmostEqual(account.equity, expected, places=11)
        self.assertAlmostEqual((bought['reward'] + sold['reward']) * 100, expected - 100, places=11)
        report = account.report(0, 2 * STEP)
        self.assertEqual(report['trades_n'], 1)
        self.assertAlmostEqual(report['net_pnl_usdt'], expected - 100, places=11)
        self.assertGreater(report['fees_usdt'], 0)
        self.assertGreater(report['turnover_usdt'], 0)

    def test_RDE04_existing_buy_is_hold_and_never_double_spends(self):
        account = self.account()
        self.step(account, 0, 2)
        cash = account.cash
        quantity = account.position['quantity']
        result, _ = self.step(account, STEP, 2, candidate='BBBUSDT')
        self.assertEqual(account.cash, cash)
        self.assertEqual(account.position['quantity'], quantity)
        self.assertEqual(account.position['symbol'], 'AAAUSDT')
        self.assertEqual(result['events'], [])
        self.assertEqual(account.counters['invalid_actions_n'], 1)
        self.assertEqual(account.report(0, 2 * STEP)['max_entry_debit_usdt'], 30.)

    def test_RDE04_buy10_buy30_and_low_cash_minimum(self):
        for action, expected in ((1, 10.), (2, 30.)):
            account = self.account(fee_bps=0, slippage_bps=0)
            self.step(account, 0, action)
            self.assertAlmostEqual(account.position['debit_usdt'], expected)
            self.assertAlmostEqual(account.cash, 100 - expected)
        account = self.account(initial=4.)
        result, calls = self.step(account, 0, 2)
        self.assertIsNone(account.position)
        self.assertEqual(calls, [])
        self.assertEqual(account.counters['entry_vetoes_n'], 1)
        self.assertEqual(result['reward'], 0.)

    def test_RDE04_flat_exit_and_wait_do_not_read_future_prices(self):
        account = self.account()
        def forbidden(*args):
            self.fail('Flat WAIT/EXIT must not inspect the future candidate bar')
        for at, action in ((0, 0), (STEP, 3)):
            result = account.step(at, 'AAAUSDT', action, forbidden)
            self.assertTrue(result['valid'])
            self.assertEqual(result['reward'], 0.)
        self.assertEqual(account.counters['invalid_actions_n'], 1)

    def test_RDE04_entry_veto_does_not_lookup_selected_future(self):
        account = self.account()
        def forbidden(*args):
            self.fail('A known entry veto must not inspect future execution')
        result = account.step(0, 'AAAUSDT', 2, forbidden, allow_entry=False)
        self.assertTrue(result['valid'])
        self.assertIsNone(account.position)
        self.assertEqual(account.counters['entry_vetoes_n'], 1)

    def test_RDE04_stop_wins_same_bar_tp_sl_and_tp_has_no_market_slip(self):
        account = self.account(fee_bps=0, slippage_bps=0)
        result, _ = self.step(account, 0, 2, candle(high=101, low=99))
        self.assertEqual(result['events'][-1]['reason'], 'SL')
        self.assertAlmostEqual(result['events'][-1]['price'], 99.5)
        self.assertAlmostEqual(account.equity, 99.85)
        account = self.account(fee_bps=0, slippage_bps=5)
        result, _ = self.step(account, 0, 2, candle(high=101, low=100))
        self.assertEqual(result['events'][-1]['reason'], 'TP')
        self.assertAlmostEqual(result['events'][-1]['price'], 100.05 * 1.005)
        self.assertAlmostEqual(account.equity, 100.15)

    def test_RDE04_gap_stop_uses_worse_open_plus_adverse_slip(self):
        account = self.account(fee_bps=0, slippage_bps=5)
        self.step(account, 0, 2)
        result, _ = self.step(account, STEP, 0, candle(open=98, high=98.1, low=97.9, close=98))
        self.assertEqual(result['events'][-1]['reason'], 'SL')
        self.assertAlmostEqual(result['events'][-1]['price'], 98 * .9995)

    def test_RDE04_trailing_uses_prior_peak_and_does_not_sell_on_new_same_bar_high(self):
        account = self.account(fee_bps=0, slippage_bps=0, exit_mode='trailing')
        first, _ = self.step(account, 0, 2, candle(high=101, low=100, close=100.8))
        self.assertIsNotNone(account.position)
        self.assertEqual(len(first['events']), 1)
        second, _ = self.step(account, STEP, 0, candle(open=100.8, high=101, low=100.7, close=100.8))
        self.assertEqual(second['events'], [])
        last, _ = self.step(account, 2 * STEP, 0, candle(open=100.8, high=100.9, low=100.4, close=100.6))
        self.assertEqual(last['events'][-1]['reason'], 'TRAIL')
        self.assertAlmostEqual(last['events'][-1]['price'], 101 * .995)

    def test_RDE04_max_hold_is_sixteenth_close_without_seventeenth_lookup(self):
        account = self.account(fee_bps=0, slippage_bps=0)
        seen = []
        def lookup(symbol, at):
            seen.append(at)
            self.assertLess(at, 16 * STEP)
            return candle()
        for i in range(16):
            result = account.step(i * STEP, 'AAAUSDT', 2 if not i else 0, lookup)
        self.assertIsNone(account.position)
        self.assertEqual(result['events'][-1]['reason'], 'HOLDING_LIMIT')
        self.assertEqual(result['events'][-1]['at_ms'], 16 * STEP - 1)
        self.assertEqual(len(seen), 16)

    def test_RDE04_interval_kill_persists_across_midnight_but_existing_exits_continue(self):
        account = self.account(fee_bps=0, slippage_bps=0)
        _, midnight = self.api.day_bounds('2026-10-28')
        self.step(account, midnight - 2 * STEP, 2)
        result, _ = self.step(account, midnight - STEP, 0, candle(open=95, high=95.1, low=94.9, close=95))
        self.assertTrue(account.entry_killed)
        self.assertIsNone(account.position)
        result, calls = self.step(account, midnight, 2)
        self.assertTrue(result['entry_killed'])
        self.assertEqual(calls, [])
        self.assertEqual(account.counters['entry_vetoes_n'], 1)
        self.assertLess(account.equity, 99)

    def test_RDE04_drawdown_kill_keeps_profitable_position_exit_available(self):
        account = self.account(fee_bps=0, slippage_bps=0, exit_mode='learned')
        self.step(account, 0, 2, candle(high=110, low=100, close=110))
        self.step(account, STEP, 0, candle(open=110, high=110, low=106, close=106))
        self.assertTrue(account.entry_killed)
        self.assertIsNotNone(account.position)
        self.assertGreater(account.equity, 100)
        result, _ = self.step(account, 2 * STEP, 3, candle(open=106))
        self.assertIsNone(account.position)
        self.assertEqual(result['events'][-1]['reason'], 'EXIT')
        self.assertGreater(account.report(0, 3 * STEP)['net_pnl_usdt'], 0)

    def test_RDE04_no_same_bar_reentry_after_exit(self):
        account = self.account(fee_bps=0, slippage_bps=0)
        self.step(account, 0, 2)
        result, calls = self.step(account, STEP, 2, candle(high=101, low=100), candidate='BBBUSDT')
        self.assertIsNone(account.position)
        self.assertEqual(calls, [('AAAUSDT', STEP)])
        self.assertEqual([e['side'] for e in result['events']], ['SELL'])

    def test_RDE05_missing_selected_execution_invalidates_without_alternative(self):
        account = self.account()
        calls = []
        def lookup(symbol, at):
            calls.append((symbol, at))
            return None
        result = account.step(0, 'AAAUSDT', 2, lookup)
        self.assertFalse(result['valid'])
        self.assertEqual(calls, [('AAAUSDT', 0)])
        self.assertEqual(account.cash, 100.)
        report = account.report(0, STEP)
        self.assertEqual(report['state'], 'INCOMPLETE')
        self.assertIsNone(report['net_pnl_usdt'])
        self.assertIn('SELECTED_EXECUTION_MISSING', report['reasons'])
        self.assertEqual(report['daily_pnl'], {})

    def test_RDE05_missing_held_execution_invalidates_and_preserves_inventory(self):
        account = self.account()
        self.step(account, 0, 2)
        quantity = account.position['quantity']
        result = account.step(STEP, 'BBBUSDT', 3, lambda *args: None)
        self.assertFalse(result['valid'])
        self.assertEqual(account.position['quantity'], quantity)
        self.assertIn('HELD_EXECUTION_MISSING', account.report(0, 2 * STEP)['reasons'])

    def test_RDE05_malformed_execution_never_manufactures_earnings(self):
        for bad in (candle(open=0), candle(high=99, low=100), candle(close=float('nan'))):
            account = self.account()
            result, _ = self.step(account, 0, 2, bad)
            self.assertFalse(result['valid'])
            self.assertEqual(account.cash, 100.)
            self.assertIsNone(account.report(0, STEP)['net_pnl_usdt'])

    def test_RDE05_gap_in_processed_timeline_invalidates_instead_of_zero_day(self):
        account = self.account()
        self.step(account, 0, 0)
        result, _ = self.step(account, 2 * STEP, 0)
        self.assertFalse(result['valid'])
        self.assertIn('NON_CONSECUTIVE_STEP', account.report(0, 3 * STEP)['reasons'])

    def test_RDE06_observation_is_fixed27_same_market_portfolio_state(self):
        account = self.account(fee_bps=0, slippage_bps=0)
        calls = []
        def vectors(symbol, at):
            calls.append((symbol, at))
            return list(range(20))
        before = account.observation(0, 'AAAUSDT', vectors)
        self.assertEqual(len(before), 27)
        self.assertEqual(before[:20], list(range(20)))
        self.assertEqual(before[20:], [1., 0., 0., 0., 0., 1., 1.])
        self.step(account, 0, 2)
        after = account.observation(STEP, 'BBBUSDT', vectors, remaining=.5)
        self.assertEqual(calls[-1], ('AAAUSDT', STEP))
        self.assertEqual(after[20], .7)
        self.assertEqual(after[21], .3)
        self.assertEqual(after[22], 0.)
        self.assertEqual(after[23], 1 / 16)
        self.assertEqual(after[-1], .5)
        self.assertEqual(after, account.observation(STEP, 'BBBUSDT', vectors, remaining=.5))

    def test_RDE06_no_candidate_is_zero_market_and_bad_vectors_fail_closed(self):
        account = self.account()
        result = account.observation(0, None, lambda *args: self.fail('No symbol to lookup'))
        self.assertEqual(result[:20], [0.] * 20)
        for values in ([0.] * 19, [float('inf')] * 20):
            with self.assertRaises(ValueError):
                account.observation(0, 'AAAUSDT', lambda *args: values)

    def test_RDE06_observation_freezes_before_execution_suffix(self):
        account = self.account()
        vector = [0.] * 20
        frozen = account.observation(0, 'AAAUSDT', lambda *args: vector)
        vector[0] = 123.
        self.assertEqual(frozen[0], 0.)
        self.step(account, 0, 2, candle(open=200, high=200, low=200, close=200))
        self.assertEqual(frozen[20], 1.)

    def test_RDE06_reward_telescopes_with_unrealized_terminal_inventory(self):
        account = self.account(fee_bps=0, slippage_bps=0, exit_mode='learned')
        rewards = []
        for i in range(4):
            result, _ = self.step(account, i * STEP, 2 if not i else 0,
                                  candle(open=100 + i, high=101 + i, low=100 + i, close=101 + i))
            rewards.append(result['reward'])
        report = account.report(0, 4 * STEP)
        self.assertIsNotNone(account.position)
        self.assertAlmostEqual(report['open_exposure_usdt'], .3 * 104)
        self.assertAlmostEqual(sum(rewards) * 100, report['net_pnl_usdt'])
        self.assertAlmostEqual(report['net_pnl_usdt'], 1.2)
        self.assertEqual(report['trades_n'], 0)
        self.assertEqual(report['evidence_kind'], 'SIMULATION')
        self.assertFalse(report['promotion_allowed'])

    def test_RDE10_dst_complete_days_include_no_trade_days(self):
        for day, expected in (('2026-03-29', 92), ('2026-10-25', 100), ('2026-10-28', 96)):
            start, end = self.api.day_bounds(day)
            account = self.account()
            for at in range(start, end, STEP):
                account.step(at, None, 0, lambda *args: self.fail('Cash needs no quote'))
            report = account.report(start, end)
            self.assertEqual(report['daily_pnl'], {day: 0.})
            self.assertEqual(report['days'][day]['points_n'], expected)
            self.assertEqual(report['complete_days_n'], 1)
            self.assertEqual(report['requested_complete_days_N'], 1)
            self.assertEqual(report['profitable_days_n'], 0)

    def test_RDE10_partial_days_are_not_eligible_zero_days(self):
        start, end = self.api.day_bounds('2026-10-28')
        account = self.account()
        for at in range(start + STEP, end, STEP):
            account.step(at, None, 0, lambda *args: None)
        report = account.report(start + STEP, end)
        self.assertEqual(report['daily_pnl'], {})
        self.assertEqual(report['complete_days_n'], 0)
        self.assertEqual(report['partial_days_n'], 1)
        self.assertIsNone(report['mean_daily_net_pnl_usdt'])
        self.assertEqual(report['net_pnl_usdt'], 0.)

    def test_RDE10_daily_accounting_reconciles_to_interval_and_does_not_reset(self):
        start, end = self.api.day_bounds('2026-10-28')
        account = self.account(fee_bps=0, slippage_bps=0, exit_mode='learned')
        for i, at in enumerate(range(start, end, STEP)):
            action = 2 if i == 0 else 3 if i == 1 else 0
            account.step(at, 'AAAUSDT', action, lambda *args: candle(open=100.2, high=100.2, low=100.2))
        report = account.report(start, end)
        self.assertAlmostEqual(sum(report['daily_pnl'].values()), report['net_pnl_usdt'])
        self.assertEqual(report['complete_days_n'], 1)

    def test_RDE10_independent_event_ledger_and_reward_reconstruction(self):
        account = self.account(exit_mode='learned')
        rewards = []
        for i in range(80):
            price = 100 + (i % 5) * .05
            action = 2 if i % 7 == 0 else 3 if i % 7 == 2 else 0
            result, _ = self.step(account, i * STEP, action, candle(open=price))
            rewards.append(result['reward'])
        report = account.report(0, 80 * STEP)
        debits = sum(e['debit_usdt'] for e in report['events'] if e['side'] == 'BUY')
        proceeds = sum(e['proceeds_usdt'] for e in report['events'] if e['side'] == 'SELL')
        fees = sum(e['fee_usdt'] for e in report['events'])
        reconstructed_cash = 100 - debits + proceeds
        reconstructed_equity = reconstructed_cash + report['open_exposure_usdt']
        self.assertAlmostEqual(reconstructed_cash, account.cash, places=10)
        self.assertAlmostEqual(reconstructed_equity - 100, report['net_pnl_usdt'], places=10)
        self.assertAlmostEqual(sum(rewards) * 100, reconstructed_equity - 100, places=10)
        self.assertAlmostEqual(fees, report['fees_usdt'], places=10)

    def test_RDE10_absent_calendar_day_is_missing_without_zero_imputation(self):
        first, second = self.api.day_bounds('2026-10-27')
        _, end = self.api.day_bounds('2026-10-29')
        samples = [dict(origin_ms=at, equity_before=100., equity_after=100.)
                   for at in range(first, second, STEP)]
        result = self.api.aggregate_days(samples, first, end)
        self.assertEqual(result['requested_complete_days_N'], 3)
        self.assertEqual(result['complete_days_n'], 1)
        self.assertEqual(result['missing_days_n'], 2)
        self.assertEqual(result['daily_pnl'], {'2026-10-27': 0.})
        self.assertIsNone(result['days']['2026-10-28']['net_pnl_usdt'])

    def test_RDE04_contract_rejects_invalid_cost_action_time_or_mode(self):
        for kwargs in (dict(fee_bps=-1), dict(slippage_bps=float('nan')), dict(initial=0), dict(exit_mode='oracle')):
            with self.assertRaises(ValueError):
                self.account(**kwargs)
        account = self.account()
        for at, action in ((1, 0), (0, 4), (0, True)):
            with self.assertRaises(ValueError):
                account.step(at, None, action, lambda *args: None)
        with self.assertRaises(ValueError):
            account.observation(0, None, lambda *args: None, remaining=1.1)


if __name__ == '__main__':
    unittest.main()
