"""Additional spec-derived boundaries fixed before the financial implementation."""

from decimal import Inexact, localcontext

from _support import BOUNDARY, D, FinancialContractCase, fill, lot, quote


class FinancialEdgeContractTests(FinancialContractCase):
    def test_FIN_07_late_new_fill_requires_ordered_replay_without_mutation(self):
        ledger = self.api.FinancialLedger(episode_id="synthetic-episode",
                                          opening_lots=[lot("USDT", "1000", "1000")])
        marks = {"USDT": "1", "AAA": "100"}
        original = fill(exchange_event_time_ms=2_000)
        ledger.apply_fill(original, marks)
        before = dict(ledger.balances())
        with self.assertRaises(self.api.AccountingError):
            ledger.apply_fill(fill("t2", quantity="1", quote_quantity="100",
                                    exchange_event_time_ms=1_000), marks)
        self.assertEqual(ledger.balances(), before)
        self.assertEqual(ledger.managed_quantity("AAA"), D("2"))
        self.assertEqual(ledger.apply_fill(fill("t2", quantity="1", quote_quantity="100",
                                                exchange_event_time_ms=3_000), marks), "APPLIED")
        self.assertEqual(ledger.apply_fill(original, marks), "DUPLICATE")

    def test_FIN_12_fifo_cost_and_managed_exit_inventory_are_independent(self):
        ledger = self.api.FinancialLedger(episode_id="synthetic-episode", opening_lots=[
            lot("USDT", "1000", "1000"), lot("AAA", "2", "200")])
        ledger.apply_fill(fill(price="120", quote_quantity="240"), {"USDT": "1", "AAA": "120"})
        ledger.apply_fill(fill("t2", side="SELL", quantity="1", price="130", quote_quantity="130"),
                          {"USDT": "1", "AAA": "130"})
        self.assertEqual(ledger.balances()["AAA"], D("3"))
        self.assertEqual(ledger.managed_quantity("AAA"), D("1"))
        result = ledger.attribution({"USDT": "1", "AAA": "130"})
        self.assertEqual(result["gross_realized_usdt"], D("30"))
        self.assertEqual(result["unrealized_usdt"], D("50"))

    def test_FIN_01_ambient_decimal_precision_and_traps_do_not_change_equity(self):
        with localcontext() as ambient:
            ambient.prec = 3
            ambient.traps[Inexact] = True
            result = self.api.value_portfolio(
                {"AAA": {"free": "1.23456789", "locked": "0"}},
                {"AAAUSDT": quote()}, {"AAA": [("AAAUSDT", False)]},
                boundary_ms=BOUNDARY, time_basis="EXCHANGE_EVENT", max_age_ms=5_000)
            self.assertEqual(result["equity_usdt"], D("123.456789"))
            self.assertEqual(ambient.prec, 3)
            self.assertTrue(ambient.traps[Inexact])

    def test_FIN_11_unknown_completeness_status_cannot_be_scored(self):
        with self.assertRaises(ValueError):
            self.api.aggregate_latest_daily(["2026-10-01"], [
                dict(day="2026-10-01", revision=1, status="SUCCESS", net_pnl_usdt="100")])

    def test_FIN_10_invalid_fill_amounts_fail_atomically(self):
        for changes in (dict(quantity="-2"), dict(commission_quantity="-1"),
                        dict(price="NaN"), dict(quantity=2.0)):
            with self.subTest(changes=changes):
                ledger = self.api.FinancialLedger(episode_id="synthetic-episode",
                                                  opening_lots=[lot("USDT", "1000", "1000")])
                before = dict(ledger.balances())
                with self.assertRaises(self.api.AccountingError):
                    ledger.apply_fill(fill(**changes), {"USDT": "1", "AAA": "100"})
                self.assertEqual(ledger.balances(), before)
