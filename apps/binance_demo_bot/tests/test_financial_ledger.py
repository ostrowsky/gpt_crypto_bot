"""P0-A fill/FIFO contracts; durable journal and OMS are separate milestones."""

from _support import D, FinancialContractCase, fill, lot


class FinancialLedgerTests(FinancialContractCase):
    def ledger(self, *extra_lots):
        return self.api.FinancialLedger(
            episode_id="synthetic-episode",
            opening_lots=[lot("USDT", "1000", "1000"), *extra_lots])

    def test_FIN_01_partial_fills_and_partial_exit_use_fifo(self):
        ledger = self.ledger()
        self.assertEqual(ledger.apply_fill(fill("t1"), {"USDT": "1", "AAA": "100"}), "APPLIED")
        self.assertEqual(ledger.apply_fill(
            fill("t2", quantity="1", price="120", quote_quantity="120"),
            {"USDT": "1", "AAA": "120"}), "APPLIED")
        self.assertEqual(ledger.apply_fill(
            fill("t3", side="SELL", quantity="1", price="130", quote_quantity="130", fee="0.5"),
            {"USDT": "1", "AAA": "130"}), "APPLIED")
        self.assertEqual(ledger.balances()["USDT"], D("809.5"))
        self.assertEqual(ledger.balances()["AAA"], D("2"))
        self.assertEqual(ledger.managed_quantity("AAA"), D("2"))
        attribution = ledger.attribution({"USDT": "1", "AAA": "130"})
        self.assertEqual(attribution["gross_realized_usdt"], D("30"))
        self.assertEqual(attribution["fee_expenses_usdt"], D("0.5"))
        self.assertEqual(attribution["unrealized_usdt"], D("40"))

    def test_FIN_02_quote_fee_is_debited_once(self):
        ledger = self.ledger()
        ledger.apply_fill(fill(fee="1"), {"USDT": "1", "AAA": "100"})
        self.assertEqual(ledger.balances()["USDT"], D("799"))
        self.assertEqual(ledger.balances()["AAA"], D("2"))
        attribution = ledger.attribution({"USDT": "1", "AAA": "110"})
        self.assertEqual(attribution["gross_realized_usdt"], D("0"))
        self.assertEqual(attribution["fee_expenses_usdt"], D("1"))
        self.assertEqual(attribution["unrealized_usdt"], D("20"))

    def test_FIN_02_base_fee_reduces_managed_net_inventory(self):
        ledger = self.ledger()
        ledger.apply_fill(fill(fee="0.02", fee_asset="AAA"), {"USDT": "1", "AAA": "100"})
        self.assertEqual(ledger.balances()["USDT"], D("800"))
        self.assertEqual(ledger.balances()["AAA"], D("1.98"))
        self.assertEqual(ledger.managed_quantity("AAA"), D("1.98"))
        attribution = ledger.attribution({"USDT": "1", "AAA": "110"})
        self.assertEqual(attribution["gross_realized_usdt"], D("0"))
        self.assertEqual(attribution["fee_expenses_usdt"], D("2"))
        self.assertEqual(attribution["unrealized_usdt"], D("19.8"))

    def test_FIN_02_appreciated_bnb_fee_has_expense_and_disposal_profit(self):
        ledger = self.ledger(lot("BNB", "1", "10"))
        ledger.apply_fill(fill(fee="0.1", fee_asset="BNB"),
                          {"USDT": "1", "AAA": "100", "BNB": "20"})
        self.assertEqual(ledger.balances()["BNB"], D("0.9"))
        attribution = ledger.attribution({"USDT": "1", "AAA": "110", "BNB": "20"})
        self.assertEqual(attribution["gross_realized_usdt"], D("1"))
        self.assertEqual(attribution["fee_expenses_usdt"], D("2"))
        self.assertEqual(attribution["unrealized_usdt"], D("29"))
        # Starting equity 1010; final 800 + 220 + 18 = 1038, net +28.
        bridge = self.api.reconcile_pnl(
            net_pnl_usdt="28", gross_realized_usdt=attribution["gross_realized_usdt"],
            fee_expenses_usdt=attribution["fee_expenses_usdt"],
            unrealized_delta_usdt=attribution["unrealized_usdt"], tolerance_usdt="0")
        self.assertEqual(bridge["status"], "COMPLETE")
        self.assertEqual(bridge["residual_usdt"], D("0"))

    def test_FIN_02_non_usdt_quote_disposal_uses_its_fifo_basis(self):
        ledger = self.ledger(lot("ETH", "2", "200"))
        ledger.apply_fill(fill(quantity="1", price="0.5", quote_quantity="0.5",
                               symbol="AAAETH", quote_asset="ETH"),
                          {"USDT": "1", "AAA": "60", "ETH": "120"})
        self.assertEqual(ledger.balances()["ETH"], D("1.5"))
        self.assertEqual(ledger.balances()["AAA"], D("1"))
        attribution = ledger.attribution({"USDT": "1", "AAA": "70", "ETH": "120"})
        self.assertEqual(attribution["gross_realized_usdt"], D("10"))
        self.assertEqual(attribution["fee_expenses_usdt"], D("0"))
        self.assertEqual(attribution["unrealized_usdt"], D("40"))

    def test_FIN_07_rest_ws_duplicate_has_no_second_fill_or_fee(self):
        ledger = self.ledger()
        marks = {"USDT": "1", "AAA": "100"}
        ledger.apply_fill(fill(fee="1", source="WS", received_at_ms=1_010), marks)
        before = dict(ledger.balances())
        attribution_before = dict(ledger.attribution(marks))
        self.assertEqual(ledger.apply_fill(
            fill(fee="1", source="REST", received_at_ms=2_000), marks), "DUPLICATE")
        self.assertEqual(ledger.balances(), before)
        self.assertEqual(ledger.attribution(marks), attribution_before)

    def test_FIN_07_conflicting_trade_is_rejected_atomically(self):
        ledger = self.ledger()
        marks = {"USDT": "1", "AAA": "100"}
        ledger.apply_fill(fill(), marks)
        before = dict(ledger.balances())
        with self.assertRaises(self.api.LedgerConflict):
            ledger.apply_fill(fill(quantity="3", quote_quantity="300"), marks)
        self.assertEqual(ledger.balances(), before)
        self.assertEqual(ledger.managed_quantity("AAA"), D("2"))

    def test_FIN_07_trade_id_is_scoped_by_symbol(self):
        ledger = self.ledger()
        marks = {"USDT": "1", "AAA": "100", "BBB": "100"}
        ledger.apply_fill(fill(), marks)
        self.assertEqual(ledger.apply_fill(fill(symbol="BBBUSDT", base_asset="BBB"), marks), "APPLIED")
        self.assertEqual(ledger.balances()["AAA"], D("2"))
        self.assertEqual(ledger.balances()["BBB"], D("2"))
        self.assertEqual(ledger.balances()["USDT"], D("600"))

    def test_FIN_10_insufficient_fee_inventory_rolls_back_whole_fill_and_dedup_key(self):
        ledger = self.ledger()
        marks = {"USDT": "1", "AAA": "100", "BNB": "20"}
        before = dict(ledger.balances())
        with self.assertRaises(self.api.AccountingError):
            ledger.apply_fill(fill(fee="0.1", fee_asset="BNB"), marks)
        self.assertEqual(ledger.balances(), before)
        self.assertEqual(ledger.apply_fill(fill(), marks), "APPLIED")
        self.assertEqual(ledger.balances()["USDT"], D("800"))

    def test_FIN_10_oversell_is_rejected_without_mutation(self):
        ledger = self.ledger()
        marks = {"USDT": "1", "AAA": "100"}
        ledger.apply_fill(fill(), marks)
        before = dict(ledger.balances())
        with self.assertRaises(self.api.AccountingError):
            ledger.apply_fill(fill("t2", side="SELL", quantity="3", quote_quantity="300"), marks)
        self.assertEqual(ledger.balances(), before)
        self.assertEqual(ledger.managed_quantity("AAA"), D("2"))

    def test_FIN_10_inconsistent_fill_notional_is_rejected(self):
        ledger = self.ledger()
        before = dict(ledger.balances())
        with self.assertRaises(self.api.AccountingError):
            ledger.apply_fill(fill(quote_quantity="201"), {"USDT": "1", "AAA": "100"})
        self.assertEqual(ledger.balances(), before)

    def test_FIN_12_seed_inventory_is_valued_but_not_sell_permission(self):
        ledger = self.ledger(lot("AAA", "2", "200"))
        self.assertEqual(ledger.balances()["AAA"], D("2"))
        self.assertEqual(ledger.managed_quantity("AAA"), D("0"))
        self.assertEqual(ledger.attribution({"USDT": "1", "AAA": "110"})["unrealized_usdt"], D("20"))
        before = dict(ledger.balances())
        with self.assertRaises(self.api.AccountingError):
            ledger.apply_fill(fill(side="SELL"), {"USDT": "1", "AAA": "100"})
        self.assertEqual(ledger.balances(), before)

    def test_FIN_12_unattributed_fill_never_becomes_managed_inventory(self):
        ledger = self.ledger()
        ledger.apply_fill(fill(ownership="UNATTRIBUTED"), {"USDT": "1", "AAA": "100"})
        self.assertEqual(ledger.balances()["AAA"], D("2"))
        self.assertEqual(ledger.managed_quantity("AAA"), D("0"))

    def test_FIN_12_foreign_app_environment_or_episode_is_rejected(self):
        for changes in (dict(application_id="legacy_bot"),
                        dict(environment="BINANCE_SPOT_LIVE"), dict(episode_id="other-episode")):
            with self.subTest(changes=changes):
                ledger = self.ledger()
                before = dict(ledger.balances())
                with self.assertRaises(self.api.AccountingError):
                    ledger.apply_fill(fill(**changes), {"USDT": "1", "AAA": "100"})
                self.assertEqual(ledger.balances(), before)
