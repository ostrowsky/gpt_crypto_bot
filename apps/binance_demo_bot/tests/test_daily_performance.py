"""P0-A daily equity/flow/completeness contracts: FIN-03/04/08/09/10/11."""

from _support import D, FinancialContractCase, snapshot


class DailyPerformanceTests(FinancialContractCase):
    def performance(self, opening=None, closing=None, flows=None, **changes):
        options = dict(reconciliation_status="COMPLETE", economic_costs_usdt=None)
        options.update(changes)
        return self.api.daily_performance(
            opening if opening is not None else snapshot(),
            closing if closing is not None else snapshot("1010", boundary=86_400_000),
            flows if flows is not None else [], **options)

    def test_FIN_03_non_usdt_deposit_and_withdrawal_use_values_at_flow(self):
        flows = [dict(effective_ms=1_000, signed_usdt="200", status="COMPLETE", classification="DEPOSIT"),
                 dict(effective_ms=2_000, signed_usdt="-50", status="COMPLETE", classification="WITHDRAWAL")]
        result = self.performance(closing=snapshot("1170", boundary=86_400_000), flows=flows)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["net_external_flow_usdt"], D("150"))
        self.assertEqual(result["net_pnl_usdt"], D("20"))
        self.assertIsNone(result["daily_return"])

    def test_FIN_03_episode_reset_cannot_be_joined_into_positive_reward(self):
        result = self.performance(closing=snapshot("5000", boundary=86_400_000, episode_id="reset-episode"))
        self.assertEqual(result["status"], "ERROR")
        self.assertIsNone(result["net_pnl_usdt"])

    def test_FIN_03_unexplained_loss_is_not_invented_as_withdrawal(self):
        result = self.performance(closing=snapshot("500", boundary=86_400_000),
                                  reconciliation_status="ERROR")
        self.assertEqual(result["status"], "ERROR")
        self.assertIsNone(result["net_pnl_usdt"])
        self.assertTrue(result["reason_codes"])

    def test_FIN_03_flow_on_right_boundary_is_excluded(self):
        flows = [dict(effective_ms=0, signed_usdt="100", status="COMPLETE", classification="DEPOSIT"),
                 dict(effective_ms=86_400_000, signed_usdt="200", status="COMPLETE", classification="DEPOSIT")]
        result = self.performance(closing=snapshot("1100", boundary=86_400_000), flows=flows)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["net_external_flow_usdt"], D("100"))
        self.assertEqual(result["net_pnl_usdt"], D("0"))

    def test_FIN_04_no_trade_overnight_loss_remains_reward(self):
        result = self.performance(closing=snapshot("970", boundary=86_400_000))
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["net_pnl_usdt"], D("-30"))
        self.assertEqual(result["daily_return"], D("-0.03"))

    def test_FIN_04_certified_flat_usdt_day_has_zero_reward(self):
        result = self.performance(closing=snapshot("1000", boundary=86_400_000), economic_costs_usdt="0")
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["net_pnl_usdt"], D("0"))
        self.assertEqual(result["daily_return"], D("0"))
        self.assertEqual(result["economic_status"], "COMPLETE")
        self.assertEqual(result["economic_pnl_usdt"], D("0"))

    def test_FIN_08_zero_start_equity_has_null_return(self):
        result = self.performance(opening=snapshot("0"), closing=snapshot("0", boundary=86_400_000))
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["net_pnl_usdt"], D("0"))
        self.assertIsNone(result["daily_return"])

    def test_FIN_08_zero_net_flow_still_disables_simple_return(self):
        flows = [dict(effective_ms=1_000, signed_usdt="100", status="COMPLETE", classification="DEPOSIT"),
                 dict(effective_ms=2_000, signed_usdt="-100", status="COMPLETE", classification="WITHDRAWAL")]
        result = self.performance(flows=flows)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["net_pnl_usdt"], D("10"))
        self.assertIsNone(result["daily_return"])

    def test_FIN_08_internal_free_locked_transfer_is_not_external_flow(self):
        flows = [dict(effective_ms=1_000, signed_usdt="300", status="COMPLETE",
                      classification="INTERNAL_ACCOUNT_TRANSFER")]
        result = self.performance(flows=flows)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["net_external_flow_usdt"], D("0"))
        self.assertEqual(result["net_pnl_usdt"], D("10"))
        self.assertEqual(result["daily_return"], D("0.01"))

    def test_FIN_09_net_equity_already_includes_trading_fees(self):
        # Two units bought at 100, sold at 105, with 2 total actual fees:
        # final cash 1008. Proceeds 210 and fees must not become extra reward/debit.
        result = self.performance(closing=snapshot("1008", boundary=86_400_000))
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["net_pnl_usdt"], D("8"))

    def test_FIN_09_unbilled_ai_cost_is_pending_separately_from_net(self):
        result = self.performance()
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["net_pnl_usdt"], D("10"))
        self.assertEqual(result["economic_status"], "PENDING")
        self.assertIsNone(result["economic_pnl_usdt"])

    def test_FIN_09_confirmed_extra_account_cost_is_subtracted_once(self):
        result = self.performance(economic_costs_usdt="3")
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["net_pnl_usdt"], D("10"))
        self.assertEqual(result["economic_status"], "COMPLETE")
        self.assertEqual(result["economic_pnl_usdt"], D("7"))

    def test_FIN_09_negative_or_nonfinite_cost_is_rejected(self):
        for cost in ("-1", "NaN", 0.1):
            with self.subTest(cost=cost):
                with self.assertRaises(ValueError):
                    self.performance(economic_costs_usdt=cost)

    def test_FIN_10_scope_mismatch_cannot_produce_reward(self):
        result = self.performance(closing=snapshot("1010", boundary=86_400_000, scope_id="other-account"))
        self.assertEqual(result["status"], "ERROR")
        self.assertIsNone(result["net_pnl_usdt"])

    def test_FIN_10_incomplete_snapshots_flow_or_reconciliation_block_reward(self):
        cases = (
            dict(opening=snapshot(None, status="PENDING")),
            dict(closing=snapshot(None, boundary=86_400_000, status="PENDING")),
            dict(reconciliation_status="PENDING"),
            dict(flows=[dict(effective_ms=1_000, signed_usdt=None,
                            status="PENDING", classification="DEPOSIT")]),
        )
        for changes in cases:
            with self.subTest(changes=changes):
                result = self.performance(**changes)
                self.assertEqual(result["status"], "PENDING")
                self.assertIsNone(result["net_pnl_usdt"])
                self.assertIsNone(result["daily_return"])
                self.assertTrue(result["reason_codes"])

    def test_FIN_10_error_is_not_downgraded_to_pending(self):
        result = self.performance(opening=snapshot(None, status="PENDING"), reconciliation_status="ERROR")
        self.assertEqual(result["status"], "ERROR")
        self.assertIsNone(result["net_pnl_usdt"])

    def test_FIN_10_reversed_or_empty_boundaries_are_rejected(self):
        for boundary in (-1, 0):
            with self.subTest(boundary=boundary):
                with self.assertRaises(ValueError):
                    self.performance(closing=snapshot("1010", boundary=boundary))


class AttributionAndReturnTests(FinancialContractCase):
    def test_FIN_08_twr_uses_flow_separated_segments(self):
        # +10% before deposit; +10% after deposit, not a 142% capital return.
        result = self.api.time_weighted_return([
            dict(start_equity_usdt="1000", end_equity_usdt="1100"),
            dict(start_equity_usdt="2200", end_equity_usdt="2420")])
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["value"], D("0.21"))

    def test_FIN_08_twr_missing_boundary_or_zero_denominator_is_pending(self):
        for segments in ([], [dict(start_equity_usdt="0", end_equity_usdt="10")],
                         [dict(start_equity_usdt="1000", end_equity_usdt=None)]):
            with self.subTest(segments=segments):
                result = self.api.time_weighted_return(segments)
                self.assertEqual(result["status"], "PENDING")
                self.assertIsNone(result["value"])

    def test_FIN_10_pnl_bridge_has_signed_residual_and_frozen_tolerance(self):
        options = dict(net_pnl_usdt="69.5", gross_realized_usdt="30",
                       fee_expenses_usdt="0.5", unrealized_delta_usdt="40", tolerance_usdt="0")
        good = self.api.reconcile_pnl(**options)
        self.assertEqual(good["status"], "COMPLETE")
        self.assertEqual(good["residual_usdt"], D("0"))
        options["net_pnl_usdt"] = "69.49"
        bad = self.api.reconcile_pnl(**options)
        self.assertEqual(bad["status"], "ERROR")
        self.assertEqual(bad["residual_usdt"], D("-0.01"))
        options["tolerance_usdt"] = "0.01"
        self.assertEqual(self.api.reconcile_pnl(**options)["status"], "COMPLETE")

    def test_FIN_10_negative_bridge_tolerance_is_rejected(self):
        with self.assertRaises(ValueError):
            self.api.reconcile_pnl(net_pnl_usdt="0", gross_realized_usdt="0",
                                   fee_expenses_usdt="0", unrealized_delta_usdt="0", tolerance_usdt="-1")


class CalendarAggregationTests(FinancialContractCase):
    DAYS = ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04"]

    def report(self, day, pnl=None, status="COMPLETE", revision=1):
        return dict(day=day, revision=revision, status=status, net_pnl_usdt=pnl)

    def test_FIN_11_downtime_and_errors_are_disclosed_not_scored_zero(self):
        result = self.api.aggregate_latest_daily(self.DAYS, [
            self.report(self.DAYS[0], "10"), self.report(self.DAYS[1], "-30"),
            self.report(self.DAYS[3], status="ERROR")])
        self.assertEqual((result["N_calendar"], result["N_complete"],
                          result["N_pending"], result["N_error"]), (4, 2, 1, 1))
        self.assertEqual(result["mean_daily_net_pnl_usdt"], D("-10"))
        self.assertEqual(result["positive_days"], 1)
        self.assertEqual(result["positive_day_fraction"], {"n": 1, "N": 2, "value": D("0.5")})

    def test_FIN_11_recovered_losing_day_reverses_favorable_average(self):
        days = self.DAYS[:2]
        before = [self.report(days[0], "10"), self.report(days[1], status="PENDING")]
        old = self.api.aggregate_latest_daily(days, before)
        recovered = self.api.aggregate_latest_daily(days, [*before, self.report(days[1], "-30", revision=2)])
        self.assertEqual(old["mean_daily_net_pnl_usdt"], D("10"))
        self.assertEqual(old["N_pending"], 1)
        self.assertEqual(recovered["mean_daily_net_pnl_usdt"], D("-10"))
        self.assertEqual(recovered["N_pending"], 0)
        self.assertEqual(before[1]["status"], "PENDING")

    def test_FIN_11_latest_revision_wins_even_if_it_removes_positive_result(self):
        result = self.api.aggregate_latest_daily(self.DAYS[:1], [
            self.report(self.DAYS[0], status="ERROR", revision=2),
            self.report(self.DAYS[0], "100", revision=1)])
        self.assertEqual(result["N_error"], 1)
        self.assertEqual(result["N_complete"], 0)
        self.assertIsNone(result["mean_daily_net_pnl_usdt"])
        self.assertEqual(result["positive_day_fraction"], {"n": 0, "N": 0, "value": None})

    def test_FIN_04_zero_complete_days_have_null_not_zero_statistics(self):
        result = self.api.aggregate_latest_daily(self.DAYS, [])
        self.assertEqual(result["N_calendar"], 4)
        self.assertEqual(result["N_pending"], 4)
        self.assertIsNone(result["mean_daily_net_pnl_usdt"])
        self.assertIsNone(result["positive_day_fraction"]["value"])

    def test_FIN_11_duplicate_revision_is_rejected_not_arbitrarily_selected(self):
        with self.assertRaises(ValueError):
            self.api.aggregate_latest_daily(self.DAYS, [
                self.report(self.DAYS[0], "10"), self.report(self.DAYS[0], "-30")])

    def test_FIN_11_report_outside_manifest_is_rejected(self):
        with self.assertRaises(ValueError):
            self.api.aggregate_latest_daily(self.DAYS, [self.report("2026-10-05", "100")])

    def test_FIN_11_complete_with_missing_pnl_is_rejected(self):
        with self.assertRaises(ValueError):
            self.api.aggregate_latest_daily(self.DAYS, [self.report(self.DAYS[0])])
