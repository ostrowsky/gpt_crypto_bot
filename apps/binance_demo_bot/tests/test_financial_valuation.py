"""P0-A valuation and clock contracts: FIN-01/04/05/06/10."""

from datetime import datetime

from _support import BOUNDARY, D, FinancialContractCase, quote


class DecimalContractTests(FinancialContractCase):
    def test_FIN_01_decimal_precision_is_preserved(self):
        value = self.api.decimal_amount("0.1000000000000000000000001")
        self.assertIsInstance(value, D)
        self.assertEqual(value, D("0.1000000000000000000000001"))
        self.assertEqual(self.api.decimal_amount(D("-2.5")), D("-2.5"))

    def test_FIN_10_binary_floats_bools_and_nonfinite_amounts_are_rejected(self):
        for value in (0.1, True, "NaN", "Infinity", "-Infinity", D("NaN")):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    self.api.decimal_amount(value)


class PortfolioValuationTests(FinancialContractCase):
    def value(self, balances, quotes=None, routes=None, **changes):
        options = dict(boundary_ms=BOUNDARY, time_basis="EXCHANGE_EVENT",
                       max_age_ms=5_000)
        options.update(changes)
        return self.api.value_portfolio(balances, quotes or {}, routes or {}, **options)

    def test_FIN_01_free_and_locked_are_counted_exactly_once(self):
        result = self.value(
            {"USDT": {"free": "500", "locked": "200"},
             "AAA": {"free": "1", "locked": "2"}},
            {"AAAUSDT": quote()}, {"AAA": [("AAAUSDT", False)]})
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["equity_usdt"], D("1000"))
        self.assertEqual(result["marks"]["AAA"], D("100"))

    def test_FIN_01_reservation_changes_neither_total_inventory_nor_equity(self):
        before = self.value({"USDT": {"free": "1000", "locked": "0"}})
        after = self.value({"USDT": {"free": "300", "locked": "700"}})
        self.assertEqual(before["status"], "COMPLETE")
        self.assertEqual(after["status"], "COMPLETE")
        self.assertEqual(before["equity_usdt"], after["equity_usdt"])

    def test_FIN_06_dust_without_a_market_is_pending_not_zero(self):
        result = self.value({"USDT": {"free": "1000", "locked": "0"},
                             "DUST": {"free": "0.00000001", "locked": "0"}})
        self.assertEqual(result["status"], "PENDING")
        self.assertIsNone(result["equity_usdt"])
        self.assertTrue(result["reason_codes"])

    def test_FIN_06_zero_inventory_requires_no_quote(self):
        result = self.value({"USDT": {"free": "1000", "locked": "0"},
                             "DUST": {"free": "0", "locked": "0"}})
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["equity_usdt"], D("1000"))

    def test_FIN_06_stablecoin_is_marked_at_market_not_assumed_one(self):
        result = self.value({"USDC": {"free": "100", "locked": "0"}},
                            {"USDCUSDT": quote(base="USDC", bid="0.97", ask="0.99")},
                            {"USDC": [("USDCUSDT", False)]})
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["equity_usdt"], D("98"))

    def test_FIN_06_two_leg_conversion_includes_fee_assets(self):
        result = self.value({"AAA": {"free": "3", "locked": "0"}},
                            {"AAABNB": quote(quote_asset="BNB", bid="1.9", ask="2.1"),
                             "BNBUSDT": quote(base="BNB", bid="9", ask="11")},
                            {"AAA": [("AAABNB", False), ("BNBUSDT", False)]})
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["equity_usdt"], D("60"))

    def test_FIN_06_inverse_uses_reciprocal_of_mid(self):
        result = self.value({"AAA": {"free": "1", "locked": "0"}},
                            {"USDTAAA": quote(base="USDT", quote_asset="AAA",
                                              bid="0.2", ask="0.3")},
                            {"AAA": [("USDTAAA", True)]})
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["equity_usdt"], D("4"))

    def test_FIN_06_one_stale_leg_invalidates_entire_equity(self):
        result = self.value({"AAA": {"free": "1", "locked": "0"}},
                            {"AAABNB": quote(quote_asset="BNB"),
                             "BNBUSDT": quote(base="BNB", event_time_ms=4_999)},
                            {"AAA": [("AAABNB", False), ("BNBUSDT", False)]})
        self.assertEqual(result["status"], "PENDING")
        self.assertIsNone(result["equity_usdt"])

    def test_FIN_06_freshness_boundary_is_inclusive(self):
        result = self.value({"AAA": {"free": "1", "locked": "0"}},
                            {"AAAUSDT": quote(event_time_ms=5_000)},
                            {"AAA": [("AAAUSDT", False)]})
        self.assertEqual(result["status"], "COMPLETE")

    def test_FIN_06_future_tick_cannot_repair_historical_boundary(self):
        result = self.value({"AAA": {"free": "1", "locked": "0"}},
                            {"AAAUSDT": quote(event_time_ms=10_001)},
                            {"AAA": [("AAAUSDT", False)]})
        self.assertEqual(result["status"], "PENDING")
        self.assertIsNone(result["equity_usdt"])

    def test_FIN_06_late_receipt_is_allowed_for_event_time_accounting_only(self):
        result = self.value({"AAA": {"free": "1", "locked": "0"}},
                            {"AAAUSDT": quote(received_at_ms=12_000)},
                            {"AAA": [("AAAUSDT", False)]})
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["equity_usdt"], D("100"))

    def test_FIN_06_observed_response_allows_unknown_tick_time(self):
        result = self.value({"AAA": {"free": "1", "locked": "0"}},
                            {"AAAUSDT": quote(event_time_ms=None)},
                            {"AAA": [("AAAUSDT", False)]}, time_basis="OBSERVED_RESPONSE")
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["equity_usdt"], D("100"))

    def test_FIN_06_observed_response_rejects_future_stale_unhealthy_intervals(self):
        cases = (
            dict(received_at_ms=10_001),
            dict(received_at_ms=4_999, response_started_at_ms=4_900),
            dict(response_started_at_ms=9_500, received_at_ms=9_100),
            dict(source_healthy=False),
        )
        for changes in cases:
            with self.subTest(changes=changes):
                result = self.value({"AAA": {"free": "1", "locked": "0"}},
                                    {"AAAUSDT": quote(event_time_ms=None, **changes)},
                                    {"AAA": [("AAAUSDT", False)]},
                                    time_basis="OBSERVED_RESPONSE")
                self.assertIn(result["status"], ("PENDING", "ERROR"))
                self.assertIsNone(result["equity_usdt"])

    def test_FIN_06_event_time_mode_does_not_invent_missing_tick_clock(self):
        result = self.value({"AAA": {"free": "1", "locked": "0"}},
                            {"AAAUSDT": quote(event_time_ms=None)},
                            {"AAA": [("AAAUSDT", False)]})
        self.assertEqual(result["status"], "PENDING")
        self.assertIsNone(result["equity_usdt"])

    def test_FIN_06_crossed_nonpositive_and_foreign_venue_quotes_are_errors(self):
        for changes in (dict(bid="102", ask="101"), dict(bid="0"),
                        dict(venue="BINANCE_SPOT_LIVE")):
            with self.subTest(changes=changes):
                result = self.value({"AAA": {"free": "1", "locked": "0"}},
                                    {"AAAUSDT": quote(**changes)},
                                    {"AAA": [("AAAUSDT", False)]})
                self.assertEqual(result["status"], "ERROR")
                self.assertIsNone(result["equity_usdt"])

    def test_FIN_06_conversion_route_cannot_cycle(self):
        result = self.value({"AAA": {"free": "1", "locked": "0"}},
                            {"AAABNB": quote(quote_asset="BNB")},
                            {"AAA": [("AAABNB", False), ("AAABNB", True)]})
        self.assertEqual(result["status"], "ERROR")
        self.assertIsNone(result["equity_usdt"])

    def test_FIN_10_negative_balance_is_error_even_if_net_quantity_positive(self):
        result = self.value({"USDT": {"free": "-1", "locked": "1001"}})
        self.assertEqual(result["status"], "ERROR")
        self.assertIsNone(result["equity_usdt"])


class LocalDayTests(FinancialContractCase):
    def test_FIN_05_dst_days_have_23_and_25_hours(self):
        for day, hours, start, end in (
            ("2026-03-29", 23, "2026-03-28T23:00:00+00:00", "2026-03-29T22:00:00+00:00"),
            ("2026-10-25", 25, "2026-10-24T22:00:00+00:00", "2026-10-25T23:00:00+00:00"),
        ):
            with self.subTest(day=day):
                actual_start, actual_end = self.api.day_bounds(day, "Europe/Budapest")
                self.assertEqual(actual_end - actual_start, hours * 3_600_000)
                self.assertEqual(actual_start, int(datetime.fromisoformat(start).timestamp() * 1000))
                self.assertEqual(actual_end, int(datetime.fromisoformat(end).timestamp() * 1000))

    def test_FIN_05_right_boundary_belongs_to_next_day_once(self):
        start, end = self.api.day_bounds("2026-10-25", "Europe/Budapest")
        next_start, _ = self.api.day_bounds("2026-10-26", "Europe/Budapest")
        self.assertEqual(end, next_start)
        self.assertEqual(self.api.day_for_timestamp(start, "Europe/Budapest"), "2026-10-25")
        self.assertEqual(self.api.day_for_timestamp(end - 1, "Europe/Budapest"), "2026-10-25")
        self.assertEqual(self.api.day_for_timestamp(end, "Europe/Budapest"), "2026-10-26")

    def test_FIN_05_invalid_dates_and_zones_fail_explicitly(self):
        for day, zone in (("2026-02-30", "Europe/Budapest"), ("2026-10-10", "invalid/zone")):
            with self.subTest(day=day, zone=zone):
                with self.assertRaises((ValueError, KeyError)):
                    self.api.day_bounds(day, zone)
