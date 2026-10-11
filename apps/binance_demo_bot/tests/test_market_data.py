"""P2A-04..06: causal quotes/bars/features and exact portfolio marks."""

import unittest
from decimal import Decimal, localcontext

from _p2_support import Fixture, NOW, pair, book, candle


class MarketDataTests(Fixture, unittest.TestCase):
    def setUp(self):
        self.setup_market()

    def test_P2A04_quote_has_null_exchange_time_explicit_observation_basis(self):
        self.refresh()
        result = self.quotes()
        quote = result["data"]["rows"][0]["quote"]
        self.assertIsNone(quote["event_time_ms"])
        self.assertEqual(quote["time_basis"], "OBSERVED_RESPONSE")
        self.assertTrue(quote["exchange_age_unknown"])
        self.assertEqual(quote["received_at_ms"], NOW)

    def test_P2A04_omitted_expected_pair_not_full_coverage(self):
        universe = self.refresh([pair(), pair("BBB")])
        self.quotes([book()], universe)
        result = self.service.availability(universe, NOW)
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["quote_ready_n"], 1)
        self.assertEqual(result["permitted_pairs_N"], 2)
        self.assertIn("QUOTE_MISSING", result["rows"][1]["reasons"])
        self.assertEqual(result["feature_ready_n"], 0)

    def test_P2A04_zero_crossed_bad_quote_rows_do_not_erase_good_pair(self):
        universe = self.refresh([pair(), pair("BBB"), pair("CCC"), pair("DDD")])
        self.quotes([book(), book("BBBUSDT", "0", "0"), book("CCCUSDT", "102", "100"), book("DDDUSDT", bidPrice="NaN")], universe)
        result = self.service.availability(universe, NOW)
        self.assertEqual(result["quote_ready_n"], 1)
        self.assertEqual(len(result["rows"]), 4)

    def test_P2A04_invalid_newer_quote_does_not_resurrect_previous(self):
        universe = self.refresh()
        self.quotes()
        self.advance(1)
        self.quotes([book(bid="0", ask="0")], universe)
        self.assertEqual(self.service.availability(universe, self.wall)["quote_ready_n"], 0)
        self.assertEqual(self.service.availability(universe, NOW)["quote_ready_n"], 1)

    def test_P2A04_quote_age_and_metadata_age_independent(self):
        universe = self.refresh()
        self.quotes()
        self.advance(5001)
        self.assertEqual(self.service.availability(universe, self.wall)["quote_ready_n"], 0)
        self.advance(900000)
        self.quotes(universe=universe)
        reasons = self.service.availability(universe, self.wall)["rows"][0]["reasons"]
        self.assertIn("METADATA_STALE", reasons)

    def test_P2A04_disabled_collection_zero_requests(self):
        universe = self.refresh()
        service = self.market.MarketDataService(self.store, self.client, enabled=False)
        before = len(self.calls)
        self.assert_reason("MARKET_DATA_DISABLED", service.poll_quotes, universe)
        self.assertEqual(len(self.calls), before)

    def test_P2A04_public_reference_coverage_denominator_null(self):
        universe = self.refresh(account=None)
        self.quotes(universe=universe)
        result = self.service.availability(universe, NOW)
        self.assertIsNone(result["permitted_pairs_N"])
        self.assertIsNone(result["quote_ready_ratio"])

    def test_P2A04_fair_bounded_candle_rotation(self):
        universe = self.refresh([pair("AAA"), pair("BBB"), pair("CCC")])
        for _ in range(2):
            self.replies.extend([[], []])
            self.service.poll_candles(universe, max_symbols=2)
        from urllib.parse import parse_qs
        symbols = [parse_qs(call[2])["symbol"][0] for call in self.calls if call[1] == "/api/v3/klines"]
        self.assertEqual(symbols[:3], ["AAAUSDT", "BBBUSDT", "CCCUSDT"])

    def test_P2A05_exact_closed_bar_feature_and_dependency_receipts(self):
        self.refresh()
        self.quotes()
        self.bars()
        value = self.service.feature("AAAUSDT", NOW)
        self.assertEqual(value["data"]["state"], "FEATURE_READY")
        self.assertEqual(Decimal(value["data"]["values"]["close_return"]), Decimal("0.02"))
        self.assertEqual(Decimal(value["data"]["values"]["quote_volume_sum"]), Decimal("3000"))
        self.assertEqual(value["data"]["values"]["trades_sum"], 30)
        self.assertTrue(value["dependencies"])
        self.assertEqual(len(value["data"]["definition_sha256"]), 64)

    def test_P2A05_open_candle_excluded_from_features(self):
        self.refresh()
        self.quotes()
        rows = [candle(NOW-180000), candle(NOW-120000, "101"), candle(NOW-60000, "102"), candle(NOW, "110")]
        observed = self.bars(rows)
        self.assertFalse(observed["data"]["rows"][-1]["closed"])
        self.assertEqual(Decimal(self.service.feature("AAAUSDT", NOW)["data"]["values"]["close_return"]), Decimal("0.02"))

    def test_P2A05_gap_or_short_history_not_interpolated(self):
        self.refresh()
        self.quotes()
        self.bars([candle(NOW-180000), candle(NOW-60000)])
        value = self.service.feature("AAAUSDT", NOW)["data"]
        self.assertEqual(value["state"], "FEATURE_UNAVAILABLE")
        self.assertIsNone(value["values"])

    def test_P2A05_duplicate_sort_and_conflicting_closed_revision(self):
        self.refresh()
        self.quotes()
        rows = [candle(NOW-60000, "102"), candle(NOW-180000), candle(NOW-120000, "101"), candle(NOW-60000, "102")]
        result = self.bars(rows)
        self.assertEqual(len(result["data"]["rows"]), 3)
        self.advance(1)
        self.bars([candle(NOW-60000, "103")])
        value = self.service.feature("AAAUSDT", self.wall)["data"]
        self.assertEqual(value["state"], "FEATURE_UNAVAILABLE")
        self.assertIn("CANDLE_CONFLICT", value["reasons"])

    def test_P2A05_late_download_never_backdates_availability(self):
        self.refresh()
        self.quotes()
        self.advance(1000)
        self.bars()
        value = self.service.feature("AAAUSDT", NOW)["data"]
        self.assertEqual(value["state"], "FEATURE_UNAVAILABLE")

    def test_P2A05_future_data_mutation_preserves_cutoff_prefix(self):
        self.refresh()
        self.quotes()
        self.bars()
        first = self.service.feature("AAAUSDT", NOW)["data"]
        self.advance(60000)
        self.bars([candle(NOW, "110")])
        second = self.service.feature("AAAUSDT", NOW)["data"]
        self.assertEqual(first["values"], second["values"])
        self.assertEqual(first["dependency_snapshot_ids"], second["dependency_snapshot_ids"])

    def test_P2A05_malformed_alignment_highlow_and_future_open_rejected(self):
        for row in (candle(NOW-60000+1), candle(NOW-60000, "130"), candle(NOW+60000)):
            self.replies.append([row])
            self.assert_reason("CANDLE_SCHEMA_INVALID", self.service.collect_candles, "AAAUSDT")

    def test_P2A05_open_revision_can_finalize_without_false_conflict(self):
        self.refresh()
        self.quotes()
        self.bars([candle(NOW, "101")])
        self.advance(60001)
        result = self.bars([candle(NOW, "102")])
        self.assertTrue(result["data"]["rows"][0]["closed"])
        self.assertTrue(result["data"]["rows"][0]["valid"])

    def test_P2A05_ambient_decimal_context_cannot_round_feature_math(self):
        self.refresh()
        self.quotes()
        self.bars()
        with localcontext() as context:
            context.prec = 1
            value = self.service.feature("AAAUSDT", NOW)["data"]
        self.assertEqual(Decimal(value["values"]["close_return"]), Decimal("0.02"))

    def test_P2A06_free_locked_direct_midpoint_equity(self):
        self.refresh()
        self.quotes()
        value = self.market.QuoteConversionGraph(self.store).value(
            {"AAA": {"free": "1", "locked": "2"}, "USDT": {"free": "100", "locked": "50"}}, NOW)
        self.assertEqual(Decimal(value["data"]["equity_usdt"]), Decimal("450"))
        self.assertEqual(value["data"]["held_assets_N"], 2)
        self.assertFalse(value["data"]["liquidation_certified"])

    def test_P2A06_inverse_uses_reciprocal_mid_not_mean_reciprocals(self):
        self.refresh([pair("USDT", "AAA")])
        # Finite 1/4 avoids claiming that P0's 50-digit rounded 1/3 * 3 is 1.
        self.quotes([book("USDTAAA", "2", "6")])
        value = self.market.QuoteConversionGraph(self.store).value({"AAA": {"free": "4", "locked": "0"}}, NOW)
        self.assertEqual(Decimal(value["data"]["equity_usdt"]), Decimal("1"))

    def test_P2A04_market_budget_reserves_account_headroom_before_dispatch(self):
        self.read_client._reservations.append((self.mono, 990))
        self.assert_reason("MARKET_RATE_BUDGET", self.client.read_exchange_info)
        self.assertEqual(self.calls, [])

    def test_P2A04_mutated_or_future_membership_rejected_before_collection(self):
        universe = self.refresh()
        universe["data"]["rows"][0]["base"] = "OTHER"
        self.assert_reason("UNIVERSE_SNAPSHOT_INVALID", self.service.poll_quotes, universe)
        self.assertEqual(len(self.calls), 1)
        self.advance(1000)
        future = self.refresh()
        self.assert_reason("UNIVERSE_NOT_AVAILABLE", self.service.availability, future, NOW)

    def test_P2A04_failed_refresh_closes_readiness_but_preserves_denominator(self):
        from binance_demo_bot.adapter import AdapterError
        universe = self.refresh()
        self.quotes()
        self.advance(1)
        self.replies.append(AdapterError("READ_UNAVAILABLE"))
        failed = self.universe_service.refresh()
        result = self.service.availability(failed, self.wall)
        self.assertEqual(result["permitted_pairs_N"], 1)
        self.assertEqual(result["quote_ready_n"], 0)
        self.assertIn("METADATA_REFRESH_FAILED", result["rows"][0]["reasons"])

    def test_P2A05_failed_candle_receipt_closes_feature_until_recovery(self):
        self.refresh()
        self.quotes()
        self.bars()
        self.advance(1)
        self.replies.append([candle(NOW-60000, "130")])
        self.assert_reason("CANDLE_SCHEMA_INVALID", self.service.collect_candles, "AAAUSDT")
        value = self.service.feature("AAAUSDT", self.wall)["data"]
        self.assertEqual(value["state"], "FEATURE_UNAVAILABLE")
        self.assertIn("CANDLE_COLLECTION_FAILED", value["reasons"])
        self.advance(1)
        self.bars()
        self.assertEqual(self.service.feature("AAAUSDT", self.wall)["data"]["state"], "FEATURE_READY")

    def test_P2A06_multileg_fee_asset_and_stablecoin_not_assumed_parity(self):
        self.refresh([pair("AAA", "BTC"), pair("BTC", "USDT"), pair("USDC", "USDT")])
        self.quotes([book("AAABTC", "1", "3"), book("BTCUSDT", "99", "101"), book("USDCUSDT", "0.98", "1.00")])
        value = self.market.QuoteConversionGraph(self.store).value(
            {"AAA": {"free": "1", "locked": "0"}, "USDC": {"free": "100", "locked": "0"}}, NOW)
        self.assertEqual(Decimal(value["data"]["equity_usdt"]), Decimal("299"))

    def test_P2A06_direct_preferred_over_price_optimizing_multileg(self):
        self.refresh([pair(), pair("AAA", "BTC"), pair("BTC", "USDT")])
        self.quotes([book(), book("AAABTC", "9", "11"), book("BTCUSDT", "99", "101")])
        value = self.market.QuoteConversionGraph(self.store).value({"AAA": {"free": "1", "locked": "0"}}, NOW)
        self.assertEqual(Decimal(value["data"]["equity_usdt"]), Decimal("100"))
        self.assertEqual(value["data"]["routes"]["AAA"], [["AAAUSDT", False]])

    def test_P2A06_one_stale_leg_entire_equity_pending(self):
        self.refresh([pair("AAA", "BTC"), pair("BTC", "USDT")])
        self.quotes([book("AAABTC"), book("BTCUSDT")])
        self.advance(5001)
        value = self.market.QuoteConversionGraph(self.store).value({"AAA": {"free": "1", "locked": "0"}}, self.wall)
        self.assertEqual(value["data"]["status"], "PENDING")
        self.assertIsNone(value["data"]["equity_usdt"])

    def test_P2A06_rest_quote_cannot_certify_exchange_event_age(self):
        self.refresh()
        self.quotes()
        value = self.market.QuoteConversionGraph(self.store).value({"AAA": {"free": "1", "locked": "0"}}, NOW, time_basis="EXCHANGE_EVENT")
        self.assertIsNone(value["data"]["equity_usdt"])

    def test_P2A06_dust_missing_route_never_zero_value(self):
        self.refresh()
        self.quotes()
        value = self.market.QuoteConversionGraph(self.store).value({"DUST": {"free": "0.00000001", "locked": "0"}}, NOW)
        self.assertEqual(value["data"]["status"], "PENDING")
        self.assertIsNone(value["data"]["equity_usdt"])

    def test_P2A06_later_quote_cannot_repair_past_boundary(self):
        self.refresh()
        self.advance(1000)
        self.quotes()
        value = self.market.QuoteConversionGraph(self.store).value({"AAA": {"free": "1", "locked": "0"}}, NOW)
        self.assertIsNone(value["data"]["equity_usdt"])

    def test_P2A06_hypothetical_zero_portfolio_denominator_null(self):
        self.refresh()
        value = self.market.QuoteConversionGraph(self.store).value({"USDT": {"free": "0", "locked": "0"}}, NOW)
        self.assertEqual(value["data"]["held_assets_N"], 0)
        self.assertIsNone(value["data"]["valued_assets_ratio"])
