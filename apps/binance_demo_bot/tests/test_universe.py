"""P2A-02/03: test-first complete membership, provenance and isolation."""

import json
import sqlite3
import unittest
from contextlib import closing
from urllib.parse import parse_qs

from _p2_support import Fixture, NOW, pair, account_snapshot, metadata


class UniverseTests(Fixture, unittest.TestCase):
    def setUp(self):
        self.setup_market()

    def test_P2A02_explicit_all_categories_without_status_or_pilot_filter(self):
        snapshot = self.refresh([pair(), pair("BBB", "BTC")])
        query = parse_qs(self.calls[0][2])
        permissions = json.loads(query["permissions"][0])
        self.assertEqual(len(permissions), 27)
        self.assertIn("TRD_GRP_025", permissions)
        self.assertEqual(query["showPermissionSets"], ["true"])
        self.assertNotIn("symbolStatus", query)
        self.assertNotIn("symbols", query)
        self.assertFalse(self.calls[0][3].get("X-MBX-APIKEY"))
        self.assertEqual(snapshot["data"]["counts"]["permitted_pairs_N"], 2)

    def test_P2A02_and_between_groups_or_within_group(self):
        row = pair(permissionSets=[["SPOT"], ["TRD_GRP_002", "TRD_GRP_003"]])
        denied = self.refresh([row])["data"]["rows"][0]
        self.assertIs(denied["permitted"], False)
        value = self.refresh([row], account=account_snapshot(["SPOT", "TRD_GRP_003"]))
        self.assertIs(value["data"]["rows"][0]["permitted"], True)

    def test_usd01_verified_spot_satisfies_or_group_with_new_native_alternative(self):
        value=self.refresh([pair(permissionSets=[['SPOT','TRD_GRP_271']])])['data']
        self.assertTrue(value['rows'][0]['permitted'])
        self.assertEqual(value['state'],'UNIVERSE_READY')
        self.assertIn('TRD_GRP_271',value['rows'][0]['unknown_permission_alternatives'])

    def test_usd01_unknown_and_group_still_blocks_spot(self):
        value=self.refresh([pair(permissionSets=[['SPOT'],['TRD_GRP_271']])])['data']
        self.assertIsNone(value['rows'][0]['permitted'])
        self.assertEqual(value['state'],'UNIVERSE_INCOMPLETE')

    def test_usd01_account_update_time_uses_validated_exchange_clock(self):
        from _p1_support import source_hash
        self.clock.observe(NOW+12000,NOW,NOW,self.mono,self.mono)
        account=account_snapshot();account['data']['update_time_ms']=NOW+12000
        account['source_sha256']=source_hash(account['data'])
        value=self.refresh(account=account)['data']
        self.assertEqual(value['state'],'UNIVERSE_READY');self.assertTrue(value['rows'][0]['permitted'])

    def test_P2A02_non_usdt_monitoring_not_shrunk_by_preview_scope(self):
        value = self.refresh([pair(), pair("BBB", "BTC")])["data"]
        rows = {r["symbol"]: r for r in value["rows"]}
        self.assertTrue(rows["BBBBTC"]["permitted"])
        self.assertFalse(rows["BBBBTC"]["metadata_entry_candidate"])
        self.assertIn("OUTSIDE_PILOT_QUOTE", rows["BBBBTC"]["reasons"])
        self.assertEqual(value["counts"]["permitted_pairs_N"], 2)

    def test_P2A02_no_account_public_reference_does_not_invent_denominator(self):
        value = self.refresh(account=None)["data"]
        self.assertEqual(value["state"], "PUBLIC_REFERENCE")
        self.assertIsNone(value["counts"]["permitted_pairs_N"])
        self.assertIsNone(value["rows"][0]["permitted"])

    def test_P2A02_unknown_account_permission_incomplete(self):
        value = self.refresh(account=account_snapshot(["SPOT", "TRD_GRP_999"]))["data"]
        self.assertEqual(value["state"], "UNIVERSE_INCOMPLETE")
        self.assertIsNone(value["counts"]["permitted_pairs_N"])

    def test_P2A02_empty_missing_unknown_permission_sets_never_vacuous_true(self):
        rows = [pair("AAA", permissionSets=[]), pair("BBB", permissionSets=[["FUTURE"]]), pair("CCC")]
        del rows[2]["permissionSets"]
        value = self.refresh(rows)["data"]
        self.assertIsNone(value["counts"]["permitted_pairs_N"])
        self.assertTrue(all(r["permitted"] is None for r in value["rows"]))

    def test_P2A02_stale_wrong_scope_bad_hash_account_uncertified(self):
        for changes in ({"received_at_ms": NOW-31000, "request_started_at_ms": NOW-31001},
                        {"portfolio_scope_id": "legacy"}, {"source_sha256": "a"*64}):
            value = self.refresh(account=account_snapshot(**changes))["data"]
            self.assertEqual(value["state"], "UNIVERSE_INCOMPLETE")
            self.assertIsNone(value["counts"]["permitted_pairs_N"])

    def test_P2A03_halt_break_cancel_unknown_kept_but_no_entry(self):
        rows = [pair(str(i), status=s) for i, s in enumerate(("HALT", "BREAK", "CANCEL_ONLY", "FUTURE"))]
        value = self.refresh(rows)["data"]
        self.assertEqual(len(value["rows"]), 4)
        self.assertFalse(any(r["metadata_entry_candidate"] for r in value["rows"]))
        self.assertFalse(any(r["execution_certified"] for r in value["rows"]))

    def test_P2A03_filter_steps_preserved_not_asset_precision(self):
        raw = pair(baseAssetPrecision=8)
        raw["filters"].append({"filterType": "FUTURE_FILTER", "limit": "123"})
        row = self.refresh([raw])["data"]["rows"][0]
        self.assertEqual(row["filters"][0]["tickSize"], "0.01")
        self.assertEqual(row["filters"][1]["stepSize"], "0.001")
        self.assertIn("FILTER_UNVERIFIED", row["reasons"])
        self.assertFalse(row["execution_certified"])

    def test_P2A03_conflicting_symbol_metadata_rejected(self):
        self.assert_reason("METADATA_CONFLICT", self.refresh, [pair(), pair(status="BREAK")])

    def test_P2A03_listing_status_and_removal_preserve_past_membership(self):
        first = self.refresh([pair(), pair("BBB")])
        self.advance(1000)
        second = self.refresh([pair(status="HALT"), pair("CCC")])
        rows = {r["symbol"]: r for r in second["data"]["rows"]}
        self.assertEqual(rows["BBBUSDT"]["status"], "REMOVED")
        kinds = {r["type"] for r in second["data"]["transitions"]}
        self.assertTrue({"ADDED", "STATUS_CHANGED", "REMOVED"}.issubset(kinds))
        self.assertEqual(self.universe_service.current(NOW)["snapshot_id"], first["snapshot_id"])
        self.assertEqual(len(first["data"]["rows"]), 2)

    def test_P2A03_failed_refresh_retains_original_denominator_as_stale(self):
        first = self.refresh([pair(), pair("BBB")])
        from binance_demo_bot.adapter import AdapterError
        self.replies.append(AdapterError("READ_UNAVAILABLE"))
        failed = self.universe_service.refresh(account_snapshot())
        self.assertEqual(failed["data"]["state"], "UNIVERSE_REFRESH_FAILED")
        self.assertEqual(failed["data"]["counts"]["permitted_pairs_N"], 2)
        self.assertEqual(self.store.as_of("UNIVERSE", NOW)["snapshot_id"], first["snapshot_id"])

    def test_P2A03_store_own_namespace_single_writer_and_wal_full(self):
        self.assertEqual(self.store.sqlite_settings(), {"journal_mode": "wal", "synchronous": 2, "foreign_keys": 1})
        with self.assertRaises(Exception):
            self.market.MarketSnapshotStore(self.paths, account_alias="demo-one", episode_id="episode-one", scope_id="whole-account")

    def test_P2A03_store_rows_immutable_and_tamper_reopen_fails(self):
        self.refresh()
        self.store.close()
        with closing(sqlite3.connect(self.paths.runtime_path("market.sqlite"))) as connection, connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE snapshots SET payload_json='{}'")
            connection.execute("DROP TRIGGER immutable_snapshots_UPDATE")
            connection.execute("UPDATE snapshots SET payload_json='{}'")
        self.assert_reason("MARKET_INTEGRITY_FAILED", self.market.MarketSnapshotStore, self.paths,
                           account_alias="demo-one", episode_id="episode-one", scope_id="whole-account")

    def test_P2A03_caller_mutation_cannot_change_saved_snapshot(self):
        result = self.refresh()
        result["data"]["rows"].clear()
        self.assertEqual(len(self.universe_service.current(NOW)["data"]["rows"]), 1)

    def test_P2A03_dependency_missing_or_future_not_publishable_early(self):
        self.assert_reason("DEPENDENCY_MISSING", self.store.append, "FEATURE", {},
                           received_at_ms=NOW, received_monotonic_ns=self.mono, dependencies=["missing"])
        self.advance(1000)
        source = self.store.append("QUOTES", {}, received_at_ms=self.wall, received_monotonic_ns=self.mono)
        value = self.store.append("FEATURE", {}, received_at_ms=NOW, received_monotonic_ns=self.mono,
                                  dependencies=[source["snapshot_id"]])
        self.assertGreaterEqual(value["available_at_ms"], NOW+1000)
        self.assertIsNone(self.store.as_of("FEATURE", NOW))

    def test_P2A03_reopen_wrong_episode_fails_no_migration(self):
        self.refresh()
        self.store.close()
        self.assert_reason("MARKET_IDENTITY_MISMATCH", self.market.MarketSnapshotStore, self.paths,
                           account_alias="demo-one", episode_id="different", scope_id="whole-account")

    def test_P2A02_whitelisted_account_provenance_and_episode_token_binding(self):
        supplied = account_snapshot(password="must-not-be-retained")
        first = self.refresh(account=supplied)
        provenance = first["data"]["account_observation"]
        self.assertNotIn("password", provenance)
        self.assertEqual(provenance["source_sha256"], supplied["source_sha256"])
        changed = account_snapshot()
        changed["data"]["account_identity"] = "f"*64
        from _p1_support import source_hash
        changed["source_sha256"] = source_hash(changed["data"])
        value = self.refresh(account=changed)["data"]
        self.assertEqual(value["state"], "UNIVERSE_INCOMPLETE")
        self.assertIsNone(value["counts"]["permitted_pairs_N"])

    def test_P2A03_metadata_schema_failure_durably_closes_current_readiness(self):
        self.refresh()
        self.advance(1)
        self.replies.append({"timezone":"FUTURE", "serverTime":NOW, "symbols":[]})
        self.assert_reason("METADATA_SCHEMA_INVALID", self.universe_service.refresh)
        self.assertEqual(self.universe_service.current(self.wall)["data"]["state"], "UNIVERSE_REFRESH_FAILED")

    def test_P2A02_account_decimal_source_hash_survives_persistence(self):
        from decimal import Decimal
        from binance_demo_bot.adapter import _sha
        supplied = account_snapshot()
        supplied["data"]["balances"] = [{"asset":"USDT", "free":Decimal("1.00000000"), "locked":Decimal("0.00000000")}]
        supplied["source_sha256"] = _sha(supplied["data"])
        saved = self.refresh(account=supplied)["data"]["account_observation"]
        self.assertEqual(_sha(saved["data"]), supplied["source_sha256"])
