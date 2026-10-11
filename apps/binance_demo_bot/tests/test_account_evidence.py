"""PB-04..07: real durable files, crash recovery and journal comparisons."""

import importlib
import json
import sqlite3
import subprocess
import sys
import unittest
from contextlib import closing
from decimal import Decimal

from _p1_support import (FixtureEnvironment, NOW, TOKEN, observation, recovery,
                         normalized_fill, normalized_event, normalized_order, list_data)
from binance_demo_bot.journal import JournalStore
from _support import lot


class AccountEvidenceTests(FixtureEnvironment, unittest.TestCase):
    def setUp(self):
        self.evidence = importlib.import_module("binance_demo_bot.account_evidence")
        self.set_fixture()

    def bind(self, session="session-one", generation="g1"):
        self.store.bind_session(session, 7, generation)

    def event(self, event=None, **changes):
        kwargs = dict(received_at_ms=NOW, received_monotonic_ns=10_000_000_000,
                      process_instance_id="fixture-process")
        kwargs.update(changes)
        return self.store.record_event("session-one", 7, event or normalized_event(), **kwargs)

    def journal(self, opening=None):
        journal = JournalStore(self.paths, episode_id="episode-one", scope_id="whole-account",
                               opening_lots=opening or [lot("USDT", "1000", "1000")])
        self.addCleanup(journal.close)
        return journal

    def account_anchor(self, balances=None, **changes):
        data = dict(balances=balances if balances is not None else
                    [dict(asset="USDT", free=Decimal("900"), locked=Decimal("100"))],
                    account_identity=TOKEN, can_trade=True, permissions=["SPOT"], update_time_ms=NOW-100)
        data.update(changes)
        return observation(data, "/api/v3/account")

    def compare(self, **changes):
        params = dict(account=self.account_anchor(), orders=observation([], "/api/v3/openOrders"),
                      lists=observation([], "/api/v3/openOrderList"), population_symbols=["BTCUSDT"],
                      population_complete=False, owned_client_ids=(), stream_health=None,
                      now_ms=NOW+100, journal_anchor_ms=NOW-5000)
        journal = changes.pop("journal", None) or self.journal()
        params.update(changes)
        return self.evidence.AccountReconciler(self.store).compare(journal, {"USDT": "1", "BTC": "100"}, **params)

    def test_PB04_namespace_and_wal_full(self):
        self.assertEqual(self.store.sqlite_settings(), {"journal_mode": "wal", "synchronous": 2, "foreign_keys": 1})
        self.close_store()
        self.assert_reason("EVIDENCE_IDENTITY_MISMATCH", self.reopen_store, account_identity="a"*64)

    def test_PB04_single_writer_lease(self):
        with self.assertRaises(Exception):
            self.evidence.AccountEvidenceStore(self.paths, account_alias="demo-one", account_identity=TOKEN,
                                              episode_id="episode-one", scope_id="whole-account")

    def test_PB04_session_mapping_conflict(self):
        self.bind()
        self.assert_reason("SESSION_CONFLICT", self.store.bind_session, "session-one", 8, "g1")

    def test_PB04_event_requires_mapped_session(self):
        self.assert_reason("UNMAPPED_SUBSCRIPTION", self.event)
        self.assertFalse(self.store.fills())

    def test_PB04_duplicate_rest_ws_fill_debits_once_preserves_receipts(self):
        self.bind()
        self.event()
        self.store.record_recovery(recovery())
        self.event(received_at_ms=NOW+1, received_monotonic_ns=10_001_000_000)
        self.assertEqual(len(self.store.fills()), 1)
        self.assertEqual(len(self.store.events()), 2)
        self.assertEqual(self.store.cursor("BTCUSDT"), 2)

    def test_PB04_conflicting_economics_block_and_preserve_previous(self):
        self.bind()
        self.event()
        self.assert_reason("FILL_CONFLICT", self.store.record_recovery,
                           recovery([normalized_fill(commission_quantity=Decimal("1"))]))
        self.assertEqual(self.store.cursor("BTCUSDT"), 0)
        self.assertEqual(len(self.store.fills()), 1)
        self.assertIn("FILL_CONFLICT", [i["data"]["reason"] for i in self.store.incidents()])

    def test_PB04_late_order_does_not_erase_later_cumulative_facts(self):
        self.bind()
        self.event(normalized_event(fill=None))
        early = normalized_event(kind="ORDER_UPDATE", transaction_time_ms=NOW-2000,
            exchange_event_time_ms=NOW-1900, execution_id=9,
            order=normalized_order(status="NEW", executed_quantity="0", cumulative_quote_quantity="0", update_time_ms=NOW-2000),
            fill=None)
        self.event(early, received_at_ms=NOW+1)
        self.assertEqual(self.store.order_states()[0]["executed_quantity"], "1")
        self.assertEqual(len(self.store.events()), 2)

    def test_PB04_credential_rotation_preserves_account_history(self):
        self.bind()
        self.event()
        self.store.bind_session("session-two", 7, "g2")
        self.close_store()
        self.reopen_store()
        self.assertEqual(len(self.store.fills()), 1)

    def test_PB05_recovery_overlapping_cursor_and_partial_tail(self):
        self.store.record_recovery(recovery([normalized_fill(1), normalized_fill(10)], next_id=11))
        self.assertEqual(self.store.recovery_start("BTCUSDT", overlap=5), 6)
        self.store.record_recovery(recovery([normalized_fill(10), normalized_fill(12)], start=6, next_id=13, exhausted=False))
        self.assertEqual(self.store.cursor("BTCUSDT"), 13)
        self.assertFalse(self.store.recovery_status("BTCUSDT")["tail_exhausted"])
        self.assertEqual(len(self.store.fills()), 3)

    def test_PB05_start_ahead_cursor_records_history_gap(self):
        self.store.record_recovery(recovery(start=100, fills=[], next_id=100))
        self.assertIn("HISTORY_GAP", [i["data"]["reason"] for i in self.store.incidents()])
        self.assertFalse(self.store.recovery_status("BTCUSDT")["origin_known"])

    def test_PB05_wrong_scope_and_hash_rejected_without_cursor(self):
        self.assert_reason("EVIDENCE_NAMESPACE_MISMATCH", self.store.record_recovery,
                           recovery(account_episode_id="old-episode"))
        self.assert_reason("EVIDENCE_HASH_MISMATCH", self.store.record_recovery,
                           recovery(source_sha256="a"*64))
        self.assertEqual(self.store.cursor("BTCUSDT"), 0)

    def test_PB05_cursor_and_fill_transaction_crash_before_after(self):
        self.close_store()
        for stage, expected in (("before_commit", 0), ("after_commit", 1)):
            options = dict(root=str(self.root), stage=stage,
                           data=recovery([normalized_fill()]))
            script = ("import json,os,sys; from binance_demo_bot.account_evidence import AccountEvidenceStore; "
                      "from binance_demo_bot.isolation import AppPaths; a=json.loads(sys.argv[1]); "
                      "s=AccountEvidenceStore(AppPaths(a['root']),account_alias='demo-one',"
                      f"account_identity='{TOKEN}',episode_id='episode-one',scope_id='whole-account',"
                      "fault_hook=lambda stage: os._exit(23) if stage==a['stage'] else None); "
                      "s.record_recovery(a['data'])")
            args = json.dumps(options, default=lambda x: format(x, "f"))
            result = subprocess.run([sys.executable, "-I", "-c", script, args], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 23, result.stderr)
            self.reopen_store()
            self.assertEqual(len(self.store.fills()), expected)
            self.assertEqual(self.store.cursor("BTCUSDT"), 2 if expected else 0)
            self.close_store()

    def test_PB05_immutable_sql_rows_and_tamper_detected_on_reopen(self):
        self.store.record_recovery(recovery())
        self.close_store()
        with closing(sqlite3.connect(self.paths.runtime_path("account-evidence.sqlite"))) as con, con:
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute("UPDATE records SET payload_json='{}'")
            con.execute("DROP TRIGGER immutable_records_UPDATE")
            con.execute("UPDATE records SET payload_json='{}'")
        self.assert_reason("EVIDENCE_INTEGRITY_FAILED", self.reopen_store)

    def test_PB06_free_locked_sum_matches_but_no_execution_permission(self):
        result = self.compare()
        self.assertEqual(result["balance_check"], "BALANCES_MATCH")
        self.assertEqual(result["matched_assets_n"], 1)
        self.assertEqual(result["assets_N"], 1)
        self.assertIsNone(result["exchange_fill_denominator"])
        self.assertFalse(result["exchange_reconciled"])
        self.assertFalse(result["order_send_enabled"])
        self.assertEqual(result["state"], "ACCOUNT_UNRECONCILED")

    def test_PB06_missing_expected_asset_is_not_zero(self):
        result = self.compare(account=self.account_anchor(balances=[]))
        self.assertEqual(result["assets"][0]["reason"], "ASSET_COVERAGE_MISSING")
        self.assertIsNone(result["assets"][0]["exchange_quantity"])
        self.assertEqual(result["matched_assets_n"], 0)

    def test_PB06_nonzero_dust_difference_preserved(self):
        result = self.compare(account=self.account_anchor(balances=[dict(asset="USDT", free=Decimal("1000.00000001"), locked=Decimal("0"))]))
        self.assertEqual(result["assets"][0]["delta"], Decimal("0.00000001"))
        self.assertEqual(result["balance_check"], "BALANCES_DIFFER")

    def test_PB06_foreign_open_orders_and_lists_block(self):
        orders = observation([normalized_order()], "/api/v3/openOrders")
        lists = observation([dict(order_list_id=31, list_client_order_id="manual-list",
                                 orders=[dict(order_id=21, client_order_id="manual-order", symbol="BTCUSDT")])], "/api/v3/openOrderList")
        result = self.compare(orders=orders, lists=lists)
        self.assertIn("FOREIGN_OPEN_ORDERS", result["reason_codes"])
        self.assertIn("FOREIGN_OPEN_LISTS", result["reason_codes"])

    def test_PB06_recovered_fill_missing_in_journal_detected_despite_matching_balance(self):
        self.store.record_recovery(recovery())
        result = self.compare()
        self.assertIn("JOURNAL_FILL_MISSING", result["reason_codes"])
        self.assertEqual(result["missing_journal_fills_n"], 1)

    def test_PB06_stale_future_wrong_account_and_namespace_rejected(self):
        journal = self.journal()
        for changes, reason in (({"received_at_ms": NOW-31000}, "ACCOUNT_SNAPSHOT_STALE"),
                                ({"received_at_ms": NOW+200}, "EVIDENCE_FROM_FUTURE"),
                                ({"account_episode_id": "foreign"}, "EVIDENCE_NAMESPACE_MISMATCH")):
            value = {**self.account_anchor(), **changes}
            if value["received_at_ms"] < NOW:
                value["request_started_at_ms"] = value["received_at_ms"]-10
            result = self.compare(account=value, journal=journal)
            self.assertIn(reason, result["reason_codes"])
        result = self.compare(account=self.account_anchor(account_identity="a"*64), journal=journal)
        self.assertIn("ACCOUNT_IDENTITY_MISMATCH", result["reason_codes"])

    def test_PB06_unclassified_flow_survives_reopen_and_matching_snapshot(self):
        self.store.record_incident("UNCLASSIFIED_FLOW")
        self.close_store()
        self.reopen_store()
        result = self.compare()
        self.assertIn("UNCLASSIFIED_FLOW", result["reason_codes"])

    def test_PB07_comparison_stale_on_new_source_not_financial_write(self):
        journal = self.journal()
        before = journal.projection({"USDT": "1"})
        result = self.compare(journal=journal)
        self.assertFalse(self.store.comparison_stale(result["comparison_id"]))
        self.store.record_incident("STREAM_GAP")
        self.assertTrue(self.store.comparison_stale(result["comparison_id"]))
        self.assertEqual(journal.projection({"USDT": "1"}), before)

    def test_PB07_zero_denominator_remains_null(self):
        journal = self.journal([lot("USDT", "0", "0")])
        result = self.compare(journal=journal, account=self.account_anchor(balances=[]), population_symbols=[])
        self.assertEqual(result["assets_N"], 0)
        self.assertIsNone(result["matched_assets_ratio"])
        self.assertIn("POPULATION_UNKNOWN", result["reason_codes"])

    def test_PB07_recover_all_explicit_symbols_not_only_holdings(self):
        self.rest_replies = [[], []]
        result = self.evidence.AccountReconciler(self.store).recover(self.adapter, ["ETHUSDT", "BTCUSDT"])
        self.assertEqual(len(result), 2)
        self.assertIn("symbol=BTCUSDT", self.rest_calls[0][2])
        self.assertIn("symbol=ETHUSDT", self.rest_calls[1][2])
        self.assertEqual(self.store.cursor("BTCUSDT"), 0)

    def test_PB07_reject_raw_secret_incident_text(self):
        error = self.assert_reason("INVALID_INCIDENT", self.store.record_incident, "synthetic-secret")
        self.assertNotIn("synthetic-secret", str(error))

    def test_PB04_cross_kind_fill_cannot_enter_through_balance_event(self):
        self.bind()
        event = dict(kind="PARTIAL_BALANCES", exchange_event_time_ms=NOW-100,
                     update_time_ms=NOW-100, balances=[], fill=normalized_fill())
        self.assert_reason("SCHEMA_INVALID", self.event, event)
        self.assertFalse(self.store.fills())

    def test_PB04_terminal_equal_time_conflict_blocks_arrival_overwrite(self):
        self.bind()
        self.event(normalized_event(fill=None, order=normalized_order(status="FILLED", original_quantity="1")))
        self.assert_reason("ORDER_STATE_CONFLICT", self.event,
                           normalized_event(fill=None, execution_id=11,
                           order=normalized_order(status="PARTIALLY_FILLED", original_quantity="1")))
        self.assertIn("ORDER_STATE_CONFLICT", [i["data"]["reason"] for i in self.store.incidents()])

    def test_PB05_arbitrary_next_cursor_cannot_skip_unobserved_history(self):
        self.assert_reason("PAGINATION_INVALID", self.store.record_recovery,
                           recovery(next_id=9999))
        self.assertEqual(self.store.cursor("BTCUSDT"), 0)

    def test_PB06_invalid_hash_cannot_yield_favorable_balance_score(self):
        result = self.compare(account={**self.account_anchor(), "source_sha256": "a"*64})
        self.assertIn("EVIDENCE_HASH_MISMATCH", result["reason_codes"])
        self.assertEqual(result["balance_check"], "UNKNOWN")
        self.assertIsNone(result["matched_assets_ratio"])

    def test_PB06_duplicate_account_rows_rejected_not_last_row_wins(self):
        rows = [dict(asset="USDT", free=Decimal("1"), locked=Decimal("0")),
                dict(asset="USDT", free=Decimal("1000"), locked=Decimal("0"))]
        result = self.compare(account=self.account_anchor(balances=rows))
        self.assertIn("SCHEMA_INVALID", result["reason_codes"])
        self.assertEqual(result["balance_check"], "UNKNOWN")

    def test_PB07_event_during_check_makes_report_stale(self):
        from unittest.mock import patch
        journal = self.journal()
        original = journal.projection
        def changed(marks):
            self.store.record_incident("STREAM_GAP")
            return original(marks)
        with patch.object(journal, "projection", side_effect=changed):
            result = self.compare(journal=journal)
        self.assertTrue(self.store.comparison_stale(result["comparison_id"]))

    def test_PB07_current_journal_hash_required_for_financial_freshness(self):
        from _support import fill
        from test_durable_journal import envelope
        journal = self.journal()
        result = self.compare(journal=journal)
        event = envelope("bridge-1", exchange=NOW-1000, received=NOW,
                         account_episode_id="episode-one", portfolio_scope_id="whole-account")
        payload = fill(trade="1", episode_id="episode-one", symbol="BTCUSDT", order_id="21", base_asset="BTC",
                       quantity="1", quote_quantity="100", price="100", ownership="UNATTRIBUTED", exchange_event_time_ms=NOW-1000)
        journal.append_fill(event, payload, {"USDT": "1", "BTC": "100"})
        self.assertTrue(self.store.comparison_stale(result["comparison_id"], journal=journal, marks={"USDT": "1", "BTC": "100"}))

    def test_PB07_actual_journal_fill_and_third_asset_fee_match(self):
        from _support import fill
        from test_durable_journal import envelope
        journal = self.journal([lot("USDT", "1000", "1000"), lot("BNB", "1", "1")])
        event = envelope("bridge-1", exchange=NOW-1000, received=NOW,
                         account_episode_id="episode-one", portfolio_scope_id="whole-account")
        payload = fill(trade="1", episode_id="episode-one", symbol="BTCUSDT", order_id="21", base_asset="BTC",
                       quantity="1", quote_quantity="100", price="100", fee="0.001", fee_asset="BNB",
                       ownership="UNATTRIBUTED", exchange_event_time_ms=NOW-1000)
        journal.append_fill(event, payload, {"USDT": "1", "BTC": "100", "BNB": "1"})
        self.store.record_recovery(recovery())
        balances = [dict(asset=a, free=Decimal(q), locked=Decimal("0")) for a, q in
                    (("USDT", "900"), ("BTC", "1"), ("BNB", "0.999"))]
        result = self.evidence.AccountReconciler(self.store).compare(journal, {"USDT": "1", "BTC": "100", "BNB": "1"},
                    account=self.account_anchor(balances), orders=observation([], "/api/v3/openOrders"),
                    lists=observation([], "/api/v3/openOrderList"), population_symbols=["BTCUSDT"],
                    now_ms=NOW+100, journal_anchor_ms=NOW-5000)
        self.assertEqual(result["missing_journal_fills_n"], 0)
        self.assertEqual(result["conflicting_journal_fills_n"], 0)
        self.assertEqual(result["matched_assets_n"], 3)

    def test_PB05_recovery_page_generation_mismatch_rejected(self):
        value = recovery()
        value["page_observations"][0]["credential_generation"] = "foreign"
        self.assert_reason("EVIDENCE_NAMESPACE_MISMATCH", self.store.record_recovery, value)
        self.assertFalse(self.store.fills())

    def test_PB06_malformed_orders_do_not_report_zero_foreign_activity(self):
        result = self.compare(orders=observation(None, "/api/v3/openOrders"))
        self.assertIn("SCHEMA_INVALID", result["reason_codes"])
        self.assertIsNone(result["foreign_open_orders_n"])

    def test_PB06_future_journal_anchor_cannot_hide_missing_fills(self):
        self.assert_reason("INVALID_JOURNAL_ANCHOR", self.compare, journal_anchor_ms=NOW+200)

    def test_PB04_durable_events_have_program_envelope_not_fabricated_model(self):
        self.bind()
        self.event()
        value = self.store.events()[0]
        self.assertEqual(value["schema_version"], 1)
        self.assertEqual(value["application_id"], "binance_demo_bot")
        self.assertEqual(value["environment"], "BINANCE_SPOT_DEMO")
        self.assertEqual(value["account_episode_id"], "episode-one")
        self.assertEqual(value["portfolio_scope_id"], "whole-account")
        self.assertEqual(value["event_id"], value["id"])
        self.assertEqual(value["event_type"], "PrivateAccountEvidenceRecorded")
        self.assertEqual(value["process_instance_id"], "fixture-process")
        self.assertEqual(value["exchange_event_time_ms"], NOW-900)
        self.assertEqual(len(value["source_sha256"]), 64)
        for key in ("causation_id", "model_id", "policy_version", "arm_id", "risk_contract_id"):
            self.assertIsNone(value[key])


class OpenListAdapterTests(unittest.TestCase):
    def setUp(self):
        from test_adapter import AdapterCase
        self.helper = AdapterCase()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)

    def test_PB06_signed_open_lists_identity(self):
        self.helper.reply([list_data()])
        result = self.helper.adapter.read_open_order_lists()
        self.assertEqual(result["data"][0]["order_list_id"], 31)
        self.assertEqual(result["data"][0]["orders"][0]["order_id"], 21)
        self.assertEqual(self.helper.calls[0][1], "/api/v3/openOrderList")

    def test_PB06_unknown_list_enum_rejected(self):
        self.helper.reply([list_data(contingencyType="UNKNOWN")])
        self.helper.assert_reason("SCHEMA_INVALID", self.helper.adapter.read_open_order_lists)
