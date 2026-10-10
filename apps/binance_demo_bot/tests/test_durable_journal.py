"""P0-B integration contracts; real SQLite files and process deaths, no broker."""

import importlib
import json
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path

from _support import D, FinancialContractCase, fill, lot, snapshot
from binance_demo_bot.financial import day_bounds
from binance_demo_bot.isolation import AppPaths


def envelope(event_id="receipt-1", *, exchange=1_000, received=5_000, **changes):
    value = dict(schema_version=1, application_id="binance_demo_bot",
                 environment="BINANCE_SPOT_DEMO", objective_contract_id="daily_net_equity_pnl_v1",
                 account_episode_id="synthetic-episode", portfolio_scope_id="synthetic-account",
                 event_id=event_id, event_type="ExchangeFillRecorded",
                 exchange_event_time_ms=exchange, received_at_ms=received,
                 received_monotonic_ns=received * 1_000_000, process_instance_id="fixture-process")
    value.update(changes)
    return value


class DurableJournalTests(FinancialContractCase):
    def setUp(self):
        self.api = importlib.import_module("binance_demo_bot.journal")
        fixture_root = Path(__file__).resolve().parents[1] / ".runtime" / "unit-fixtures"
        fixture_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=fixture_root)
        self.base = Path(temporary.name).resolve()
        self.assertTrue(self.base.is_relative_to(fixture_root.resolve()))
        self.addCleanup(temporary.cleanup)
        self.root = self.base / "app"
        self.root.mkdir()
        self.paths = AppPaths(self.root)
        self.opening = [lot("USDT", "1000", "1000")]
        self.marks = {"USDT": "1", "AAA": "100"}
        self.store = None
        self.addCleanup(self.close)

    def open(self, **changes):
        options = dict(episode_id="synthetic-episode", scope_id="synthetic-account",
                       opening_lots=self.opening)
        options.update(changes)
        self.store = self.api.JournalStore(self.paths, **options)
        return self.store

    def close(self):
        if self.store is not None:
            self.store.close()
            self.store = None

    def append(self, event=None, payload=None, marks=None):
        return self.store.append_fill(event or envelope(), payload or fill(), marks or self.marks)

    def reports_inputs(self, closing="1010", **changes):
        start, end = day_bounds("2026-10-01", "Europe/Budapest")
        options = dict(opening=snapshot(boundary=start), closing=snapshot(closing, boundary=end),
                       flows=[], reconciliation_status="COMPLETE", economic_costs_usdt="0")
        options.update(changes)
        return options

    def crash_child(self, stage, operation="fill"):
        arguments = dict(root=str(self.root), opening=self.opening, envelope=envelope(),
                         payload=fill(fee="1"), marks=self.marks, stage=stage,
                         operation=operation, report_inputs=self.reports_inputs())
        script = (
            "import json,os,sys; from binance_demo_bot.journal import JournalStore; "
            "from binance_demo_bot.isolation import AppPaths; "
            "args=json.loads(sys.argv[1]); "
            "hook=lambda stage: os._exit(23) if stage==args['stage'] else None; "
            "store=JournalStore(AppPaths(args['root']),episode_id='synthetic-episode',"
            "scope_id='synthetic-account',opening_lots=args['opening'],fault_hook=hook); "
            "store.record_daily_report('2026-10-01',**args['report_inputs']) "
            "if args['operation']=='report' else "
            "store.append_fill(args['envelope'],args['payload'],args['marks'])"
        )
        result = subprocess.run([sys.executable, "-I", "-c", script, json.dumps(arguments)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 23, result.stderr)

    def test_DUR_01_reopen_preserves_balances_fees_and_projection_hash(self):
        self.open()
        self.assertEqual(self.append(payload=fill(fee="1")), "APPLIED")
        before = self.store.projection({"USDT": "1", "AAA": "110"})
        self.assertEqual(before["balances"]["USDT"], D("799"))
        self.assertEqual(before["attribution"]["fee_expenses_usdt"], D("1"))
        self.assertEqual(before["attribution"]["unrealized_usdt"], D("20"))
        self.assertIs(before["exchange_reconciled"], False)
        self.assertEqual(before["evidence_kind"], "LOCAL_PROJECTION")
        self.close()
        self.open()
        self.assertEqual(self.store.projection({"USDT": "1", "AAA": "110"}), before)

    def test_DUR_01_sqlite_wal_and_full_synchronous_are_registered(self):
        self.open()
        with closing(sqlite3.connect(self.paths.runtime_path("ledger.sqlite"))) as connection, connection:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        self.assertEqual(self.store.sqlite_settings(),
                         {"journal_mode": "wal", "synchronous": 2, "foreign_keys": 1})

    def test_DUR_02_exact_retry_does_not_add_delivery_event_or_fee(self):
        self.open()
        self.append(payload=fill(fee="1"))
        before = self.store.projection(self.marks)
        self.assertEqual(self.append(payload=fill(fee="1")), "DUPLICATE")
        self.assertEqual(len(self.store.events()), 1)
        self.assertEqual(len(self.store.receipts()), 1)
        self.assertEqual(len(self.store.pending_outbox()), 1)
        self.assertEqual(self.store.projection(self.marks), before)

    def test_DUR_02_rest_ws_duplicate_preserves_separate_actual_receipts(self):
        self.open()
        self.append(payload=fill(fee="1", source="WS"))
        self.assertEqual(self.append(envelope("receipt-2", received=6_000,
                                              process_instance_id="restarted-process"),
                                     fill(quantity="2.0", quote_quantity="200.00", fee="1.0", source="REST")),
                         "DUPLICATE")
        self.assertEqual(len(self.store.events()), 1)
        receipts = self.store.receipts()
        self.assertEqual([r["envelope"]["received_at_ms"] for r in receipts], [5_000, 6_000])
        self.assertEqual(receipts[1]["envelope"]["process_instance_id"], "restarted-process")
        self.assertEqual(self.store.projection(self.marks)["balances"]["USDT"], D("799"))

    def test_DUR_03_conflicting_fill_or_mark_is_rejected_atomically(self):
        self.open()
        self.append()
        before = self.store.projection(self.marks)
        for payload, marks in ((fill(quantity="3", quote_quantity="300"), self.marks),
                               (fill(), {"USDT": "1", "AAA": "101"})):
            with self.subTest(conflict_type="economic payload or valuation"):
                with self.assertRaises(self.api.LedgerConflict):
                    self.append(envelope("other-delivery"), payload, marks)
                self.assertEqual(self.store.projection(self.marks), before)
                self.assertEqual(len(self.store.events()), 1)
                self.assertEqual(len(self.store.receipts()), 1)
                self.assertEqual(len(self.store.pending_outbox()), 1)

    def test_DUR_03_event_id_cannot_be_reused_for_another_trade(self):
        self.open()
        self.append()
        with self.assertRaises(self.api.LedgerConflict):
            self.append(payload=fill("other-trade"))
        self.assertEqual(len(self.store.events()), 1)

    def test_DUR_03_bad_namespace_schema_or_clocks_are_not_accepted(self):
        self.open()
        for changes in (dict(schema_version=2), dict(application_id="legacy_bot"),
                        dict(environment="BINANCE_SPOT_LIVE"),
                        dict(account_episode_id="foreign-episode"),
                        dict(portfolio_scope_id="foreign-account"),
                        dict(objective_contract_id="sales_revenue"),
                        dict(received_at_ms=1.0), dict(received_monotonic_ns=-1),
                        dict(exchange_event_time_ms=2_000)):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.append(envelope(**changes))
        self.assertEqual(self.store.events(), [])
        self.assertEqual(self.store.pending_outbox(), [])

    def test_DUR_04_exception_before_commit_rolls_back_all_projections(self):
        self.open()
        before = self.store.projection(self.marks)
        def fail(stage):
            if stage == "before_commit":
                raise RuntimeError("synthetic fault before commit")
        self.store.fault_hook = fail
        with self.assertRaises(RuntimeError):
            self.append()
        self.store.fault_hook = None
        self.assertEqual(self.store.projection(self.marks), before)
        self.assertEqual(self.store.events(), [])
        self.assertEqual(self.store.receipts(), [])
        self.assertEqual(self.store.pending_outbox(), [])
        self.assertEqual(self.append(), "APPLIED")

    def test_DUR_04_real_process_death_before_commit_is_recoverable(self):
        self.crash_child("before_commit")
        self.open()
        self.assertEqual(self.store.events(), [])
        self.assertEqual(self.store.pending_outbox(), [])
        self.assertEqual(self.store.projection(self.marks)["balances"], {"USDT": D("1000")})
        self.assertEqual(self.append(payload=fill(fee="1")), "APPLIED")

    def test_DUR_05_real_process_death_after_commit_retry_is_idempotent(self):
        self.crash_child("after_commit")
        self.open()
        self.assertEqual(self.append(payload=fill(fee="1")), "DUPLICATE")
        self.assertEqual(len(self.store.events()), 1)
        self.assertEqual(len(self.store.receipts()), 1)
        self.assertEqual(len(self.store.pending_outbox()), 1)
        self.assertEqual(self.store.projection(self.marks)["balances"]["USDT"], D("799"))

    def test_DUR_06_late_fill_rebuilds_fifo_in_exchange_order(self):
        self.open()
        self.append(envelope("buy-expensive", exchange=2_000),
                    fill("t2", quantity="1", quote_quantity="120", price="120", exchange_event_time_ms=2_000))
        self.append(envelope("sell", exchange=3_000),
                    fill("t3", side="SELL", quantity="1", quote_quantity="130", price="130",
                         exchange_event_time_ms=3_000))
        self.assertEqual(self.store.projection(self.marks)["attribution"]["gross_realized_usdt"], D("10"))
        self.append(envelope("late-cheap-buy", received=9_000),
                    fill("t1", quantity="1", quote_quantity="100", exchange_event_time_ms=1_000))
        result = self.store.projection({"USDT": "1", "AAA": "130"})
        self.assertEqual(result["balances"], {"USDT": D("910"), "AAA": D("1")})
        self.assertEqual(result["attribution"]["gross_realized_usdt"], D("30"))
        self.assertEqual(result["attribution"]["unrealized_usdt"], D("10"))
        self.close()
        self.open()
        self.assertEqual(self.store.projection({"USDT": "1", "AAA": "130"}), result)

    def test_DUR_07_late_accounting_correction_does_not_change_past_knowledge(self):
        self.open()
        self.append(envelope("known", exchange=2_000, received=3_000),
                    fill("t2", quantity="1", quote_quantity="120", price="120", exchange_event_time_ms=2_000))
        before = self.store.projection(self.marks, as_of_received_ms=4_000)
        self.append(envelope("late", received=9_000), fill("t1", quantity="1", quote_quantity="100"))
        self.assertEqual(self.store.projection(self.marks, as_of_received_ms=4_000), before)
        self.assertEqual(self.store.projection(self.marks)["balances"]["AAA"], D("2"))
        self.close()
        self.open()
        self.assertEqual(self.store.projection(self.marks, as_of_received_ms=4_000), before)

    def test_DUR_07_cutoff_includes_boundary_and_excludes_future_exchange_clock(self):
        self.open()
        self.append(envelope(received=5_000))
        self.assertEqual(self.store.projection(self.marks, as_of_received_ms=4_999)["balances"],
                         {"USDT": D("1000")})
        self.assertEqual(self.store.projection(self.marks, as_of_received_ms=5_000)["balances"]["AAA"], D("2"))
        self.append(envelope("future-clock", exchange=8_000, received=6_000),
                    fill("t2", quantity="1", quote_quantity="100", exchange_event_time_ms=8_000))
        self.assertEqual(self.store.projection(self.marks, as_of_received_ms=7_000)["balances"]["AAA"], D("2"))

    def test_DUR_08_outbox_repeats_same_id_until_idempotent_ack(self):
        self.open()
        self.append()
        pending = self.store.pending_outbox()
        self.assertEqual(pending[0]["topic"], "projection_changed")
        self.assertEqual(self.store.pending_outbox(), pending)
        self.close()
        self.open()
        self.assertEqual(self.store.pending_outbox(), pending)
        self.assertTrue(self.store.ack_outbox(pending[0]["id"]))
        self.assertFalse(self.store.ack_outbox(pending[0]["id"]))
        with self.assertRaises(ValueError):
            self.store.ack_outbox(999_999)
        self.close()
        self.open()
        self.assertEqual(self.store.pending_outbox(), [])

    def test_DUR_09_corrected_losing_report_preserves_old_favorable_revision(self):
        self.open()
        first = self.store.record_daily_report("2026-10-01", **self.reports_inputs())
        self.assertEqual(first["revision"], 1)
        self.assertEqual(first["metrics"]["net_pnl_usdt"], D("10"))
        self.append()
        self.assertTrue(self.store.reports("2026-10-01")[0]["stale"])
        second = self.store.record_daily_report("2026-10-01", **self.reports_inputs(closing="970"))
        self.assertEqual(second["revision"], 2)
        self.assertEqual(second["supersedes"], first["id"])
        self.assertEqual(second["metrics"]["net_pnl_usdt"], D("-30"))
        reports = self.store.reports("2026-10-01")
        self.assertEqual([r["metrics"]["net_pnl_usdt"] for r in reports], [D("10"), D("-30")])
        self.assertEqual([r["stale"] for r in reports], [True, False])
        self.assertIs(second["exchange_reconciled"], False)
        self.close()
        self.open()
        self.assertEqual(self.store.reports("2026-10-01"), reports)

    def test_DUR_09_pending_revision_is_kept_not_replaced_by_last_complete(self):
        self.open()
        self.store.record_daily_report("2026-10-01", **self.reports_inputs())
        self.store.record_daily_report("2026-10-01", **self.reports_inputs(reconciliation_status="PENDING"))
        reports = self.store.reports("2026-10-01")
        self.assertEqual(reports[-1]["metrics"]["status"], "PENDING")
        self.assertIsNone(reports[-1]["metrics"]["net_pnl_usdt"])

    def test_DUR_10_report_failure_rolls_back_revision_and_outbox(self):
        self.open()
        def fail(stage):
            if stage == "before_commit":
                raise RuntimeError("synthetic report fault")
        self.store.fault_hook = fail
        with self.assertRaises(RuntimeError):
            self.store.record_daily_report("2026-10-01", **self.reports_inputs())
        self.store.fault_hook = None
        self.assertEqual(self.store.reports("2026-10-01"), [])
        self.assertEqual(self.store.pending_outbox(), [])
        result = self.store.record_daily_report("2026-10-01", **self.reports_inputs())
        self.assertEqual(result["revision"], 1)
        self.assertEqual(self.store.pending_outbox()[0]["topic"], "daily_report_revision")

    def test_DUR_10_exact_report_retry_keeps_revision_and_outbox(self):
        self.open()
        original = self.store.record_daily_report("2026-10-01", **self.reports_inputs())
        same_values = self.reports_inputs(closing="1010.00")
        same_values["opening"]["equity_usdt"] = "1000.0"
        repeated = self.store.record_daily_report("2026-10-01", **same_values)
        self.assertEqual(repeated, original)
        self.assertEqual(len(self.store.reports("2026-10-01")), 1)
        self.assertEqual(len(self.store.pending_outbox()), 1)

    def test_DUR_10_report_process_death_after_commit_retry_has_no_new_revision(self):
        self.crash_child("after_commit", operation="report")
        self.open()
        before = self.store.reports("2026-10-01")[0]
        recovered = self.store.record_daily_report("2026-10-01", **self.reports_inputs())
        self.assertEqual(recovered, before)
        self.assertEqual(len(self.store.reports("2026-10-01")), 1)
        self.assertEqual(len(self.store.pending_outbox()), 1)

    def test_DUR_10_mismatched_scope_partial_day_or_future_day_is_rejected(self):
        self.open()
        for changes in (dict(opening=snapshot(boundary=0)),
                        dict(closing=snapshot("1010", boundary=day_bounds("2026-10-01", "Europe/Budapest")[1],
                                              scope_id="foreign-account"))):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.store.record_daily_report("2026-10-01", **self.reports_inputs(**changes))
        start, end = day_bounds("2099-01-01", "Europe/Budapest")
        with self.assertRaises(ValueError):
            self.store.record_daily_report("2099-01-01", opening=snapshot(boundary=start),
                closing=snapshot("1010", boundary=end), flows=[], reconciliation_status="COMPLETE", economic_costs_usdt="0")
        self.assertEqual(self.store.pending_outbox(), [])

    def test_DUR_11_immutable_events_and_reports_reject_sql_updates(self):
        self.open()
        self.append()
        self.store.record_daily_report("2026-10-01", **self.reports_inputs())
        with closing(sqlite3.connect(self.paths.runtime_path("ledger.sqlite"))) as connection, connection:
            for sql in ("DELETE FROM events", "UPDATE events SET payload_json='{}'",
                        "DELETE FROM receipts", "UPDATE reports SET content_json='{}'",
                        "UPDATE meta SET value='{}'"):
                with self.subTest(sql=sql):
                    with self.assertRaises(sqlite3.IntegrityError):
                        connection.execute(sql)
                    connection.rollback()
        self.assertEqual(len(self.store.events()), 1)

    def test_DUR_11_materialized_cache_tamper_fails_closed_on_reopen(self):
        self.open()
        self.append()
        self.close()
        with closing(sqlite3.connect(self.paths.runtime_path("ledger.sqlite"))) as connection, connection:
            connection.execute("UPDATE projection SET content_json='{}'")
        with self.assertRaises(self.api.JournalIntegrityError):
            self.open()

    def test_DUR_11_reopen_different_identity_or_inventory_is_refused(self):
        self.open()
        self.close()
        for changes in (dict(episode_id="foreign-episode"), dict(scope_id="foreign-account"),
                        dict(opening_lots=[lot("USDT", "2000", "2000")])):
            with self.subTest(changes=changes):
                with self.assertRaises(self.api.JournalIntegrityError):
                    self.open(**changes)
        self.open()
        self.assertEqual(self.store.projection(self.marks)["balances"], {"USDT": D("1000")})

    def test_DUR_11_unknown_existing_schema_is_not_migrated(self):
        self.paths.runtime_path("ledger.sqlite").parent.mkdir(parents=True)
        with closing(sqlite3.connect(self.paths.runtime_path("ledger.sqlite"))) as connection, connection:
            connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("INSERT INTO meta VALUES ('identity', ?)", (json.dumps({"schema_version": 99}),))
        with self.assertRaises(self.api.JournalIntegrityError):
            self.open()
        with closing(sqlite3.connect(self.paths.runtime_path("ledger.sqlite"))) as connection, connection:
            self.assertEqual(json.loads(connection.execute("SELECT value FROM meta").fetchone()[0])["schema_version"], 99)

    def test_DUR_12_second_writer_refused_without_stopping_or_deleting_owner(self):
        self.open()
        with self.assertRaises(self.api.WriterBusy):
            self.api.JournalStore(self.paths, episode_id="synthetic-episode", scope_id="synthetic-account",
                                  opening_lots=self.opening)
        self.assertEqual(self.append(), "APPLIED")
        self.close()
        self.open()
        self.assertEqual(len(self.store.events()), 1)

    def test_DUR_08_outbox_id_and_payload_cannot_be_changed_or_deleted(self):
        self.open()
        self.append()
        expected = self.store.pending_outbox()
        with closing(sqlite3.connect(self.paths.runtime_path("ledger.sqlite"))) as connection, connection:
            for statement in ("DELETE FROM outbox", "UPDATE outbox SET id=id+100",
                              "UPDATE outbox SET payload_json='{}'"):
                with self.subTest(statement=statement):
                    try:
                        with self.assertRaises(sqlite3.IntegrityError):
                            connection.execute(statement)
                    finally:
                        connection.rollback()
        self.assertEqual(self.store.pending_outbox(), expected)

    def test_DUR_07_optional_decision_provenance_is_preserved_not_discarded(self):
        self.open()
        provenance = dict(decision_id="decision-fixture", causation_id="intent-fixture",
                          arm_id=None, model_id=None, policy_version="policy-fixture",
                          source_sha256="a" * 64, risk_contract_id="risk-fixture")
        self.append(envelope(**provenance))
        recorded = self.store.receipts()[0]["envelope"]
        for key, value in provenance.items():
            self.assertEqual(recorded[key], value)
        self.assertGreaterEqual(recorded["recorded_at_ms"], recorded["received_at_ms"])
        with self.assertRaises(ValueError):
            self.append(envelope("invalid-provenance", decision_id=123))
        with closing(sqlite3.connect(self.paths.runtime_path("ledger.sqlite"))) as connection:
            identity = json.loads(connection.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
        self.assertRegex(identity["accounting_code_sha256"], r"^[a-f0-9]{64}$")

    def test_DUR_11_corrupt_outbox_content_is_detected_on_reopen(self):
        self.open()
        self.append()
        self.close()
        with closing(sqlite3.connect(self.paths.runtime_path("ledger.sqlite"))) as connection, connection:
            connection.execute("DROP TRIGGER immutable_outbox_payload")
            connection.execute("UPDATE outbox SET payload_json='{}'")
        with self.assertRaises(self.api.JournalIntegrityError):
            self.open()

    def test_DUR_12_database_path_cannot_escape_owned_runtime(self):
        for name in ("../../foreign.sqlite", str(self.base / "foreign.sqlite")):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.open(database=name)
        self.assertFalse((self.base / "foreign.sqlite").exists())
