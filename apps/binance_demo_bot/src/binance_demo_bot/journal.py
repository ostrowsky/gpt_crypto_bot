"""App-local durable accounting projection and at-least-once notification outbox.

No broker or order dispatcher. LOCAL_PROJECTION never certifies exchange data.
"""

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from .financial import AccountingError, FinancialLedger, LedgerConflict, daily_performance, day_bounds, decimal_amount
from .isolation import AppPaths


class WriterBusy(ValueError):
    """Another owner holds this app-local journal's OS lease."""


class JournalIntegrityError(ValueError):
    """Frozen identity, event content or derived state does not verify."""


FILL_FIELDS = (
    "application_id", "environment", "episode_id", "symbol", "trade_id", "order_id",
    "client_order_id", "base_asset", "quote_asset", "side", "quantity", "quote_quantity",
    "price", "commission_asset", "commission_quantity", "exchange_event_time_ms", "ownership",
)
MONEY_FIELDS = {"quantity", "quote_quantity", "price", "commission_quantity"}
METRIC_AMOUNTS = {"net_pnl_usdt", "net_external_flow_usdt", "daily_return", "economic_pnl_usdt"}
OPTIONAL_PROVENANCE = ("causation_id", "decision_id", "arm_id", "model_id", "policy_version",
                       "source_sha256", "risk_contract_id")


def _money(value):
    amount = decimal_amount(value)
    if amount == 0:
        return "0"
    text = format(amount, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _json_value(value):
    if isinstance(value, Decimal):
        return _money(value)
    if isinstance(value, float):
        raise ValueError("Binary floats are forbidden in journal payloads")
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise ValueError("Unsupported journal value")


def _json(value):
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _now_ms():
    return time.time_ns() // 1_000_000


def _accounting_code_hash():
    package = Path(__file__).resolve().parent
    return _digest({name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                    for name in ("financial.py", "journal.py", "isolation.py")})


class _WriterLease:
    def __init__(self, path):
        self.handle = open(path, "a+b")
        try:
            self.handle.seek(0, 2)
            if not self.handle.tell():
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise WriterBusy("Journal writer lease is already held") from None

    def close(self):
        if not self.handle.closed:
            self.handle.close()  # OS releases the lock, including after process death.


SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE events (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, trade_key TEXT NOT NULL UNIQUE,
 exchange_ms INTEGER NOT NULL, payload_json TEXT NOT NULL, marks_json TEXT NOT NULL,
 content_sha256 TEXT NOT NULL);
CREATE TABLE receipts (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
 economic_seq INTEGER NOT NULL REFERENCES events(seq), envelope_json TEXT NOT NULL,
 request_json TEXT NOT NULL, content_sha256 TEXT NOT NULL);
CREATE TABLE projection (id INTEGER PRIMARY KEY CHECK(id=1), content_json TEXT NOT NULL);
CREATE TABLE reports (
 id TEXT PRIMARY KEY, day TEXT NOT NULL, revision INTEGER NOT NULL,
 content_json TEXT NOT NULL, content_sha256 TEXT NOT NULL,
 request_sha256 TEXT NOT NULL UNIQUE, UNIQUE(day,revision));
CREATE TABLE outbox (
 id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, payload_json TEXT NOT NULL,
 content_sha256 TEXT NOT NULL,
 acked INTEGER NOT NULL DEFAULT 0 CHECK(acked IN (0,1)));
"""


class JournalStore:
    def __init__(self, paths, *, episode_id, scope_id, opening_lots,
                 database="ledger.sqlite", fault_hook=None):
        if not isinstance(paths, AppPaths):
            raise ValueError("An explicit AppPaths capability is required")
        if not isinstance(scope_id, str) or not scope_id:
            raise ValueError("A portfolio scope is required")
        FinancialLedger(episode_id=episode_id, opening_lots=opening_lots)
        normalized_lots = [dict(asset=lot["asset"], quantity=_money(lot["quantity"]),
                                basis_usdt=_money(lot["basis_usdt"]), ownership=lot["ownership"])
                           for lot in opening_lots]
        self.identity = dict(schema_version=1, application_id="binance_demo_bot",
                             environment="BINANCE_SPOT_DEMO", objective_contract_id="daily_net_equity_pnl_v1",
                             episode_id=episode_id, scope_id=scope_id, opening_lots=normalized_lots,
                             accounting_code_sha256=_accounting_code_hash())
        self.paths, self.fault_hook = paths, fault_hook
        self.database_path = paths.runtime_path(database)
        lease_path = paths.runtime_path(str(database) + ".writer.lock")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lease = _WriterLease(lease_path)
        self._connection = None
        try:
            self._connection = sqlite3.connect(self.database_path, timeout=0, isolation_level=None)
            self._connection.row_factory = sqlite3.Row
            tables = {row[0] for row in self._connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables:
                self._check_identity()
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            if not tables:
                self._initialize()
            self._verify()
        except BaseException:
            self.close()
            raise

    def _initialize(self):
        # Schema and genesis are committed together; incomplete initialization fails closed.
        statements = SCHEMA.split(";")
        with self._transaction(inject=False):
            for statement in statements:
                if statement.strip():
                    self._connection.execute(statement)
            for table in ("meta", "events", "receipts", "reports"):
                for operation in ("UPDATE", "DELETE"):
                    self._connection.execute(
                        f"CREATE TRIGGER immutable_{table}_{operation} BEFORE {operation} ON {table} "
                        "BEGIN SELECT RAISE(ABORT,'immutable journal row'); END")
            self._connection.execute(
                "CREATE TRIGGER immutable_outbox_payload BEFORE UPDATE OF id,topic,payload_json,content_sha256 ON outbox "
                "BEGIN SELECT RAISE(ABORT,'immutable outbox payload'); END")
            self._connection.execute(
                "CREATE TRIGGER immutable_outbox_delete BEFORE DELETE ON outbox "
                "BEGIN SELECT RAISE(ABORT,'immutable outbox message'); END")
            self._connection.execute("INSERT INTO meta VALUES ('identity',?)", (_json(self.identity),))
            self._connection.execute("INSERT INTO projection VALUES (1,?)", (_json(self._cache([])),))

    def _check_identity(self):
        try:
            row = self._connection.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
            if row is None or json.loads(row[0]) != self.identity:
                raise JournalIntegrityError("Journal schema or frozen account identity mismatch")
        except (sqlite3.Error, ValueError, TypeError):
            raise JournalIntegrityError("Journal schema or frozen account identity mismatch") from None

    @contextmanager
    def _transaction(self, *, inject=True):
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            if inject and self.fault_hook:
                self.fault_hook("before_commit")
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        if inject and self.fault_hook:
            self.fault_hook("after_commit")

    def sqlite_settings(self):
        return {key: self._connection.execute("PRAGMA " + key).fetchone()[0]
                for key in ("journal_mode", "synchronous", "foreign_keys")}

    def _event_rows(self):
        return [dict(row) for row in self._connection.execute("SELECT * FROM events ORDER BY seq")]

    def _source_hash(self, rows):
        return _digest(dict(identity=self.identity,
                            events=[dict(seq=r["seq"], content_sha256=r["content_sha256"]) for r in rows]))

    def _replay(self, rows):
        ledger = FinancialLedger(episode_id=self.identity["episode_id"], opening_lots=self.identity["opening_lots"])
        for row in sorted(rows, key=lambda r: (r["exchange_ms"], r["seq"])):
            ledger.apply_fill(json.loads(row["payload_json"]), json.loads(row["marks_json"]))
        return ledger

    def _cache(self, rows):
        ledger = self._replay(rows)
        balances = ledger.balances()
        return dict(balances=balances, managed_quantities={a: ledger.managed_quantity(a) for a in balances},
                    source_sha256=self._source_hash(rows))

    @staticmethod
    def _trade_key(payload):
        return _json([payload["application_id"], payload["environment"], payload["episode_id"],
                      payload["symbol"], payload["trade_id"]])

    def _envelope(self, envelope, payload):
        fixed = dict(schema_version=1, application_id="binance_demo_bot", environment="BINANCE_SPOT_DEMO",
                     objective_contract_id="daily_net_equity_pnl_v1", event_type="ExchangeFillRecorded",
                     account_episode_id=self.identity["episode_id"], portfolio_scope_id=self.identity["scope_id"])
        if type(envelope.get("schema_version")) is not int or any(envelope.get(k) != v for k, v in fixed.items()):
            raise ValueError("Invalid journal event namespace or schema")
        result = dict(fixed)
        for key in ("event_id", "process_instance_id"):
            value = envelope.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError("Event/process identity is required")
            result[key] = value
        for key in OPTIONAL_PROVENANCE:
            if key in envelope:
                value = envelope[key]
                if value is not None and (not isinstance(value, str) or not value):
                    raise ValueError("Invalid optional event provenance")
                if key == "source_sha256" and value is not None and (
                        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
                    raise ValueError("Invalid source artifact hash")
                result[key] = value
        for key in ("exchange_event_time_ms", "received_at_ms", "received_monotonic_ns"):
            value = envelope.get(key)
            if type(value) is not int or value < 0:
                raise ValueError("Invalid journal event clock")
            result[key] = value
        if (result["exchange_event_time_ms"] != payload["exchange_event_time_ms"] or
                result["received_at_ms"] > _now_ms()):
            raise ValueError("Event clock mismatch or future receipt")
        return result

    def append_fill(self, envelope, fill, marks):
        try:
            payload = {key: _money(fill[key]) if key in MONEY_FIELDS else fill[key] for key in FILL_FIELDS}
            normalized_marks = {asset: _money(value) for asset, value in marks.items()}
            # Validates the monetary form without applying a late fill to current FIFO.
            for key in MONEY_FIELDS:
                amount = decimal_amount(payload[key])
                if amount < 0 or (key != "commission_quantity" and amount == 0):
                    raise AccountingError("Invalid fill amount")
            event = self._envelope(envelope, payload)
            request = dict(envelope=event, payload=payload, marks=normalized_marks)
            request_json = _json(request)
        except (KeyError, TypeError, ArithmeticError):
            raise ValueError("Malformed journal fill request") from None
        trade_key = self._trade_key(payload)
        content_hash = _digest(dict(payload=payload, marks=normalized_marks))
        with self._transaction():
            previous_receipt = self._connection.execute(
                "SELECT request_json FROM receipts WHERE event_id=?", (event["event_id"],)).fetchone()
            if previous_receipt:
                if previous_receipt[0] != request_json:
                    raise LedgerConflict("Conflicting delivery under an accepted event ID")
                outcome = "DUPLICATE"
            else:
                existing = self._connection.execute("SELECT seq,content_sha256 FROM events WHERE trade_key=?",
                                                    (trade_key,)).fetchone()
                if existing and existing["content_sha256"] != content_hash:
                    raise LedgerConflict("Conflicting trade payload or contemporaneous valuation")
                if existing:
                    sequence, outcome = existing["seq"], "DUPLICATE"
                else:
                    cursor = self._connection.execute(
                        "INSERT INTO events(trade_key,exchange_ms,payload_json,marks_json,content_sha256) VALUES (?,?,?,?,?)",
                        (trade_key, payload["exchange_event_time_ms"], _json(payload), _json(normalized_marks), content_hash))
                    sequence, outcome = cursor.lastrowid, "APPLIED"
                stored_event = dict(event, recorded_at_ms=_now_ms())
                receipt_hash = _digest(dict(envelope=stored_event, request=request, economic_seq=sequence))
                self._connection.execute(
                    "INSERT INTO receipts(event_id,economic_seq,envelope_json,request_json,content_sha256) VALUES (?,?,?,?,?)",
                    (event["event_id"], sequence, _json(stored_event), request_json, receipt_hash))
                if outcome == "APPLIED":
                    cache = self._cache(self._event_rows())  # New journal is replayed in exchange-time order.
                    self._connection.execute("UPDATE projection SET content_json=? WHERE id=1", (_json(cache),))
                    self._enqueue("projection_changed", dict(event_seq=sequence, source_sha256=cache["source_sha256"]))
        return outcome

    def events(self):
        return [dict(seq=r["seq"], payload=json.loads(r["payload_json"]), marks=json.loads(r["marks_json"]),
                     content_sha256=r["content_sha256"]) for r in self._event_rows()]

    def receipts(self):
        return [dict(seq=r["seq"], economic_seq=r["economic_seq"], envelope=json.loads(r["envelope_json"]))
                for r in self._connection.execute("SELECT * FROM receipts ORDER BY seq")]

    def projection(self, marks, *, as_of_received_ms=None):
        rows = self._event_rows()
        if as_of_received_ms is not None:
            if type(as_of_received_ms) is not int:
                raise ValueError("Receipt cutoff must be UTC milliseconds")
            available = {r["economic_seq"] for r in self.receipts()
                         if r["envelope"]["received_at_ms"] <= as_of_received_ms}
            rows = [r for r in rows if r["seq"] in available and r["exchange_ms"] <= as_of_received_ms]
        ledger = self._replay(rows)
        return dict(self._cache(rows), attribution=ledger.attribution(marks),
                    evidence_kind="LOCAL_PROJECTION", exchange_reconciled=False)

    def _enqueue(self, topic, payload):
        self._connection.execute("INSERT INTO outbox(topic,payload_json,content_sha256) VALUES (?,?,?)",
                                 (topic, _json(payload), _digest(dict(topic=topic, payload=payload))))

    def pending_outbox(self, limit=100):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Outbox batch limit must be between 1 and 1000")
        return [dict(id=r["id"], topic=r["topic"], payload=json.loads(r["payload_json"]))
                for r in self._connection.execute("SELECT * FROM outbox WHERE acked=0 ORDER BY id LIMIT ?", (limit,))]

    def ack_outbox(self, message_id):
        if type(message_id) is not int:
            raise ValueError("Invalid outbox message ID")
        with self._transaction(inject=False):
            row = self._connection.execute("SELECT acked FROM outbox WHERE id=?", (message_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown outbox message ID")
            self._connection.execute("UPDATE outbox SET acked=1 WHERE id=?", (message_id,))
        return row["acked"] == 0

    def record_daily_report(self, day, **inputs):
        inputs = json.loads(_json(inputs))  # Capture an owned immutable input snapshot.
        for name in ("opening", "closing"):
            if inputs[name]["equity_usdt"] is not None:
                inputs[name]["equity_usdt"] = _money(inputs[name]["equity_usdt"])
        for flow in inputs["flows"]:
            if flow["signed_usdt"] is not None:
                flow["signed_usdt"] = _money(flow["signed_usdt"])
        if inputs["economic_costs_usdt"] is not None:
            inputs["economic_costs_usdt"] = _money(inputs["economic_costs_usdt"])
        start, end = day_bounds(day, "Europe/Budapest")
        if end > _now_ms():
            raise ValueError("An unended day cannot be finalized")
        for name, boundary in (("opening", start), ("closing", end)):
            snapshot = inputs[name]
            if (snapshot["boundary_ms"] != boundary or snapshot["episode_id"] != self.identity["episode_id"]
                    or snapshot["scope_id"] != self.identity["scope_id"]):
                raise ValueError("Report boundaries, episode or scope mismatch")
        metrics = daily_performance(**inputs)
        with self._transaction():
            source = self._source_hash(self._event_rows())
            request_hash = _digest(dict(day=day, inputs=inputs, source_sha256=source))
            existing = self._connection.execute("SELECT id FROM reports WHERE request_sha256=?",
                                                (request_hash,)).fetchone()
            if existing:
                return next(report for report in self.reports(day) if report["id"] == existing["id"])
            prior = self._connection.execute("SELECT id,revision FROM reports WHERE day=? ORDER BY revision DESC LIMIT 1",
                                             (day,)).fetchone()
            body = dict(day=day, revision=prior["revision"] + 1 if prior else 1,
                        supersedes=prior["id"] if prior else None, inputs=inputs, metrics=metrics,
                        source_sha256=source, recorded_at_ms=_now_ms(),
                        evidence_kind="LOCAL_PROJECTION", exchange_reconciled=False)
            digest = _digest(body)
            content = dict(body, id=digest, content_sha256=digest)
            self._connection.execute("INSERT INTO reports VALUES (?,?,?,?,?,?)",
                                     (digest, day, body["revision"], _json(content), digest, request_hash))
            self._enqueue("daily_report_revision", dict(report_id=digest, day=day, revision=body["revision"]))
        return self.reports(day)[-1]

    def reports(self, day):
        current_source = self._source_hash(self._event_rows())
        result = []
        for row in self._connection.execute("SELECT content_json FROM reports WHERE day=? ORDER BY revision", (day,)):
            report = json.loads(row[0])
            report["stale"] = report["source_sha256"] != current_source
            report["metrics"] = {key: decimal_amount(value) if key in METRIC_AMOUNTS and value is not None else value
                                 for key, value in report["metrics"].items()}
            result.append(report)
        return result

    def _verify(self):
        try:
            self._check_identity()
            rows = self._event_rows()
            by_sequence = {r["seq"]: r for r in rows}
            for row in rows:
                payload, marks = json.loads(row["payload_json"]), json.loads(row["marks_json"])
                if (row["content_sha256"] != _digest(dict(payload=payload, marks=marks))
                        or row["trade_key"] != self._trade_key(payload)
                        or row["exchange_ms"] != payload["exchange_event_time_ms"]):
                    raise JournalIntegrityError("Economic event hash mismatch")
            observed = set()
            for row in self._connection.execute("SELECT * FROM receipts ORDER BY seq"):
                envelope, request = json.loads(row["envelope_json"]), json.loads(row["request_json"])
                event = by_sequence[row["economic_seq"]]
                expected = _digest(dict(envelope=envelope, request=request, economic_seq=row["economic_seq"]))
                if (expected != row["content_sha256"] or row["event_id"] != envelope["event_id"]
                        or request["payload"] != json.loads(event["payload_json"])
                        or request["marks"] != json.loads(event["marks_json"])):
                    raise JournalIntegrityError("Receipt hash or economic association mismatch")
                observed.add(row["economic_seq"])
            if observed != set(by_sequence):
                raise JournalIntegrityError("Economic event has no delivery evidence")
            stored = self._connection.execute("SELECT content_json FROM projection WHERE id=1").fetchone()
            if stored is None or stored[0] != _json(self._cache(rows)):
                raise JournalIntegrityError("Materialized projection differs from journal replay")
            for row in self._connection.execute("SELECT * FROM reports ORDER BY day,revision"):
                content = json.loads(row["content_json"])
                digest, identifier = content.pop("content_sha256"), content.pop("id")
                if (digest != _digest(content) or digest != row["content_sha256"] or identifier != row["id"]
                        or content["day"] != row["day"] or content["revision"] != row["revision"]
                        or row["request_sha256"] != _digest(dict(day=content["day"], inputs=content["inputs"],
                                                                 source_sha256=content["source_sha256"]))):
                    raise JournalIntegrityError("Immutable report hash mismatch")
            notifications = set()
            for row in self._connection.execute("SELECT * FROM outbox ORDER BY id"):
                payload = json.loads(row["payload_json"])
                if row["content_sha256"] != _digest(dict(topic=row["topic"], payload=payload)):
                    raise JournalIntegrityError("Outbox payload hash mismatch")
                if row["topic"] == "projection_changed":
                    sequence = payload["event_seq"]
                    if sequence not in by_sequence or payload["source_sha256"] != self._source_hash(
                            [event for event in rows if event["seq"] <= sequence]):
                        raise JournalIntegrityError("Outbox projection association mismatch")
                    identity = (row["topic"], sequence)
                elif row["topic"] == "daily_report_revision":
                    report = self._connection.execute("SELECT day,revision FROM reports WHERE id=?",
                                                      (payload["report_id"],)).fetchone()
                    if report is None or report["day"] != payload["day"] or report["revision"] != payload["revision"]:
                        raise JournalIntegrityError("Outbox report association mismatch")
                    identity = (row["topic"], payload["report_id"])
                else:
                    raise JournalIntegrityError("Unknown outbox notification topic")
                if identity in notifications:
                    raise JournalIntegrityError("Duplicate outbox notification")
                notifications.add(identity)
            expected_notifications = {("projection_changed", seq) for seq in by_sequence}
            expected_notifications.update(("daily_report_revision", row[0])
                                          for row in self._connection.execute("SELECT id FROM reports"))
            if notifications != expected_notifications:
                raise JournalIntegrityError("Missing durable outbox notification")
            if self._connection.execute("PRAGMA foreign_key_check").fetchall():
                raise JournalIntegrityError("Journal foreign-key integrity mismatch")
        except (sqlite3.Error, KeyError, ValueError, TypeError):
            raise JournalIntegrityError("Journal content or projection integrity failed") from None

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._lease.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
