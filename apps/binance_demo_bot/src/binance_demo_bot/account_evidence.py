"""Durable private observations and conservative diagnostics, without execution.

This store cannot certify account history, seed anchors or trading ownership.
It never writes into FinancialLedger/JournalStore; unresolved incidents survive.
"""

from contextlib import contextmanager
from decimal import Decimal, Context, localcontext
import json
import sqlite3
import threading
import time
import uuid

from .adapter import AdapterError, _integer, _text, _sha, _money
from .isolation import AppPaths
from .journal import JournalStore, _WriterLease, _json, _digest


FIXED = dict(application_id="binance_demo_bot", environment="BINANCE_SPOT_DEMO",
             objective_contract_id="daily_net_equity_pnl_v1")
PROCESS_INSTANCE_ID = uuid.uuid4().hex
INCIDENTS = frozenset(("STREAM_GAP", "FILL_CONFLICT", "HISTORY_GAP", "UNCLASSIFIED_FLOW",
    "UNKNOWN_EVENT", "STREAM_TERMINATED", "EXTERNAL_LOCK", "UNMAPPED_SUBSCRIPTION",
    "WS_BUFFER_OVERFLOW", "WS_AUTH_DENIED", "WS_RATE_LIMITED", "WS_BANNED", "WS_PROTOCOL_ERROR",
    "WS_UNAVAILABLE", "WS_HEARTBEAT_FAILED", "SCHEMA_INVALID", "ACCOUNT_IDENTITY_MISMATCH",
    "ORDER_STATE_CONFLICT", "READER_STOP_TIMEOUT"))
FILL_FIELDS = frozenset(("symbol", "trade_id", "order_id", "order_list_id", "price", "quantity",
    "quote_quantity", "commission_quantity", "commission_asset", "side", "is_maker",
    "exchange_event_time_ms", "ownership"))


def _amount(value, positive=False):
    return _money(format(value, "f") if isinstance(value, Decimal) else value, positive)


def normalized_fill(data):
    if not isinstance(data, dict) or set(data) - FILL_FIELDS:
        raise AdapterError("SCHEMA_INVALID")
    try:
        result = {k: data[k] for k in FILL_FIELDS}
        for k in ("symbol",):
            _text(result[k])
        for k in ("trade_id", "order_id", "exchange_event_time_ms"):
            _integer(result[k])
        _integer(result["order_list_id"], -1)
        if result["side"] not in ("BUY", "SELL") or type(result["is_maker"]) is not bool or result["ownership"] != "UNATTRIBUTED":
            raise AdapterError("SCHEMA_INVALID")
        for k in ("price", "quantity", "quote_quantity", "commission_quantity"):
            result[k] = _amount(result[k], k != "commission_quantity")
        if result["commission_quantity"] == 0:
            result["commission_asset"] = None
        else:
            _text(result["commission_asset"])
        return json.loads(_json(result))
    except (KeyError, TypeError, ValueError):
        raise AdapterError("SCHEMA_INVALID") from None


def _event(data):
    """Defend the durable boundary, even when called outside the wire reader."""
    if not isinstance(data, dict) or data.get("kind") not in {"ORDER_UPDATE", "PARTIAL_BALANCES", "FLOW", "LIST_UPDATE"}:
        raise AdapterError("SCHEMA_INVALID")
    fields = {"ORDER_UPDATE": {"transaction_time_ms", "execution_id", "order", "fill"},
              "PARTIAL_BALANCES": {"balances", "update_time_ms"}, "FLOW": {"asset", "delta", "transaction_time_ms"},
              "LIST_UPDATE": {"list"}}
    allowed = fields[data["kind"]] | {"kind", "exchange_event_time_ms"}
    if set(data) != allowed:
        raise AdapterError("SCHEMA_INVALID")
    result = json.loads(_json(data))
    _integer(result["exchange_event_time_ms"])
    if result["kind"] == "ORDER_UPDATE":
        _integer(result["transaction_time_ms"])
        _integer(result["execution_id"])
        if not isinstance(result["order"], dict):
            raise AdapterError("SCHEMA_INVALID")
        # Fields are normalized and whitelisted by the adapter's order parser.
        order_fields = {"symbol", "order_id", "order_list_id", "client_order_id", "status", "side", "type",
                        "price", "original_quantity", "executed_quantity", "cumulative_quote_quantity",
                        "time_ms", "update_time_ms", "ownership"}
        if set(result["order"]) != order_fields:
            raise AdapterError("SCHEMA_INVALID")
        o = result["order"]
        for name in ("symbol", "client_order_id"):
            _text(o[name])
        for name in ("order_id", "time_ms", "update_time_ms"):
            _integer(o[name])
        _integer(o["order_list_id"], -1)
        if o["side"] not in ("BUY", "SELL") or o["ownership"] != "UNATTRIBUTED":
            raise AdapterError("SCHEMA_INVALID")
        if o["status"] not in {"NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", "PENDING_CANCEL", "REJECTED", "EXPIRED", "EXPIRED_IN_MATCH", "PENDING_NEW"}:
            raise AdapterError("SCHEMA_INVALID")
        if o["type"] not in {"LIMIT", "MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT", "TAKE_PROFIT", "TAKE_PROFIT_LIMIT", "LIMIT_MAKER"}:
            raise AdapterError("SCHEMA_INVALID")
        for name in ("price", "original_quantity", "executed_quantity", "cumulative_quote_quantity"):
            _amount(o[name])
        if result.get("fill") is not None:
            result["fill"] = normalized_fill(result["fill"])
            if (result["fill"]["symbol"] != o["symbol"] or result["fill"]["order_id"] != o["order_id"]
                    or result["fill"]["side"] != o["side"]):
                raise AdapterError("SCHEMA_INVALID")
    elif result["kind"] == "PARTIAL_BALANCES":
        _integer(result["update_time_ms"])
        if not isinstance(result["balances"], list):
            raise AdapterError("SCHEMA_INVALID")
        seen = set()
        for row in result["balances"]:
            if not isinstance(row, dict) or set(row) != {"asset", "free", "locked"}:
                raise AdapterError("SCHEMA_INVALID")
            asset = _text(row["asset"])
            if asset in seen:
                raise AdapterError("SCHEMA_INVALID")
            seen.add(asset)
            _amount(row["free"])
            _amount(row["locked"])
    elif result["kind"] == "FLOW":
        _text(result["asset"])
        _integer(result["transaction_time_ms"])
        value = result["delta"]
        _amount(value[1:] if isinstance(value, str) and value.startswith("-") else value)
    elif result["kind"] == "LIST_UPDATE":
        if not isinstance(result["list"], dict):
            raise AdapterError("SCHEMA_INVALID")
        # The adapter validates full enums and members before producing this kind.
        value = result["list"]
        if set(value) != {"order_list_id", "symbol", "contingency_type", "list_status_type", "list_order_status",
                          "list_client_order_id", "transaction_time_ms", "orders", "ownership"}:
            raise AdapterError("SCHEMA_INVALID")
        _integer(value["order_list_id"])
        _text(value["list_client_order_id"])
        if value["ownership"] != "UNATTRIBUTED":
            raise AdapterError("SCHEMA_INVALID")
        for member in value["orders"]:
            if set(member) != {"symbol", "order_id", "client_order_id"}:
                raise AdapterError("SCHEMA_INVALID")
            _integer(member["order_id"])
            _text(member["client_order_id"])
    return result


class AccountEvidenceStore:
    def __init__(self, paths, *, account_alias, account_identity, episode_id, scope_id,
                 database="account-evidence.sqlite", fault_hook=None):
        if not isinstance(paths, AppPaths):
            raise AdapterError("EVIDENCE_IDENTITY_MISMATCH")
        for value in (account_alias, episode_id, scope_id):
            _text(value)
        if not isinstance(account_identity, str) or len(account_identity) != 64 or any(c not in "0123456789abcdef" for c in account_identity):
            raise AdapterError("EVIDENCE_IDENTITY_MISMATCH")
        self.identity = {**FIXED, "schema_version": 1, "account_alias": account_alias,
                         "account_identity": account_identity, "account_episode_id": episode_id,
                         "portfolio_scope_id": scope_id}
        self.path = paths.runtime_path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._lease = _WriterLease(paths.runtime_path(database+".writer.lock"))
        self._con = None
        self.fault_hook = fault_hook
        try:
            self._con = sqlite3.connect(self.path, timeout=0, isolation_level=None, check_same_thread=False)
            self._con.row_factory = sqlite3.Row
            tables = {r[0] for r in self._con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA synchronous=FULL")
            self._con.execute("PRAGMA foreign_keys=ON")
            if not tables:
                with self._transaction(inject=False):
                    self._con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
                    self._con.execute("CREATE TABLE records (seq INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,kind TEXT NOT NULL,payload_json TEXT NOT NULL,content_sha256 TEXT NOT NULL)")
                    self._con.execute("CREATE TABLE fills (trade_key TEXT PRIMARY KEY,record_seq INTEGER NOT NULL REFERENCES records(seq),data_json TEXT NOT NULL,content_sha256 TEXT NOT NULL)")
                    self._con.execute("INSERT INTO meta VALUES ('identity',?)", (_json(self.identity),))
                    for table in ("meta", "records", "fills"):
                        for operation in ("UPDATE", "DELETE"):
                            self._con.execute(f"CREATE TRIGGER immutable_{table}_{operation} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT,'immutable evidence'); END")
            else:
                try:
                    row = self._con.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
                    if not row or json.loads(row[0]) != self.identity:
                        raise AdapterError("EVIDENCE_IDENTITY_MISMATCH")
                except sqlite3.Error:
                    raise AdapterError("EVIDENCE_IDENTITY_MISMATCH") from None
            self._verify()
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _transaction(self, inject=True):
        with self._lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                yield
                if inject and self.fault_hook:
                    self.fault_hook("before_commit")
                self._con.commit()
            except BaseException:
                self._con.rollback()
                raise
            if inject and self.fault_hook:
                self.fault_hook("after_commit")

    def _append(self, kind, data, **provenance):
        request = dict(self.identity, kind=kind, data=data, event_type="PrivateAccountEvidenceRecorded",
                       exchange_event_time_ms=data.get("exchange_event_time_ms") if kind == "EVENT" else None,
                       received_at_ms=time.time_ns()//1_000_000, received_monotonic_ns=time.monotonic_ns(),
                       process_instance_id=PROCESS_INSTANCE_ID, source_sha256=_digest(data),
                       causation_id=None, model_id=None, policy_version=None, arm_id=None, risk_contract_id=None)
        request.update(provenance)
        identity = _digest(request)
        existing = self._con.execute("SELECT seq FROM records WHERE id=?", (identity,)).fetchone()
        if existing:
            return existing[0]
        payload = {**request, "event_id": identity, "recorded_at_ms": time.time_ns()//1_000_000}
        return self._con.execute("INSERT INTO records(id,kind,payload_json,content_sha256) VALUES (?,?,?,?)",
            (identity, kind, _json(payload), _digest(payload))).lastrowid

    def _rows(self, kind=None):
        with self._lock:
            sql, args = ("SELECT * FROM records ORDER BY seq", ()) if kind is None else ("SELECT * FROM records WHERE kind=? ORDER BY seq", (kind,))
            return [dict(seq=r["seq"], id=r["id"], **json.loads(r["payload_json"])) for r in self._con.execute(sql, args)]

    def generation(self):
        with self._lock:
            return self._con.execute("SELECT COALESCE(MAX(seq),0) FROM records WHERE kind!='COMPARISON'").fetchone()[0]

    def sqlite_settings(self):
        with self._lock:
            return {k: self._con.execute("PRAGMA "+k).fetchone()[0] for k in ("journal_mode", "synchronous", "foreign_keys")}

    def bind_session(self, session_id, subscription_id, credential_generation):
        _text(session_id)
        _integer(subscription_id)
        _text(credential_generation)
        data = dict(session_id=session_id, subscription_id=subscription_id, credential_generation=credential_generation)
        with self._transaction():
            existing = [r["data"] for r in self._rows("SESSION") if r["data"]["session_id"] == session_id]
            if existing and existing[0] != data:
                raise AdapterError("SESSION_CONFLICT")
            if existing:
                return
            self._append("SESSION", data)

    def record_incident(self, reason):
        if reason not in INCIDENTS:
            raise AdapterError("INVALID_INCIDENT")
        with self._transaction(inject=False):
            self._append("INCIDENT", {"reason": reason}, received_at_ms=time.time_ns()//1_000_000,
                         received_monotonic_ns=time.monotonic_ns())

    def _insert_fill(self, data, sequence):
        fill = normalized_fill(data)
        key = _json([fill["symbol"], fill["trade_id"]])
        existing = self._con.execute("SELECT data_json FROM fills WHERE trade_key=?", (key,)).fetchone()
        if existing:
            if existing[0] != _json(fill):
                raise AdapterError("FILL_CONFLICT")
        else:
            self._con.execute("INSERT INTO fills VALUES (?,?,?,?)", (key, sequence, _json(fill), _digest(fill)))

    def record_event(self, session_id, subscription_id, normalized_event, received_at_ms,
                     received_monotonic_ns, process_instance_id):
        _integer(received_at_ms)
        _integer(received_monotonic_ns)
        _text(process_instance_id)
        if received_at_ms > time.time_ns()//1_000_000:
            raise AdapterError("EVIDENCE_FROM_FUTURE")
        data = _event(normalized_event)
        with self._lock:
            mapping = [r["data"] for r in self._rows("SESSION") if r["data"]["session_id"] == session_id]
            if not mapping or mapping[0]["subscription_id"] != subscription_id:
                raise AdapterError("UNMAPPED_SUBSCRIPTION")
            try:
                with self._transaction():
                    seq = self._append("EVENT", data, session_id=session_id, subscription_id=subscription_id,
                        credential_generation=mapping[0]["credential_generation"], received_at_ms=received_at_ms,
                        received_monotonic_ns=received_monotonic_ns, process_instance_id=process_instance_id)
                    if data.get("fill"):
                        self._insert_fill(data["fill"], seq)
                if data["kind"] == "FLOW":
                    self.record_incident("UNCLASSIFIED_FLOW")
                self.order_states()  # Validate consistent cumulative facts.
                return seq
            except AdapterError as error:
                if error.reason in INCIDENTS:
                    self.record_incident(error.reason)
                raise

    def validate_observation(self, value, endpoint):
        if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
                or value["schema_version"] != 1 or any(value.get(k) != v for k, v in FIXED.items())
                or any(value.get(k) != self.identity[k] for k in ("account_alias", "account_episode_id", "portfolio_scope_id"))):
            raise AdapterError("EVIDENCE_NAMESPACE_MISMATCH")
        if value.get("source_endpoint") != endpoint or value.get("source_method") != "GET":
            raise AdapterError("EVIDENCE_NAMESPACE_MISMATCH")
        _text(value["credential_generation"])
        if value.get("source_sha256") != _sha(value["data"]):
            raise AdapterError("EVIDENCE_HASH_MISMATCH")
        _integer(value["received_at_ms"])
        _integer(value["request_started_at_ms"])
        _integer(value["received_monotonic_ns"])
        if value["request_started_at_ms"] > value["received_at_ms"]:
            raise AdapterError("SCHEMA_INVALID")
        if value["received_at_ms"] > time.time_ns()//1_000_000:
            raise AdapterError("EVIDENCE_FROM_FUTURE")

    def record_recovery(self, result):
        self.validate_observation(result, "/api/v3/myTrades")
        try:
            symbol = _text(result["symbol"])
            start, next_id = _integer(result["starting_from_id"]), _integer(result["next_from_id"])
            if type(result["tail_exhausted"]) is not bool or not isinstance(result["data"], list):
                raise AdapterError("SCHEMA_INVALID")
            fills = [normalized_fill(row) for row in result["data"]]
            expected_next = max([start-1]+[f["trade_id"] for f in fills])+1
            if next_id != expected_next:
                raise AdapterError("PAGINATION_INVALID")
            if any(r["symbol"] != symbol or r["trade_id"] < start or r["trade_id"] >= next_id for r in fills):
                raise AdapterError("PAGINATION_INVALID")
            if next_id < start or not isinstance(result["page_observations"], list) or not result["page_observations"]:
                raise AdapterError("PAGINATION_INVALID")
            pages = []
            for page in result["page_observations"]:
                self.validate_observation(page, "/api/v3/myTrades")
                if page["credential_generation"] != result["credential_generation"]:
                    raise AdapterError("EVIDENCE_NAMESPACE_MISMATCH")
                page_fills = [normalized_fill(row) for row in page["data"]]
                if any(r["symbol"] != symbol for r in page_fills):
                    raise AdapterError("PAGINATION_INVALID")
                pages.append(dict(data=page_fills, source_sha256=page["source_sha256"],
                                  received_at_ms=page["received_at_ms"], credential_generation=page["credential_generation"]))
            if {_digest(f) for p in pages for f in p["data"]} != {_digest(f) for f in fills}:
                raise AdapterError("EVIDENCE_HASH_MISMATCH")
            data = dict(symbol=symbol, starting_from_id=start, next_from_id=next_id,
                        tail_exhausted=result["tail_exhausted"], fills=fills, pages=pages,
                        source_sha256=result["source_sha256"], credential_generation=result["credential_generation"])
            with self._lock:
                previous = self.cursor(symbol)
                try:
                    with self._transaction():
                        sequence = self._append("RECOVERY", data, received_at_ms=result["received_at_ms"],
                                                received_monotonic_ns=result["received_monotonic_ns"])
                        for fill in fills:
                            self._insert_fill(fill, sequence)
                        if start > previous:
                            self._append("INCIDENT", {"reason": "HISTORY_GAP"})
                except AdapterError as error:
                    if error.reason == "FILL_CONFLICT":
                        self.record_incident(error.reason)
                    raise
                return sequence
        except (KeyError, TypeError, ValueError):
            raise AdapterError("SCHEMA_INVALID") from None

    def recovery_status(self, symbol):
        _text(symbol)
        rows = [r for r in self._rows("RECOVERY") if r["data"]["symbol"] == symbol]
        cursor, origin, tail = 0, False, False
        for row in rows:
            data = row["data"]
            origin = origin or data["starting_from_id"] == 0
            if data["next_from_id"] >= cursor:
                cursor, tail = data["next_from_id"], data["tail_exhausted"]
        return dict(next_from_id=cursor, origin_known=origin, tail_exhausted=tail, observations_n=len(rows))

    def cursor(self, symbol):
        return self.recovery_status(symbol)["next_from_id"]

    def recovery_start(self, symbol, overlap=100):
        _integer(overlap)
        return max(0, self.cursor(symbol)-overlap)

    def fills(self):
        with self._lock:
            return [dict(record_seq=r["record_seq"], data=json.loads(r["data_json"]), source_sha256=r["content_sha256"])
                    for r in self._con.execute("SELECT * FROM fills ORDER BY record_seq,trade_key")]

    def events(self):
        return self._rows("EVENT")

    def incidents(self):
        return self._rows("INCIDENT")

    def order_states(self):
        latest = {}
        rows = sorted((r for r in self.events() if r["data"]["kind"] == "ORDER_UPDATE"),
                      key=lambda r: (r["data"]["transaction_time_ms"], r["seq"]))
        for row in rows:
            o = row["data"]["order"]
            key = o["symbol"], o["order_id"]
            if key in latest and Decimal(o["executed_quantity"]) < Decimal(latest[key]["executed_quantity"]):
                raise AdapterError("ORDER_STATE_CONFLICT")
            if (key in latest and latest[key]["status"] in {"FILLED", "CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "REJECTED"}
                    and latest[key]["status"] != o["status"]):
                raise AdapterError("ORDER_STATE_CONFLICT")
            latest[key] = o
        return [latest[k] for k in sorted(latest)]

    def save_comparison(self, data):
        with self._transaction(inject=False):
            seq = self._append("COMPARISON", data)
            return next(r["id"] for r in self._rows("COMPARISON") if r["seq"] == seq)

    def comparison_stale(self, identity, journal=None, marks=None):
        row = next((r for r in self._rows("COMPARISON") if r["id"] == identity), None)
        if row is None:
            raise AdapterError("COMPARISON_UNKNOWN")
        stale = row["data"]["evidence_generation"] != self.generation()
        if (journal is None) != (marks is None):
            raise AdapterError("INVALID_PARAMETERS")
        if journal is not None:
            if not isinstance(journal, JournalStore):
                raise AdapterError("JOURNAL_NAMESPACE_MISMATCH")
            stale = stale or journal.projection(marks)["source_sha256"] != row["data"]["journal_source_sha256"]
        return stale

    def _verify(self):
        try:
            derived = {}
            for row in self._con.execute("SELECT * FROM records ORDER BY seq"):
                payload = json.loads(row["payload_json"])
                if _digest(payload) != row["content_sha256"] or row["kind"] != payload["kind"]:
                    raise ValueError()
                if (any(payload.get(k) != v for k, v in self.identity.items()) or payload.get("event_id") != row["id"]
                        or payload.get("event_type") != "PrivateAccountEvidenceRecorded"
                        or payload.get("source_sha256") != _digest(payload["data"])):
                    raise ValueError()
                request = {k: v for k, v in payload.items() if k not in {"recorded_at_ms", "event_id"}}
                if _digest(request) != row["id"]:
                    raise ValueError()
                fills = payload["data"].get("fills", []) if row["kind"] == "RECOVERY" else (
                    [payload["data"]["fill"]] if row["kind"] == "EVENT" and payload["data"].get("fill") else [])
                for fill in fills:
                    fill = normalized_fill(fill)
                    key = _json([fill["symbol"], fill["trade_id"]])
                    if key in derived and derived[key][0] != fill:
                        raise ValueError()
                    derived.setdefault(key, (fill, row["seq"]))
            stored = {}
            for row in self._con.execute("SELECT * FROM fills"):
                data = json.loads(row["data_json"])
                if _digest(data) != row["content_sha256"]:
                    raise ValueError()
                stored[row["trade_key"]] = data, row["record_seq"]
            if stored != derived or self._con.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError()
        except (KeyError, TypeError, ValueError, sqlite3.Error, AdapterError):
            raise AdapterError("EVIDENCE_INTEGRITY_FAILED") from None

    def close(self):
        with self._lock:
            if self._con is not None:
                self._con.close()
                self._con = None
            self._lease.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class AccountReconciler:
    def __init__(self, store):
        if not isinstance(store, AccountEvidenceStore):
            raise AdapterError("EVIDENCE_IDENTITY_MISMATCH")
        self.store = store

    def recover(self, adapter, symbols, *, page_size=1000, max_pages=100):
        if not isinstance(symbols, (list, tuple, set)) or any(not isinstance(s, str) or not s for s in symbols):
            raise AdapterError("INVALID_PARAMETERS")
        result = []
        for symbol in sorted(set(symbols)):
            response = adapter.recover_fills(symbol, from_id=self.store.recovery_start(symbol),
                                            page_size=page_size, max_pages=max_pages)
            self.store.record_recovery(response)
            result.append(response)
        return result

    def compare(self, journal, marks, *, account, orders, lists, population_symbols,
                population_complete=False, owned_client_ids=(), stream_health=None, now_ms=None,
                journal_anchor_ms):
        if not isinstance(journal, JournalStore):
            raise AdapterError("EVIDENCE_IDENTITY_MISMATCH")
        _integer(journal_anchor_ms)
        now = time.time_ns()//1_000_000 if now_ms is None else _integer(now_ms)
        if journal_anchor_ms > now:
            raise AdapterError("INVALID_JOURNAL_ANCHOR")
        generation = self.store.generation()
        reasons = []
        valid = {}
        for value, endpoint in ((account, "/api/v3/account"), (orders, "/api/v3/openOrders"), (lists, "/api/v3/openOrderList")):
            valid[endpoint] = True
            try:
                self.store.validate_observation(value, endpoint)
                if value["received_at_ms"] > now:
                    reasons.append("EVIDENCE_FROM_FUTURE")
                    valid[endpoint] = False
                elif now-value["received_at_ms"] > 30000:
                    reasons.append("ACCOUNT_SNAPSHOT_STALE")
                    valid[endpoint] = False
            except AdapterError as error:
                reasons.append(error.reason)
                valid[endpoint] = False
            except (KeyError, TypeError, ValueError):
                reasons.append("SCHEMA_INVALID")
                valid[endpoint] = False
        account_data = account.get("data", {}) if isinstance(account, dict) else {}
        if not isinstance(account_data, dict):
            account_data = {}
            valid["/api/v3/account"] = False
            reasons.append("SCHEMA_INVALID")
        if account_data.get("account_identity") != self.store.identity["account_identity"]:
            reasons.append("ACCOUNT_IDENTITY_MISMATCH")
            valid["/api/v3/account"] = False
        if (journal.identity["episode_id"] != self.store.identity["account_episode_id"]
                or journal.identity["scope_id"] != self.store.identity["portfolio_scope_id"]):
            reasons.append("JOURNAL_NAMESPACE_MISMATCH")
        projection = journal.projection(marks)
        balances = projection["balances"]
        returned = {}
        try:
            if not isinstance(account_data.get("balances"), list):
                raise AdapterError("SCHEMA_INVALID")
            for row in account_data["balances"]:
                if not isinstance(row, dict) or set(row) != {"asset", "free", "locked"}:
                    raise AdapterError("SCHEMA_INVALID")
                asset = _text(row["asset"])
                if asset in returned:
                    raise AdapterError("SCHEMA_INVALID")
                returned[asset] = dict(asset=asset, free=_amount(row["free"]), locked=_amount(row["locked"]))
        except (AdapterError, KeyError, TypeError, ValueError):
            reasons.append("SCHEMA_INVALID")
            valid["/api/v3/account"] = False
        if not valid["/api/v3/account"]:
            returned = {}
        required = {a for a, q in balances.items() if Decimal(q) != 0}
        required.update(a for a, r in returned.items() if r["free"] != 0 or r["locked"] != 0)
        rows = []
        with localcontext(Context(prec=50)):
            for asset in sorted(required):
                expected = Decimal(balances.get(asset, "0"))
                actual = None if asset not in returned else _amount(returned[asset]["free"])+_amount(returned[asset]["locked"])
                delta = None if actual is None else actual-expected
                reason = "ASSET_COVERAGE_MISSING" if actual is None else None if delta == 0 else "BALANCE_MISMATCH"
                if reason:
                    reasons.append(reason)
                rows.append(dict(asset=asset, journal_quantity=expected, exchange_quantity=actual, delta=delta, reason=reason))
        n, total = sum(r["reason"] is None for r in rows), len(rows)
        owned = set(owned_client_ids)
        foreign_orders, foreign_lists = [], []
        try:
            if not valid["/api/v3/openOrders"] or not isinstance(orders["data"], list):
                raise AdapterError("SCHEMA_INVALID")
            foreign_orders = [o for o in orders["data"] if _text(o["client_order_id"]) not in owned]
        except (AdapterError, KeyError, TypeError):
            valid["/api/v3/openOrders"] = False
            reasons.append("SCHEMA_INVALID")
        try:
            if not valid["/api/v3/openOrderList"] or not isinstance(lists["data"], list):
                raise AdapterError("SCHEMA_INVALID")
            foreign_lists = [o for o in lists["data"] if _text(o["list_client_order_id"]) not in owned
                             or any(_text(c["client_order_id"]) not in owned for c in o["orders"])]
        except (AdapterError, KeyError, TypeError):
            valid["/api/v3/openOrderList"] = False
            reasons.append("SCHEMA_INVALID")
        if foreign_orders:
            reasons.append("FOREIGN_OPEN_ORDERS")
        if foreign_lists:
            reasons.append("FOREIGN_OPEN_LISTS")
        if not population_complete or not population_symbols:
            reasons.append("POPULATION_UNKNOWN")
        for symbol in population_symbols:
            status = self.store.recovery_status(symbol)
            if not status["origin_known"] or not status["tail_exhausted"]:
                reasons.append("FILL_HISTORY_UNCERTIFIED")
        if not stream_health or not stream_health.get("transport_healthy") or not stream_health.get("connected"):
            reasons.append("PRIVATE_STREAM_UNHEALTHY")
        reasons.extend(r["data"]["reason"] for r in self.store.incidents())
        existing = {(e["payload"]["symbol"], str(e["payload"]["trade_id"])): e["payload"] for e in journal.events()}
        missing, conflict = 0, 0
        for row in self.store.fills():
            f = row["data"]
            if f["exchange_event_time_ms"] <= journal_anchor_ms:
                continue
            recorded = existing.get((f["symbol"], str(f["trade_id"])))
            if recorded is None:
                missing += 1
                continue
            fields = ("quantity", "quote_quantity", "price", "commission_quantity")
            if (any(Decimal(recorded[k]) != Decimal(f[k]) for k in fields)
                    or recorded["side"] != f["side"] or str(recorded["order_id"]) != str(f["order_id"])
                    or (Decimal(f["commission_quantity"]) != 0 and recorded["commission_asset"] != f["commission_asset"])):
                conflict += 1
        if missing:
            reasons.append("JOURNAL_FILL_MISSING")
        if conflict:
            reasons.append("JOURNAL_FILL_CONFLICT")
        result = dict(**FIXED, state="ACCOUNT_UNRECONCILED" if reasons else "READ_CHECK_MATCH",
            exchange_reconciled=False, order_send_enabled=False, assets=rows,
            balance_check="UNKNOWN" if not valid["/api/v3/account"] else "BALANCES_MATCH" if n == total else "BALANCES_DIFFER",
            matched_assets_n=n, assets_N=total, matched_assets_ratio=None if total == 0 or not valid["/api/v3/account"] else Decimal(n)/Decimal(total),
            exchange_fill_denominator=None, missing_journal_fills_n=missing, conflicting_journal_fills_n=conflict,
            foreign_open_orders_n=len(foreign_orders) if valid["/api/v3/openOrders"] else None,
            foreign_open_lists_n=len(foreign_lists) if valid["/api/v3/openOrderList"] else None, reason_codes=sorted(set(reasons)),
            journal_source_sha256=projection["source_sha256"], evidence_generation=generation,
            checked_at_ms=now, journal_anchor_ms=journal_anchor_ms,
            input_hashes=[value.get("source_sha256") for value in (account, orders, lists)])
        result["comparison_id"] = self.store.save_comparison(result)
        return result
