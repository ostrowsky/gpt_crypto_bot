"""Complete registered-category discovery and received-as-of membership."""

from copy import deepcopy
from decimal import Decimal
import json

from .adapter import AdapterError, _integer, _text, _sha

PERMISSIONS = ("SPOT", "MARGIN", "LEVERAGED") + tuple(f"TRD_GRP_{n:03d}" for n in range(2, 26))
STATUSES = frozenset(("TRADING", "END_OF_DAY", "HALT", "BREAK", "CANCEL_ONLY"))
ORDERS = frozenset(("LIMIT", "MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT", "TAKE_PROFIT", "TAKE_PROFIT_LIMIT", "LIMIT_MAKER"))
FILTERS = frozenset(("PRICE_FILTER", "PERCENT_PRICE", "PERCENT_PRICE_BY_SIDE", "LOT_SIZE", "MIN_NOTIONAL", "NOTIONAL", "ICEBERG_PARTS", "MARKET_LOT_SIZE", "MAX_NUM_ORDERS", "MAX_NUM_ALGO_ORDERS", "MAX_NUM_ICEBERG_ORDERS", "MAX_POSITION", "TRAILING_DELTA", "MAX_NUM_ORDER_AMENDS", "MAX_NUM_ORDER_LISTS"))
SYMBOL_FIELDS = dict(symbol="symbol",baseAsset="base",quoteAsset="quote",status="status",
                     isSpotTradingAllowed="spot_allowed",permissionSets="permission_sets",filters="filters",orderTypes="order_types")


def symbol_source(row):
    if "raw" in row:
        return deepcopy(row["raw"])
    result = deepcopy(row["metadata_extra"])
    result.update({key:deepcopy(row[field]) for key,field in SYMBOL_FIELDS.items() if key in row["source_fields"]})
    result.update(deepcopy(row.get("source_overrides",{})))
    return result


def universe_source(snapshot):
    data = snapshot["data"]
    if data.get("format_version") != "universe-v2":
        return deepcopy(data["observation"]["data"])
    result = deepcopy(data["observation"]["source_header"])
    rows = {r["symbol"]:r for r in data["rows"]}
    result["symbols"] = [symbol_source(rows[s]) for s in data["source_symbol_order"]]
    return result


class UniverseService:
    def __init__(self, store, client):
        self.store, self.client = store, client

    def current(self, cutoff_ms):
        result = self.store.as_of("UNIVERSE", cutoff_ms)
        if result and self.store.last_failure("UNIVERSE_REFRESH_FAILED", "UNIVERSE", cutoff_ms):
            result["data"].update(state="UNIVERSE_REFRESH_FAILED", stale=True)
        return result

    def _account_permissions(self, account, now):
        if account is None:
            return None, "PUBLIC_REFERENCE"
        try:
            for k in ("application_id", "environment", "objective_contract_id", "account_alias", "account_episode_id", "portfolio_scope_id"):
                if account[k] != self.store.identity[k]:
                    raise ValueError()
            if (account["source_endpoint"] != "/api/v3/account" or account["source_method"] != "GET"
                    or account["source_sha256"] != _sha(account["data"])
                    or not isinstance(account.get("credential_generation"), str) or not account["credential_generation"]):
                raise ValueError()
            received = _integer(account["received_at_ms"])
            started = _integer(account["request_started_at_ms"])
            if not 0 <= now-received <= 30_000 or started > received or now-started > 30_000:
                raise ValueError()
            if not 0 <= self.client.clock.monotonic_ns()-_integer(account["received_monotonic_ns"]) <= 30_000_000_000:
                raise ValueError()
            if account["clock_uncertainty_ms"] is None or not 0 <= Decimal(str(account["clock_uncertainty_ms"])) <= 250:
                raise ValueError()
            data = account["data"]
            if account.get("schema_version") != 1 or not set(data) <= {"permissions", "can_trade", "account_identity", "balances", "update_time_ms", "returned_assets", "coverage"}:
                raise ValueError()
            if _integer(data["update_time_ms"]) > self.client.clock.timestamp_ms() + 250:
                raise ValueError()
            token = data["account_identity"]
            if not isinstance(token, str) or len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
                raise ValueError()
            permissions = data["permissions"]
            if not isinstance(permissions, list) or not permissions or not set(permissions) <= set(PERMISSIONS):
                raise ValueError()
            if type(data["can_trade"]) is not bool:
                raise ValueError()
            adapter_identity = getattr(self.client.client, "identity", {})
            if adapter_identity and any(account.get(k) != v for k, v in adapter_identity.items()):
                raise ValueError()
            for snapshot in self.store.records("UNIVERSE"):
                prior = snapshot["data"].get("account_observation")
                if snapshot["data"]["state"] == "UNIVERSE_READY" and prior:
                    if token != prior["data"]["account_identity"]:
                        raise ValueError()
                    break
            return set(permissions) if data["can_trade"] else set(), "UNIVERSE_READY"
        except (AdapterError, KeyError, TypeError, ValueError, ArithmeticError):
            return None, "UNIVERSE_INCOMPLETE"

    def refresh(self, account=None):
        try:
            return self._refresh(account)
        except AdapterError as error:
            self.store.append("INCIDENT", dict(reason=error.reason, operation="UNIVERSE_REFRESH_FAILED"),
                              received_at_ms=self.client.clock.wall_ms(), received_monotonic_ns=self.client.clock.monotonic_ns())
            raise

    def _refresh(self, account=None):
        try:
            observation = self.client.read_exchange_info()
        except AdapterError as error:
            now, mono = self.client.clock.wall_ms(), self.client.clock.monotonic_ns()
            self.store.append("INCIDENT", dict(reason=error.reason, operation="UNIVERSE_REFRESH_FAILED"),
                              received_at_ms=now, received_monotonic_ns=mono)
            previous = self.current(now)
            if previous:
                previous["data"].update(state="UNIVERSE_REFRESH_FAILED", stale=True, last_error=error.reason)
                return previous
            raise
        raw = observation["data"]
        try:
            if not isinstance(raw, dict) or not isinstance(raw["symbols"], list) or raw["timezone"] != "UTC":
                raise AdapterError("METADATA_SCHEMA_INVALID")
            _integer(raw["serverTime"])
            now = observation["received_at_ms"]
            permissions, state = self._account_permissions(account, now)
            rows, seen = [], {}
            for source in raw["symbols"]:
                symbol = _text(source["symbol"])
                if symbol in seen:
                    if seen[symbol] != source:
                        raise AdapterError("METADATA_CONFLICT")
                    continue
                seen[symbol] = source
                base, quote = _text(source["baseAsset"]), _text(source["quoteAsset"])
                status = _text(source["status"])
                reasons = []
                groups = source.get("permissionSets")
                structure = (isinstance(groups,list) and bool(groups) and all(isinstance(g,list) and bool(g)
                    and all(isinstance(p,str) and bool(p) for p in g) for g in groups))
                known = structure and all(all(p in PERMISSIONS for p in g) or
                    (permissions is not None and any(p in permissions for p in g)) for g in groups)
                unknown_alternatives=sorted({p for g in groups for p in g if p not in PERMISSIONS}) if structure else []
                if not known:
                    reasons.append("PERMISSIONS_UNVERIFIED")
                    if account is not None:
                        state = "UNIVERSE_INCOMPLETE"
                spot = source.get("isSpotTradingAllowed")
                if type(spot) is not bool:
                    reasons.append("SPOT_FLAG_UNVERIFIED")
                    state = "UNIVERSE_INCOMPLETE" if account is not None else state
                permitted = (all(any(p in permissions for p in g) for g in groups) and spot) if known and permissions is not None and type(spot) is bool else None
                if status not in STATUSES:
                    reasons.append("STATUS_UNVERIFIED")
                elif status != "TRADING":
                    reasons.append("NOT_TRADING")
                if quote != "USDT":
                    reasons.append("OUTSIDE_PILOT_QUOTE")
                filters = source.get("filters")
                if not isinstance(filters, list) or any(not isinstance(f, dict) or f.get("filterType") not in FILTERS for f in filters):
                    reasons.append("FILTER_UNVERIFIED")
                orders = source.get("orderTypes")
                if not isinstance(orders, list) or not orders or any(o not in ORDERS for o in orders):
                    reasons.append("ORDER_TYPES_UNVERIFIED")
                if permitted is not True:
                    reasons.append("ACCOUNT_NOT_PERMITTED" if permitted is False else "ACCOUNT_UNVERIFIED")
                rows.append(dict(symbol=symbol, base=base, quote=quote, status=status,
                                 spot_allowed=spot, permission_sets=groups, permitted=permitted,
                                 filters=filters, order_types=orders,
                                 metadata_extra={k:v for k,v in source.items() if k not in SYMBOL_FIELDS},
                                 source_fields=sorted(set(source)&set(SYMBOL_FIELDS)),
                                 monitored=True, metadata_entry_candidate=not reasons,
                                 execution_certified=False, reasons=reasons,
                                 unknown_permission_alternatives=unknown_alternatives))
            previous = self.current(now)
            old = {r["symbol"]: r for r in previous["data"]["rows"]} if previous else {}
            transitions = []
            for row in rows:
                prior = old.get(row["symbol"])
                if prior is None or prior["status"] == "REMOVED":
                    transitions.append(dict(symbol=row["symbol"], type="ADDED"))
                elif prior["status"] != row["status"]:
                    transitions.append(dict(symbol=row["symbol"], type="STATUS_CHANGED", previous_status=prior["status"], status=row["status"]))
            for symbol, prior in old.items():
                if symbol not in seen:
                    if prior["status"] != "REMOVED":
                        transitions.append(dict(symbol=symbol, type="REMOVED"))
                    tombstone = deepcopy(prior)
                    tombstone.setdefault("source_overrides",{}).setdefault("status",prior["status"])
                    tombstone.update(status="REMOVED", permitted=False, metadata_entry_candidate=False,
                                     reasons=["REMOVED_FROM_METADATA"], execution_certified=False)
                    rows.append(tombstone)
            count = sum(r["permitted"] is True and r["status"] != "REMOVED" for r in rows) if state == "UNIVERSE_READY" else None
            compact_observation = {k:v for k,v in observation.items() if k != "data"}
            compact_observation["source_header"] = {k:v for k,v in raw.items() if k != "symbols"}
            data = dict(state=state, rows=rows, transitions=transitions, stale=False,format_version="universe-v2",
                        source_symbol_order=[r["symbol"] for r in raw["symbols"]],
                        counts=dict(discovered_pairs_n=len(seen), permitted_pairs_N=count),
                        permission_registry_version="spot-permissions-2026-10-11",
                        observation=compact_observation, account_source_sha256=account.get("source_sha256") if isinstance(account, dict) else None,
                        exchange_filters=raw.get("exchangeFilters"), rate_limits=raw.get("rateLimits"))
            account_fields = {"application_id", "environment", "objective_contract_id", "account_alias", "credential_generation", "account_episode_id", "portfolio_scope_id", "schema_version", "data", "source_endpoint", "source_method", "source_sha256", "received_at_ms", "request_started_at_ms", "received_monotonic_ns", "clock_uncertainty_ms", "exchange_reconciled"}
            data["account_observation"] = {k: v for k, v in account.items() if k in account_fields} if isinstance(account, dict) and permissions is not None else None
            if data["account_observation"]:
                # P1 hashes fixed-point Decimal spelling; market canonical money
                # strips zeros, so preserve the P1 object's strings explicitly.
                data["account_observation"]["data"] = json.loads(json.dumps(account["data"],
                    default=lambda value: format(value, "f") if isinstance(value, Decimal) else None,
                    allow_nan=False))
            return self.store.append("UNIVERSE", data, received_at_ms=now,
                                     received_monotonic_ns=observation["received_monotonic_ns"])
        except (KeyError, TypeError, ValueError):
            raise AdapterError("METADATA_SCHEMA_INVALID") from None
