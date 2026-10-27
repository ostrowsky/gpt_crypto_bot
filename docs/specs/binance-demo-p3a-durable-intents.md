# Binance Demo P3-A: durable order intents and reservations

Registered: 2026-10-27. Status: P3-A implemented; 39 focused /746 app GREEN; execution disabled.
Parent: [phase 3](binance-demo-phase3-order-management-risk.md).
Objective: `daily_net_equity_pnl_v1`; environment: `BINANCE_SPOT_DEMO`.

## Purpose and authority

Move execution development to the first OMS slice. A pending intent must survive
restart without losing its reservation, acquiring another arm's budget, or being
sent twice after an uncertain result. This slice is a local journal, not a broker,
exchange reconciliation, risk approval, or a profitable policy. It supplies no
network client, credentials, runnable trading worker, release ticket or order-send
capability. `execution_authorized` is always false. No legacy runtime changes.

Actual Demo trading remains blocked until dedicated credentials, signed account
checks, certified ownership/valuation, fresh market data, full pre-send risk and
exchange-filter checks, audited sender and verified TP/SL/recovery are available.
The last all-market latency verdict is FAIL. A new policy requires causal replay
on the maximum available history and separate forward/canary evidence. Neither
the tests below nor arbitrary constructor balances satisfy these gates.

## Interface and durable contract

`OrderIntentJournal(paths, *, account_alias, episode_id, scope_id, allocations,
max_intents=1000, fault_hook=None)` owns `oms-intents.sqlite` only. It reuses the
existing AuditJournal OS writer lease, immutable hash chain and WAL/FULL ACK
contract. A successful RESERVE record contains the complete immutable intent,
its reservation and pending outbox identity in one durable event. There is no
separately committed reserve or dispatch row. Partial/unknown commits make the
instance fail closed; a verified reopen may recover committed records.

`allocations` is a nonempty mapping of arm IDs to nonnegative asset amounts.
They are explicit local seed budgets, not certified exchange balances, and may
not change on reopen. Commission reserve belongs to the debited asset: quote
for BUY, base for SELL. Multi-asset fee reservation, fills/consumption, allocation
top-ups and terminal release of possibly-sent intents are deferred to P3-B.
Unsent cancellation alone may release a reservation. No automatic balancing,
borrowing, deriving SELL inventory from requested BUYs, or base-asset minting.

`reserve(intent, *, now_ms)` returns a detached current state. Exact fields:
`arm_id, decision_id, action_seq, symbol, base_asset, quote_asset, side, quantity,
limit_price, fee_reserve, take_profit, stop_loss, asof_ms, expires_at_ms,
market_snapshot_id, account_snapshot_id, policy_version, model_id,
source_sha256, risk_contract_id`. Only LIMIT BUY/SELL for USDT-quoted symbols;
symbol must equal base+quote. BUY requires `0 < stop_loss < limit_price <
take_profit`; SELL requires both levels null. Prices/amounts accept finite
Decimal or decimal text, never binary float, bool, NaN or infinity. Quantity
and price are positive; fee nonnegative. Decimal arithmetic must be exact even
under the caller's low-precision decimal context. Extreme exponents/digits are
rejected within a documented bound: at most 28 significant digits and absolute
exponent <=18. Product/sum/reservation arithmetic uses precision 96.

Snapshot IDs and versions are bounded, nonempty simple identifiers, not arbitrary
messages. Source digest is 64 lowercase hex. Snapshot availability is represented
by `asof_ms`; require `0 <= asof_ms <= now_ms < expires_at_ms`. This is caller
provenance, not certification of as-of correctness. Logical key is
`(arm_id, decision_id, action_seq)`. Same canonical payload returns the existing
state even after expiry/kill/cancel/restart; it appends a RETRY decision linking
the existing ID and state version, without another reservation. Conflicting reuse fails. Different
decisions and action sequences are independent. Deterministic client order ID
(`bd_` plus 32 hex, <=36 characters) includes episode/scope/account and logical
key; it never changes across attempts and is not an exactly-once guarantee.

BUY reserve = quantity*limit_price + fee_reserve in USDT. SELL reserve =
quantity + fee_reserve in base. Sum of noncanceled reservations per arm/asset
cannot exceed that arm's allocation. Canceled tombstones count toward capacity,
so IDs cannot be reused. Only `RESERVED`, `SEND_STARTED`, `QUERY_REQUIRED`,
`CANCELED` exist in this slice; none means FILLED or OPEN_PROTECTED.

Each state has increasing `state_version`. `annotate_send_started(intent_id,
*, expected_version, lease_epoch, now_ms)` records a local attempt annotation
before a future sender could perform I/O, requires RESERVED/unexpired, current
epoch and exact version; it DOES NOT authorize I/O. It returns no transport or
release capability. A second annotation cannot create another send. `mark_unknown`
requires SEND_STARTED and moves to QUERY_REQUIRED without releasing funds.
`cancel_unsent` requires RESERVED and explicit reason; possibly-sent/unknown
intents cannot be canceled/released here. All transitions are audited before ACK.

Every open appends an EPOCH event while holding the OS lease. On reopen,
SEND_STARTED becomes QUERY_REQUIRED before serving the instance; it retains its
reservation. Old epoch and stale CAS operations fail. This is local journal
fencing, not proof of fencing a remote Binance request. An incomplete run or
unknown attempt must not generate a new order. A sticky `kill_entries(reason)`
blocks new BUY reservation/attempts, persists across restart, and preserves SELL
and existing reservations. There is no automatic unkill method in this slice.

`get(intent_id)` returns a detached view; `summary()` includes per-arm/asset
allocated/reserved/available decimal text, intents N, unresolved n/N, entry kill,
lease epoch, local-only authority and chain pin. Unknown denominator yields null.
Startup verifies event schema, canonical intent/reservation, replay transitions,
CAS, budgets, epoch sequence and audit hashes, not only JSON readability.
BLOCK references, when present, must be canonical intent IDs; malformed BLOCKs
cannot masquerade as valid rejection evidence. Malformed or illegal events fail closed, including bool substituted for integer
versions/epochs or false authority. A pristine empty audit database may initialize;
a nonempty history missing its OMS genesis must fail, not seed new balances.
All operations serialized under the
journal lock. Bound intent count and durable DECISION count (<=10000); exhaust
capacity by failing, never pruning tombstones. No cross-account DB reuse.

## Test-first acceptance scenarios

| ID | Observable contract |
|---|---|
| P3A-01 | Atomic durable reservation+outbox, canonical Decimal, provenance and no execution authority |
| P3A-02 | Same-key retries stable; changed payload conflict; distinct seq/episode separate IDs |
| P3A-03 | Arm/asset aggregate budgets, BUY fees and SELL base fees; no borrowed inventory |
| P3A-04 | Reject invalid schemas, symbols, float/nonfinite/oversize amounts, future/expired sources and protection |
| P3A-05 | CAS/epoch/expiry and exactly one SEND_STARTED annotation; unknown retains reserve |
| P3A-06 | Unsent cancellation only, tombstones durable, immutable returned views |
| P3A-07 | Restart recovery, persistent kill, old epoch denied, exclusive writer/identity |
| P3A-08 | Native SQLite rollback and uncertain post-commit ACK failure; sticky failure and verified reopen |
| P3A-09 | Native thread budget race; integrity/schema tampering and bounded capacity denied |
| P3A-10 | Auditable rejection decisions, reason codes, no fabricated fills/PnL/protection or send method |

## Verification and rollback

Use app-owned fixtures without keys/network; native SQLite triggers, competing
threads, process death and reopen verify real storage/concurrency. Both uncertain
reservation ACK and uncertain SEND_STARTED ACK are tested; the latter must become
QUERY_REQUIRED on restart. Record expected RED before
code, then focused GREEN, app regressions, diff review, staged Truth Harness,
commit/push. TH-01..12 apply; existing full TH-11 FAIL remains separate.
Test-first evidence:29 missing-module RED, then a fixture connection-lifecycle
correction (SQLite context managers commit but do not close connections).
Four added boundaries reproduced3 failures before repair: boolean state version,
missing genesis, unaudited idempotent retry. Six further cases reproduced1
malformed-BLOCK failure; the preserving epoch/CAS, transaction rollback,
uncertain started ACK and actual process-death cases already passed. Final39
focused GREEN (8.652s),746 app GREEN (148.261s),14 root spec tests GREEN;
no skips/xfail or weakened assertions. Decimal expected value was independently
checked using rational arithmetic before implementation. No credential reads,
exchange calls, live orders or profitability evidence. Full Harness remains
FAIL TH-11 because historical portfolio alpha has a stale replay source hash.

Rollback means stop using this local journal interface while preserving its
evidence and all reservations. No endpoint is enabled and no risk gate relaxed;
historical policy validation cannot be replaced with synthetic unit tests.
