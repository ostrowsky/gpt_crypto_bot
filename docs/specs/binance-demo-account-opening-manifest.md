# Demo account ownership and provisional opening inventory (P0-C1)

Date: 2026-10-27. Status: implemented; 39 focused GREEN; final authenticated
capture and regression evidence recorded below. Certification is provisional.
Application: `apps/binance_demo_bot` only. Objective: `daily_net_equity_pnl_v1`.

## Purpose and scope

The user explicitly attested that the Demo account is dedicated to this new bot,
without manual trading or other order-placing applications. Persist that declaration
against the actual HMAC account identity, then preserve a source-backed candidate
for starting inventory. Empty order lists alone do not establish exclusivity.
This supplies opening evidence for [P0](binance-demo-phase0-financial-contract.md);
it does not activate a financial episode, certify reconciliation or permit orders.

## Contract

`AccountOpeningManifest` uses an app-owned WAL/FULL `AuditJournal` database,
immutable namespace and hash chain. No credentials, raw UID or legacy state.
`attest(account_identity, declaration_id, declared_at_ms, exclusive, now_ms)`
requires explicit boolean true, a 64-character HMAC identity, identifier-only
declaration reference and nonfuture UTC receipt time. Exact retry is idempotent;
changing an existing attestation conflicts. Persist before returning success.
On verified reopen the same declaration survives; unknown commit poisons the
current writer, including cached reads. An unexplained pre-attestation history
must not silently create a new ownership declaration.

`capture(candidate_id, account_before, account_after, orders, order_lists,
metadata, tickers, cutoff_ms)` requires a declaration. Inputs are normalized
adapter observations, not arbitrary balances or user-written equity. Verify
GET endpoint/environment, source hash, durable audit references, receipt/request
wall and monotonic order, clock uncertainty <=250ms and a common private namespace,
credential generation and HMAC account identity. Every account balances row is
unique, nonnegative exact decimal and agrees with returned-assets metadata.
Asset and market-symbol identifiers are exact opaque Unicode word strings,
up to 64 characters, as permitted by the adapter's market reader. Do not assume
ASCII, uppercase, transliterate, case-fold or merge visually similar names.
Zero-valued non-ASCII assets still belong to the returned-schema check; nonzero
unpriced non-ASCII holdings block valuation just like any other holding.
The account response coverage remains RETURNED_BALANCES_ONLY, not a claim of an
exchange-wide inventory schema. All nonzero returned free+locked holdings enter
the valuation denominator, including dust and nontradable holdings.

Before/after account reads must bracket open-orders, open-order-lists and ticker
requests. All private observations must be <=30s old at cutoff; metadata <=15min;
quotes <=5s old. These are observed-response bounds, not exchange-event freshness.
Malformed/future/stale/hash/identity errors reject capture without overwriting a
previous candidate. Metadata and quotes are matched by symbol, base and quote;
no implicit stablecoin peg. Registered mark convention: Demo bid/ask midpoint,
OBSERVED_RESPONSE, exchange quote age unknown. Deterministic shortest lexical
routes, up to three edges, end in USDT. Missing routes/quotes or invalid books
prevent complete valuation; never remove the held asset or replace its value
with zero. Fixed precision arithmetic is independent of caller Decimal context.

Changed balances, any nonzero locked balance, any open order/list, or absent
Spot trade permission produce a BLOCKED candidate. No adoption, cancellation,
selling or repair of such inventory/orders. Equal before/after balances are an
observation check only, not proof that no intervening trades/flows occurred.
With stable balances and complete valuation the state is PROVISIONAL_OPENING.
Opening lots have ownership SEED and total basis quantity * opening mark;
managed inventory is zero. No rounding to trade filters or sale authorization.
If valuation is incomplete, state is BLOCKED and equity/basis are not invented.

Every saved candidate freezes declaration, normalized nonzero holdings, seed
lots, valuation, source hashes/audit links, receipt bounds, code hash and reasons.
Candidate ID retry returns the same content; conflicting payload is rejected.
At most 100 candidates per database; return detached copies. Invalid inputs do
not append a candidate. Durable source responses remain in the accompanying own
audit/evidence store; public metadata is pinned by hash rather than duplicated
unboundedly. Caller must preserve that source evidence. Inputs are trusted local
adapter objects; a matching hash is not a cryptographic proof of Binance origin.

Every candidate has exchange_reconciled=false, financial_episode_opened=false,
execution_authorized=false, daily_net_pnl_usdt=null and opening equity is NOT
income. A mid-day candidate is not a midnight/day anchor. Historical stream gaps,
fill/flow coverage, reset evidence and certified cut/recovery remain unresolved.
SEED initialization cannot serve as independent financial reconciliation.

## Acceptance scenarios (synthetic, offline)

| ID | Required evidence |
|---|---|
| OPN-01 | Explicit exclusivity, identity, declaration time and exact retry; reject false/missing/foreign/conflicting declaration. |
| OPN-02 | Full namespace, generation, endpoint/hash/audit provenance and both clocks; reject stale/future/misordered input. |
| OPN-03 | Bracketed stable inventory, full returned denominator; duplicate/negative/float balances rejected. |
| OPN-04 | USDC marked from actual book; direct/inverse/multileg deterministic conversion; unknown/dust asset blocks completeness. |
| OPN-05 | Changing balances, locks, open orders/lists or permission denial block; no side effects or managed SEED sales. |
| OPN-06 | Exact SEED basis, caller Decimal context, no profit/day/episode/trade certification. |
| OPN-07 | Immutable candidate retries/conflicts, detached reads, reopen, namespace isolation, writer failure and uncertain commit. |

## Validation and rollout

Apply TH-01..12, focused RED/GREEN, affected regressions, root spec checks,
diff review and staged Harness. Run one bounded authenticated read-only capture
against the already authorized dedicated Demo account, retain balances only in
ignored app runtime, scan for credentials/raw UID and report its actual state.
No trading-policy relaxation: maximum-history replay and fresh canary still
required for a future trading policy. Rollback preserves evidence and blocks
entries; it never clears source gaps or rewrites opening capital.

Implementation: `apps/binance_demo_bot/src/binance_demo_bot/account_opening.py`;
tests: `apps/binance_demo_bot/tests/test_account_opening.py`.
Initial26 missing-module RED preceded implementation. Eight additional contract
boundaries passed without artificial failure. Authenticated read-only integration
then exposed two genuine adapter-format mismatches: Decimal clock uncertainty
(1 regression RED) and Unicode asset/symbol names (3 regression RED). Both were
repaired after their tests;39 focused tests now pass, with limits preserved.
An offline diagnostic on the retained normalized source observations validates
all6 source envelopes,866 returned balance rows, bracketing and COMPLETE midpoint
valuation after those repairs. This replay is historical diagnostic evidence,
not a fresh snapshot or financial reconciliation. Both rejected capture attempts
and the original immutable declaration remain in ignored own runtime.

Final fresh authenticated capture (2026-10-27, final source):
PROVISIONAL_OPENING,866 returned assets,2 nonzero holdings,2 SEED lots,
COMPLETE observed Demo midpoint valuation,0 managed assets and0 open orders/lists.
Opening capital/balances are retained only in ignored runtime. Source audit
records8/8 REST BEGIN and END with0 unresolved attempts; these are durable
application counters, not wire or financial completeness. Human declaration
remains the same across all three attempts; no economic reset or historical
stream-gap clearing occurred. No matching-engine requests were made.
Privacy scan:28/28 JSON files and66/66 records across three source audits plus
the opening-manifest chain,0 credential values/raw UID fields found.

Final own full suite:829 GREEN (97.268s); focused39 GREEN (5.272s), root14 spec
tests GREEN. `git diff --check` passes. Initial and subsequent RED evidence plus
final outputs are in ignored app `.runtime/test-evidence/`.
Full Truth Harness remains FAIL TH-11 because the legacy canonical portfolio
alpha report pins a stale replay source hash. Unit success and this provisional
opening do not override that failure or prove trading profitability. Next:
certified cut/recovery and fill/flow anchors, audited sender/risk/protection,
then qualified bounded Demo canary; ORDER_SEND remains disabled.
