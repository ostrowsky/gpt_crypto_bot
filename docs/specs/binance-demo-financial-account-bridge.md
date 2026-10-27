# P0-C2: source-backed ledger bridge and independent balance checks

Date: 2026-10-27. Status: implemented;39 focused GREEN; financial certification
remains PENDING. Fresh authenticated/regression verification below.
App: `apps/binance_demo_bot`; objective `daily_net_equity_pnl_v1`.
Parent: [P0 financial contract](binance-demo-phase0-financial-contract.md),
[P0-C1 opening manifest](binance-demo-account-opening-manifest.md).

## Behavior

`FinancialAccount(paths, opening_manifest, candidate_id, ...)` binds an own
WAL/FULL audit database and separate existing `JournalStore` to one persisted
PROVISIONAL_OPENING candidate. Validate namespace, HMAC account identity, hash
chain, complete valuation and SEED lots. Use the original seed basis, never the
latest balance. Persist BIND before creating the ledger, then LEDGER_READY;
reopen verifies the same candidate and accounting fingerprint. A later candidate
cannot rebase this ledger. If a previously ready ledger disappears, fail rather
than silently replacing its trades. A ready ledger that is truncated/empty or
missing its identity record must be rejected before invoking the core's new-file
initialization; do not initialize a replacement seed. BIND-only interrupted initialization can
resume. The frozen financial/journal/isolation core is unchanged.

`reconcile(check_id, account_before, account_after, orders, order_lists,
metadata, tickers, cutoff_ms, evidence)` reuses C1 observation validation:
all source hashes/audit links, namespace/generation, receipt clocks, brackets,
30s account /5s quote /15min metadata bounds, exact Unicode asset identifiers.
Account-before request must be later than the bound opening cutoff. This is an
independent subsequent observation, not a seeded-equality certification.
Retain original full returned-asset membership: an explicitly returned zero is
zero; omission of a previously held asset is UNKNOWN, not zero. Quantities
compare as exact decimals, without ad-hoc tolerances or trade-filter rounding.
Persist returned membership as SHA256 hashes of exact identifiers plus its
count, pinned to the full source response. A legitimate zero-balance asset named
TOKEN or COOKIE must not be mistaken for authentication data or discarded. Never
weaken the global credential redactor to preserve this metadata.

Read an integrity-verified frozen prefix of the own `AccountEvidenceStore`.
Source scope/HMAC must match. Import available post-anchor fills chronologically
using the actual source receipt/process/event identity. Every imported fill is
UNATTRIBUTED; this layer has no OMS ownership authority. No app-managed inventory
or bot profit is created by an external fill. REST/WS duplicates debit once.
Pre-anchor fills are disclosed/excluded; future events or receipts remain pending.

Initial supported fill accounting: metadata observed before the fill, direct
USDT quote, fee in base/USDT or zero. Base-fee mark uses the transaction price;
USDT mark is one. Third-asset fees, non-USDT quote, unavailable historical mapping
or invalid arithmetic remain explicit unimported evidence. Never value an old
fee using today's book. Preserve native quote/price/quantity; do not alter a
fill to make the frozen core accept it. No synthetic fills or inferred transfers.
An already imported economic fill reuses its original immutable mapping/marks
and source recipe. A later metadata refresh must neither reprice it nor falsely
classify that accepted import as missing historical evidence.

Each import uses durable IMPORT_BEGIN with the exact source recipe, then the
idempotent journal write and IMPORT_END. No cross-database atomicity claim.
An unknown journal commit fails the coordinator closed; reopen verifies the
ledger and replays the identical pending recipe, debiting once. Known accounting
or conflict failures are recorded, not discarded. Own audit failure also poisons
cached operations. Before-commit failure must not create an economic event.

Persist an immutable comparison with bound opening hash, original and current
source pins, ledger/evidence prefix hashes, per-asset expected/actual/residual,
matched n / required N, fill dispositions, incidents, locks and orders/lists.
Residuals cause MISMATCH; missing coverage causes UNKNOWN; equal quantities
produce BALANCES_MATCH but financial status remains PENDING. A generation change
during processing invalidates freshness; saved historical checks are never live
permissions. Identical check retry is immutable/idempotent; changed input under
the same ID conflicts. Return detached results and cap comparisons/imports.

Unclassified flow/order/stream incidents remain visible and block certification.
Do not label a loss/reset from a residual, clear stream gaps, or adopt/cancel any
order. Valuation includes every nonzero holding. Unadjusted equity change may
be shown as a diagnostic, but daily_net_pnl_usdt is always null in this slice.
Full-day boundaries, classified flows, cut/delivery/population completeness and
OMS ownership/protection still require integration. financial_episode_opened,
exchange_reconciled and execution_authorized remain false, even if quantities
match and the socket is healthy.

## Acceptance cases

| ID | Required result |
|---|---|
| FAC-01 | Original seed bound once; reject missing/blocked/foreign/rebased candidate; reopen never reseeds. |
| FAC-02 | Independent fresh sources; reject same opening snapshot, tampering, future/stale/foreign/mixed sources. |
| FAC-03 | Exact balance comparison, explicit zero vs omission, Unicode/dust/new asset and locks; no discrepancy repair. |
| FAC-04 | Real stored fill -> UNATTRIBUTED journal entry with original provenance; chronological/late ordering and duplicate fees. |
| FAC-05 | Third-fee/non-USDT/late metadata/pre-anchor/future source coverage disclosed; no current-price historical substitution. |
| FAC-06 | Native commit failures/reopen, missing ledger, immutable checks and writer leases fail closed. |
| FAC-07 | Matching quantities still PENDING; flows/gaps/foreign events and evidence mutation remain visible; no daily-profit certification. |

## Verification and rollback

Offline synthetic tests must demonstrate RED before implementation, then focused
and full regressions, root spec checks, diff review and staged Truth Harness.
Use a bounded authenticated GET-only capture on the authorized dedicated Demo
account to compare the existing frozen opening with new independent observations;
preserve both source attempts and privacy-check own runtime. No matching orders.
No policy relaxation: maximum-history causal replay and forward/canary gates
remain required before later trading enablement. Rollback stops new imports/checks
and entries, preserves all committed fills/bindings and replays their verified log.
Full Harness FAIL must remain FAIL; this bridge is not profitability evidence.

Implementation: `apps/binance_demo_bot/src/binance_demo_bot/financial_account.py`;
tests: `apps/binance_demo_bot/tests/test_financial_account.py`.
30 missing-module RED before source implementation, then30 GREEN. Seven added
boundaries reproduced one genuine historical-metadata refresh failure; repaired
after RED. Review then reproduced silent seed reinitialization after truncating
a ready ledger; read-only identity inspection now rejects it without altering
the damaged file. An actual read-only capture exposed a legitimate returned
zero asset COOKIE colliding with the generic secret redactor. TOKEN/COOKIE
membership regression RED preceded hashed membership storage. Global redaction
was not relaxed. Final39 focused GREEN (14.052s).

The first authenticated attempt was retained as FAIL FINANCE_SOURCE_INVALID;
no comparison or fabricated profitability result was saved. It did create the
original pinned local SEED ledger, which subsequent attempts reopen rather than
rebase. Source replay verifies866 returned assets and COMPLETE valuation but is
explicitly historical diagnostic evidence. A healthy private subscription and
quiet polling do not certify delivery; intentional closure records STREAM_GAP.

Final authenticated independent comparison (2026-10-27, final source):
BALANCES_MATCH,2/2 required nonzero assets;866 returned asset identifiers retained
as membership hashes. The original C1 opening hash/basis remained unchanged.
Stored/imported fills0/0 in the available evidence, global denominator unknown;
no actual economic fill has qualified this importer. Unit fixtures exercise it.
Status PENDING with cut, fill/flow coverage, delivery and prior stream incidents
explicit; no daily PnL and no matching orders. At comparison the background reader
was alive, connected and transport-healthy; private delivery remained uncertified.
After deliberate close evidence generation advanced4->6; saved comparison became
stale. Four incidents (two WS_UNAVAILABLE and two STREAM_GAP) remain from the
two sessions; they were neither cleared nor presented as continuous coverage.
The final source audit retained10/10 REST and11/11 WS BEGIN/END,1/1 processing
marker and10/10 control records,0 unresolved completed operations. These are
application-boundary counters, not a complete wire/event denominator.

Final full regression868 GREEN (138.258s), focused39 GREEN (14.052s), root14
spec tests GREEN, diff check passes. Privacy scan:23/23 JSON files,135/135 audit/
evidence records and2/2 ledger JSON rows,0 credentials/raw UID fields found.
Actual sources, ledger and all reports stay in ignored runtime. Full Truth
Harness remains FAIL TH-11 (stale legacy portfolio replay source hash); staged
change checks are separate. Certified cut/recovery, classified flows, historic
fee conversion, OMS ownership/sender/protection and bounded canary remain required.
