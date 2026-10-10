# Independent Demo bot: first P0 test batch

Specification: [P0-A public contract](../../../docs/specs/binance-demo-phase0-financial-contract.md#p0-a-публичный-контракт-первой-партии-unit-тестов).
Required order: **SPEC → TESTS → CODE** in [app rules](../AGENTS.md).

These are offline behavioral tests of the future `binance_demo_bot.financial`
module. No application implementation is included in this test-only milestone.
Inputs are synthetic and contain no credentials or private exchange payloads.
Expected numbers are fixed independently of the future implementation.

| Test file | Spec coverage in this batch |
|---|---|
| `test_financial_valuation.py` | FIN-01 reserve identity; FIN-04/06 full inventory, dust and stablecoins; FIN-05 DST/event-day boundaries; FIN-06 conversion routes, historical clocks and quote freshness; FIN-10 invalid balances. |
| `test_financial_ledger.py` | FIN-01 partial fills/FIFO; FIN-02 base/quote/BNB fees and non-USDT quote disposal; FIN-07 duplicate/conflicting fills; FIN-10 atomic accounting failures; FIN-12 seed/foreign/unattributed ownership. |
| `test_daily_performance.py` | FIN-03 flows and episode boundaries; FIN-04 no-trade MTM and null denominators; FIN-08 flow-adjusted return/TWR; FIN-09 equity reward vs proceeds/extra costs; FIN-10 reconciliation; FIN-11 calendar gaps and adverse corrected revisions. |

Run from repository root, using the bundled interpreter as an **engineering
bootstrap runner only** (not a new app runtime dependency):

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
pyembed\python.exe -m unittest discover -s apps/binance_demo_bot/tests -p 'test_*.py' -v
```

After the app has its own environment and package, the same tests must run with
that environment's Python. Tests add only the new app's own `src` to the import
path. They do not import the existing bot or read its config/state/environment.

Verified 2026-10-10: **64 discovered tests, 64 errors, exit code 1 (expected RED)**.
At this milestone each discoverable test fails
in `setUp` with `ModuleNotFoundError: No module named 'binance_demo_bot'`.
This is the intentionally missing public interface, permitted by the app's TDD
contract. The financial assertions have **not passed or executed yet**. There
are no skips, xfail cases, substitute implementation or always-green mocks.
The next implementation step must make these unchanged contracts GREEN.

Not covered yet: SQLite transactions/WAL/outbox, restart recovery, immutable
report storage, exchange reconciliation/actual fills, episode reset evidence,
cross-arm sum, account exclusivity and full process/environment isolation.
FIN scenario labels identify partial contract coverage, not completion of all
phase acceptance criteria. P1–P7 test batches remain planned. Passing this
batch later will not certify profitability or enable order sending.

Local RED output belongs under `../.runtime/test-evidence/` and is not committed.
