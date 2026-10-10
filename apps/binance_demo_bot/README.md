# Independent Binance Demo bot — P0-A financial core

This separate application implements offline Decimal accounting, valuation,
daily equity PnL, FIFO/fees and initial path/configuration isolation. It does
not send orders, run learning workers or connect to Binance. SQLite/outbox,
reconciliation adapters and the P1–P7 services remain planned.

Specifications: [financial contract](../../docs/specs/binance-demo-phase0-financial-contract.md),
[application isolation](../../docs/specs/binance-demo-application-isolation.md).
Development order: **SPEC → TESTS → CODE**; see [AGENTS.md](AGENTS.md).

From `apps/binance_demo_bot` on this workstation:

```powershell
uv venv --python C:\Python314\python.exe --no-config --cache-dir .runtime/uv-cache .venv
uv sync --locked --no-config --cache-dir .runtime/uv-cache
.venv\Scripts\python.exe -I -m unittest discover -s tests -p 'test_*.py' -v
.venv\Scripts\python.exe -I -m binance_demo_bot
```

Use a separately installed Python 3.14 on other machines. The lock freezes
runtime dependencies; the build backend is explicitly pinned. All cache,
fixtures and runtime evidence belong under this app's ignored `.runtime`.
The existing bot's Python, environment, state and workers are not dependencies.

`-I` prevents CWD/PYTHONPATH from substituting an installed package. The offline
entrypoint also rejects any nonempty inherited PYTHONPATH and returns JSON
with `order_send_enabled=false`. It reads no credentials and starts no services.
The current entrypoint supports this editable project layout, not a deployed
wheel/service. Always test future deployment packaging separately.

When the account adapter is implemented, provision a dedicated demo pair in
this app's ignored `.env`, based on `.env.example`, or pass the complete pair
explicitly. No parent-env discovery, old-config imports or legacy key fallback.
Real credentials have not been copied or validated in this milestone.

`FinancialLedger` is an **in-memory projection**, not durable production storage.
Its caller supplies contemporaneous marks; daily reporting takes certified
snapshots and explicit reconciliation status. The kernel cannot itself certify
exchange coverage or absence of manual account activity. Do not enable trading
until durable ledger, account ownership, adapter, OMS/risk and release gates pass.
