# Independent Binance Demo bot: mandatory development rules

User instruction recorded 2026-10-10. These rules persist for all work in this
application. They supplement the repository rules and the canonical
[spec-first workflow](../../docs/specs/spec-first-workflow.md).

## Required order: SPEC → TESTS → CODE

For every executable code or behavior change, including bug fixes:

1. Write/update and register the behavioral specification before implementation.
   Use the phase's acceptance criteria and named scenarios. Clarify contract
   changes in the spec before changing tests or implementation.
2. Write focused automated tests from that spec **before implementation code**.
   Test observable contracts, normal cases, boundaries and relevant failures.
3. Run those tests before implementation. For new/changed behavior, verify RED:
   they fail for the expected missing/incorrect behavior. A missing intended
   new public interface can establish RED; broken
   tooling, unrelated imports, credentials, network or permissions cannot.
   Record the exact command, failing scenario IDs and reason in runtime evidence
   or the task report. No secrets/account data in committed fixtures.
4. Only after the test-first verification (RED for new/changed behavior), write
   the minimum implementation that makes those tests pass.
5. Verify GREEN, then refactor while keeping tests green. Run appropriate
   regressions, diff checks and the required evidence/release gates.
6. Review and stage only intended spec/test/source files; run staged Truth
   Harness, commit, push and report both verification and implementation status.

Do not write the implementation first and add tests afterward. Do not weaken
assertions, add skips/xfail, rewrite expected outputs from the implementation or
use always-passing mocks to manufacture GREEN. Existing passing tests alone do
not demonstrate a changed contract; add a failing reproduction where applicable.
For behavior-preserving refactoring, strengthen contract coverage first and
record why existing behavior legitimately remains GREEN; do not fake a failure.
Pure documentation updates are not executable implementation and do not require
artificial unit tests; check their links, registration and consistency.

## Independent application boundary

Follow [application isolation](../../docs/specs/binance-demo-application-isolation.md)
and [program priorities](../../docs/specs/binance-demo-autonomous-trading-program.md).
Own package, dependencies, environment, credentials, ledger, models, UI and
processes. No runtime imports or state/config/credential fallback from the
current bot; never stop or mutate its workers. Tests live under this app's
`tests/` and use app-owned fixtures. No real orders or credential reads in unit
tests. Runtime, keys, balances, logs and trained artifacts are never committed.

## Evidence is separate from green tests

Green tests establish implementation correctness for their scope. They do not
prove earnings, current account permission or profitable deployment. Apply
TH-01..TH-12 and the maximum available causal portfolio backtest plus fresh
forward/canary gates before adopting trading-policy changes. Keep risk limits,
unknowns, rejected hypotheses, rollback and actual/simulated evidence distinct.
