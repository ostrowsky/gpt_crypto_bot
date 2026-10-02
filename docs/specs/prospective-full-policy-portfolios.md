# Prospective full-policy portfolios and logical controller

2026-10-02. TH-01..TH-12. No production policy relaxation.

Priority 1: the full BUY/SELL replay engine accepts persistent JSON state. In
streaming mode open positions are not liquidated at tick end; cooldowns, last
exits and suspicious re-entry state survive. Only strictly increasing event-clock
frames execute. Closed market history must retain stable indices (append-only).
The producer freezes champion/candidate/general/base-ranker bytes, source hashes
and the point-in-time active watchlist, then computes both arms on the same closed
frame. Missing symbols/grids, late arrival (>120s), source/model drift or gaps
block the frame. No historical replay is admitted as prospective observation.
Warmup may precede registration but cannot create trades before observation start.
Unsigned sealed bundles include explicit position_open=true entries without
SELL events. Account valuation marks those holdings after estimated liquidation
costs; they do not count toward the >=100 completed-trade denominator. No missing
open holding is dropped and no boundary SELL is invented to close the account.
The source journal is hash chained. State and frame publication form one atomic
checkpoint; retries cannot execute twice. Actual model exposure embargo remains
48h. T+5 labels are not portfolio outcomes. Completed and open trades are separate.

Priority 2: user explicitly accepts logical same-user separation. Local authority
has a distinct signed contract identifying this weaker mode, not a fabricated
no_trainer_holdout_access assertion. Trainer receives no authority keys through
the process environment. Lenovo could read keys/files; this is an accepted trust
boundary, not Windows isolation. Missing certifications still block confirmation.
Numerical comparison, complete populations, untouched chronological holdout,
minimum windows, canary assignment, source binding and rollback gates unchanged.

Priority 3: Harness recognizes v1 and v2 numeric contracts. V2 requires complete
closed BTC grid and finite continuous drawdown, in addition to existing checks.
Missing data or stale source hashes continue to FAIL. Never relabel an incomplete
report or update provenance hashes without rerunning the actual computation.
Maximum 181-day candidate replay currently has delta=0: no winning candidate.

Priority 4: production adoption remains off until genuine maximum-history,
sealed/forward and canary evidence passes. The same signed pointer/model route
must be consumed by the bot with receipt and rollback verification. No success
receipt is inferred from controller authorization or synthetic tests.

Validation: incremental vs batch engine parity (without forced final liquidation),
restart, cooldown, duplicate/gap rejection, immutable model/data/source bindings,
late/partial frames, logical-vs-OS authority mismatch, v2 measurement failures,
and existing portfolio/account/release tests. Maximum-period parity replay is
required before any production adoption. No runtime/model/key artifacts committed.
Rollback: stop local producer/controller; unchanged champion remains production.
