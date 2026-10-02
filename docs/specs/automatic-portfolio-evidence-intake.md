# Automatic independent portfolio evidence intake

2026-10-02. Implements priority 1 validation tooling and priority 2 intake;
does not approve a strategy, invent certificates or claim closed-loop completion.

Maximum-period stream validation binds archive, frozen model/source receipts and
uses identical candidates/event clocks for batch vs restarted incremental BUY/SELL
execution. Open positions are not forcibly sold for comparison. This checks the
execution-state contract, NOT candidate generation/live discovery parity. A PASS
cannot certify point-in-time population or raw market provenance by itself.

The controller automatically reads a bound input registration, copies no models,
discovers historical/sealed/shadow/canary bundle and certification pairs, validates
signatures and model hashes, recomputes complete accounts, and publishes the
existing independent intake contract. Missing phases remain UNKNOWN; failed
numeric phases remain REJECTED with numerator/window/result preserved. A signed
intake is not fabricated from an unsigned replay or trainer summary. Missing,
stale, mismatched or failed evidence never enables a stage or reuses a stale
ready request. The existing controller owns rollback and release permissions.

CANARY requires historical+sealed+shadow; PROMOTED additionally requires actual
assignment-certified canary. Phase chronology, costs, risk, confidence, maximum
history and >=30 days / >=100 closed trades remain unchanged. No production gate
relaxation. Separate cohorts/real admission receipts remain required work.

Morning report shows full-policy producer status/embargo and independent intake
status separately from proxy model outcomes. Unknown is not zero or improvement.

Tests: bound-path traversal, model/hash/signature mismatch, missing and rejected
phases, CANARY vs PROMOTED selection, chronology rejection delegated to gate,
stream/restart state parity and report freshness. Rollback: disable pipeline;
existing fail-closed controller and unchanged champion remain in force.
