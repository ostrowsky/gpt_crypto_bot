# Automatic independent portfolio evidence intake

## 2026-10-04: checkpoint accounting and explicit parity differences

Research/measurement repair only; no production BUY/SELL relaxation. The original
181-day execution parity FAIL is retained. A reproduction showed that batch
closed-trade objects receive later cooldown annotations through shared references,
whereas JSON-restored stream objects do not update the consumer's closed history.
Both the verifier and prospective paper producer must reconcile the three
cooldown annotations from `last_closed_by_symbol` into their closed-history
records after every frame. Identity is (symbol, timeframe, entry timestamp,
exit timestamp); ambiguous, absent or regressing records fail closed. Do not
discard these fields from comparison or change execution prices/decisions.
The history index is transient and rebuilt after restart; no synthetic trades
or forward labels are created. Old incomplete history is not automatically
certified by this repair. Missing historical annotations remain a provenance gap.

The repeated maximum check uses the same frozen archive/models, a new output
filename and bound verifier/accounting source hashes. Save every differing leaf
with its structural path, both values and presence flags in per-arm sidecars;
publish total counts and bounded examples in the result. Preserve list order,
multiplicity, missing-vs-null and numeric representation differences. Do not
normalize away execution or cooldown differences. Progress names loading,
candidate generation, batch simulation, streaming, comparison and finalization.
Interrupted computation or source drift is UNKNOWN, never PASS. A PASS still
does not prove full live BUY parity, raw/PIT certification or a closed loop.

Tests: real simulator close/cooldown/checkpoint reproduction, repeated updates,
replacement identity, malformed/unknown/duplicate/regressing history, producer
integration, exact difference paths/presence, hashes and refusal to overwrite.
Verification: focused regressions, full/staged Truth Harness and a sequential
maximum-period rerun on one CPU, BelowNormal, numerical library threads=1.
Rollback: stop the diagnostic run/revert accounting integration; production
champion is unchanged. Runtime evidence and model files are not committed.

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
