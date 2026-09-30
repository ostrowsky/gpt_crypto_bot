# Causal entry reconstruction and duplicate integrity

Date: 2026-09-30. Applies TH-01 through TH-12.

This repairs measurement, not BUY/SELL policy. Preserve the original frozen audit,
model, exposure embargo, prospective manifest and candidate-outcome-v2 labels.
No training on repaired targets and no production promotion in this stage.

## Duplicate contract

ID uniqueness checks and append must be inside one cross-process dataset IO lock.
A second producer/restart must update the existing decision through the existing
priority rules, not append another observation. Malformed history fails closed.
A changed decision (action/score/gates) gets its actual new decision timestamp;
archive the prior decision and provenance. An identical rescan preserves time.
The ID cache is only an IO optimization under the lock: scan another producer's
appended tail and rescan an atomically replaced file; never trust process memory
alone. Cooperative writers must not edit an existing prefix in place.
Historical repair is an immutable derived dataset. Collapse identical duplicate
payloads; differing features, sequences, decisions, provenance or labels quarantine
the entire ID. Keep the first source row only as an UNKNOWN marker, never choose
the variant with better returns. Preserve original bytes, counts and source hash.
An invalid member makes its entire comparison group incomplete, not a smaller
favorable selection pool. Live cleanup must first archive the entire original
stream under its IO lock, then atomically replace it with unique nonconflicting
IDs only. Conflicting IDs are excluded from training, not merged. Preserve the
archive and quarantine-ID manifest permanently; never delete the evidence.

## Price contract

The old closed-bar close is not the decision-time execution price. Binance 1m
history supplies a reference: the last fully closed minute before the recorded
decision. It does NOT supply the historical ask, fill or exact intra-minute quote.
Those quantities remain UNKNOWN. A separate retrospective diagnostic simulates
entry at the next minute open strictly after the decision (delay >0 and <=60s).
The next-minute high/low/close cannot select the entry price. Require raw validated
OHLCV, exact timestamp, successor proof and source-response SHA256. Missing bars,
conflicting history, nonpositive/nonfinite prices or future targets fail closed.
Compare at the original T+5 close only when entry precedes that close and every
timeframe bar from feature to target and its proof bar is present. Validate original
ret_5 against fetched closes; preserve it unchanged and calculate the new diagnostic
return separately. Never reconstruct target price by inverting rounded old returns.

Freeze the entire maximum available source file, candidate and sources; query all
eligible unexposed observations after the maximum train/validation/test exposure
plus 48h, without choosing dates or symbols for favorable results. Publish a
separate immutable report and response manifests; do not replace the prospective
experiment or point the release controller to this changed target contract.
Report exclusion/unknown counts, pairs, per-day effects and 30-day CI requirement.
Asynchronous decision times and fixed bar endpoints make this a ranking diagnostic,
not a synchronous executable selection strategy. Fees, spread, slippage, exits,
portfolio capacity, PIT universe and canary remain unverified. Runtime eligible is
always false and business improvement UNKNOWN regardless of proxy direction.

## Tests

Cross-process ID race/restart; malformed history; exact versus conflicting
duplicates; missing-minute proof; strict entry timing; full target continuity;
invalid/future/nonfinite prices; old-target reconciliation; quarantine whole groups;
prediction cannot see price reconstruction/outcomes; exposure embargo unchanged;
original input immutability; hashes/receipt verification and no production mutation.
