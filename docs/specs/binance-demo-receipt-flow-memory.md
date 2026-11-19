# RFP-1 bounded-memory execution continuation (MR-1)

Registered 2026-11-19 after RFP-v2 MemoryError; original attempts stay INCOMPLETE.
No financial/model/threshold/risk changes. Parent RFP-1 and journal recovery.
SPEC -> TESTS -> CODE. No keys, orders, worker/expiry changes or promotion.

MR01: replace the eager all-origin Python row/prediction dictionaries with a
lazy time lookup into the exact original frozen NPZ arrays. Keep original
float64 values, global source-row identities, chronological candidate order,
feature hashes and every issued row including those without labels. Load only
one arm/horizon prediction vector at a time. Do not load future targets or refit
any model, including the already frozen TRAIN-only microprice coefficients.

MR02: stream the exact original full-cohort provenance SHA using canonical JSON
array bytes and compare to the parent's signals manifest before economic work.
Use the ORIGINAL signal_rows helper on each current row, ORIGINAL Policy,
simulator, quote reader, financial audit, summary and renderer. There is no new
admission/exit logic. Current-origin signal rows are ephemeral, not retained
for four days; only the latest selected decision needed for the first-trade plot
is retained by the event sink. All events still go to the durable journal.

MR03: quote LRU maximum4 chunks, within original bound16. Cache size cannot
change as-of answers. Monetary accounting remains Decimal, predictor values
remain exact original float64. OOS and TEST dates, initial100USDT per case,
continuous inventory, costs, latches and all61 cases remain unchanged.

MR04: accept all sealed completed parent cases, independent of outcomes, under
the existing strict VerifiedParent gates. Re-emit and re-audit them; wholly
recompute incomplete cases. Source input hashes, old economic registration,
baseline and signals manifests must match exactly. Reused decompressed bytes
must match parent journals. Keep failed parents, partial files and incidents.

MR05: the new outer memory registration pins active adapter/writer/verifier/
economic sources and this spec, parent incident and complete reuse vector.
The inner original economic registration is an explicit unchanged reference
contract; it is not a claim that only its old controller executed. Extra source
provenance is mandatory. No dynamic module/global monkeypatches are needed.
Original raw-validation proof and current used-chunk hashes remain separate.

MR06: complete all61 financial/source/journal audits before completion. Full
Truth Harness FAIL TH-11 and conditional simulation limitations remain. No
production adoption, actual-fill training or earnings claim. Runtime untracked.
Rollback stops only own offline process, preserving every source and result.

Tests before implementation: lazy-vs-original signal parity for all15 arms/
horizons, exact decision/fill/ledger parity, no target reads, no future-label
filter, prefix invariance, original full provenance SHA, no per-origin dict
retention and cache-size invariance. Run relevant and app regressions, diff
review, staged Truth Harness, commit and push. No artificial RED or weakened
economic assertions to make a resource repair appear successful.
