# Live BUY/replay contract and loss audit

2026-10-04. Diagnostic only; no production strategy or model change. TH-01..12.

Audit the existing maximum-period frozen paired bundle, never a shortened winning
sample. Verify its receipt, bounds and original policy/account source hashes.
Preserve originals and refuse to overwrite the new report. Use one CPU,
BelowNormal and one numerical-library thread. No exchange calls, live writes,
learning-role restart, model training or release authorization.

Reproduce narrow contract differences, not a fabricated full live BUY PASS:
hourly maximum holding period through the real replay picker and the source-bound
live assignment; external-agent capacity through the real live capacity gate and
real replay simulator on the same conditional occupancy fixture. Mock only entry
detectors/input data, temporal event loading and external occupancy. Fixtures
prove conditional behavior, not the historical frequency or PnL impact of a gap.
Known inequality makes equivalence FAIL; untested full `_poll_coin` remains UNKNOWN.
Differences in report forecasts, fetch horizon, discovery cadence, group/cluster
gates and admission ordering require separate causal historical verification.

On all recorded champion trades, report numerator/denominator, holding periods,
exit-reason and timeframe/mode diagnostics, gross returns and simple round-trip
cost-adjusted trade returns. These unweighted trade statistics are not portfolio
alpha or causal rule-ablation effects. Use the independently audited canonical
portfolio results for capital-weighted loss/cost impact; missing values are
UNKNOWN, not zero. Keep boundary-finalized positions separate from natural exits.
Report how many archived hourly parameters differ from the live base assignment.

The historical reconstruction is unsigned, overlaps candidate model exposure,
does not include the external agent's complete portfolio, and lacks full live BUY
and raw/PIT certification. Loss figures cannot be attributed to the running bot's
actual account. Identifying a mismatch does not authorize fixing/relaxing a rule.
Follow-up: align contracts, test full live decision path offline on closed-prefix
inputs, repeat the maximum paired portfolios, then untouched forward cohorts and
existing risk/rollback gates. Reject this candidate unless all independent gates
pass; never remove gates to obtain a winner.

Tests: real conditional probes, exact cost calculation, empty denominators,
exit classification, receipt/source drift, output preservation and no full-live
success claim. Verification: focused tests, full/staged Harness and max-bundle
audit. Rollback: stop/revert the diagnostic tooling; champion remains unchanged.
