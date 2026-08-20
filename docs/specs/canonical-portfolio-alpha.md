# Canonical Unified Portfolio Alpha

Status: measurement-only implementation
Last updated: 2026-08-13

## Problem

`replay_backtest.py` historically reported `pnl_total` as the arithmetic sum of
per-trade percentage returns. That number ignores capital allocation, idle cash,
concurrent positions, compounding, fees, slippage, and the return available from
a named benchmark. It is useful as a trade-outcome diagnostic but it is not
portfolio return and must not be used as evidence of trading alpha.

The bot objective requires one symbol-deduplicated portfolio with at most ten
open positions. A decision-grade profitability claim therefore needs a single
capital curve for that exact contract.

## Objective fit

This evaluator supports:

- selection under the unified ten-position cap;
- truthful comparison of policy variants after executable costs;
- safe roadmap decisions that cannot substitute per-trade PnL for portfolio
  alpha.

It does not optimize entry or exit policy by itself.

## Scope

### In scope

- Replay all admitted trades through one cash account.
- Enforce at most ten concurrent positions and one position per symbol across
  the unified candidate stream.
- Allocate each new position at most one tenth of current liquidation equity,
  capped by available cash; idle cash remains in the account.
- Apply taker fee and conservative static slippage on every entry, partial exit,
  and final exit.
- Mark open positions on closed candles at a 15-minute valuation cadence.
- Compare net portfolio return with `BTCUSDT` buy-and-hold over the same first
  and last observable closed bars using the same costs.
- Emit a standalone canonical artifact and embed the same result in replay JSON.
- Label legacy `pnl_total` explicitly as a non-capital-weighted diagnostic sum.

### Out of scope

- Live BUY, SELL, replacement, sizing, or portfolio-policy changes.
- Claiming that a replay candidate stream proves real fill quality.
- Hindsight reconstruction of main/agent positions from incomplete event logs.
- Order-book slippage reconstruction where historical depth is unavailable.

The static slippage assumption is deliberately explicit. A later depth-aware
evaluator may replace it only under a new versioned metric contract.

## Metric contract

Primary metric:

- `net_alpha_after_costs = portfolio.net_return_after_costs_pct -
  benchmark.net_return_after_costs_pct`.

Supporting metrics:

- portfolio net and gross return;
- total fee and slippage cost;
- maximum drawdown of the closed-bar liquidation curve;
- maximum concurrent positions and average gross utilization;
- trade count, valuation coverage, and contract violations;
- named benchmark entry/exit timestamps and prices.

`totals.pnl_total` remains available for compatibility only and is tagged
`non_capital_weighted_diagnostic_sum`. It is never portfolio alpha.

## Evidence grades

A result is `decision_grade=true` only when all conditions hold:

1. capacity is exactly `10`;
2. the replay covers the established maximum feasible engine window of `30`
   requested days and at least `95%` of that interval has benchmark candles;
3. at least one trade closes;
4. all positions are symbol-deduplicated and no capacity violation occurs;
5. fee and slippage assumptions are both positive and are applied identically
   to portfolio and benchmark;
6. valuation uses observable closed bars with no mark-price fallback;
7. benchmark and universe/policy provenance are present.

Shorter or incomplete runs remain useful diagnostics but cannot approve a
production hypothesis.

## Acceptance criteria

- Unit tests prove capital weighting, two-sided costs, benchmark parity,
  symbol/capacity fail-closed behavior, and decision-grade gating.
- Replay JSON exposes the canonical result without removing compatibility
  fields.
- Truth Harness TH-11 validates the complete contract, recomputes alpha
  consistency, checks the current policy epoch, and refuses a stale/newer
  malformed artifact instead of falling back to an older valid file.
- A maximum-period full-watchlist run produces the standalone artifact.
- The replay accepts an explicit timezone-aware `--end-at` cutoff so a data
  outage after the last complete market bar cannot turn "maximum available"
  into an empty wall-clock window. The requested cutoff is recorded in the
  artifact; it may move the window backward but may not shorten the registered
  30-day period.
- Diagnostic/incomplete artifacts render safely with `n/a` alpha instead of
  crashing after the JSON has been written.
- The final result is reported as evidence, not as a live-policy promotion.

## Risks and trade-offs

- Static slippage is an assumption, not a reconstructed fill. It is conservative
  and visible, but it can still differ from real liquidity.
- Equal-slot capital allocation may differ from future live sizing. Changing the
  sizing contract requires a schema/version change, not silent reinterpretation.
- A BTC benchmark can outperform or underperform the watchlist opportunity set;
  this is intentional because the alternative use of capital must be named and
  stable.

## Backtest and verification gate

Run:

```powershell
pyembed\python.exe -m unittest files.test_portfolio_alpha files.test_truth_harness
pyembed\python.exe files\replay_backtest.py --days 30 --end-at <latest-complete-UTC-cutoff> --max-open-positions 10 --variant replacement_block_non_losing --top-gainer-score-min 34 --objective-top-n 15 --no-baseline --portfolio-alpha-output .runtime\reports\canonical_portfolio_alpha_30d_latest.json --json
pyembed\python.exe files\truth_harness.py full
```

The full-watchlist `30d` run is the established maximum feasible replay-engine
gate. A shorter run can debug the evaluator but cannot set `decision_grade`.

## Maximum-period evidence (2026-08-13)

The canonical command completed over `103` watchlist symbols (`15m` and `1h`),
`2026-07-14T18:22Z .. 2026-08-13T18:22Z`:

- requested/observed period: `30d / 29.979d` (`99.93%`);
- closed trades: `1630`; max concurrent positions: `10`;
- closed-bar valuation coverage: `100%`; contract violations: `0`;
- gross capital return before costs: `-24.90%`;
- net capital return after costs: `-49.98%`;
- BTC buy-and-hold after the same costs: `-2.49%`;
- canonical net alpha after costs: `-47.49pp`;
- cost drag: `25.08pp`; max drawdown after costs: about `50.00%`;
- average gross capital utilization: `56.72%`.

The legacy diagnostic trade sum was `-281.69%`; its large difference from the
capital result confirms why it cannot represent portfolio performance.

Decision: the evaluator is accepted as decision-grade measurement, while the
evaluated `replacement_block_non_losing` production profile does **not** prove
positive portfolio alpha. No live behavior is relaxed. Earlier policy decisions
whose profitability evidence relied on `pnl_total` must be revalidated with the
canonical capital contract before they can be cited as profitable. The next
research priority is turnover/cost attribution and paired canonical-alpha replay
of frozen production alternatives, not threshold relaxation.

## Rollback

Remove the optional `--portfolio-alpha-output` integration and the embedded
`portfolio_alpha` block. Since this feature does not affect trading decisions,
rollback does not require position migration.
