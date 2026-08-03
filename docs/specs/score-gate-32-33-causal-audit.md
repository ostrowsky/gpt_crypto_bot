# Score-gate 32–33 Frozen Causal Audit

Date: 2026-08-03
Status: research-only

## Objective

Test whether candidates blocked immediately below the live top-gainer score
floor should advance to WATCH/shadow. The audit must not lower the BUY floor.

## Frozen cohort

- first `top_gainer_score_gate` event per local day and symbol;
- candidate score `>=32` and `<34`;
- actual blocked floor exactly `34`;
- only days with a final top-gainer critic;
- repeated scan events are deduplicated while retaining their count.

## Outcomes and attribution

The audit joins candidates to canonical watchlist-top labels, actual earlier and
later entries, other same-day blockers, reconstructed portfolio occupancy, and
cached 15m/1h candle paths. A candidate is admission-eligible only when portfolio
capacity existed and the symbol had not already been bought before the band
event. T+5/T+10 returns include 20 bps round-trip cost.

It reports detection, score admission, other blockers, portfolio capacity,
later control entry, top-mover precision, net returns, and downside. Evaluation
uses a chronological 70/30 split and a separate recent 14-day stability window.

## Gate

WATCH/shadow requires all-period, holdout, and recent windows to each have:

- at least 10 capacity-eligible mature cases;
- at least one earlier canonical top-mover opportunity;
- positive average and median T+10 net return;
- at least 50% positive T+10 outcomes.

BUY is always false at this stage. Any later BUY proposal requires independent
forward WATCH evidence and the normal early-capture, precision, PnL, drawdown,
turnover, and capacity guardrails.
