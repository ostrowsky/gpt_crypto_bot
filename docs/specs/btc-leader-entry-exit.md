# BTC leader entry and exit hypothesis

2026-10-03. Research only; TH-01..TH-12. No live gate relaxation.

Test whether BTC growth precedes SOL/ETH, whether their subsequent percentage
return is at least twice BTC's, and whether BTC weakening improves exits.
Use the maximum common locally available closed 1h history, merging cached raw
Binance candles and the recovered maximum archive. Conflicting duplicates fail;
gaps invalidate windows, never imply zero returns. Hash every input, source and
the output contract. Hourly data cannot resolve sub-hour ordering. Present both
same-hour and delayed return correlations; correlation is not causality.

Pre-register BTC 1h triggers +0.25%, +0.5%, +1%; forward horizons 1/3/6/12/24h.
Report positive, negative and >=2x BTC event counts with denominators, BTC-positive
amplification denominator, unconditional target baseline and counterexamples.
Events are spaced 24h to avoid overlapping target windows. No guaranteed profit
or universal trend-onset prediction can be established by finite history.

Chronological train 70%, test 30%; purge 24 bars at the train boundary. Select
trigger on train only using the fixed-24h equal-weight SOL/ETH basket, then freeze
it for test. Enter at next bar open, exit at next bar open following a closed BTC
return <=0%, BTC return <=-0.25%, target return <=0%, or fixed 24h timeout.
Compare exits on the same entries, with at most one basket and 24h spacing;
include 7.5bps fee and 5bps slippage per side. Report individual net returns,
equal-weight basket compounded return and drawdown at trade closes (not intratrade
max drawdown or full live-bot alpha). Other-symbol studies are future exploratory
work; this primary pre-registration is SOL/ETH only. Reject automatic promotion.

Run after the existing serial replay terminates, never concurrently; one logical
CPU, BelowNormal priority, one numerical-library thread. If the prior process
stops without results, retain its UNKNOWN and do not call it success. Preserve
all runtime outputs outside Git. Rollback: stop this research job only.

Tests cover next-open causality, gaps, duplicate conflicts, costs, no future
selection, overlap/purge, zero denominators, amplification and exit pairing.
