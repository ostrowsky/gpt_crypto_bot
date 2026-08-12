# P0 Observability Foundation

Last updated: 2026-08-12

## Problem

The bot has strong objective metrics, but diagnosing missed opportunities still
requires manual log archaeology across blocked events, critic records, and
post-factum evaluator output. In addition, evaluator windows can become
uninformative without explaining why.

## Objective Fit

This feature improves:

- safer and more explainable operation;
- faster identification of missed top-mover causes;
- better decision quality before any future BUY / EXIT optimization.

It is intentionally **measurement-first** and must not change BUY / SELL behavior.

## Scope

Implement now:

1. structured blocked logging fields for bot and market-agent blocked events;
2. a read-only why-no-signal trace report over critic candidate history and
   the canonical runtime blocked-event journal;
3. evaluator coverage diagnostics that explain empty or weakly-covered windows;
4. a canonical metrics map document for operator / roadmap decisions.
5. daily critic summaries that surface the dominant blocker and the latest
   blocker detail for missed watchlist winners.

Explicitly out of scope:

- changing live gates;
- changing ranker / portfolio logic;
- auto-promoting hypotheses;
- adding new user-facing alert rules.

## Primary Metrics

- `blocked_winners`
- `missed_reason_counts`
- `blocked_reason_harm`
- evaluator reliability / coverage status
- decision consistency through canonical metric mapping

## Acceptance Criteria

- blocked events contain normalized `reason_code`, `gate`, and richer optional context;
- why-no-signal report can summarize blocker chains for a requested
  symbol/window even when critic persistence is missing, while repeated polls
  of the same closed bar are shown as one decision with a repeat count;
- evaluator output includes a machine-readable `coverage` section and human-readable explanation when the window is empty or incomplete;
- canonical metrics map exists under `docs/specs/`;
- daily critic output contains concise why-no-signal summaries for missed winners;
- no BUY / SELL behavior changes are introduced.

## Risk / Trade-offs

- richer logging slightly increases event size;
- some legacy blocked events will not have the new fields;
- legacy diagnostics that consult only the critic dataset can miss real runtime
  blocks; the runtime journal is therefore an explicit fallback source;
- canonical metrics reduce ambiguity but may expose conflicts that require later roadmap decisions.

## Verification Gate

- compile touched Python files;
- run the new why-no-signal report against existing local critic data;
- run the evaluator in JSON mode and verify the new `coverage` section on a recent window;
- review one generated text report for readability.

## Rollback Switch

- logging additions are additive only;
- the new report is standalone and can be ignored or removed without runtime impact;
- evaluator coverage fields are additive and do not affect evaluator scoring.

## Status

Shipped first additive slice:

- structured blocked events now emit normalized `reason_code` and `gate`;
- `files/why_no_signal_report.py` provides a read-only blocker-chain report;
- the report streams and normalizes both `critic_dataset.jsonl` and
  `bot_events.jsonl`, scans append-only journals backwards only through the
  requested time window, exposes raw versus unique counts, and records
  provenance;
- evaluator output now includes additive `coverage` diagnostics;
- canonical metrics map added in `docs/specs/metrics-canonical.md`.

Next likely extension:

- pass richer per-candidate context from remaining live call sites into blocked events;
- add an operator-facing command wrapper around `why_no_signal_report.py`.
