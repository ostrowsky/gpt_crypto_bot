# V2 Shadow Operator Notification Policy

Status: expedited shadow-only  
Last updated: 2026-05-18

## Decision

Until the learned v2 policy exists, **real-time Telegram notifications are disabled**.

The provisional observer remains valuable as telemetry:

- decision trace;
- material transition log;
- daily summary;
- fast why / why-not lookup.

But `emerging_move` transitions are still too noisy to deserve live operator attention.

## Operator Surface

- no per-symbol v2 live Telegram messages;
- one daily v2 summary;
- ad hoc explainer on demand.

## Re-enable Gate

Real-time v2 alerts may return only after one of:

1. learned shadow policy proves higher precision than the provisional observer; or
2. a hand-reviewed alert policy reaches an explicit precision / volume target.

