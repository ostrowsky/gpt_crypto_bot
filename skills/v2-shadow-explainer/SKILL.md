---
name: v2-shadow-explainer
description: Quickly explain why the v2 shadow observer did or did not emit a signal for a symbol using the compact v2 decision trace, without reanalyzing full bot history.
---

# V2 Shadow Explainer

Use this skill when the operator asks:

- why did v2 signal on a coin;
- why did v2 not signal on a coin;
- what did v2 see for a coin today.

## Fast Path

1. Do **not** grep full bot history first.
2. Run:

```powershell
pyembed\python.exe files\explain_v2_shadow_signal.py --symbol SYMBOL --date YYYY-MM-DD --json
```

Add `--timeframe 15m` or `--timeframe 1h` when the operator asks for a specific timeframe.

## Source Of Truth

- `files/v2_shadow_decisions.jsonl`
  - every deduplicated per-bar shadow decision;
- `files/v2_shadow_events.jsonl`
  - material transitions only.

## Answer Shape

Always answer:

1. latest observed v2 state and action;
2. whether a material shadow signal occurred;
3. if not, the explicit `why_no_signal` reason;
4. the most relevant feature snapshot;
5. whether the only observation was a bootstrap row rather than a real transition;
6. note that this is provisional shadow logic, not the learned v2 model.
