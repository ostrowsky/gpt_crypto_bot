# Repository Skills Tooling

Status: implemented local tooling  
Last updated: 2026-05-17

## Purpose

Keep recurring repository-specific review and reporting workflows explicit, reusable, and
aligned with this bot's actual objective instead of relying on ad-hoc prompts each time.

## Included Skills

- `bot-progress-report`
  - builds recurring progress summaries against the scout objective;
  - separates scout, quality, ML, and RL/data health.
- `trading-app-review`
  - compares external trading systems against this bot and professional practice;
  - converts findings into a prioritized roadmap and replay requirements.
- `pr-description`
  - keeps pull-request summaries consistent and concise.

## Non-Goals

- No production trading behavior.
- No automatic code mutation.
- No replacement for replay/backtest gates.
- No claim that a skill result is production evidence by itself.

## Acceptance Criteria

1. Each skill has a clear trigger description and workflow.
2. `bot-progress-report` includes an executable report-building script.
3. `trading-app-review` includes a rubric and user-facing template.
4. The skills preserve this repo's objective contract:
   - early same-day top-mover capture;
   - BUY/WATCH separation;
   - replay-gated adoption.

## Verification

- `python -m py_compile skills/bot-progress-report/scripts/build_progress_report.py`
- run the progress report script against the local reports directory;
- inspect skill metadata and companion references for consistency.
