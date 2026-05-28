# Dynamic Menu Build Badge

Status: shipped operational fix  
Date: 2026-05-28

## Objective

The Telegram menu must show the actual code version currently running, not a
hardcoded historical marker.

## Problem

`files/bot.py` had hardcoded:

- `BUILD_ID = "menu_build_v30"`
- `BUILD_APPLIED_AT = "2026-05-16 13:29:11 +02:00"`

After code deployments, the menu could still display the old build string,
making it impossible to verify from Telegram whether the running process is on
the latest release.

## Requirements

1. Build badge is derived from local git metadata at runtime:
   - short commit hash;
   - commit date converted to local timezone.
2. Material dirty working trees are marked with `+dirty`; runtime telemetry,
   local state, positions, and generated model reports do not mark the deployed
   version dirty.
3. If git metadata is unavailable, fall back to the `bot.py` modification time.
4. The implementation must be outside `bot.py` import side effects so it can be
   unit-tested without starting the Telegram bot or acquiring `bot.lock`.
5. No trading logic changes.

## Acceptance Criteria

- `/menu` badge shows a current commit-like version, for example:
  `v:d5f7a1b` `build:2026-05-28 16:42:10 +02:00`
- On material dirty local code it shows `v:d5f7a1b+dirty`.
- Unit tests cover clean git metadata, dirty suffix, and fallback behavior.
