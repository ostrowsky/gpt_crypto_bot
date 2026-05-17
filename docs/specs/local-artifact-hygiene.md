# Local Artifact Hygiene

Status: implemented local tooling  
Last updated: 2026-05-17

## Purpose

Keep local runtime state, credentials, and process artifacts out of version control so release
worktrees stay readable and production handoffs are not polluted by machine-specific files.

## Scope

- Normalize the repository `.gitignore` into a real ignore file.
- Ignore embedded runtimes, virtual environments, caches, logs, runtime reports, local secrets,
  lock files, and transient dataset artifacts.

## Non-Goals

- No deletion of operator-local state.
- No automatic mutation of chat recipients.
- No cleanup of historical artifacts already intentionally tracked.

## Acceptance Criteria

1. `.gitignore` contains plain ignore patterns, not shell-script text.
2. Local `.env`, lock files, runtime reports, and temporary dataset files no longer appear as
   untracked noise.
3. Existing tracked files are not silently removed by this hygiene change.
