# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Blockers (in order)

- [x] Create archive script to finalize TODO.md
  - Path: `ci/scripts/archive_todo.sh`
  - Behavior: move root `TODO.md` to `.docs/DONE/TODO-[auto-increment].md` (next number), create directories if missing, print final path.
  - Acceptance: running the script moves the file; re-run when no `TODO.md` present fails with a clear message; numbering is monotonic.

## P1 — Quality of Life

- [ ] Link docs from the main README
  - Add links in `README.md` to `README_LLM.md` and `.docs/workflow.md` for quick discovery.

## Progress Log (what changed)

- 2025-09-15: Added `README_LLM.md` and aligned it with `.docs/workflow.md`. Added rules for root `TODO.md` lifecycle and the archive script.
- 2025-09-15: Created archive script at `ci/scripts/archive_todo.sh` and set it executable. Usage: `bash ci/scripts/archive_todo.sh`.
