# E2E Haiku — Implementation Plan (Anti‑Cheat Gate)

Scope: Real‑model end‑to‑end test driving only OrchQueue v1, enforcing anti‑cheat and measuring token deltas.

## Stages and Deliverables

- Stage 15 — Real‑Model E2E (Haiku)
  - Runner in `test-harness/e2e-haiku/` issues `POST /v1/tasks` and streams via `GET /v1/tasks/:id/stream`.
  - Prompt uses `minute_words` (TZ=Europe/Amsterdam) and an 8‑char `nonce`.
  - Pre/post `/metrics` scrape to assert `tokens_out_total` delta > 0; engine/version surfaced.
  - Anti‑cheat checks:
    - Require `REQUIRE_REAL_LLAMA=1` and a real Worker; CPU allowed only as a clearly marked CI fallback.
    - Forbid fixtures (`fixtures/haiku*`) and hardcoded outputs; repo scan for lines containing both `minute_words` and `nonce`.
    - Fail if mock/stub engine detected or `/metrics` absent.

## Tests

- `test-harness/e2e-haiku/tests/e2e_client.rs` and additional cases for failure modes.
- BDD features in `test-harness/bdd/tests/features/e2e_haiku/` (optional mirror).

## Acceptance Criteria

- Pass within ≤ 30 s (single retry allowed if minute flips during run).
- Anti‑cheat criteria satisfied; metrics token delta observed; engine/model visible.

## Backlog (initial)

- Local GPU runner config; CI CPU carve‑out harness with explicit warning.
- Helper to compute `minute_words` deterministically; snapshot of metric families for debugging.
