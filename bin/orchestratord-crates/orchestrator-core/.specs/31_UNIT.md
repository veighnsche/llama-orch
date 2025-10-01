# orchestrator-core — Unit Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Queue invariants (Reject vs Drop-LRU), cancel semantics, capacity snapshots.

## Notes

- Pure logic; no I/O.

## Test Catalog

- Queue Admission & Guardrails
  - Enqueue within capacity → accepted; `queue_position` increments.
  - Exceed capacity with multiple priorities → lowest priority cohort drops; within cohort drop oldest (LRU) first.
  - Over-budget requests (ctx, max_tokens) → typed error surfaced (no silent truncation).

- Ordering & FIFO Within Priority
  - Given interleaved enqueues of priorities A > B:
    - All A tasks preserve FIFO among A; all B tasks preserve FIFO among B.
    - No cross-priority reordering of A vs B in effective schedule.

- Cancel Semantics
  - Cancel on queued id → removed; positions of trailing tasks decrement exactly by 1.
  - Cancel on running id → state flips to cancelled; mark for deterministic stop signal at boundary (mock asserted as flag only in unit).
  - Cancel idempotency → duplicate cancel returns Ok/Noop without panics.

- Snapshots & Readouts
  - Capacity snapshot returns immutable view; repeated calls do not mutate internal queue.
  - Predicted start ms computation with fixed service time hints is monotonic with enqueue order for single-slot.

- Error Typing (internal-only)
  - Errors are crate-internal types; carry reason codes suitable for mapping at HTTP boundary (delegated to `orchestratord`).

## Structure & Conventions

- Place unit tests next to the logic they exercise using `#[cfg(test)] mod tests {}` with focused helpers.
- Table-driven tests for admission/cancel matrices; each row documents parameters and expected outcomes.
- Name tests with traceability hints where applicable, e.g., `no_silent_truncation_orch_3016`.

## Execution

- Run all: `cargo test -p orchestrator-core -- --nocapture`
- Filter: `cargo test -p orchestrator-core queue_admission -- --nocapture`

## Traceability

- ORCH-3016: No silent truncation on over-budget inputs.
- ORCH-3093/3094: Capability pre-validation expectations are enforced here as internal guards where delegated from handlers.

## Refinement Opportunities

- Add edge cases for cancel on empty/duplicate ids.
