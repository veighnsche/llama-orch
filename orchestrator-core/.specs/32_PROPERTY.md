# orchestrator-core — Property Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- FIFO within priority, Drop-LRU behavior under random workloads.

## Invariants

- FIFO-within-priority: for any sequence of enqueues and cancels, the relative order of tasks within the same priority equals arrival order after removing cancelled tasks.
- Drop-LRU-by-priority: upon exceeding capacity, drop lowest priority cohort; if multiple tasks within that cohort, drop oldest arrival first.
- Deterministic tie-breaks: with identical inputs, placement order and predicted start times are stable across runs.
- No silent truncation: over-budget attributes (ctx, max_tokens) never self-adjust; instead produce a typed error at the core boundary.

## Strategies & Generators

- Task generator
  - Random `priority ∈ {interactive, batch, background}`
  - Arrival index (monotonic sequence number)
  - Budgets within a bounded range; include over-budget cases with low probability to probe guardrails
- Operation generator
  - Weighted sequence of `enqueue(task)` and `cancel(task_id)`; ensure some cancel targets are absent to test idempotency
  - Capacity changes (optional) under a constrained model to ensure invariants hold when slots change

## Oracles

- Reference queue model implemented in-test (pure, simple) to assert FIFO-within-priority and Drop-LRU decisions.
- For predicted start times, compute using synthetic fixed service time per task to assert monotonicity.

## Execution

- Suggested tool: `proptest`
- Example filter: `cargo test -p orchestrator-core property -- --nocapture`
- Configure shrinking to reveal minimal counterexamples; include seed in failure output for reproduction.

## Traceability

- ORCH-3016: No silent truncation on over-budget inputs.
- ORCH-3093/3094: Capability pre-validation surfaces as core guardrails; property sequences include violations.

## Refinement Opportunities

- Model-based tests for placement scoring stability.
