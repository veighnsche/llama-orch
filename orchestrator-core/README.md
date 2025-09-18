# orchestrator-core — orchestrator-core (core)

## 1. Name & Purpose

`orchestrator-core` provides core orchestration primitives used by daemons. It focuses on queueing/admission policies and invariants (bounded FIFO with priorities, full-queue behavior), with property-style tests. No HTTP or adapter IO lives here.

## 2. Why it exists (Spec traceability)

Traceability follows the leading workspace specs:

- Core orchestrator spec: [.specs/00_llama-orch.md](../.specs/00_llama-orch.md)
  - Admission & bounded queues: ORCH-3004, ORCH-3005
  - FIFO within priority class: ORCH-3008
  - Placement/readiness hooks (informative for this crate): ORCH-3010, ORCH-3011
  - Observability fields (consumers emit metrics): ORCH-3027, ORCH-3028
- Home profile overlay: [.specs/00_home_profile.md](../.specs/00_home_profile.md) — informs defaults around queue depth and developer experience in the single-host profile.


## 3. Public API surface

- Rust crate API (internal)

## 4. How it fits

- Part of the core orchestrator. Upstream: adapters, Downstream: workers.

```mermaid
flowchart LR
  callers[Clients] --> orch[Orchestrator]
  orch --> adapters[Worker Adapters]
  adapters --> engines[Engines]
```

#### Detailed behavior (High / Mid / Low)

- High-level
  - In-memory, bounded queue with two priorities: `Interactive` and `Batch`.
  - Full-queue behavior via policy enum: `Reject` or `DropLru`.

- Mid-level
  - Types: `Priority`, `Policy`, `InMemoryQueue` with separate deques per priority.
  - Core ops: `enqueue(id, prio)`, `cancel(id)`, `snapshot_priority(prio)`, `len()`, `capacity()`.
  - Tests in `src/queue.rs` assert key invariants: boundedness + reject, drop-lru preference, FIFO within class.

- Low-level (from `src/queue.rs`)
  - `enqueue`: if `len >= capacity` and policy is `Reject`, returns `EnqueueError::QueueFullReject`.
  - `enqueue` with `DropLru`: drops oldest Batch first, else oldest Interactive, then pushes new item into the requested priority.
  - `cancel`: removes the first occurrence of `id` from either priority queue and returns `true` if removed.

## 5. Build & Test

- Workspace fmt/clippy: `cargo fmt --all -- --check` and `cargo clippy --all-targets --all-features
-- -D warnings`
- Tests for this crate: `cargo test -p orchestrator-core -- --nocapture`


## 6. Contracts

- None


## 7. Config & Env

- See deployment configs and environment variables used by the daemons.

## 8. Metrics & Logs

- Emits queue depth, latency percentiles, and engine/version labels.

## 9. Runbook (Dev)

- Regenerate artifacts: `cargo xtask regen-openapi && cargo xtask regen-schema`
- Rebuild docs: `cargo run -p tools-readme-index --quiet`


## 10. Status & Owners

- Status: alpha
- Owners: @llama-orch-maintainers

## 11. Changelog pointers

- None

## 12. Footnotes

- Specs:
  - Core: [.specs/00_llama-orch.md](../.specs/00_llama-orch.md)
  - Home overlay: [.specs/00_home_profile.md](../.specs/00_home_profile.md)
- Requirements: [requirements/00_llama-orch.yaml](../requirements/00_llama-orch.yaml)

### Additional Details
- Queue invariants and property tests overview (capacity, rejection policies, session affinity helpers).
- Capacity policies and bounded FIFO behavior.


## What this crate is not

- Not a general-purpose inference server; focuses on orchestration.
