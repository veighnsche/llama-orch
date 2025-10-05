# orchestrator-core-bdd
**BDD test suite for orchestrator-core**
`libs/orchestrator-core/bdd` — Cucumber-based behavior-driven development tests for the orchestrator-core queue library.
---
## What This Crate Does
This is the **BDD test harness** for orchestrator-core. It provides:
- **Cucumber/Gherkin scenarios** testing queue behavior
- **Step definitions** in Rust using the `cucumber` crate
- **Unit-level BDD tests** for queue invariants and policies
- ** output** for test artifacts
**Tests**:
- Queue admission (accept/reject)
- Capacity limits and backpressure
- FIFO ordering
- Admission policies (drop-LRU, reject-new, fail-fast)
- Queue depth and metrics
---
## Running Tests
### All Scenarios
```bash
# Run all BDD tests
cargo test -p orchestrator-core-bdd -- --nocapture
# Or use the BDD runner binary (if available)
cargo run -p orchestrator-core-bdd --bin bdd-runner
```
### Specific Feature
```bash
# Set environment variable to target specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/queue_capacity.feature \
cargo test -p orchestrator-core-bdd -- --nocapture
```
---
## Feature Files
Located in `tests/features/`:
- `admission.feature` — Task admission, accept/reject logic
- `capacity.feature` — Queue capacity limits, backpressure
- `fifo.feature` — FIFO ordering guarantees
- `policies.feature` — Admission policies (drop-LRU, reject-new)
- `metrics.feature` — Queue depth, enqueue/dequeue counts
---
## Example Scenario
```gherkin
Feature: Queue Capacity
  Scenario: Reject task when queue is full
    Given a queue with capacity 10
    And the queue has 10 tasks
    When I enqueue a new task
    Then the task should be rejected
    And the rejection reason should be "QueueFull"
    And the queue depth should remain 10
```
---
## Step Definitions
Located in `src/steps/`:
- `queue.rs` — Queue creation and manipulation steps
- `admission.rs` — Admission and rejection steps
- `assertions.rs` — Queue state assertions
---
## Testing
```bash
# Run all tests
cargo test -p orchestrator-core-bdd -- --nocapture
# Check for undefined steps
cargo test -p orchestrator-core-bdd --lib -- features_have_no_undefined_or_ambiguous_steps
```
---
## Dependencies
### Parent Crate
- `orchestrator-core` — The library being tested
### Test Infrastructure
- `cucumber` — BDD framework
- `tokio` — Async runtime for tests
- `` — Test artifact output
---
## Specifications
Tests verify requirements from:
- ORCH-3004 (Admission control)
- ORCH-3005 (Queue capacity)
- ORCH-3008 (Backpressure)
- ORCH-3010 (FIFO ordering)
- ORCH-3011 (Rejection policies)
- ORCH-3016, ORCH-3017, ORCH-3027, ORCH-3028
- ORCH-3044, ORCH-3045
See `.specs/00_llama-orch.md` for full requirements.
---
## Status
- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
