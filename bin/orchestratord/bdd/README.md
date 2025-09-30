# orchestratord-bdd

**BDD test suite for orchestratord**

`bin/orchestratord/bdd` — Cucumber-based behavior-driven development tests for the orchestratord HTTP API.

---

## What This Crate Does

This is the **BDD test harness** for orchestratord. It provides:

- **Cucumber/Gherkin scenarios** testing orchestratord behavior
- **Step definitions** in Rust using the `cucumber` crate
- **Integration tests** covering HTTP API, SSE streaming, catalog, sessions
- **Proof bundle output** for test artifacts and traceability

**Tests**:
- Task admission and queueing
- SSE streaming (started → token → metrics → end)
- Session management (create, query, delete)
- Catalog operations (register, verify, lifecycle)
- Multi-node registration and heartbeats
- Error handling and edge cases

---

## Running Tests

### All Scenarios

```bash
# Run all BDD tests
cargo test -p orchestratord-bdd -- --nocapture

# Or use the BDD runner binary
cargo run -p orchestratord-bdd --bin bdd-runner
```

### Specific Feature

```bash
# Set environment variable to target specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/admission.feature \
cargo test -p orchestratord-bdd -- --nocapture
```

### Check for Undefined Steps

```bash
# Verify all steps are implemented
cargo test -p orchestratord-bdd --lib -- features_have_no_undefined_or_ambiguous_steps
```

---

## Test Status

**Current Coverage** (as of 2025-09-30):
- **18 features**, 41 scenarios
- **84/108 steps passing** (78%)
- **Core features**: 100% passing (admission, streaming, sessions)
- **New features**: Partial implementation (catalog, multi-node)

### Passing Features

✅ Task admission and rejection  
✅ SSE streaming lifecycle  
✅ Session management  
✅ Basic catalog operations  
✅ Queue backpressure  

### In Progress

🚧 Advanced catalog (verification, lifecycle states)  
🚧 Multi-node registration and heartbeats  
🚧 Placement with model filtering  

---

## Feature Files

Located in `tests/features/`:

- `admission.feature` — Task admission, queue capacity, rejection
- `streaming.feature` — SSE lifecycle, token streaming, metrics
- `sessions.feature` — Session create, query, delete, TTL
- `catalog.feature` — Model registration, verification, lifecycle
- `nodes.feature` — Multi-node registration, heartbeats, deregistration
- `placement.feature` — Pool selection, model-aware placement
- `backpressure.feature` — Queue full, retry-after headers

---

## Step Definitions

Located in `src/steps/`:

- `background.rs` — Setup and teardown (orchestratord instance, mock adapters)
- `admission.rs` — Task admission steps
- `streaming.rs` — SSE streaming steps
- `sessions.rs` — Session management steps
- `catalog.rs` — Catalog operation steps
- `nodes.rs` — Multi-node registration steps
- `assertions.rs` — Common assertion helpers

---

## World State

The `World` struct (in `src/world.rs`) maintains test state:

```rust
pub struct World {
    pub orchestratord_url: String,
    pub last_response: Option<Response>,
    pub last_job_id: Option<String>,
    pub last_session_id: Option<String>,
    pub sse_events: Vec<SseEvent>,
    pub mock_adapter: Option<MockAdapter>,
}
```

---

## Example Scenario

```gherkin
Feature: Task Admission

  Scenario: Successfully admit a task
    Given orchestratord is running
    And a mock adapter is configured
    When I POST a task to /v2/tasks with:
      | model      | llama-3.1-8b-instruct |
      | max_tokens | 100                   |
    Then the response status should be 202
    And the response should contain a job_id
    And the admission queue depth should be 1
```

---

## Proof Bundle Output

BDD tests emit proof bundles to `.proof_bundle/bdd/<run_id>/`:

- `scenarios.ndjson` — Scenario results
- `steps.ndjson` — Step execution details
- `metadata.json` — Test run metadata
- `seeds.txt` — Random seeds for reproducibility

Set `LLORCH_RUN_ID` to control the output directory.

---

## Dependencies

### Parent Crate

- `orchestratord` — The binary being tested

### Test Infrastructure

- `cucumber` — BDD framework
- `tokio` — Async runtime for tests
- `reqwest` — HTTP client for API calls
- `serde_json` — JSON parsing
- `proof-bundle` — Test artifact output

---

## Writing New Tests

### 1. Add Feature File

Create `tests/features/my_feature.feature`:

```gherkin
Feature: My New Feature

  Scenario: Test something
    Given orchestratord is running
    When I do something
    Then something should happen
```

### 2. Implement Steps

Add to `src/steps/my_steps.rs`:

```rust
use cucumber::{given, when, then};
use crate::world::World;

#[when("I do something")]
async fn when_i_do_something(world: &mut World) {
    // Implementation
}

#[then("something should happen")]
async fn then_something_happens(world: &mut World) {
    // Assertions
}
```

### 3. Run Tests

```bash
cargo test -p orchestratord-bdd -- --nocapture
```

---

## Specifications

Tests verify requirements from:
- ORCH-3004, ORCH-3005, ORCH-3008, ORCH-3010, ORCH-3011
- ORCH-3016, ORCH-3017, ORCH-3027, ORCH-3028
- ORCH-3044, ORCH-3045

See `.specs/00_llama-orch.md` for full requirements.

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Coverage**: 78% (84/108 steps passing)
- **Maintainers**: @llama-orch-maintainers
