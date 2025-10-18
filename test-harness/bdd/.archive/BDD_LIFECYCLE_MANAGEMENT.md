# BDD Lifecycle Management

**TEAM-085: Global daemon lifecycle for BDD tests**

---

## Architecture

BDD tests use **global singleton instances** of daemons to avoid port conflicts and improve performance.

```
┌─────────────────────────────────────────────────────────┐
│ BDD Test Suite                                          │
├─────────────────────────────────────────────────────────┤
│ Global queen-rbee (port 8080) - Always running         │
│ Global rbee-hive (port 9200) - On-demand for localhost │
│ Workers (port 8001+) - Spawned by rbee-hive as needed  │
└─────────────────────────────────────────────────────────┘
```

---

## Global Instances

### 1. Global queen-rbee (`global_queen.rs`)

**Created by:** TEAM-051  
**Port:** 8080  
**Lifecycle:** Started once before all tests, cleaned up at end

**Purpose:** Orchestrator daemon for all BDD tests

**Usage:**
```rust
use crate::steps::global_queen;

// Auto-started by test runner
let url = global_queen::get_global_queen_url();
```

### 2. Global rbee-hive (`global_hive.rs`)

**Created by:** TEAM-085  
**Port:** 9200  
**Lifecycle:** Started on-demand when localhost tests need it

**Purpose:** Pool manager for localhost inference tests

**Usage:**
```rust
use crate::steps::global_hive;

// Start on-demand
global_hive::start_global_rbee_hive().await;

let url = global_hive::get_global_hive_url();
```

---

## When to Use Each

### Use Global queen-rbee (always available)
- Testing queen-rbee registry operations
- Testing orchestration logic
- Any test that needs queen-rbee

### Use Global rbee-hive (start on-demand)
- Testing localhost inference
- Testing model provisioning
- Testing worker spawning
- Integration tests that need full stack

---

## Lifecycle Flow

### Test Suite Startup
1. `cucumber.rs` starts test runner
2. Global queen-rbee auto-starts (port 8080)
3. Tests run
4. Cleanup kills all global instances

### Individual Test
1. Test step calls `start_global_rbee_hive()` if needed
2. Singleton pattern ensures only one instance
3. Test uses the global instance
4. No cleanup needed (handled at suite end)

---

## Implementation Details

### Singleton Pattern
Both modules use `std::sync::OnceLock` for thread-safe singleton:

```rust
static GLOBAL_QUEEN: OnceLock<GlobalQueenRbee> = OnceLock::new();
static GLOBAL_HIVE: OnceLock<GlobalRbeeHive> = OnceLock::new();
```

### Startup Logic
1. Check if already initialized
2. Find binary in `target/debug/`
3. Spawn process with proper args
4. Wait for health endpoint (30s timeout)
5. Store in singleton

### Cleanup Logic
1. `cleanup_global_hive()` - Kill rbee-hive
2. `cleanup_global_queen()` - Kill queen-rbee
3. Called from `cucumber.rs` before exit

---

## Port Allocation

| Component | Port | Lifecycle |
|-----------|------|-----------|
| queen-rbee | 8080 | Always running |
| rbee-hive | 9200 | On-demand |
| worker-001 | 8001 | Spawned by rbee-hive |
| worker-002 | 8002 | Spawned by rbee-hive |
| worker-N | 8000+N | Spawned by rbee-hive |

---

## Example: Integration Test

```gherkin
Feature: End-to-End Integration Tests
  Scenario: Complete inference workflow
    Given queen-rbee is running           # Uses global instance
    And rbee-hive is running on workstation  # Starts global rbee-hive
    When user submits inference request
    Then tokens are streamed back
```

**Step implementation:**
```rust
#[given(expr = "rbee-hive is running on workstation")]
pub async fn given_rbee_hive_running(world: &mut World) {
    // TEAM-085: Auto-start global rbee-hive
    crate::steps::global_hive::start_global_rbee_hive().await;
    
    tracing::info!("✅ rbee-hive is running");
}
```

---

## Benefits

1. **No Port Conflicts** - Single instance per daemon
2. **Fast Tests** - No repeated startup/shutdown
3. **Realistic** - Tests actual daemon processes
4. **Clean** - Automatic cleanup at suite end

---

## Files

- `src/steps/global_queen.rs` - queen-rbee lifecycle
- `src/steps/global_hive.rs` - rbee-hive lifecycle (TEAM-085)
- `tests/cucumber.rs` - Test runner with cleanup
- `src/steps/mod.rs` - Module exports

---

**Created by:** TEAM-051 (queen), TEAM-085 (hive)  
**Date:** 2025-10-11  
**Status:** ACTIVE
