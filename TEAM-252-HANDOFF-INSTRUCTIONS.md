# TEAM-252 Handoff Instructions

**Date:** Oct 22, 2025  
**From:** TEAM-251  
**To:** TEAM-252  
**Mission:** Implement Weeks 3-4 of Integration Testing (State Machine + Chaos Tests)

---

## What TEAM-251 Completed (Weeks 1-2) ✅

### Week 1: Test Harness
- ✅ `xtask/src/integration/harness.rs` - Complete test harness
  - Spawns actual binaries
  - Manages process lifecycle
  - Captures stdout/stderr
  - Validates state via health checks
  - Automatic cleanup
  - Test isolation (temp dirs, dynamic ports)

- ✅ `xtask/src/integration/assertions.rs` - Assertion library
  - `assert_success()` / `assert_failure()`
  - `assert_output_contains()`
  - `assert_running()` / `assert_stopped()`
  - 10+ reusable assertions

### Week 2: Command Tests
- ✅ `xtask/src/integration/commands/queen_commands.rs` - 11 tests
  - Queen start (stopped → running)
  - Queen start (already running → idempotent)
  - Queen stop (running → stopped)
  - Queen stop (already stopped → idempotent)
  - Full lifecycle tests
  - Health check tests

- ✅ `xtask/src/integration/commands/hive_commands.rs` - 15 tests
  - Hive start (both stopped → both running)
  - Hive start (queen running → hive starts)
  - Hive start (already running → idempotent)
  - Hive stop tests
  - Hive list tests
  - Hive status tests
  - Full lifecycle tests
  - Heartbeat tests

**Total: 26 integration tests implemented**

---

## Your Mission (Weeks 3-4) 🎯

### Week 3: State Machine Tests
Implement `xtask/src/integration/state_machine.rs`

### Week 4: Chaos Tests
Implement `xtask/src/chaos/` module

---

## Week 3: State Machine Tests (Detailed Instructions)

### Goal
Test **all valid state transitions** systematically using a state machine approach.

### File to Create
`xtask/src/integration/state_machine.rs`

### State Machine Definition

```rust
// TEAM-252: State machine tests
// Purpose: Test all valid state transitions systematically

use crate::integration::harness::{TestHarness, ProcessState, SystemState};
use crate::integration::assertions::*;
use anyhow::Result;
use std::time::Duration;

/// All possible state transitions
#[derive(Debug, Clone)]
pub struct StateTransition {
    pub initial: SystemState,
    pub command: Vec<String>,
    pub expected: SystemState,
    pub description: String,
}

/// Get all valid state transitions
pub fn get_all_transitions() -> Vec<StateTransition> {
    vec![
        // Queen transitions
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            command: vec!["queen".to_string(), "start".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            description: "Queen: Stopped → Running (hive stays stopped)".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            command: vec!["queen".to_string(), "stop".to_string()],
            expected: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            description: "Queen: Running → Stopped".to_string(),
        },
        
        // Hive transitions
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            command: vec!["hive".to_string(), "start".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            description: "Hive start: Both stopped → Both running".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            command: vec!["hive".to_string(), "start".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            description: "Hive start: Queen running → Hive starts".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            command: vec!["hive".to_string(), "stop".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            description: "Hive stop: Hive stops, queen stays running".to_string(),
        },
        
        // Add more transitions here...
        // TODO: Add idempotent transitions (start when running, stop when stopped)
        // TODO: Add cascade transitions (queen stop when hive running)
    ]
}

#[tokio::test]
async fn test_all_state_transitions() {
    // TEAM-252: Test all state transitions systematically
    
    let transitions = get_all_transitions();
    let total = transitions.len();
    
    println!("🔄 Testing {} state transitions...\n", total);
    
    for (i, transition) in transitions.iter().enumerate() {
        println!("📝 Test {}/{}: {}", i + 1, total, transition.description);
        
        let mut harness = TestHarness::new().await.unwrap();
        
        // Set up initial state
        harness.set_state(transition.initial.clone()).await.unwrap();
        
        // Verify initial state
        let actual_initial = harness.get_state().await;
        assert_eq!(actual_initial, transition.initial, 
            "Failed to set up initial state for: {}", transition.description);
        
        // Execute command
        let cmd_refs: Vec<&str> = transition.command.iter().map(|s| s.as_str()).collect();
        let result = harness.run_command(&cmd_refs).await.unwrap();
        assert_success(&result);
        
        // Wait for state to stabilize
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Verify final state
        let actual_final = harness.get_state().await;
        assert_eq!(actual_final, transition.expected, 
            "State transition failed: {}\nExpected: {:?}\nActual: {:?}",
            transition.description, transition.expected, actual_final);
        
        println!("✅ Passed\n");
        
        harness.cleanup().await.unwrap();
    }
    
    println!("🎉 All {} state transitions passed!", total);
}
```

### Your Tasks

1. **Complete the transition list** (add 10+ more transitions)
   - Idempotent transitions (start when running, stop when stopped)
   - Cascade transitions (queen stop when hive running)
   - Edge cases

2. **Add individual transition tests** (one test per transition)
   ```rust
   #[tokio::test]
   async fn test_transition_queen_start_from_stopped() {
       // Test single transition in isolation
   }
   ```

3. **Add transition matrix test** (test all combinations)
   ```rust
   #[tokio::test]
   async fn test_transition_matrix() {
       // Test all possible state combinations
   }
   ```

4. **Add invalid transition tests** (should be rejected or handled gracefully)
   ```rust
   #[tokio::test]
   async fn test_invalid_transitions() {
       // Test transitions that shouldn't be allowed
   }
   ```

### Success Criteria
- ✅ 20+ state transitions defined
- ✅ All transitions tested
- ✅ All tests pass
- ✅ Test execution time < 5 minutes

---

## Week 4: Chaos Tests (Detailed Instructions)

### Goal
Test **error handling and recovery** in failure scenarios.

### Files to Create

#### 1. `xtask/src/chaos/mod.rs`
```rust
// TEAM-252: Chaos testing module
// Purpose: Test error handling and recovery

pub mod binary_failures;
pub mod network_failures;
pub mod process_failures;
pub mod resource_failures;
```

#### 2. `xtask/src/chaos/binary_failures.rs`

```rust
// TEAM-252: Binary failure tests
// Purpose: Test behavior when binaries are missing or corrupt

use crate::integration::harness::TestHarness;
use crate::integration::assertions::*;
use std::fs;
use std::path::PathBuf;

#[tokio::test]
async fn test_binary_not_found() {
    // TEAM-252: Test rbee-hive binary not found
    
    let mut harness = TestHarness::new().await.unwrap();
    
    // Hide the binary (rename it temporarily)
    let binary_path = PathBuf::from("target/debug/rbee-hive");
    let hidden_path = PathBuf::from("target/debug/rbee-hive.hidden");
    
    if binary_path.exists() {
        fs::rename(&binary_path, &hidden_path).unwrap();
    }
    
    // Try to start hive
    let result = harness.run_command(&["hive", "start"]).await.unwrap();
    
    // Should fail with helpful error
    assert_failure(&result);
    assert_output_contains(&result, "binary not found");
    assert_output_contains(&result, "cargo build");
    
    // Restore binary
    if hidden_path.exists() {
        fs::rename(&hidden_path, &binary_path).unwrap();
    }
    
    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_queen_binary_not_found() {
    // TEAM-252: Test queen-rbee binary not found
    
    // Similar to above but for queen-rbee
    // TODO: Implement
}

#[tokio::test]
async fn test_keeper_binary_not_found() {
    // TEAM-252: Test rbee-keeper binary not found
    
    // TODO: Implement
}
```

#### 3. `xtask/src/chaos/network_failures.rs`

```rust
// TEAM-252: Network failure tests
// Purpose: Test behavior with network issues

use crate::integration::harness::TestHarness;
use crate::integration::assertions::*;
use std::net::TcpListener;
use std::time::Duration;

#[tokio::test]
async fn test_port_already_in_use() {
    // TEAM-252: Test queen start when port 9000 is already in use
    
    let mut harness = TestHarness::new().await.unwrap();
    
    // Block port 9000
    let _listener = TcpListener::bind("127.0.0.1:9000").unwrap();
    
    // Try to start queen (should fail or use different port)
    let result = harness.run_command(&["queen", "start"]).await.unwrap();
    
    // Should either fail with helpful error or succeed with different port
    if result.exit_code != Some(0) {
        assert_output_contains(&result, "port");
    }
    
    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_queen_unreachable() {
    // TEAM-252: Test hive list when queen is unreachable
    
    // TODO: Implement
}

#[tokio::test]
async fn test_connection_timeout() {
    // TEAM-252: Test timeout when connecting to queen
    
    // TODO: Implement
}
```

#### 4. `xtask/src/chaos/process_failures.rs`

```rust
// TEAM-252: Process failure tests
// Purpose: Test behavior when processes crash

use crate::integration::harness::TestHarness;
use crate::integration::assertions::*;
use std::time::Duration;

#[tokio::test]
async fn test_queen_crash_during_operation() {
    // TEAM-252: Test queen crash mid-operation
    
    let mut harness = TestHarness::new().await.unwrap();
    
    // Start queen
    let _ = harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();
    
    // Kill queen forcefully
    harness.kill_process("queen-rbee").await.unwrap();
    
    // Try to run command
    let result = harness.run_command(&["hive", "list"]).await.unwrap();
    
    // Should detect queen is down
    assert_failure(&result);
    assert_output_contains(&result, "queen");
    
    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_hive_crash_during_operation() {
    // TEAM-252: Test hive crash mid-operation
    
    // TODO: Implement
}

#[tokio::test]
async fn test_process_hang() {
    // TEAM-252: Test process hang (timeout)
    
    // TODO: Implement
}
```

#### 5. `xtask/src/chaos/resource_failures.rs`

```rust
// TEAM-252: Resource failure tests
// Purpose: Test behavior with resource constraints

use crate::integration::harness::TestHarness;
use crate::integration::assertions::*;

#[tokio::test]
async fn test_disk_full() {
    // TEAM-252: Test behavior when disk is full
    
    // This is hard to test without actually filling disk
    // Consider skipping or mocking
    
    // TODO: Implement or skip
}

#[tokio::test]
async fn test_permission_denied() {
    // TEAM-252: Test permission denied errors
    
    // TODO: Implement
}

#[tokio::test]
async fn test_config_file_missing() {
    // TEAM-252: Test missing config file
    
    // TODO: Implement
}
```

### Your Tasks

1. **Implement all chaos test files** (4 files)
2. **Add 15+ chaos tests total**
   - Binary failures (3-5 tests)
   - Network failures (3-5 tests)
   - Process failures (3-5 tests)
   - Resource failures (3-5 tests)
3. **Add chaos module to main.rs**
4. **Document all tests**

### Success Criteria
- ✅ 15+ chaos tests implemented
- ✅ All error scenarios covered
- ✅ Helpful error messages validated
- ✅ All tests pass

---

## Integration with xtask

### Add to `xtask/src/main.rs`

```rust
mod chaos; // TEAM-252: Chaos testing infrastructure
```

### Add CLI Commands (Optional)

If you want to add CLI commands for running tests:

1. Edit `xtask/src/cli.rs`:
```rust
#[command(name = "integration:test")]
IntegrationTest,

#[command(name = "chaos:test")]
ChaosTest,
```

2. Edit `xtask/src/main.rs`:
```rust
Cmd::IntegrationTest => {
    // Run integration tests
    println!("Running integration tests...");
    std::process::Command::new("cargo")
        .args(&["test", "--package", "xtask", "--lib", "integration"])
        .status()?;
}
Cmd::ChaosTest => {
    // Run chaos tests
    println!("Running chaos tests...");
    std::process::Command::new("cargo")
        .args(&["test", "--package", "xtask", "--lib", "chaos"])
        .status()?;
}
```

---

## Running Tests

### Run All Integration Tests
```bash
# Run all integration tests (TEAM-251 + TEAM-252)
cargo test --package xtask --lib integration

# Run specific test file
cargo test --package xtask --lib integration::commands::queen_commands

# Run with output
cargo test --package xtask --lib integration -- --nocapture
```

### Run State Machine Tests
```bash
cargo test --package xtask --lib integration::state_machine
```

### Run Chaos Tests
```bash
cargo test --package xtask --lib chaos
```

---

## Tips & Best Practices

### 1. Test Isolation
- Each test gets its own `TestHarness`
- Always call `harness.cleanup().await.unwrap()` at the end
- Use temp directories (automatic in harness)

### 2. Timing
- Use `tokio::time::sleep()` for delays
- Use `harness.wait_for_ready()` for health checks
- Be generous with timeouts (10s for startup)

### 3. Assertions
- Use assertion library from `assertions.rs`
- Add custom assertions if needed
- Always check exit codes AND output

### 4. Debugging
- Tests print live output (helpful for debugging)
- Check test logs in console
- Use `-- --nocapture` to see all output

### 5. Error Handling
- Use `.unwrap()` in tests (fail fast)
- Validate error messages are helpful
- Test both success and failure paths

---

## Deliverables Checklist

### Week 3: State Machine Tests
- [ ] `xtask/src/integration/state_machine.rs` created
- [ ] 20+ state transitions defined
- [ ] `test_all_state_transitions()` implemented
- [ ] Individual transition tests added
- [ ] All tests pass
- [ ] Documentation complete

### Week 4: Chaos Tests
- [ ] `xtask/src/chaos/mod.rs` created
- [ ] `xtask/src/chaos/binary_failures.rs` (3-5 tests)
- [ ] `xtask/src/chaos/network_failures.rs` (3-5 tests)
- [ ] `xtask/src/chaos/process_failures.rs` (3-5 tests)
- [ ] `xtask/src/chaos/resource_failures.rs` (3-5 tests)
- [ ] 15+ chaos tests total
- [ ] All tests pass
- [ ] Documentation complete

### Final Handoff
- [ ] `TEAM-252-SUMMARY.md` created
- [ ] All tests documented
- [ ] Test count: 26 (TEAM-251) + 20 (state machine) + 15 (chaos) = 61 total
- [ ] CI/CD integration documented

---

## Questions?

If you have questions:
1. Check `INTEGRATION-TESTING-MASTER-PLAN.md` for detailed architecture
2. Look at existing tests in `commands/` for examples
3. Review `harness.rs` for available methods
4. Check `assertions.rs` for available assertions

---

## Success Metrics

### TEAM-252 Goals
- ✅ 20+ state machine tests
- ✅ 15+ chaos tests
- ✅ 100% of defined transitions tested
- ✅ All error scenarios covered
- ✅ Test execution time < 10 minutes total

### Combined (TEAM-251 + TEAM-252)
- ✅ 61+ integration tests total
- ✅ All commands tested in all states
- ✅ All failure scenarios tested
- ✅ Ready for CI/CD

---

**Good luck, TEAM-252! You've got this! 🚀**

**Status:** 📋 Handoff Complete  
**From:** TEAM-251  
**To:** TEAM-252  
**Date:** Oct 22, 2025
