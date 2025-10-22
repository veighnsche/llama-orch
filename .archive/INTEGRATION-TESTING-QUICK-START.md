# Integration Testing Quick Start

**TL;DR:** Test all commands with actual binaries in all states using a custom test harness

---

## The Plan

### 3-Tier Approach

1. **BDD Tests** (Gherkin) - User scenarios
2. **State Machine Tests** (Rust) - All state transitions  
3. **Chaos Tests** (Rust) - Failure scenarios

### Test Harness

Custom Rust harness that:
- ✅ Spawns actual binaries (`rbee-keeper`, `queen-rbee`, `rbee-hive`)
- ✅ Manages process lifecycle
- ✅ Captures stdout/stderr
- ✅ Validates state transitions
- ✅ Cleans up automatically
- ✅ Isolates each test (temp dirs, dynamic ports)

---

## Quick Example

```rust
#[tokio::test]
async fn test_queen_start_when_stopped() {
    let mut harness = TestHarness::new().await?;
    
    // Precondition: Queen is stopped
    assert!(!harness.is_running("queen"));
    
    // Action: Start queen
    let result = harness.run_command(&["queen", "start"]).await?;
    
    // Assertions
    assert_eq!(result.exit_code, Some(0));
    assert!(result.stdout.contains("✅"));
    
    // Postcondition: Queen is running
    harness.wait_for_ready("queen", Duration::from_secs(10)).await?;
    assert!(harness.is_running("queen"));
    
    harness.cleanup().await?;
}
```

---

## What Gets Tested

### All Commands
- `rbee queen start` (stopped → running)
- `rbee queen start` (already running → idempotent)
- `rbee queen stop` (running → stopped)
- `rbee queen stop` (already stopped → idempotent)
- `rbee hive start` (auto-starts queen if needed)
- `rbee hive stop`
- `rbee hive list`
- `rbee hive status`
- `rbee hive get`
- ... (all commands)

### All State Transitions
- Queen: Stopped → Running
- Queen: Running → Stopped
- Hive: Stopped → Running (with Queen: Stopped)
- Hive: Stopped → Running (with Queen: Running)
- Hive: Running → Stopped
- ... (all valid transitions)

### All Failure Scenarios
- Binary not found
- Port already in use
- Process crashes mid-operation
- Network failures
- Timeout scenarios
- Permission denied
- ... (all error cases)

---

## File Structure

```
xtask/src/
├── integration/
│   ├── mod.rs                    # ✅ Created
│   ├── harness.rs                # ✅ Created (test harness)
│   ├── commands/
│   │   ├── queen_commands.rs     # ⏳ TODO
│   │   ├── hive_commands.rs      # ⏳ TODO
│   │   └── ...
│   ├── state_machine.rs          # ⏳ TODO
│   └── assertions.rs             # ⏳ TODO
├── chaos/
│   └── ...                       # ⏳ TODO
└── e2e/
    ├── queen_lifecycle.rs        # ✅ Exists
    ├── hive_lifecycle.rs         # ✅ Exists
    └── cascade_shutdown.rs       # ✅ Exists
```

---

## Running Tests

```bash
# All integration tests
cargo xtask integration:test

# Specific command tests
cargo xtask integration:test --filter queen

# State machine tests
cargo xtask integration:state-machine

# Chaos tests
cargo xtask chaos:test

# BDD tests (existing)
cargo xtask bdd:test

# All tests
cargo xtask test:all
```

---

## Key Features

### Isolation
- ✅ Each test gets unique temp directory
- ✅ Dynamic port allocation (no conflicts)
- ✅ Separate config/data per test
- ✅ Automatic cleanup on completion

### Validation
- ✅ Exit code checking
- ✅ Output validation (stdout/stderr)
- ✅ State verification (is process running?)
- ✅ Health endpoint checking

### Debugging
- ✅ Live output during tests
- ✅ Test ID for tracking
- ✅ Detailed error messages
- ✅ Test logs preserved on failure

---

## Implementation Status

### ✅ Complete
- Master plan document
- Test harness skeleton
- Integration module structure

### ⏳ TODO (4 weeks)
- Week 1: Complete test harness
- Week 2: Implement command tests
- Week 3: Implement state machine tests
- Week 4: Implement chaos tests

---

## Next Steps

1. Review `INTEGRATION-TESTING-MASTER-PLAN.md`
2. Complete test harness implementation
3. Add command tests one by one
4. Add to CI/CD pipeline

---

**Status:** 📋 Plan Complete, Implementation Started  
**Estimated Effort:** 4 weeks (1 developer)  
**Expected Value:** 100% command coverage, all states tested
