# Integration Testing Master Plan

**Date:** Oct 22, 2025  
**Purpose:** Test all commands with actual binaries in different states  
**Approach:** Hybrid - BDD + Custom Test Harness + State Machine Testing

---

## Executive Summary

**Goal:** Test every command (`rbee hive start`, `rbee queen stop`, etc.) in every valid state combination.

**Strategy:** 3-tier testing approach:
1. **BDD Tests** (Gherkin) - User-facing scenarios
2. **State Machine Tests** (Rust) - All state transitions
3. **Chaos Tests** (Rust) - Failure scenarios

**Infrastructure:** Custom test harness that:
- Spawns actual binaries
- Manages process lifecycle
- Captures stdout/stderr
- Validates state transitions
- Cleans up automatically

---

## Current State Analysis

### ✅ What Exists
1. **BDD Infrastructure** (`xtask/src/tasks/bdd/`)
   - Runner with live output
   - Tag filtering
   - Failure reporting
   - Auto-rerun failing tests

2. **E2E Tests** (`xtask/src/e2e/`)
   - Queen lifecycle test
   - Hive lifecycle test
   - Cascade shutdown test
   - Uses `.spawn()` for live narration

3. **xtask Commands**
   - `cargo xtask bdd:test` - Run BDD tests
   - `cargo xtask e2e:queen` - Test queen lifecycle
   - `cargo xtask e2e:hive` - Test hive lifecycle

### ❌ What's Missing
1. **State Machine Testing** - Test all state transitions
2. **Command Matrix** - Test all commands in all states
3. **Isolation** - Each test in clean environment
4. **Assertions** - Validate output, exit codes, side effects
5. **Parallel Execution** - Run tests concurrently
6. **Failure Injection** - Test error scenarios

---

## Proposed Architecture

### Tier 1: BDD Tests (User Scenarios)

**Purpose:** Test user-facing workflows in Gherkin

**Example:**
```gherkin
Feature: Hive Management
  Scenario: Start and stop a hive
    Given queen is running
    When I run "rbee hive start"
    Then I should see "✅ Hive 'localhost' started successfully"
    And the hive should be reachable
    When I run "rbee hive stop"
    Then I should see "✅ Hive 'localhost' stopped successfully"
```

**Location:** `bin/*/bdd/tests/features/*.feature`

**Run:** `cargo xtask bdd:test`

---

### Tier 2: State Machine Tests (All Transitions)

**Purpose:** Test every command in every valid state

**State Matrix:**

| State | Command | Expected Result |
|-------|---------|-----------------|
| Queen: Stopped | `rbee queen start` | ✅ Starts |
| Queen: Running | `rbee queen start` | ✅ Already running |
| Queen: Running | `rbee queen stop` | ✅ Stops |
| Queen: Stopped | `rbee queen stop` | ✅ Not running |
| Hive: Stopped, Queen: Stopped | `rbee hive start` | ✅ Starts queen + hive |
| Hive: Stopped, Queen: Running | `rbee hive start` | ✅ Starts hive |
| Hive: Running | `rbee hive start` | ✅ Already running |
| Hive: Running | `rbee hive stop` | ✅ Stops |
| Hive: Stopped | `rbee hive stop` | ✅ Not running |

**Implementation:** Custom Rust test harness

**Location:** `xtask/src/integration/`

**Run:** `cargo xtask integration:test`

---

### Tier 3: Chaos Tests (Failure Scenarios)

**Purpose:** Test error handling and recovery

**Scenarios:**
- Binary not found
- Port already in use
- Network failures
- Process crashes
- Timeout scenarios
- Disk full
- Permission denied

**Location:** `xtask/src/chaos/`

**Run:** `cargo xtask chaos:test`

---

## Implementation Plan

### Phase 1: Test Harness (Week 1)

Create `xtask/src/integration/harness.rs`:

```rust
/// Test harness for integration tests
pub struct TestHarness {
    /// Temporary directory for test isolation
    temp_dir: TempDir,
    /// Running processes
    processes: HashMap<String, Child>,
    /// Test state
    state: TestState,
}

impl TestHarness {
    /// Create new isolated test environment
    pub fn new() -> Result<Self> {
        let temp_dir = TempDir::new()?;
        
        // Set up isolated environment
        env::set_var("RBEE_CONFIG_DIR", temp_dir.path());
        env::set_var("RBEE_DATA_DIR", temp_dir.path());
        
        Ok(Self {
            temp_dir,
            processes: HashMap::new(),
            state: TestState::default(),
        })
    }
    
    /// Run a command and capture output
    pub async fn run_command(&mut self, cmd: &[&str]) -> CommandResult {
        let binary = self.find_binary("rbee-keeper")?;
        
        let mut child = Command::new(binary)
            .args(cmd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        
        // Capture output
        let output = child.wait_with_output().await?;
        
        CommandResult {
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        }
    }
    
    /// Check if process is running
    pub fn is_running(&self, name: &str) -> bool {
        // Check via health endpoint or process list
        self.check_health(name).is_ok()
    }
    
    /// Wait for process to be ready
    pub async fn wait_for_ready(&self, name: &str, timeout: Duration) -> Result<()> {
        let start = Instant::now();
        
        loop {
            if self.is_running(name) {
                return Ok(());
            }
            
            if start.elapsed() > timeout {
                bail!("Timeout waiting for {} to be ready", name);
            }
            
            sleep(Duration::from_millis(100)).await;
        }
    }
    
    /// Clean up all processes
    pub async fn cleanup(&mut self) -> Result<()> {
        // Kill all processes
        for (name, mut child) in self.processes.drain() {
            let _ = child.kill().await;
        }
        
        // Clean up temp dir (automatic on drop)
        Ok(())
    }
}

impl Drop for TestHarness {
    fn drop(&mut self) {
        // Ensure cleanup happens
        let _ = futures::executor::block_on(self.cleanup());
    }
}
```

---

### Phase 2: Command Tests (Week 2)

Create `xtask/src/integration/commands/` with tests for each command:

#### `queen_commands.rs`:
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
    assert!(result.stdout.contains("started"));
    
    // Postcondition: Queen is running
    harness.wait_for_ready("queen", Duration::from_secs(10)).await?;
    assert!(harness.is_running("queen"));
    
    harness.cleanup().await?;
}

#[tokio::test]
async fn test_queen_start_when_already_running() {
    let mut harness = TestHarness::new().await?;
    
    // Precondition: Queen is running
    harness.run_command(&["queen", "start"]).await?;
    harness.wait_for_ready("queen", Duration::from_secs(10)).await?;
    
    // Action: Start queen again
    let result = harness.run_command(&["queen", "start"]).await?;
    
    // Assertions
    assert_eq!(result.exit_code, Some(0));
    assert!(result.stdout.contains("already running"));
    
    harness.cleanup().await?;
}

#[tokio::test]
async fn test_queen_stop_when_running() {
    let mut harness = TestHarness::new().await?;
    
    // Precondition: Queen is running
    harness.run_command(&["queen", "start"]).await?;
    harness.wait_for_ready("queen", Duration::from_secs(10)).await?;
    
    // Action: Stop queen
    let result = harness.run_command(&["queen", "stop"]).await?;
    
    // Assertions
    assert_eq!(result.exit_code, Some(0));
    assert!(result.stdout.contains("✅"));
    assert!(result.stdout.contains("stopped"));
    
    // Postcondition: Queen is stopped
    sleep(Duration::from_secs(1)).await;
    assert!(!harness.is_running("queen"));
    
    harness.cleanup().await?;
}

#[tokio::test]
async fn test_queen_stop_when_already_stopped() {
    let mut harness = TestHarness::new().await?;
    
    // Precondition: Queen is stopped
    assert!(!harness.is_running("queen"));
    
    // Action: Stop queen
    let result = harness.run_command(&["queen", "stop"]).await?;
    
    // Assertions
    assert_eq!(result.exit_code, Some(0));
    assert!(result.stdout.contains("not running") || result.stdout.contains("already stopped"));
    
    harness.cleanup().await?;
}
```

#### `hive_commands.rs`:
```rust
#[tokio::test]
async fn test_hive_start_starts_queen_if_needed() {
    let mut harness = TestHarness::new().await?;
    
    // Precondition: Nothing running
    assert!(!harness.is_running("queen"));
    assert!(!harness.is_running("hive"));
    
    // Action: Start hive
    let result = harness.run_command(&["hive", "start"]).await?;
    
    // Assertions
    assert_eq!(result.exit_code, Some(0));
    assert!(result.stdout.contains("✅"));
    
    // Postcondition: Both running
    harness.wait_for_ready("queen", Duration::from_secs(10)).await?;
    harness.wait_for_ready("hive", Duration::from_secs(10)).await?;
    assert!(harness.is_running("queen"));
    assert!(harness.is_running("hive"));
    
    harness.cleanup().await?;
}

#[tokio::test]
async fn test_hive_list() {
    let mut harness = TestHarness::new().await?;
    
    // Precondition: Queen running
    harness.run_command(&["queen", "start"]).await?;
    harness.wait_for_ready("queen", Duration::from_secs(10)).await?;
    
    // Action: List hives
    let result = harness.run_command(&["hive", "list"]).await?;
    
    // Assertions
    assert_eq!(result.exit_code, Some(0));
    assert!(result.stdout.contains("localhost") || result.stdout.contains("hive"));
    
    harness.cleanup().await?;
}

#[tokio::test]
async fn test_hive_status() {
    let mut harness = TestHarness::new().await?;
    
    // Precondition: Hive running
    harness.run_command(&["hive", "start"]).await?;
    harness.wait_for_ready("hive", Duration::from_secs(10)).await?;
    
    // Action: Check status
    let result = harness.run_command(&["hive", "status"]).await?;
    
    // Assertions
    assert_eq!(result.exit_code, Some(0));
    assert!(result.stdout.contains("running") || result.stdout.contains("✅"));
    
    harness.cleanup().await?;
}
```

---

### Phase 3: State Matrix Tests (Week 3)

Create `xtask/src/integration/state_machine.rs`:

```rust
/// All possible system states
#[derive(Debug, Clone, PartialEq)]
pub struct SystemState {
    queen: ProcessState,
    hive: ProcessState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProcessState {
    Stopped,
    Starting,
    Running,
    Stopping,
}

/// Test all state transitions
#[tokio::test]
async fn test_all_state_transitions() {
    let transitions = vec![
        // (Initial State, Command, Expected Final State)
        (
            SystemState { queen: Stopped, hive: Stopped },
            &["queen", "start"],
            SystemState { queen: Running, hive: Stopped },
        ),
        (
            SystemState { queen: Running, hive: Stopped },
            &["hive", "start"],
            SystemState { queen: Running, hive: Running },
        ),
        (
            SystemState { queen: Running, hive: Running },
            &["hive", "stop"],
            SystemState { queen: Running, hive: Stopped },
        ),
        (
            SystemState { queen: Running, hive: Stopped },
            &["queen", "stop"],
            SystemState { queen: Stopped, hive: Stopped },
        ),
        // ... more transitions
    ];
    
    for (initial, command, expected) in transitions {
        let mut harness = TestHarness::new().await?;
        
        // Set up initial state
        harness.set_state(initial).await?;
        
        // Execute command
        let result = harness.run_command(command).await?;
        assert_eq!(result.exit_code, Some(0));
        
        // Verify final state
        let actual = harness.get_state().await?;
        assert_eq!(actual, expected, 
            "State transition failed: {:?} + {:?} should result in {:?}, got {:?}",
            initial, command, expected, actual
        );
        
        harness.cleanup().await?;
    }
}
```

---

### Phase 4: Chaos Tests (Week 4)

Create `xtask/src/chaos/`:

```rust
#[tokio::test]
async fn test_binary_not_found() {
    let mut harness = TestHarness::new().await?;
    
    // Remove binary
    harness.hide_binary("rbee-hive")?;
    
    // Try to start hive
    let result = harness.run_command(&["hive", "start"]).await?;
    
    // Should fail with helpful error
    assert_ne!(result.exit_code, Some(0));
    assert!(result.stderr.contains("binary not found"));
    assert!(result.stderr.contains("cargo build"));
}

#[tokio::test]
async fn test_port_already_in_use() {
    let mut harness = TestHarness::new().await?;
    
    // Start process on port 9000
    let _blocker = harness.block_port(9000).await?;
    
    // Try to start queen
    let result = harness.run_command(&["queen", "start"]).await?;
    
    // Should fail with helpful error
    assert_ne!(result.exit_code, Some(0));
    assert!(result.stderr.contains("port") || result.stderr.contains("address in use"));
}

#[tokio::test]
async fn test_process_crash_during_operation() {
    let mut harness = TestHarness::new().await?;
    
    // Start queen
    harness.run_command(&["queen", "start"]).await?;
    harness.wait_for_ready("queen", Duration::from_secs(10)).await?;
    
    // Kill queen mid-operation
    harness.kill_process("queen").await?;
    
    // Try to run command
    let result = harness.run_command(&["hive", "list"]).await?;
    
    // Should detect queen is down
    assert_ne!(result.exit_code, Some(0));
    assert!(result.stderr.contains("queen") && result.stderr.contains("not running"));
}
```

---

## Test Organization

```
xtask/src/
├── integration/
│   ├── mod.rs                    # Main integration test module
│   ├── harness.rs                # Test harness implementation
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── queen_commands.rs     # Queen start/stop tests
│   │   ├── hive_commands.rs      # Hive start/stop/list/status tests
│   │   ├── worker_commands.rs    # Worker tests (future)
│   │   └── model_commands.rs     # Model tests (future)
│   ├── state_machine.rs          # State transition tests
│   └── assertions.rs             # Custom assertions
├── chaos/
│   ├── mod.rs
│   ├── binary_failures.rs        # Binary not found, corrupt, etc.
│   ├── network_failures.rs       # Port conflicts, timeouts
│   ├── process_failures.rs       # Crashes, hangs
│   └── resource_failures.rs      # Disk full, memory exhausted
└── e2e/
    ├── mod.rs                    # Existing E2E tests
    ├── queen_lifecycle.rs
    ├── hive_lifecycle.rs
    └── cascade_shutdown.rs
```

---

## Running Tests

### All Integration Tests
```bash
cargo xtask integration:test
```

### Specific Command Tests
```bash
cargo xtask integration:test --filter queen
cargo xtask integration:test --filter hive
```

### State Machine Tests
```bash
cargo xtask integration:state-machine
```

### Chaos Tests
```bash
cargo xtask chaos:test
```

### BDD Tests (User Scenarios)
```bash
cargo xtask bdd:test
cargo xtask bdd:test --tags @hive
cargo xtask bdd:test --feature lifecycle
```

### All Tests (Full Suite)
```bash
cargo xtask test:all
```

---

## Test Isolation Strategy

### Per-Test Isolation
1. **Temporary Directory** - Each test gets unique temp dir
2. **Environment Variables** - Override config/data paths
3. **Port Allocation** - Dynamic port assignment
4. **Process Cleanup** - Automatic on test completion
5. **Database Isolation** - Separate DB per test

### Implementation:
```rust
impl TestHarness {
    pub fn new() -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let test_id = Uuid::new_v4();
        
        // Isolated config
        env::set_var("RBEE_CONFIG_DIR", temp_dir.path().join("config"));
        env::set_var("RBEE_DATA_DIR", temp_dir.path().join("data"));
        env::set_var("RBEE_TEST_ID", test_id.to_string());
        
        // Dynamic ports
        let queen_port = find_free_port()?;
        let hive_port = find_free_port()?;
        
        env::set_var("RBEE_QUEEN_PORT", queen_port.to_string());
        env::set_var("RBEE_HIVE_PORT", hive_port.to_string());
        
        Ok(Self { temp_dir, test_id, queen_port, hive_port, ... })
    }
}
```

---

## Assertions Library

Create `xtask/src/integration/assertions.rs`:

```rust
/// Assert command succeeded
pub fn assert_success(result: &CommandResult) {
    assert_eq!(result.exit_code, Some(0), 
        "Command failed with exit code {:?}\nStderr: {}", 
        result.exit_code, result.stderr
    );
}

/// Assert command failed
pub fn assert_failure(result: &CommandResult) {
    assert_ne!(result.exit_code, Some(0), 
        "Command should have failed but succeeded"
    );
}

/// Assert output contains text
pub fn assert_output_contains(result: &CommandResult, text: &str) {
    assert!(
        result.stdout.contains(text) || result.stderr.contains(text),
        "Output should contain '{}'\nStdout: {}\nStderr: {}",
        text, result.stdout, result.stderr
    );
}

/// Assert process is running
pub async fn assert_running(harness: &TestHarness, name: &str) {
    assert!(harness.is_running(name), "{} should be running", name);
}

/// Assert process is stopped
pub async fn assert_stopped(harness: &TestHarness, name: &str) {
    assert!(!harness.is_running(name), "{} should be stopped", name);
}
```

---

## CI/CD Integration

### GitHub Actions Workflow:
```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build binaries
        run: cargo build --workspace
      
      - name: Run integration tests
        run: cargo xtask integration:test
      
      - name: Run state machine tests
        run: cargo xtask integration:state-machine
      
      - name: Run chaos tests
        run: cargo xtask chaos:test
      
      - name: Upload test logs
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: test-logs
          path: xtask/.test-logs/
```

---

## Success Metrics

### Coverage Goals
- ✅ All commands tested in all valid states
- ✅ All error scenarios tested
- ✅ All state transitions tested
- ✅ 100% of user-facing commands covered

### Quality Goals
- ✅ Tests run in <5 minutes
- ✅ 99%+ pass rate (no flaky tests)
- ✅ Automatic cleanup (no leftover processes)
- ✅ Clear failure messages

### Maintenance Goals
- ✅ Easy to add new command tests
- ✅ Easy to add new state transitions
- ✅ Easy to debug failures
- ✅ Self-documenting test code

---

## Timeline

### Week 1: Test Harness
- Day 1-2: Implement TestHarness
- Day 3-4: Implement isolation
- Day 5: Implement assertions

### Week 2: Command Tests
- Day 1-2: Queen commands
- Day 3-4: Hive commands
- Day 5: Integration with xtask

### Week 3: State Machine
- Day 1-3: Implement state machine tests
- Day 4-5: Test all transitions

### Week 4: Chaos Tests
- Day 1-2: Binary failures
- Day 3-4: Network/process failures
- Day 5: CI/CD integration

**Total: 4 weeks (1 developer)**

---

## Next Steps

1. ✅ Review this plan
2. ⏳ Implement TestHarness (Week 1)
3. ⏳ Implement command tests (Week 2)
4. ⏳ Implement state machine tests (Week 3)
5. ⏳ Implement chaos tests (Week 4)
6. ⏳ Integrate into CI/CD

---

**Status:** 📋 PLAN COMPLETE - READY FOR IMPLEMENTATION  
**Estimated Effort:** 4 weeks (1 developer)  
**Expected Value:** 100% command coverage, all states tested
