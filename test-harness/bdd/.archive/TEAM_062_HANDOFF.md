# TEAM-062 HANDOFF: Error Handling Implementation

**From:** TEAM-061  
**To:** TEAM-062  
**Date:** 2025-10-10  
**Status:** 🚀 READY FOR IMPLEMENTATION

---

## Mission

Implement actual error handling logic in the BDD test suite step definitions, connecting them to real binaries and error scenarios. Transform mock step definitions into production-ready test implementations.

---

## What TEAM-061 Completed

### ✅ Infrastructure Ready
1. **Timeout system** - All HTTP requests, process spawns, and operations have timeouts
2. **Error scenarios** - 35 error scenarios documented in feature file and spec
3. **Step definitions** - 90+ step definitions created as mocks in `src/steps/error_handling.rs`
4. **Documentation** - Complete error taxonomy, retry strategies, and timeout values

### ✅ Files Ready for You
- `src/steps/error_handling.rs` - 90+ mock step definitions (YOUR PRIMARY WORK)
- `tests/features/test-001.feature` - 28 error scenarios to implement
- `bin/.specs/.gherkin/test-001.md` - Error handling specification
- `TEAM_061_ERROR_HANDLING_ANALYSIS.md` - Complete error analysis

---

## Your Mission: Wire Up Step Definitions to Binaries

### Goal
Transform mock step definitions into real implementations that:
1. Actually spawn processes (queen-rbee, rbee-hive, workers)
2. Actually detect errors (timeouts, crashes, connection failures)
3. Actually verify error messages and exit codes
4. Actually clean up processes on failure

### Key Principle
**"Make the tests interact with real binaries, not just log debug messages."**

---

## Architecture Overview

### Current State (TEAM-061)
```rust
#[given(expr = "SSH key at {string} has wrong permissions")]
pub async fn given_ssh_key_wrong_permissions(_world: &mut World, _key_path: String) {
    tracing::debug!("SSH key has wrong permissions");  // ← Just logging
}
```

### Target State (TEAM-062)
```rust
#[given(expr = "SSH key at {string} has wrong permissions")]
pub async fn given_ssh_key_wrong_permissions(world: &mut World, key_path: String) {
    // 1. Create a temporary SSH key with wrong permissions
    let temp_key = world.temp_dir.path().join("bad_key");
    std::fs::write(&temp_key, "fake key").unwrap();
    std::fs::set_permissions(&temp_key, std::fs::Permissions::from_mode(0o644)).unwrap();
    
    // 2. Store in world state for later use
    world.ssh_key_path = Some(temp_key);
    
    tracing::debug!("Created SSH key with wrong permissions");
}
```

---

## Implementation Strategy

### Phase 1: Extend World State (Week 1)
**File:** `src/steps/world.rs`

Add error injection state:
```rust
pub struct World {
    // Existing fields...
    
    // NEW: Error injection state
    pub error_injection: ErrorInjection,
    pub temp_dir: TempDir,
    pub ssh_key_path: Option<PathBuf>,
    pub expected_error: Option<ExpectedError>,
}

pub struct ErrorInjection {
    pub simulate_ssh_timeout: bool,
    pub simulate_http_timeout: bool,
    pub simulate_worker_crash: bool,
    pub simulate_oom: bool,
    pub simulate_disk_full: bool,
}
```

### Phase 2: Add Error Injection to Mock Binaries (Week 1)
**Files:** `src/mock_rbee_hive.rs`, `src/bin/mock-worker.rs`

Add environment variable support:
```rust
// In mock-worker.rs
if std::env::var("SIMULATE_CRASH_ON_STARTUP").is_ok() {
    eprintln!("Worker simulating crash");
    std::process::exit(1);
}

if std::env::var("SIMULATE_OOM").is_ok() {
    std::process::exit(137);  // SIGKILL exit code
}
```

### Phase 3: Implement SSH Errors (Week 1-2)
Connect SSH error scenarios to actual SSH commands with timeouts and retries.

### Phase 4: Implement HTTP Errors (Week 2)
Add HTTP timeout detection, malformed JSON handling, connection loss detection.

### Phase 5: Implement Resource Errors (Week 2-3)
Add RAM/VRAM/disk space checks with actual resource monitoring.

### Phase 6: Implement Worker Lifecycle (Week 3)
Add worker startup, crash detection, port conflict resolution, graceful shutdown.

### Phase 7: Implement Validation & Auth (Week 4)
Add input validation, API key checking, device number validation.

### Phase 8: Implement Cancellation (Week 4)
Add Ctrl+C handling, stream closure detection, DELETE endpoint.

---

## Helper Functions You'll Need

### File: `src/steps/helpers.rs` (NEW)

```rust
/// Wait for process to exit with timeout
pub async fn wait_for_process_exit(pid: u32, timeout: Duration) -> Result<()> {
    let start = std::time::Instant::now();
    loop {
        if !is_process_running(pid) {
            return Ok(());
        }
        if start.elapsed() > timeout {
            anyhow::bail!("Process {} did not exit within {:?}", pid, timeout);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Check if process is running
pub fn is_process_running(pid: u32) -> bool {
    std::process::Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Find next available port
pub async fn find_available_port(base: u16) -> Result<u16> {
    for port in base..base + 100 {
        if tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await.is_ok() {
            return Ok(port);
        }
    }
    anyhow::bail!("No available ports")
}
```

---

## Testing Your Implementation

### Run Individual Scenarios
```bash
cd test-harness/bdd

# Run specific error scenario
cargo run --bin bdd-runner -- --tags @error-handling

# Run specific category
cargo run --bin bdd-runner -- --tags @validation
```

### Verify Timeouts Work
```bash
# Should complete within 5 minutes
timeout 360 cargo run --bin bdd-runner
echo $?  # Should be 0 or 1, not 124 (timeout)
```

### Verify Cleanup Works
```bash
cargo run --bin bdd-runner &
PID=$!
sleep 5
kill -INT $PID
sleep 2
ps aux | grep -E "queen|worker|hive"  # Should be empty
```

---

## Success Criteria

### ✅ Functional Requirements
- [ ] All 28 error scenarios pass
- [ ] Errors are actually triggered (not mocked)
- [ ] Error messages are verified
- [ ] Exit codes are verified
- [ ] Cleanup happens on all paths

### ✅ Non-Functional Requirements
- [ ] Tests complete within 5 minutes
- [ ] No processes left running
- [ ] Ctrl+C cleanly shuts down
- [ ] Tests are deterministic
- [ ] Clear error messages

---

## Resources

### Documentation
- `TEAM_061_ERROR_HANDLING_ANALYSIS.md` - Complete error analysis
- `TEAM_061_TIMEOUT_IMPLEMENTATION.md` - Timeout infrastructure
- `bin/.specs/.gherkin/test-001.md` - Error handling spec

### Code References
- `src/steps/world.rs` - World state and helpers
- `src/steps/error_handling.rs` - Step definitions to implement
- `src/mock_rbee_hive.rs` - Mock server for testing
- `src/bin/mock-worker.rs` - Mock worker for testing

---

## Common Pitfalls to Avoid

1. **Don't forget cleanup** - Always track spawned processes in World
2. **Don't use infinite waits** - Always use timeouts
3. **Don't ignore exit codes** - Verify expected exit codes
4. **Don't hardcode ports** - Find available ports dynamically

---

**TEAM-061 signing off. Good luck TEAM-062!**

**Your primary task:** Implement the 90+ step definitions in `src/steps/error_handling.rs`

🎯 **Transform mocks into real error detection and verification.** 🔥
