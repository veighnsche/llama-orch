# TEAM-061 HANDOFF - TIMEOUTS & ROBUST ERROR HANDLING

**From:** TEAM-060  
**To:** TEAM-061  
**Date:** 2025-10-10  
**Priority:** 🔴 CRITICAL - Tests Currently Hang  
**Status:** Phase 5 Complete, Phase 6 Blocked by Hangs

---

## Executive Summary

**CRITICAL ISSUE:** The BDD test suite hangs indefinitely during execution. TEAM-060 completed Phase 5 (real edge cases), but Phase 6 (verification) is blocked because tests never complete.

**Your Mission:** 
1. Add comprehensive timeouts to prevent hangs
2. Analyze and implement robust error handling throughout the test infrastructure
3. Get the test suite to 62/62 passing with proper cleanup

**Why This Matters:** Without timeouts, a single hanging test blocks all development. Without robust error handling, we can't diagnose failures. This is a **blocker for the entire project**.

---

## The Hanging Problem

### Current Behavior

```bash
$ cargo run --bin bdd-runner
# ... starts running ...
# ... hangs forever, never completes ...
# Ctrl+C doesn't work cleanly
# Must kill -9 all processes
```

### Where It Hangs (Suspected)

Based on TEAM-058 and TEAM-059 analysis:

1. **Global queen-rbee initialization** (`src/steps/global_queen.rs`)
   - Starts queen-rbee process
   - Waits for port 8080 to be ready
   - May hang if port check never succeeds

2. **Mock rbee-hive startup** (`src/mock_rbee_hive.rs`)
   - Binds to port 9200
   - May hang if port already in use
   - No timeout on server startup

3. **Worker spawning** (`src/bin/mock-worker.rs`)
   - Spawns worker processes
   - Sends ready callback
   - May hang if callback never completes

4. **HTTP requests without timeouts** (various step files)
   - `reqwest::Client::new()` has no default timeout
   - Requests can hang forever waiting for response

5. **Process cleanup** (`Drop` implementations)
   - May hang waiting for processes to die
   - No timeout on `wait()` calls

---

## Priority 1: Add Timeouts to ALL Async Operations

### 1.1 HTTP Client Timeouts

**File:** `src/steps/world.rs` or create `src/http_client.rs`

**Problem:** All HTTP requests use `reqwest::Client::new()` with no timeout.

**Solution:**
```rust
// TEAM-061: Create HTTP client with aggressive timeouts
pub fn create_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))        // Total request timeout
        .connect_timeout(Duration::from_secs(5)) // Connection timeout
        .build()
        .expect("Failed to create HTTP client")
}
```

**Files to update:**
- `src/steps/beehive_registry.rs` - All HTTP calls
- `src/steps/happy_path.rs` - Worker spawn calls
- `src/steps/lifecycle.rs` - Worker management calls
- `src/steps/edge_cases.rs` - HTTP auth tests

**Pattern:**
```rust
// BEFORE (WRONG - can hang forever):
let client = reqwest::Client::new();
let resp = client.get(url).send().await?;

// AFTER (RIGHT - times out after 10s):
let client = create_http_client();
let resp = client.get(url).send().await?;
```

### 1.2 Process Spawn Timeouts

**File:** `src/steps/global_queen.rs`

**Problem:** Waits forever for queen-rbee to be ready.

**Current code (lines 60-80):**
```rust
// Wait for queen-rbee to be ready
loop {
    if let Ok(_) = std::net::TcpStream::connect_timeout(...) {
        break;
    }
    sleep(Duration::from_millis(100)).await;
    // NO TIMEOUT - HANGS FOREVER!
}
```

**Solution:**
```rust
// TEAM-061: Add timeout to queen-rbee readiness check
let start = std::time::Instant::now();
let timeout = Duration::from_secs(30);

loop {
    if start.elapsed() > timeout {
        tracing::error!("❌ Queen-rbee failed to start within 30s");
        panic!("Queen-rbee startup timeout");
    }
    
    if let Ok(_) = std::net::TcpStream::connect_timeout(...) {
        tracing::info!("✅ Queen-rbee ready after {:?}", start.elapsed());
        break;
    }
    
    sleep(Duration::from_millis(100)).await;
}
```

### 1.3 Mock rbee-hive Startup Timeout

**File:** `src/mock_rbee_hive.rs`

**Problem:** `start_mock_rbee_hive()` may hang if port is in use.

**Solution:**
```rust
// TEAM-061: Add timeout wrapper for server startup
pub async fn start_mock_rbee_hive_with_timeout() -> String {
    tokio::time::timeout(
        Duration::from_secs(10),
        start_mock_rbee_hive()
    )
    .await
    .expect("Mock rbee-hive startup timeout after 10s")
}
```

### 1.4 Worker Ready Callback Timeout

**File:** `src/bin/mock-worker.rs`

**Problem:** `send_ready_callback()` may hang if queen-rbee not responding.

**Current code (lines 100-120):**
```rust
async fn send_ready_callback(...) -> Result<()> {
    let client = reqwest::Client::new();  // NO TIMEOUT!
    let resp = client.post(url).json(&payload).send().await?;
    // Can hang forever
}
```

**Solution:**
```rust
// TEAM-061: Add timeout to ready callback
async fn send_ready_callback(...) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    
    let resp = tokio::time::timeout(
        Duration::from_secs(10),
        client.post(url).json(&payload).send()
    )
    .await
    .map_err(|_| anyhow::anyhow!("Ready callback timeout after 10s"))??;
    
    Ok(())
}
```

### 1.5 Test Scenario Timeout Wrapper

**File:** `src/main.rs` or `src/lib.rs`

**Problem:** Individual scenarios can hang forever.

**Solution:** Add timeout to cucumber runner:
```rust
// TEAM-061: Wrap entire test execution in timeout
#[tokio::main]
async fn main() {
    let result = tokio::time::timeout(
        Duration::from_secs(300), // 5 minutes for entire suite
        run_tests()
    )
    .await;
    
    match result {
        Ok(Ok(_)) => {
            tracing::info!("✅ All tests completed");
            std::process::exit(0);
        }
        Ok(Err(e)) => {
            tracing::error!("❌ Tests failed: {}", e);
            std::process::exit(1);
        }
        Err(_) => {
            tracing::error!("❌ Test suite timeout after 5 minutes");
            cleanup_all_processes();
            std::process::exit(124); // Timeout exit code
        }
    }
}
```

---

## Priority 2: Robust Error Handling Analysis

### 2.1 Process Cleanup on Failure

**Problem:** When tests fail or hang, processes are left running.

**Files to audit:**
- `src/steps/global_queen.rs` - Queen-rbee cleanup
- `src/mock_rbee_hive.rs` - Mock server cleanup
- `src/steps/world.rs` - World cleanup on scenario end

**Required improvements:**

1. **Add cleanup hook to World:**
```rust
// TEAM-061: Ensure cleanup always runs
impl Drop for World {
    fn drop(&mut self) {
        tracing::info!("🧹 Cleaning up World resources");
        
        // Kill all spawned workers
        for worker in &mut self.worker_processes {
            let _ = worker.start_kill();
        }
        
        // Kill mock rbee-hive if running
        if let Some(mut hive) = self.rbee_hive_process.take() {
            let _ = hive.start_kill();
        }
        
        // Give processes time to die
        std::thread::sleep(Duration::from_millis(500));
        
        tracing::info!("✅ World cleanup complete");
    }
}
```

2. **Add signal handler for Ctrl+C:**
```rust
// TEAM-061: Clean shutdown on Ctrl+C
tokio::spawn(async {
    tokio::signal::ctrl_c().await.ok();
    tracing::warn!("🛑 Ctrl+C received, cleaning up...");
    cleanup_all_processes();
    std::process::exit(130);
});
```

3. **Add panic handler:**
```rust
// TEAM-061: Cleanup on panic
std::panic::set_hook(Box::new(|panic_info| {
    eprintln!("💥 Panic: {:?}", panic_info);
    cleanup_all_processes();
}));
```

### 2.2 Port Conflict Detection

**Problem:** Tests fail silently if ports are already in use.

**Solution:**
```rust
// TEAM-061: Check port availability before binding
fn check_port_available(port: u16) -> Result<()> {
    match std::net::TcpListener::bind(("127.0.0.1", port)) {
        Ok(_) => Ok(()),
        Err(e) => {
            tracing::error!("❌ Port {} already in use", port);
            
            // Try to find what's using it
            let output = std::process::Command::new("lsof")
                .arg("-i")
                .arg(format!(":{}", port))
                .output()
                .ok();
            
            if let Some(output) = output {
                let processes = String::from_utf8_lossy(&output.stdout);
                tracing::error!("Processes using port {}:\n{}", port, processes);
            }
            
            Err(anyhow::anyhow!("Port {} conflict: {}", port, e))
        }
    }
}

// Use before starting servers:
check_port_available(8080)?; // queen-rbee
check_port_available(9200)?; // mock rbee-hive
```

### 2.3 Binary Existence Checks

**Problem:** Tests fail with cryptic errors if binaries don't exist.

**Solution:**
```rust
// TEAM-061: Verify binaries exist before spawning
fn check_binary_exists(name: &str) -> Result<PathBuf> {
    let workspace_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(|p| PathBuf::from(p).parent().unwrap().parent().unwrap().to_path_buf())
        .unwrap_or_else(|_| PathBuf::from("/home/vince/Projects/llama-orch"));
    
    let binary_path = workspace_dir.join("target/debug").join(name);
    
    if !binary_path.exists() {
        return Err(anyhow::anyhow!(
            "Binary not found: {}\n\
             Run: cargo build --bin {}",
            binary_path.display(),
            name
        ));
    }
    
    Ok(binary_path)
}

// Use before spawning:
let queen_path = check_binary_exists("queen-rbee")?;
let worker_path = check_binary_exists("mock-worker")?;
```

### 2.4 Detailed Error Context

**Problem:** When tests fail, we don't know why.

**Solution:** Add context to all errors:
```rust
// TEAM-061: Rich error context
use anyhow::Context;

// BEFORE:
let resp = client.get(url).send().await?;

// AFTER:
let resp = client.get(url).send().await
    .context(format!("Failed to GET {}", url))?;

// BEFORE:
let output = Command::new(binary).spawn()?;

// AFTER:
let output = Command::new(binary).spawn()
    .context(format!("Failed to spawn binary: {}", binary))?;
```

### 2.5 Retry Logic for Flaky Operations

**Problem:** Network operations can fail transiently.

**Solution:**
```rust
// TEAM-061: Retry helper with exponential backoff
async fn retry_with_backoff<F, Fut, T>(
    operation: F,
    max_attempts: u32,
    operation_name: &str,
) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut attempt = 1;
    let mut delay = Duration::from_millis(100);
    
    loop {
        match operation().await {
            Ok(result) => {
                if attempt > 1 {
                    tracing::info!("✅ {} succeeded on attempt {}", operation_name, attempt);
                }
                return Ok(result);
            }
            Err(e) if attempt >= max_attempts => {
                tracing::error!("❌ {} failed after {} attempts: {}", 
                    operation_name, max_attempts, e);
                return Err(e);
            }
            Err(e) => {
                tracing::warn!("⚠️  {} failed (attempt {}/{}): {}", 
                    operation_name, attempt, max_attempts, e);
                tokio::time::sleep(delay).await;
                delay *= 2; // Exponential backoff
                attempt += 1;
            }
        }
    }
}

// Usage:
let resp = retry_with_backoff(
    || client.get(url).send(),
    3,
    "HTTP GET to worker"
).await?;
```

---

## Priority 3: Diagnostic Improvements

### 3.1 Process Visibility

**Add to all process spawns:**
```rust
// TEAM-061: Log all spawned processes
tracing::info!("🚀 Spawning process: {} (PID will be assigned)", binary_name);

let mut child = Command::new(binary)
    .args(args)
    .spawn()
    .context("Failed to spawn process")?;

if let Some(pid) = child.id() {
    tracing::info!("✅ Process spawned: {} (PID: {})", binary_name, pid);
    
    // Store PID for cleanup
    SPAWNED_PIDS.lock().unwrap().push(pid);
}
```

### 3.2 Hanging Detection

**Add watchdog timer:**
```rust
// TEAM-061: Detect hanging operations
async fn with_hang_detection<F, T>(
    operation: F,
    timeout: Duration,
    operation_name: &str,
) -> Result<T>
where
    F: std::future::Future<Output = T>,
{
    let start = std::time::Instant::now();
    
    // Spawn watchdog
    let watchdog = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        loop {
            interval.tick().await;
            let elapsed = start.elapsed();
            tracing::warn!("⏱️  {} still running after {:?}", operation_name, elapsed);
            
            if elapsed > timeout {
                tracing::error!("🚨 {} HANGING - exceeded timeout!", operation_name);
                break;
            }
        }
    });
    
    let result = tokio::time::timeout(timeout, operation).await;
    watchdog.abort();
    
    result.map_err(|_| anyhow::anyhow!("{} timeout after {:?}", operation_name, timeout))
}
```

### 3.3 Test Progress Logging

**Add to cucumber runner:**
```rust
// TEAM-061: Log test progress
World::cucumber()
    .before(|_feature, _rule, scenario, _world| {
        Box::pin(async move {
            tracing::info!("▶️  Starting scenario: {}", scenario.name);
        })
    })
    .after(|_feature, _rule, scenario, _world| {
        Box::pin(async move {
            tracing::info!("✅ Completed scenario: {}", scenario.name);
        })
    })
    .run(...)
    .await;
```

---

## Priority 4: Specific Hang Fixes

### 4.1 Global Queen-rbee Initialization

**File:** `src/steps/global_queen.rs`

**Changes needed:**
1. Add 30s timeout to port readiness check (line 60-80)
2. Add binary existence check before spawn (line 55)
3. Add better error messages on failure
4. Add PID logging

### 4.2 Mock rbee-hive Server

**File:** `src/mock_rbee_hive.rs`

**Changes needed:**
1. Check port 9200 available before binding
2. Add timeout to `axum::serve()` startup
3. Add graceful shutdown handler
4. Log when server is ready

### 4.3 Worker Spawning

**File:** `src/steps/happy_path.rs` and `src/steps/lifecycle.rs`

**Changes needed:**
1. Add timeout to HTTP POST `/v1/workers/spawn`
2. Add retry logic (3 attempts with backoff)
3. Verify worker process actually started
4. Add timeout to ready callback wait

### 4.4 Mock Worker Binary

**File:** `src/bin/mock-worker.rs`

**Changes needed:**
1. Add timeout to ready callback HTTP POST
2. Add retry logic for callback (3 attempts)
3. Log when server is ready
4. Add graceful shutdown on timeout

---

## Testing Strategy

### Phase 1: Add Timeouts (Day 1-2)

1. **HTTP client timeouts** - All requests timeout after 10s
2. **Process spawn timeouts** - All spawns timeout after 30s
3. **Scenario timeout** - Entire suite times out after 5 minutes

**Verification:**
```bash
# Should complete or timeout within 5 minutes
timeout 360 cargo run --bin bdd-runner
echo "Exit code: $?"  # Should be 0 (success) or 124 (timeout)
```

### Phase 2: Error Handling (Day 2-3)

1. **Port conflict detection** - Clear error if ports in use
2. **Binary checks** - Clear error if binaries missing
3. **Cleanup on failure** - All processes die on test failure
4. **Ctrl+C handling** - Clean shutdown on interrupt

**Verification:**
```bash
# Test port conflict
nc -l 8080 &  # Occupy port
cargo run --bin bdd-runner  # Should fail with clear message
pkill nc

# Test missing binary
mv target/debug/mock-worker /tmp/
cargo run --bin bdd-runner  # Should fail with clear message
mv /tmp/mock-worker target/debug/

# Test Ctrl+C
cargo run --bin bdd-runner &
sleep 5
kill -INT $!  # Should cleanup and exit 130
ps aux | grep -E "queen|worker|hive"  # Should be empty
```

### Phase 3: Diagnostics (Day 3-4)

1. **Process logging** - All PIDs logged
2. **Hang detection** - Warnings every 5s for long operations
3. **Progress logging** - Each scenario start/end logged

**Verification:**
```bash
cargo run --bin bdd-runner 2>&1 | tee test.log

# Check for process logs
grep "PID:" test.log

# Check for hang warnings
grep "still running" test.log

# Check for progress
grep -E "(Starting|Completed) scenario" test.log
```

### Phase 4: Full Suite Run (Day 4-5)

1. Run full suite with timeouts
2. Identify which scenarios still hang
3. Add targeted fixes
4. Achieve 62/62 passing

**Verification:**
```bash
# Full run with timeout
timeout 360 cargo run --bin bdd-runner 2>&1 | tee full_run.log

# Check results
tail -20 full_run.log | grep "scenarios"
# Target: "62 scenarios (62 passed)"
```

---

## Files to Modify

### Critical (Must Fix for Tests to Run)

1. **`src/steps/global_queen.rs`** - Add timeout to queen-rbee startup
2. **`src/mock_rbee_hive.rs`** - Add timeout to server startup
3. **`src/steps/world.rs`** - Add HTTP client factory, cleanup in Drop
4. **`src/bin/mock-worker.rs`** - Add timeout to ready callback

### Important (For Robust Error Handling)

5. **`src/steps/beehive_registry.rs`** - Use timeout HTTP client
6. **`src/steps/happy_path.rs`** - Use timeout HTTP client, add retries
7. **`src/steps/lifecycle.rs`** - Use timeout HTTP client, add retries
8. **`src/main.rs`** or `src/lib.rs` - Add scenario timeout wrapper

### Nice to Have (For Better Diagnostics)

9. **`src/utils/process.rs`** (NEW) - Process management utilities
10. **`src/utils/retry.rs`** (NEW) - Retry logic helper
11. **`src/utils/timeout.rs`** (NEW) - Timeout wrapper utilities

---

## Success Criteria

### Must Have (Blocker Resolution)

✅ Test suite completes within 5 minutes (no infinite hangs)  
✅ Ctrl+C cleanly shuts down all processes  
✅ Clear error messages when ports are in use  
✅ Clear error messages when binaries are missing  
✅ All spawned processes are logged with PIDs  
✅ All spawned processes die when tests complete

### Should Have (Robust Error Handling)

✅ HTTP requests timeout after 10s  
✅ Process spawns timeout after 30s  
✅ Retry logic for flaky network operations  
✅ Rich error context on all failures  
✅ Hang detection warnings every 5s  
✅ Progress logging for each scenario

### Nice to Have (Full Verification)

✅ 62/62 scenarios passing  
✅ Test suite runs in under 2 minutes  
✅ Zero dangling processes after tests  
✅ Comprehensive error handling audit complete

---

## Measurement

### Daily Check-In Questions

1. Can the test suite complete without hanging? (Target: Yes by Day 2)
2. How many scenarios timeout? (Target: 0 by Day 4)
3. Are all processes cleaned up? (Target: Yes by Day 3)
4. What's the test suite runtime? (Target: <5 minutes by Day 4)

### Exit Criteria

**Minimum (Unblock Development):**
- Test suite completes (pass or fail, but doesn't hang)
- All processes cleanup on exit
- Clear error messages

**Target (Production Ready):**
- 62/62 scenarios passing
- <2 minute runtime
- Comprehensive error handling
- Zero dangling processes

---

## Resources

### What TEAM-060 Built

- ✅ Real edge case command execution (7 steps)
- ✅ No fake exit codes
- ✅ Compilation successful
- ❌ Test suite hangs (your job to fix)

### Documentation

- **TEAM-059 work:** `TEAM_059_SUMMARY.md`
- **TEAM-060 work:** `TEAM_060_QUICK_START.md`
- **Original mandate:** `TEAM_059_HANDOFF_REAL_TESTING.md`
- **Dev-bee rules:** `../../.windsurf/rules/dev-bee-rules.md`

### Debugging Commands

```bash
# Check for hanging processes
ps aux | grep -E "queen|worker|hive|bdd-runner"

# Kill all test processes
pkill -9 -f "bdd-runner|mock-worker|queen-rbee"

# Check port usage
lsof -i :8080  # queen-rbee
lsof -i :9200  # mock rbee-hive
lsof -i :8001  # workers

# Watch logs in real-time
cargo run --bin bdd-runner 2>&1 | tee test.log | grep -E "(ERROR|WARN|Starting|Completed)"
```

---

## Anti-Patterns to Avoid

### ❌ DON'T: Ignore timeouts

```rust
// NO! This can hang forever
let resp = client.get(url).send().await?;
```

### ✅ DO: Always use timeouts

```rust
// YES! This will timeout after 10s
let client = reqwest::Client::builder()
    .timeout(Duration::from_secs(10))
    .build()?;
let resp = client.get(url).send().await?;
```

### ❌ DON'T: Spawn processes without tracking

```rust
// NO! Lost process, can't cleanup
Command::new("worker").spawn()?;
```

### ✅ DO: Track all spawned processes

```rust
// YES! Can cleanup later
let child = Command::new("worker").spawn()?;
world.worker_processes.push(child);
```

### ❌ DON'T: Silent failures

```rust
// NO! We don't know what failed
let _ = operation().await;
```

### ✅ DO: Log all failures

```rust
// YES! We can debug failures
if let Err(e) = operation().await {
    tracing::error!("Operation failed: {:?}", e);
    return Err(e);
}
```

---

## Final Instructions

1. **Start with Priority 1** - Add timeouts to HTTP client and process spawns
2. **Test incrementally** - After each change, run tests to verify no hangs
3. **Focus on unblocking** - Getting tests to complete is more important than passing
4. **Document hangs** - If you find a hang, document exactly where and why
5. **Clean up processes** - Always verify no dangling processes after tests

**The goal is simple: Make the test suite complete without hanging.**

Once tests complete reliably, we can work on making them pass. But first, they must finish.

---

**TEAM-060 signing off.**

**Status:** Phase 5 complete, Phase 6 blocked by hangs  
**Handoff:** Add timeouts and robust error handling  
**Timeline:** 4-5 days to unblock and verify  
**Philosophy:** Tests that hang are worse than tests that fail. Make them finish.

**Fix the hangs. Make it robust. Get to 62/62.** 🎯🔥

**We believe in you. Make it bulletproof.** 💪
