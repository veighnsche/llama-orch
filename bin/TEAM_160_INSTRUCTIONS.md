# TEAM-160: Implement Real Integration Tests

**Date:** 2025-10-20  
**Priority:** 🔴 CRITICAL  
**Estimated Time:** 6-10 hours

---

## ⚠️ CRITICAL WARNING: Don't Be Stupid About Orchestration

**READ THIS BEFORE YOU START:**

If your test manually spawns rbee-hive, **you're testing the WRONG thing**.

The whole point of queen-rbee is **ORCHESTRATION**. Queen is supposed to:
1. Accept "add localhost to catalog" command
2. **SPAWN rbee-hive daemon** (via SSH or local process)
3. Wait for first heartbeat
4. Do device detection

If the **test harness** spawns rbee-hive instead of queen, then:
- ❌ Your test passes
- ❌ Production fails because queen doesn't know how to spawn hives
- ❌ You wasted everyone's time testing HTTP endpoints, not orchestration

**Ask yourself:** If the test spawns the hive, how do we know queen can?

**What's the point of an orchestrator that doesn't orchestrate?**

The `spawn_rbee_hive()` function in `integration_steps.rs` is marked with warnings.
It should only exist temporarily until queen has orchestration logic. Once queen
can spawn hives, **DELETE that function** and test that queen does it.

---

## Mission

Implement **REAL integration tests** that verify **QUEEN spawns rbee-hive** and handles the first heartbeat flow.

**What you're testing:**
1. Queen-rbee daemon starts
2. Rbee-hive daemon starts
3. Rbee-hive sends first heartbeat to queen
4. Queen receives heartbeat and triggers device detection
5. Queen makes HTTP call to rbee-hive's /v1/devices
6. Queen stores device capabilities
7. Queen updates hive status to Online

**NO MOCKS** - This tests actual daemon-to-daemon communication.

---

## What TEAM-159 Left You

### ✅ Files Created:
1. **`integration_first_heartbeat.feature`** - 3 test scenarios (ready to use)
2. **`integration_steps.rs`** - Step definitions (skeleton, needs implementation)
3. **`TEAM_159_INTEGRATION_TEST_PLAN.md`** - Detailed implementation plan

### ✅ What Works:
- Unit tests with mocks (9 scenarios passing)
- Heartbeat receiving logic (consolidated in shared crate)
- Device storage logic
- Catalog CRUD operations

### ❌ What's Missing:
- Daemon spawning logic
- Process management
- Health check polling
- Catalog verification
- Cleanup on test failure

---

## Implementation Checklist

### Phase 1: Daemon Spawning (2-3 hours)

#### Task 1.1: Update BddWorld to Store Processes
**File:** `bdd/src/steps/world.rs`

```rust
use std::process::Child;

#[derive(World)]
pub struct BddWorld {
    // ... existing fields ...
    
    // TEAM-160: Process handles for integration tests
    pub queen_process: Option<Child>,
    pub hive_process: Option<Child>,
    pub queen_port: Option<u16>,
    pub hive_port: Option<u16>,
}

// TEAM-160: Cleanup spawned processes
impl Drop for BddWorld {
    fn drop(&mut self) {
        if let Some(mut process) = self.queen_process.take() {
            let _ = process.kill();
            let _ = process.wait();
        }
        if let Some(mut process) = self.hive_process.take() {
            let _ = process.kill();
            let _ = process.wait();
        }
    }
}
```

#### Task 1.2: Build Binaries Before Tests
**File:** `bdd/src/steps/integration_steps.rs`

Add at the top:
```rust
use std::sync::Once;

static BUILD_BINARIES: Once = Once::new();

fn ensure_binaries_built() {
    BUILD_BINARIES.call_once(|| {
        println!("Building binaries for integration tests...");
        
        // Build queen-rbee
        let status = Command::new("cargo")
            .args(&["build", "--bin", "queen-rbee"])
            .status()
            .expect("Failed to build queen-rbee");
        assert!(status.success(), "Failed to build queen-rbee");
        
        // Build rbee-hive
        let status = Command::new("cargo")
            .args(&["build", "--bin", "rbee-hive"])
            .status()
            .expect("Failed to build rbee-hive");
        assert!(status.success(), "Failed to build rbee-hive");
        
        println!("Binaries built successfully");
    });
}
```

#### Task 1.3: Implement spawn_queen_rbee()
**File:** `bdd/src/steps/integration_steps.rs`

```rust
fn spawn_queen_rbee(port: u16, db_path: &str) -> Result<Child, std::io::Error> {
    ensure_binaries_built();
    
    let child = Command::new("target/debug/queen-rbee")
        .arg("--port")
        .arg(port.to_string())
        .arg("--db")
        .arg(db_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    
    Ok(child)
}
```

#### Task 1.4: Implement spawn_rbee_hive()
**File:** `bdd/src/steps/integration_steps.rs`

```rust
fn spawn_rbee_hive(port: u16, queen_url: &str, db_path: &str) -> Result<Child, std::io::Error> {
    ensure_binaries_built();
    
    let child = Command::new("target/debug/rbee-hive")
        .arg("--port")
        .arg(port.to_string())
        .arg("--queen-url")
        .arg(queen_url)
        .arg("--db")
        .arg(db_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    
    Ok(child)
}
```

**NOTE:** Check actual command-line args for queen-rbee and rbee-hive. They might be different!

---

### Phase 2: Health Check Polling (1-2 hours)

#### Task 2.1: Implement wait_for_queen_ready()
**File:** `bdd/src/steps/integration_steps.rs`

```rust
async fn wait_for_queen_ready(port: u16, timeout_secs: u64) -> Result<(), String> {
    let client = reqwest::Client::new();
    let health_url = format!("http://localhost:{}/health", port);
    let max_attempts = (timeout_secs * 2) as usize; // Check every 500ms
    
    for attempt in 0..max_attempts {
        match client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => {
                println!("Queen-rbee ready after {} attempts", attempt + 1);
                return Ok(());
            }
            _ => {
                sleep(Duration::from_millis(500)).await;
            }
        }
    }
    
    Err(format!("Queen-rbee failed to start on port {} after {}s", port, timeout_secs))
}
```

#### Task 2.2: Implement wait_for_hive_ready()
**File:** `bdd/src/steps/integration_steps.rs`

```rust
async fn wait_for_hive_ready(port: u16, timeout_secs: u64) -> Result<(), String> {
    let client = reqwest::Client::new();
    let health_url = format!("http://localhost:{}/health", port);
    let max_attempts = (timeout_secs * 2) as usize;
    
    for attempt in 0..max_attempts {
        match client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => {
                println!("Rbee-hive ready after {} attempts", attempt + 1);
                return Ok(());
            }
            _ => {
                sleep(Duration::from_millis(500)).await;
            }
        }
    }
    
    Err(format!("Rbee-hive failed to start on port {} after {}s", port, timeout_secs))
}
```

---

### Phase 3: Implement Given Steps (1-2 hours)

#### Task 3.1: given_queen_running()
**File:** `bdd/src/steps/integration_steps.rs`

```rust
#[given(expr = "queen-rbee HTTP server is running on port {int}")]
async fn given_queen_running(world: &mut BddWorld, port: u16) {
    let db_path = world.temp_dir.as_ref()
        .expect("No temp dir - did you forget 'Given a temporary directory'?")
        .path()
        .join("queen.db")
        .to_str()
        .expect("Invalid path")
        .to_string();
    
    // Spawn queen-rbee
    let child = spawn_queen_rbee(port, &db_path)
        .expect("Failed to spawn queen-rbee");
    
    // Store process handle
    world.queen_process = Some(child);
    world.queen_port = Some(port);
    
    // Wait for queen to be ready
    wait_for_queen_ready(port, 10).await
        .expect("Queen-rbee failed to start");
    
    println!("✅ Queen-rbee running on port {}", port);
}
```

#### Task 3.2: Implement Other Given Steps
Follow the same pattern for:
- `given_hive_entry_points()` - Add hive to catalog with Unknown status
- `given_rbee_hive_running()` - Spawn hive daemon
- `given_rbee_hive_sent_heartbeat()` - Verify heartbeat in catalog

---

### Phase 4: Implement When Steps (1-2 hours)

#### Task 4.1: when_rbee_hive_starts()
**File:** `bdd/src/steps/integration_steps.rs`

```rust
#[when(expr = "rbee-hive daemon starts on port {int}")]
async fn when_rbee_hive_starts(world: &mut BddWorld, port: u16) {
    let queen_port = world.queen_port.expect("Queen not running");
    let queen_url = format!("http://localhost:{}", queen_port);
    
    let db_path = world.temp_dir.as_ref()
        .expect("No temp dir")
        .path()
        .join("hive.db")
        .to_str()
        .expect("Invalid path")
        .to_string();
    
    // Spawn rbee-hive
    let child = spawn_rbee_hive(port, &queen_url, &db_path)
        .expect("Failed to spawn rbee-hive");
    
    // Store process handle
    world.hive_process = Some(child);
    world.hive_port = Some(port);
    
    // Wait for hive to be ready
    wait_for_hive_ready(port, 10).await
        .expect("Rbee-hive failed to start");
    
    println!("✅ Rbee-hive running on port {}", port);
}
```

#### Task 4.2: Implement Other When Steps
- `when_wait_for_init()` - Already implemented (just sleep)
- `when_wait_for_heartbeat()` - Already implemented
- `when_kill_rbee_hive()` - Kill stored process handle
- `when_query_catalog()` - Query catalog and store result

---

### Phase 5: Implement Then Steps (2-3 hours)

#### Task 5.1: then_queen_receives_heartbeat()
**File:** `bdd/src/steps/integration_steps.rs`

```rust
#[then("queen should receive the heartbeat")]
async fn then_queen_receives_heartbeat(world: &mut BddWorld) {
    let catalog_path = world.temp_dir.as_ref()
        .expect("No temp dir")
        .path()
        .join("queen.db");
    
    let catalog = HiveCatalog::new(&catalog_path).await
        .expect("Failed to open catalog");
    
    let hive = catalog.get_hive("localhost").await
        .expect("Failed to query catalog")
        .expect("Hive not found");
    
    assert!(hive.last_heartbeat_ms.is_some(), 
        "Hive should have last_heartbeat timestamp");
    
    println!("✅ Queen received heartbeat");
}
```

#### Task 5.2: then_queen_stores_capabilities()
```rust
#[then("queen should store the device capabilities in the catalog")]
async fn then_queen_stores_capabilities(world: &mut BddWorld) {
    let catalog_path = world.temp_dir.as_ref()
        .expect("No temp dir")
        .path()
        .join("queen.db");
    
    let catalog = HiveCatalog::new(&catalog_path).await
        .expect("Failed to open catalog");
    
    let hive = catalog.get_hive("localhost").await
        .expect("Failed to query catalog")
        .expect("Hive not found");
    
    assert!(hive.devices.is_some(), 
        "Hive should have device capabilities stored");
    
    let devices = hive.devices.unwrap();
    assert!(devices.cpu.is_some(), "Should have CPU info");
    
    println!("✅ Device capabilities stored: {} cores, {} GPUs", 
        devices.cpu.unwrap().cores, 
        devices.gpus.len());
}
```

#### Task 5.3: then_hive_has_status()
```rust
#[then(expr = "the hive should have status {string}")]
async fn then_hive_has_status(world: &mut BddWorld, expected_status: String) {
    let catalog_path = world.temp_dir.as_ref()
        .expect("No temp dir")
        .path()
        .join("queen.db");
    
    let catalog = HiveCatalog::new(&catalog_path).await
        .expect("Failed to open catalog");
    
    let hive = catalog.get_hive("localhost").await
        .expect("Failed to query catalog")
        .expect("Hive not found");
    
    let expected = match expected_status.as_str() {
        "Online" => HiveStatus::Online,
        "Offline" => HiveStatus::Offline,
        "Unknown" => HiveStatus::Unknown,
        _ => panic!("Invalid status: {}", expected_status),
    };
    
    assert_eq!(hive.status, expected, 
        "Expected status {:?}, got {:?}", expected, hive.status);
    
    println!("✅ Hive status is {:?}", hive.status);
}
```

#### Task 5.4: Implement Remaining Then Steps
- `then_rbee_hive_sends_heartbeat()` - Check catalog updated
- `then_queen_triggers_detection()` - Check narration or logs
- `then_rbee_hive_responds()` - Verify /v1/devices endpoint works
- `then_queen_emits_narration()` - Capture stdout/stderr or query narration DB
- `then_hive_has_capabilities()` - Check devices field not null
- `then_hive_has_recent_heartbeat()` - Check timestamp within 30s
- `then_queen_does_not_trigger_detection()` - Verify no "Checking capabilities" narration
- `then_timestamp_updated()` - Compare old vs new timestamp
- `then_queen_detects_missed()` - Requires heartbeat monitoring (may need to implement)
- `then_queen_updates_status()` - Check status changed
- `then_queen_emits_offline_narration()` - Check narration

---

## Testing Your Implementation

### Run Integration Tests
```bash
cd bin/10_queen_rbee/bdd
cargo run --bin bdd-runner -- tests/features/integration_first_heartbeat.feature
```

### Expected Output
```
Feature: Integration Test - First Heartbeat from Real rbee-hive
  Scenario: Queen receives first heartbeat from spawned rbee-hive
    Building binaries for integration tests...
    Binaries built successfully
    ✔  Given a temporary directory for test databases
    ✔  And queen-rbee is configured to use the test database
    ✅ Queen-rbee running on port 18500
    ✔  And queen-rbee HTTP server is running on port 18500
    ✔  And the hive catalog contains a hive "localhost" with status "Unknown"
    ✔  And the hive entry points to "localhost:18600"
    ✅ Rbee-hive running on port 18600
    ✔  When rbee-hive daemon starts on port 18600
    ✔  And rbee-hive is configured to send heartbeats to "http://localhost:18500"
    ✔  And we wait 2 seconds for rbee-hive to initialize
    ✅ Queen received heartbeat
    ✔  Then rbee-hive should send its first heartbeat to queen
    ✔  And queen should receive the heartbeat
    ✔  And queen should trigger device detection to "http://localhost:18600/v1/devices"
    ✔  And rbee-hive should respond with real device information
    ✅ Device capabilities stored: 8 cores, 2 GPUs
    ✔  And queen should store the device capabilities in the catalog
    ✅ Hive status is Online
    ✔  And queen should update hive status to "Online"
    ✔  And queen should emit narration "First heartbeat from localhost"
    ✔  And queen should emit narration "Checking capabilities"
```

---

## Common Issues & Solutions

### Issue 1: Binaries Not Found
**Error:** `No such file or directory (os error 2)`

**Solution:** Make sure binaries are built:
```bash
cargo build --bin queen-rbee
cargo build --bin rbee-hive
```

### Issue 2: Port Already in Use
**Error:** `Address already in use (os error 48)`

**Solution:** Use different ports (18500, 18600) instead of production ports (8500, 8600)

### Issue 3: Process Not Cleaned Up
**Error:** Tests hang or fail on second run

**Solution:** Verify `Drop` implementation in `BddWorld` kills processes

### Issue 4: Timeout Waiting for Daemon
**Error:** `Queen-rbee failed to start on port 18500 after 10s`

**Solution:** 
- Check if binary actually starts: `target/debug/queen-rbee --help`
- Check stdout/stderr from spawned process
- Increase timeout from 10s to 30s

### Issue 5: Command-Line Args Wrong
**Error:** `error: unexpected argument '--db' found`

**Solution:** Check actual args by running:
```bash
target/debug/queen-rbee --help
target/debug/rbee-hive --help
```

Update spawn functions with correct args.

---

## Verification Checklist

Before marking complete, verify:

- [ ] All 3 integration test scenarios pass
- [ ] Processes are cleaned up after tests (no zombie processes)
- [ ] Tests can run multiple times without conflicts
- [ ] Catalog is queried and verified correctly
- [ ] Device capabilities are stored
- [ ] Hive status changes from Unknown → Online
- [ ] Narration events are captured (or documented as TODO)
- [ ] Tests run in CI (add to GitHub Actions)

---

## Deliverables

1. **Fully implemented `integration_steps.rs`** - All step definitions working
2. **Updated `world.rs`** - Process handles and cleanup
3. **Test run output** - Screenshot or log showing all tests passing
4. **Documentation** - Any gotchas or issues you encountered
5. **CI integration** - Add integration tests to `.github/workflows/`

---

## Time Estimates

| Phase | Task | Time |
|-------|------|------|
| 1 | Daemon spawning | 2-3h |
| 2 | Health check polling | 1-2h |
| 3 | Given steps | 1-2h |
| 4 | When steps | 1-2h |
| 5 | Then steps | 2-3h |
| **Total** | | **7-12h** |

---

## Success Criteria

✅ **You're done when:**
1. All 3 integration test scenarios pass
2. Real queen-rbee and rbee-hive daemons communicate
3. First heartbeat triggers device detection
4. Device capabilities are stored in catalog
5. Hive status changes to Online
6. Tests clean up processes properly

---

**TEAM-160: This is REAL integration testing. No mocks. Actual daemons. Good luck! 🚀**
