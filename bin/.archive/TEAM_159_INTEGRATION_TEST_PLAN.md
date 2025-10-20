# TEAM-159: Real Integration Test Plan

**Date:** 2025-10-20  
**Status:** 🚧 CREATED - Needs Implementation

---

## The Problem

You asked: **"We tested queen-rbee starting rbee-hive and waiting for first heartbeat, right?"**

**Answer:** ❌ **NO** - We only tested:
- Mock-based unit tests (90% real code, 10% mocked HTTP)
- Catalog CRUD operations
- Heartbeat receiving logic

**We did NOT test:**
- Queen spawning rbee-hive daemon
- Real daemon-to-daemon communication
- Actual first heartbeat from a real rbee-hive
- Real device detection HTTP calls

---

## What Previous Teams Claimed

Previous teams created `happy_flow_part1.feature` with scenarios like:
- "Queen adds localhost to hive catalog and starts rbee-hive"
- "Rbee-hive automatically sends heartbeat to queen"
- "Queen triggers device detection on first heartbeat"

**But ALL these scenarios are SKIPPED** because the background step `Given all services are stopped` was never implemented.

**Result:** 9 skipped scenarios, 0 real integration tests.

---

## What We Need: REAL Integration Tests

### Test 1: First Heartbeat Flow (End-to-End)

```gherkin
Scenario: Queen receives first heartbeat from spawned rbee-hive
  Given queen-rbee HTTP server is running on port 18500
  And the hive catalog contains a hive "localhost" with status "Unknown"
  And the hive entry points to "localhost:18600"
  
  When rbee-hive daemon starts on port 18600
  And rbee-hive is configured to send heartbeats to "http://localhost:18500"
  And we wait 2 seconds for rbee-hive to initialize
  
  Then rbee-hive should send its first heartbeat to queen
  And queen should receive the heartbeat
  And queen should trigger device detection to "http://localhost:18600/v1/devices"
  And rbee-hive should respond with real device information
  And queen should store the device capabilities in the catalog
  And queen should update hive status to "Online"
```

**What this tests:**
- ✅ Real queen-rbee daemon running
- ✅ Real rbee-hive daemon running
- ✅ Real HTTP communication between them
- ✅ Real device detection HTTP call
- ✅ Real database operations
- ✅ Real narration events

**No mocks** - This is actual integration testing.

---

## Implementation Status

### ✅ Created Files:
1. `integration_first_heartbeat.feature` - Feature file with 3 scenarios
2. `integration_steps.rs` - Step definitions (skeleton)

### 🚧 What Needs Implementation:

#### 1. Build Binaries First
```bash
cargo build --bin queen-rbee
cargo build --bin rbee-hive
```

#### 2. Process Management
```rust
// Store process handles in BddWorld
pub struct BddWorld {
    pub queen_process: Option<Child>,
    pub hive_process: Option<Child>,
    // ...
}
```

#### 3. Daemon Spawning
```rust
fn spawn_queen_rbee(port: u16, db_path: &str) -> Result<Child, std::io::Error> {
    Command::new("target/debug/queen-rbee")
        .arg("--port").arg(port.to_string())
        .arg("--db").arg(db_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
}
```

#### 4. Health Check Polling
```rust
// Wait for daemon to be ready
for _ in 0..10 {
    if let Ok(response) = client.get(&health_url).send().await {
        if response.status().is_success() {
            return Ok(());
        }
    }
    sleep(Duration::from_millis(500)).await;
}
```

#### 5. Catalog Verification
```rust
// Query catalog and verify state
let catalog = HiveCatalog::new(&db_path).await?;
let hive = catalog.get_hive("localhost").await?;
assert_eq!(hive.status, HiveStatus::Online);
assert!(hive.devices.is_some());
```

#### 6. Narration Capture
```rust
// Capture narration events from stdout/stderr
// Or query narration database if persisted
```

#### 7. Cleanup
```rust
impl Drop for BddWorld {
    fn drop(&mut self) {
        // Kill spawned processes
        if let Some(mut process) = self.queen_process.take() {
            let _ = process.kill();
        }
        if let Some(mut process) = self.hive_process.take() {
            let _ = process.kill();
        }
    }
}
```

---

## Test Scenarios

### Scenario 1: First Heartbeat (Primary Flow)
**Tests:** Queen receives first heartbeat from real rbee-hive, triggers device detection, stores capabilities

**Expected Duration:** ~5 seconds
- 2s for daemons to start
- 1s for first heartbeat
- 2s for device detection

### Scenario 2: Periodic Heartbeats
**Tests:** Subsequent heartbeats don't trigger device detection

**Expected Duration:** ~20 seconds
- Wait for next heartbeat cycle (15s default)

### Scenario 3: Hive Goes Offline
**Tests:** Queen detects when rbee-hive stops sending heartbeats

**Expected Duration:** ~65 seconds
- Kill daemon
- Wait for timeout (60s)

---

## Comparison: Unit Tests vs Integration Tests

### What We Have (Unit Tests with Mocks):
```
┌─────────────────────────────────────┐
│ Test Code                           │
│   ├─> Mock HTTP Server             │ 🎭 MOCK
│   ├─> Real Catalog (SQLite)        │ ✅ REAL
│   ├─> Real Heartbeat Handler       │ ✅ REAL
│   └─> Real Type Conversions        │ ✅ REAL
└─────────────────────────────────────┘
```
**Good for:** Testing business logic, fast feedback  
**Missing:** Actual daemon communication

### What We Need (Integration Tests):
```
┌─────────────────────────────────────┐
│ Real queen-rbee daemon              │ ✅ REAL
│   ├─> HTTP Server (Axum)           │ ✅ REAL
│   ├─> SQLite Database              │ ✅ REAL
│   └─> Heartbeat Handler            │ ✅ REAL
└─────────────────────────────────────┘
         ↓ HTTP POST /heartbeat
┌─────────────────────────────────────┐
│ Real rbee-hive daemon               │ ✅ REAL
│   ├─> HTTP Server (Axum)           │ ✅ REAL
│   ├─> Heartbeat Sender             │ ✅ REAL
│   └─> Device Detection             │ ✅ REAL
└─────────────────────────────────────┘
```
**Good for:** Verifying actual system behavior, catching integration bugs  
**Slower:** ~5-65 seconds per test

---

## Why This Matters

### Integration tests catch:
1. **Port conflicts** - Does queen actually bind to the port?
2. **Network issues** - Can hive reach queen over HTTP?
3. **Timing issues** - Does heartbeat arrive before timeout?
4. **Configuration issues** - Are command-line args parsed correctly?
5. **Process lifecycle** - Do daemons start/stop cleanly?
6. **Real device detection** - Does the HTTP call actually work?

### Unit tests with mocks miss:
- Daemon startup failures
- HTTP routing issues
- Port binding problems
- Real timing behavior
- Process management bugs

---

## Recommendation

**Implement the integration tests in this order:**

### Phase 1: Basic Daemon Spawning (1-2 hours)
- [ ] Build binaries
- [ ] Spawn queen-rbee
- [ ] Verify health endpoint responds
- [ ] Clean shutdown

### Phase 2: Hive Communication (2-3 hours)
- [ ] Spawn rbee-hive
- [ ] Configure hive to send to queen
- [ ] Verify heartbeat arrives
- [ ] Check catalog updated

### Phase 3: Device Detection (1-2 hours)
- [ ] Verify queen calls hive's /v1/devices
- [ ] Verify hive responds with real data
- [ ] Verify queen stores capabilities
- [ ] Verify status changes to Online

### Phase 4: Advanced Scenarios (2-3 hours)
- [ ] Periodic heartbeats
- [ ] Hive offline detection
- [ ] Multiple hives
- [ ] Restart scenarios

**Total Estimate:** 6-10 hours for complete integration test suite

---

## Current Test Coverage

| Type | What | Coverage | Status |
|------|------|----------|--------|
| **Unit Tests** | Heartbeat logic | 90% | ✅ Done |
| **Unit Tests** | Catalog CRUD | 100% | ✅ Done |
| **Unit Tests** | Device storage | 100% | ✅ Done |
| **Integration Tests** | Daemon spawning | 0% | ❌ TODO |
| **Integration Tests** | HTTP communication | 0% | ❌ TODO |
| **Integration Tests** | First heartbeat flow | 0% | ❌ TODO |
| **Integration Tests** | Device detection | 0% | ❌ TODO |

---

## Bottom Line

**You were right to ask.** We have good unit tests, but **ZERO real integration tests**.

The integration test skeleton is now created. Next step is to implement the daemon spawning and verification logic.

**This is what you asked for** - testing the actual queen → hive → heartbeat → device detection flow with real daemons.

---

**TEAM-159: Integration test plan created. Ready for implementation.**
