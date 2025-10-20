# TEAM-160 → TEAM-161 HANDOFF

**Date:** 2025-10-20  
**From:** TEAM-160 (E2E Testing Framework)  
**To:** TEAM-161 (E2E Implementation & Cleanup)  
**Mission:** Make all E2E tests pass and eliminate ALL warnings

---

## 🎯 Your Mission

**Make everything pristine:**
1. Fix all compilation errors
2. Implement missing functionality
3. Make all 3 E2E tests pass
4. Eliminate ALL warnings (zero tolerance)

---

## 📋 Priority Checklist

### Priority 1: Fix Queen-Rbee Compilation (CRITICAL)

**Location:** `bin/10_queen_rbee/`

**4 compilation errors blocking everything:**

#### Error 1: Missing async-trait
```
error[E0432]: unresolved import `async_trait`
 --> bin/10_queen_rbee/src/http/device_detector.rs:8:5
```

**Fix:**
```toml
# Add to bin/10_queen_rbee/Cargo.toml
[dependencies]
async-trait = "0.1"
```

#### Error 2: Lifetime mismatch
```
error[E0195]: lifetime parameters or bounds on method `detect_devices` do not match the trait declaration
  --> bin/10_queen_rbee/src/http/device_detector.rs:35:28
```

**Fix:** Check the trait definition in the heartbeat crate and match the lifetime signature.

#### Error 3: Type mismatch
```
error[E0308]: mismatched types
  --> bin/10_queen_rbee/src/http/heartbeat.rs:37:5
   expected `rbee_heartbeat::HeartbeatAcknowledgement`
   found `rbee_heartbeat::queen_receiver::HeartbeatAcknowledgement`
```

**Fix:** Use the correct type from `rbee_heartbeat` crate. Check which type is exported.

#### Error 4: Missing field
```
error[E0063]: missing field `device_detector` in initializer of `HeartbeatState`
   --> bin/10_queen_rbee/src/main.rs:119:27
```

**Fix:**
```rust
// bin/10_queen_rbee/src/main.rs:119
let heartbeat_state = http::heartbeat::HeartbeatState { 
    hive_catalog: hive_catalog.clone(),
    device_detector: Arc::new(/* create device detector */)
};
```

**Verification:**
```bash
cargo check --bin queen-rbee
# Should compile with 0 errors
```

---

### Priority 2: Implement Queen Spawning Logic

**Location:** `bin/10_queen_rbee/src/http/add_hive.rs`

**Current state:** Adds hive to catalog but doesn't spawn process

**What you need to implement:**

```rust
pub async fn handle_add_hive(
    State(catalog): State<AddHiveState>,
    Json(payload): Json<AddHiveRequest>,
) -> Result<(StatusCode, Json<AddHiveResponse>), (StatusCode, String)> {
    println!("👑 Adding hive {} to catalog", payload.host);

    // Step 1: Add to hive catalog (✅ DONE)
    let now_ms = chrono::Utc::now().timestamp_millis();
    let hive = HiveRecord { /* ... */ };
    catalog.add_hive(hive).await?;
    
    // Step 2: If localhost, spawn rbee-hive process (TODO: IMPLEMENT THIS)
    if payload.host == "localhost" {
        println!("👑 Spawning rbee-hive on localhost:{}...", payload.port);
        
        let child = Command::new("target/debug/rbee-hive")
            .arg("--port")
            .arg(payload.port.to_string())
            .arg("--queen-url")
            .arg("http://localhost:8500")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        
        // TODO: Store child process handle somewhere
        // TODO: Monitor process lifecycle
    }
    
    Ok(Json(AddHiveResponse { /* ... */ }))
}
```

**Verification:**
```bash
cargo run --bin rbee-keeper -- hive start
# Should spawn both queen AND hive
ps aux | grep rbee
# Should show both queen-rbee and rbee-hive processes
```

---

### Priority 3: Implement Hive Heartbeat Sender

**Location:** `bin/20_rbee_hive/src/` (create new file)

**What you need to implement:**

```rust
// bin/20_rbee_hive/src/heartbeat_sender.rs
use std::time::Duration;
use tokio::time::sleep;

pub async fn start_heartbeat_task(queen_url: String, hive_id: String) {
    let client = reqwest::Client::new();
    let heartbeat_url = format!("{}/heartbeat", queen_url);
    
    loop {
        let payload = serde_json::json!({
            "hive_id": hive_id,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "workers": [] // Empty initially
        });
        
        match client.post(&heartbeat_url).json(&payload).send().await {
            Ok(response) if response.status().is_success() => {
                println!("🐝 Heartbeat sent to queen");
            }
            Ok(response) => {
                eprintln!("⚠️  Heartbeat failed: {}", response.status());
            }
            Err(e) => {
                eprintln!("⚠️  Heartbeat error: {}", e);
            }
        }
        
        sleep(Duration::from_secs(15)).await;
    }
}
```

**Wire it up in main.rs:**
```rust
// bin/20_rbee_hive/src/main.rs
tokio::spawn(heartbeat_sender::start_heartbeat_task(
    args.queen_url.clone(),
    "localhost".to_string(),
));
```

**Verification:**
```bash
cargo run --bin rbee-keeper -- hive start
# Wait 15 seconds
# Check queen logs - should see heartbeat received
```

---

### Priority 4: Implement Device Detection Trigger

**Location:** `bin/10_queen_rbee/src/http/heartbeat.rs`

**Current state:** Receives heartbeat but doesn't trigger device detection

**What you need to implement:**

```rust
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<Json<HeartbeatAcknowledgement>, (StatusCode, String)> {
    // Check if this is first heartbeat (devices unknown)
    let hive = state.hive_catalog.get_hive(&payload.hive_id).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    if hive.devices.is_none() {
        println!("👑 First heartbeat from {} - checking capabilities...", payload.hive_id);
        
        // TODO: IMPLEMENT THIS
        // Call GET /v1/devices on the hive
        let hive_url = format!("http://{}:{}", hive.host, hive.port);
        let devices_url = format!("{}/v1/devices", hive_url);
        
        let client = reqwest::Client::new();
        let response = client.get(&devices_url).send().await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        
        let devices: DeviceCapabilities = response.json().await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        
        // Store in catalog
        state.hive_catalog.update_devices(&payload.hive_id, devices).await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        
        println!("👑 Hive {} capabilities stored", payload.hive_id);
    }
    
    // Update last_heartbeat timestamp
    let now_ms = chrono::Utc::now().timestamp_millis();
    state.hive_catalog.update_heartbeat(&payload.hive_id, now_ms).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(HeartbeatAcknowledgement { status: "ok" }))
}
```

**Verification:**
```bash
cargo run --bin rbee-keeper -- hive start
# Wait for first heartbeat
# Check queen logs - should see "checking capabilities"
# Check catalog - hive should have devices stored
```

---

### Priority 5: Implement Cascading Shutdown

**Location:** `bin/10_queen_rbee/src/http/shutdown.rs`

**Current state:** Shutdown endpoint exists but doesn't cascade

**What you need to implement:**

```rust
pub async fn handle_shutdown(State(state): State<ShutdownState>) -> StatusCode {
    println!("👑 Received shutdown signal");
    
    // Step 1: Get all hives from catalog
    let hives = state.hive_catalog.list_hives().await.unwrap_or_default();
    
    println!("👑 Shutting down {} hives...", hives.len());
    
    // Step 2: Send shutdown to each hive
    let client = reqwest::Client::new();
    for hive in hives {
        let shutdown_url = format!("http://{}:{}/shutdown", hive.host, hive.port);
        match client.post(&shutdown_url).send().await {
            Ok(_) => println!("👑 Shutdown signal sent to {}", hive.id),
            Err(e) => eprintln!("⚠️  Failed to shutdown {}: {}", hive.id, e),
        }
    }
    
    // Step 3: Wait for hives to shutdown
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    
    println!("👑 Queen shutting down");
    
    // Step 4: Exit
    std::process::exit(0);
}
```

**Verification:**
```bash
cargo run --bin rbee-keeper -- hive start
# Wait for hive to be online
cargo run --bin rbee-keeper -- queen stop
# Should see both queen and hive shutdown
ps aux | grep rbee
# Should show NO rbee processes
```

---

### Priority 6: Eliminate ALL Warnings

**Current warnings:**

#### xtask warnings:
```
warning: unused variable: `pattern`
warning: function `start_queen` is never used
warning: function `wait_for_queen` is never used
warning: function `start_hive` is never used
warning: function `wait_for_hive` is never used
warning: function `kill_process` is never used
warning: function `extract_failures` is never used
warning: function `extract_failure_context` is never used
warning: field `really_quiet` is never read
warning: struct `FailureInfo` is never constructed
```

**Fix:** Either use these functions or add `#[allow(dead_code)]` if they're for future use.

#### rbee-keeper warning:
```
warning: unreachable pattern
   --> bin/00_rbee_keeper/src/main.rs:525:9
```

**Fix:** Already fixed by TEAM-160 (removed duplicate Hive command)

#### rbee-types warnings:
```
warning: missing documentation for a struct field (39 warnings)
```

**Fix:** Add doc comments to all public fields in `bin/99_shared_crates/rbee-types/src/`.

**Verification:**
```bash
cargo check 2>&1 | grep warning
# Should output NOTHING
```

---

## 🧪 Testing Checklist

Run each test and verify it passes:

### Test 1: Queen Lifecycle
```bash
cargo xtask e2e:queen
```

**Expected output:**
```
✅ E2E Test PASSED: Queen Lifecycle
```

**If it fails:**
- Check queen compilation errors are fixed
- Check queen spawns correctly
- Check shutdown endpoint works

---

### Test 2: Hive Lifecycle
```bash
cargo xtask e2e:hive
```

**Expected output:**
```
✅ E2E Test PASSED: Hive Lifecycle
```

**If it fails:**
- Check queen spawns hive correctly
- Check hive sends heartbeats
- Check hive shutdown works
- Check queen stays running after hive stops

---

### Test 3: Cascade Shutdown
```bash
cargo xtask e2e:cascade
```

**Expected output:**
```
✅ E2E Test PASSED: Cascade Shutdown
   Queen stopped → Hive stopped automatically
```

**If it fails:**
- Check queen shutdown cascades to hives
- Check both processes actually terminate
- Check no zombie processes left

---

## 📊 Success Criteria

**You are DONE when:**

✅ All 3 E2E tests pass  
✅ Zero compilation errors  
✅ Zero warnings  
✅ All processes start correctly  
✅ All processes stop correctly  
✅ Cascading shutdown works  
✅ No zombie processes  
✅ Clean `cargo check` output  

---

## 🔍 Verification Commands

Run these in order:

```bash
# 1. Check compilation
cargo check 2>&1 | grep -E "(error|warning)"
# Expected: NO OUTPUT

# 2. Build all binaries
cargo build --bin queen-rbee
cargo build --bin rbee-keeper
cargo build --bin rbee-hive
# Expected: All succeed

# 3. Run E2E tests
cargo xtask e2e:queen
cargo xtask e2e:hive
cargo xtask e2e:cascade
# Expected: All pass

# 4. Check for zombie processes
ps aux | grep rbee
# Expected: NO rbee processes running
```

---

## 📁 Files You'll Touch

**Must modify:**
1. `bin/10_queen_rbee/Cargo.toml` - Add async-trait
2. `bin/10_queen_rbee/src/http/device_detector.rs` - Fix lifetime
3. `bin/10_queen_rbee/src/http/heartbeat.rs` - Fix type, add device detection
4. `bin/10_queen_rbee/src/main.rs` - Add device_detector field
5. `bin/10_queen_rbee/src/http/add_hive.rs` - Add process spawning
6. `bin/10_queen_rbee/src/http/shutdown.rs` - Add cascading shutdown
7. `bin/20_rbee_hive/src/heartbeat_sender.rs` - Create heartbeat sender
8. `bin/20_rbee_hive/src/main.rs` - Wire up heartbeat
9. `bin/99_shared_crates/rbee-types/src/*.rs` - Add doc comments
10. `xtask/src/e2e/helpers.rs` - Fix dead code warnings
11. `xtask/src/tasks/bdd/*.rs` - Fix unused variable warnings

**May need to create:**
- `bin/10_queen_rbee/src/process_manager.rs` - Store/manage child processes
- `bin/20_rbee_hive/src/heartbeat_sender.rs` - Heartbeat logic

---

## 🚨 Common Pitfalls

### 1. Process Management
**Problem:** Spawned processes become zombies  
**Solution:** Store process handles, implement proper cleanup

### 2. Heartbeat Timing
**Problem:** Test times out waiting for heartbeat  
**Solution:** Reduce heartbeat interval to 5s for testing

### 3. Port Conflicts
**Problem:** "Address already in use"  
**Solution:** Kill all rbee processes before testing: `pkill -9 rbee`

### 4. Shutdown Race Conditions
**Problem:** Hive doesn't receive shutdown signal  
**Solution:** Add retry logic, increase wait time

### 5. Catalog State
**Problem:** Old hive entries in catalog  
**Solution:** Delete `queen-hive-catalog.db` between tests

---

## 📚 Reference Documentation

**TEAM-160 created these docs:**
1. `xtask/src/e2e/README.md` - E2E test documentation
2. `bin/TEAM_160_E2E_COMPLETE.md` - Complete implementation summary
3. `bin/TEAM_160_HAPPY_FLOW_BREAKDOWN.md` - Step-by-step flow
4. `bin/a_human_wrote_this.md` - Original happy flow specification

**Read these FIRST before starting.**

---

## 🎯 Your TODO List

Copy this to your working document and check off as you go:

```
Priority 1: Fix Compilation
[ ] Add async-trait to queen-rbee Cargo.toml
[ ] Fix lifetime mismatch in device_detector.rs
[ ] Fix type mismatch in heartbeat.rs
[ ] Add device_detector field to HeartbeatState
[ ] Verify: cargo check --bin queen-rbee (0 errors)

Priority 2: Implement Queen Spawning
[ ] Add process spawning to handle_add_hive
[ ] Store child process handle
[ ] Test: rbee hive start spawns both processes
[ ] Verify: ps aux | grep rbee shows 2 processes

Priority 3: Implement Hive Heartbeat
[ ] Create heartbeat_sender.rs in rbee-hive
[ ] Wire up in main.rs
[ ] Test: Wait 15s, check queen receives heartbeat
[ ] Verify: Queen logs show "Heartbeat received"

Priority 4: Implement Device Detection
[ ] Add device detection trigger in handle_heartbeat
[ ] Call GET /v1/devices on first heartbeat
[ ] Store devices in catalog
[ ] Test: Check catalog has device info
[ ] Verify: Hive status changes to "Online"

Priority 5: Implement Cascading Shutdown
[ ] Update handle_shutdown to cascade
[ ] Send shutdown to all hives
[ ] Test: rbee queen stop kills both processes
[ ] Verify: ps aux | grep rbee shows 0 processes

Priority 6: Eliminate Warnings
[ ] Fix unused variable warnings in xtask
[ ] Fix dead code warnings in helpers.rs
[ ] Add doc comments to rbee-types
[ ] Verify: cargo check (0 warnings)

Final Verification
[ ] cargo xtask e2e:queen PASSES
[ ] cargo xtask e2e:hive PASSES
[ ] cargo xtask e2e:cascade PASSES
[ ] cargo check (0 errors, 0 warnings)
[ ] No zombie processes after tests
```

---

## 💬 Questions?

If you get stuck:

1. **Read the docs** - TEAM-160 left extensive documentation
2. **Check the happy flow** - `bin/a_human_wrote_this.md` has the spec
3. **Look at existing code** - Similar patterns exist in other binaries
4. **Test incrementally** - Don't try to fix everything at once

---

## 🎉 Success Looks Like

```bash
$ cargo check
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.50s

$ cargo xtask e2e:queen
✅ E2E Test PASSED: Queen Lifecycle

$ cargo xtask e2e:hive
✅ E2E Test PASSED: Hive Lifecycle

$ cargo xtask e2e:cascade
✅ E2E Test PASSED: Cascade Shutdown
   Queen stopped → Hive stopped automatically

$ ps aux | grep rbee
(no output)
```

**When you see this, you're DONE. Ship it! 🚀**

---

**TEAM-160 → TEAM-161: E2E framework is ready. Make it pristine. Good luck! 🎯**
