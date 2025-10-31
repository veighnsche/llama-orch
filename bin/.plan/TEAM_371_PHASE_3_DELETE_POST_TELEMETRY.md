# TEAM-371 Phase 3: DELETE Old POST Telemetry Logic + Registry Consolidation

**Status:** 📋 READY FOR IMPLEMENTATION  
**Team:** TEAM-374 (or whoever implements this)  
**Depends on:** Phase 2 complete (Queen subscribes to SSE)

---

## Mission

1. **DELETE** old POST-based continuous telemetry logic now that SSE is working
2. **CONSOLIDATE** registries: Rename `hive-registry` → `telemetry-registry`, DELETE `worker-registry`

**RULE ZERO:** Break cleanly. Let compiler find all call sites. No backwards compatibility.

---

## Part A: Registry Consolidation (Do First)

### Step 1: Rename Crate Directory

```bash
mv bin/15_queen_rbee_crates/hive-registry \
   bin/15_queen_rbee_crates/telemetry-registry
```

### Step 2: Update Cargo.toml

**File:** `bin/15_queen_rbee_crates/telemetry-registry/Cargo.toml`

```toml
[package]
name = "queen-rbee-telemetry-registry"
```

### Step 3: Rename Struct in Code

**File:** `bin/15_queen_rbee_crates/telemetry-registry/src/registry.rs`

```rust
// Rename HiveRegistry → TelemetryRegistry
pub struct TelemetryRegistry {
    hives: HeartbeatRegistry<HiveHeartbeat>,
    workers: RwLock<HashMap<String, Vec<ProcessStats>>>,
}
```

### Step 4: Update lib.rs

**File:** `bin/15_queen_rbee_crates/telemetry-registry/src/lib.rs`

```rust
pub use registry::TelemetryRegistry;
```

### Step 5: Delete worker-registry

```bash
rm -rf bin/15_queen_rbee_crates/worker-registry
```

### Step 6: Update Queen Dependencies

**File:** `bin/10_queen_rbee/Cargo.toml`

```toml
# DELETE:
# queen-rbee-hive-registry = { path = "../15_queen_rbee_crates/hive-registry" }
# queen-rbee-worker-registry = { path = "../15_queen_rbee_crates/worker-registry" }

# ADD:
queen-rbee-telemetry-registry = { path = "../15_queen_rbee_crates/telemetry-registry" }
```

### Step 7: Update All Imports in Queen

**Files to update:**
- `bin/10_queen_rbee/src/main.rs`
- `bin/10_queen_rbee/src/http/heartbeat.rs`
- `bin/10_queen_rbee/src/http/heartbeat_stream.rs`
- `bin/10_queen_rbee/src/hive_subscriber.rs`
- `bin/10_queen_rbee/src/job_router.rs`

```rust
// BEFORE:
use queen_rbee_hive_registry::HiveRegistry;
use queen_rbee_worker_registry::WorkerRegistry;

// AFTER:
use queen_rbee_telemetry_registry::TelemetryRegistry;
```

### Step 8: Fix Field Names

**File:** `bin/10_queen_rbee/src/job_router.rs`

```rust
// BEFORE (confusing):
pub struct JobState {
    pub hive_registry: Arc<queen_rbee_worker_registry::WorkerRegistry>,
}

// AFTER (clear):
pub struct JobState {
    pub telemetry: Arc<TelemetryRegistry>,
}
```

---

## Part B: DELETE Old POST Telemetry Logic

---

## What Gets DELETED

### 1. Hive: Old Continuous POST Logic

**DELETE ENTIRE FUNCTION:** `bin/20_rbee_hive/src/heartbeat.rs:196-252`

```rust
// TEAM-374: DELETE THIS - replaced by SSE broadcaster
async fn start_normal_telemetry_task(hive_info: HiveInfo, queen_url: String) {
    // ... DELETE ALL 56 LINES ...
}
```

**DELETE ENTIRE FUNCTION:** `bin/20_rbee_hive/src/heartbeat.rs:54-92`

```rust
// TEAM-374: DELETE THIS - replaced by ready callback
pub async fn send_heartbeat_to_queen(
    hive_info: &HiveInfo, 
    queen_url: &str,
    capabilities: Option<Vec<HiveDevice>>,
) -> Result<()> {
    // ... DELETE ALL 38 LINES ...
}
```

**UPDATE FUNCTION:** `bin/20_rbee_hive/src/heartbeat.rs:158-189`

```rust
// In start_discovery_with_backoff()
// Change from send_heartbeat_to_queen() to send_ready_callback_to_queen()

// BEFORE (DELETE):
match send_heartbeat_to_queen(&hive_info, &queen_url, Some(capabilities)).await {

// AFTER (KEEP):
match send_ready_callback_to_queen(&hive_info, &queen_url).await {
```

### 2. Queen: Old POST Receiver

**DELETE OLD FUNCTION:** `bin/10_queen_rbee/src/http/heartbeat.rs:68-95`

```rust
// TEAM-374: DELETE THIS - replaced by SSE subscription
pub async fn handle_hive_heartbeat(
    State(state): State<HeartbeatState>,
    Json(heartbeat): Json<HiveHeartbeat>,
) -> Result<Json<HttpHeartbeatAcknowledgement>, (StatusCode, String)> {
    // ... DELETE ALL 27 LINES ...
}
```

**Note:** `handle_hive_ready()` from Phase 2 stays - that's the discovery callback.

### 3. Contracts: Simplify HiveHeartbeat

**UPDATE:** `bin/97_contracts/hive-contract/src/heartbeat.rs`

**HiveHeartbeat struct is no longer needed for continuous telemetry:**

```rust
// TEAM-374: This struct is now ONLY used during discovery (with capabilities)
// Normal telemetry flows via SSE HiveHeartbeatEvent instead

// Option A: Keep for discovery, rename to HiveDiscoveryPayload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveDiscoveryPayload {
    pub hive: HiveInfo,
    pub capabilities: Vec<HiveDevice>,
}

// Option B: Delete entirely, use HiveReadyCallback instead
// (Recommended - simpler)
```

**Recommendation:** DELETE `HiveHeartbeat` struct, use `HiveReadyCallback` for discovery.

---

## Compiler Will Find These Call Sites

**Run after deletions:**

```bash
cargo check --bin rbee-hive
cargo check --bin queen-rbee
```

**Expected errors:**
- ✅ `send_heartbeat_to_queen` not found → Update to `send_ready_callback_to_queen`
- ✅ `handle_hive_heartbeat` not found → Remove route registration
- ✅ `HiveHeartbeat` not found → Use `HiveReadyCallback` or remove import

**Fix each compilation error - no guessing, compiler tells you exactly what to fix.**

---

## Implementation Steps

### Step 1: DELETE Hive Functions

```rust
// bin/20_rbee_hive/src/heartbeat.rs

// TEAM-374: DELETED send_heartbeat_to_queen() - replaced by send_ready_callback_to_queen()
// TEAM-374: DELETED start_normal_telemetry_task() - replaced by SSE broadcaster

// KEEP:
// - start_heartbeat_task() (entry point)
// - start_discovery_with_backoff() (discovery logic)
// - send_ready_callback_to_queen() (discovery callback)
// - detect_capabilities() (still used during discovery)
```

### Step 2: Fix Hive Compilation Errors

```bash
cargo check --bin rbee-hive 2>&1 | grep error

# Fix each error:
# - Replace send_heartbeat_to_queen() calls with send_ready_callback_to_queen()
# - Remove unused imports
# - Update function signatures if needed
```

### Step 3: DELETE Queen Function

```rust
// bin/10_queen_rbee/src/http/heartbeat.rs

// TEAM-374: DELETED handle_hive_heartbeat() - replaced by handle_hive_ready() + SSE subscription

// KEEP:
// - handle_hive_ready() (discovery callback from Phase 2)
// - HeartbeatState (used by SSE stream)
// - HeartbeatEvent (forwarded from hive SSE)
```

### Step 4: Fix Queen Compilation Errors

```bash
cargo check --bin queen-rbee 2>&1 | grep error

# Fix each error:
# - Remove route registration for handle_hive_heartbeat
# - Remove unused imports
# - Update tests if needed
```

### Step 5: Update Integration Tests

**MODIFY:** `bin/20_rbee_hive/tests/heartbeat_tests.rs`

```rust
// TEAM-374: UPDATE TEST - discovery callback, not continuous telemetry

#[tokio::test]
async fn test_discovery_callback() {
    let mock_queen = MockServer::start().await;
    
    // BEFORE: Mock expected POST /v1/hive-heartbeat every 1s
    // AFTER: Mock expects POST /v1/hive/ready (one-time)
    
    let ready_mock = mock_queen
        .mock("POST", "/v1/hive/ready")
        .with_status(200)
        .with_body(r#"{"status":"ok","message":"Subscribed"}"#)
        .expect(1) // ONE-TIME callback
        .create();
    
    // Send ready callback
    let result = send_ready_callback_to_queen(&hive_info, &mock_queen.url()).await;
    assert!(result.is_ok());
    
    // Verify callback received
    ready_mock.assert();
}
```

---

## Files Modified Summary

### DELETED (RULE ZERO)
1. `bin/20_rbee_hive/src/heartbeat.rs:54-92` - `send_heartbeat_to_queen()` function
2. `bin/20_rbee_hive/src/heartbeat.rs:196-252` - `start_normal_telemetry_task()` function
3. `bin/10_queen_rbee/src/http/heartbeat.rs:68-95` - `handle_hive_heartbeat()` function
4. `bin/97_contracts/hive-contract/src/heartbeat.rs` - `HiveHeartbeat` struct (optional)

### UPDATED
1. `bin/20_rbee_hive/src/heartbeat.rs` - Update discovery to use new callback
2. `bin/10_queen_rbee/src/main.rs` - Remove old route registration
3. `bin/20_rbee_hive/tests/heartbeat_tests.rs` - Update tests for callback

**Estimated LOC deleted:** ~150-200 lines

---

## Testing After Cleanup

### Manual Test

```bash
# Terminal 1: Start Queen
cargo run --bin queen-rbee -- --port 7833

# Terminal 2: Start Hive
cargo run --bin rbee-hive -- --port 7835 --queen-url http://localhost:7833

# Expected:
# 1. Hive sends ONE POST /v1/hive/ready
# 2. Queen subscribes to GET /v1/heartbeats/stream
# 3. Telemetry flows via SSE (every 1s)

# Terminal 3: Check Queen SSE
curl -N http://localhost:7833/v1/heartbeats/stream

# Should see HiveTelemetry events every 1s

# Terminal 4: Check Hive SSE
curl -N http://localhost:7835/v1/heartbeats/stream

# Should see Telemetry events every 1s
```

### Verification Checklist

- [ ] `cargo check --bin rbee-hive` passes
- [ ] `cargo check --bin queen-rbee` passes
- [ ] `cargo test --bin rbee-hive` passes
- [ ] Discovery handshake works (both scenarios)
- [ ] SSE telemetry flows correctly
- [ ] No POST /v1/hive-heartbeat endpoint exists
- [ ] No continuous POST calls in logs

---

## Rule ZERO Compliance

✅ **DELETED old code** - No deprecation, no wrappers  
✅ **Compiler-driven** - Let compiler find all call sites  
✅ **Clean codebase** - Single source of truth (SSE only)  
✅ **No backwards compatibility** - Breaking change is temporary pain  

**Quote from RULE ZERO:**
> "Breaking changes are temporary. Entropy is forever."

**We broke cleanly, fixed compilation errors, moved on.**

---

## Success Criteria

1. ✅ No POST-based continuous telemetry code exists
2. ✅ All telemetry flows via SSE
3. ✅ Discovery callback triggers SSE subscription
4. ✅ Compilation succeeds with no warnings
5. ✅ Integration tests pass
6. ✅ Manual testing confirms SSE works end-to-end

---

## Architecture After Phase 3

```
┌─────────────────────────────────────────────────────┐
│                   FINAL ARCHITECTURE                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Hive Discovery:                                    │
│    1. Hive → POST /v1/hive/ready → Queen           │
│       (ONE-TIME callback with exponential backoff)  │
│                                                     │
│  Continuous Telemetry:                              │
│    2. Queen → Subscribe to GET /v1/heartbeats/stream│
│       (SSE connection to hive)                      │
│                                                     │
│  Queen Aggregation:                                 │
│    3. Queen → Forward to GET /v1/heartbeats/stream  │
│       (Queen's SSE for all hives)                   │
│                                                     │
│  Client Access:                                     │
│    - Hive SDK → Direct to hive SSE                  │
│    - Queen SDK → Aggregated queen SSE               │
│                                                     │
│  Handshake Preserved:                               │
│    ✅ Bidirectional startup discovery               │
│    ✅ Exponential backoff                           │
│    ✅ Queen restart detection                       │
│    ✅ SSH config reading                            │
│                                                     │
│  Changed:                                           │
│    ❌ POST continuous telemetry → SSE streams       │
│    ❌ Active push → Passive pull                    │
└─────────────────────────────────────────────────────┘
```

---

## What We Preserved (Handshake)

✅ **Discovery intact:**
- Exponential backoff (0s, 2s, 4s, 8s, 16s)
- Bidirectional startup (Queen first or Hive first)
- Queen restart detection (resend callback)
- SSH config reading

✅ **What changed:**
- Discovery callback is ONE-TIME (not continuous)
- After callback, SSE handles continuous telemetry
- Simpler, cleaner, more efficient

---

**TEAM-374: DELETE old code. Let compiler guide you. No fear.**
