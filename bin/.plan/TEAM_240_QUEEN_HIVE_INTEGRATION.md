# TEAM-240: Queen ↔ Hive Integration Flow Inventory

**Date:** Oct 22, 2025  
**Components:** `queen-rbee` ↔ `rbee-hive`  
**Complexity:** High  
**Status:** ✅ COMPLETE

// TEAM-240: Investigated

---

## 1. Happy Path Flows

### 1.1 Hive Spawn Flow

**Complete Flow:**
```text
1. Queen: execute_hive_start() called
2. Queen: Load config from ~/.config/rbee/
3. Queen: Validate hive exists in hives.conf
4. Queen: Resolve binary path (config → debug → release)
5. Queen: Create DaemonManager with args
6. Queen: Spawn rbee-hive daemon (Stdio::null())
   - Args: ["--port", "8081", "--hive-id", "localhost", "--queen-url", "http://localhost:8500"]
7. Hive: Process starts, binds to port 8081
8. Hive: Start heartbeat task (5s interval)
9. Hive: Listen on /health and /capabilities
10. Queen: Poll GET http://localhost:8081/health (15s timeout)
11. Hive: Respond 200 OK
12. Queen: Health check succeeds
```

**Key Files:**
- Queen: `queen-rbee-hive-lifecycle/src/start.rs`
- Hive: `rbee-hive/src/main.rs` (lines 59-117)

### 1.2 Heartbeat Flow

**Complete Flow:**
```text
1. Hive: Heartbeat task wakes up (every 5s)
2. Hive: Collect worker states from WorkerStateProvider
3. Hive: Build HiveHeartbeatPayload {
     hive_id: "localhost",
     timestamp: now(),
     workers: vec![], // Empty for now
   }
4. Hive: POST http://localhost:8500/v1/heartbeat
5. Queen: Receive heartbeat at /v1/heartbeat endpoint
6. Queen: handle_hive_heartbeat() called
7. Queen: Update hive_registry with timestamp
8. Queen: Update hive status (based on workers)
9. Queen: Return HeartbeatAcknowledgement
10. Hive: Receive acknowledgement
11. Hive: Sleep 5s, repeat
```

**Frequency:** 5 seconds (configurable)

**Key Files:**
- Hive: `rbee-hive/src/main.rs` (lines 74-90)
- Queen: `queen-rbee/src/http/heartbeat.rs`

### 1.3 Capabilities Discovery Flow

**Complete Flow:**
```text
1. Queen: execute_hive_refresh_capabilities() called
2. Queen: Validate hive exists in config
3. Queen: Build capabilities URL: http://localhost:8081/capabilities
4. Queen: GET http://localhost:8081/capabilities (15s timeout with job_id)
5. Hive: Receive request at /capabilities endpoint
6. Hive: Detect GPUs using nvml (if available)
7. Hive: Add CPU device (always present)
8. Hive: Build CapabilitiesResponse {
     devices: [
       { id: "GPU-0", name: "RTX 4090", vram_gb: 24, ... },
       { id: "CPU", name: "CPU", vram_gb: 0, ... }
     ]
   }
9. Hive: Return JSON response
10. Queen: Parse response
11. Queen: Update capabilities cache
12. Queen: Save to ~/.config/rbee/capabilities.yaml
```

**Key Files:**
- Queen: `queen-rbee-hive-lifecycle/src/capabilities.rs`
- Hive: `rbee-hive/src/main.rs` (capabilities endpoint)

---

## 2. SSH Integration

### 2.1 SSH Connection Establishment

**Flow:**
```text
1. Queen: execute_ssh_test() called
2. Queen: Get SSH details from hives.conf:
   - hostname: 192.168.1.100
   - port: 22
   - user: admin
3. Queen: Build SSH command: ssh -p 22 admin@192.168.1.100 'echo "SSH OK"'
4. Queen: Execute command with timeout
5. Remote: SSH connection established
6. Remote: Execute echo command
7. Remote: Return "SSH OK"
8. Queen: Receive output
9. Queen: Return success
```

**Key Files:**
- Queen: `queen-rbee-hive-lifecycle/src/ssh_test.rs`

### 2.2 Remote Hive Start

**Flow (NOT YET IMPLEMENTED):**
```text
1. Queen: Detect remote hive (hostname != localhost)
2. Queen: Build SSH command to start hive
3. Queen: Execute: ssh user@host 'nohup /path/to/rbee-hive --port 8081 &'
4. Remote: Hive starts in background
5. Queen: Poll remote health endpoint
6. Queen: Fetch remote capabilities
```

**Status:** Placeholder (returns "not yet implemented")

### 2.3 SSH Error Handling

**Common Errors:**
- Connection refused → Firewall or SSH not running
- Authentication failed → Wrong credentials
- Host key verification failed → Unknown host
- Timeout → Network unreachable

**Handling:**
```rust
if !response.success {
    return Err(anyhow::anyhow!(
        "SSH connection failed: {}",
        response.error.unwrap_or_else(|| "Unknown error".to_string())
    ));
}
```

---

## 3. Worker Operations

### 3.1 Worker Lifecycle Delegation

**Flow (NOT YET IMPLEMENTED):**
```text
1. Keeper: ./rbee worker spawn --hive-id localhost --model llama2 --worker cpu --device 0
2. Keeper: POST /v1/jobs to queen
3. Queen: Parse Operation::WorkerSpawn
4. Queen: Lookup hive in registry
5. Queen: Forward to hive: POST http://localhost:8081/v1/jobs
6. Hive: Create job, spawn worker
7. Hive: Stream results via SSE
8. Queen: Proxy SSE stream back to keeper
9. Keeper: Display results
```

**Status:** TODO markers in job_router.rs (lines 274-315)

### 3.2 Worker Status Reporting

**Flow:**
```text
1. Worker: Send heartbeat to hive (30s interval)
2. Hive: Update worker registry
3. Hive: Collect worker states
4. Hive: Include in heartbeat to queen (5s interval)
5. Queen: Update hive_registry with worker states
6. Keeper: ./rbee status
7. Queen: Query hive_registry
8. Queen: Display all hives and workers
```

**Key Files:**
- Hive: `rbee-hive/src/main.rs` (WorkerStateProvider)
- Queen: `queen-rbee/src/job_router.rs` (Status operation, lines 134-201)

---

## 4. Heartbeat System

### 4.1 Heartbeat Frequency

**Hive → Queen:** 5 seconds (configurable)

**Why 5s:**
- Fast staleness detection
- Low network overhead
- Acceptable for local network

**Configuration:**
```rust
let heartbeat_config = HiveHeartbeatConfig::new(
    args.hive_id.clone(),
    args.queen_url.clone(),
    "".to_string(), // Auth token
)
.with_interval(5); // 5 seconds
```

### 4.2 Heartbeat Payload

**Structure:**
```rust
pub struct HiveHeartbeatPayload {
    pub hive_id: String,
    pub timestamp: chrono::DateTime<Utc>,
    pub workers: Vec<WorkerState>,
}

pub struct WorkerState {
    pub worker_id: String,
    pub health_status: HealthStatus,
    pub last_seen: chrono::DateTime<Utc>,
}
```

**Example:**
```json
{
  "hive_id": "localhost",
  "timestamp": "2025-10-22T15:30:00Z",
  "workers": [
    {
      "worker_id": "worker-123",
      "health_status": "Healthy",
      "last_seen": "2025-10-22T15:29:55Z"
    }
  ]
}
```

### 4.3 Staleness Detection

**Queen Logic:**
```rust
// Get all active hives (heartbeat within last 30 seconds)
let active_hive_ids = state.hive_registry.list_active_hives(30_000);
```

**Staleness Threshold:** 30 seconds (6 missed heartbeats)

**Why 30s:**
- Tolerates network hiccups
- Fast enough for user visibility
- Prevents false positives

### 4.4 Hive Re-registration

**Flow:**
```text
1. Hive: Crashes or network partition
2. Queen: No heartbeat for 30s
3. Queen: Mark hive as inactive
4. Hive: Restarts or network recovers
5. Hive: Send heartbeat
6. Queen: Receive heartbeat, mark active
7. Queen: Update last_seen timestamp
```

**No Explicit Re-registration:** Heartbeat is the registration

---

## 5. Capabilities Flow

### 5.1 Capabilities Refresh Trigger

**Manual Trigger:**
```bash
./rbee hive refresh-capabilities --alias localhost
```

**Automatic Trigger:**
- On hive start (after health check succeeds)

**Why Manual:**
- Capabilities rarely change
- GPU detection is expensive
- User controls when to refresh

### 5.2 Device Detection

**GPU Detection:**
```rust
// Try NVML first (NVIDIA GPUs)
if let Ok(nvml) = nvml::Nvml::init() {
    for device in nvml.device_count()? {
        let device = nvml.device_by_index(device)?;
        let name = device.name()?;
        let memory = device.memory_info()?;
        // Add to devices list
    }
}
```

**CPU Detection:**
```rust
// Always add CPU device
devices.push(DeviceInfo {
    id: "CPU".to_string(),
    name: "CPU".to_string(),
    vram_gb: 0,
    compute_capability: None,
    device_type: DeviceType::Cpu,
});
```

**Fallback:** If GPU detection fails, only CPU is reported

### 5.3 Capabilities Caching

**Cache Location:** `~/.config/rbee/capabilities.yaml`

**Cache Structure:**
```yaml
last_updated: "2025-10-22T15:30:00Z"
hives:
  localhost:
    hive_id: "localhost"
    base_url: "http://localhost:8081"
    last_updated: "2025-10-22T15:30:00Z"
    devices:
      - id: "GPU-0"
        name: "RTX 4090"
        vram_gb: 24
        compute_capability: "8.9"
        device_type: "gpu"
      - id: "CPU"
        name: "CPU"
        vram_gb: 0
        device_type: "cpu"
```

**Cache Invalidation:** Manual only (no TTL)

### 5.4 Timeout Handling

**Capabilities Fetch Timeout:** 15 seconds

**Why 15s:**
- GPU detection can be slow (5-10s)
- Network latency
- Prevents hanging forever

**Implementation:**
```rust
TimeoutEnforcer::new(Duration::from_secs(15))
    .with_label("Fetching capabilities")
    .with_job_id(&job_id)  // ← CRITICAL for SSE routing
    .enforce(fetch_capabilities(url)).await?;
```

---

## 6. Error Propagation

### 6.1 Hive Unreachable

**Scenario:** Hive not running

**Flow:**
```text
1. Queen: GET http://localhost:8081/health
2. HTTP: Connection refused
3. Queen: Return error "Hive not reachable"
4. Queen: Emit error narration with job_id
5. Narration → SSE → Keeper
6. Keeper: Display "❌ Hive not reachable"
```

**Handled:** Yes (with narration)

### 6.2 SSH Failures

**Scenario:** SSH connection fails

**Flow:**
```text
1. Queen: Execute SSH command
2. SSH: Connection refused / Auth failed / Timeout
3. Queen: Parse stderr output
4. Queen: Return error with SSH error message
5. Queen: Emit error narration with job_id
6. Narration → SSE → Keeper
7. Keeper: Display "❌ SSH connection failed: <error>"
```

**Handled:** Yes (with detailed error messages)

### 6.3 Heartbeat Failures

**Scenario:** Heartbeat POST fails

**Flow:**
```text
1. Hive: POST /v1/heartbeat to queen
2. HTTP: Connection refused / Timeout
3. Hive: Log error (no narration - background task)
4. Hive: Retry on next interval (5s)
5. Queen: No heartbeat received
6. Queen: Mark hive as inactive after 30s
```

**Handled:** Yes (with retry and staleness detection)

### 6.4 Capabilities Timeout

**Scenario:** GPU detection hangs

**Flow:**
```text
1. Queen: GET /capabilities with 15s timeout
2. Hive: GPU detection hangs
3. After 15s: TimeoutEnforcer fires
4. Queen: Emit timeout narration with job_id
5. Narration → SSE → Keeper
6. Keeper: Display "❌ Capabilities fetch TIMED OUT after 15s"
```

**Handled:** Yes (with TimeoutEnforcer)

---

## 7. State Synchronization

### 7.1 Hive Registry State (Queen)

**Structure:**
```rust
pub struct HiveRegistry {
    hives: Arc<Mutex<HashMap<String, HiveState>>>,
}

pub struct HiveState {
    pub hive_id: String,
    pub last_seen: i64, // Timestamp in milliseconds
    pub workers: Vec<WorkerState>,
}
```

**Updates:**
- On heartbeat received: Update last_seen + workers
- On staleness check: Filter by last_seen > (now - 30s)

**Storage:** In-memory only (no persistence)

### 7.2 Hive Status Tracking

**Status Derivation:**
```rust
// Active: Heartbeat within last 30s
if last_seen > (now - 30_000) {
    HiveStatus::Active
} else {
    HiveStatus::Inactive
}
```

**No Explicit Status Field:** Derived from last_seen timestamp

### 7.3 Worker Status Aggregation

**Flow:**
```text
1. Worker: Send heartbeat to hive
2. Hive: Update worker registry (local)
3. Hive: Collect all worker states
4. Hive: Include in heartbeat to queen
5. Queen: Update hive_registry with worker states
6. Queen: Store workers in HiveState
```

**Aggregation:** Hive collects all workers, queen stores aggregated state

---

## 8. Critical Invariants

### 8.1 Stdio::null() for Daemon Spawn

**Invariant:** Hive daemon MUST use Stdio::null()

**Why:** Prevents pipe hangs when queen runs via Command::output()

**Enforcement:** DaemonManager always sets Stdio::null()

### 8.2 Heartbeat Interval < Staleness Threshold

**Invariant:** Heartbeat interval (5s) < Staleness threshold (30s)

**Why:** Ensures at least 6 heartbeats before marked inactive

**Enforcement:** Hardcoded values (5s, 30s)

### 8.3 Capabilities Cached After Start

**Invariant:** Hive start MUST fetch and cache capabilities

**Why:** Ensures capabilities available for worker operations

**Enforcement:** execute_hive_start() calls fetch_capabilities()

### 8.4 job_id in All Narration

**Invariant:** All queen narration MUST include job_id

**Why:** Without job_id, events don't reach SSE stream

**Enforcement:** All operation handlers receive job_id parameter

---

## 9. Existing Test Coverage

### 9.1 Integration Tests

**E2E Tests:**
- `bin/test_happy_flow.sh` - Includes hive start
- Manual testing of heartbeat flow

**Coverage:**
- ✅ Hive spawn
- ✅ Health polling
- ✅ Capabilities fetch
- ✅ Heartbeat flow (manual)

### 9.2 Test Gaps

**Missing Tests:**
- ❌ SSH connection (success/failure)
- ❌ Remote hive start
- ❌ Heartbeat retry on failure
- ❌ Staleness detection (30s threshold)
- ❌ Hive re-registration after crash
- ❌ Capabilities timeout
- ❌ GPU detection failure fallback
- ❌ Worker status aggregation
- ❌ Concurrent hive operations
- ❌ Hive crash mid-operation

---

## 10. Flow Checklist

- [x] All happy paths documented
- [x] All error paths documented
- [x] All state transitions documented
- [x] All cleanup flows documented
- [x] All edge cases documented
- [x] Test coverage gaps identified

---

**Handoff:** Ready for Phase 6 (test planning)  
**Next:** TEAM-241 (hive-worker integration)
