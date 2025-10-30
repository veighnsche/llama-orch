# CRITICAL TELEMETRY PIPELINE INVESTIGATION

**Date:** Oct 30, 2025  
**Status:** 🔍 INVESTIGATION COMPLETE  
**Purpose:** Document ALL behaviors for comprehensive testing

---

## 🎯 EXECUTIVE SUMMARY

The telemetry pipeline is a **6-stage data flow** that collects worker metrics and delivers them to the UI in real-time. Every component has **specific failure modes** that must be tested.

**Critical Dependencies:**
- Linux cgroup v2 (worker placement + monitoring)
- nvidia-smi (GPU stats)
- /proc filesystem (PID, cmdline, uptime)
- HTTP/SSE (Hive → Queen → UI)

**Testing Gap:** No end-to-end tests, no fault injection, no performance tests

---

## 📊 COMPONENT BEHAVIORS

### **1. WORKER SPAWN (lifecycle-local)**

**File:** `bin/96_lifecycle/lifecycle-local/src/start.rs`

**Behavior:**
```rust
ProcessMonitor::spawn_monitored(
    MonitorConfig {
        group: "llm",
        instance: "8080",
        cpu_limit: Some("200%"),
        memory_limit: Some("4G"),
    },
    "/path/to/llm-worker",
    vec!["--model", "llama-3.2-1b", "--port", "8080"]
)
```

**Steps:**
1. Create cgroup directory: `/sys/fs/cgroup/rbee.slice/llm/8080/`
2. Write resource limits:
   - `cpu.max`: "200000 100000" (200% = 2 cores)
   - `memory.max`: "4294967296" (4GB)
3. Spawn process with `Stdio::null()` (detached)
4. Write PID to `cgroup.procs`: `echo {pid} >> /sys/fs/cgroup/rbee.slice/llm/8080/cgroup.procs`
5. Return PID

**Success Conditions:**
- cgroup directory created
- PID appears in `cgroup.procs`
- Process is running
- Resource limits applied

**Failure Modes:**
- cgroup filesystem not writable (permissions)
- Invalid resource limit format
- Binary not found
- Process fails immediately after spawn
- PID cannot be moved to cgroup (already dead)

**Platform Behavior:**
- Linux: Full cgroup v2 support
- macOS/Windows: Plain spawn, no cgroups, no limits

---

### **2. TELEMETRY COLLECTION (rbee-hive-monitor)**

**File:** `bin/25_rbee_hive_crates/monitor/src/monitor.rs`

**Behavior:**
```rust
let stats = ProcessMonitor::collect_stats("llm", "8080").await?;
// Returns ProcessStats with 12 fields
```

**Data Sources:**

#### **2A. Cgroup Stats (Linux)**
```bash
# CPU (usage_usec)
cat /sys/fs/cgroup/rbee.slice/llm/8080/cpu.stat
# usage_usec 123456789

# Memory (RSS)
cat /sys/fs/cgroup/rbee.slice/llm/8080/memory.current
# 4294967296

# PIDs
cat /sys/fs/cgroup/rbee.slice/llm/8080/cgroup.procs
# 12345

# I/O (placeholder - returns 0.0)
cat /sys/fs/cgroup/rbee.slice/llm/8080/io.stat
# 8:0 rbytes=1234 wbytes=5678
```

**CPU Percentage:** Currently returns `0.0` (TODO: needs time-delta calculation)  
**I/O Rates:** Currently returns `0.0` (TODO: needs rate calculation)

#### **2B. GPU Stats (nvidia-smi)**
```bash
nvidia-smi --query-compute-apps=pid,used_memory,sm --format=csv,noheader,nounits
# 12345, 8192, 85
```

**Behavior:**
- If nvidia-smi not found → returns `(0.0, 0)`
- If command fails → returns `(0.0, 0)`
- If PID not in output → returns `(0.0, 0)` (process not using GPU)
- If PID found → returns `(gpu_util_pct, vram_mb)`

**Graceful Degradation:** GPU stats are optional, missing data doesn't fail collection

#### **2C. Model Detection (/proc/pid/cmdline)**
```bash
cat /proc/12345/cmdline | tr '\0' ' '
# llm-worker --model llama-3.2-1b --port 8080
```

**Parsing Logic:**
- Split by null bytes (`\0`)
- Find `--model` argument
- Return next argument as model name
- If `--model` not found → returns `None`

#### **2D. Process Uptime (/proc/pid/stat)**
```bash
cat /proc/12345/stat
# 12345 (llm-worker) R ... {starttime_jiffies} ...

cat /proc/uptime
# 3600.50 14400.00
```

**Calculation:**
```
starttime_secs = starttime_jiffies / hz (100 Hz typical)
uptime_s = system_uptime_secs - starttime_secs
```

**ProcessStats Output:**
```rust
ProcessStats {
    pid: 12345,
    group: "llm",
    instance: "8080",
    cpu_pct: 0.0,              // TODO: needs time-delta
    rss_mb: 4096,
    io_r_mb_s: 0.0,            // TODO: needs rate calculation
    io_w_mb_s: 0.0,
    uptime_s: 3600,
    gpu_util_pct: 85.0,        // 0.0 = idle
    vram_mb: 8192,
    model: Some("llama-3.2-1b"),
}
```

**Success Conditions:**
- PID exists in cgroup.procs
- All files readable
- Parsing succeeds
- Returns populated ProcessStats

**Failure Modes:**
- cgroup directory missing (worker died)
- cgroup.procs empty (no processes)
- memory.current unreadable
- /proc/{pid}/stat missing (process died)
- /proc/{pid}/cmdline missing
- nvidia-smi hangs (need timeout)
- Invalid PID format
- Invalid number parsing

---

### **3. HIVE HEARTBEAT (rbee-hive)**

**File:** `bin/20_rbee_hive/src/heartbeat.rs`

**Behavior:**
```rust
// Every 1 second
tokio::time::interval(Duration::from_secs(1));

// Collect all workers
let workers = rbee_hive_monitor::collect_all_workers().await?;

// Build heartbeat
let heartbeat = HiveHeartbeat::with_workers(hive_info.clone(), workers);

// Send to Queen
POST http://localhost:7833/v1/hive-heartbeat
Content-Type: application/json

{
  "hive": {
    "id": "localhost",
    "hostname": "127.0.0.1",
    "port": 7835,
    "operational_status": "Ready",
    "health_status": "Healthy",
    "version": "0.1.0"
  },
  "timestamp": "2025-10-30T18:30:00Z",
  "workers": [
    {
      "pid": 12345,
      "group": "llm",
      "instance": "8080",
      "cpu_pct": 0.0,
      "rss_mb": 4096,
      "io_r_mb_s": 0.0,
      "io_w_mb_s": 0.0,
      "uptime_s": 3600,
      "gpu_util_pct": 85.0,
      "vram_mb": 8192,
      "model": "llama-3.2-1b"
    }
  ]
}
```

**Frequency:** Exactly 1 second intervals (using tokio::time::interval)

**Error Handling:**
- Collection failure → send empty workers array
- HTTP failure → log warning, continue loop
- Queen not ready → retry next interval

**Success Conditions:**
- Heartbeat sent every 1s
- 200 OK response from Queen
- Workers array populated correctly

**Failure Modes:**
- Queen unreachable (network)
- Queen returns 4xx/5xx
- Timeout (no timeout enforcer!)
- Collection hangs
- JSON serialization fails

---

### **4. QUEEN STORAGE (HiveRegistry)**

**File:** `bin/15_queen_rbee_crates/hive-registry/src/registry.rs`

**Behavior:**
```rust
// Receive heartbeat
POST /v1/hive-heartbeat
→ handle_hive_heartbeat()

// Store hive info
hive_registry.update_hive(heartbeat.clone());
// HashMap<String, (HiveHeartbeat, Instant)>

// Store workers
hive_registry.update_workers(&hive_id, workers);
// HashMap<String, Vec<ProcessStats>>

// Broadcast to SSE
event_tx.send(HeartbeatEvent::HiveTelemetry { ... });
```

**Data Structures:**
```rust
// HiveRegistry
inner: HeartbeatRegistry<HiveHeartbeat> {
    entries: RwLock<HashMap<String, (HiveHeartbeat, Instant)>>
}

workers: RwLock<HashMap<String, Vec<ProcessStats>>> {
    "localhost" => [ProcessStats, ProcessStats, ...]
}
```

**Thread Safety:**
- RwLock for concurrent access
- Multiple readers allowed
- Single writer blocks all

**Stale Cleanup:**
- `cleanup_stale()` removes entries older than 90 seconds
- NOT automatic - must be called explicitly

**Success Conditions:**
- Hive stored in registry
- Workers stored in workers map
- SSE event broadcast succeeds

**Failure Modes:**
- RwLock poisoned (panic in holder)
- Memory leak (no cleanup)
- Broadcast channel full (slow consumers)

---

### **5. SCHEDULING QUERIES (HiveRegistry)**

**File:** `bin/15_queen_rbee_crates/hive-registry/src/registry.rs`

**APIs:**
```rust
// Find idle workers
hive_registry.find_idle_workers()
// Returns workers where gpu_util_pct == 0.0

// Find workers with model
hive_registry.find_workers_with_model("llama-3.2-1b")
// Exact string match on model field

// Find workers with VRAM capacity
hive_registry.find_workers_with_capacity(4096)
// Returns workers where vram_mb + 4096 < 24576

// Get all workers
hive_registry.get_all_workers()
// Flattens all hive workers
```

**Idle Detection:**
```rust
gpu_util_pct == 0.0  // Exactly zero means idle
```

**VRAM Capacity Check:**
```rust
const TOTAL_VRAM: u64 = 24576;  // Hardcoded 24GB
if worker.vram_mb + required_vram_mb < TOTAL_VRAM {
    // Worker has capacity
}
```

**Model Matching:**
```rust
worker.model.as_deref() == Some(model)  // Exact match
```

**Success Conditions:**
- Queries return correct workers
- Filters work as expected
- Empty results handled

**Failure Modes:**
- Stale data (worker died but still in registry)
- Hardcoded VRAM limit wrong (multi-GPU, different GPUs)
- Model name mismatch (case sensitivity)
- Race condition (worker state changes mid-query)

---

### **6. SSE STREAMING (Queen → UI)**

**File:** `bin/10_queen_rbee/src/http/heartbeat_stream.rs`

**Behavior:**
```rust
GET /v1/heartbeats/stream
→ handle_heartbeat_stream()

// Stream events
tokio::select! {
    // Queen heartbeat every 2.5s
    _ = queen_interval.tick() => {
        event: heartbeat
        data: {"type":"queen","workers_online":3,...}
    }
    
    // Hive telemetry every 1s (forwarded)
    Ok(event) = event_rx.recv() => {
        event: heartbeat
        data: {"type":"hive_telemetry","hive_id":"localhost","workers":[...]}
    }
}
```

**Event Types:**

#### **Queen Heartbeat (2.5s interval)**
```json
{
  "type": "queen",
  "workers_online": 3,
  "workers_available": 2,
  "hives_online": 1,
  "hives_available": 1,
  "worker_ids": ["worker-1", "worker-2"],
  "hive_ids": ["localhost"],
  "timestamp": "2025-10-30T18:30:00Z"
}
```

#### **Hive Telemetry (1s interval)**
```json
{
  "type": "hive_telemetry",
  "hive_id": "localhost",
  "timestamp": "2025-10-30T18:30:00Z",
  "workers": [
    {
      "pid": 12345,
      "group": "llm",
      "instance": "8080",
      "gpu_util_pct": 85.0,
      "vram_mb": 8192,
      "model": "llama-3.2-1b",
      ...
    }
  ]
}
```

**Broadcast Channel:**
- `tokio::sync::broadcast` with capacity (default 16)
- Multiple subscribers allowed
- Slow consumer handling: `RecvError::Lagged`

**Success Conditions:**
- SSE connection established
- Events stream continuously
- Both event types delivered

**Failure Modes:**
- Client disconnects (expect and handle)
- Broadcast channel full (lagged messages)
- Serialization failures
- Network timeout

---

## 🐛 FAILURE SCENARIOS

### **1. Worker Dies Mid-Collection**

**Symptom:** cgroup.procs empty or /proc files missing

**Current Behavior:** Returns error, breaks collection loop

**Expected Behavior:** Skip dead worker, continue collecting others

**Test:** Kill worker during collection

---

### **2. nvidia-smi Hangs**

**Symptom:** Collection hangs indefinitely

**Current Behavior:** No timeout! Blocks forever

**Expected Behavior:** 5-second timeout, return (0.0, 0)

**Test:** Mock nvidia-smi that sleeps

---

### **3. Queen Unreachable**

**Symptom:** Hive heartbeat fails

**Current Behavior:** Log warning, retry next interval

**Expected Behavior:** Same (correct)

**Test:** Stop Queen, verify Hive continues

---

### **4. Stale Workers in Registry**

**Symptom:** Scheduling queries return dead workers

**Current Behavior:** No automatic cleanup

**Expected Behavior:** cleanup_stale() called periodically (TODO)

**Test:** Stop worker, verify removal after timeout

---

### **5. Broadcast Channel Full**

**Symptom:** SSE events dropped

**Current Behavior:** Sender doesn't care

**Expected Behavior:** Metrics/logging (TODO)

**Test:** Slow consumer, verify lagged handling

---

### **6. cgroup Permission Denied**

**Symptom:** Worker spawn fails

**Current Behavior:** Returns error

**Expected Behavior:** Fallback to plain spawn (TODO)

**Test:** Run without root, verify graceful degradation

---

### **7. Multi-GPU Workers**

**Symptom:** nvidia-smi returns multiple PIDs

**Current Behavior:** Only finds first GPU usage

**Expected Behavior:** Sum GPU util across all GPUs (TODO)

**Test:** Worker using 2 GPUs

---

## 📊 DATA FLOW TIMING

```
Worker Spawn (t=0)
    ↓
    ↓ [cgroup created]
    ↓
Hive Collection (t=1s, t=2s, t=3s, ...)
    ↓ [every 1s]
    ↓
Hive → Queen HTTP (t=1s, t=2s, t=3s, ...)
    ↓ [every 1s]
    ↓
Queen Storage (immediate)
    ↓
    ↓ [RwLock write]
    ↓
SSE Broadcast (immediate)
    ↓
    ↓ [tokio channel]
    ↓
UI Receives (t=1s, t=2s, t=3s, ...)
    ↓ [every 1s]
    
Total Latency: <100ms (spawn → UI)
```

**Frequency Summary:**
- Worker collection: 1s
- Hive → Queen: 1s
- Hive telemetry events: 1s
- Queen heartbeat events: 2.5s

---

## 🧪 TESTING REQUIREMENTS

### **Unit Tests (Per Component)**

#### **1. ProcessMonitor (monitor.rs)**
```rust
#[test]
fn spawn_creates_cgroup()
#[test]
fn spawn_applies_cpu_limit()
#[test]
fn spawn_applies_memory_limit()
#[test]
fn spawn_returns_pid()
#[test]
fn collect_reads_cgroup_stats()
#[test]
fn collect_queries_nvidia_smi()
#[test]
fn collect_parses_cmdline()
#[test]
fn collect_calculates_uptime()
#[test]
fn collect_handles_missing_gpu()
#[test]
fn collect_handles_dead_process()
#[test]
fn enumerate_walks_cgroup_tree()
```

#### **2. Telemetry Collection (telemetry.rs)**
```rust
#[test]
fn collect_all_workers_returns_all()
#[test]
fn collect_group_filters_by_group()
#[test]
fn collect_instance_single_worker()
#[test]
fn collect_handles_empty_cgroup()
```

#### **3. HiveRegistry (registry.rs)**
```rust
#[test]
fn update_workers_stores_correctly()
#[test]
fn get_workers_returns_stored()
#[test]
fn get_all_workers_flattens()
#[test]
fn find_idle_workers_filters()
#[test]
fn find_workers_with_model_matches()
#[test]
fn find_workers_with_capacity_checks_vram()
#[test]
fn update_workers_thread_safe()
#[test]
fn cleanup_stale_removes_old_workers()
```

#### **4. Heartbeat Sending (heartbeat.rs)**
```rust
#[test]
fn send_heartbeat_posts_to_queen()
#[test]
fn send_heartbeat_includes_workers()
#[test]
fn send_heartbeat_handles_collection_failure()
#[test]
fn start_heartbeat_task_sends_every_1s()
```

#### **5. SSE Streaming (heartbeat_stream.rs)**
```rust
#[test]
fn stream_sends_queen_heartbeat()
#[test]
fn stream_forwards_hive_telemetry()
#[test]
fn stream_handles_broadcast_lag()
#[test]
fn stream_handles_client_disconnect()
```

---

### **Integration Tests**

```rust
#[tokio::test]
async fn test_end_to_end_telemetry_flow() {
    // 1. Start Queen
    // 2. Start Hive
    // 3. Spawn worker via ProcessMonitor
    // 4. Wait 2 seconds
    // 5. Verify worker appears in Queen registry
    // 6. Verify SSE stream contains worker
}

#[tokio::test]
async fn test_worker_dies_removed_from_registry() {
    // 1. Start Queen + Hive
    // 2. Spawn worker
    // 3. Kill worker
    // 4. Wait 90 seconds (stale timeout)
    // 5. Verify worker removed from registry
}

#[tokio::test]
async fn test_scheduling_queries() {
    // 1. Start Queen + Hive
    // 2. Spawn 3 workers (different models)
    // 3. Verify find_workers_with_model()
    // 4. Verify find_idle_workers()
    // 5. Verify find_workers_with_capacity()
}

#[tokio::test]
async fn test_queen_restart_recovers() {
    // 1. Start Hive
    // 2. Start Queen
    // 3. Spawn workers
    // 4. Stop Queen
    // 5. Restart Queen
    // 6. Verify workers reappear after next heartbeat
}

#[tokio::test]
async fn test_hive_restart_clears_workers() {
    // 1. Start Queen + Hive
    // 2. Spawn workers
    // 3. Stop Hive
    // 4. Verify workers removed from Queen after timeout
}
```

---

### **Performance Tests**

```rust
#[tokio::test]
async fn bench_collection_10_workers() {
    // Measure time to collect stats for 10 workers
    // Target: <10ms
}

#[tokio::test]
async fn bench_heartbeat_payload_size() {
    // Measure JSON size for 100 workers
    // Target: <100KB
}

#[tokio::test]
async fn bench_sse_latency() {
    // Measure time from Hive send to UI receive
    // Target: <100ms
}

#[tokio::test]
async fn stress_1000_workers() {
    // Spawn 1000 workers
    // Measure collection time
    // Target: <1s
}
```

---

### **Fault Injection Tests**

```rust
#[tokio::test]
async fn test_nvidia_smi_timeout() {
    // Mock nvidia-smi that sleeps 10s
    // Verify collection continues with (0.0, 0)
}

#[tokio::test]
async fn test_cgroup_permission_denied() {
    // Mock cgroup write failure
    // Verify spawn returns error
}

#[tokio::test]
async fn test_queen_5xx_error() {
    // Mock Queen returning 500
    // Verify Hive retries next interval
}

#[tokio::test]
async fn test_broadcast_channel_full() {
    // Start SSE client that doesn't read
    // Verify slow consumer handling
}
```

---

## 🚨 CRITICAL ISSUES FOUND

### **1. NO TIMEOUT ON nvidia-smi**

**Risk:** HIGH - Can hang collection indefinitely

**Location:** `bin/25_rbee_hive_crates/monitor/src/monitor.rs:363`

**Fix:**
```rust
let output = tokio::time::timeout(
    Duration::from_secs(5),
    Command::new("nvidia-smi").args(...).output()
).await??;
```

---

### **2. CPU% ALWAYS RETURNS 0.0**

**Risk:** MEDIUM - Scheduling decisions can't use CPU

**Location:** `bin/25_rbee_hive_crates/monitor/src/monitor.rs:339`

**Fix:** Track previous usage_usec, calculate delta

---

### **3. I/O RATES ALWAYS RETURN 0.0**

**Risk:** LOW - Not currently used for scheduling

**Location:** `bin/25_rbee_hive_crates/monitor/src/monitor.rs:354`

**Fix:** Track previous io.stat, calculate rate

---

### **4. NO AUTOMATIC STALE CLEANUP**

**Risk:** MEDIUM - Dead workers accumulate

**Location:** `bin/15_queen_rbee_crates/hive-registry/src/registry.rs:161`

**Fix:** Spawn background task calling cleanup_stale() every 60s

---

### **5. HARDCODED 24GB VRAM LIMIT**

**Risk:** MEDIUM - Breaks on different GPUs

**Location:** `bin/15_queen_rbee_crates/hive-registry/src/registry.rs:151`

**Fix:** Detect GPU VRAM from nvidia-smi during collection

---

### **6. NO TIMEOUT ON HEARTBEAT HTTP**

**Risk:** MEDIUM - Can hang Hive heartbeat loop

**Location:** `bin/20_rbee_hive/src/heartbeat.rs:31`

**Fix:**
```rust
let client = reqwest::Client::builder()
    .timeout(Duration::from_secs(5))
    .build()?;
```

---

### **7. COLLECTION FAILS IF ANY WORKER FAILS**

**Risk:** MEDIUM - One dead worker breaks all telemetry

**Location:** `bin/25_rbee_hive_crates/monitor/src/monitor.rs:256`

**Fix:** Continue on error, collect what you can

---

## ✅ TESTING PRIORITY

### **P0 (Critical - Must Test)**
1. End-to-end flow (spawn → SSE)
2. Worker death handling
3. nvidia-smi timeout
4. Queen unreachable
5. Stale worker cleanup

### **P1 (High - Should Test)**
1. Scheduling query correctness
2. Thread safety (concurrent access)
3. SSE broadcast lag handling
4. Multi-GPU workers
5. cgroup permission errors

### **P2 (Medium - Nice to Have)**
1. Performance benchmarks
2. CPU% calculation
3. I/O rate calculation
4. VRAM detection
5. Model name variations

---

## 📝 TESTING IMPLEMENTATION PLAN

### **Phase 1: Unit Tests (2-3 days)**
- ProcessMonitor (11 tests)
- Telemetry collection (4 tests)
- HiveRegistry (8 tests)
- Heartbeat sending (4 tests)
- SSE streaming (4 tests)

**Total:** 31 unit tests

### **Phase 2: Integration Tests (3-4 days)**
- End-to-end flow (5 tests)
- Fault injection (4 tests)

**Total:** 9 integration tests

### **Phase 3: Performance Tests (2 days)**
- Collection benchmarks (2 tests)
- Stress tests (2 tests)

**Total:** 4 performance tests

**Grand Total:** 44 tests

---

**Investigation Complete:** Oct 30, 2025  
**Next Step:** Create testing stubs
