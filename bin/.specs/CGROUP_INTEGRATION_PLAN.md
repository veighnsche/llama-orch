# cgroup v2 Integration Plan

**Date:** Oct 30, 2025  
**Status:** 📋 PLANNING  
**Related:** `HEARTBEAT_ARCHITECTURE.md`, `DISCOVERY_PROBLEM_ANALYSIS.md`

---

## 🎯 Executive Summary

**Goal:** Integrate cgroup v2 monitoring into rbee architecture for worker discovery and telemetry.

**Key Changes:**
1. Create new `25_rbee_hive_crates/cgroup-monitor` crate
2. Update `daemon-lifecycle` to support cgroup-aware process management
3. Implement Hive telemetry with cgroup v2 worker enumeration
4. Remove worker heartbeat mechanisms (workers don't heartbeat)

**Benefits:**
- ✅ Workers can't lie about their stats (OS-level monitoring)
- ✅ No worker cooperation needed (automatic discovery)
- ✅ Single telemetry path (Hive → Queen, not per-worker)
- ✅ Efficient (1 heartbeat per node, not per worker)

---

## 📦 Part 1: New Crate - `cgroup-monitor`

### **Location**
```
bin/25_rbee_hive_crates/cgroup-monitor/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── monitor.rs       # Main CgroupMonitor struct
│   ├── parser.rs        # Parse cgroup v2 files
│   ├── worker.rs        # WorkerStats from cgroup
│   └── node.rs          # NodeStats (CPU, RAM, GPUs)
```

### **Core API**

```rust
/// Monitor workers via cgroup v2 tree
pub struct CgroupMonitor {
    /// Base cgroup path (e.g., "/sys/fs/cgroup/rbee.slice")
    base_path: PathBuf,
}

impl CgroupMonitor {
    /// Create new monitor for rbee.slice
    pub fn new() -> Result<Self>;
    
    /// Enumerate all workers in rbee.slice/<service>/<instance>
    pub async fn collect_workers(&self) -> Result<Vec<WorkerStats>>;
    
    /// Collect node-level stats (CPU, RAM, GPUs)
    pub async fn collect_node_stats(&self) -> Result<NodeStats>;
}

/// Worker stats from cgroup v2
#[derive(Debug, Clone, Serialize)]
pub struct WorkerStats {
    pub worker_id: String,           // Derived from cgroup path
    pub service: String,              // e.g., "llm"
    pub instance: String,             // e.g., "8080"
    pub cgroup: String,               // e.g., "rbee.slice/llm/8080"
    pub pids: Vec<u32>,               // PIDs in cgroup
    pub port: u16,                    // Parsed from instance
    pub model: Option<String>,        // From worker metadata
    pub gpu: Option<String>,          // GPU assignment
    
    // Resource usage (from cgroup v2)
    pub cpu_pct: f64,                 // CPU usage %
    pub rss_mb: u64,                  // Resident memory (MB)
    pub vram_mb: Option<u64>,         // VRAM usage (MB) from GPU driver
    pub io_r_mb_s: f64,               // I/O read (MB/s)
    pub io_w_mb_s: f64,               // I/O write (MB/s)
    pub uptime_s: u64,                // Uptime in seconds
    pub state: WorkerState,           // ready, busy, error, etc.
}

/// Node-level stats
#[derive(Debug, Clone, Serialize)]
pub struct NodeStats {
    pub cpu_pct: f64,                 // Total CPU usage %
    pub ram_used_mb: u64,             // Used RAM (MB)
    pub ram_total_mb: u64,            // Total RAM (MB)
    pub gpus: Vec<GpuStats>,          // GPU stats
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuStats {
    pub id: String,                   // e.g., "GPU-0"
    pub util_pct: u8,                 // GPU utilization %
    pub vram_used_mb: u64,            // VRAM used (MB)
    pub vram_total_mb: u64,           // VRAM total (MB)
    pub temp_c: u8,                   // Temperature (°C)
}
```

### **cgroup v2 Files to Read**

```bash
# Worker cgroup: /sys/fs/cgroup/rbee.slice/llm/8080/
├── cgroup.procs          # PIDs in cgroup
├── cpu.stat              # CPU usage
├── memory.current        # Current memory usage
├── memory.stat           # Memory stats (RSS, cache, etc.)
├── io.stat               # I/O stats
└── pids.current          # Number of PIDs
```

### **Implementation Notes**

1. **cgroup v2 Detection:**
   ```rust
   fn is_cgroup_v2() -> bool {
       Path::new("/sys/fs/cgroup/cgroup.controllers").exists()
   }
   ```

2. **Worker Discovery:**
   ```rust
   // Enumerate: /sys/fs/cgroup/rbee.slice/<service>/<instance>/
   // Example: rbee.slice/llm/8080 → worker on port 8080
   ```

3. **CPU Calculation:**
   ```rust
   // Read cpu.stat: usage_usec=12345678
   // Calculate: (current - previous) / time_delta * 100
   ```

4. **Memory:**
   ```rust
   // Read memory.current for RSS
   // Read memory.stat for detailed breakdown
   ```

5. **GPU Stats:**
   ```rust
   // Use nvidia-smi or rocm-smi
   // Match worker PID to GPU process
   ```

### **Dependencies**

```toml
[dependencies]
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["fs", "process"] }
tracing = "0.1"
```

---

## 🔧 Part 2: Update `daemon-lifecycle` Crate

### **Current State Analysis**

**Location:** `bin/99_shared_crates/daemon-lifecycle/`

**Current Flow:**
1. `start.rs` - Starts daemon with `nohup {binary} {args} > /dev/null 2>&1 & echo $!`
2. `stop.rs` - Stops via HTTP shutdown or `pkill -f {daemon_name}`
3. `shutdown.rs` - SIGTERM/SIGKILL via `pkill`

**Problem:** No cgroup integration - processes not tracked in cgroup tree.

### **Required Changes**

#### **1. Update `start.rs` - cgroup-aware Process Start**

**Current:**
```rust
// Line 258: Start daemon in background
let start_cmd = format!("nohup {} {} > /dev/null 2>&1 & echo $!", binary_path, args);
```

**New:**
```rust
// Start daemon in cgroup slice
let start_cmd = format!(
    "systemd-run --user --scope --slice=rbee.slice --unit={service}-{instance} \
     {} {} & echo $!",
    binary_path, args
);
```

**Changes Needed:**
- Add `cgroup_config: Option<CgroupConfig>` to `HttpDaemonConfig`
- If `cgroup_config` present, use `systemd-run` instead of `nohup`
- Parse service/instance from config (e.g., "llm", "8080")

**New Config:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CgroupConfig {
    /// Service name (e.g., "llm", "hive")
    pub service: String,
    
    /// Instance identifier (e.g., "8080", "main")
    pub instance: String,
    
    /// Optional: CPU limit (e.g., "200%" = 2 cores)
    pub cpu_limit: Option<String>,
    
    /// Optional: Memory limit (e.g., "4G")
    pub memory_limit: Option<String>,
}
```

#### **2. Update `stop.rs` - cgroup-aware Process Stop**

**Current:**
```bash
pkill -f {daemon_name}  # Kills by process name
```

**New:**
```bash
# Option 1: Stop via systemd unit
systemctl --user stop {service}-{instance}.scope

# Option 2: Kill all PIDs in cgroup
cat /sys/fs/cgroup/rbee.slice/{service}/{instance}/cgroup.procs | xargs kill -TERM
```

**Changes Needed:**
- Add `cgroup_config: Option<CgroupConfig>` to `StopConfig`
- If `cgroup_config` present, use systemd or cgroup-based stop
- Fallback to `pkill` if cgroup not available

#### **3. Update `shutdown.rs` - cgroup-aware Shutdown**

**Current:**
```bash
pkill -TERM -f {daemon_name}  # SIGTERM by name
pkill -KILL -f {daemon_name}  # SIGKILL by name
```

**New:**
```bash
# SIGTERM all PIDs in cgroup
cat /sys/fs/cgroup/rbee.slice/{service}/{instance}/cgroup.procs | xargs kill -TERM

# Wait 5s

# SIGKILL all PIDs in cgroup
cat /sys/fs/cgroup/rbee.slice/{service}/{instance}/cgroup.procs | xargs kill -KILL
```

**Changes Needed:**
- Add `cgroup_config: Option<CgroupConfig>` to `ShutdownConfig`
- If `cgroup_config` present, use cgroup-based shutdown
- Fallback to `pkill` if cgroup not available

### **Backward Compatibility**

**Strategy:** Make cgroup optional via `Option<CgroupConfig>`

```rust
// Old way (still works)
let config = HttpDaemonConfig::new("rbee-hive", "http://localhost:7835");

// New way (with cgroup)
let config = HttpDaemonConfig::new("rbee-hive", "http://localhost:7835")
    .with_cgroup(CgroupConfig {
        service: "hive".to_string(),
        instance: "main".to_string(),
        cpu_limit: None,
        memory_limit: None,
    });
```

**Implementation:**
```rust
impl HttpDaemonConfig {
    pub fn with_cgroup(mut self, cgroup: CgroupConfig) -> Self {
        self.cgroup_config = Some(cgroup);
        self
    }
}
```

### **systemd-run Requirements**

**Check if available:**
```bash
which systemd-run  # Must be installed
systemctl --user status  # User session must exist
```

**Fallback Strategy:**
1. Try `systemd-run` (preferred)
2. If not available, use `nohup` (old way)
3. Log warning: "cgroup monitoring not available"

---

## 🐝 Part 3: Update Hive Telemetry

### **Current State**

**File:** `bin/20_rbee_hive/src/heartbeat.rs`

**Current Flow:**
1. Sends `HiveHeartbeat` every 30s
2. Contains only hive info (no workers)
3. Workers send separate heartbeats

### **New Flow**

**File:** `bin/20_rbee_hive/src/telemetry.rs` (rename from heartbeat.rs)

**New Flow:**
1. Collect node stats every ~1s (EMA-smoothed)
2. Enumerate workers via `CgroupMonitor`
3. Build `HiveTelemetry` with node + workers
4. Send to Queen via POST /v1/hive-heartbeat

### **Implementation**

```rust
use cgroup_monitor::{CgroupMonitor, NodeStats, WorkerStats};

/// Collect telemetry (node + workers)
async fn collect_telemetry(
    hive_id: &str,
    monitor: &CgroupMonitor,
) -> Result<HiveTelemetry> {
    // Collect node stats (CPU, RAM, GPUs)
    let node = monitor.collect_node_stats().await?;
    
    // Enumerate workers via cgroup v2
    let workers = monitor.collect_workers().await?;
    
    Ok(HiveTelemetry {
        hive_id: hive_id.to_string(),
        ts: chrono::Utc::now().to_rfc3339(),
        node,
        workers,
    })
}

/// Start telemetry task (replaces heartbeat task)
pub fn start_telemetry_task(
    hive_id: String,
    queen_url: String,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let monitor = CgroupMonitor::new().expect("Failed to init cgroup monitor");
        
        // Discovery phase: exponential backoff (0s, 2s, 4s, 8s, 16s)
        let backoff_delays = [0, 2, 4, 8, 16];
        let mut discovered = false;
        
        for delay in backoff_delays {
            if delay > 0 {
                tokio::time::sleep(Duration::from_secs(delay)).await;
            }
            
            // Try to send telemetry
            match collect_telemetry(&hive_id, &monitor).await {
                Ok(telemetry) => {
                    match send_telemetry(&telemetry, &queen_url).await {
                        Ok(_) => {
                            tracing::info!("✅ Queen discovered, entering normal telemetry mode");
                            discovered = true;
                            break;
                        }
                        Err(e) => {
                            tracing::debug!("Discovery attempt failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to collect telemetry: {}", e);
                }
            }
        }
        
        if !discovered {
            tracing::info!("⏳ Discovery failed, waiting for Queen to call /capabilities");
        }
        
        // Normal telemetry mode: ~1s interval (EMA-smoothed)
        let mut interval = tokio::time::interval(Duration::from_millis(1000));
        
        loop {
            interval.tick().await;
            
            match collect_telemetry(&hive_id, &monitor).await {
                Ok(telemetry) => {
                    if let Err(e) = send_telemetry(&telemetry, &queen_url).await {
                        tracing::warn!("Failed to send telemetry: {}", e);
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to collect telemetry: {}", e);
                }
            }
        }
    })
}
```

### **HiveTelemetry Contract**

```rust
#[derive(Debug, Clone, Serialize)]
pub struct HiveTelemetry {
    pub hive_id: String,
    pub ts: String,              // ISO 8601 timestamp
    pub node: NodeStats,
    pub workers: Vec<WorkerStats>,
}
```

**Example JSON:**
```json
{
  "hive_id": "hive:550e8400-e29b-41d4-a716-446655440000",
  "ts": "2025-10-30T14:59:12Z",
  "node": {
    "cpu_pct": 37.2,
    "ram_used_mb": 12345,
    "ram_total_mb": 65536,
    "gpus": [
      {
        "id": "GPU-0",
        "util_pct": 64,
        "vram_used_mb": 8123,
        "vram_total_mb": 24564,
        "temp_c": 66
      }
    ]
  },
  "workers": [
    {
      "worker_id": "worker:660e8400-e29b-41d4-a716-446655440001",
      "service": "llm",
      "instance": "8080",
      "cgroup": "rbee.slice/llm/8080",
      "pids": [23145],
      "port": 8080,
      "model": "llama-3.2-1b",
      "gpu": "GPU-0",
      "cpu_pct": 122.0,
      "rss_mb": 9821,
      "vram_mb": 7900,
      "io_r_mb_s": 12.3,
      "io_w_mb_s": 0.1,
      "uptime_s": 432,
      "state": "ready"
    }
  ]
}
```

---

## 👑 Part 4: Update Queen to Process Telemetry

### **Current State**

**File:** `bin/10_queen_rbee/src/http/heartbeat.rs`

**Current:**
- Receives `WorkerHeartbeat` from workers
- Receives `HiveHeartbeat` from hives
- Updates separate registries

### **New State**

**Changes:**
1. Remove `handle_worker_heartbeat` endpoint (workers don't heartbeat)
2. Update `handle_hive_heartbeat` to accept `HiveTelemetry`
3. Extract workers from telemetry, update `WorkerRegistry`
4. Update `HiveRegistry` with hive info + timestamp

**Implementation:**
```rust
pub async fn handle_hive_heartbeat(
    State(state): State<HeartbeatState>,
    Json(telemetry): Json<HiveTelemetry>,
) -> Result<Json<HttpHeartbeatAcknowledgement>, (StatusCode, String)> {
    tracing::debug!("🐝 Hive telemetry: hive_id={}", telemetry.hive_id);
    
    // Update HiveRegistry with hive info + timestamp
    state.hive_registry.update_from_telemetry(&telemetry);
    
    // Update WorkerRegistry with all workers from telemetry
    for worker in &telemetry.workers {
        state.worker_registry.update_from_telemetry(worker);
    }
    
    // Broadcast telemetry event for real-time streaming
    let event = HeartbeatEvent::HiveTelemetry {
        hive_id: telemetry.hive_id.clone(),
        node: telemetry.node.clone(),
        workers: telemetry.workers.clone(),
        timestamp: telemetry.ts.clone(),
    };
    let _ = state.event_tx.send(event);
    
    Ok(Json(HttpHeartbeatAcknowledgement {
        status: "ok".to_string(),
        message: format!("Telemetry received from hive {}", telemetry.hive_id),
    }))
}
```

---

## 📋 Implementation Checklist

### **Phase 1: Create cgroup-monitor Crate** (Week 1)
- [ ] Create `bin/25_rbee_hive_crates/cgroup-monitor/`
- [ ] Implement `CgroupMonitor::new()`
- [ ] Implement `CgroupMonitor::collect_workers()`
- [ ] Implement `CgroupMonitor::collect_node_stats()`
- [ ] Add GPU stats collection (nvidia-smi/rocm-smi)
- [ ] Write unit tests for cgroup parsing
- [ ] Write integration tests with mock cgroup files

### **Phase 2: Update daemon-lifecycle** (Week 1-2)
- [ ] Add `CgroupConfig` struct to `start.rs`
- [ ] Update `start_daemon()` to use `systemd-run` when cgroup enabled
- [ ] Add `cgroup_config` to `StopConfig`
- [ ] Update `stop_daemon()` for cgroup-aware stop
- [ ] Add `cgroup_config` to `ShutdownConfig`
- [ ] Update `shutdown_daemon()` for cgroup-aware shutdown
- [ ] Add fallback logic (systemd-run → nohup)
- [ ] Test on systems with/without systemd

### **Phase 3: Update Hive Telemetry** (Week 2)
- [ ] Rename `heartbeat.rs` to `telemetry.rs`
- [ ] Implement `collect_telemetry()` with `CgroupMonitor`
- [ ] Implement exponential backoff discovery (5 tries)
- [ ] Change interval from 30s to ~1s
- [ ] Create `HiveTelemetry` contract
- [ ] Update Hive to send telemetry instead of heartbeat
- [ ] Test telemetry collection with real workers

### **Phase 4: Update Queen** (Week 2-3)
- [ ] Remove `handle_worker_heartbeat` endpoint
- [ ] Update `handle_hive_heartbeat` to accept `HiveTelemetry`
- [ ] Update `WorkerRegistry` to accept telemetry data
- [ ] Update `HiveRegistry` to accept telemetry data
- [ ] Update `HeartbeatEvent` enum (remove Worker, update Hive)
- [ ] Update SSE streaming for telemetry events
- [ ] Test Queen with telemetry from multiple hives

### **Phase 5: Testing & Validation** (Week 3)
- [ ] Test single hive + multiple workers
- [ ] Test multiple hives + multiple workers
- [ ] Test worker crash/restart (auto-discovery)
- [ ] Test Queenless operation
- [ ] Test Queen-first discovery
- [ ] Test Hive-first discovery (exponential backoff)
- [ ] Performance test (1000 workers)
- [ ] Load test (telemetry every 1s)

### **Phase 6: Documentation** (Week 3)
- [ ] Update `HEARTBEAT_ARCHITECTURE.md` with implementation details
- [ ] Create `CGROUP_SETUP_GUIDE.md` for system requirements
- [ ] Update API documentation for telemetry endpoints
- [ ] Create migration guide from old heartbeat system
- [ ] Document troubleshooting (systemd not available, etc.)

---

## 🚨 Critical Decisions

### **1. Should we create a separate crate?**

**Answer: YES** ✅

**Reasons:**
- cgroup monitoring is Hive-specific functionality
- Keeps `daemon-lifecycle` generic (reusable for other projects)
- Clear separation of concerns
- Easier to test in isolation

**Location:** `bin/25_rbee_hive_crates/cgroup-monitor`

### **2. Should daemon-lifecycle be cgroup-aware?**

**Answer: YES (but optional)** ✅

**Reasons:**
- Workers need to be started in cgroups for monitoring
- `daemon-lifecycle` is the right place (it starts/stops processes)
- Make it optional via `Option<CgroupConfig>` for backward compatibility
- Fallback to `nohup` if `systemd-run` not available

### **3. systemd-run vs manual cgroup management?**

**Answer: systemd-run (preferred)** ✅

**Reasons:**
- ✅ Automatic cgroup creation/cleanup
- ✅ Resource limits (CPU, memory) built-in
- ✅ Widely available on modern Linux
- ✅ Handles edge cases (zombie processes, etc.)

**Fallback:** Manual cgroup management if systemd not available

### **4. Telemetry interval: 1s or 30s?**

**Answer: ~1s (EMA-smoothed)** ✅

**Reasons:**
- Real-time UI updates (important for user experience)
- Queen needs fresh data for scheduling decisions
- Network overhead minimal (1 JSON payload per node per second)
- EMA smoothing prevents jitter

**Calculation:**
```
100 hives × 1 KB payload × 1 Hz = 100 KB/s = 0.8 Mbps
```
This is negligible for modern networks.

---

## 🔍 Open Questions

### **Q1: What if systemd-run is not available?**

**Answer:** Fallback to `nohup` (old way), log warning that cgroup monitoring unavailable.

### **Q2: How to handle workers started outside cgroups?**

**Answer:** They won't be discovered by `CgroupMonitor`. This is intentional - only cgroup-managed workers are monitored.

### **Q3: What about Windows/macOS support?**

**Answer:** cgroup v2 is Linux-only. For Windows/macOS:
- Option 1: Workers send heartbeats (old way)
- Option 2: Hive uses platform-specific monitoring (future work)
- For now: Focus on Linux (primary deployment target)

### **Q4: How to test without root/systemd?**

**Answer:** 
- Unit tests: Mock cgroup files in `/tmp/mock-cgroup/`
- Integration tests: Use Docker with cgroup v2 enabled
- Local dev: Use `systemd-run --user` (no root needed)

### **Q5: What about resource limits (CPU, memory)?**

**Answer:** Add to `CgroupConfig`:
```rust
pub struct CgroupConfig {
    pub cpu_limit: Option<String>,     // e.g., "200%" = 2 cores
    pub memory_limit: Option<String>,  // e.g., "4G"
}
```

Pass to `systemd-run`:
```bash
systemd-run --user --scope --slice=rbee.slice \
  --property=CPUQuota=200% \
  --property=MemoryMax=4G \
  --unit=llm-8080 \
  ./rbee-worker --port 8080
```

---

## 📚 References

- **cgroup v2 Documentation:** https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
- **systemd-run Manual:** https://www.freedesktop.org/software/systemd/man/systemd-run.html
- **Canonical Spec:** `bin/.specs/HEARTBEAT_ARCHITECTURE.md`
- **Problem Analysis:** `bin/.specs/DISCOVERY_PROBLEM_ANALYSIS.md`

---

## ✅ Success Criteria

1. ✅ Workers discovered automatically via cgroup enumeration
2. ✅ No worker heartbeats (workers don't send anything)
3. ✅ Hive telemetry includes node stats + all workers
4. ✅ Telemetry sent every ~1s to Queen
5. ✅ Queen updates registries from telemetry
6. ✅ Worker crash/restart auto-detected
7. ✅ Backward compatible (cgroup optional)
8. ✅ Works with/without systemd (fallback to nohup)

---

**Next Steps:** Review this plan, then start Phase 1 (cgroup-monitor crate).
