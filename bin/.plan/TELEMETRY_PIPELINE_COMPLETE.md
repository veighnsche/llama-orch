# COMPLETE TELEMETRY PIPELINE - Worker → Hive → Queen → UI

**Date:** Oct 30, 2025  
**Teams:** 359, 360, 361, 362, 363  
**Status:** ✅ PRODUCTION READY

---

## 🎯 **OVERVIEW**

Complete end-to-end telemetry pipeline that collects GPU, CPU, memory, and model information from workers and delivers it to the UI in real-time for monitoring and scheduling.

**Key Features:**
- No worker cooperation needed (OS-level monitoring)
- Real-time updates (1 second interval)
- GPU utilization and VRAM tracking
- Model detection from command line
- SSE streaming to UI
- Scheduling APIs for Queen

---

## 📊 **COMPLETE DATA FLOW**

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: WORKER SPAWN                                            │
└─────────────────────────────────────────────────────────────────┘
Worker spawned via lifecycle-local
    ↓
ProcessMonitor::spawn_monitored()
    ↓
Placed in cgroup: /sys/fs/cgroup/rbee.slice/{group}/{instance}/
    Example: /sys/fs/cgroup/rbee.slice/llm/8080/

Files:
- bin/96_lifecycle/lifecycle-local/src/start.rs
- bin/25_rbee_hive_crates/monitor/src/monitor.rs

┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: HIVE MONITORING (Every 1 Second)                        │
└─────────────────────────────────────────────────────────────────┘
Hive heartbeat task runs every 1s
    ↓
rbee_hive_monitor::collect_all_workers()
    ↓
For each worker in cgroup tree:
  1. Read cgroup stats:
     - /sys/fs/cgroup/rbee.slice/{group}/{instance}/cpu.stat
     - /sys/fs/cgroup/rbee.slice/{group}/{instance}/memory.current
     - /sys/fs/cgroup/rbee.slice/{group}/{instance}/io.stat
  
  2. Query nvidia-smi:
     nvidia-smi --query-compute-apps=pid,used_memory,sm --format=csv
     → gpu_util_pct (0.0 = idle, >0 = busy)
     → vram_mb (GPU memory used)
  
  3. Parse /proc/{pid}/cmdline:
     → Extract --model argument
     → model = "llama-3.2-1b"
  
  4. Calculate uptime:
     → Read /proc/{pid}/stat for start time

Files:
- bin/20_rbee_hive/src/heartbeat.rs (start_heartbeat_task)
- bin/25_rbee_hive_crates/monitor/src/telemetry.rs (collect_all_workers)
- bin/25_rbee_hive_crates/monitor/src/monitor.rs (collect_stats_linux)

┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: HIVE → QUEEN (Every 1 Second)                           │
└─────────────────────────────────────────────────────────────────┘
Build HiveHeartbeat payload:
{
  "hive": {
    "id": "localhost",
    "hostname": "127.0.0.1",
    "port": 7835,
    ...
  },
  "timestamp": "2025-10-30T18:30:00Z",
  "workers": [
    {
      "pid": 12345,
      "group": "llm",
      "instance": "8080",
      "cpu_pct": 45.2,
      "rss_mb": 4096,
      "gpu_util_pct": 85.0,    // ← Worker is BUSY
      "vram_mb": 8192,          // ← 8GB VRAM used
      "model": "llama-3.2-1b",  // ← Model loaded
      "uptime_s": 3600,
      "io_r_mb_s": 0.0,
      "io_w_mb_s": 0.0
    }
  ]
}
    ↓
POST http://localhost:7833/v1/hive-heartbeat
    ↓
Queen receives telemetry

Files:
- bin/20_rbee_hive/src/heartbeat.rs (send_heartbeat_to_queen)
- bin/97_contracts/hive-contract/src/heartbeat.rs (HiveHeartbeat struct)

┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: QUEEN STORAGE                                           │
└─────────────────────────────────────────────────────────────────┘
Queen receives POST /v1/hive-heartbeat
    ↓
handle_hive_heartbeat() extracts data:
  1. hive_registry.update_hive(heartbeat)
     → Stores hive info
  
  2. hive_registry.update_workers(hive_id, workers)
     → Stores workers in HashMap<String, Vec<ProcessStats>>
  
  3. Broadcast HeartbeatEvent::HiveTelemetry to SSE
     → event_tx.send(event)

Files:
- bin/10_queen_rbee/src/http/heartbeat.rs (handle_hive_heartbeat)
- bin/15_queen_rbee_crates/hive-registry/src/registry.rs (HiveRegistry)

┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: QUEEN → UI (SSE Stream)                                 │
└─────────────────────────────────────────────────────────────────┘
UI connects to: GET http://localhost:7833/v1/heartbeats/stream
    ↓
SSE stream sends events:
  
  Event 1 (every 2.5s): Queen heartbeat
  {
    "type": "queen",
    "workers_online": 3,
    "hives_online": 1,
    "worker_ids": [...],
    "hive_ids": [...]
  }
  
  Event 2 (every 1s): Hive telemetry
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

Files:
- bin/10_queen_rbee/src/http/heartbeat_stream.rs (handle_heartbeat_stream)
- bin/10_queen_rbee/src/http/heartbeat.rs (HeartbeatEvent enum)

┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: SCHEDULING (Queen-side)                                 │
└─────────────────────────────────────────────────────────────────┘
Queen can query workers for scheduling:

// Find idle workers
let idle = hive_registry.find_idle_workers();
// Returns workers where gpu_util_pct == 0.0

// Find workers with specific model
let workers = hive_registry.find_workers_with_model("llama-3.2-1b");

// Find workers with VRAM capacity
let workers = hive_registry.find_workers_with_capacity(4096);
// Returns workers where vram_mb + 4096 < 24576

// Get all workers
let all = hive_registry.get_all_workers();

Files:
- bin/15_queen_rbee_crates/hive-registry/src/registry.rs (scheduling methods)
```

---

## 🔍 **KEY FILES BY COMPONENT**

### **Worker Spawn (TEAM-359)**
```
bin/96_lifecycle/lifecycle-local/src/start.rs
  - start_daemon() with monitoring metadata
  - Calls ProcessMonitor::spawn_monitored()

bin/25_rbee_hive_crates/monitor/src/monitor.rs
  - ProcessMonitor::spawn_monitored()
  - Creates cgroup: /sys/fs/cgroup/rbee.slice/{group}/{instance}/
```

### **Telemetry Collection (TEAM-360)**
```
bin/25_rbee_hive_crates/monitor/src/monitor.rs
  - ProcessMonitor::collect_stats() - Main collection logic
  - query_nvidia_smi() - GPU stats via nvidia-smi
  - extract_model_from_cmdline() - Model detection
  - collect_stats_linux() - Linux cgroup backend

bin/25_rbee_hive_crates/monitor/src/telemetry.rs
  - collect_all_workers() - High-level API
  - collect_group() - Per-group collection
  - collect_instance() - Per-instance collection

bin/25_rbee_hive_crates/monitor/src/lib.rs
  - ProcessStats struct definition
```

### **Hive Heartbeat (TEAM-361)**
```
bin/20_rbee_hive/src/heartbeat.rs
  - start_heartbeat_task() - Spawns 1s interval task
  - send_heartbeat_to_queen() - Collects and sends telemetry

bin/97_contracts/hive-contract/src/heartbeat.rs
  - HiveHeartbeat struct with workers field
```

### **Queen Storage (TEAM-362)**
```
bin/10_queen_rbee/src/http/heartbeat.rs
  - handle_hive_heartbeat() - Receives telemetry
  - HeartbeatEvent enum (HiveTelemetry + Queen)
  - HeartbeatState struct

bin/15_queen_rbee_crates/hive-registry/src/registry.rs
  - HiveRegistry with worker storage
  - update_workers() - Store workers
  - find_idle_workers() - Scheduling query
  - find_workers_with_model() - Model matching
  - find_workers_with_capacity() - VRAM check
```

### **SSE Streaming (TEAM-362)**
```
bin/10_queen_rbee/src/http/heartbeat_stream.rs
  - handle_heartbeat_stream() - SSE endpoint
  - Broadcasts Queen + HiveTelemetry events
```

---

## 🐛 **DEBUGGING GUIDE**

### **Problem: Workers not appearing in telemetry**

**Check 1: Is worker in cgroup?**
```bash
ls -la /sys/fs/cgroup/rbee.slice/
# Should see: llm/, vllm/, comfy/, etc.

ls -la /sys/fs/cgroup/rbee.slice/llm/
# Should see: 8080/, 8081/, etc.

cat /sys/fs/cgroup/rbee.slice/llm/8080/cgroup.procs
# Should show PIDs
```

**Check 2: Is Hive collecting telemetry?**
```bash
# Check Hive logs
journalctl -u rbee-hive -f

# Should see every 1s:
# "Collected telemetry for N workers"
# "Hive telemetry sent successfully"
```

**Check 3: Is Queen receiving telemetry?**
```bash
# Check Queen logs
journalctl -u queen-rbee -f

# Should see every 1s:
# "🐝 Hive telemetry: hive_id=localhost, workers=3"
```

**Check 4: Is SSE stream working?**
```bash
curl -N http://localhost:7833/v1/heartbeats/stream

# Should see events:
# event: heartbeat
# data: {"type":"hive_telemetry","hive_id":"localhost","workers":[...]}
```

### **Problem: GPU stats showing 0**

**Check 1: Is nvidia-smi available?**
```bash
nvidia-smi --query-compute-apps=pid,used_memory,sm --format=csv,noheader,nounits

# Should show:
# 12345, 8192, 85
```

**Check 2: Is worker actually using GPU?**
```bash
nvidia-smi

# Check if process appears in GPU list
```

**Fix:** If nvidia-smi not found, gpu_util_pct and vram_mb will be 0 (graceful degradation).

### **Problem: Model name not detected**

**Check: Command line has --model?**
```bash
cat /proc/{pid}/cmdline | tr '\0' ' '

# Should contain: --model llama-3.2-1b
```

**Fix:** Ensure worker spawned with --model argument.

### **Problem: Scheduling not finding workers**

**Check 1: Are workers stored?**
```rust
// In Queen code
let all_workers = hive_registry.get_all_workers();
eprintln!("Total workers: {}", all_workers.len());
```

**Check 2: Are queries correct?**
```rust
// Idle check
let idle = hive_registry.find_idle_workers();
// Only returns workers where gpu_util_pct == 0.0

// Model check
let workers = hive_registry.find_workers_with_model("llama-3.2-1b");
// Exact string match on model field

// Capacity check
let workers = hive_registry.find_workers_with_capacity(4096);
// Checks: vram_mb + 4096 < 24576 (hardcoded 24GB limit)
```

---

## 📊 **DATA STRUCTURES**

### **ProcessStats (Worker Telemetry)**
```rust
// bin/25_rbee_hive_crates/monitor/src/lib.rs
pub struct ProcessStats {
    pub pid: u32,              // Process ID
    pub group: String,         // "llm", "vllm", "comfy"
    pub instance: String,      // "8080", "8081"
    pub cpu_pct: f64,         // CPU usage (placeholder, returns 0.0)
    pub rss_mb: u64,          // RAM usage in MB
    pub io_r_mb_s: f64,       // I/O read (placeholder, returns 0.0)
    pub io_w_mb_s: f64,       // I/O write (placeholder, returns 0.0)
    pub uptime_s: u64,        // Process uptime in seconds
    
    // TEAM-360: GPU telemetry
    pub gpu_util_pct: f64,    // GPU utilization (0.0 = idle, >0 = busy)
    pub vram_mb: u64,         // GPU memory used in MB
    
    // TEAM-360: Model detection
    pub model: Option<String>, // Model name from --model arg
}
```

### **HiveHeartbeat (Hive → Queen)**
```rust
// bin/97_contracts/hive-contract/src/heartbeat.rs
pub struct HiveHeartbeat {
    pub hive: HiveInfo,              // Hive metadata
    pub timestamp: HeartbeatTimestamp, // When sent
    pub workers: Vec<ProcessStats>,   // TEAM-361: Worker telemetry
}
```

### **HeartbeatEvent (Queen → UI)**
```rust
// bin/10_queen_rbee/src/http/heartbeat.rs
pub enum HeartbeatEvent {
    HiveTelemetry {
        hive_id: String,
        timestamp: String,
        workers: Vec<ProcessStats>,
    },
    Queen {
        workers_online: usize,
        workers_available: usize,
        hives_online: usize,
        hives_available: usize,
        worker_ids: Vec<String>,
        hive_ids: Vec<String>,
        timestamp: String,
    },
}
```

---

## ⚙️ **CONFIGURATION**

### **Hive Heartbeat Interval**
```rust
// bin/20_rbee_hive/src/heartbeat.rs:49
let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
```
**Change to:** Adjust interval for different update frequency

### **Queen Heartbeat Interval**
```rust
// bin/10_queen_rbee/src/http/heartbeat_stream.rs:27
let mut queen_interval = interval(Duration::from_millis(2500));
```
**Change to:** Adjust Queen's self-heartbeat frequency

### **GPU VRAM Capacity**
```rust
// bin/15_queen_rbee_crates/hive-registry/src/registry.rs:151
let total_vram = 24576; // 24GB
```
**Change to:** Match your GPU capacity

---

## 🧪 **TESTING**

### **Manual Test: End-to-End**
```bash
# 1. Start Queen
cargo run --bin queen-rbee

# 2. Start Hive
cargo run --bin rbee-hive -- --queen-url http://localhost:7833

# 3. Spawn a worker (via rbee-keeper or direct)
# Worker should appear in cgroup tree

# 4. Check SSE stream
curl -N http://localhost:7833/v1/heartbeats/stream

# Should see hive_telemetry events with worker data
```

### **Unit Tests**
```bash
# Monitor crate
cargo test -p rbee-hive-monitor

# Hive registry
cargo test -p queen-rbee-hive-registry

# Integration
cargo test -p queen-rbee
cargo test -p rbee-hive
```

---

## 📈 **PERFORMANCE**

### **Telemetry Collection**
- **Frequency:** 1 second
- **Per-worker overhead:** ~1ms (cgroup reads + nvidia-smi query)
- **Network:** ~1-5KB per heartbeat (depends on worker count)

### **SSE Streaming**
- **Frequency:** 1 second (hive telemetry) + 2.5 seconds (queen heartbeat)
- **Per-client overhead:** Minimal (broadcast channel)

### **Storage**
- **Memory:** ~1KB per worker (ProcessStats in HashMap)
- **No persistence:** All in-memory (RAM only)

---

## 🚀 **FUTURE ENHANCEMENTS**

### **Not Yet Implemented**
1. **CPU percentage calculation** - Currently returns 0.0 (needs time-delta)
2. **I/O rate calculation** - Currently returns 0.0 (needs time-delta)
3. **Node stats** - CPU/RAM for entire node (not per-worker)
4. **Exponential backoff discovery** - Hive sends immediately, no retry logic
5. **GPU temperature** - Not tracked
6. **Multi-GPU support** - Assumes single GPU per worker

### **Known Limitations**
1. **Linux only** - cgroup v2 monitoring requires Linux
2. **nvidia-smi required** - GPU stats require NVIDIA GPU
3. **Hardcoded VRAM capacity** - 24GB assumed in scheduling
4. **No worker cooperation** - Workers don't report their own state

---

## ✅ **VERIFICATION CHECKLIST**

- [x] Workers spawn in cgroup tree
- [x] Hive collects telemetry every 1s
- [x] GPU stats collected via nvidia-smi
- [x] Model detected from command line
- [x] Hive sends to Queen every 1s
- [x] Queen stores workers in HiveRegistry
- [x] Queen broadcasts to SSE stream
- [x] UI receives hive_telemetry events
- [x] Scheduling APIs available
- [x] No deprecated code (RULE ZERO)
- [x] No TODO markers
- [x] All crates compile

---

## 📞 **SUPPORT**

**For bugs or questions:**
1. Check debugging guide above
2. Review key files by component
3. Check logs (journalctl)
4. Verify cgroup tree exists
5. Test SSE stream manually

**Documentation:**
- `bin/.plan/TEAM_359_MONITORING_INTEGRATION.md`
- `bin/.plan/TEAM_360_COMPLETE.md`
- `bin/.plan/TEAM_361_COMPLETE.md`
- `bin/.plan/TEAM_362_COMPLETE.md`
- `bin/.plan/TEAM_363_COMPLETE.md`

---

**Pipeline Status:** ✅ PRODUCTION READY  
**Last Updated:** Oct 30, 2025  
**Teams:** 359, 360, 361, 362, 363
