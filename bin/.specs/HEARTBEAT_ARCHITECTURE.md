# Heartbeat Architecture

**Date:** Oct 30, 2025  
**Status:** CANONICAL SPECIFICATION  
**Purpose:** Define telemetry and discovery protocol for Queen ↔ Hive communication

---

## Table of Contents

1. [Core Principle](#core-principle)
2. [Discovery Protocol](#discovery-protocol)
3. [Hive Telemetry Flow](#hive-telemetry-flow)
4. [Worker Monitoring (No Worker Heartbeats)](#worker-monitoring-no-worker-heartbeats)
5. [Telemetry Payload Specification](#telemetry-payload-specification)
6. [State Machine](#state-machine)
7. [Health Determination](#health-determination)
8. [Implementation Details](#implementation-details)

---

## Core Principle

**Bidirectional discovery with Hive-side monitoring. Workers never heartbeat.**

### **Key Design Points:**

1. **Discovery Plane** (Handshake)
   - Queen discovers hives via SSH config → GET /capabilities
   - Hive discovers Queen via exponential backoff (5 tries: 0s, 2s, 4s, 8s, 16s)
   - Both mechanisms work independently (no dependency order)

2. **Telemetry Plane** (Live Stats)
   - **Hive → Queen:** Periodic telemetry heartbeats (~1s, EMA-smoothed)
   - **Workers → NO ONE:** Workers never send heartbeats
   - **Hive monitors workers:** Via cgroup v2 tree (rbee.slice/<service>/<instance>)

3. **Health Determination**
   - Queen derives health from telemetry freshness
   - Healthy: < 3× interval
   - Down: After 10 missed intervals

### **Why This Design?**

1. **No worker cooperation** - Hive monitors via OS (cgroups), workers can't lie
2. **Single telemetry path** - All metrics flow Hive → Queen (no fan-out)
3. **No dependency order** - Queen and Hive start independently
4. **Efficient** - 1 heartbeat per node (not per worker)
5. **Resilient** - Exponential backoff handles startup races

---

## Discovery Protocol

**Two independent discovery paths. Both work; neither requires the other.**

---

### Scenario 1: Queen Starts First (Pull-based Discovery via SSH)

**When Queen starts:**

1. Queen waits 5 seconds (allow services to stabilize)
2. Queen reads SSH config (list of hive hostnames/IPs)
   - **Note:** SSH config parsing logic needs shared crate extraction
   - Currently in: `bin/00_rbee_keeper/src/tauri_commands.rs::ssh_list()`
3. Queen sends parallel GET requests to all configured hives:
   ```
   GET http://{hive_hostname}:7835/capabilities?queen_url=http://queen-host:7833
   ```
4. Hive receives request:
   - Extracts `queen_url` from query parameter
   - Stores `queen_url` in memory
   - Responds with capabilities (devices, models)
5. **Trigger:** Hive immediately starts normal telemetry heartbeats to `queen_url` (~1s interval)
6. Queen stores capabilities in HiveRegistry

**Result:** All online hives discovered and begin sending telemetry.

---

### Scenario 2: Hive Starts First (Push-based Discovery with Exponential Backoff)

**When Hive starts:**

1. Hive has Queen URL configured (via CLI arg or config file)
2. Hive sends **discovery telemetry heartbeat** to Queen
3. Hive uses **exponential backoff** for first 5 attempts:
   - **Attempt 1:** Immediate (0s)
   - **Attempt 2:** 2s delay
   - **Attempt 3:** 4s delay
   - **Attempt 4:** 8s delay
   - **Attempt 5:** 16s delay

4. **On first `200 OK` response:**
   - Hive transitions to **normal telemetry mode** (~1s interval)
   - Discovery phase complete

5. **If all 5 attempts fail:**
   - Hive stops discovery heartbeats
   - Waits for Queen to initiate via `/capabilities` (Scenario 1)

**Discovery telemetry payload (same format as normal telemetry):**
```json
{
  "hive_id": "hive:550e8400-e29b-41d4-a716-446655440000",
  "ts": "2025-10-30T15:13:00Z",
  "node": {
    "cpu_pct": 45.2,
    "ram_used_mb": 12800,
    "ram_total_mb": 65536,
    "gpus": [
      {
        "id": "GPU-0",
        "util_pct": 0,
        "vram_used_mb": 0,
        "vram_total_mb": 24564,
        "temp_c": 42
      }
    ]
  },
  "workers": []
}
```

**Queen responses:**
- `200 OK` → Telemetry received, Hive enters normal mode
- `404 Not Found` → Queen not ready, retry with backoff
- `503 Service Unavailable` → Queen overloaded, retry with backoff

**Result:** Hive discovered by Queen, enters normal telemetry mode.

---

### Scenario 3: Both Start Simultaneously (Race Condition Handling)

**When both start at the same time:**

1. **Hive side:**
   - Starts exponential backoff discovery telemetry (Scenario 2)
   - Attempt 1 (0s): Queen not ready → 404
   - Attempt 2 (2s): Queen not ready → 404
   - Attempt 3 (4s): Queen not ready → 404

2. **Queen side (after 5s wait):**
   - Reads SSH config
   - Sends `GET /capabilities?queen_url=...` to all hives (Scenario 1)
   - Hive responds with capabilities
   - **Trigger:** Hive starts normal telemetry immediately

3. **Hive side (attempt 4 at 8s):**
   - Already in normal telemetry mode (triggered by capabilities request)
   - Discovery backoff stops

**Result:** Both mechanisms work in parallel. First success wins. No duplicate work.

---

## Hive Telemetry Flow

**After discovery, Hive sends telemetry heartbeats to Queen (~1s interval, EMA-smoothed):**

**Endpoint:** `POST http://{queen_url}/v1/hive-heartbeat`

**Payload:** Node stats + all workers (monitored via cgroup v2)

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
      "worker_id": "worker:a1b2c3d4-e5f6-7890-abcd-ef1234567890",
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

**Queen responses:**
- `200 OK` - Telemetry acknowledged
- `400 Bad Request` - Invalid payload
- `503 Service Unavailable` - Queen overloaded (Hive retries)

---

## Worker Monitoring (No Worker Heartbeats)

**CRITICAL: Workers never send heartbeats. Hive monitors workers via OS.**

### **How Hive Monitors Workers**

**Method:** cgroup v2 tree inspection

**cgroup Structure:**
```
/sys/fs/cgroup/rbee.slice/
├── llm/
│   ├── 8080/
│   │   ├── cgroup.procs      # PIDs
│   │   ├── cpu.stat          # CPU usage
│   │   ├── memory.current    # RSS
│   │   └── io.stat           # I/O rates
│   └── 8081/
│       └── ...
├── vllm/
│   └── 8000/
│       └── ...
└── comfy/
    └── 8188/
        └── ...
```

### **Monitor Loop (Hive)**

```rust
// Hive monitor (runs every ~1s)
async fn collect_worker_telemetry() -> Vec<WorkerTelemetry> {
    let mut workers = vec![];
    
    // Enumerate cgroup tree
    for service in ["llm", "vllm", "comfy"] {
        for instance in enumerate_instances(&format!("rbee.slice/{}", service)) {
            let cgroup_path = format!("rbee.slice/{}/{}", service, instance);
            
            // Read OS stats (no worker cooperation)
            let pids = read_pids(&cgroup_path)?;
            let cpu_pct = read_cpu_stat(&cgroup_path)?;
            let rss_mb = read_memory_current(&cgroup_path)?;
            let (io_r, io_w) = read_io_stat(&cgroup_path)?;
            
            // Query GPU driver for VRAM (if worker uses GPU)
            let vram_mb = query_nvidia_smi_for_pids(&pids)?;
            
            workers.push(WorkerTelemetry {
                worker_id: generate_worker_id(service, instance),
                service,
                instance,
                cgroup: cgroup_path,
                pids,
                port: instance.parse()?,
                cpu_pct,
                rss_mb,
                vram_mb,
                io_r_mb_s: io_r,
                io_w_mb_s: io_w,
                uptime_s: calculate_uptime(&pids[0]),
                state: infer_state(cpu_pct, rss_mb),
                ...
            });
        }
    }
    
    workers
}
```

### **Why No Worker Heartbeats?**

1. **Workers can't lie** - OS reports ground truth
2. **Single telemetry path** - Hive → Queen (not worker → Queen)
3. **No worker cooperation** - Works even if worker crashes/hangs
4. **Efficient** - 1 heartbeat per node (not per worker)
5. **Consistent** - Same monitoring for all worker types

---

## State Machine

### Hive States

```
┌─────────────────┐
│     STARTUP     │ (Initial state - Hive just started)
└────────┬────────┘
         │
         ├──────────────────────────────────────────┐
         │                                          │
         │ Path 1: Hive has Queen URL configured   │ Path 2: Wait for Queen discovery
         │                                          │
         ▼                                          ▼
┌─────────────────┐                        ┌─────────────────┐
│ DISCOVERY_PUSH  │                        │ DISCOVERY_WAIT  │
│ (Exponential    │                        │ (Waiting for    │
│  backoff: 5     │                        │  GET /caps)     │
│  tries)         │                        │                 │
└────────┬────────┘                        └────────┬────────┘
         │                                          │
         │ Attempt 1: 0s  → 404                     │
         │ Attempt 2: 2s  → 404                     │ Queen sends GET /capabilities?queen_url=...
         │ Attempt 3: 4s  → 404                     │
         │ Attempt 4: 8s  → 200 OK!                 │
         │ Attempt 5: 16s → STOP                    │
         │                                          │
         └──────────────┬───────────────────────────┘
                        │
                        │ First 200 OK received
                        │
                        ▼
                ┌─────────────────┐
                │   TELEMETRY     │ (Sending telemetry every ~1s)
                └─────────────────┘
                         │
                         │ Monitor loop runs continuously:
                         │ - Poll cgroups for worker stats
                         │ - Read node stats (CPU, RAM, GPU)
                         │ - Send telemetry to Queen
                         │
                         └──────► Loop forever
```

**State Descriptions:**

- **STARTUP** - Hive just started, decides which discovery path
- **DISCOVERY_PUSH** - Hive sends telemetry with exponential backoff (5 tries: 0s, 2s, 4s, 8s, 16s)
- **DISCOVERY_WAIT** - Hive waits for Queen to send GET /capabilities?queen_url=...
- **TELEMETRY** - Normal operation, sending telemetry every ~1s (EMA-smoothed)

---

## Telemetry Payload Specification

### Contract (Authoritative)

```rust
// bin/97_contracts/hive-contract/src/telemetry.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveTelemetry {
    pub hive_id: String,              // "hive:UUID"
    pub ts: DateTime<Utc>,            // ISO 8601
    pub node: NodeStats,
    pub workers: Vec<WorkerTelemetry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStats {
    pub cpu_pct: f32,                 // 0-100 per core (can exceed 100)
    pub ram_used_mb: u64,
    pub ram_total_mb: u64,
    pub gpus: Vec<GpuStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStats {
    pub id: String,                   // "GPU-0", "GPU-1", etc.
    pub util_pct: f32,                // 0-100
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub temp_c: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerTelemetry {
    pub worker_id: String,            // "worker:UUID"
    pub service: String,              // "llm", "vllm", "comfy"
    pub instance: String,             // "8080", "8081", etc. (port)
    pub cgroup: String,               // "rbee.slice/llm/8080"
    pub pids: Vec<u32>,
    pub port: u16,
    pub model: Option<String>,        // "llama-3.2-1b"
    pub gpu: Option<String>,          // "GPU-0" or null for CPU
    pub cpu_pct: f32,                 // 0-100 per core
    pub rss_mb: u64,                  // Resident set size
    pub vram_mb: u64,                 // 0 if CPU worker
    pub io_r_mb_s: f32,               // Read MB/s
    pub io_w_mb_s: f32,               // Write MB/s
    pub uptime_s: u64,
    pub state: WorkerState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkerState {
    Starting,
    Ready,
    Busy,
    Error,
}
```

---

## Health Determination

**Queen derives health from telemetry freshness. No explicit health endpoints.**

### **Health States**

| State | Condition | Action |
|-------|-----------|--------|
| **Healthy** | Last telemetry < 3× interval (< 3s) | Normal operation |
| **Degraded** | Last telemetry 3-10× interval (3-10s) | Warning |
| **Down** | No telemetry for 10× interval (> 10s) | Mark as offline |

### **Calculation**

```rust
// Queen health checker (runs every 1s)
async fn check_hive_health(hive: &HiveEntry) -> HealthStatus {
    let now = Utc::now();
    let last_seen = hive.last_telemetry_ts;
    let elapsed = now - last_seen;
    
    const INTERVAL: Duration = Duration::from_secs(1);  // ~1s telemetry
    
    if elapsed < INTERVAL * 3 {
        HealthStatus::Healthy
    } else if elapsed < INTERVAL * 10 {
        HealthStatus::Degraded
    } else {
        HealthStatus::Down
    }
}
```

### **Worker Health**

Workers are monitored via **Hive telemetry only**. No separate worker health checks.

```rust
// Worker health derived from telemetry presence
fn check_worker_health(worker: &WorkerTelemetry, now: DateTime<Utc>) -> HealthStatus {
    // If worker appears in telemetry, it's alive
    // If worker disappears from telemetry (Hive removes it), it's down
    
    match worker.state {
        WorkerState::Ready | WorkerState::Busy => HealthStatus::Healthy,
        WorkerState::Starting => HealthStatus::Degraded,
        WorkerState::Error => HealthStatus::Down,
    }
}
```

---

## Implementation Details

### Hive Telemetry Manager

**File:** `bin/20_rbee_hive/src/telemetry.rs`

```rust
pub struct TelemetryManager {
    queen_url: Arc<RwLock<Option<String>>>,
    hive_id: String,
    cgroup_monitor: Arc<CgroupMonitor>,
}

impl TelemetryManager {
    /// Called when Queen sends GET /capabilities?queen_url=...
    pub async fn discover(&self, queen_url: String) {
        *self.queen_url.write().await = Some(queen_url.clone());
        self.start_telemetry_task(queen_url).await;
    }
    
    async fn start_telemetry_task(&self, queen_url: String) {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            
            loop {
                interval.tick().await;
                
                // Collect node stats
                let node = collect_node_stats().await;
                
                // Enumerate cgroup tree for workers
                let workers = self.cgroup_monitor.collect_workers().await;
                
                // Build telemetry
                let telemetry = HiveTelemetry {
                    hive_id: self.hive_id.clone(),
                    ts: Utc::now(),
                    node,
                    workers,
                };
                
                // Send to Queen
                if let Err(e) = send_telemetry(&queen_url, &telemetry).await {
                    tracing::warn!("Failed to send telemetry: {}", e);
                }
            }
        });
    }
}
```

---

### Queen Implementation

**File:** `bin/10_queen_rbee/src/discovery.rs`

```rust
pub struct HiveDiscovery {
    ssh_config: SshConfig,
    hive_registry: Arc<HiveRegistry>,
}

impl HiveDiscovery {
    /// Called on Queen startup
    pub async fn discover_all_hives(&self) -> Result<()> {
        let queen_url = "http://localhost:7833"; // TODO: Make configurable
        
        for hive_config in &self.ssh_config.hives {
            match self.discover_hive(hive_config, queen_url).await {
                Ok(capabilities) => {
                    tracing::info!("Discovered hive: {}", hive_config.alias);
                    
                    // Store capabilities in registry
                    self.hive_registry.register_hive(
                        hive_config.alias.clone(),
                        capabilities,
                    );
                }
                Err(e) => {
                    tracing::warn!("Failed to discover hive {}: {}", hive_config.alias, e);
                }
            }
        }
        
        Ok(())
    }
    
    async fn discover_hive(
        &self,
        hive_config: &HiveConfig,
        queen_url: &str,
    ) -> Result<HiveCapabilities> {
        let url = format!(
            "http://{}:{}/capabilities?queen_url={}",
            hive_config.hostname,
            hive_config.port,
            urlencoding::encode(queen_url)
        );
        
        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            .timeout(Duration::from_secs(10))
            .send()
            .await?;
        
        if !response.status().is_success() {
            anyhow::bail!("Failed to fetch capabilities: {}", response.status());
        }
        
        let capabilities: HiveCapabilities = response.json().await?;
        Ok(capabilities)
    }
}
```

**File:** `bin/10_queen_rbee/src/http/heartbeat.rs`

```rust
pub async fn handle_hive_heartbeat(
    State(state): State<HeartbeatState>,
    Json(heartbeat): Json<HiveHeartbeat>,
) -> Result<StatusCode, (StatusCode, String)> {
    // Update monitor data in registry
    state.hive_registry.update_monitor_data(
        &heartbeat.hive_id,
        heartbeat.monitor_data,
    );
    
    // Handle capability changes if present
    if let Some(changes) = heartbeat.capability_changes {
        state.hive_registry.apply_capability_changes(
            &heartbeat.hive_id,
            changes,
        );
    }
    
    // Broadcast event for SSE streaming
    let event = HeartbeatEvent::Hive {
        hive_id: heartbeat.hive_id.clone(),
        timestamp: heartbeat.timestamp.to_rfc3339(),
    };
    let _ = state.event_tx.send(event);
    
    Ok(StatusCode::NO_CONTENT)
}
```

---

## Capabilities Endpoint Enhancement

**File:** `bin/20_rbee_hive/src/main.rs`

```rust
async fn get_capabilities(
    Query(params): Query<CapabilitiesQuery>,
) -> Json<CapabilitiesResponse> {
    // Extract queen_url from query parameter
    if let Some(queen_url) = params.queen_url {
        // Store queen_url and start heartbeat task
        HEARTBEAT_MANAGER.discover(queen_url).await;
    }
    
    // Return capabilities (devices, models, workers)
    let devices = detect_devices().await;
    let models = list_models().await;
    let workers = list_workers().await;
    
    Json(CapabilitiesResponse {
        devices,
        models,
        workers,
    })
}

#[derive(Debug, Deserialize)]
struct CapabilitiesQuery {
    queen_url: Option<String>,
}
```

---

## Prerequisites: SSH Config Shared Crate

**Current Issue:** SSH config parsing logic is duplicated in rbee-keeper.

**Current Location:** `bin/00_rbee_keeper/src/tauri_commands.rs::ssh_list()`

**Required Action:** Extract SSH config parsing to shared crate.

**Proposed Crate:** `bin/99_shared_crates/ssh-config-parser`

**API:**
```rust
pub struct SshTarget {
    pub host: String,              // Alias from SSH config
    pub hostname: String,          // Actual hostname/IP
    pub user: String,
    pub port: u16,
}

pub fn parse_ssh_config(path: &Path) -> Result<Vec<SshTarget>>;
pub fn get_default_ssh_config_path() -> PathBuf;  // ~/.ssh/config
```

**Usage in Queen:**
```rust
use ssh_config_parser::{parse_ssh_config, get_default_ssh_config_path};

let ssh_config_path = get_default_ssh_config_path();
let targets = parse_ssh_config(&ssh_config_path)?;

for target in targets {
    // Fetch capabilities from each hive
    let url = format!("http://{}:{}/capabilities?queen_url={}", 
        target.hostname, 7835, queen_url);
    // ...
}
```

**Priority:** HIGH - Required before implementing Queen discovery.

---

## Summary

### Key Design Decisions

1. **Bidirectional discovery** - Both Queen and Hive can initiate
2. **Exponential backoff** - Hive retries 5 times with increasing delays
3. **No dependency order** - Queen and Hive can start in any order
4. **Shared protocol** - Same mechanism for hives and workers
5. **Monitor data in heartbeat** - No separate monitoring endpoint
6. **Capability changes in heartbeat** - Incremental updates, not full refresh

### Benefits

1. **No dependency order** - Queen and Hive can start independently
2. **Explicit control** - Queen decides which hives to monitor
3. **Resilient** - Queen can rediscover after restart
4. **Efficient** - Monitor data piggybacks on heartbeat
5. **Scalable** - No polling, push-based updates

### Implementation Priority

**Phase 1 (Critical):**
- [ ] Enhance `/capabilities` endpoint with `queen_url` parameter
- [ ] Implement `HeartbeatManager` in hive
- [ ] Implement `HiveDiscovery` in Queen
- [ ] Update `HiveHeartbeat` contract with monitor data

**Phase 2 (Important):**
- [ ] Implement capability change tracking
- [ ] Add retry logic with backoff
- [ ] Handle Queen unreachable scenario

**Phase 3 (Nice-to-have):**
- [ ] Heartbeat metrics and monitoring
- [ ] Configurable heartbeat interval
- [ ] Heartbeat compression for large payloads

---

**This document is the canonical specification for heartbeat architecture. All other documents must refer to this specification.**
