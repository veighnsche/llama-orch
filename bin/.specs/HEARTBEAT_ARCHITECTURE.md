# Heartbeat Architecture

**Date:** Oct 30, 2025  
**Status:** CANONICAL SPECIFICATION  
**Purpose:** Define heartbeat protocol for Queen ↔ Hive ↔ Worker communication

---

## Table of Contents

1. [Core Principle](#core-principle)
2. [Heartbeat Discovery Protocol](#heartbeat-discovery-protocol)
3. [Hive Heartbeat Flow](#hive-heartbeat-flow)
4. [Worker Heartbeat Flow](#worker-heartbeat-flow)
5. [Heartbeat Payload Specification](#heartbeat-payload-specification)
6. [State Machine](#state-machine)
7. [Implementation Details](#implementation-details)

---

## Core Principle

**Bidirectional startup discovery with exponential backoff.**

- **Both Queen and Hive can start independently** - No dependency order required
- **Queen discovers hives on startup** - Fetches capabilities from SSH config
- **Hive discovers Queen on startup** - Sends heartbeats with exponential backoff
- **Resilient to race conditions** - Handles simultaneous startup gracefully

**Startup Scenarios:**

1. **Hive starts before Queen** - Hive sends 5 exponential backoff heartbeats with capabilities until Queen responds
2. **Queen starts before Hive** - Queen fetches capabilities, triggering Hive to send heartbeats
3. **Both start simultaneously** - Both mechanisms work in parallel, first success wins

**Why this design?**

1. **No dependency order** - Queen and Hive can start in any order
2. **Resilient** - Handles network delays, race conditions, restarts
3. **Efficient** - Exponential backoff prevents flooding
4. **Simple** - Clear state transitions, easy to debug

---

## Heartbeat Discovery Protocol

### Scenario 1: Queen Starts First (Pull-based Discovery)

**When Queen starts:**

1. Queen waits 5 seconds (allow services to stabilize)
2. Queen reads SSH config (list of hive hostnames/IPs)
   - **Note:** SSH config parsing logic needs to be moved to shared crate
   - Currently in: `bin/00_rbee_keeper/src/tauri_commands.rs::ssh_list()`
3. Queen sends parallel GET requests to all configured hives:
   ```
   GET http://{hive_hostname}:7835/capabilities?queen_url=http://queen-host:7833
   ```
4. Hive receives request, extracts `queen_url`, stores it
5. Hive responds with capabilities (devices, models, workers)
6. Queen stores capabilities in HiveRegistry
7. **Trigger:** Hive starts sending heartbeats every 30s

**Result:** All online hives discovered and monitored.

---

### Scenario 2: Hive Starts First (Push-based Discovery with Exponential Backoff)

**When Hive starts:**

1. Hive has Queen URL configured (via CLI arg or config file)
2. Hive sends heartbeat with **full capabilities** (not just monitor data)
3. Hive uses **exponential backoff** for first 5 attempts:
   - Attempt 1: Immediate (0s)
   - Attempt 2: 2s delay
   - Attempt 3: 4s delay
   - Attempt 4: 8s delay
   - Attempt 5: 16s delay
4. If Queen responds with `200 OK`:
   - Hive transitions to normal heartbeat mode (monitor data every 30s)
   - Capabilities only sent when changed
5. If all 5 attempts fail:
   - Hive stops sending heartbeats
   - Waits for Queen to fetch capabilities (Scenario 1)

**Heartbeat payload during discovery:**
```json
{
  "hive_id": "localhost",
  "timestamp": "2025-10-30T15:13:00Z",
  "capabilities": {
    "devices": [...],
    "models": [...],
    "workers": [...]
  },
  "monitor_data": {
    "cpu_usage_percent": 45.2,
    "ram_used_gb": 12.5,
    "ram_total_gb": 64.0,
    "uptime_seconds": 10,
    "devices": [...]
  }
}
```

**Queen response:**
- `200 OK` - Capabilities received, start normal heartbeats
- `404 Not Found` - Queen not ready, retry with backoff
- `503 Service Unavailable` - Queen overloaded, retry with backoff

**Result:** Hive discovered by Queen, starts normal monitoring.

---

### Scenario 3: Both Start Simultaneously (Race Condition Handling)

**When both start at the same time:**

1. **Hive side:**
   - Starts exponential backoff heartbeats (Scenario 2)
   - Attempt 1 (0s): Queen not ready yet → 404
   - Attempt 2 (2s): Queen not ready yet → 404
   - Attempt 3 (4s): Queen not ready yet → 404

2. **Queen side (after 5s wait):**
   - Fetches capabilities from all hives (Scenario 1)
   - Hive responds with capabilities
   - Queen stores in registry

3. **Hive side (attempt 4 at 8s):**
   - Sends heartbeat → Queen responds 200 OK
   - Hive transitions to normal heartbeat mode

**Result:** Both mechanisms work in parallel, first success wins. No duplicate work.

---

### Phase 2: Hive Heartbeat (Monitoring)

**After discovery, hive sends heartbeats every 30 seconds:**

```
POST http://{queen_url}/v1/hive-heartbeat
{
  "hive_id": "localhost",
  "timestamp": "2025-10-30T14:52:00Z",
  "monitor_data": {
    "cpu_usage_percent": 45.2,
    "ram_used_gb": 12.5,
    "ram_total_gb": 64.0,
    "uptime_seconds": 86400,
    "devices": [
      {
        "device_id": "GPU-0",
        "vram_used_gb": 8.2,
        "vram_total_gb": 24.0,
        "temperature_celsius": 65.0
      }
    ]
  },
  "capability_changes": null  // Only set when capabilities change
}
```

**Queen responds:**
- `204 No Content` - Heartbeat acknowledged
- `400 Bad Request` - Invalid payload
- `503 Service Unavailable` - Queen overloaded (hive should retry)

---

### Phase 3: Capability Changes

**When hive capabilities change (model downloaded, worker installed):**

```
POST http://{queen_url}/v1/hive-heartbeat
{
  "hive_id": "localhost",
  "timestamp": "2025-10-30T14:52:00Z",
  "monitor_data": { ... },
  "capability_changes": {
    "models": [
      {
        "action": "added",
        "model_id": "llama-3.2-1b",
        "size_gb": 2.5
      }
    ],
    "workers": [
      {
        "action": "removed",
        "worker_type": "cuda",
        "version": "0.1.0"
      }
    ]
  }
}
```

**Queen responds:**
- `204 No Content` - Changes acknowledged, registry updated

---

### Phase 4: Worker Discovery

**Workers follow the same protocol:**

1. Hive spawns worker process
2. Hive passes Queen URL to worker via CLI arg:
   ```bash
   ./llm-worker-rbee --queen-url http://queen-host:7833 ...
   ```
3. Worker starts, sends initial registration:
   ```
   POST http://{queen_url}/v1/worker-heartbeat
   {
     "worker_id": "worker-cuda-9001",
     "hive_id": "localhost",
     "timestamp": "2025-10-30T14:52:00Z",
     "status": "starting",
     "model": "llama-3.2-1b",
     "device": "GPU-0",
     "port": 9001
   }
   ```
4. Queen stores worker in WorkerRegistry
5. Worker continues sending heartbeats every 30 seconds

**Note:** Workers receive Queen URL from hive at spawn time, not via discovery.

---

## Hive Heartbeat Flow

### State Machine

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
│  attempts)      │                        │                 │
└────────┬────────┘                        └────────┬────────┘
         │                                          │
         │ Attempt 1: 0s  → 404                     │
         │ Attempt 2: 2s  → 404                     │ Queen sends GET /capabilities?queen_url=...
         │ Attempt 3: 4s  → 404                     │
         │ Attempt 4: 8s  → 200 OK!                 │
         │ Attempt 5: 16s                           │
         │                                          │
         └──────────────┬───────────────────────────┘
                        │
                        ▼
                ┌─────────────────┐
                │   DISCOVERED    │ (Has Queen URL, Queen has capabilities)
                └────────┬────────┘
                         │
                         │ Send heartbeat every 30s
                         │
                         ▼
                ┌─────────────────┐
                │   HEARTBEATING  │ (Sending monitor data)
                └────────┬────────┘
                         │
                         │ Capabilities change (model added/removed)
                         │
                         ▼
                ┌─────────────────┐
                │ CAPABILITY_SYNC │ (Send capability_changes in next heartbeat)
                └────────┬────────┘
                         │
                         │ Queen responds 200 OK
                         │
                         └──────► Back to HEARTBEATING
```

**State Descriptions:**

- **STARTUP** - Hive just started, decides which discovery path to take
- **DISCOVERY_PUSH** - Hive actively pushes heartbeats with exponential backoff (5 attempts)
- **DISCOVERY_WAIT** - Hive waits for Queen to fetch capabilities via GET request
- **DISCOVERED** - Queen knows about Hive, Hive knows about Queen
- **HEARTBEATING** - Normal operation, sending monitor data every 30s
- **CAPABILITY_SYNC** - Temporary state when capabilities changed, next heartbeat includes changes

---

### Heartbeat Payload Types

**Type 1: Monitor Data (Normal)**
```json
{
  "hive_id": "localhost",
  "timestamp": "2025-10-30T14:52:00Z",
  "monitor_data": {
    "cpu_usage_percent": 45.2,
    "ram_used_gb": 12.5,
    "ram_total_gb": 64.0,
    "uptime_seconds": 86400,
    "devices": [...]
  },
  "capability_changes": null
}
```

**Type 2: Capability Changes**
```json
{
  "hive_id": "localhost",
  "timestamp": "2025-10-30T14:52:00Z",
  "monitor_data": { ... },
  "capability_changes": {
    "models": [
      { "action": "added", "model_id": "...", "size_gb": 2.5 }
    ],
    "workers": []
  }
}
```

---

## Worker Heartbeat Flow

### State Machine

```
┌─────────────────┐
│    SPAWNED      │ (Worker process started by hive)
└────────┬────────┘
         │
         │ Hive passes --queen-url flag
         │
         ▼
┌─────────────────┐
│   REGISTERED    │ (Send initial heartbeat with status=starting)
└────────┬────────┘
         │
         │ Model loaded, ready to serve
         │
         ▼
┌─────────────────┐
│     READY       │ (Send heartbeat every 30s with status=ready)
└────────┬────────┘
         │
         │ Receive inference request
         │
         ▼
┌─────────────────┐
│      BUSY       │ (Send heartbeat with status=busy)
└────────┬────────┘
         │
         │ Inference complete
         │
         └──────► Back to READY
```

---

### Worker Heartbeat Payload

```json
{
  "worker_id": "worker-cuda-9001",
  "hive_id": "localhost",
  "timestamp": "2025-10-30T14:52:00Z",
  "status": "ready",  // "starting", "ready", "busy", "error"
  "model": "llama-3.2-1b",
  "device": "GPU-0",
  "port": 9001,
  "requests_served": 42,
  "uptime_seconds": 3600
}
```

---

## Heartbeat Payload Specification

### HiveHeartbeat Contract

```rust
// bin/97_contracts/hive-contract/src/heartbeat.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveHeartbeat {
    pub hive_id: String,
    pub timestamp: DateTime<Utc>,
    pub monitor_data: MonitorData,
    pub capability_changes: Option<CapabilityChanges>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorData {
    pub cpu_usage_percent: f32,
    pub ram_used_gb: f32,
    pub ram_total_gb: f32,
    pub uptime_seconds: u64,
    pub devices: Vec<DeviceMonitorData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMonitorData {
    pub device_id: String,
    pub vram_used_gb: f32,
    pub vram_total_gb: f32,
    pub temperature_celsius: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityChanges {
    pub models: Vec<ModelChange>,
    pub workers: Vec<WorkerChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelChange {
    pub action: ChangeAction,  // "added", "removed"
    pub model_id: String,
    pub size_gb: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerChange {
    pub action: ChangeAction,
    pub worker_type: String,  // "cpu", "cuda", "metal"
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChangeAction {
    Added,
    Removed,
}
```

---

### WorkerHeartbeat Contract

```rust
// bin/97_contracts/worker-contract/src/heartbeat.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeat {
    pub worker_id: String,
    pub hive_id: String,
    pub timestamp: DateTime<Utc>,
    pub status: WorkerStatus,
    pub model: String,
    pub device: String,
    pub port: u16,
    pub requests_served: u64,
    pub uptime_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkerStatus {
    Starting,
    Ready,
    Busy,
    Error,
}
```

---

## State Machine

### Hive State Transitions

| Current State | Event | Next State | Action |
|--------------|-------|------------|--------|
| NOT_DISCOVERED | Queen sends GET /capabilities?queen_url=X | DISCOVERED | Store queen_url, start heartbeat task |
| DISCOVERED | Heartbeat task starts | HEARTBEATING | Send first heartbeat with monitor data |
| HEARTBEATING | Model downloaded | CAPABILITY_SYNC | Set capability_changes in next heartbeat |
| CAPABILITY_SYNC | Queen responds 204 | HEARTBEATING | Clear capability_changes |
| HEARTBEATING | Queen unreachable (5 failures) | DISCOVERED | Stop heartbeat task, wait for rediscovery |

---

### Worker State Transitions

| Current State | Event | Next State | Action |
|--------------|-------|------------|--------|
| SPAWNED | Hive passes --queen-url | REGISTERED | Send initial heartbeat (status=starting) |
| REGISTERED | Model loaded | READY | Send heartbeat (status=ready) |
| READY | Inference request received | BUSY | Send heartbeat (status=busy) |
| BUSY | Inference complete | READY | Send heartbeat (status=ready) |
| READY | Shutdown signal | STOPPED | Send final heartbeat, exit |

---

## Implementation Details

### Hive Implementation

**File:** `bin/20_rbee_hive/src/heartbeat.rs`

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct HeartbeatManager {
    queen_url: Arc<RwLock<Option<String>>>,
    hive_id: String,
    monitor: Arc<dyn SystemMonitor>,
    capability_tracker: Arc<CapabilityTracker>,
}

impl HeartbeatManager {
    pub fn new(hive_id: String) -> Self {
        Self {
            queen_url: Arc::new(RwLock::new(None)),
            hive_id,
            monitor: Arc::new(SystemMonitorImpl::new()),
            capability_tracker: Arc::new(CapabilityTracker::new()),
        }
    }
    
    /// Called when Queen sends GET /capabilities?queen_url=...
    pub async fn discover(&self, queen_url: String) {
        let mut url = self.queen_url.write().await;
        *url = Some(queen_url.clone());
        
        // Start heartbeat task
        self.start_heartbeat_task(queen_url).await;
    }
    
    async fn start_heartbeat_task(&self, queen_url: String) {
        let hive_id = self.hive_id.clone();
        let monitor = self.monitor.clone();
        let capability_tracker = self.capability_tracker.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Collect monitor data
                let monitor_data = monitor.collect().await;
                
                // Check for capability changes
                let capability_changes = capability_tracker.get_changes().await;
                
                // Build heartbeat
                let heartbeat = HiveHeartbeat {
                    hive_id: hive_id.clone(),
                    timestamp: Utc::now(),
                    monitor_data,
                    capability_changes,
                };
                
                // Send to Queen
                match send_heartbeat(&queen_url, &heartbeat).await {
                    Ok(_) => {
                        // Clear capability changes after successful send
                        if heartbeat.capability_changes.is_some() {
                            capability_tracker.clear_changes().await;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to send heartbeat: {}", e);
                        // TODO: Implement retry logic with backoff
                    }
                }
            }
        });
    }
}

async fn send_heartbeat(queen_url: &str, heartbeat: &HiveHeartbeat) -> Result<()> {
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/v1/hive-heartbeat", queen_url))
        .json(heartbeat)
        .send()
        .await?;
    
    if response.status() == StatusCode::NO_CONTENT {
        Ok(())
    } else {
        Err(anyhow::anyhow!("Unexpected response: {}", response.status()))
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
