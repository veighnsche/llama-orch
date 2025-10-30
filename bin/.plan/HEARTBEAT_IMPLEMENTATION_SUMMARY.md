# Heartbeat Architecture Implementation Summary

**Date:** Oct 30, 2025  
**Status:** SPECIFICATION COMPLETE  
**Canonical Spec:** `/home/vince/Projects/llama-orch/bin/.specs/HEARTBEAT_ARCHITECTURE.md`

---

## What Changed (Rule Zero Applied)

### ❌ Deprecated (Removed)

1. **Old idea:** Hive sends blind heartbeats every 30s, even if Queen is offline
   - **Problem:** Wasted resources, no way to know if Queen exists
   
2. **Old idea:** Hive caches capabilities when Queen is offline (400 response)
   - **Problem:** Complex caching logic, unclear cache invalidation
   
3. **Old idea:** Queen fetches capabilities on first heartbeat
   - **Problem:** Race condition, Queen doesn't know hive exists until heartbeat

4. **Document deleted:** `bin/.plan/QUEEN_HIVE_COMMUNICATION.md`
   - **Reason:** Contained outdated architecture

5. **Document deprecated:** `bin/20_rbee_hive/DEVICE_MONITORING_ANALYSIS.md`
   - **Reason:** Heartbeat section is outdated
   - **Status:** Marked as deprecated with pointer to new spec

---

## ✅ New Architecture (Canonical)

### Core Principle

**Pull-based discovery, push-based monitoring.**

```
Queen initiates discovery → Hive responds and starts heartbeats
```

**No blind heartbeats. No wasted resources.**

---

### Discovery Flow

```
1. Queen starts
   ↓
2. Queen reads SSH config (list of hive hostnames)
   ↓
3. Queen sends GET /capabilities?queen_url=http://queen:7833 to each hive
   ↓
4. Hive receives request, extracts queen_url
   ↓
5. Hive starts heartbeat task (send every 30s)
   ↓
6. Hive responds with capabilities (devices, models, workers)
   ↓
7. Queen stores capabilities in HiveRegistry
```

**Result:** All configured hives discovered and monitored.

---

### Heartbeat Payload

**Type 1: Normal Heartbeat (Monitor Data)**
```json
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
      { "action": "added", "model_id": "llama-3.2-1b", "size_gb": 2.5 }
    ],
    "workers": [
      { "action": "removed", "worker_type": "cuda", "version": "0.1.0" }
    ]
  }
}
```

**Queen Response:**
- `204 No Content` - Heartbeat acknowledged
- `400 Bad Request` - Invalid payload
- `503 Service Unavailable` - Queen overloaded

---

### Worker Heartbeat (Same Protocol)

**Workers follow the same discovery pattern:**

1. Hive spawns worker with `--queen-url` flag
2. Worker sends initial heartbeat (status=starting)
3. Worker continues heartbeat every 30s
4. Queen tracks worker in WorkerRegistry

**Worker Payload:**
```json
{
  "worker_id": "worker-cuda-9001",
  "hive_id": "localhost",
  "timestamp": "2025-10-30T14:52:00Z",
  "status": "ready",
  "model": "llama-3.2-1b",
  "device": "GPU-0",
  "port": 9001,
  "requests_served": 42,
  "uptime_seconds": 3600
}
```

---

## Key Benefits

### 1. No Wasted Resources
- Hives don't send heartbeats to non-existent Queen
- No blind network calls
- No complex caching logic

### 2. Explicit Control
- Queen decides which hives to monitor
- Queen initiates discovery
- Clear ownership

### 3. Resilient
- Queen can rediscover hives after restart
- Hives automatically reconnect when Queen comes online
- No state synchronization issues

### 4. Efficient
- Monitor data piggybacks on heartbeat
- No separate monitoring endpoint
- Capability changes sent incrementally

### 5. Shared Protocol
- Same discovery mechanism for hives and workers
- Consistent architecture
- Easy to understand

---

## Implementation Checklist

### Phase 1: Contracts (Priority 1)

**File:** `bin/97_contracts/hive-contract/src/heartbeat.rs`

- [ ] Create `HiveHeartbeat` struct
  - `hive_id: String`
  - `timestamp: DateTime<Utc>`
  - `monitor_data: MonitorData`
  - `capability_changes: Option<CapabilityChanges>`

- [ ] Create `MonitorData` struct
  - `cpu_usage_percent: f32`
  - `ram_used_gb: f32`
  - `ram_total_gb: f32`
  - `uptime_seconds: u64`
  - `devices: Vec<DeviceMonitorData>`

- [ ] Create `CapabilityChanges` struct
  - `models: Vec<ModelChange>`
  - `workers: Vec<WorkerChange>`

- [ ] Create `ModelChange` and `WorkerChange` structs

**File:** `bin/97_contracts/worker-contract/src/heartbeat.rs`

- [ ] Create `WorkerHeartbeat` struct
- [ ] Create `WorkerStatus` enum (Starting, Ready, Busy, Error)

---

### Phase 2: Hive Implementation (Priority 1)

**File:** `bin/20_rbee_hive/src/heartbeat.rs`

- [ ] Create `HeartbeatManager` struct
  - `queen_url: Arc<RwLock<Option<String>>>`
  - `hive_id: String`
  - `monitor: Arc<dyn SystemMonitor>`
  - `capability_tracker: Arc<CapabilityTracker>`

- [ ] Implement `discover()` method
  - Store queen_url
  - Start heartbeat task

- [ ] Implement `start_heartbeat_task()`
  - Collect monitor data every 30s
  - Check for capability changes
  - Send heartbeat to Queen
  - Handle 204/400/503 responses

**File:** `bin/20_rbee_hive/src/main.rs`

- [ ] Enhance `/capabilities` endpoint
  - Accept `queen_url` query parameter
  - Call `HeartbeatManager::discover(queen_url)`
  - Return capabilities (devices, models, workers)

---

### Phase 3: Monitor Implementation (Priority 1)

**File:** `bin/25_rbee_hive_crates/monitor/src/lib.rs`

- [ ] Implement `SystemMonitor` trait
  - `collect() -> MonitorData`

- [ ] Implement `SystemMonitorImpl`
  - Collect CPU usage (via `sysinfo` crate)
  - Collect RAM usage
  - Collect uptime
  - Collect device stats (VRAM, temperature)

**File:** `bin/25_rbee_hive_crates/monitor/src/device_monitor.rs`

- [ ] Implement device monitoring
  - Query nvidia-smi for GPU stats
  - Parse VRAM usage, temperature
  - Return `Vec<DeviceMonitorData>`

---

### Phase 4: Capability Tracking (Priority 2)

**File:** `bin/20_rbee_hive/src/capability_tracker.rs`

- [ ] Create `CapabilityTracker` struct
  - Track model additions/removals
  - Track worker additions/removals
  - Return changes since last heartbeat

- [ ] Implement `track_model_added(model_id, size_gb)`
- [ ] Implement `track_model_removed(model_id)`
- [ ] Implement `track_worker_added(worker_type, version)`
- [ ] Implement `track_worker_removed(worker_type)`
- [ ] Implement `get_changes() -> Option<CapabilityChanges>`
- [ ] Implement `clear_changes()`

---

### Phase 5: Queen Implementation (Priority 1)

**File:** `bin/10_queen_rbee/src/discovery.rs`

- [ ] Create `HiveDiscovery` struct
  - `ssh_config: SshConfig`
  - `hive_registry: Arc<HiveRegistry>`

- [ ] Implement `discover_all_hives()`
  - Read SSH config
  - For each hive: send GET /capabilities?queen_url=...
  - Store capabilities in HiveRegistry

- [ ] Implement `discover_hive(hive_config, queen_url)`
  - Send GET request with timeout
  - Parse capabilities response
  - Return `HiveCapabilities`

**File:** `bin/10_queen_rbee/src/http/heartbeat.rs`

- [ ] Update `handle_hive_heartbeat()`
  - Extract monitor_data
  - Update HiveRegistry with monitor data
  - If capability_changes present: apply to registry
  - Return 204 No Content

**File:** `bin/10_queen_rbee/src/main.rs`

- [ ] Call `HiveDiscovery::discover_all_hives()` on startup

---

### Phase 6: Registry Enhancement (Priority 2)

**File:** `bin/15_queen_rbee_crates/hive-registry/src/registry.rs`

- [ ] Add `update_monitor_data(hive_id, monitor_data)`
- [ ] Add `apply_capability_changes(hive_id, changes)`
- [ ] Store capabilities in HiveInfo
- [ ] Store monitor data in HiveInfo

---

## Testing Strategy

### Unit Tests

1. **HeartbeatManager**
   - Test discovery flow
   - Test heartbeat task start
   - Test capability change tracking

2. **SystemMonitor**
   - Test CPU/RAM collection
   - Test device stats collection
   - Test error handling

3. **CapabilityTracker**
   - Test model tracking
   - Test worker tracking
   - Test change accumulation
   - Test change clearing

4. **HiveDiscovery**
   - Test discovery flow
   - Test timeout handling
   - Test error handling

### Integration Tests

1. **End-to-End Discovery**
   - Start Queen
   - Start Hive
   - Verify Queen discovers Hive
   - Verify heartbeats received

2. **Capability Changes**
   - Download model on Hive
   - Verify capability_changes in next heartbeat
   - Verify Queen registry updated

3. **Worker Discovery**
   - Spawn worker on Hive
   - Verify worker heartbeat received
   - Verify WorkerRegistry updated

---

## Migration Path

### Step 1: Implement Contracts
- Create heartbeat contracts
- No breaking changes yet

### Step 2: Implement Monitor Crate
- Implement system monitoring
- Can be tested independently

### Step 3: Enhance Hive
- Add HeartbeatManager
- Enhance /capabilities endpoint
- **Breaking change:** Hive now requires discovery

### Step 4: Enhance Queen
- Add HiveDiscovery
- Update heartbeat handler
- **Breaking change:** Queen must discover hives on startup

### Step 5: Update Existing Hives
- Restart hives to pick up new code
- Queen will discover them automatically

---

## Documents Updated

### ✅ Created
- `/home/vince/Projects/llama-orch/bin/.specs/HEARTBEAT_ARCHITECTURE.md` (canonical spec)
- `/home/vince/Projects/llama-orch/bin/.plan/HEARTBEAT_IMPLEMENTATION_SUMMARY.md` (this doc)

### ✅ Updated
- `/home/vince/Projects/llama-orch/bin/20_rbee_hive/HIVE_RESPONSIBILITIES.md` (references new spec)

### ✅ Deprecated
- `/home/vince/Projects/llama-orch/bin/20_rbee_hive/DEVICE_MONITORING_ANALYSIS.md` (marked deprecated)

### ✅ Deleted
- `/home/vince/Projects/llama-orch/bin/.plan/QUEEN_HIVE_COMMUNICATION.md` (outdated)

---

## Summary

**Rule Zero applied:** Deprecated ideas removed, single source of truth established.

**Canonical spec:** `/home/vince/Projects/llama-orch/bin/.specs/HEARTBEAT_ARCHITECTURE.md`

**Key changes:**
1. Pull-based discovery (Queen initiates)
2. No blind heartbeats (hives only send after discovery)
3. Monitor data in heartbeat payload
4. Capability changes sent incrementally
5. Shared protocol for hives and workers

**Implementation priority:**
1. Contracts (Phase 1)
2. Monitor crate (Phase 3)
3. Hive enhancement (Phase 2)
4. Queen enhancement (Phase 5)
5. Capability tracking (Phase 4)
6. Registry enhancement (Phase 6)

**All documents now reference the canonical spec. No entropy.**
