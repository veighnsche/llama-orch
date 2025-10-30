# Heartbeat Logic Consolidation Analysis

**Date:** Oct 30, 2025  
**Purpose:** Analyze current heartbeat logic and plan Rule Zero consolidation  
**Canonical Spec:** `/home/vince/Projects/llama-orch/bin/.specs/HEARTBEAT_ARCHITECTURE.md`

---

## Current Heartbeat Logic Inventory

### 📦 Existing Crates/Files

#### **1. Sender Logic (3 locations)**

**A. Hive Heartbeat Sender**
- **File:** `bin/20_rbee_hive/src/heartbeat.rs` (67 LOC)
- **Functions:**
  - `send_heartbeat_to_queen()` - Sends POST /v1/hive-heartbeat
  - `start_heartbeat_task()` - Spawns task, sends every 30s
- **Status:** ✅ IMPLEMENTED (simple, no exponential backoff)
- **Payload:** Static `HiveInfo` (no monitor data, no capabilities)

**B. Worker Heartbeat Sender**
- **File:** `bin/30_llm_worker_rbee/src/heartbeat.rs` (85 LOC)
- **Functions:**
  - `send_heartbeat_to_queen()` - TODO: Not implemented (commented out)
  - `start_heartbeat_task()` - Spawns task, sends every 30s
- **Status:** ⏳ STUB (HTTP POST commented out)
- **Payload:** Static `WorkerInfo`

**C. Queen Heartbeat Sender (Self-heartbeat)**
- **File:** `bin/10_queen_rbee/src/http/heartbeat_stream.rs`
- **Purpose:** Queen sends own heartbeat every 2.5s for UI
- **Status:** ✅ IMPLEMENTED
- **Payload:** Queen stats (workers_online, hives_online, etc.)

---

#### **2. Receiver Logic (1 location)**

**Queen Heartbeat Receiver**
- **File:** `bin/10_queen_rbee/src/http/heartbeat.rs` (166 LOC)
- **Endpoints:**
  - `POST /v1/worker-heartbeat` - Receives worker heartbeats
  - `POST /v1/hive-heartbeat` - Receives hive heartbeats
- **Functions:**
  - `handle_worker_heartbeat()` - Updates WorkerRegistry, broadcasts event
  - `handle_hive_heartbeat()` - Updates HiveRegistry, broadcasts event
- **Status:** ✅ IMPLEMENTED (simple, no capability handling)

---

#### **3. Registry Logic (1 shared crate)**

**Generic Heartbeat Registry**
- **Crate:** `bin/99_shared_crates/heartbeat-registry` (388 LOC)
- **Purpose:** Generic registry for tracking component state via heartbeats
- **Pattern:** `Component → Heartbeat → Registry → Query API`
- **Trait:** `HeartbeatItem` (id, info, is_recent, is_available)
- **Status:** ✅ IMPLEMENTED (generic, reusable)
- **Used by:** WorkerRegistry, HiveRegistry

---

#### **4. Contract Types (3 locations)**

**A. Worker Contract**
- **File:** `bin/97_contracts/worker-contract/src/heartbeat.rs` (49 matches)
- **Types:** `WorkerHeartbeat`, `WorkerInfo`
- **Status:** ✅ IMPLEMENTED

**B. Hive Contract**
- **File:** `bin/97_contracts/hive-contract/src/heartbeat.rs` (46 matches)
- **Types:** `HiveHeartbeat`, `HiveInfo`
- **Status:** ✅ IMPLEMENTED (missing monitor data, capabilities)

**C. Shared Contract**
- **File:** `bin/97_contracts/shared-contract/src/heartbeat.rs` (18 matches)
- **Types:** `HeartbeatTimestamp`, `OperationalStatus`, `HealthStatus`
- **Status:** ✅ IMPLEMENTED

---

## What's Missing (Per New Spec)

### ❌ **Not Implemented:**

1. **Exponential Backoff (Hive/Worker)**
   - Current: Simple 30s interval
   - Needed: 5 attempts with 0s, 2s, 4s, 8s, 16s delays

2. **Discovery Heartbeats (Hive)**
   - Current: Sends static HiveInfo
   - Needed: Send full capabilities + monitor data during discovery

3. **Monitor Data Collection**
   - Current: No monitor data
   - Needed: CPU%, RAM%, VRAM%, uptime, device stats

4. **Capability Changes Tracking**
   - Current: No capability tracking
   - Needed: Track model/worker additions/removals

5. **Queen Discovery (Pull-based)**
   - Current: No discovery mechanism
   - Needed: Queen fetches capabilities on startup

6. **Capabilities Endpoint Enhancement**
   - Current: Returns device list only
   - Needed: Accept `queen_url` parameter, trigger heartbeat task

7. **State Machine**
   - Current: No state tracking
   - Needed: STARTUP → DISCOVERY_PUSH/WAIT → DISCOVERED → HEARTBEATING

---

## Rule Zero Analysis: Can We Consolidate?

### ✅ **YES - Create Unified Heartbeat Crate**

**Proposed Crate:** `bin/99_shared_crates/heartbeat-client`

**Why Consolidate:**
1. **Eliminate duplication** - Hive and Worker have nearly identical sender logic
2. **Single source of truth** - Exponential backoff, state machine in one place
3. **Easier testing** - Test heartbeat logic once, not twice
4. **Consistent behavior** - Hive and Worker behave identically
5. **Shared discovery** - Same discovery protocol for both

**What to Consolidate:**

```
heartbeat-client/
├── src/
│   ├── lib.rs              # Public API
│   ├── sender.rs           # HeartbeatSender trait + impl
│   ├── exponential.rs      # Exponential backoff logic
│   ├── state_machine.rs    # Discovery state machine
│   ├── monitor.rs          # Monitor data collection (optional)
│   └── types.rs            # Common types
```

---

## Proposed Architecture

### **Unified Heartbeat Client API**

```rust
// bin/99_shared_crates/heartbeat-client/src/lib.rs

pub struct HeartbeatClient<T: HeartbeatPayload> {
    queen_url: Arc<RwLock<Option<String>>>,
    component_id: String,
    state: Arc<RwLock<HeartbeatState>>,
    payload_builder: Arc<dyn Fn() -> T>,
}

pub trait HeartbeatPayload: Serialize + Clone + Send + Sync {
    fn component_id(&self) -> &str;
    fn endpoint(&self) -> &str;  // "/v1/worker-heartbeat" or "/v1/hive-heartbeat"
}

impl<T: HeartbeatPayload> HeartbeatClient<T> {
    pub fn new(component_id: String, payload_builder: impl Fn() -> T + 'static) -> Self;
    
    /// Start discovery with exponential backoff (5 attempts)
    pub async fn start_discovery(&self, queen_url: String);
    
    /// Transition to normal heartbeat mode (every 30s)
    pub async fn start_heartbeating(&self);
    
    /// Send single heartbeat
    pub async fn send_heartbeat(&self) -> Result<HeartbeatResponse>;
    
    /// Get current state
    pub fn state(&self) -> HeartbeatState;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeartbeatState {
    Startup,
    DiscoveryPush,
    DiscoveryWait,
    Discovered,
    Heartbeating,
}

pub struct HeartbeatResponse {
    pub status_code: u16,
    pub success: bool,
}
```

---

### **Usage in Hive**

```rust
// bin/20_rbee_hive/src/heartbeat.rs (REWRITE)

use heartbeat_client::{HeartbeatClient, HeartbeatPayload};
use hive_contract::HiveHeartbeat;

impl HeartbeatPayload for HiveHeartbeat {
    fn component_id(&self) -> &str {
        &self.hive.id
    }
    
    fn endpoint(&self) -> &str {
        "/v1/hive-heartbeat"
    }
}

pub struct HiveHeartbeatManager {
    client: HeartbeatClient<HiveHeartbeat>,
}

impl HiveHeartbeatManager {
    pub fn new(hive_id: String) -> Self {
        let client = HeartbeatClient::new(hive_id.clone(), move || {
            // Build HiveHeartbeat with monitor data + capabilities
            let monitor_data = collect_monitor_data();
            let capabilities = get_capabilities();
            HiveHeartbeat {
                hive_id: hive_id.clone(),
                timestamp: Utc::now(),
                monitor_data,
                capabilities: Some(capabilities),
            }
        });
        
        Self { client }
    }
    
    /// Called when Queen sends GET /capabilities?queen_url=...
    pub async fn discover(&self, queen_url: String) {
        self.client.start_discovery(queen_url).await;
    }
    
    /// Called on Hive startup (if Queen URL configured)
    pub async fn start_with_queen_url(&self, queen_url: String) {
        self.client.start_discovery(queen_url).await;
    }
}
```

---

### **Usage in Worker**

```rust
// bin/30_llm_worker_rbee/src/heartbeat.rs (REWRITE)

use heartbeat_client::{HeartbeatClient, HeartbeatPayload};
use worker_contract::WorkerHeartbeat;

impl HeartbeatPayload for WorkerHeartbeat {
    fn component_id(&self) -> &str {
        &self.worker.id
    }
    
    fn endpoint(&self) -> &str {
        "/v1/worker-heartbeat"
    }
}

pub struct WorkerHeartbeatManager {
    client: HeartbeatClient<WorkerHeartbeat>,
}

impl WorkerHeartbeatManager {
    pub fn new(worker_id: String, worker_info: WorkerInfo) -> Self {
        let client = HeartbeatClient::new(worker_id.clone(), move || {
            // Build WorkerHeartbeat
            WorkerHeartbeat::new(worker_info.clone())
        });
        
        Self { client }
    }
    
    /// Called when worker spawns with --queen-url flag
    pub async fn start(&self, queen_url: String) {
        self.client.start_discovery(queen_url).await;
    }
}
```

---

### **Queen Receiver (Keep Separate)**

**File:** `bin/10_queen_rbee/src/http/heartbeat.rs`

**Why keep separate:**
- Queen is the receiver, not sender
- Already well-structured
- Needs to handle both worker and hive heartbeats
- Needs to update registries and broadcast events

**Changes needed:**
- Update `handle_hive_heartbeat()` to accept monitor data + capabilities
- Return `200 OK` instead of `204 No Content` (for discovery)
- Handle capability changes

---

## Rule Zero: What to Delete

### ❌ **DELETE (Outdated Logic)**

1. **`bin/20_rbee_hive/src/heartbeat.rs`** (67 LOC)
   - Replace with new `HiveHeartbeatManager` using `heartbeat-client`
   
2. **`bin/30_llm_worker_rbee/src/heartbeat.rs`** (85 LOC)
   - Replace with new `WorkerHeartbeatManager` using `heartbeat-client`

### ✅ **KEEP (Still Useful)**

1. **`bin/99_shared_crates/heartbeat-registry`** (388 LOC)
   - Generic registry pattern is solid
   - Used by WorkerRegistry and HiveRegistry
   - No changes needed

2. **`bin/10_queen_rbee/src/http/heartbeat.rs`** (166 LOC)
   - Receiver logic is separate concern
   - Needs updates, not deletion
   - Keep structure, enhance functionality

3. **Contract types** (worker-contract, hive-contract, shared-contract)
   - Need updates (add monitor data, capabilities)
   - Keep structure, enhance types

---

## Implementation Plan

### **Phase 1: Create Unified Heartbeat Client**

**Priority:** CRITICAL

**Tasks:**
- [ ] Create `bin/99_shared_crates/heartbeat-client` crate
- [ ] Implement `HeartbeatClient<T>` with generic payload
- [ ] Implement exponential backoff logic (5 attempts: 0s, 2s, 4s, 8s, 16s)
- [ ] Implement state machine (STARTUP → DISCOVERY → HEARTBEATING)
- [ ] Add tests for exponential backoff
- [ ] Add tests for state transitions

**Files to create:**
- `bin/99_shared_crates/heartbeat-client/Cargo.toml`
- `bin/99_shared_crates/heartbeat-client/src/lib.rs`
- `bin/99_shared_crates/heartbeat-client/src/sender.rs`
- `bin/99_shared_crates/heartbeat-client/src/exponential.rs`
- `bin/99_shared_crates/heartbeat-client/src/state_machine.rs`

---

### **Phase 2: Enhance Contracts**

**Priority:** CRITICAL

**Tasks:**
- [ ] Add `MonitorData` to `HiveHeartbeat`
- [ ] Add `Capabilities` to `HiveHeartbeat`
- [ ] Add `CapabilityChanges` to `HiveHeartbeat`
- [ ] Update `HiveInfo` to include capabilities
- [ ] Add monitor data collection types

**Files to modify:**
- `bin/97_contracts/hive-contract/src/heartbeat.rs`
- `bin/97_contracts/hive-contract/src/types.rs`

---

### **Phase 3: Rewrite Hive Heartbeat**

**Priority:** HIGH

**Tasks:**
- [ ] Delete old `bin/20_rbee_hive/src/heartbeat.rs`
- [ ] Create new `HiveHeartbeatManager` using `heartbeat-client`
- [ ] Implement monitor data collection
- [ ] Implement capability tracking
- [ ] Enhance `/capabilities` endpoint (accept `queen_url` parameter)
- [ ] Wire up discovery on startup

**Files to modify:**
- `bin/20_rbee_hive/src/heartbeat.rs` (rewrite)
- `bin/20_rbee_hive/src/main.rs` (enhance /capabilities endpoint)

---

### **Phase 4: Rewrite Worker Heartbeat**

**Priority:** HIGH

**Tasks:**
- [ ] Delete old `bin/30_llm_worker_rbee/src/heartbeat.rs`
- [ ] Create new `WorkerHeartbeatManager` using `heartbeat-client`
- [ ] Implement actual HTTP POST (currently commented out)
- [ ] Wire up discovery on startup

**Files to modify:**
- `bin/30_llm_worker_rbee/src/heartbeat.rs` (rewrite)
- `bin/30_llm_worker_rbee/src/main.rs` (wire up heartbeat)

---

### **Phase 5: Enhance Queen Receiver**

**Priority:** HIGH

**Tasks:**
- [ ] Update `handle_hive_heartbeat()` to accept monitor data
- [ ] Update `handle_hive_heartbeat()` to accept capabilities
- [ ] Handle capability changes
- [ ] Return `200 OK` instead of `204 No Content`
- [ ] Update HiveRegistry to store monitor data + capabilities

**Files to modify:**
- `bin/10_queen_rbee/src/http/heartbeat.rs`
- `bin/15_queen_rbee_crates/hive-registry/src/registry.rs`

---

### **Phase 6: Implement Queen Discovery**

**Priority:** HIGH

**Tasks:**
- [ ] Create SSH config shared crate
- [ ] Create `HiveDiscovery` in Queen
- [ ] Implement `discover_all_hives()` on startup
- [ ] Wait 5s, fetch capabilities from all hives in parallel

**Files to create:**
- `bin/99_shared_crates/ssh-config-parser/` (new crate)
- `bin/10_queen_rbee/src/discovery.rs` (new file)

**Files to modify:**
- `bin/10_queen_rbee/src/main.rs` (call discovery on startup)

---

## Summary

### **Current State:**

| Component | Sender | Receiver | Status |
|-----------|--------|----------|--------|
| Hive | ✅ Simple (no backoff) | ✅ Simple (no monitor data) | Needs rewrite |
| Worker | ⏳ Stub (commented out) | ✅ Simple | Needs implementation |
| Queen | ✅ Self-heartbeat | ✅ Handles both | Needs enhancement |

### **After Consolidation:**

| Component | Sender | Receiver | Status |
|-----------|--------|----------|--------|
| Hive | ✅ `heartbeat-client` | ✅ Enhanced | Unified |
| Worker | ✅ `heartbeat-client` | ✅ Enhanced | Unified |
| Queen | ✅ Self-heartbeat | ✅ Enhanced | Separate |

### **Benefits:**

1. **Single source of truth** - Exponential backoff logic in one place
2. **Consistent behavior** - Hive and Worker use same client
3. **Easier testing** - Test heartbeat logic once
4. **Rule Zero applied** - Delete duplicate logic, keep generic registry
5. **Clear separation** - Sender logic unified, receiver logic separate

### **LOC Impact:**

- **Delete:** 152 LOC (67 hive + 85 worker)
- **Create:** ~500 LOC (heartbeat-client crate)
- **Net:** +348 LOC (but eliminates duplication, adds exponential backoff, state machine)

---

## Decision: Consolidate or Not?

### ✅ **YES - Consolidate into `heartbeat-client` crate**

**Reasons:**
1. Hive and Worker sender logic is 90% identical
2. Exponential backoff needs to be consistent
3. State machine needs to be consistent
4. Easier to maintain one implementation
5. Easier to test one implementation
6. Rule Zero: Eliminate duplication

**Keep Separate:**
- Queen receiver logic (different concern)
- Generic heartbeat-registry (already shared, works well)
- Contract types (need enhancement, not consolidation)

---

**Next Step:** Create `heartbeat-client` crate with unified sender logic.
