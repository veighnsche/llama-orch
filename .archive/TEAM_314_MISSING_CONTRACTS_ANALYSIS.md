# TEAM-314: Missing Contracts Analysis

**Status:** 🔍 ANALYSIS  
**Date:** 2025-10-27  
**Purpose:** Identify types that should be contracts but aren't

---

## Correct Contracts Location

✅ **CORRECT:** `/home/vince/Projects/llama-orch/bin/97_contracts/`  
❌ **ANCIENT:** `/home/vince/Projects/llama-orch/contracts/` (DO NOT USE)

---

## Current Contracts (97_contracts)

### 1. shared-contract
**Purpose:** Common types shared between workers and hives

**Contains:**
- Status types (HealthStatus, OperationalStatus)
- Heartbeat protocol (HeartbeatPayload, HeartbeatTimestamp)
- Error types (ContractError)
- Constants (timeouts, intervals)

### 2. worker-contract
**Purpose:** Worker-specific contracts

### 3. hive-contract
**Purpose:** Hive-specific contracts

### 4. operations-contract
**Purpose:** Operation definitions (formerly rbee-operations)

### 5. jobs-contract
**Purpose:** Job-related contracts

---

## Missing Contracts

### 1. ✅ **DaemonHandle** (CRITICAL)

**Current Location:** `queen-lifecycle/src/types.rs`  
**Should Be:** `97_contracts/daemon-contract/` or `daemon-lifecycle/src/handle.rs`

```rust
// Currently in queen-lifecycle only
pub struct QueenHandle {
    started_by_us: bool,
    base_url: String,
    pid: Option<u32>,
}
```

**Why It's a Contract:**
- ✅ Used by multiple daemons (queen, hive, workers)
- ✅ Defines lifecycle management protocol
- ✅ Shared behavior across all daemons
- ✅ Should be generic: `DaemonHandle`

**Recommendation:** Create `daemon-contract` or move to `daemon-lifecycle`

**Impact:** HIGH - Affects all daemon lifecycle management

---

### 2. ✅ **SshTarget** (DUPLICATED)

**Current Locations:**
- `ssh-config/src/lib.rs` (authoritative)
- `rbee-keeper/src/tauri_commands.rs` (duplicate)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshTarget {
    pub host: String,
    pub host_subtitle: Option<String>,
    pub hostname: String,
    pub user: String,
    pub port: u16,
    pub status: SshTargetStatus,
}
```

**Why It's a Contract:**
- ✅ Serialized for API responses
- ✅ Used by CLI, UI, and backend
- ✅ Defines SSH host discovery protocol
- ✅ Shared between keeper and UI

**Recommendation:** Move to `97_contracts/ssh-contract/`

**Impact:** MEDIUM - Used by keeper and Tauri UI

---

### 3. ✅ **Config** (KEEPER CONFIGURATION)

**Current Location:** `rbee-keeper/src/config.rs`  
**Should Be:** `97_contracts/keeper-config-contract/`

```rust
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub queen_port: u16,
}
```

**Why It's a Contract:**
- ✅ Serialized to TOML
- ✅ Defines keeper configuration schema
- ✅ Used by CLI and potentially UI
- ✅ Needs to be stable across versions

**Recommendation:** Move to contracts with schema validation

**Impact:** LOW - Only used by keeper, but should be a contract for stability

---

### 4. ✅ **StatusRequest/StatusResponse** (DAEMON STATUS)

**Current Location:** `daemon-lifecycle/src/status.rs`  
**Should Be:** `97_contracts/daemon-contract/`

```rust
#[derive(Debug, Clone)]
pub struct StatusRequest {
    pub id: String,
    pub job_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StatusResponse {
    pub id: String,
    pub is_running: bool,
    pub health_status: Option<String>,
}
```

**Why It's a Contract:**
- ✅ Defines daemon status protocol
- ✅ Used across all daemons (queen, hive, workers)
- ✅ Serialized for API responses
- ✅ Should be stable

**Recommendation:** Move to `daemon-contract` with other daemon types

**Impact:** MEDIUM - Used by all daemon status checks

---

### 5. ✅ **InstallConfig/InstallResult** (DAEMON INSTALLATION)

**Current Location:** `daemon-lifecycle/src/install.rs`  
**Should Be:** `97_contracts/daemon-contract/`

```rust
pub struct InstallConfig {
    pub binary_name: String,
    pub binary_path: Option<String>,
    pub target_path: Option<String>,
    pub job_id: Option<String>,
}

pub struct InstallResult {
    pub binary_path: String,
    pub install_time: std::time::SystemTime,
}
```

**Why It's a Contract:**
- ✅ Defines installation protocol
- ✅ Used by all lifecycle crates
- ✅ Should be consistent across daemons
- ✅ Part of daemon lifecycle contract

**Recommendation:** Move to `daemon-contract`

**Impact:** MEDIUM - Used by all install operations

---

### 6. ✅ **HttpDaemonConfig** (DAEMON HTTP CONFIG)

**Current Location:** `daemon-lifecycle/src/lifecycle.rs`  
**Should Be:** `97_contracts/daemon-contract/`

```rust
#[derive(Clone)]
pub struct HttpDaemonConfig {
    pub daemon_name: String,
    pub health_url: String,
    pub shutdown_endpoint: Option<String>,
    pub job_id: Option<String>,
}
```

**Why It's a Contract:**
- ✅ Defines HTTP daemon configuration protocol
- ✅ Used by all HTTP-based daemons
- ✅ Should be consistent
- ✅ Part of daemon contract

**Recommendation:** Move to `daemon-contract`

**Impact:** MEDIUM - Used by queen, hive, workers

---

### 7. ❓ **CommandResponse** (TAURI RESPONSES)

**Current Location:** `rbee-keeper/src/tauri_commands.rs`  
**Should Be:** `97_contracts/ui-contract/` (maybe?)

```rust
#[derive(Serialize, Deserialize)]
pub struct CommandResponse {
    pub success: bool,
    pub message: String,
    pub data: Option<String>,
}
```

**Why It Might Be a Contract:**
- ✅ Defines UI command response format
- ✅ Used by Tauri frontend
- ✅ Should be stable

**Why It Might Not:**
- ❌ Very generic, might be Tauri-specific
- ❌ Could be internal to keeper

**Recommendation:** Consider `ui-contract` if UI becomes more complex

**Impact:** LOW - Internal to keeper/UI communication

---

## Proposed New Contracts

### 1. daemon-contract

**Purpose:** Generic daemon lifecycle contracts

**Should Contain:**
- `DaemonHandle` - Generic handle for all daemons
- `StatusRequest/StatusResponse` - Status check protocol
- `InstallConfig/InstallResult` - Installation protocol
- `UninstallConfig` - Uninstallation protocol
- `HttpDaemonConfig` - HTTP daemon configuration
- `ShutdownConfig` - Shutdown configuration

**Benefits:**
- ✅ Single source of truth for daemon contracts
- ✅ Consistent across all daemons
- ✅ Easy to version and evolve
- ✅ Clear API boundaries

### 2. ssh-contract

**Purpose:** SSH-related contracts

**Should Contain:**
- `SshTarget` - SSH host information
- `SshTargetStatus` - Connection status
- `SshConfig` - SSH configuration (if needed)

**Benefits:**
- ✅ Removes duplication (ssh-config + tauri_commands)
- ✅ Clear SSH protocol definition
- ✅ Shared by CLI and UI

### 3. keeper-config-contract

**Purpose:** Keeper configuration schema

**Should Contain:**
- `KeeperConfig` - Main configuration
- `QueenConfig` - Queen-specific config
- `HiveConfig` - Hive-specific config (if needed)

**Benefits:**
- ✅ Stable configuration schema
- ✅ Versioned configuration
- ✅ Schema validation

### 4. ui-contract (OPTIONAL)

**Purpose:** UI-specific contracts

**Should Contain:**
- `CommandResponse` - Generic command response
- `UiState` - UI state (if needed)
- `UiEvent` - UI events (if needed)

**Benefits:**
- ✅ Clear UI/backend boundary
- ✅ Type-safe UI communication

---

## Priority Ranking

### 🔴 CRITICAL (Do First)

1. **DaemonHandle** - Affects all daemon lifecycle
   - Create `daemon-contract` crate
   - Move `QueenHandle` → `DaemonHandle`
   - Add `HiveHandle`, `WorkerHandle` as type aliases

### 🟡 HIGH (Do Soon)

2. **SshTarget** - Duplicated code
   - Create `ssh-contract` crate
   - Move from `ssh-config` and `tauri_commands`
   - Update all consumers

3. **StatusRequest/StatusResponse** - Used everywhere
   - Move to `daemon-contract`
   - Standardize status protocol

### 🟢 MEDIUM (Do Eventually)

4. **InstallConfig/InstallResult** - Installation protocol
   - Move to `daemon-contract`
   - Standardize installation

5. **HttpDaemonConfig** - HTTP daemon config
   - Move to `daemon-contract`
   - Standardize HTTP daemons

6. **Config** - Keeper configuration
   - Create `keeper-config-contract`
   - Add schema validation

### 🔵 LOW (Nice to Have)

7. **CommandResponse** - UI responses
   - Consider `ui-contract`
   - Only if UI grows more complex

---

## Implementation Plan

### Phase 1: Create daemon-contract

```bash
# Create new contract crate
cd bin/97_contracts
cargo new --lib daemon-contract

# Add dependencies
cd daemon-contract
# Add serde, anyhow, etc.
```

**Structure:**
```
daemon-contract/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs
    ├── handle.rs        # DaemonHandle
    ├── status.rs        # Status types
    ├── install.rs       # Install types
    ├── lifecycle.rs     # Lifecycle types
    └── config.rs        # Config types
```

### Phase 2: Migrate DaemonHandle

1. Create `daemon-contract/src/handle.rs`
2. Implement generic `DaemonHandle`
3. Update `queen-lifecycle` to use it
4. Add `HiveHandle` type alias
5. Update all consumers

### Phase 3: Create ssh-contract

1. Create `ssh-contract` crate
2. Move `SshTarget` from `ssh-config`
3. Remove duplicate from `tauri_commands`
4. Update all consumers

### Phase 4: Migrate Other Types

1. Move status types to `daemon-contract`
2. Move install types to `daemon-contract`
3. Move config types to `daemon-contract`
4. Update all consumers

---

## Benefits of Contracts

### 1. Clear API Boundaries

Contracts define the interface between components:
- CLI ↔ Backend
- UI ↔ Backend
- Daemon ↔ Daemon

### 2. Versioning

Contracts can be versioned independently:
- `daemon-contract` v1.0.0
- `ssh-contract` v1.0.0
- Breaking changes are explicit

### 3. Documentation

Contracts serve as documentation:
- What types are shared?
- What's the protocol?
- What's stable vs internal?

### 4. Testing

Contracts can be tested independently:
- Serialization/deserialization
- Schema validation
- Protocol compliance

### 5. Code Reuse

Contracts eliminate duplication:
- One definition, many consumers
- DRY principle
- Single source of truth

---

## Anti-Patterns to Avoid

### ❌ Don't Put Everything in Contracts

**Bad:**
```rust
// daemon-contract/src/internal_helper.rs
pub fn internal_helper_function() { ... }  // ❌ Internal, not a contract
```

**Good:**
```rust
// daemon-contract/src/handle.rs
pub struct DaemonHandle { ... }  // ✅ Shared type, is a contract
```

### ❌ Don't Create Micro-Contracts

**Bad:**
```
97_contracts/
├── daemon-handle-contract/  # ❌ Too granular
├── daemon-status-contract/  # ❌ Too granular
└── daemon-install-contract/ # ❌ Too granular
```

**Good:**
```
97_contracts/
└── daemon-contract/  # ✅ Related types together
    ├── handle.rs
    ├── status.rs
    └── install.rs
```

### ❌ Don't Mix Concerns

**Bad:**
```rust
// daemon-contract/src/lib.rs
pub struct DaemonHandle { ... }  // ✅ Contract
pub fn spawn_daemon() { ... }    // ❌ Implementation, not contract
```

**Good:**
```rust
// daemon-contract/src/lib.rs
pub struct DaemonHandle { ... }  // ✅ Contract

// daemon-lifecycle/src/lib.rs
pub fn spawn_daemon() { ... }    // ✅ Implementation
```

---

## Summary

**Missing Contracts Identified:**

1. 🔴 **DaemonHandle** - CRITICAL, affects all daemons
2. 🟡 **SshTarget** - HIGH, duplicated code
3. 🟡 **StatusRequest/StatusResponse** - HIGH, used everywhere
4. 🟢 **InstallConfig/InstallResult** - MEDIUM, installation protocol
5. 🟢 **HttpDaemonConfig** - MEDIUM, HTTP daemon config
6. 🟢 **Config** - MEDIUM, keeper configuration
7. 🔵 **CommandResponse** - LOW, UI responses

**Recommended Actions:**

1. Create `daemon-contract` crate
2. Migrate `DaemonHandle` (generic)
3. Create `ssh-contract` crate
4. Migrate other daemon types
5. Consider `keeper-config-contract`

**Benefits:**
- ✅ Clear API boundaries
- ✅ No duplication
- ✅ Consistent protocols
- ✅ Better testing
- ✅ Easier versioning

---

**Maintained by:** TEAM-314  
**Last Updated:** 2025-10-27  
**Status:** ANALYSIS 🔍
