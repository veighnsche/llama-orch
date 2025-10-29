# Contract Audit Report - COMPREHENSIVE

**Date:** Oct 29, 2025  
**Auditor:** Cascade AI  
**Status:** 🔍 REVIEW REQUIRED  
**Scope:** All contracts in `/bin/97_contracts` + `/contracts`

---

## Executive Summary

Audited all contracts against current architecture:
- **Queen's job API:** ONLY handles `Status` and `Infer` operations
- **Hive's job API:** Handles worker/model lifecycle operations  
- **NO PROXYING:** rbee-keeper talks directly to queen AND hive
- **Heartbeat-based registries:** No SQLite, RAM-only

**Findings:**
- ✅ 5 contracts are correct and actively used
- ❌ 1 contract should be DELETED (ssh-contract)
- ⚠️ 1 contract needs review (keeper-config-contract)
- ⚠️ 1 contract needs MAJOR updates (operations-contract)
- ✅ 2 external contracts are correct (api-types, config-schema)

---

## Contracts in `/bin/97_contracts`

## Contract Status

### ✅ KEEP - Core Contracts (5)

#### 1. **shared-contract** ✅
**Status:** KEEP - Foundation for all contracts  
**Purpose:** Common types (HealthStatus, OperationalStatus, heartbeat protocol)  
**Used by:** worker-contract, hive-contract  
**Action:** None - correct as-is

---

#### 2. **worker-contract** ✅
**Status:** KEEP - Worker protocol  
**Purpose:** Worker types, heartbeat protocol, API specification  
**Used by:** llm-worker, queen-rbee (worker registry)  
**Action:** Update README - queen port is 7833, not 8500

**Changes needed:**
```diff
- // POST to http://queen:8500/v1/worker-heartbeat
+ // POST to http://queen:7833/v1/worker-heartbeat
```

---

#### 3. **hive-contract** ✅
**Status:** KEEP - Hive protocol  
**Purpose:** Hive types, heartbeat protocol, API specification  
**Used by:** rbee-hive, queen-rbee (hive registry)  
**Action:** Update README - queen port is 7833, not 9200 for hive

**Changes needed:**
```diff
- port: 9200,
+ port: 7835,

- // POST http://queen:8500/v1/hive-heartbeat
+ // POST http://queen:7833/v1/hive-heartbeat
```

---

#### 4. **operations-contract** ⚠️
**Status:** KEEP - But needs major updates  
**Purpose:** Operation types for job submissions  
**Used by:** rbee-keeper, queen-rbee, rbee-hive  

**CRITICAL ISSUES:**

1. **`should_forward_to_hive()` is DEPRECATED**
   - Queen does NOT forward operations anymore
   - rbee-keeper talks directly to hive
   - This method should be DELETED or marked deprecated

2. **README is outdated**
   - Shows operations that don't exist (HiveStart, HiveStop, HiveCreate, etc.)
   - Doesn't reflect current architecture

3. **Documentation says "forwarded to hive"**
   - This is wrong - queen doesn't forward
   - Operations are sent directly to hive by rbee-keeper

**Actions needed:**
- [ ] DELETE `should_forward_to_hive()` method (or mark deprecated)
- [ ] Update README to reflect current architecture
- [ ] Add section explaining queen vs hive operations
- [ ] Remove references to "forwarding"

---

#### 5. **jobs-contract** ✅
**Status:** KEEP - Breaks circular dependency  
**Purpose:** JobRegistry interface for narration-core tests  
**Used by:** job-server, narration-core test binaries  
**Action:** None - correct as-is

---

### ❌ DELETE - Deprecated Contracts (1)

#### 6. **ssh-contract** ❌
**Status:** DELETE - Not used  
**Purpose:** SSH-related types  
**Used by:** NONE (only self-reference in its own Cargo.toml)  
**Reason:** SSH operations were removed in TEAM-284

**Action:** DELETE entire directory

---

### ⚠️ REVIEW - Unclear Usage (1)

#### 7. **keeper-config-contract** ⚠️
**Status:** REVIEW - Minimal usage  
**Purpose:** rbee-keeper configuration schema  
**Used by:** rbee-keeper only  
**Reason:** Only 1 external usage found

**Questions:**
- Is this contract necessary or could it live in rbee-keeper directly?
- Does it provide value as a separate contract?

**Recommendation:** 
- If only rbee-keeper uses it, move to `bin/00_rbee_keeper/src/config.rs`
- If GUI also needs it, keep as contract

---

## Detailed Issues

### operations-contract - Critical Updates Needed

#### Issue 1: `should_forward_to_hive()` Method

**Current code:**
```rust
pub fn should_forward_to_hive(&self) -> bool {
    matches!(
        self,
        Operation::WorkerSpawn(_)
            | Operation::WorkerProcessList(_)
            | Operation::WorkerProcessGet(_)
            | Operation::WorkerProcessDelete(_)
            | Operation::ModelDownload(_)
            | Operation::ModelList(_)
            | Operation::ModelGet(_)
            | Operation::ModelDelete(_)
    )
}
```

**Problem:** This implies queen forwards operations to hive, which is FALSE.

**Solution:** DELETE this method entirely. It's misleading.

**Alternative:** If needed for rbee-keeper to know which server to talk to:
```rust
pub fn target_server(&self) -> TargetServer {
    match self {
        Operation::Status | Operation::Infer(_) => TargetServer::Queen,
        Operation::WorkerSpawn(_) 
            | Operation::WorkerProcessList(_)
            | Operation::WorkerProcessGet(_)
            | Operation::WorkerProcessDelete(_)
            | Operation::ModelDownload(_)
            | Operation::ModelList(_)
            | Operation::ModelGet(_)
            | Operation::ModelDelete(_) => TargetServer::Hive,
        _ => TargetServer::Queen, // Default
    }
}

pub enum TargetServer {
    Queen,  // http://localhost:7833
    Hive,   // http://localhost:7835
}
```

---

#### Issue 2: Outdated README

**Current README shows:**
```
### Hive Operations
- `HiveStart { hive_id }`
- `HiveStop { hive_id }`
- `HiveCreate { host, port }`
- `HiveUpdate { id }`
- `HiveDelete { id }`
```

**Problem:** These operations don't exist anymore (deleted in TEAM-285, TEAM-323).

**Solution:** Update README to show actual operations:
```
### Queen Operations (http://localhost:7833/v1/jobs)
- `Status` - Query registries
- `Infer { ... }` - Schedule and route inference

### Hive Operations (http://localhost:7835/v1/jobs)
- `WorkerSpawn { ... }` - Spawn worker process
- `WorkerProcessList { ... }` - List worker processes
- `WorkerProcessGet { ... }` - Get worker process
- `WorkerProcessDelete { ... }` - Delete worker process
- `ModelDownload { ... }` - Download model
- `ModelList { ... }` - List models
- `ModelGet { ... }` - Get model details
- `ModelDelete { ... }` - Delete model
```

---

#### Issue 3: Architecture Diagram

**Current README shows:**
```
rbee-keeper                    queen-rbee
    ↓                              ↓
Operation enum              Operation enum
    ↓                              ↓
serde_json::to_value()      serde_json::from_value()
    ↓                              ↓
POST /v1/jobs  ────────────→  Pattern match & route
```

**Problem:** Implies all operations go to queen.

**Solution:** Update to show dual servers:
```
rbee-keeper
    ↓
Operation enum
    ↓
    ├─→ Queen Operations (Status, Infer)
    │   POST http://localhost:7833/v1/jobs
    │
    └─→ Hive Operations (Worker/Model lifecycle)
        POST http://localhost:7835/v1/jobs
```

---

## Missing Contracts

### 1. **api-types** (Consider Adding)

**Purpose:** Common API response types  
**Reason:** Avoid duplication between queen, hive, worker APIs

**Types to include:**
- `HealthResponse` - Standard health check response
- `InfoResponse` - Standard info endpoint response
- `ErrorResponse` - Standard error format
- `JobResponse` - Standard job submission response

**Example:**
```rust
#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,  // "ok"
}

#[derive(Serialize, Deserialize)]
pub struct JobResponse {
    pub job_id: String,
    pub status: String,
}

#[derive(Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: Option<String>,
}
```

---

## Action Items

### Immediate (Delete)
- [ ] DELETE `bin/97_contracts/ssh-contract/` (not used)

### High Priority (Fix Critical Issues)
- [ ] UPDATE `operations-contract/src/lib.rs`:
  - [ ] DELETE or deprecate `should_forward_to_hive()` method
  - [ ] ADD `target_server()` method if needed for routing
- [ ] UPDATE `operations-contract/README.md`:
  - [ ] Remove references to deleted operations
  - [ ] Add section on queen vs hive operations
  - [ ] Update architecture diagram
  - [ ] Remove "forwarding" language

### Medium Priority (Correctness)
- [ ] UPDATE `worker-contract/README.md`:
  - [ ] Fix queen port (8500 → 7833)
- [ ] UPDATE `hive-contract/README.md`:
  - [ ] Fix hive port (9200 → 7835)
  - [ ] Fix queen port (8500 → 7833)

### Low Priority (Review)
- [ ] REVIEW `keeper-config-contract`:
  - [ ] Determine if it should remain a contract
  - [ ] Consider moving to rbee-keeper if only used there

### Consider (New Contracts)
- [ ] CONSIDER creating `api-types` contract for common API types

---

## Actual Usage Analysis

### Used By Multiple Binaries (Core Contracts)

**1. shared-contract**
- Used by: worker-contract, hive-contract
- Imports: 2 files
- Status: ✅ CORE FOUNDATION

**2. worker-contract**
- Used by: llm-worker-rbee, queen-rbee (worker-registry)
- Imports: 7 files across 2 binaries
- Status: ✅ ACTIVELY USED

**3. hive-contract**
- Used by: rbee-hive, queen-rbee (hive-registry)
- Imports: 5 files across 2 binaries
- Status: ✅ ACTIVELY USED

**4. operations-contract**
- Used by: rbee-keeper, queen-rbee, rbee-hive
- Imports: 6 files across 3 binaries
- Status: ⚠️ ACTIVELY USED BUT NEEDS UPDATES

**5. jobs-contract**
- Used by: job-server (99_shared_crates)
- Imports: 1 file
- Status: ✅ ACTIVELY USED (breaks circular dependency)

### Used By Single Binary (Questionable)

**6. keeper-config-contract**
- Used by: rbee-keeper ONLY
- Imports: 1 file (bin/00_rbee_keeper/src/config.rs)
- Status: ⚠️ REVIEW - Could be inlined into rbee-keeper

### Not Used At All (Delete)

**7. ssh-contract**
- Used by: NONE (only self-reference in its own src/target.rs)
- Imports: 0 external files
- Status: ❌ DELETE - Completely unused

---

## Contracts in `/contracts` (External)

### contracts/api-types ✅
**Purpose:** Generated API types from OpenAI spec  
**Used by:** tools/openapi-client  
**Status:** ✅ KEEP - Used for OpenAI compatibility  
**Action:** None

### contracts/config-schema ✅
**Purpose:** Configuration schema types (pools, provisioning, etc.)  
**Used by:** xtask  
**Status:** ✅ KEEP - Used for config validation  
**Action:** None

---

## Summary Table

| Contract | Location | Used By | Imports | Status | Action | Priority |
|----------|----------|---------|---------|--------|--------|----------|
| **shared-contract** | 97_contracts | worker-contract, hive-contract | 2 | ✅ Keep | None | - |
| **worker-contract** | 97_contracts | llm-worker, queen | 7 | ✅ Keep | Fix ports | Medium |
| **hive-contract** | 97_contracts | rbee-hive, queen | 5 | ✅ Keep | Fix ports | Medium |
| **operations-contract** | 97_contracts | keeper, queen, hive | 6 | ⚠️ Keep | Major updates | High |
| **jobs-contract** | 97_contracts | job-server | 1 | ✅ Keep | None | - |
| **ssh-contract** | 97_contracts | NONE | 0 | ❌ Delete | Delete dir | Immediate |
| **keeper-config-contract** | 97_contracts | rbee-keeper | 1 | ⚠️ Review | Consider inline | Low |
| **api-types** | contracts | openapi-client | 1 | ✅ Keep | None | - |
| **config-schema** | contracts | xtask | 1 | ✅ Keep | None | - |

---

## Dependency Graph

### Contract Dependencies (Internal)

```
shared-contract (foundation)
    ├─→ worker-contract
    │   ├─→ llm-worker-rbee (bin/30)
    │   └─→ queen-rbee-worker-registry (bin/15)
    │       └─→ queen-rbee (bin/10)
    │
    └─→ hive-contract
        ├─→ rbee-hive (bin/20)
        └─→ queen-rbee-hive-registry (bin/15)
            └─→ queen-rbee (bin/10)

operations-contract
    ├─→ rbee-keeper (bin/00)
    ├─→ queen-rbee (bin/10)
    └─→ rbee-hive (bin/20)

jobs-contract
    └─→ job-server (bin/99_shared_crates)
        ├─→ queen-rbee (bin/10)
        ├─→ rbee-hive (bin/20)
        └─→ llm-worker-rbee (bin/30)

keeper-config-contract
    └─→ rbee-keeper (bin/00) ONLY

ssh-contract
    └─→ NONE (unused)
```

### External Contract Dependencies

```
contracts/api-types
    └─→ tools/openapi-client

contracts/config-schema
    └─→ xtask
```

---

## Usage Heatmap

| Contract | Binaries Using | Files Importing | Criticality |
|----------|----------------|-----------------|-------------|
| **shared-contract** | 2 (indirect: 4) | 2 | 🔴 CRITICAL |
| **worker-contract** | 2 | 7 | 🔴 CRITICAL |
| **hive-contract** | 2 | 5 | 🔴 CRITICAL |
| **operations-contract** | 3 | 6 | 🔴 CRITICAL |
| **jobs-contract** | 1 (indirect: 3) | 1 | 🟡 IMPORTANT |
| **keeper-config-contract** | 1 | 1 | 🟢 LOW |
| **ssh-contract** | 0 | 0 | ⚪ UNUSED |
| **api-types** | 1 | 1 | 🟢 LOW |
| **config-schema** | 1 | 1 | 🟢 LOW |

**Legend:**
- 🔴 CRITICAL: Used by multiple binaries, breaking change impacts many components
- 🟡 IMPORTANT: Used indirectly by multiple binaries
- 🟢 LOW: Used by single binary
- ⚪ UNUSED: Not used at all

---

## Recommendations

### 1. Delete ssh-contract immediately
It's not used anywhere and was deprecated in TEAM-284.

### 2. Fix operations-contract urgently
The `should_forward_to_hive()` method is actively misleading and contradicts the current architecture.

### 3. Update port numbers
Ensure all documentation reflects correct ports:
- Queen: 7833
- Hive: 7835
- Worker: 8080

### 4. Consider api-types contract
Would reduce duplication and ensure consistent API responses.

---

---

## Implementation Checklist

### Phase 1: Immediate Cleanup (1 hour)
- [ ] Delete `bin/97_contracts/ssh-contract/` directory
- [ ] Remove `ssh-contract` from workspace Cargo.toml
- [ ] Verify no broken imports with `cargo check --workspace`

### Phase 2: Fix operations-contract (2-3 hours)
- [ ] Delete or deprecate `should_forward_to_hive()` method
- [ ] Add `target_server()` method if needed for routing
- [ ] Update README with correct architecture
- [ ] Update all code examples
- [ ] Remove "forwarding" language throughout

### Phase 3: Fix Port Numbers (30 minutes)
- [ ] Update worker-contract README (queen port 8500 → 7833)
- [ ] Update hive-contract README (hive port 9200 → 7835, queen port 8500 → 7833)
- [ ] Verify all examples use correct ports

### Phase 4: Review keeper-config-contract (1 hour)
- [ ] Determine if it should remain a contract
- [ ] If only rbee-keeper uses it, move to `bin/00_rbee_keeper/src/config/`
- [ ] Update imports if moved
- [ ] Remove from workspace if moved

---

## Risk Assessment

### High Risk Changes
**operations-contract updates:**
- **Impact:** 3 binaries (rbee-keeper, queen-rbee, rbee-hive)
- **Risk:** Breaking changes if method signatures change
- **Mitigation:** Deprecate old methods, add new ones, then remove old

### Medium Risk Changes
**Port number updates:**
- **Impact:** Documentation only
- **Risk:** Low - no code changes
- **Mitigation:** None needed

### Low Risk Changes
**ssh-contract deletion:**
- **Impact:** None (unused)
- **Risk:** None
- **Mitigation:** None needed

**keeper-config-contract review:**
- **Impact:** 1 binary (rbee-keeper)
- **Risk:** Low - single consumer
- **Mitigation:** Update imports if moved

---

## Testing Strategy

### After operations-contract Changes
```bash
# Verify all binaries compile
cargo check --workspace

# Test rbee-keeper CLI
cargo run --bin rbee-keeper -- status

# Test queen-rbee
cargo run --bin queen-rbee

# Test rbee-hive
cargo run --bin rbee-hive
```

### After ssh-contract Deletion
```bash
# Verify no broken imports
cargo check --workspace

# Should pass with no errors
```

---

**Document Version:** 2.0  
**Last Updated:** Oct 29, 2025  
**Methodology:** Analyzed actual usage via grep, Cargo.toml dependencies, and import statements  
**Next Review:** After implementing action items
