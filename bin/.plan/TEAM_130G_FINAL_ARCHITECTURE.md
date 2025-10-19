# TEAM-130G: FINAL ARCHITECTURE DOCUMENT

**Date:** 2025-10-19  
**Team:** TEAM-130G  
**Status:** 📋 BASED ON TEAM-130D, 130E, 130F ANALYSIS

---

## 🎯 PURPOSE

Document the **CORRECT** architecture based on:
- TEAM-130D: Identified crates (logical groupings)
- TEAM-130E: Identified consolidation opportunities + corrections
- TEAM-130F: Created implementation plans

---

## 📁 FINAL DIRECTORY STRUCTURE (CORRECT - WITH SEPARATE CRATES!)

```
bin/
├─ shared-crates/                    ← System-wide (2+ binaries)
│  ├─ daemon-lifecycle/              (NEW - Phase 3)
│  │  ├─ src/lib.rs
│  │  └─ Cargo.toml
│  ├─ rbee-http-client/              (NEW - Phase 3)
│  │  ├─ src/lib.rs
│  │  └─ Cargo.toml
│  ├─ rbee-types/                    (NEW - Phase 3)
│  │  ├─ src/lib.rs
│  │  └─ Cargo.toml
│  ├─ auth-min/                      (EXISTING)
│  ├─ input-validation/              (EXISTING)
│  ├─ audit-logging/                 (EXISTING)
│  ├─ deadline-propagation/          (EXISTING)
│  └─ observability-narration-core/  (EXISTING)
│
├─ rbee-keeper/
│  ├─ src/
│  │  ├─ main.rs                     (~12 LOC) - Entry point ONLY
│  │  └─ lib.rs                      (~10 LOC) - Re-exports from crates
│  └─ Cargo.toml
├─ rbee-keeper-crates/               ← Binary-specific crates
│  ├─ config/
│  │  ├─ src/lib.rs                  (~44 LOC)
│  │  └─ Cargo.toml
│  ├─ cli/
│  │  ├─ src/lib.rs                  (~180 LOC)
│  │  └─ Cargo.toml
│  └─ commands/
│     ├─ src/
│     │  ├─ lib.rs
│     │  ├─ infer.rs
│     │  ├─ setup.rs
│     │  ├─ workers.rs
│     │  ├─ logs.rs                  (REVISED: no SSH)
│     │  └─ install.rs
│     └─ Cargo.toml
│
├─ queen-rbee/
│  ├─ src/
│  │  ├─ main.rs                     (~50 LOC) - Entry point ONLY
│  │  └─ lib.rs                      (~30 LOC) - Re-exports from crates
│  └─ Cargo.toml
├─ queen-rbee-crates/                ← Binary-specific crates
│  ├─ ssh-client/                    (NEW - Phase 3) - ONLY queen uses SSH!
│  │  ├─ src/lib.rs                  (~120 LOC)
│  │  └─ Cargo.toml
│  ├─ hive-registry/
│  │  ├─ src/lib.rs                  (~200 LOC)
│  │  └─ Cargo.toml
│  ├─ worker-registry/
│  │  ├─ src/lib.rs                  (~210 LOC) - Routing context
│  │  └─ Cargo.toml
│  ├─ hive-lifecycle/
│  │  ├─ src/lib.rs                  (~300 LOC) - NEW
│  │  └─ Cargo.toml
│  ├─ http-server/
│  │  ├─ src/
│  │  │  ├─ lib.rs
│  │  │  ├─ routes.rs
│  │  │  ├─ beehives.rs
│  │  │  ├─ workers.rs
│  │  │  └─ ...
│  │  └─ Cargo.toml
│  └─ preflight/
│     ├─ src/lib.rs
│     └─ Cargo.toml
│
├─ rbee-hive/
│  ├─ src/
│  │  ├─ main.rs                     (~50 LOC) - Entry point + daemon args (clap)
│  │  └─ lib.rs                      (~30 LOC) - Re-exports from crates
│  └─ Cargo.toml
├─ rbee-hive-crates/                 ← Binary-specific crates (NO CLI!)
│  ├─ worker-lifecycle/
│  │  ├─ src/lib.rs                  (~150 LOC)
│  │  └─ Cargo.toml
│  ├─ worker-registry/
│  │  ├─ src/lib.rs                  (~250 LOC) - Lifecycle context
│  │  └─ Cargo.toml
│  ├─ model-catalog/                 (MOVED from shared-crates)
│  │  ├─ src/
│  │  │  ├─ lib.rs
│  │  │  ├─ catalog.rs
│  │  │  └─ types.rs
│  │  └─ Cargo.toml
│  ├─ model-provisioner/
│  │  ├─ src/lib.rs
│  │  └─ Cargo.toml
│  ├─ monitor/
│  │  ├─ src/lib.rs                  (~200 LOC) - REFACTORED
│  │  └─ Cargo.toml
│  ├─ http-server/
│  │  ├─ src/lib.rs
│  │  └─ Cargo.toml
│  └─ ... (other crates)
│
├─ llm-worker-rbee/
│  ├─ src/
│  │  ├─ main.rs                     (~50 LOC) - Entry point ONLY
│  │  ├─ lib.rs                      (~80 LOC) - Re-exports from crates
│  │  ├─ bin/
│  │  │  ├─ cpu.rs                   (~30 LOC)
│  │  │  ├─ cuda.rs                  (~30 LOC)
│  │  │  └─ metal.rs                 (~30 LOC)
│  │  └─ backend/                    (~2,500 LOC) - LLM-SPECIFIC, stays in binary!
│  │     ├─ inference.rs
│  │     ├─ sampling.rs
│  │     ├─ tokenizer_loader.rs
│  │     ├─ gguf_tokenizer.rs
│  │     └─ models/
│  │        ├─ llama.rs
│  │        ├─ mistral.rs
│  │        ├─ phi.rs
│  │        └─ qwen.rs
│  └─ Cargo.toml
└─ llm-worker-rbee-crates/           ← Binary-specific crates (NOT worker-rbee-crates!)
   ├─ http-server/
   │  ├─ src/
   │  │  ├─ lib.rs
   │  │  ├─ validation.rs            (~50 LOC) - REPLACED
   │  │  └─ ...
   │  └─ Cargo.toml
   ├─ device-detection/
   │  ├─ src/lib.rs
   │  └─ Cargo.toml
   ├─ heartbeat/
   │  ├─ src/lib.rs
   │  └─ Cargo.toml
   └─ ... (other crates)
```

---

## 🔑 KEY ARCHITECTURE PRINCIPLES

### 1. Binary Structure (CRITICAL!)

**Each binary contains ONLY:**
- `main.rs` - Entry point ONLY (~12-50 LOC)
  - **rbee-keeper:** CLI parsing (clap) + command dispatch
  - **queen-rbee:** Daemon startup
  - **rbee-hive:** Daemon args only (`--port`, `--config`) - NO CLI commands!
  - **llm-worker-rbee:** Worker startup
- `lib.rs` - Re-exports from crates (~10-80 LOC)
- **EXCEPTION:** `llm-worker-rbee` also has `src/backend/` (~2,500 LOC) - LLM-specific inference

**Binary does NOT contain business logic - ALL logic is in separate crates!**

**WHY SEPARATE CRATES:**
- **Prevents AI drift** - AI cannot add files to `src/` without explicitly creating a new crate
- **Clear boundaries** - Each crate has its own Cargo.toml
- **Obvious violations** - Adding SSH to `rbee-keeper/src/` is easy, creating `rbee-keeper-crates/ssh/` is obvious violation
- **Modular** - Each crate can be tested/compiled independently

### 2. Crate Organization

**Three levels:**
1. **`bin/shared-crates/`** - System-wide (ALL binaries)
2. **`bin/<binary>-crates/`** - Binary-specific crates
3. **`bin/<binary>/src/backend/`** - ONLY for llm-worker-rbee (LLM-specific inference)

### 3. Shared Crates (NEW - Phase 3)

| Crate | Purpose | Used By | Location |
|-------|---------|---------|----------|
| **daemon-lifecycle** | Daemon spawning/management | rbee-keeper, queen-rbee, rbee-hive | `shared-crates/` |
| **rbee-http-client** | HTTP client wrapper | ALL 4 binaries | `shared-crates/` |
| **rbee-types** | Shared types | ALL 4 binaries | `shared-crates/` |

**NOTE:** Only 3 NEW shared crates (not 4!)

### 3.1. Queen-Specific Crate (NOT Shared!)

| Crate | Purpose | Used By | Location |
|-------|---------|---------|----------|
| **ssh-client** | SSH client wrapper | **queen-rbee ONLY** | `queen-rbee-crates/` |

**CRITICAL: SSH is ONLY in queen-rbee!**
- **Why:** In network mode, rbee-hive runs on remote machines
- **How:** queen-rbee uses SSH to start/stop remote hives
- **Flow:** queen-rbee SSH → remote hive (start/stop), then HTTP for operations
- **NOT shared:** Only queen-rbee needs SSH, so it's in `queen-rbee-crates/`, NOT `shared-crates/`

### 4. SSH Architecture (CRITICAL!)

**ONLY queen-rbee has SSH access!**

```
Network Mode (Remote Hives):
┌─────────────┐
│ queen-rbee  │
└──────┬──────┘
       │
       │ SSH (start/stop hive daemon)
       ↓
┌─────────────────────┐
│ Remote Machine      │
│ ┌─────────────────┐ │
│ │  rbee-hive      │ │
│ │  (daemon)       │ │
│ └─────────────────┘ │
└─────────────────────┘
       ↑
       │ HTTP (all operations after startup)
       │
┌──────┴──────┐
│ queen-rbee  │
└─────────────┘
```

**Why SSH in queen-rbee:**
1. **Network mode:** rbee-hive runs on remote machines
2. **Lifecycle:** queen-rbee needs to start/stop remote hives via SSH
3. **After startup:** All communication via HTTP (no more SSH)

**Why NO SSH in rbee-keeper:**
- rbee-keeper only talks to queen-rbee (HTTP)
- queen-rbee handles all SSH to remote hives
- Single SSH entry point = better security

**Why NO SSH in rbee-hive:**
- rbee-hive is the daemon being managed
- rbee-hive doesn't manage other daemons via SSH
- rbee-hive spawns workers locally (no SSH needed)

**Why NO SSH in llm-worker:**
- llm-worker is managed by rbee-hive
- llm-worker doesn't manage anything
- llm-worker just does inference

### 5. What is Shared vs Binary-Internal

**Shared (in `bin/shared-crates/`):**
- Used by **2+ binaries**
- Generic/reusable logic
- No binary-specific context
- Examples: daemon-lifecycle (3 binaries), rbee-http-client (4 binaries), rbee-types (4 binaries)

**Binary-Internal (in `bin/<binary>-crates/`):**
- Used by **1 binary only**
- Binary-specific context
- Not reusable across binaries
- Examples: 
  - `rbee-keeper-crates/commands` (only rbee-keeper)
  - `queen-rbee-crates/hive-registry` (only queen-rbee)
  - `queen-rbee-crates/ssh-client` (only queen-rbee) ← **SSH is here!**

---

## 📊 CRITICAL DECISIONS FROM TEAM-130E/F

### 1. WorkerInfo is NOT Shared

**queen-rbee WorkerInfo (routing context):**
```rust
pub struct WorkerInfo {
    pub id: String,
    pub url: String,
    pub model_ref: String,
    pub state: rbee_types::WorkerState,  // Shared enum
    pub node_name: String,               // Routing-specific
    pub slots_available: u32,            // Load balancing
    pub vram_bytes: Option<u64>,         // Capacity planning
}
```

**rbee-hive WorkerInfo (lifecycle context):**
```rust
pub struct WorkerInfo {
    pub id: String,
    pub url: String,
    pub model_ref: String,
    pub state: rbee_types::WorkerState,  // Shared enum
    pub pid: Option<u32>,                // Lifecycle-specific
    pub restart_count: u32,              // Restart policy
    pub failed_health_checks: u32,       // Health monitoring
    pub last_heartbeat: Option<SystemTime>, // Stale detection
}
```

**Only share WorkerState enum** - NOT the full WorkerInfo struct!

---

### 2. model-catalog is NOT Shared

**TEAM-130E mistake:** Listed as shared crate

**TEAM-130F correction:** Move to `bin/rbee-hive/src/model_catalog/`

**Why:**
- Only rbee-hive uses it
- Tracks models on THIS hive (local SQLite)
- Single-hive scope

---

### 3. inference-base Stays in Binary

**llm-worker-rbee keeps inference in `src/backend/`:**
- Tightly coupled to Candle
- Worker-specific inference logic
- Not generic enough for reuse

---

### 4. SSH is ONLY in queen-rbee (CRITICAL!)

**Network Mode Lifecycle:**

```
1. queen-rbee needs to start remote hive:
   queen-rbee → SSH → remote machine → start rbee-hive daemon
   
2. Wait for hive to be ready:
   queen-rbee → HTTP health check → remote hive
   
3. All operations after startup:
   queen-rbee → HTTP → remote hive (no more SSH)
   
4. queen-rbee needs to stop remote hive:
   queen-rbee → HTTP shutdown → remote hive (graceful)
   OR
   queen-rbee → SSH → remote machine → kill rbee-hive (force)
```

**Why this architecture:**
- **Single SSH entry point:** Only queen-rbee has SSH credentials
- **Security:** rbee-keeper never touches SSH
- **Simplicity:** rbee-hive doesn't need SSH (it's the daemon being managed)
- **Network mode:** SSH needed to start daemons on remote machines
- **Local mode:** No SSH needed (direct process spawn)

---

### 4. NO Backward Compatibility

**Deleted:**
- ❌ `rbee-keeper/src/pool_client.rs` (115 LOC)

**Why:**
- Creates confusion and drift
- Not actually compatible
- No users depend on it

---

## 📋 NEW CRATE DETAILS

### Shared Crates (in `bin/shared-crates/`)

#### 1. daemon-lifecycle (~500 LOC)

**Purpose:** Daemon spawning/management

**API:**
```rust
pub struct DaemonLifecycle {
    config: LifecycleConfig,
}

impl DaemonLifecycle {
    pub async fn ensure_running(&self) -> Result<DaemonHandle>;
    pub async fn stop(&self, handle: DaemonHandle) -> Result<()>;
}
```

**Usage:**
- rbee-keeper → queen-rbee lifecycle
- queen-rbee → rbee-hive lifecycle (local)
- rbee-hive → llm-worker lifecycle

---

#### 2. rbee-http-client (~120 LOC)

**Purpose:** HTTP client wrapper

**API:**
```rust
pub struct RbeeHttpClient {
    client: reqwest::Client,
    base_url: Option<String>,
}

impl RbeeHttpClient {
    pub async fn post_json<T, R>(&self, path: &str, body: &T) -> Result<R>;
    pub async fn get_json<R>(&self, path: &str) -> Result<R>;
    pub async fn health_check(&self, path: &str) -> bool;
}
```

**Usage:**
- All 4 binaries (27 call sites)

---

#### 3. rbee-types (~220 LOC)

**Purpose:** Shared types

**Contents:**
```rust
// Beehive types
pub struct BeehiveNode { ... }  // Shared across keeper + queen
pub type HiveId = String;

// Worker types
pub enum WorkerState {          // Shared across queen + hive + worker
    Loading,
    Idle,
    Busy,
}

// HTTP request/response types
pub struct ReadyRequest { ... }
pub struct AddNodeRequest { ... }
pub struct AddNodeResponse { ... }
```

**NOT included:**
- ❌ WorkerInfo (different contexts - routing vs lifecycle)

---

### Queen-Specific Crate (in `bin/queen-rbee-crates/`)

#### ssh-client (~120 LOC)

**Purpose:** SSH client wrapper for remote hive management

**API:**
```rust
pub struct SshClient {
    host: String,
    port: u16,
    user: String,
    key_path: Option<PathBuf>,
}

impl SshClient {
    pub async fn test_connection(&self) -> Result<bool>;
    pub async fn exec(&self, command: &str) -> Result<ExecResult>;
    pub async fn exec_detached(&self, command: &str) -> Result<()>;
}
```

**Usage:**
- **queen-rbee ONLY** (network mode hive lifecycle)
- Start/stop remote hive daemons
- NOT in `shared-crates/` because only 1 binary uses it

**Why NOT shared:**
- Only queen-rbee needs SSH
- rbee-keeper talks to queen via HTTP (no SSH)
- rbee-hive is the daemon being managed (no SSH)
- llm-worker is managed by hive (no SSH)

---

## 📊 LOC IMPACT SUMMARY

| Binary | Before | After | Change | Key Changes |
|--------|--------|-------|--------|-------------|
| rbee-keeper | 1,252 | 985 | **-267 LOC** | Delete SSH, pool_client, use shared crates |
| queen-rbee | 2,015 | 2,100 | **+85 LOC** | Add hive lifecycle (critical missing) |
| rbee-hive | 4,184 | 3,887 | **-297 LOC** | Remove CLI, move model-catalog |
| llm-worker | 5,026 | 4,435 | **-591 LOC** | Fix validation (641 LOC!) |
| **TOTAL** | **12,477** | **11,407** | **-1,070 LOC** | Net savings |

**System-wide savings:** 8.6% reduction

---

## 🚀 IMPLEMENTATION PHASES

### Phase 1: Create Shared Crates (Week 1)
1. Create `daemon-lifecycle` crate (in `shared-crates/`)
2. Create `rbee-http-client` crate (in `shared-crates/`)
3. Create `rbee-types` crate (in `shared-crates/`)

**NOTE:** Only 3 shared crates! SSH is NOT shared.

### Phase 2: Create Binary-Specific Crates (Week 1)
1. Create `queen-rbee-crates/ssh-client` (queen-rbee ONLY)
2. Create `queen-rbee-crates/hive-lifecycle` (uses ssh-client)
3. Create other binary-specific crates as needed

### Phase 3: Integrate & Remove Violations (Week 2)
1. **rbee-keeper:** Remove SSH, pool_client, use shared crates
2. **queen-rbee:** Add hive lifecycle (uses ssh-client)
3. **rbee-hive:** Remove CLI commands, move model-catalog to `rbee-hive-crates/`
4. **llm-worker:** Fix validation (use input-validation)

### Phase 4: Testing & Cleanup (Week 3)
1. Unit tests for all shared crates
2. Unit tests for all binary-specific crates
3. Integration tests for all binaries
4. Remove unused dependencies
5. Update documentation

---

## ✅ VALIDATION CHECKLIST

### Architecture Correctness
- [ ] NO SSH in rbee-keeper
- [ ] NO CLI commands in rbee-hive (daemon only)
- [ ] WorkerInfo NOT shared (different contexts)
- [ ] model-catalog in rbee-hive, NOT shared-crates
- [ ] inference-base in llm-worker binary, NOT shared

### Shared Crate Usage
- [ ] daemon-lifecycle used by 3 binaries
- [ ] rbee-http-client used by 4 binaries
- [ ] rbee-ssh-client used by queen-rbee only
- [ ] rbee-types used by 4 binaries
- [ ] WorkerState enum shared, WorkerInfo NOT shared

### LOC Targets
- [ ] rbee-keeper: -267 LOC (1,252 → 985)
- [ ] queen-rbee: +85 LOC (2,015 → 2,100) - Adding missing functionality
- [ ] rbee-hive: -297 LOC (4,184 → 3,887)
- [ ] llm-worker: -591 LOC (5,026 → 4,435)
- [ ] Total: -1,070 LOC (8.6% reduction)

---

## 📝 LESSONS LEARNED

### 1. Read Previous Team Documentation FIRST
- TEAM-130D identified the crates (logical groupings)
- TEAM-130E analyzed consolidation + made corrections
- TEAM-130F created implementation plans
- Don't reinvent - build on their work

### 2. Context Matters More Than Name
- WorkerInfo looks similar but serves different purposes
- queen-rbee: Routing context
- rbee-hive: Lifecycle context
- Don't consolidate by name alone

### 3. Verify Usage, Not Just Location
- model-catalog in shared-crates doesn't mean it's shared
- Check actual usage across binaries
- Move to correct location

### 4. Descriptive Naming
- ✅ `daemon-lifecycle` (specific, reusable)
- ❌ `lifecycle` (too vague)
- ✅ `rbee-http-client` (specific, descriptive)
- ❌ `http-util` (too generic)

---

**Status:** 📋 ARCHITECTURE DOCUMENTED  
**Based On:** TEAM-130D, TEAM-130E (+ corrections), TEAM-130F  
**Next:** Implementation (Phase 1-3)  
**Total Impact:** -1,070 LOC (8.6% reduction)

---

**END OF TEAM-130G ARCHITECTURE DOCUMENT**
