# rbee Project Technical Summary

**Generated:** 2025-10-19  
**Purpose:** Comprehensive technical overview of all crates in the `bin/` directory  
**Status:** ✅ Current (based on actual folder structure)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Principles](#architecture-principles)
3. [Main Binaries](#main-binaries)
4. [Binary-Specific Crates](#binary-specific-crates)
5. [Shared Crates](#shared-crates)
6. [Workspace Structure](#workspace-structure)
7. [Key Design Decisions](#key-design-decisions)

---

## 🎯 Project Overview

**rbee** is a distributed LLM inference system written in Rust, consisting of 4 main binaries and 38+ supporting crates organized in a modular workspace architecture.

### System Components

```
┌─────────────────┐
│  rbee-keeper    │  CLI tool for managing rbee infrastructure
│  (User CLI)     │
└────────┬────────┘
         │ HTTP
         ↓
┌─────────────────┐
│  queen-rbee     │  Daemon managing rbee-hive instances
│  (Orchestrator) │  (supports SSH for remote hives)
└────────┬────────┘
         │ HTTP (+ SSH for remote startup)
         ↓
┌─────────────────┐
│  rbee-hive      │  Daemon managing LLM worker instances
│  (Pool Manager) │
└────────┬────────┘
         │ Local process spawn
         ↓
┌─────────────────┐
│ llm-worker-rbee │  LLM inference worker daemon
│ (Worker)        │  (CPU/CUDA/Metal variants)
└─────────────────┘
```

### Statistics

- **Main Binaries:** 4
- **Binary-Specific Crates:** 19
- **Shared Crates:** 19
- **Total Crates:** 42 (excluding BDD test crates)
- **License:** GPL-3.0-or-later
- **Rust Edition:** 2021

---

## 🏗️ Architecture Principles

### 1. Minimal Binary Pattern

Each binary follows a strict pattern:
- **`main.rs`**: Entry point ONLY (12-50 LOC)
- **`lib.rs`**: Re-exports from crates (10-80 LOC)
- **Exception**: `llm-worker-rbee` contains `src/backend/` for LLM-specific inference logic

**Why?** Prevents AI drift and enforces clear boundaries. Adding functionality requires explicitly creating a new crate, making violations obvious.

### 2. Three-Tier Crate Organization

```
bin/
├─ 99_shared_crates/          # System-wide (used by 2+ binaries)
├─ XX_<binary>_crates/        # Binary-specific (used by 1 binary)
└─ XX_<binary>/               # Binary entry points
```

### 3. Crate Naming Convention

- **Binaries**: `rbee-keeper`, `queen-rbee`, `rbee-hive`, `llm-worker-rbee`
- **Binary-specific crates**: `<binary>-<feature>` (e.g., `rbee-keeper-cli`)
- **Shared crates**: Generic names (e.g., `daemon-lifecycle`, `rbee-http-client`)

### 4. No Code Duplication

Shared functionality is consolidated into shared crates. Binary-specific crates contain only logic unique to that binary.

---

## 🔧 Main Binaries

### 1. rbee-keeper (00_rbee_keeper)

**Purpose:** CLI tool for managing rbee infrastructure  
**Type:** User-facing command-line interface  
**Communication:** HTTP to queen-rbee

**Structure:**
```
00_rbee_keeper/
├─ src/
│  ├─ main.rs          # Entry point + CLI parsing (clap)
│  └─ lib.rs           # Re-exports from crates
├─ bdd/                # BDD integration tests
└─ Cargo.toml
```

**Key Features:**
- CLI commands: `infer`, `setup`, `workers`, `logs`, `install`
- Manages queen-rbee lifecycle
- User-friendly interface to the rbee system

**Dependencies:** See [rbee-keeper crates](#rbee-keeper-crates-05_rbee_keeper_crates)

---

### 2. queen-rbee (10_queen_rbee)

**Purpose:** Daemon for managing rbee-hive instances  
**Type:** Long-running orchestrator daemon  
**Communication:** HTTP API + SSH (for remote hive management)

**Structure:**
```
10_queen_rbee/
├─ src/
│  ├─ main.rs          # Entry point + daemon startup
│  └─ lib.rs           # Re-exports from crates
├─ bdd/                # BDD integration tests
└─ Cargo.toml
```

**Key Features:**
- Manages multiple rbee-hive instances (local or remote)
- SSH support for remote hive lifecycle management
- HTTP API for hive operations
- Worker registry for routing
- Health monitoring and preflight checks

**Critical Design:** ONLY queen-rbee has SSH access (for remote hive management)

**Dependencies:** See [queen-rbee crates](#queen-rbee-crates-15_queen_rbee_crates)

---

### 3. rbee-hive (20_rbee_hive)

**Purpose:** Daemon for managing LLM worker instances  
**Type:** Long-running pool manager daemon  
**Communication:** HTTP API (managed by queen-rbee)

**Structure:**
```
20_rbee_hive/
├─ src/
│  ├─ main.rs          # Entry point + daemon args (--port, --config)
│  └─ lib.rs           # Re-exports from crates
├─ bdd/                # BDD integration tests
└─ Cargo.toml
```

**Key Features:**
- Manages LLM worker lifecycle (spawn, monitor, restart)
- Model catalog and provisioning
- Device detection (GPU/CPU)
- Download tracking for models
- Worker registry for local workers
- HTTP API for operations

**Critical Design:** NO CLI commands (daemon only), NO SSH (managed by queen-rbee)

**Dependencies:** See [rbee-hive crates](#rbee-hive-crates-25_rbee_hive_crates)

---

### 4. llm-worker-rbee (30_llm_worker_rbee)

**Purpose:** LLM inference worker daemon  
**Type:** Long-running inference worker  
**Communication:** HTTP API (managed by rbee-hive)

**Structure:**
```
30_llm_worker_rbee/
├─ src/
│  ├─ main.rs          # Entry point + worker startup
│  ├─ lib.rs           # Re-exports from crates
│  ├─ bin/
│  │  ├─ cpu.rs        # CPU variant
│  │  ├─ cuda.rs       # CUDA variant
│  │  └─ metal.rs      # Metal variant
│  └─ backend/         # LLM-specific inference (~2,500 LOC)
│     ├─ inference.rs
│     ├─ sampling.rs
│     ├─ tokenizer_loader.rs
│     ├─ gguf_tokenizer.rs
│     └─ models/
│        ├─ llama.rs
│        ├─ mistral.rs
│        ├─ phi.rs
│        └─ qwen.rs
├─ bdd/                # BDD integration tests
└─ Cargo.toml
```

**Key Features:**
- LLM inference using Candle framework
- Multiple backend support (CPU, CUDA, Metal)
- Model-specific implementations (Llama, Mistral, Phi, Qwen)
- Tokenization and sampling
- HTTP API for inference requests
- Heartbeat to rbee-hive

**Critical Design:** Contains `src/backend/` for LLM-specific logic (exception to minimal binary rule)

**Dependencies:** See [worker-rbee crates](#worker-rbee-crates-39_worker_rbee_crates)

---

## 📦 Binary-Specific Crates

### rbee-keeper Crates (05_rbee_keeper_crates)

Binary-specific crates for `rbee-keeper`:

#### 1. cli/
**Package:** `rbee-keeper-cli`  
**Purpose:** CLI argument parsing and command dispatch  
**Key Features:**
- Clap-based CLI structure
- Command definitions
- Help text and validation

#### 2. commands/
**Package:** `rbee-keeper-commands`  
**Purpose:** Implementation of CLI commands  
**Key Features:**
- `infer` - Run inference requests
- `setup` - Initialize rbee infrastructure
- `workers` - Manage workers
- `logs` - View logs (HTTP-based, NO SSH)
- `install` - Install rbee components

#### 3. config/
**Package:** `rbee-keeper-config`  
**Purpose:** Configuration management  
**Key Features:**
- Config file parsing
- Environment variable handling
- Validation

#### 4. queen-lifecycle/
**Package:** `rbee-keeper-queen-lifecycle`  
**Purpose:** Managing queen-rbee daemon lifecycle  
**Key Features:**
- Start/stop queen-rbee
- Health checks
- Process management

---

### queen-rbee Crates (15_queen_rbee_crates)

Binary-specific crates for `queen-rbee`:

#### 1. ssh-client/
**Package:** `queen-rbee-ssh-client`  
**Purpose:** SSH client for remote hive management  
**Key Features:**
- SSH connection management
- Remote command execution
- Detached process spawning
- **CRITICAL:** ONLY queen-rbee uses SSH!

#### 2. hive-lifecycle/
**Package:** `queen-rbee-hive-lifecycle`  
**Purpose:** Managing rbee-hive daemon lifecycle  
**Key Features:**
- Start/stop hives (local and remote)
- Uses ssh-client for remote hives
- Health monitoring
- Graceful shutdown

#### 3. hive-registry/
**Package:** `queen-rbee-hive-registry`  
**Purpose:** Registry of managed hives  
**Key Features:**
- Hive state tracking
- Routing information
- Capacity planning

#### 4. worker-registry/
**Package:** `queen-rbee-worker-registry`  
**Purpose:** Registry of workers across all hives (routing context)  
**Key Features:**
- Worker state tracking
- Load balancing data
- Routing decisions
- **Note:** Different from rbee-hive's worker-registry (different context)

#### 5. http-server/
**Package:** `queen-rbee-http-server`  
**Purpose:** HTTP API server  
**Key Features:**
- REST API endpoints
- Request routing
- Authentication/authorization
- Health checks

#### 6. preflight/
**Package:** `queen-rbee-preflight`  
**Purpose:** Preflight checks before operations  
**Key Features:**
- System validation
- Dependency checks
- Configuration validation

---

### rbee-hive Crates (25_rbee_hive_crates)

Binary-specific crates for `rbee-hive`:

#### 1. worker-lifecycle/
**Package:** `rbee-hive-worker-lifecycle`  
**Purpose:** Managing LLM worker lifecycle  
**Key Features:**
- Spawn workers locally
- Monitor worker health
- Restart policies
- Graceful shutdown

#### 2. worker-registry/
**Package:** `rbee-hive-worker-registry`  
**Purpose:** Registry of workers on THIS hive (lifecycle context)  
**Key Features:**
- Worker state tracking
- PID management
- Restart counts
- Health check tracking
- **Note:** Different from queen-rbee's worker-registry (different context)

#### 3. model-catalog/
**Package:** `rbee-hive-model-catalog`  
**Purpose:** Catalog of models on THIS hive  
**Key Features:**
- Model metadata storage (SQLite)
- Model availability tracking
- Single-hive scope
- **Note:** NOT shared (hive-specific)

#### 4. model-provisioner/
**Package:** `rbee-hive-model-provisioner`  
**Purpose:** Model download and provisioning  
**Key Features:**
- Model downloading
- Checksum verification
- Storage management

#### 5. monitor/
**Package:** `rbee-hive-monitor`  
**Purpose:** System monitoring  
**Key Features:**
- Resource usage tracking
- Performance metrics
- Health monitoring

#### 6. http-server/
**Package:** `rbee-hive-http-server`  
**Purpose:** HTTP API server  
**Key Features:**
- REST API endpoints
- Worker management endpoints
- Model management endpoints
- Health checks

#### 7. download-tracker/
**Package:** `rbee-hive-download-tracker`  
**Purpose:** Track model download progress  
**Key Features:**
- Download state management
- Progress reporting
- Resume support

#### 8. device-detection/
**Package:** `rbee-hive-device-detection`  
**Purpose:** Detect available compute devices  
**Key Features:**
- GPU detection (CUDA, Metal)
- CPU capabilities
- VRAM/RAM detection

---

### worker-rbee Crates (39_worker_rbee_crates)

Binary-specific crates for `llm-worker-rbee`:

#### 1. http-server/
**Package:** `worker-rbee-http-server`  
**Purpose:** HTTP API server for inference  
**Key Features:**
- Inference endpoints
- Request validation
- Response formatting
- Health checks

**Note:** Only 1 binary-specific crate for worker (inference logic is in binary's `src/backend/`)

---

## 🔄 Shared Crates (99_shared_crates)

System-wide crates used by 2+ binaries:

### Core Infrastructure

#### 1. daemon-lifecycle/
**Package:** `daemon-lifecycle`  
**Used By:** rbee-keeper, queen-rbee, rbee-hive  
**Purpose:** Generic daemon spawning and management  
**Key Features:**
- Process spawning
- Health checks
- Graceful shutdown
- Restart policies

#### 2. rbee-http-client/
**Package:** `rbee-http-client`  
**Used By:** All 4 binaries  
**Purpose:** HTTP client wrapper  
**Key Features:**
- Consistent HTTP client interface
- Error handling
- Retry logic
- Health checks

#### 3. rbee-types/
**Package:** `rbee-types`  
**Used By:** All 4 binaries  
**Purpose:** Shared types and enums  
**Key Features:**
- `WorkerState` enum (Loading, Idle, Busy)
- `BeehiveNode` struct
- HTTP request/response types
- **Note:** Does NOT include full `WorkerInfo` (context-specific)

#### 4. heartbeat/
**Package:** `heartbeat`  
**Used By:** llm-worker-rbee, rbee-hive  
**Purpose:** Heartbeat mechanism  
**Key Features:**
- Periodic heartbeat sending
- Timeout detection
- Health status reporting

---

### Security & Authentication

#### 5. auth-min/
**Package:** `auth-min`  
**Purpose:** Minimal authentication  
**Key Features:**
- Basic authentication
- Token validation

#### 6. jwt-guardian/
**Package:** `jwt-guardian`  
**Purpose:** JWT token management  
**Key Features:**
- JWT creation
- JWT validation
- Token refresh

#### 7. secrets-management/
**Package:** `secrets-management`  
**Purpose:** Secure secrets handling  
**Key Features:**
- Secret storage
- Encryption
- Key rotation

---

### Observability

#### 8. narration-core/
**Package:** `narration-core`  
**Purpose:** Structured logging and tracing  
**Key Features:**
- Structured log events
- Trace context
- Log aggregation

#### 9. narration-macros/
**Package:** `narration-macros`  
**Purpose:** Macros for narration-core  
**Key Features:**
- Logging macros
- Trace macros

#### 10. audit-logging/
**Package:** `audit-logging`  
**Purpose:** Audit trail logging  
**Key Features:**
- Audit event recording
- Compliance logging
- Tamper-proof logs

---

### Utilities

#### 11. input-validation/
**Package:** `input-validation`  
**Purpose:** Input validation utilities  
**Key Features:**
- Request validation
- Schema validation
- Sanitization

#### 12. deadline-propagation/
**Package:** `deadline-propagation`  
**Purpose:** Request deadline propagation  
**Key Features:**
- Timeout management
- Deadline headers
- Cancellation propagation

---

### Legacy Crates (To Be Removed)

#### 13. hive-core/
**Status:** ⚠️ DEPRECATED (renamed to rbee-types)  
**Action:** Remove after migration complete

#### 14. model-catalog/
**Status:** ⚠️ MISPLACED (should be in rbee-hive-crates)  
**Action:** Remove (duplicate of rbee-hive-model-catalog)

---

## 📊 Workspace Structure

### Directory Layout

```
bin/
├─ 00_rbee_keeper/              # Binary: rbee-keeper
│  ├─ src/
│  │  ├─ main.rs
│  │  └─ lib.rs
│  ├─ bdd/                      # BDD tests
│  └─ Cargo.toml
│
├─ 05_rbee_keeper_crates/       # Binary-specific crates
│  ├─ cli/
│  ├─ commands/
│  ├─ config/
│  └─ queen-lifecycle/
│
├─ 10_queen_rbee/               # Binary: queen-rbee
│  ├─ src/
│  │  ├─ main.rs
│  │  └─ lib.rs
│  ├─ bdd/
│  └─ Cargo.toml
│
├─ 15_queen_rbee_crates/        # Binary-specific crates
│  ├─ ssh-client/               ⚠️ ONLY queen-rbee has SSH!
│  ├─ hive-lifecycle/
│  ├─ hive-registry/
│  ├─ worker-registry/
│  ├─ http-server/
│  └─ preflight/
│
├─ 20_rbee_hive/                # Binary: rbee-hive
│  ├─ src/
│  │  ├─ main.rs
│  │  └─ lib.rs
│  ├─ bdd/
│  └─ Cargo.toml
│
├─ 25_rbee_hive_crates/         # Binary-specific crates
│  ├─ worker-lifecycle/
│  ├─ worker-registry/
│  ├─ model-catalog/
│  ├─ model-provisioner/
│  ├─ monitor/
│  ├─ http-server/
│  ├─ download-tracker/
│  └─ device-detection/
│
├─ 30_llm_worker_rbee/          # Binary: llm-worker-rbee
│  ├─ src/
│  │  ├─ main.rs
│  │  ├─ lib.rs
│  │  ├─ bin/
│  │  │  ├─ cpu.rs
│  │  │  ├─ cuda.rs
│  │  │  └─ metal.rs
│  │  └─ backend/              ⚠️ Exception: LLM-specific logic
│  │     ├─ inference.rs
│  │     ├─ sampling.rs
│  │     └─ models/
│  ├─ bdd/
│  └─ Cargo.toml
│
├─ 39_worker_rbee_crates/       # Binary-specific crates
│  └─ http-server/
│
└─ 99_shared_crates/            # System-wide crates
   ├─ daemon-lifecycle/
   ├─ rbee-http-client/
   ├─ rbee-types/
   ├─ heartbeat/
   ├─ auth-min/
   ├─ jwt-guardian/
   ├─ secrets-management/
   ├─ narration-core/
   ├─ narration-macros/
   ├─ audit-logging/
   ├─ input-validation/
   ├─ deadline-propagation/
   ├─ hive-core/               ⚠️ DEPRECATED
   └─ model-catalog/           ⚠️ MISPLACED
```

### Workspace Members (from root Cargo.toml)

**Total Members:** 42 crates + BDD test crates

**Main Binaries:** 4
- `bin/00_rbee_keeper`
- `bin/10_queen_rbee`
- `bin/20_rbee_hive`
- `bin/30_llm_worker_rbee`

**Binary-Specific Crates:** 19
- rbee-keeper: 4 crates
- queen-rbee: 6 crates
- rbee-hive: 8 crates
- llm-worker-rbee: 1 crate

**Shared Crates:** 19 crates (including 2 deprecated)

**BDD Test Crates:** 1 per binary + per crate (not counted in totals)

---

## 🎯 Key Design Decisions

### 1. SSH Architecture

**CRITICAL:** Only queen-rbee has SSH access!

**Rationale:**
- Network mode: rbee-hive runs on remote machines
- queen-rbee needs to start/stop remote hives via SSH
- After startup: All communication via HTTP (no more SSH)
- Single SSH entry point = better security

**Flow:**
```
1. queen-rbee → SSH → remote machine → start rbee-hive daemon
2. queen-rbee → HTTP health check → remote hive
3. queen-rbee → HTTP → remote hive (all operations)
4. queen-rbee → HTTP shutdown → remote hive (graceful)
   OR
   queen-rbee → SSH → remote machine → kill rbee-hive (force)
```

**Why NO SSH in other binaries:**
- rbee-keeper: Only talks to queen-rbee (HTTP)
- rbee-hive: Is the daemon being managed (doesn't manage others)
- llm-worker: Managed by rbee-hive (no management role)

---

### 2. WorkerInfo is NOT Shared

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

**Decision:** Only share `WorkerState` enum, NOT the full `WorkerInfo` struct.

**Rationale:** Different contexts require different fields. Forcing a shared struct would create coupling and confusion.

---

### 3. model-catalog is NOT Shared

**Current Location:** `bin/99_shared_crates/model-catalog/` ⚠️ INCORRECT  
**Correct Location:** `bin/25_rbee_hive_crates/model-catalog/` ✅

**Rationale:**
- Only rbee-hive uses it
- Tracks models on THIS hive (local SQLite)
- Single-hive scope
- Not generic enough for reuse

**Action:** Remove from shared-crates (duplicate exists in rbee-hive-crates)

---

### 4. inference-base Stays in Binary

**Location:** `bin/30_llm_worker_rbee/src/backend/`

**Rationale:**
- Tightly coupled to Candle framework
- Worker-specific inference logic
- Model-specific implementations
- Not generic enough for reuse
- Exception to minimal binary rule (justified)

---

### 5. No Backward Compatibility

**Deleted:**
- ❌ `rbee-keeper/src/pool_client.rs` (115 LOC)

**Rationale:**
- Creates confusion and drift
- Not actually compatible
- No users depend on it
- Pre-1.0 allows breaking changes

---

### 6. Separate Crates Prevent AI Drift

**Problem:** AI assistants tend to add files to existing directories, leading to bloat.

**Solution:** Separate crates with explicit boundaries.

**Example:**
- ❌ Easy: Add `rbee-keeper/src/ssh_client.rs` (AI might do this)
- ✅ Obvious violation: Create `rbee-keeper-crates/ssh-client/` (clearly wrong)

**Benefit:** Violations become obvious, preventing architectural drift.

---

## 📈 Metrics

### Crate Count by Category

| Category | Count | Percentage |
|----------|-------|------------|
| Main Binaries | 4 | 9.5% |
| Binary-Specific Crates | 19 | 45.2% |
| Shared Crates | 19 | 45.2% |
| **Total** | **42** | **100%** |

### Crate Distribution by Binary

| Binary | Binary-Specific Crates | Uses Shared Crates |
|--------|------------------------|-------------------|
| rbee-keeper | 4 | ~10 |
| queen-rbee | 6 | ~12 |
| rbee-hive | 8 | ~10 |
| llm-worker-rbee | 1 | ~8 |

### Shared Crate Usage

| Crate | Used By | Binaries |
|-------|---------|----------|
| rbee-http-client | All | 4 |
| rbee-types | All | 4 |
| daemon-lifecycle | Most | 3 |
| heartbeat | Workers | 2 |
| auth-min | Servers | 3 |
| narration-core | All | 4 |
| input-validation | All | 4 |

---

## 🚀 Development Workflow

### Adding New Functionality

1. **Determine scope:**
   - Used by 1 binary? → Binary-specific crate
   - Used by 2+ binaries? → Shared crate

2. **Create crate:**
   ```bash
   # Binary-specific
   mkdir -p bin/XX_<binary>_crates/<feature>
   
   # Shared
   mkdir -p bin/99_shared_crates/<feature>
   ```

3. **Add to workspace:**
   Edit `Cargo.toml` to add crate to `[workspace.members]`

4. **Implement:**
   - Keep binary's `main.rs` minimal
   - Put logic in crates
   - Follow existing patterns

### Testing

Each crate has:
- Unit tests in `src/` (using `#[cfg(test)]`)
- BDD tests in `bdd/` subdirectory (optional)

System-level BDD tests in `test-harness/bdd/`

---

## 🔍 Common Patterns

### HTTP Server Pattern

All HTTP servers follow similar structure:
```rust
// In <binary>-http-server crate
pub struct Server {
    router: Router,
    config: ServerConfig,
}

impl Server {
    pub async fn serve(&self) -> Result<()> {
        // Axum server setup
    }
}
```

### Registry Pattern

Registries (hive-registry, worker-registry) follow:
```rust
pub struct Registry {
    entries: HashMap<String, Entry>,
}

impl Registry {
    pub fn register(&mut self, entry: Entry) -> Result<()>;
    pub fn get(&self, id: &str) -> Option<&Entry>;
    pub fn list(&self) -> Vec<&Entry>;
}
```

### Lifecycle Pattern

Lifecycle managers follow:
```rust
pub struct Lifecycle {
    config: LifecycleConfig,
}

impl Lifecycle {
    pub async fn ensure_running(&self) -> Result<Handle>;
    pub async fn stop(&self, handle: Handle) -> Result<()>;
    pub async fn health_check(&self) -> bool;
}
```

---

## 📝 Maintenance Notes

### Cleanup Tasks

1. **Remove deprecated crates:**
   - `bin/99_shared_crates/hive-core/` (renamed to rbee-types)
   - `bin/99_shared_crates/model-catalog/` (duplicate, use rbee-hive version)

2. **Remove old directories:**
   - `bin/old.rbee-keeper/`
   - `bin/old.queen-rbee/`
   - `bin/old.rbee-hive/`
   - `bin/old.llm-worker-rbee/`

3. **Verify BDD test coverage:**
   - Each crate should have BDD tests where applicable
   - System-level tests in `test-harness/bdd/`

### Documentation Updates Needed

1. Update README files in each crate
2. Add API documentation (rustdoc)
3. Create architecture diagrams
4. Document deployment procedures

---

## 🎓 Lessons Learned

### 1. Context Matters More Than Name

Don't consolidate by name alone. `WorkerInfo` in queen-rbee and rbee-hive serve different purposes despite similar names.

### 2. Verify Usage, Not Just Location

Just because a crate is in `shared-crates/` doesn't mean it's actually shared. Check actual usage.

### 3. Descriptive Naming

- ✅ `daemon-lifecycle` (specific, reusable)
- ❌ `lifecycle` (too vague)
- ✅ `rbee-http-client` (specific, descriptive)
- ❌ `http-util` (too generic)

### 4. Separate Crates Enforce Boundaries

Physical separation prevents logical drift. AI assistants can't accidentally add files to wrong places.

---

## 📚 References

- **Root Cargo.toml:** `/home/vince/Projects/llama-orch/Cargo.toml`
- **Architecture Doc:** `/home/vince/Projects/llama-orch/bin/.plan/TEAM_130G_FINAL_ARCHITECTURE.md`
- **Directory Structure:** `/home/vince/Projects/llama-orch/bin/DIRECTORY_STRUCTURE.md`

---

**Status:** ✅ CURRENT (2025-10-19)  
**Next Review:** After Phase 1 implementation  
**Maintainer:** TEAM-135

---

**END OF TECHNICAL SUMMARY**
