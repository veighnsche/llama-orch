# TEAM-130B: rbee-keeper - PART 1: METRICS & CRATES

**Binary:** `bin/rbee-keeper` (CLI tool `rbee`)  
**Phase:** Phase 2, Day 7-8  
**Date:** 2025-10-19

---

## 🎯 EXECUTIVE SUMMARY

**Current:** CLI remote control tool (1,252 LOC code-only, 13 files)  
**Proposed:** 5 focused crates under `rbee-keeper-crates/`  
**Risk:** LOW (smallest binary, clear boundaries)  
**Timeline:** 4 days (30 hours)

**Phase 1 Cross-Binary Corrections Applied:**
- ✅ Only 1/11 shared crates used (minimal integration)
- ✅ 2 bugs identified: workers.rs missing queen-lifecycle, logs.rs wrong pattern
- ✅ BeehiveNode duplication with queen-rbee → needs hive-core
- ✅ narration-core opportunity for better UX

**Reference:** `TEAM_130B_CROSS_BINARY_ANALYSIS.md`

---

## 📊 GROUND TRUTH METRICS

```bash
$ cloc bin/rbee-keeper/src --quiet
Files: 13 | Code: 1,252 | Comments: 300 | Blanks: 214
Total Lines: 1,766
```

**Team 134 Accuracy:** 100% ✅ (perfect LOC match)

**Largest Files:**
1. commands/setup.rs - 222 LOC (node registry)
2. commands/workers.rs - 197 LOC (worker management)
3. commands/infer.rs - 186 LOC (inference CLI)
4. cli.rs - 175 LOC (CLI parsing)
5. pool_client.rs - 115 LOC (HTTP client)

**Structure:**
```
commands/       703 LOC (6 command files)
cli.rs          175 LOC (clap parsing - stays in binary)
pool_client.rs  115 LOC (rbee-hive client)
queen_lifecycle  75 LOC (auto-start daemon)
config.rs        44 LOC (config loading)
ssh.rs           14 LOC (SSH wrapper)
main.rs          12 LOC (entry point)
```

---

## 🏗️ 5 PROPOSED CRATES

| # | Crate | LOC | Purpose | Risk |
|---|-------|-----|---------|------|
| 1 | config | 44 | Configuration loading | Low |
| 2 | ssh-client | 14 | SSH wrapper | Low |
| 3 | pool-client | 115 | rbee-hive HTTP client | Low |
| 4 | queen-lifecycle | 75 | Auto-start queen-rbee | Medium |
| 5 | commands | 817 | All CLI commands | Medium |

**Total:** 1,065 LOC in libraries + 187 LOC binary (main + cli.rs)

**Why cli.rs stays in binary:**
- Tightly coupled to binary (clap derive macros)
- Changes with every command addition
- No benefit from extraction

---

## 📦 CRATE SPECIFICATIONS

### CRATE 1: rbee-keeper-config (44 LOC)

**Purpose:** Configuration file management  
**Files:** config.rs (44)

**API:**
```rust
pub struct Config {
    pub pool: PoolConfig,
    pub paths: PathsConfig,
    pub remote: Option<RemoteConfig>,
}

pub struct PoolConfig {
    pub default_url: String,
    pub api_key: String,
}

impl Config {
    pub fn load() -> Result<Self>;
}
```

**Config Paths (priority order):**
1. `$RBEE_CONFIG` environment variable
2. `~/.config/rbee/config.toml`
3. `/etc/rbee/config.toml`

**Dependencies:** serde, toml, dirs, anyhow

**Test Strategy:**
- Load from different paths
- Missing config handling
- TOML parse errors
- Environment variable overrides

**Cross-Binary:** CLI-specific (not reusable)

**Risk:** LOW (simple file loading)

---

### CRATE 2: rbee-keeper-ssh-client (14 LOC)

**Purpose:** SSH command execution wrapper  
**Files:** ssh.rs (14)

**API:**
```rust
pub fn execute_remote_command_streaming(
    host: &str,
    command: &str,
) -> Result<()>;
```

**Implementation:** Uses system `ssh` binary (respects ~/.ssh/config)

**Dependencies:** indicatif (progress spinner), anyhow

**Advantages of system ssh:**
- ✅ Respects ~/.ssh/config
- ✅ Uses system SSH keys
- ✅ No custom SSH implementation
- ✅ Works with SSH agents

**Test Strategy:**
- Mock SSH command execution
- Progress spinner display
- Error handling (connection refused, auth failed)

**Cross-Binary:**
- Different from queen-rbee SSH (that has command injection vulnerability!)
- rbee-keeper uses system ssh (safer)
- Opportunity: Standardize SSH approach after fixing queen-rbee

**Risk:** LOW (simple wrapper, delegates to system ssh)

---

### CRATE 3: rbee-keeper-pool-client (115 LOC)

**Purpose:** HTTP client for rbee-hive (direct pool mode, rarely used)  
**Files:** pool_client.rs (115)

**API:**
```rust
pub struct PoolClient {
    base_url: String,
    api_key: String,
    client: reqwest::Client,
}

impl PoolClient {
    pub fn new(base_url: String, api_key: String) -> Self;
    
    pub async fn health_check(&self) -> Result<HealthResponse>;
    pub async fn spawn_worker(&self, req: SpawnWorkerRequest) -> Result<SpawnWorkerResponse>;
}
```

**Current Status:** ✅ Already has 5 unit tests!

**Usage:** Direct pool mode (bypasses queen-rbee)  
**Future:** May be used when queen-rbee orchestration not needed

**Dependencies:** reqwest, serde, serde_json, anyhow

**Test Strategy:** Already tested ✅

**Cross-Binary:**
- Duplicate HTTP client pattern (reqwest usage)
- Candidate for `rbee-http-client` shared crate

**Risk:** LOW (already tested, standalone)

---

### CRATE 4: rbee-keeper-queen-lifecycle (75 LOC)

**Purpose:** Auto-start queen-rbee daemon if not running  
**Files:** queen_lifecycle.rs (75)

**API:**
```rust
pub async fn ensure_queen_rbee_running(
    client: &reqwest::Client,
    queen_url: &str,
) -> Result<()>;
```

**Behavior:**
1. Check if queen-rbee is running (GET /health)
2. If not running, spawn queen-rbee process
3. Wait for ready (30s timeout)
4. Detach process (continue running in background)

**Features:**
- `RBEE_SILENT=1` env var for quiet mode
- Progress updates during startup
- Ephemeral database mode

**Dependencies:** tokio, reqwest, colored, anyhow

**Critical Bugs from Phase 1:**

**Bug #1: workers.rs doesn't call this (MEDIUM severity)**
```rust
// commands/workers.rs currently missing:
pub async fn handle(action: WorkersAction) -> Result<()> {
    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8080";
    
    // ❌ MISSING THIS:
    ensure_queen_rbee_running(&client, queen_url).await?;
    
    match action { ... }
}
```

**Bug #2: logs.rs uses wrong pattern (LOW severity)**
```rust
// logs.rs should use SSH directly, not queen-rbee API
// Current: Uses queen-rbee API (unnecessary dependency)
// Fix: Use ssh-client to fetch logs directly
```

**Test Strategy:**
- Mock queen-rbee health endpoint
- Timeout handling (queen fails to start)
- Process spawn errors

**Cross-Binary:**
- Unique to rbee-keeper (CLI auto-starts daemon)
- Similar pattern could be used for rbee-hive auto-start

**Risk:** MEDIUM (process spawning, integration)

---

### CRATE 5: rbee-keeper-commands (817 LOC)

**Purpose:** All CLI command implementations  
**Files:** commands/*.rs (all 6 command files)

**LOC Breakdown:**
```
setup.rs        222 LOC    Node registry (add/remove/list beehives)
workers.rs      197 LOC    Worker management (spawn/list/stop)
infer.rs        186 LOC    Inference command
install.rs       98 LOC    Installation
hive.rs          84 LOC    Hive commands (models/worker/status)
logs.rs          24 LOC    Log streaming
mod.rs            6 LOC    Exports
─────────────────────
Total:          817 LOC
```

**API:**
```rust
pub mod install {
    pub fn handle(system: bool) -> Result<()>;
}

pub mod setup {
    pub async fn handle(action: SetupAction) -> Result<()>;
}

pub mod hive {
    pub fn handle(action: HiveAction) -> Result<()>;
}

pub mod infer {
    pub async fn handle(
        model: String,
        prompt: String,
        max_tokens: u32,
    ) -> Result<()>;
}

pub mod workers {
    pub async fn handle(action: WorkersAction) -> Result<()>;
}

pub mod logs {
    pub async fn handle(node: String, follow: bool) -> Result<()>;
}
```

**Dependencies:**
- Internal: config, ssh-client, pool-client, queen-lifecycle
- Shared: input-validation (used in setup.rs, infer.rs)
- External: reqwest, serde, colored, futures, tokio

**Command Details:**

**setup.rs (222 LOC):**
- Beehive node registry (add/remove/list)
- Defines BeehiveNode locally (⚠️ duplicated with queen-rbee!)
- Uses input-validation for node names
- Calls queen-lifecycle for auto-start

**workers.rs (197 LOC) - HAS BUG:**
- ❌ Missing ensure_queen_rbee_running() call
- Worker spawn/list/stop via queen-rbee API
- Should be fixed during extraction

**infer.rs (186 LOC):**
- Inference via queen-rbee orchestration
- SSE streaming client (receives tokens)
- Uses input-validation for prompt
- Colored output for results

**install.rs (98 LOC):**
- System installation script
- Downloads binaries
- Configures systemd services
- Uses config for paths

**hive.rs (84 LOC):**
- Direct SSH to hive nodes
- Runs rbee-hive CLI commands remotely
- Uses ssh-client

**logs.rs (24 LOC) - HAS BUG:**
- ❌ Currently uses queen-rbee API (wrong!)
- Should use SSH directly to fetch logs
- Fix during extraction

**Test Strategy:**
- Unit: Command logic with mocks
- Integration: Full commands with test server
- BDD: CLI scenarios per command

**Cross-Binary:**
- Similar structure to rbee-hive-cli
- Shared pattern: Command implementations as modules
- Difference: rbee-keeper remote control, rbee-hive daemon

**Risk:** MEDIUM (largest crate, many integrations)

---

## 📊 DEPENDENCY GRAPH

```
Layer 0 (Standalone):
- config (44 LOC)
- ssh-client (14 LOC)
- pool-client (115 LOC)
- queen-lifecycle (75 LOC)

Layer 1 (Commands - Uses all Layer 0):
- commands (817 LOC) → uses config, ssh, pool, queen-lifecycle, input-validation

Binary (187 LOC):
- main.rs (12 LOC) → uses commands
- cli.rs (175 LOC) → stays in binary (clap parsing)
```

**No circular dependencies ✅**

**Migration Order:**
1. config (no deps)
2. ssh-client (no deps)
3. pool-client (no deps, already tested)
4. queen-lifecycle (no deps)
5. commands (depends on all Layer 0)

**Parallelization:** Layer 0 can be done in parallel (4 crates simultaneously)

---

## 🔗 CROSS-BINARY CONTEXT

### Type Duplication (CRITICAL)

**BeehiveNode defined in setup.rs:**
```rust
struct BeehiveNode {
    node_name: String,
    ssh_host: String,
    ssh_port: u16,
    ssh_user: String,
    ssh_key_path: Option<String>,
    git_repo_url: String,
    git_branch: String,
    install_path: String,
    // 8 fields total
}
```

**Also defined in queen-rbee beehive_registry.rs:**
```rust
pub struct BeehiveNode {
    pub node_name: String,
    pub ssh_host: String,
    pub ssh_port: u16,
    // ... 12 fields total (MORE fields!)
}
```

**Problem:** Schema drift! queen-rbee has 12 fields, keeper has 8 fields

**Fix (Phase 3):**
- Move to `hive-core` shared crate
- Both binaries import from hive-core
- Single source of truth

### Integration Points

**rbee-keeper → queen-rbee:**
- 8 API endpoints used:
  - `/health` - Auto-start detection
  - `/v2/tasks` - Inference orchestration
  - `/v2/registry/beehives/*` - Node registry
  - `/v2/workers/*` - Worker management
  - `/v2/logs` - Log streaming

**rbee-keeper → rbee-hive (via SSH):**
- Direct SSH command execution
- `rbee-hive models|workers|status` commands

**rbee-keeper → queen-rbee (auto-start):**
- Spawns queen-rbee process if not running
- Waits for health endpoint ready

### Shared Crate Opportunities

**Current Usage (1/11):**
- ✅ input-validation (2 files: setup.rs, infer.rs)

**Should Add (from Phase 1):**
- ⚠️ audit-logging (MEDIUM priority - track node add/remove)
- ⚠️ narration-core (MEDIUM priority - better UX)

**Not Needed:**
- ❌ auth-min (no authentication yet)
- ❌ secrets-management (system SSH handles credentials)
- ❌ deadline-propagation (simple CLI timeouts OK)
- ❌ jwt-guardian (no JWT requirement)
- ❌ model-catalog (server-side validation)
- ❌ gpu-info (remote control, no local GPU)

### narration-core Opportunity (Phase 1 Finding)

**Current (basic colored output):**
```rust
println!("{}", "🎯 Inference complete".green());
println!("{}", format!("Generated {} tokens", count).cyan());
```

**With narration-core (better):**
```rust
narrate(NarrationFields {
    actor: "rbee-cli",
    action: "inference_complete",
    target: task_id,
    human: format!("Generated {} tokens", count),
    cute: Some("🎉 Your AI masterpiece is ready!"),
    correlation_id: Some(correlation_id),
    ...
});
```

**Benefits:**
- Correlation IDs for tracing
- Structured output (machine-parseable)
- Consistent format with other binaries
- "Cute mode" for better UX

**Estimate:** ~15-20 narration points

### HTTP Client Duplication

**rbee-keeper duplicates reqwest patterns:**
- infer.rs: SSE streaming client
- workers.rs: queen-rbee API calls
- setup.rs: queen-rbee API calls

**Opportunity:** `rbee-http-client` shared crate (after Phase 1 analysis)

---

## 🐛 BUGS TO FIX DURING MIGRATION

### Bug #1: workers.rs Missing queen-lifecycle Call

**Severity:** MEDIUM  
**Impact:** Commands fail if queen-rbee not running

**Current Code:**
```rust
// commands/workers.rs
pub async fn handle(action: WorkersAction) -> Result<()> {
    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8080";
    
    // ❌ MISSING: ensure_queen_rbee_running(&client, queen_url).await?;
    
    match action {
        WorkersAction::Spawn { ... } => { ... }
        WorkersAction::List => { ... }
        WorkersAction::Stop { ... } => { ... }
    }
}
```

**Fix:**
```rust
pub async fn handle(action: WorkersAction) -> Result<()> {
    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8080";
    
    // ✅ ADD THIS:
    ensure_queen_rbee_running(&client, queen_url).await?;
    
    match action { ... }
}
```

**Test:**
```rust
#[test]
async fn test_workers_auto_starts_queen() {
    // Mock queen-rbee not running initially
    // Verify ensure_queen_rbee_running called
    // Verify command succeeds after auto-start
}
```

### Bug #2: logs.rs Wrong Integration Pattern

**Severity:** LOW  
**Impact:** Unnecessary queen-rbee dependency

**Current:** Uses queen-rbee `/v2/logs` API  
**Comment in code:** "TEAM-085: Does NOT need queen-rbee - direct SSH"

**Fix:**
```rust
// OLD (via queen-rbee):
let response = client.get(format!("{}/v2/logs?node={}", queen_url, node)).await?;

// NEW (direct SSH):
use rbee_keeper_ssh_client::execute_remote_command_streaming;
execute_remote_command_streaming(&node, "journalctl -u rbee-hive -f")?;
```

**Benefit:** Removes unnecessary queen-rbee dependency for logs

---

## 📋 COMPARISON WITH OTHER BINARIES

### Size Comparison

| Binary | LOC | Crates | Avg Crate Size | Complexity |
|--------|-----|--------|----------------|------------|
| rbee-keeper | 1,252 | 5 | 213 LOC | Low (CLI) |
| queen-rbee | 2,015 | 4 | 504 LOC | Medium (orchestration) |
| rbee-hive | 4,184 | 10 | 408 LOC | Medium (pool mgmt) |
| llm-worker | 5,026 | 6 | 674 LOC | High (inference) |

**Insight:** rbee-keeper is smallest, simplest binary (CLI tool)

### CLI Structure Comparison

| Binary | CLI Crate | Commands | LOC |
|--------|-----------|----------|-----|
| rbee-keeper | commands/ | 6 commands | 817 LOC |
| rbee-hive | cli/ | 4 modes | 719 LOC |

**Insight:** Both use similar command module structure

### Integration Complexity

| Binary | Integrations | Complexity |
|--------|-------------|------------|
| rbee-keeper | queen-rbee (8 endpoints) + SSH | Medium |
| queen-rbee | rbee-hive + llm-worker | High |
| rbee-hive | llm-worker spawn | Medium |
| llm-worker | rbee-hive callback | Low |

**Insight:** rbee-keeper has medium integration complexity (8 API endpoints + SSH)

---

## ✅ PHASE 1 CORRECTIONS APPLIED

**Team 134 Findings Verified:**
1. ✅ LOC accuracy: 1,252 (100% correct)
2. ✅ Clean architecture: No circular deps
3. ✅ 2 bugs identified (workers.rs, logs.rs)
4. ✅ Strong migration strategy (30 hours)

**Cross-Binary Corrections:**
1. ✅ BeehiveNode duplication identified → needs hive-core
2. ✅ narration-core opportunity identified (~15-20 points)
3. ✅ HTTP client duplication → rbee-http-client opportunity
4. ✅ Minimal shared crate usage (1/11) is appropriate for CLI

**Bugs to Fix:**
1. workers.rs: Add ensure_queen_rbee_running() call
2. logs.rs: Use SSH directly instead of queen-rbee API

---

**Status:** Part 1 Complete - Metrics & Crate Design Established  
**Next:** Part 2 (Phase 3) - External Library Analysis  
**Key Focus:** Bug fixes, BeehiveNode sharing, narration-core consideration  
**Reference:** TEAM_130B_CROSS_BINARY_ANALYSIS.md for full context
