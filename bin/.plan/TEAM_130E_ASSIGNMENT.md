# TEAM-130E: CONSOLIDATION & SHARED CRATE OPPORTUNITIES

**Phase:** Phase 3 (Days 9-12)  
**Mission:** Find duplicated code patterns and create new shared crates  
**Date:** 2025-10-19

---

## 🎯 MISSION STATEMENT

**PRIMARY GOAL:** Identify code duplication across all 4 binaries and propose new shared crates to eliminate redundancy.

**CRITICAL INSIGHT FROM TEAM-130D:**
All 3 daemon-managing binaries have lifecycle management, but each implements it differently:
- rbee-keeper → queen-rbee lifecycle
- queen-rbee → rbee-hive lifecycle  
- rbee-hive → llm-worker-rbee lifecycle

**This is the EXACT type of duplication we need to eliminate.**

---

## 🔍 INVESTIGATION AREAS

### 1. LIFECYCLE MANAGEMENT PATTERNS (CRITICAL)

**Evidence of Duplication:**

**rbee-keeper (queen_lifecycle.rs - 132 LOC):**
```rust
pub async fn ensure_queen_rbee_running(client: &reqwest::Client, queen_url: &str) -> Result<()> {
    // 1. Health check
    if health_check(queen_url).is_ok() { return Ok(()); }
    
    // 2. Find binary
    let binary = find_binary("queen-rbee")?;
    
    // 3. Spawn daemon
    let child = Command::new(&binary).arg("--port").arg("8080").spawn()?;
    
    // 4. Wait for ready (30s)
    wait_for_health(queen_url, Duration::from_secs(30)).await?;
    
    // 5. Detach
    std::mem::forget(child);
    
    Ok(())
}
```

**queen-rbee (hive-lifecycle - ~800 LOC proposed):**
```rust
pub async fn start_local_hive(config: LocalHiveConfig) -> Result<HiveId> {
    // 1. Health check
    if health_check(&config.health_url).is_ok() { return Ok(config.hive_id); }
    
    // 2. Find binary
    let binary = find_binary("rbee-hive")?;
    
    // 3. Spawn daemon
    let child = Command::new(&binary).arg("--port").arg(config.port).spawn()?;
    
    // 4. Wait for ready (30s)
    wait_for_health(&config.health_url, Duration::from_secs(30)).await?;
    
    // 5. Store PID
    store_pid(config.hive_id, child.id())?;
    
    // 6. Detach
    std::mem::forget(child);
    
    Ok(config.hive_id)
}
```

**rbee-hive (monitor/spawn.rs - ~386 LOC):**
```rust
pub async fn spawn_worker(request: SpawnWorkerRequest) -> Result<WorkerId> {
    // 1. Preflight checks
    preflight::check(&request)?;
    
    // 2. Find binary
    let binary = find_worker_binary(&request.backend)?;
    
    // 3. Spawn worker
    let child = Command::new(&binary)
        .arg("--model").arg(&request.model_ref)
        .arg("--port").arg(find_free_port()?)
        .spawn()?;
    
    // 4. Wait for ready callback (300s)
    wait_for_worker_ready(worker_id, Duration::from_secs(300)).await?;
    
    // 5. Register
    worker_registry.register(worker_info).await?;
    
    Ok(worker_id)
}
```

**PATTERN IDENTIFIED:**
1. Health check / preflight
2. Find binary
3. Spawn process
4. Wait for ready (health check or callback)
5. Register / store PID
6. Detach (optional)

**PROPOSED SHARED CRATE: `daemon-lifecycle`**
```rust
pub struct DaemonLifecycle<T> {
    config: LifecycleConfig,
    registry: Option<Arc<dyn Registry<T>>>,
}

pub struct LifecycleConfig {
    pub binary_name: String,
    pub args: Vec<String>,
    pub health_url: Option<String>,
    pub callback_url: Option<String>,
    pub ready_timeout: Duration,
    pub detach: bool,
}

impl<T> DaemonLifecycle<T> {
    pub async fn ensure_running(&self, id: &str) -> Result<T> {
        // Generic lifecycle: check → spawn → wait → register
    }
    
    pub async fn stop(&self, id: &str) -> Result<()> {
        // Generic shutdown: SIGTERM → wait → SIGKILL
    }
}
```

**INVESTIGATION REQUIRED:**
- [ ] Compare all 3 lifecycle implementations line-by-line
- [ ] Extract common patterns into traits
- [ ] Identify what's generic vs what's binary-specific
- [ ] Propose unified `daemon-lifecycle` crate API
- [ ] Calculate LOC savings (estimate: ~500-800 LOC)

---

### 2. HTTP CLIENT PATTERNS (CRITICAL)

**Evidence of Duplication:**

All 4 binaries use reqwest with similar patterns:

**rbee-keeper:**
```rust
let client = reqwest::Client::new();
let response = client
    .post(format!("{}/v2/tasks", queen_url))
    .json(&request)
    .timeout(Duration::from_secs(30))
    .send().await?;
```

**queen-rbee:**
```rust
let response = reqwest::Client::new()
    .post(format!("{}/v1/workers/spawn", hive_url))
    .json(&json!({ "model_ref": model_ref }))
    .send().await?;
```

**rbee-hive:**
```rust
let response = client
    .post(format!("{}/v1/admin/shutdown", worker.url))
    .send().await?;
```

**llm-worker:**
```rust
let response = reqwest::Client::new()
    .post(format!("{}/v1/workers/ready", callback_url))
    .json(&ready_request)
    .send().await?;
```

**PATTERN IDENTIFIED:**
- All use reqwest::Client
- All construct URLs with format!()
- All send JSON payloads
- All need error handling
- All need timeout configuration

**PROPOSED SHARED CRATE: `rbee-http-client`**
```rust
pub struct RbeeHttpClient {
    client: reqwest::Client,
    base_url: String,
    default_timeout: Duration,
}

impl RbeeHttpClient {
    pub async fn post<T, R>(&self, path: &str, body: &T) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        // Standardized POST with error handling, timeouts, retries
    }
    
    pub async fn get<R>(&self, path: &str) -> Result<R>
    where
        R: DeserializeOwned,
    {
        // Standardized GET
    }
}
```

**INVESTIGATION REQUIRED:**
- [ ] Grep all reqwest usage across 4 binaries
- [ ] Identify common patterns (JSON, timeouts, errors)
- [ ] Propose unified HTTP client API
- [ ] Calculate LOC savings (estimate: ~200-400 LOC)

---

### 3. TYPE DUPLICATION (CRITICAL)

**Evidence of Duplication:**

**BeehiveNode defined in TWO places:**

`queen-rbee/beehive_registry.rs`:
```rust
pub struct BeehiveNode {
    pub node_name: String,
    pub ssh_host: String,
    pub ssh_port: u16,
    pub ssh_user: String,
    pub ssh_key_path: Option<String>,
    // ... 12 fields total
}
```

`rbee-keeper/commands/setup.rs`:
```rust
struct BeehiveNode {
    node_name: String,
    ssh_host: String,
    ssh_port: u16,
    // ... 8 fields total (DIFFERENT!)
}
```

**Worker Types duplicated:**

- WorkerInfo in queen-rbee (worker_registry.rs)
- WorkerInfo in rbee-hive (registry/)
- SpawnWorkerRequest in multiple places
- ReadyRequest in llm-worker

**PROPOSED SHARED CRATE: `rbee-types` or `rbee-http-types`**
```rust
// Shared types for ALL binaries

pub struct BeehiveNode {
    // Single source of truth
}

pub struct WorkerInfo {
    // Single source of truth
}

pub struct SpawnWorkerRequest {
    // Single source of truth
}

pub struct ReadyRequest {
    // Single source of truth
}
```

**INVESTIGATION REQUIRED:**
- [ ] List ALL type definitions across 4 binaries
- [ ] Identify duplicates and near-duplicates
- [ ] Propose unified type definitions
- [ ] Decide: One `rbee-types` or split into `rbee-http-types`, `rbee-worker-types`, etc.
- [ ] Calculate LOC savings (estimate: ~300-500 LOC)

---

### 4. HEALTH CHECK PATTERNS

**Evidence of Duplication:**

All binaries implement health check waiting:

**rbee-keeper:**
```rust
async fn wait_for_health(url: &str, timeout: Duration) -> Result<()> {
    let start = Instant::now();
    loop {
        if start.elapsed() > timeout { return Err(...); }
        
        match reqwest::get(format!("{}/health", url)).await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            _ => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
}
```

**Pattern exists in:** rbee-keeper, queen-rbee (proposed), rbee-hive

**PROPOSED:** Part of `daemon-lifecycle` crate

**INVESTIGATION REQUIRED:**
- [ ] Compare health check implementations
- [ ] Standardize timeout/retry logic
- [ ] Include in `daemon-lifecycle` crate

---

### 5. SSH CLIENT PATTERNS

**Evidence of Duplication:**

**queen-rbee has SSH (current):**
```rust
// ssh.rs (76 LOC) - only for shutdown
pub async fn execute_remote_command(...) -> Result<(bool, String, String)>;
```

**queen-rbee needs SSH (missing):**
```rust
// For hive lifecycle START (800 LOC proposed)
pub async fn start_network_hive(...) -> Result<HiveId>;
```

**Both use similar patterns:**
- SSH connection
- Command execution
- Output capture
- Error handling

**PROPOSED SHARED CRATE: `rbee-ssh-client`**
```rust
pub struct SshClient {
    host: String,
    port: u16,
    user: String,
    key_path: Option<PathBuf>,
}

impl SshClient {
    pub async fn connect(...) -> Result<Self>;
    pub async fn exec(&self, command: &str) -> Result<ExecResult>;
    pub async fn exec_detached(&self, command: &str) -> Result<()>;
}

pub struct ExecResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}
```

**INVESTIGATION REQUIRED:**
- [ ] Analyze current ssh.rs (76 LOC)
- [ ] Design full SSH client for hive lifecycle
- [ ] Identify shared patterns
- [ ] Decide: Wrapper around system `ssh` or use ssh2 crate?
- [ ] Calculate LOC savings (estimate: ~100-200 LOC)

---

### 6. PROCESS MANAGEMENT PATTERNS

**Evidence of Duplication:**

**Process spawning appears in:**
- rbee-keeper: spawns queen-rbee
- queen-rbee: spawns rbee-hive (local mode)
- rbee-hive: spawns llm-worker

**Common operations:**
- Find binary path
- Build command with args
- Spawn child process
- Store PID
- Detach process
- Send signals (SIGTERM, SIGKILL)
- Wait for exit

**PROPOSED:** Part of `daemon-lifecycle` crate

**INVESTIGATION REQUIRED:**
- [ ] Extract process management utilities
- [ ] Standardize PID storage/retrieval
- [ ] Standardize signal handling

---

### 7. SHARED CRATE USAGE ANALYSIS

**Current shared crates (from Phase 1):**
1. auth-min
2. audit-logging
3. input-validation
4. deadline-propagation
5. secrets-management
6. hive-core (proposed, not created)
7. narration-core
8. narration-macros
9. model-catalog
10. gpu-info
11. jwt-guardian

**INVESTIGATION REQUIRED:**

**For EACH shared crate:**
- [ ] Which binaries use it?
- [ ] Is usage consistent?
- [ ] Are there binaries that SHOULD use it but don't?
- [ ] Are there crates used by only ONE binary? (Should be binary crate!)
- [ ] Are there gaps (needed but not created)?

**Example Questions:**
- Why doesn't rbee-hive use narration-core? (llm-worker has 15× usage)
- Why doesn't queen-rbee use narration-core?
- Is secrets-management actually needed? (unused in llm-worker)
- Should hive-core be created for BeehiveNode?

---

### 8. VALIDATION PATTERN DUPLICATION

**Evidence:**

**llm-worker has 691 LOC manual validation** but `input-validation` exists!

**INVESTIGATION REQUIRED:**
- [ ] Compare manual validation in llm-worker with input-validation crate
- [ ] Identify what's missing from input-validation
- [ ] Propose additions to input-validation crate
- [ ] Calculate LOC savings (691 LOC waste!)

---

### 9. ERROR TYPE PATTERNS

**Evidence:**

All binaries have error types:
- worker-rbee-error (llm-worker) - 336 LOC ✅ extracted
- Similar errors in rbee-keeper, queen-rbee, rbee-hive

**INVESTIGATION REQUIRED:**
- [ ] Compare error types across binaries
- [ ] Identify common error categories
- [ ] Propose shared error traits
- [ ] Decision: One shared error crate or binary-specific?

---

### 10. METRICS & OBSERVABILITY PATTERNS

**Evidence:**

**llm-worker:** 15× narration-core usage (GOLD STANDARD)  
**rbee-hive:** Prometheus metrics (176 LOC)  
**queen-rbee:** MISSING observability  
**rbee-keeper:** Basic colored output

**INVESTIGATION REQUIRED:**
- [ ] Why doesn't queen-rbee use narration-core?
- [ ] Should rbee-hive adopt narration-core?
- [ ] Standardize observability across all binaries
- [ ] Propose ~40-60 narration points for queen-rbee

---

## 📋 DELIVERABLES (4 Documents)

### Document 1: TEAM_130E_LIFECYCLE_CONSOLIDATION.md
**Focus:** Lifecycle management patterns
- Line-by-line comparison of all 3 lifecycle implementations
- Proposed `daemon-lifecycle` shared crate API
- Migration plan for each binary
- LOC savings estimate

### Document 2: TEAM_130E_HTTP_PATTERNS.md
**Focus:** HTTP client and type duplication
- All reqwest usage across binaries
- Type duplication analysis (BeehiveNode, WorkerInfo, etc.)
- Proposed `rbee-http-client` and `rbee-types` crates
- LOC savings estimate

### Document 3: TEAM_130E_SHARED_CRATE_AUDIT.md
**Focus:** Existing shared crates analysis
- Usage matrix (which binary uses which crate)
- Inconsistencies (should use but doesn't)
- Unused crates (declared but not used)
- Missing crates (needed but not created)
- Recommendations for each shared crate

### Document 4: TEAM_130E_CONSOLIDATION_SUMMARY.md
**Focus:** Complete consolidation opportunities
- All NEW shared crates proposed
- Total LOC savings estimate
- Priority ranking (high/medium/low)
- Implementation roadmap

---

## 🎯 SUCCESS METRICS

**Target:** Identify opportunities to save **1,500-2,500 LOC** through consolidation

**Expected NEW Shared Crates:**
1. `daemon-lifecycle` (~500-800 LOC savings)
2. `rbee-http-client` (~200-400 LOC savings)
3. `rbee-types` or `rbee-http-types` (~300-500 LOC savings)
4. `rbee-ssh-client` (~100-200 LOC savings)
5. Expand `input-validation` usage (~691 LOC savings in llm-worker alone)

**Expected Improvements:**
- All 3 lifecycle implementations use same pattern
- All HTTP calls use unified client
- Zero type duplication
- Consistent observability
- Consistent shared crate usage

---

## ⚠️ CRITICAL INSTRUCTIONS

**DO NOT:**
- ❌ Focus on external libraries (axum, tokio, etc.) - that's NOT the goal
- ❌ Just list dependencies - that's pointless
- ❌ Miss consolidation opportunities by looking at binaries in isolation

**DO:**
- ✅ Compare implementations ACROSS binaries
- ✅ Identify duplicated patterns
- ✅ Propose NEW shared crates
- ✅ Calculate LOC savings
- ✅ Prioritize high-impact consolidations

**REMEMBER:**
The user expected Phase 3 to find consolidation opportunities like lifecycle management patterns. TEAM-130D missed this because we looked at each binary in isolation instead of cross-binary analysis.

---

**Team:** 130E  
**Duration:** 4 days  
**Output:** 4 consolidation analysis documents  
**Goal:** Propose 5+ new shared crates, save 1,500-2,500 LOC
