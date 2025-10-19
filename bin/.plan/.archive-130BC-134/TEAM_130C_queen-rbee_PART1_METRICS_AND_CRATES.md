# TEAM-130B: queen-rbee - PART 1: METRICS & CRATES

**Binary:** `bin/queen-rbee`  
**Phase:** Phase 2, Day 5-6  
**Date:** 2025-10-19

---

## 🎯 EXECUTIVE SUMMARY

**Current:** Orchestrator daemon (2,015 LOC code-only, 17 files)  
**Proposed:** 4 focused crates under `queen-rbee-crates/`  
**Risk:** LOW  
**Timeline:** 2.5 days (20 hours)

**Phase 1 Cross-Binary Corrections Applied:**
- ✅ Shared crate audit INCOMPLETE - only 5/11 audited (Team 132 error)
- ✅ narration-core gap identified - missing observability
- ✅ hive-core integration - BeehiveNode should move to shared crate
- ✅ secrets-management unused → REMOVE
- 🔴 CRITICAL: Command injection vulnerability → MUST FIX in Phase 4

**Reference:** `TEAM_130B_CROSS_BINARY_ANALYSIS.md`

---

## 📊 GROUND TRUTH METRICS

```bash
$ cloc bin/queen-rbee/src --quiet
Files: 17 | Code: 2,015 | Comments: 367 | Blanks: 428
Total Lines: 2,810
```

**Team 132 Accuracy:** 100% ✅ (perfect LOC match)

**Largest Files:**
1. http/inference.rs - 466 LOC (orchestration)
2. http/handlers/workers.rs - 260 LOC (worker endpoints)
3. beehive_registry.rs - 200 LOC (node registry)
4. worker_registry.rs - 153 LOC (worker tracking)
5. orchestrator.rs - 144 LOC (orchestration core)

---

## 🏗️ 4 PROPOSED CRATES

| # | Crate | LOC | Purpose | Risk |
|---|-------|-----|---------|------|
| 1 | registry | 353 | Dual registry (beehive + worker) | Low |
| 2 | remote | 182 | SSH + preflight (CMD INJECTION!) | Medium¹ |
| 3 | http-server | 897 | HTTP endpoints | Medium |
| 4 | orchestrator | 610 | Orchestration logic | Medium |

**Total:** 2,042 LOC in libraries + 283 LOC binary

**Critical Security Issue:**
- ¹ remote crate has command injection in ssh.rs (MUST FIX in Phase 4)

---

## 📦 CRATE SPECIFICATIONS

### CRATE 1: queen-rbee-registry (353 LOC)

**Purpose:** Dual registry system (persistent beehive nodes + ephemeral workers)  
**Files:** beehive_registry.rs (200) + worker_registry.rs (153)

**API:**
```rust
// Beehive Registry (SQLite persistent)
pub struct BeehiveRegistry {
    db: Arc<Pool<Sqlite>>,
}

pub struct BeehiveNode {
    pub node_name: String,
    pub ssh_host: String,
    pub ssh_port: u16,
    pub status: String,
    // ... 12 fields total
}

impl BeehiveRegistry {
    pub async fn register_node(&self, node: BeehiveNode) -> Result<()>;
    pub async fn list_nodes(&self) -> Result<Vec<BeehiveNode>>;
    pub async fn get_node(&self, name: &str) -> Result<BeehiveNode>;
}

// Worker Registry (In-memory ephemeral)
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<String, WorkerInfo>>>,
}

impl WorkerRegistry {
    pub fn register_worker(&self, info: WorkerInfo);
    pub fn list_workers(&self) -> Vec<WorkerInfo>;
    pub fn remove_worker(&self, worker_id: &str);
}
```

**Dependencies:** sqlx (sqlite), tokio, serde

**Cross-Binary Context:**
- **CRITICAL:** BeehiveNode duplicated between queen-rbee and rbee-keeper
- **Resolution:** Move to `hive-core` shared crate (prevents schema drift)
- **Similar:** rbee-hive also has worker registry (persistent vs ephemeral difference)

**Phase 1 Correction:**
- Team 132 didn't analyze hive-core usage (audit incomplete)
- BeehiveNode should be in hive-core, not defined locally

---

### CRATE 2: queen-rbee-remote (182 LOC)

**Purpose:** Remote operations (SSH + preflight checks)  
**Files:** ssh.rs (76) + http_client.rs (76) + preflight/*.rs (60)

**API:**
```rust
pub async fn execute_remote_command(
    host: &str,
    user: &str,
    command: &str,
) -> Result<String>;

pub async fn preflight_check(node: &BeehiveNode) -> Result<PreflightResult>;
```

**🔴 CRITICAL SECURITY VULNERABILITY:**
```rust
// CURRENT (UNSAFE - ssh.rs:79):
.arg(command)  // ← Direct user input, command injection!

// MUST FIX IN PHASE 4:
use shellwords;
let sanitized = shellwords::split(command)?;
// Validate no dangerous patterns (&&, ||, ;)
.arg("--")  // Force argument boundary
.args(&sanitized)
```

**Dependencies:** tokio, anyhow, shellwords (ADD for fix)

**Test Required:**
```rust
#[test]
fn test_command_injection_blocked() {
    let result = execute_remote_command(..., "echo && rm -rf /").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("injection"));
}
```

**Cross-Binary Context:**
- rbee-keeper also uses SSH (system `ssh` binary, respects ~/.ssh/config)
- Both should use consistent SSH pattern
- Opportunity: Extract to `rbee-ssh-client` shared crate after fixing vulnerability

---

### CRATE 3: queen-rbee-http-server (897 LOC)

**Purpose:** HTTP server with all endpoints  
**Files:** http/{server,routes,handlers/*,types,middleware}.rs

**API:**
```rust
pub struct QueenHttpServer {
    beehive_registry: Arc<BeehiveRegistry>,
    worker_registry: Arc<WorkerRegistry>,
    orchestrator: Arc<Orchestrator>,
}

// Endpoints
pub async fn inference_endpoint(...) -> Result<Response>; // SSE streaming
pub async fn register_beehive(...) -> Result<Json<RegisterResponse>>;
pub async fn spawn_worker(...) -> Result<Json<SpawnResponse>>;
pub async fn list_workers(...) -> Result<Json<Vec<WorkerInfo>>>;
```

**Dependencies:** axum, tower, tokio, serde_json, auth-min, audit-logging, input-validation, deadline-propagation

**Shared Crate Usage (EXCELLENT):**
- ✅ auth-min: Full implementation with timing-safe comparison
- ✅ audit-logging: Auth events (AuthSuccess, AuthFailure)
- ✅ input-validation: Request validation
- ✅ deadline-propagation: Timeout handling

**Cross-Binary Context:**
- Integration: queen-rbee → rbee-hive (worker spawn via HTTP)
- Integration: queen-rbee → llm-worker (inference via HTTP + SSE)
- Opportunity: Extract shared types to `rbee-http-types` (WorkerSpawnRequest, etc.)

**Phase 1 Corrections:**
- Team 133 peer review: Incomplete shared crate audit (only 5/11 checked)
- Missing: narration-core analysis (should be added for observability)

---

### CRATE 4: queen-rbee-orchestrator (610 LOC)

**Purpose:** Inference orchestration logic  
**Files:** orchestrator.rs (144) + lifecycle.rs (100) + http/inference.rs (466 partial)

**API:**
```rust
pub struct Orchestrator {
    worker_registry: Arc<WorkerRegistry>,
    beehive_registry: Arc<BeehiveRegistry>,
}

impl Orchestrator {
    pub async fn orchestrate_inference(
        &self,
        req: InferenceRequest,
    ) -> Result<impl Stream<Item = InferenceEvent>>;
    
    pub async fn select_worker(&self, model_ref: &str) -> Result<WorkerInfo>;
    pub async fn spawn_worker_if_needed(&self, model: &str) -> Result<String>;
}
```

**Dependencies:** tokio, futures, reqwest  
**Internal:** registry, remote

**Team 132 Concern (from peer review):**
- Boundary between http-server and orchestrator unclear
- Some orchestration logic in http/inference.rs (466 LOC)
- Recommendation: Clarify which code stays in HTTP crate vs orchestrator

**Cross-Binary Context:**
- Core orchestration logic unique to queen-rbee
- Coordinates: beehive selection → worker spawn → inference routing
- Different from rbee-hive (pool management) and llm-worker (execution)

---

## 📊 DEPENDENCY GRAPH

```
Layer 0 (Standalone):
- remote (182 LOC)

Layer 1 (Core):
- registry (353 LOC)
  - beehive_registry (SQLite)
  - worker_registry (in-memory)

Layer 2 (Orchestration):
- orchestrator (610 LOC) → uses registry, remote

Layer 3 (HTTP):
- http-server (897 LOC) → uses registry, orchestrator

Binary (283 LOC) → uses http-server
```

**No circular dependencies ✅**

**Crate Extraction Order:**
1. registry (no dependencies)
2. remote (no dependencies, but FIX vulnerability first!)
3. orchestrator (depends on registry, remote)
4. http-server (depends on all)

---

## 🔗 CROSS-BINARY CONTEXT

### Shared Patterns

**Worker Registry:**
- queen-rbee: Ephemeral (request scope)
- rbee-hive: Persistent (process lifecycle)
- Opportunity: Extract common WorkerState enum

**Auth Middleware (EXCELLENT ✅):**
- queen-rbee pattern is GOLD STANDARD:
  - timing-safe token comparison
  - audit events for success/failure
  - proper error handling
- rbee-hive should adopt this pattern
- llm-worker should adopt this pattern

**Graceful Shutdown:**
- Similar to rbee-hive (coordinate cleanup)
- Opportunity: Extract shared pattern

### Integration Points

**queen-rbee → rbee-hive:**
- HTTP POST to spawn workers
- Receives ready callbacks from rbee-hive
- Contract types should go in `rbee-http-types`

**queen-rbee → llm-worker:**
- HTTP POST for inference
- SSE streaming for token output
- Deadline propagation via x-deadline header

**rbee-keeper → queen-rbee:**
- HTTP API for all operations
- Auto-start daemon if not running
- 8 API endpoints used by keeper

### Cross-Binary Opportunities (Phase 1)

**1. Type Duplication (CRITICAL):**
- BeehiveNode duplicated in queen-rbee and rbee-keeper
- **Fix:** Move to `hive-core` shared crate
- **Benefit:** Prevents schema drift between binaries

**2. Missing Observability (CRITICAL):**
- queen-rbee has ZERO narration-core usage
- llm-worker has 15× narration-core usage (excellent)
- **Fix:** Add narration-core (~40-60 narration points)
- **Benefit:** Full request tracing with correlation IDs

**3. HTTP Client Patterns:**
- queen-rbee uses reqwest for rbee-hive/worker calls
- Same patterns as rbee-hive, llm-worker, rbee-keeper
- **Opportunity:** `rbee-http-client` shared crate

**4. API Contracts:**
- Worker spawn types duplicated
- **Opportunity:** `rbee-http-types` shared crate

**5. secrets-management Waste:**
- Declared in Cargo.toml (line 63)
- NEVER used (0 grep matches)
- **Fix:** REMOVE dependency

### Shared Crate Audit (CORRECTED)

Team 132 only audited 5/11 shared crates (45% complete). Full audit:

| Shared Crate | Usage | Status |
|--------------|-------|--------|
| auth-min | ✅ Excellent | ✅ Keep |
| audit-logging | ✅ Excellent | ✅ Keep |
| input-validation | ✅ Good | ✅ Keep |
| deadline-propagation | ✅ Good | ✅ Keep |
| secrets-management | ❌ Declared, unused | 🔴 Remove |
| hive-core | ❌ Should use (BeehiveNode) | 🔴 Add |
| narration-core | ❌ Missing | 🔴 Add |
| narration-macros | ❌ Not needed | ⏸️ Skip |
| model-catalog | ❌ Not dependency | ⏸️ Skip |
| gpu-info | ❌ Not needed | ⏸️ Skip |
| jwt-guardian | ❌ Not needed | ⏸️ Skip |

**Corrections from Team 132:**
- Audit was 55% incomplete (only 5/11 checked)
- Missed hive-core opportunity (BeehiveNode duplication)
- Missed narration-core gap (no observability)
- Incorrectly claimed secrets-management "minimal use" (actually ZERO use)

---

## 🔴 CRITICAL ISSUES TO ADDRESS

### Issue #1: Command Injection Vulnerability

**Location:** queen-rbee-remote crate, ssh.rs line 79  
**Severity:** CRITICAL (RCE possible)  
**Status:** MUST FIX in Phase 4 (Migration)

**Current Code:**
```rust
Command::new("ssh")
    .arg(format!("{}@{}", user, host))
    .arg(command)  // ← UNSAFE!
```

**Attack Vector:**
```bash
command = "ls && rm -rf /important"
command = "ls; cat /etc/passwd"
command = "ls || malicious_script.sh"
```

**Fix (Phase 4):**
```rust
use shellwords;

// Parse and sanitize
let parts = shellwords::split(command)
    .map_err(|e| anyhow!("Invalid command syntax: {}", e))?;

// Validate no shell operators
for part in &parts {
    if part.contains("&&") || part.contains("||") || part.contains(";") {
        anyhow::bail!("Command injection attempt detected");
    }
}

// Safe execution
Command::new("ssh")
    .arg(format!("{}@{}", user, host))
    .arg("--")  // Force argument boundary
    .args(&parts)  // Separate arguments
```

**Testing:**
```rust
#[test]
fn test_blocks_shell_operators() {
    assert!(execute_remote_command(..., "ls && rm -rf /").await.is_err());
    assert!(execute_remote_command(..., "ls; cat /etc/passwd").await.is_err());
    assert!(execute_remote_command(..., "ls || evil").await.is_err());
}

#[test]
fn test_allows_safe_commands() {
    assert!(execute_remote_command(..., "ls -la /tmp").await.is_ok());
    assert!(execute_remote_command(..., "echo hello world").await.is_ok());
}
```

### Issue #2: BeehiveNode Type Duplication

**Impact:** Schema drift between queen-rbee and rbee-keeper  
**Fix:** Move to hive-core shared crate (Phase 3)

### Issue #3: Missing Observability

**Impact:** Cannot trace requests across system  
**Fix:** Add narration-core (Phase 3)  
**Estimate:** ~40-60 narration points

### Issue #4: Unused Dependency

**Impact:** Wasted compilation time  
**Fix:** Remove secrets-management from Cargo.toml (Phase 4)

---

## 📋 COMPARISON WITH OTHER BINARIES

### Registry Comparison

| Binary | Registry Type | Persistence | LOC |
|--------|--------------|-------------|-----|
| queen-rbee | Dual (beehive + worker) | SQLite + in-memory | 353 |
| rbee-hive | Worker only | In-memory (process) | 644 |
| llm-worker | None | N/A | 0 |
| rbee-keeper | None (client) | N/A | 0 |

**Insight:** queen-rbee has unique dual registry design

### Auth Middleware Comparison

| Binary | Pattern | Audit Events | Quality |
|--------|---------|--------------|---------|
| queen-rbee | Timing-safe + audit | ✅ Yes | EXCELLENT |
| rbee-hive | Basic + audit | ✅ Yes | Good |
| llm-worker | Basic, no audit | ❌ No | Basic |
| rbee-keeper | No auth yet | N/A | N/A |

**Insight:** queen-rbee auth pattern is GOLD STANDARD (other binaries should adopt)

### Crate Count Comparison

| Binary | LOC | Crates | Avg Crate Size |
|--------|-----|--------|----------------|
| queen-rbee | 2,015 | 4 | 504 LOC |
| rbee-hive | 4,184 | 10 | 408 LOC |
| llm-worker | 5,026 | 6 | 674 LOC |
| rbee-keeper | 1,252 | 5 | 213 LOC |

**Insight:** queen-rbee has largest average crate size (fewer, focused crates)

---

## ✅ PHASE 1 CORRECTIONS APPLIED

**Team 132 Errors Corrected:**
1. ✅ Shared crate audit incomplete (5/11 → now 11/11)
2. ✅ hive-core missing (BeehiveNode duplication identified)
3. ✅ narration-core gap identified (~40-60 points needed)
4. ✅ secrets-management waste confirmed (REMOVE)
5. ✅ Command injection documented (CRITICAL fix in Phase 4)

**Cross-Binary Context Integrated:**
- Worker spawn contracts → rbee-http-types opportunity
- Auth pattern comparison (queen-rbee is gold standard)
- BeehiveNode type sharing requirement
- HTTP client duplication patterns

---

**Status:** Part 1 Complete - Metrics & Crate Design Established  
**Next:** Part 2 (Phase 3) - External Library Analysis  
**Critical:** Command injection MUST be fixed in Phase 4  
**Reference:** TEAM_130B_CROSS_BINARY_ANALYSIS.md for full context
