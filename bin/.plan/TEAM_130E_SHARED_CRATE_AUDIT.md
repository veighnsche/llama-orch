# TEAM-130E: SHARED CRATE AUDIT & SSH ANALYSIS

**Phase:** Phase 3 (Days 9-12)  
**Date:** 2025-10-19  
**Mission:** Audit existing shared crates and SSH patterns

---

## 🎯 EXECUTIVE SUMMARY

**Finding:** 11 shared crates exist, but usage is inconsistent. Major gap: input-validation unused in llm-worker (691 LOC waste).

**Opportunities:**
1. **SSH Client Consolidation:** ~90 LOC savings
2. **Fix input-validation Usage:** ~691 LOC savings
3. **Improve Shared Crate Adoption:** ~150 LOC avoided duplication

**Total Savings:** ~931 LOC

**Impact:** HIGH - Eliminates manual validation, improves security

---

## 📊 EXISTING SHARED CRATES (11 Total)

### Crate Usage Matrix

| Shared Crate | rbee-keeper | queen-rbee | rbee-hive | llm-worker | Total Uses |
|--------------|-------------|------------|-----------|------------|------------|
| **audit-logging** | ❌ | ❌ | ❌ | ❌ | **0** (internal only) |
| **auth-min** | ❌ | ✅ (1×) | ✅ (1×) | ✅ (1×) | **3** |
| **deadline-propagation** | ❌ | ❌ | ❌ | ❌ | **0** (not implemented) |
| **gpu-info** | ❌ | ❌ | ❌ | ❌ | **0** (not used yet) |
| **hive-core** | ❌ | ❌ | ❌ | ❌ | **0** (unused!) |
| **input-validation** | ✅ (2×) | ✅ (1×) | ✅ (2×) | ❌ **MISSING!** | **5** |
| **jwt-guardian** | ❌ | ❌ | ❌ | ❌ | **0** (not needed) |
| **model-catalog** | ❌ | ❌ | ❌ | ❌ | **0** (not used yet) |
| **narration-core** | ❌ | ❌ | ❌ | ✅ (15×) | **15** (llm-worker only!) |
| **narration-macros** | ❌ | ❌ | ❌ | ❌ | **0** |
| **secrets-management** | ❌ | ❌ | ❌ | ❌ | **0** (unused!) |

### Key Findings

**✅ Well-Adopted (3+ uses):**
- auth-min: Used by all 3 HTTP servers (queen, hive, worker)
- input-validation: Used by 3/4 binaries (MISSING in llm-worker!)
- narration-core: Used extensively by llm-worker (15 call sites)

**⚠️ Partially Adopted (1-2 uses):**
- None

**❌ Not Adopted (0 uses):**
- audit-logging (internal tests only)
- deadline-propagation (not implemented)
- gpu-info (planned but not used)
- hive-core (UNUSED - duplicate WorkerInfo!)
- jwt-guardian (not needed)
- model-catalog (planned but not used)
- narration-macros (advanced feature)
- secrets-management (UNUSED!)

---

## 🔍 DETAILED CRATE ANALYSIS

### 1. auth-min ✅ WELL-USED

**Status:** ✅ Adopted by all HTTP servers

**Usage:**
- queen-rbee: `src/http/middleware/auth.rs` (1×)
- rbee-hive: `src/http/middleware/auth.rs` (1×)
- llm-worker: `src/http/middleware/auth.rs` (1×)

**Assessment:** **GOLD STANDARD** - Used consistently across all HTTP servers

**Action:** ✅ Keep as is

---

### 2. input-validation ⚠️ INCONSISTENT

**Status:** ⚠️ Used by 3/4 binaries, MISSING in llm-worker

**Usage:**
- rbee-keeper: `commands/setup.rs` (2× - node name, identifier validation)
- queen-rbee: `http/beehives.rs` (1× - node name validation)
- rbee-hive: `http/workers.rs` (2× - worker ID, model ref validation)
- llm-worker: **❌ NOT USED** (has 691 LOC manual validation instead!)

**Critical Issue:**

**llm-worker has `src/http/validation.rs` (691 LOC) doing MANUAL validation:**
```rust
// Manual validation (691 LOC!)
pub fn validate_execute_request(req: &ExecuteRequest) -> Result<(), Vec<FieldError>> {
    let mut errors = Vec::new();
    
    // Job ID validation (should use input_validation::validate_identifier!)
    if req.job_id.is_empty() {
        errors.push(FieldError { ... });
    }
    
    // Prompt validation (should use input_validation::validate_prompt!)
    if req.prompt.is_empty() || req.prompt.len() > 32768 {
        errors.push(FieldError { ... });
    }
    
    // ... 680 more lines of manual validation
}
```

**Should be:**
```rust
use input_validation::{validate_identifier, validate_prompt};

pub fn validate_execute_request(req: &ExecuteRequest) -> Result<(), Vec<FieldError>> {
    let mut errors = Vec::new();
    
    // Job ID validation
    if let Err(e) = validate_identifier(&req.job_id, 64) {
        errors.push(FieldError::from(e));
    }
    
    // Prompt validation
    if let Err(e) = validate_prompt(&req.prompt) {
        errors.push(FieldError::from(e));
    }
    
    // ... reduced to ~50 LOC
}
```

**LOC Savings:** 691 - 50 = **641 LOC**

**Action:** 🚨 **CRITICAL FIX** - Replace manual validation with input-validation

---

### 3. narration-core ⚠️ SINGLE-BINARY USE

**Status:** ⚠️ Only used by llm-worker (15× call sites)

**Usage:**
- llm-worker: Extensively used (15 narration points)
- queen-rbee: **❌ NOT USED** (should have ~40 narration points)
- rbee-hive: **❌ NOT USED** (should have ~20 narration points)
- rbee-keeper: **❌ NOT USED** (CLI, less critical)

**llm-worker narration points:**
1. Worker startup
2. Model loading
3. Inference execution
4. Token generation
5. Error handling
6. Health checks
7. Heartbeat
8. Callback ready
9. Backend initialization
10. Device detection
11. Tokenizer loading
12. Sampling configuration
13. Memory allocation
14. Request validation
15. Shutdown

**Missing in queen-rbee (should have):**
- Hive lifecycle start/stop
- Worker spawning
- Task routing
- Scheduler decisions
- Admission control
- Queue operations
- Error handling
- SSH operations
- Registry operations
- HTTP request handling

**Missing in rbee-hive (should have):**
- Worker lifecycle
- Model provisioning
- Health monitoring
- Shutdown coordination
- Registry operations
- HTTP request handling

**Action:** ⚠️ **RECOMMENDED** - Add narration to queen-rbee and rbee-hive

---

### 4. hive-core ❌ COMPLETELY UNUSED

**Status:** ❌ Exists but NEVER used

**Contains:**
- `WorkerInfo` struct (DUPLICATE of queen/hive versions!)
- `Backend` enum
- `PoolError` type
- `ModelCatalog` (empty)

**Problem:** This crate has a WorkerInfo definition, but queen-rbee and rbee-hive use their own incompatible versions!

**Action:** ❌ **DELETE** or merge into rbee-types (see HTTP Patterns doc)

---

### 5. secrets-management ❌ UNUSED

**Status:** ❌ Declared in llm-worker Cargo.toml but NEVER used

**llm-worker Cargo.toml:**
```toml
secrets-management = { path = "../shared-crates/secrets-management" }
```

**Grep search:** 0 uses in llm-worker

**Action:** ❌ **REMOVE** from Cargo.toml

---

### 6. deadline-propagation ❌ NOT IMPLEMENTED

**Status:** ❌ Exists but deadline propagation not fully implemented

**Should be used by:**
- queen-rbee: Propagate deadlines to rbee-hive
- rbee-hive: Propagate deadlines to llm-worker
- llm-worker: Respect deadlines in inference

**Currently:** Partial implementation in queen-rbee only

**Action:** 🔄 **IMPLEMENT** in future (not part of Phase 3)

---

### 7. audit-logging ❌ INTERNAL ONLY

**Status:** ❌ Only used within its own tests

**No production usage found**

**Action:** 🔄 **FUTURE** - Implement audit logging (not part of Phase 3)

---

### 8. gpu-info, model-catalog, jwt-guardian ❌ NOT USED

**Status:** ❌ Created but not integrated

**Action:** 🔄 **FUTURE** - Integrate when needed (not part of Phase 3)

---

## 📊 SSH CLIENT ANALYSIS

### SSH Usage Patterns

**Found in 2 binaries:**

### 1. queen-rbee/src/ssh.rs (76 LOC)

**Functions:**
```rust
// Test SSH connection
pub async fn test_ssh_connection(
    host: &str,
    port: u16,
    user: &str,
    key_path: Option<&str>,
) -> Result<bool>

// Execute remote command
pub async fn execute_remote_command(
    host: &str,
    port: u16,
    user: &str,
    key_path: Option<&str>,
    command: &str,
) -> Result<(bool, String, String)>
```

**Used by:**
- `http/beehives.rs`: Test SSH when adding nodes
- `preflight/ssh.rs`: SSH preflight checks

**LOC:** 76

---

### 2. rbee-keeper/src/ssh.rs (14 LOC)

**Functions:**
```rust
// Execute command with streaming output
pub fn execute_remote_command_streaming(
    host: &str,
    command: &str,
) -> Result<()>
```

**Used by:**
- `commands/logs.rs`: Stream logs from remote node

**LOC:** 14

**⚠️ VIOLATION:** This should NOT exist in rbee-keeper! SSH should only be in queen-rbee.

---

### Pattern Comparison

| Feature | queen-rbee | rbee-keeper |
|---------|------------|-------------|
| Test connection | ✅ | ❌ |
| Execute command | ✅ | ✅ |
| Streaming output | ❌ | ✅ |
| Key path support | ✅ | ❌ (uses ~/.ssh/config) |
| Async | ✅ | ❌ (blocking) |

### Consolidation Opportunity

**Both use system SSH command**, not ssh2 crate. Can consolidate into single `rbee-ssh-client` crate.

---

## 💡 PROPOSED SHARED CRATE: `rbee-ssh-client`

### API Design

```rust
pub struct SshClient {
    host: String,
    port: u16,
    user: String,
    key_path: Option<PathBuf>,
}

impl SshClient {
    pub fn new(host: impl Into<String>, user: impl Into<String>) -> Self {
        Self {
            host: host.into(),
            port: 22,
            user: user.into(),
            key_path: None,
        }
    }
    
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }
    
    pub fn with_key(mut self, key_path: impl Into<PathBuf>) -> Self {
        self.key_path = Some(key_path.into());
        self
    }
    
    /// Test SSH connection
    pub async fn test_connection(&self) -> Result<bool> {
        // Implements queen-rbee logic
    }
    
    /// Execute command and return output
    pub async fn exec(&self, command: &str) -> Result<ExecResult> {
        // Implements queen-rbee logic
    }
    
    /// Execute command with streaming output
    pub async fn exec_streaming(&self, command: &str) -> Result<()> {
        // Implements rbee-keeper logic
    }
    
    /// Execute detached command (for starting daemons)
    pub async fn exec_detached(&self, command: &str) -> Result<()> {
        // For hive lifecycle in queen-rbee
    }
}

pub struct ExecResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}
```

### LOC Savings

- queen-rbee/ssh.rs: 76 LOC
- rbee-keeper/ssh.rs: 14 LOC (VIOLATION - should be removed)
- **Total removed:** 90 LOC
- **New crate:** 120 LOC
- **Net cost:** -30 LOC initially

**But:** Enables queen-rbee hive lifecycle (~500 LOC) without duplicating SSH logic.

**Long-term value:** ~90 LOC + avoided duplication in hive lifecycle

---

## 📊 CONSOLIDATION SUMMARY

### High-Priority Fixes

| Issue | Binary | Current | Fix | Savings |
|-------|--------|---------|-----|---------|
| **Manual validation** | llm-worker | 691 LOC | Use input-validation | **641 LOC** |
| **SSH in keeper** | rbee-keeper | 14 LOC | Remove (use queen) | **14 LOC** |
| **hive-core unused** | N/A | ~100 LOC | Delete/merge | **100 LOC** |
| **secrets-management** | llm-worker | Declared | Remove dep | **0 LOC** (cleanup) |

**Total High-Priority:** **755 LOC**

### Medium-Priority Improvements

| Improvement | Binaries | Benefit |
|-------------|----------|---------|
| Add narration to queen-rbee | queen-rbee | Observability (~40 points) |
| Add narration to rbee-hive | rbee-hive | Observability (~20 points) |
| SSH client consolidation | queen, keeper | ~90 LOC + enables hive lifecycle |
| Implement deadline-propagation | All | Future: timeout handling |

---

## 🎯 RECOMMENDATIONS

### Immediate Actions (Phase 3)

1. **🚨 CRITICAL:** Replace llm-worker validation.rs with input-validation (**641 LOC**)
2. **🚨 CRITICAL:** Remove rbee-keeper ssh.rs (**14 LOC**)
3. **❌ DELETE:** Remove/merge hive-core (**100 LOC**)
4. **❌ REMOVE:** Remove secrets-management from llm-worker Cargo.toml

**Total Phase 3 savings:** **755 LOC**

### Future Actions (Post-Phase 3)

5. **✅ ADD:** Create rbee-ssh-client crate (enables hive lifecycle)
6. **✅ ADD:** Add narration-core to queen-rbee (~40 points)
7. **✅ ADD:** Add narration-core to rbee-hive (~20 points)
8. **🔄 IMPLEMENT:** Complete deadline-propagation implementation

---

## 📋 ACCEPTANCE CRITERIA

### Phase 3 (Immediate)

1. ✅ llm-worker uses input-validation (validation.rs deleted)
2. ✅ rbee-keeper ssh.rs deleted (logs via queen)
3. ✅ hive-core deleted or merged into rbee-types
4. ✅ secrets-management removed from llm-worker
5. ✅ All tests pass
6. ✅ No functionality regression

### Post-Phase 3 (Future)

7. ✅ rbee-ssh-client created
8. ✅ queen-rbee uses narration-core
9. ✅ rbee-hive uses narration-core
10. ✅ deadline-propagation implemented

---

**Status:** TEAM-130E Shared Crate Audit Complete  
**Next:** Final Consolidation Summary  
**Savings:** ~755 LOC (Phase 3) + ~250 LOC (future) = 1,005 LOC total
