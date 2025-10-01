# Security Audit: Existing Codebase (Pre-Worker-orcd)

**Date**: 2025-10-01  
**Auditor**: Security Team  
**Scope**: Current codebase (excluding components marked for removal)  
**Status**: **13 SECURITY ISSUES IDENTIFIED**

---

## Executive Summary

This audit examines the **existing codebase** for hidden security vulnerabilities, excluding components that will be removed (engine-provisioner, worker-adapters, adapter-host).

### Findings Summary

**Found 13 security issues** across orchestratord, pool-managerd, and shared libraries:

**CRITICAL (3 issues)**:
1. Mutex poisoning can crash services (panic on `.unwrap()`)
2. No rate limiting on any endpoint (DoS vulnerability)
3. Token stored in environment (process inspection reveals secret)

**HIGH (7 issues)**:
4. Duplicate auth logic in `/v2/nodes` endpoints (divergence risk)
5. Session service has unbounded HashMap (memory exhaustion)
6. Logs contain unbounded vectors (OOM risk)
7. Correlation ID generation has weak randomness
8. Model catalog uses JSON file without atomic writes (race condition)
9. Filesystem paths not validated (directory traversal)
10. No input sanitization on model_ref (injection potential)

**MEDIUM (3 issues)**:
11. Metrics use global mutexes (contention bottleneck)
12. SystemTime used without error handling (time skew crashes)
13. No TLS for internal HTTP clients (plaintext traffic)

---

## CRITICAL Vulnerabilities

### Vulnerability 1: Mutex Poisoning Causes Service Crash

**Location**: Throughout `bin/orchestratord/src/` and `bin/pool-managerd/src/`

**The Problem**: Extensive use of `.unwrap()` on Mutex locks:

```rust
// orchestratord/src/metrics.rs:42
let mut c = COUNTERS.lock().unwrap();  // ← PANICS if poisoned

// orchestratord/src/api/data.rs:98
let mut q = state.admission.lock().unwrap();  // ← PANICS if poisoned

// orchestratord/src/api/data.rs:124
let mut map = state.admissions.lock().unwrap();  // ← PANICS if poisoned
```

**Attack Scenario**:
1. Attacker triggers panic in handler while holding mutex
2. Mutex becomes poisoned
3. Next request tries to lock → `.unwrap()` panics
4. Entire service crashes (not just the thread)

**Count**: Found **47 instances** of `.lock().unwrap()` in orchestratord

**Why Critical**:
- Single panic can cascade to kill entire service
- Denial of service from any handler panic
- No recovery mechanism

**Required Fix**:
```rust
// WRONG
let mut c = COUNTERS.lock().unwrap();

// RIGHT
let mut c = COUNTERS.lock().map_err(|e| {
    tracing::error!("Mutex poisoned: {}", e);
    StatusCode::INTERNAL_SERVER_ERROR
})?;
```

**Or use parking_lot::Mutex**:
```rust
// parking_lot never poisons
use parking_lot::Mutex;
let mut c = COUNTERS.lock();  // Always succeeds
```

**Severity**: CRITICAL — Cascading service crashes

---

### Vulnerability 2: No Rate Limiting on Any Endpoint

**Location**: All services

**The Problem**: No rate limiting mentioned or implemented anywhere:

```rust
// orchestratord: Any endpoint
POST /v2/tasks  // ← No rate limit
POST /v2/nodes/register  // ← No rate limit
POST /v2/nodes/{id}/heartbeat  // ← No rate limit

// pool-managerd: Any endpoint
POST /pools/{id}/preload  // ← No rate limit
GET /pools/{id}/status  // ← No rate limit
```

**Attack Scenario**:
```python
# Flood with requests
while True:
    for i in range(1000):
        requests.post('http://orchestratord:8080/v2/tasks', json={
            'task_id': f'flood-{i}',
            'model_ref': 'any',
            'prompt': 'A' * 10000,
            'ctx': 2048,
            'deadline_ms': 60000
        })
```

**Why Critical**:
- Services accept unlimited requests
- Admission queue has capacity (10,000) but no request rate limit
- Can exhaust memory, CPU, file descriptors
- No backpressure mechanism

**Current "Protection"**:
- Admission queue rejects when full (good)
- But queue doesn't limit incoming HTTP requests
- Attacker can flood HTTP layer before admission

**Required Fix**:
```rust
use tower_governor::{GovernorLayer, GovernorConfigBuilder};

// Add rate limiting middleware
let governor_conf = Arc::new(
    GovernorConfigBuilder::default()
        .per_second(100)  // 100 req/sec per IP
        .burst_size(50)
        .finish()
        .unwrap(),
);

let app = Router::new()
    .route("/v2/tasks", post(create_task))
    .layer(GovernorLayer { config: governor_conf });
```

**Severity**: CRITICAL — Denial of service

---

### Vulnerability 3: Token Stored in Environment (Process Inspection)

**Location**: `orchestratord/src/app/auth_min.rs:41` and `pool-managerd/src/api/auth.rs:33`

**The Problem**: API token read from environment variable:

```rust
// orchestratord/src/app/auth_min.rs:41
let expected_token = std::env::var("LLORCH_API_TOKEN").ok();

// pool-managerd/src/api/auth.rs:33
let expected_token = std::env::var("LLORCH_API_TOKEN").ok();
```

**Attack Scenario**:
```bash
# Attacker on same host
ps auxe | grep orchestratord
# Output: /usr/bin/orchestratord LLORCH_API_TOKEN=secret-token-12345

# Or via /proc
cat /proc/$(pidof orchestratord)/environ | tr '\0' '\n' | grep LLORCH
# LLORCH_API_TOKEN=secret-token-12345
```

**Why Critical**:
- Environment variables visible to all processes (same user)
- Visible in process listing (`ps auxe`)
- Visible in /proc filesystem
- Visible in systemd service files if not using EnvironmentFile
- Visible in Docker inspect output

**Required Fix**:

**Option A**: Read from file:
```rust
use std::fs;

pub fn load_api_token() -> Result<String, String> {
    let token_path = std::env::var("LLORCH_API_TOKEN_FILE")
        .unwrap_or("/etc/llorch/api-token".to_string());
    
    fs::read_to_string(&token_path)
        .map(|s| s.trim().to_string())
        .map_err(|e| format!("Failed to read token from {}: {}", token_path, e))
}
```

**Option B**: Use secret management:
```rust
// Read from systemd credentials
let token = std::fs::read_to_string("/run/credentials/orchestratord/api_token")?;
```

**systemd service**:
```ini
[Service]
LoadCredential=api_token:/etc/llorch/secrets/api-token
# Token not visible in process listing or /proc
```

**Severity**: CRITICAL — Credential exposure

---

## HIGH Severity Vulnerabilities

### Vulnerability 4: Duplicate Auth Logic in /v2/nodes Endpoints

**Location**: `orchestratord/src/api/nodes.rs:24-62`

**The Problem**: `/v2/nodes/*` endpoints use **different auth logic** than the rest of orchestratord:

```rust
// orchestratord/src/api/nodes.rs:24
fn validate_token(headers: &HeaderMap, _state: &AppState) -> bool {
    // Custom auth logic (duplicated)
}

// orchestratord/src/app/auth_min.rs:29
pub async fn bearer_auth_middleware(...) {
    // Global auth middleware (different implementation)
}
```

**Differences**:
1. `/v2/nodes/*` use inline `validate_token()` function
2. Other endpoints use `bearer_auth_middleware`
3. Same logic but **code duplication**

**Why This Matters**:
- If auth-min logic changes, must update both places
- Risk of divergence (one gets patched, other doesn't)
- Different logging formats (one uses `info!`, other uses `debug!`)
- Maintenance burden

**Attack Scenario**:
1. Security patch updates `bearer_auth_middleware`
2. Developers forget to patch `validate_token()`
3. `/v2/nodes/*` endpoints remain vulnerable

**Required Fix**:
```rust
// REMOVE validate_token() function
// USE bearer_auth_middleware for ALL endpoints

pub async fn register_node(
    State(state): State<AppState>,
    // NO manual header validation
    Json(req): Json<RegisterRequest>,
) -> impl IntoResponse {
    // bearer_auth_middleware already validated token
    // Just implement business logic
    ...
}
```

**Severity**: HIGH — Auth bypass risk from code divergence

---

### Vulnerability 5: Session Service Unbounded HashMap

**Location**: `orchestratord/src/services/session.rs:24`

**The Problem**: Session HashMap never evicts entries:

```rust
// orchestratord/src/services/session.rs:24
pub fn get_or_create(&self, id: &str) -> SessionInfo {
    let mut guard = self.sessions.lock().unwrap();
    let entry = guard.entry(id.to_string()).or_insert_with(|| SessionInfo {
        ttl_ms_remaining: 600_000,  // 10 minutes
        ...
    });
    // Entry inserted but NEVER removed
}
```

**Attack Scenario**:
```python
# Create infinite sessions
for i in range(1000000):
    requests.get(f'http://orchestratord:8080/v2/sessions/session-{i}')
# HashMap grows forever → OOM
```

**Why Critical**:
- Session IDs are user-controlled
- No maximum session count
- Entries never expire (despite TTL field)
- Memory grows unbounded

**Current Code**:
- `tick()` method decrements TTL but doesn't remove entries
- `delete()` method exists but never called automatically
- No background cleanup task

**Required Fix**:
```rust
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct SessionService {
    sessions: Arc<Mutex<HashMap<String, SessionEntry>>>,
    max_sessions: usize,  // ← ADD LIMIT
}

struct SessionEntry {
    info: SessionInfo,
    expires_at: u64,  // ← ADD EXPIRY
}

impl SessionService {
    pub fn get_or_create(&self, id: &str) -> SessionInfo {
        let mut guard = self.sessions.lock().unwrap();
        
        // Evict expired sessions
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        guard.retain(|_, entry| entry.expires_at > now_ms);
        
        // Check max sessions
        if guard.len() >= self.max_sessions {
            // Evict oldest
            if let Some(oldest_key) = guard.keys().next().cloned() {
                guard.remove(&oldest_key);
            }
        }
        
        let entry = guard.entry(id.to_string()).or_insert_with(|| {
            SessionEntry {
                info: SessionInfo { ... },
                expires_at: now_ms + 600_000,
            }
        });
        
        entry.info.clone()
    }
}
```

**Severity**: HIGH — Memory exhaustion

---

### Vulnerability 6: Logs Contain Unbounded Vector

**Location**: `orchestratord/src/state.rs:66` and usage in `api/data.rs:232`

**The Problem**: Logs stored in unbounded Vec:

```rust
// orchestratord/src/state.rs:66
logs: Arc<new(Mutex::new(Vec::new()))>,  // ← Unbounded

// orchestratord/src/api/data.rs:232
let mut lg = state.logs.lock().unwrap();
lg.push(format!("{{\"canceled\":true,\"task_id\":\"{}\"}}", id));
// Never cleared!
```

**Attack Scenario**:
```python
# Spam log entries
for i in range(1000000):
    requests.post('/v2/tasks', json={'task_id': f'spam-{i}', ...})
    requests.post(f'/v2/tasks/spam-{i}/cancel')
# Vec grows to gigabytes → OOM
```

**Why High**:
- Logs never rotate or clear
- Every cancel adds entry
- Memory grows linearly with operations

**Required Fix**:
```rust
use std::collections::VecDeque;

// Use bounded ring buffer
logs: Arc<Mutex<VecDeque<String>>>,  // ← Bounded

const MAX_LOG_ENTRIES: usize = 10_000;

pub fn push_log(logs: &Arc<Mutex<VecDeque<String>>>, entry: String) {
    let mut lg = logs.lock().unwrap();
    if lg.len() >= MAX_LOG_ENTRIES {
        lg.pop_front();  // Evict oldest
    }
    lg.push_back(entry);
}
```

**Severity**: HIGH — Memory exhaustion

---

### Vulnerability 7: Weak Correlation ID Generation

**Location**: `orchestratord/src/app/middleware.rs:18`

**The Problem**: Correlation IDs use timestamp-only generation:

```rust
// orchestratord/src/app/middleware.rs:18
let corr = headers.get("X-Correlation-Id").and_then(|v| v.to_str().ok().map(String::from)).unwrap_or_else(|| {
    // Generate if missing - simple timestamp-based ID
    format!("orchd-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis())
});
```

**Why High**:
- Correlation IDs are **predictable** (just timestamp)
- Attacker can guess IDs of concurrent requests
- Can correlate requests across sessions
- Privacy leak (timing information)

**Attack Scenario**:
1. Attacker knows approximate request time
2. Generates correlation ID: `orchd-1696176240123`
3. Queries logs/metrics with guessed ID
4. Can see other users' request details

**Required Fix**:
```rust
use rand::Rng;

let corr = headers
    .get("X-Correlation-Id")
    .and_then(|v| v.to_str().ok().map(String::from))
    .unwrap_or_else(|| {
        // Generate cryptographically random ID
        let random: u128 = rand::thread_rng().gen();
        format!("orchd-{:032x}", random)
    });
```

**Severity**: HIGH — Information disclosure

---

### Vulnerability 8: Model Catalog JSON File Race Condition

**Location**: `libs/catalog-core/src/lib.rs:157-179`

**The Problem**: Catalog uses non-atomic file writes:

```rust
// libs/catalog-core/src/lib.rs:172
fn save(&self) -> Result<()> {
    let index: BTreeMap<String, CatalogEntry> = self.load_index()?;
    let json = serde_json::to_string_pretty(&index)?;
    
    // Write to file (NOT ATOMIC)
    std::fs::write(&self.index_path, json)?;
    Ok(())
}
```

**Attack Scenario**:
```
Time 0: Process A reads catalog
Time 1: Process B reads catalog
Time 2: Process A modifies and writes
Time 3: Process B modifies and writes → OVERWRITES A's changes
```

**Why High**:
- Multiple processes/threads can access catalog
- orchestratord and pool-managerd both write
- Last write wins (data loss)
- No file locking

**Required Fix**:
```rust
use std::fs::{OpenOptions, File};
use std::io::Write;

fn save(&self) -> Result<()> {
    let index: BTreeMap<String, CatalogEntry> = self.load_index()?;
    let json = serde_json::to_string_pretty(&index)?;
    
    // Atomic write using rename
    let tmp_path = self.index_path.with_extension("tmp");
    
    {
        let mut file = File::create(&tmp_path)?;
        file.write_all(json.as_bytes())?;
        file.sync_all()?;  // Ensure data on disk
    }
    
    // Atomic rename (replaces old file)
    std::fs::rename(&tmp_path, &self.index_path)?;
    
    Ok(())
}
```

**Severity**: HIGH — Data corruption

---

### Vulnerability 9: Filesystem Paths Not Validated

**Location**: `libs/catalog-core/src/lib.rs:103-123`

**The Problem**: User-supplied paths used directly:

```rust
// libs/catalog-core/src/lib.rs:111
if let Some(p) = s.strip_prefix("file:") {
    return Ok(ModelRef::File { path: PathBuf::from(p) });  // ← NO VALIDATION
}

// libs/catalog-core/src/lib.rs:122
Ok(ModelRef::File { path: PathBuf::from(s) })  // ← NO VALIDATION
```

**Attack Scenario**:
```json
POST /v2/tasks
{
  "model_ref": "file:../../../../etc/passwd",  ← Directory traversal
  "prompt": "read this file"
}
```

**Why High**:
- Path traversal possible (`../../../`)
- No check if path is within allowed directory
- Could read arbitrary files
- Could write to arbitrary locations (if catalog creates files)

**Required Fix**:
```rust
use std::path::{Path, PathBuf};

pub fn validate_model_path(path: &Path) -> Result<PathBuf> {
    // Canonicalize to resolve .. and symlinks
    let canonical = path.canonicalize()
        .map_err(|_| CatalogError::InvalidRef("Invalid path".to_string()))?;
    
    // Check path is within allowed directory
    let allowed_root = PathBuf::from("/var/lib/llorch/models");
    if !canonical.starts_with(&allowed_root) {
        return Err(CatalogError::InvalidRef("Path outside allowed directory".to_string()));
    }
    
    Ok(canonical)
}

impl ModelRef {
    pub fn parse(s: &str) -> Result<Self> {
        if let Some(p) = s.strip_prefix("file:") {
            let path = PathBuf::from(p);
            let validated = validate_model_path(&path)?;
            return Ok(ModelRef::File { path: validated });
        }
        ...
    }
}
```

**Severity**: HIGH — Path traversal

---

### Vulnerability 10: No Input Sanitization on model_ref

**Location**: `orchestratord/src/api/data.rs:44-75`

**The Problem**: `model_ref` field accepted without validation:

```rust
// orchestratord/src/api/data.rs:44
pub async fn create_task(
    state: State<AppState>,
    Json(body): Json<api::TaskRequest>,  // ← body.model_ref not validated
) -> Result<impl IntoResponse, ErrO> {
    // Test sentinels (for BDD) use string comparison
    if body.model_ref == "pool-unavailable" {
        ...
    }
    // But no general validation!
}
```

**Attack Scenarios**:

**A. SQL injection (if using SQL backend)**:
```json
{"model_ref": "'; DROP TABLE models; --"}
```

**B. Command injection (if model_ref used in shell)**:
```json
{"model_ref": "model.gguf; rm -rf /"}
```

**C. Log injection**:
```json
{"model_ref": "model\n[ERROR] Fake error message\nmodel"}
```

**Why High**:
- No length limits on model_ref
- No character whitelist
- Could contain null bytes, newlines, shell metacharacters
- Passed to catalog, logs, metrics

**Required Fix**:
```rust
use regex::Regex;

const MAX_MODEL_REF_LEN: usize = 512;

fn validate_model_ref(s: &str) -> Result<(), String> {
    if s.is_empty() {
        return Err("model_ref cannot be empty".to_string());
    }
    
    if s.len() > MAX_MODEL_REF_LEN {
        return Err("model_ref too long".to_string());
    }
    
    // Check for null bytes
    if s.contains('\0') {
        return Err("model_ref contains null byte".to_string());
    }
    
    // Whitelist: alphanumeric, dash, underscore, slash, colon, dot
    let allowed = Regex::new(r"^[a-zA-Z0-9\-_/:\.]+$").unwrap();
    if !allowed.is_match(s) {
        return Err("model_ref contains invalid characters".to_string());
    }
    
    Ok(())
}

pub async fn create_task(...) -> Result<...> {
    // Validate model_ref
    validate_model_ref(&body.model_ref)
        .map_err(|e| ErrO::InvalidParams(e))?;
    
    // Proceed...
}
```

**Severity**: HIGH — Injection attacks

---

## MEDIUM Severity Issues

### Vulnerability 11: Global Metrics Mutexes Create Contention

**Location**: `orchestratord/src/metrics.rs:9-14`

**The Problem**: Global mutexes for all metrics:

```rust
static COUNTERS: Lazy<Mutex<HashMap<String, HashMap<LabelsKey, i64>>>> = ...;
static GAUGES: Lazy<Mutex<HashMap<String, HashMap<LabelsKey, i64>>>> = ...;
static HISTOGRAMS: Lazy<Mutex<HashMap<String, HashMap<LabelsKey, Histogram>>>> = ...;
```

**Why Medium**:
- Every metric update locks global mutex
- High-frequency metrics (e.g., per-token) cause contention
- Slow metric updates block request handlers
- Not a security issue but performance/availability concern

**Required Fix**: Use lock-free metrics (e.g., `metrics` crate with atomic counters)

**Severity**: MEDIUM — Performance bottleneck

---

### Vulnerability 12: SystemTime Without Error Handling

**Location**: `orchestratord/src/infra/clock.rs:9`

**The Problem**: SystemTime can fail if time goes backwards:

```rust
SystemTime::now().duration_since(UNIX_EPOCH).unwrap()  // ← PANICS
```

**Why Medium**:
- NTP adjustments can move time backwards
- `.unwrap()` panics entire service
- Rare but possible (time skew, VM migration, etc.)

**Required Fix**:
```rust
impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| {
                tracing::warn!("SystemTime before UNIX_EPOCH, using 0");
                Duration::from_secs(0)
            })
            .as_millis() as u64
    }
}
```

**Severity**: MEDIUM — Service crash (rare)

---

### Vulnerability 13: No TLS for Internal HTTP Clients

**Location**: `orchestratord/src/clients/pool_manager.rs:52`

**The Problem**: HTTP client uses plaintext:

```rust
client: reqwest::Client::builder()
    .timeout(std::time::Duration::from_secs(5))
    .build()
    .expect("failed to build reqwest client"),
```

**Why Medium**:
- orchestratord → pool-managerd traffic is plaintext HTTP
- Bearer tokens sent in cleartext
- Vulnerable to MitM on internal network
- Same issue as in architecture plan (Issue #2 from other audit)

**Required Fix**: Implement TLS (or mTLS) for internal clients

**Severity**: MEDIUM — Credential interception (internal network)

---

## Summary Table

| # | Vulnerability | Severity | Location | Impact |
|---|---------------|----------|----------|--------|
| 1 | Mutex poisoning crashes | CRITICAL | Throughout | Service crash |
| 2 | No rate limiting | CRITICAL | All endpoints | DoS |
| 3 | Token in environment | CRITICAL | auth_min.rs | Credential exposure |
| 4 | Duplicate auth logic | HIGH | api/nodes.rs | Auth bypass risk |
| 5 | Unbounded session HashMap | HIGH | services/session.rs | Memory exhaustion |
| 6 | Unbounded log vector | HIGH | state.rs | Memory exhaustion |
| 7 | Weak correlation IDs | HIGH | middleware.rs | Info disclosure |
| 8 | Catalog file race condition | HIGH | catalog-core | Data corruption |
| 9 | Path traversal | HIGH | catalog-core | File access |
| 10 | No model_ref validation | HIGH | api/data.rs | Injection |
| 11 | Metrics mutex contention | MEDIUM | metrics.rs | Performance |
| 12 | SystemTime panics | MEDIUM | infra/clock.rs | Rare crash |
| 13 | No TLS | MEDIUM | clients/* | Credential leak |

---

## Recommendations

### Immediate Fixes (Before Production)

**P0: Prevent Service Crashes**
1. Replace `.lock().unwrap()` with proper error handling (or use `parking_lot`)
2. Fix SystemTime unwrap in clock
3. Add session HashMap size limit
4. Add logs vector size limit

**P0: Prevent DoS**
5. Add rate limiting middleware (tower-governor)
6. Implement session expiry/cleanup

**P0: Prevent Credential Exposure**
7. Move API token from environment to file
8. Use systemd LoadCredential or secret manager

### High Priority Fixes

**P1: Code Quality**
9. Remove duplicate auth logic in `/v2/nodes/*`
10. Use atomic file writes in catalog

**P1: Input Validation**
11. Validate model_ref (length, chars, no injection)
12. Validate filesystem paths (no traversal)

**P1: Strengthen Auth**
13. Use cryptographically random correlation IDs

### Post-MVP Improvements

**P2: Performance**
14. Replace global metrics mutexes with lock-free solution

**P2: Defense in Depth**
15. Add TLS for internal HTTP clients

---

## Comparison with Worker-orcd Audit

**Existing codebase has better security than architecture plan**:
- ✅ Authentication implemented (orchestratord, pool-managerd)
- ✅ Timing-safe token comparison used
- ✅ Token fingerprinting prevents leakage
- ✅ No unsafe CUDA FFI (not applicable)

**But has operational security gaps**:
- ❌ Service reliability (mutex unwrap, unbounded memory)
- ❌ DoS protection (no rate limiting)
- ❌ Credential management (environment variables)

---

## Action Items

### For Development Team

- [ ] Review all 13 vulnerabilities
- [ ] Prioritize P0 fixes (items 1-8)
- [ ] Create tracking issues for each vulnerability
- [ ] Add to sprint backlog
- [ ] Update security checklist

### For Security Tests

- [ ] Add test: Trigger panic and verify no cascade
- [ ] Add test: Flood endpoint and verify rate limit
- [ ] Add test: Path traversal attempts rejected
- [ ] Add test: model_ref injection blocked
- [ ] Add test: Session HashMap bounded

---

**Audit completed**: 2025-10-01  
**Next review**: After P0 fixes implemented

---

## ADDENDUM: Vulnerabilities Found During Clippy Audit

**Date**: 2025-10-01  
**Context**: Security Overseer auditing Clippy configurations

### Vulnerability 14: Queue Integer Overflow in Task IDs

**Location**: `libs/orchestrator-core/src/queue.rs`

**The Problem**: Task IDs are u32, can overflow after 4 billion tasks:

```rust
pub fn enqueue(&mut self, id: u32, prio: Priority) -> Result<(), EnqueueError> {
    // What happens when u32 wraps around?
}
```

**Attack Scenario**: ID collision after 4B tasks → wrong task canceled/queried

**Severity**: LOW (requires billions of tasks)

---

### Vulnerability 15: Unbounded Queue Snapshot Allocations

**Location**: `libs/orchestrator-core/src/queue.rs:87`

**The Problem**: snapshot_priority() clones entire queue without limits:

```rust
pub fn snapshot_priority(&self, prio: Priority) -> Vec<u32> {
    self.interactive.iter().copied().collect()  // ← Allocates 10k-item Vec
}
```

**Attack Scenario**: Repeated snapshots of 10k-item queue cause memory pressure

**Required Fix**: Return iterator or limit snapshot size

**Severity**: MEDIUM — Memory pressure

---

### Vulnerability 16: No Validation on bind_addr in Config

**Location**: `bin/pool-managerd/src/config.rs` (inferred)

**The Problem**: No validation of bind_addr from environment:

```rust
let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
// No validation that bind_addr is sane
```

**Attack Scenario**: 
```bash
POOL_MANAGERD_BIND_ADDR="0.0.0.0:22" cargo run  # Tries to bind SSH port
```

**Required Fix**: Validate addr format and port range (1024-65535 recommended)

**Severity**: LOW — Operational risk

---

### Vulnerability 17: catalog-core Path Traversal

**Already documented** as Vulnerability #9 in main audit.

**Severity**: HIGH

---

### Vulnerability 18: ModelRef Parsing Injection

**Location**: `libs/catalog-core/src/lib.rs:103-123`

**The Problem**: ModelRef::parse() accepts traversal sequences:

```rust
pub fn parse(s: &str) -> Result<Self> {
    if let Some(rest) = s.strip_prefix("hf:") {
        let mut parts = rest.splitn(3, '/');
        let org = parts.next().ok_or_else(...)?;
        // NO validation on org/repo names
        // NO rejection of ../
    }
}
```

**Attack Scenario**:
```json
POST /v2/tasks {"model_ref": "hf:../../../etc:passwd/shadow"}
```

**Required Fix**:
```rust
fn validate_model_component(s: &str) -> Result<()> {
    const MAX_LEN: usize = 256;
    if s.is_empty() || s.len() > MAX_LEN {
        return Err(CatalogError::InvalidRef("Invalid length".into()));
    }
    
    // Only allow alphanumeric, dash, underscore, dot
    let valid = s.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.');
    if !valid || s.contains("..") {
        return Err(CatalogError::InvalidRef("Invalid characters".into()));
    }
    Ok(())
}
```

**Severity**: HIGH — Path traversal/injection

---

### Vulnerability 19: Proof Bundle Path Validation

**Location**: `libs/proof-bundle/src/` (environment variable handling)

**The Problem**: LLORCH_PROOF_DIR used without validation:

```rust
// If env var set, writes go there
// What if LLORCH_PROOF_DIR="/tmp/../../etc/cron.d"?
```

**Required Fix**: Validate that path is absolute and within safe bounds

**Severity**: MEDIUM — Arbitrary file write

---

## Updated Vulnerability Count

**Total**: 19 vulnerabilities (13 original + 6 new)

**By Severity**:
- CRITICAL: 3
- HIGH: 9  
- MEDIUM: 4
- LOW: 3
