# TEAM-131: Shared Crate Usage Audit

**Binary:** rbee-hive  
**Date:** 2025-10-19  
**Auditor:** TEAM-131

---

## AUDIT SUMMARY

**Total Shared Crates Declared:** 9  
**Actually Used:** 5 ✅  
**Unused (Declared but not imported):** 3 ⚠️  
**Missing Opportunities:** 2 💡

---

## ✅ WELL-USED SHARED CRATES

### 1. hive-core
**Status:** ✅ **ACTIVELY USED**  
**Declared in:** `Cargo.toml` line 21  
**Used by:**
- `src/registry.rs` - Uses core worker types
- `src/http/routes.rs` - Uses shared state types

**Coverage:** Good  
**Recommendation:** None - continue using as is

---

### 2. model-catalog
**Status:** ✅ **ACTIVELY USED**  
**Declared in:** `Cargo.toml` line 22  
**Used by:**
- `src/provisioner/catalog.rs` - Model lookup and registration
- `src/provisioner/operations.rs` - Model metadata access
- `src/http/workers.rs` - Model provisioning integration
- `src/commands/models.rs` - CLI model operations

**Usage Pattern:**
```rust
use model_catalog::{ModelCatalog, ModelInfo};

let catalog = ModelCatalog::open(db_path).await?;
let model = catalog.find_model(reference, provider).await?;
catalog.register_model(&model_info).await?;
```

**Coverage:** Excellent  
**Recommendation:** None - proper integration

---

### 3. gpu-info
**Status:** ✅ **ACTIVELY USED**  
**Declared in:** `Cargo.toml` line 23  
**Used by:**
- `src/resources.rs` - VRAM checks and GPU validation
- `src/commands/detect.rs` - Backend detection

**Usage Pattern:**
```rust
use gpu_info::detect_gpus;

let gpu_info = detect_gpus();
if !gpu_info.available {
    anyhow::bail!("No GPU detected");
}
let gpu_device = gpu_info.validate_device(device)?;
```

**Coverage:** Good  
**Recommendation:** None - proper usage for GPU operations

---

### 4. auth-min
**Status:** ✅ **ACTIVELY USED**  
**Declared in:** `Cargo.toml` line 56  
**Used by:**
- `src/http/middleware/auth.rs` - Authentication middleware

**Usage Pattern:**
```rust
use auth_min::verify_token;

pub async fn auth_middleware(
    TypedHeader(authorization): TypedHeader<Authorization<Bearer>>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    verify_token(authorization.token())?;
    Ok(next.run(request).await)
}
```

**Coverage:** Good - All HTTP endpoints protected  
**Recommendation:** Verify all sensitive endpoints use middleware

---

### 5. input-validation
**Status:** ✅ **ACTIVELY USED**  
**Declared in:** `Cargo.toml` line 58  
**Used by:**
- `src/http/workers.rs` - Validate spawn requests (TEAM-103)
- `src/http/models.rs` - Validate model operations

**Usage Pattern:**
```rust
use input_validation::{validate_identifier, validate_model_ref};

validate_model_ref(&request.model_ref)
    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid model_ref: {}", e)))?;

validate_identifier(&request.backend, 64)
    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid backend: {}", e)))?;
```

**Coverage:** Good  
**Added by:** TEAM-103  
**Recommendation:** None - proper validation applied

---

## ⚠️ UNUSED SHARED CRATES

### 6. secrets-management
**Status:** ⚠️ **DECLARED BUT NOT USED**  
**Declared in:** `Cargo.toml` line 57  
**Import searches:** 0 matches for `use secrets_management`  
**grep results:** No usage found in any source file

**Evidence:**
```bash
$ grep -r "secrets_management" src/
# No matches found
```

**Recommendation:** 🔴 **REMOVE from Cargo.toml**
```toml
# DELETE THIS LINE:
secrets-management = { path = "../shared-crates/secrets-management" }
```

**Rationale:** 
- No secrets currently managed by rbee-hive
- Worker processes don't use API keys
- If secrets needed in future, add dependency then

---

### 7. audit-logging
**Status:** ⚠️ **DECLARED BUT NOT USED**  
**Declared in:** `Cargo.toml` line 61  
**Import searches:** 0 matches for `use audit_logging`  
**grep results:** No usage found in any source file

**Evidence:**
```bash
$ grep -r "audit_logging" src/
# No matches found
```

**Recommendation:** 🟡 **ADD USAGE** (Don't remove - should be used!)

**Where to add:**
1. **Worker spawn** - `src/http/workers.rs:handle_spawn_worker()`
   ```rust
   use audit_logging::log_event;
   
   log_event(audit_logging::Event {
       action: "worker.spawned",
       resource_id: &worker_id,
       user_id: Some(&user_id),  // Extract from auth
       metadata: json!({
           "model_ref": &request.model_ref,
           "backend": &request.backend,
           "device": request.device,
       }),
   }).await?;
   ```

2. **Worker stop** - `src/http/workers.rs` (when stop endpoint added)
   ```rust
   log_event(audit_logging::Event {
       action: "worker.stopped",
       resource_id: worker_id,
       user_id: Some(&user_id),
       metadata: json!({"reason": reason}),
   }).await?;
   ```

3. **Model download** - `src/provisioner/download.rs`
   ```rust
   log_event(audit_logging::Event {
       action: "model.downloaded",
       resource_id: &model_ref,
       user_id: None,  // System action
       metadata: json!({"size_bytes": size, "duration_secs": duration}),
   }).await?;
   ```

**Impact:** Medium - Audit trail is important for production systems

---

### 8. deadline-propagation
**Status:** ⚠️ **DECLARED BUT NOT USED**  
**Declared in:** `Cargo.toml` line 62  
**Import searches:** 0 matches for `use deadline_propagation`  
**grep results:** No usage found in any source file

**Evidence:**
```bash
$ grep -r "deadline_propagation" src/
# No matches found
```

**Recommendation:** 🟡 **ADD USAGE** (Don't remove - should be used!)

**Where to add:**
1. **HTTP handler context** - `src/http/routes.rs`
   ```rust
   use deadline_propagation::{extract_deadline, DeadlineLayer};
   
   // Add middleware:
   let app = Router::new()
       .route("/v1/workers/spawn", post(handle_spawn_worker))
       .layer(DeadlineLayer::new())
       .with_state(state);
   ```

2. **Pass deadline to provisioner** - `src/http/workers.rs`
   ```rust
   use deadline_propagation::Deadline;
   
   pub async fn handle_spawn_worker(
       Extension(deadline): Extension<Deadline>,
       State(state): State<AppState>,
       Json(request): Json<SpawnWorkerRequest>,
   ) -> Result<...> {
       // Pass deadline to download:
       state.provisioner
           .download_model_with_deadline(&reference, &provider, deadline)
           .await?;
   }
   ```

3. **Timeout enforcement** - `src/provisioner/download.rs`
   ```rust
   use deadline_propagation::Deadline;
   
   pub async fn download_model_with_deadline(
       &self,
       reference: &str,
       provider: &str,
       deadline: Deadline,
   ) -> Result<PathBuf> {
       let timeout = deadline.remaining()?;
       tokio::time::timeout(timeout, self.download_impl(reference, provider)).await??;
   }
   ```

**Impact:** Medium - Timeout propagation prevents hung requests

---

## 💡 MISSING SHARED CRATE OPPORTUNITIES

### 9. HTTP Client Wrapper (DUPLICATE CODE FOUND)

**Issue:** Inconsistent HTTP client usage across multiple modules

**Current Pattern:**
```rust
// In monitor.rs:
let client = reqwest::Client::new();
match client.get(format!("{}/v1/health", worker.url))
    .timeout(Duration::from_secs(5))
    .send()
    .await { ... }

// In shutdown.rs:
let client = reqwest::Client::new();
let response = client
    .post(format!("{}/v1/shutdown", worker_url))
    .timeout(timeout)
    .send()
    .await?;

// In http/workers.rs:
let client = reqwest::Client::new();
// More similar patterns...
```

**Problems:**
- Timeout values scattered (5s, variable)
- No retry logic
- No circuit breaker
- No connection pooling
- Duplicate error handling

**Recommendation:** 🟢 **CREATE `rbee-http-client` shared crate**

**Proposed API:**
```rust
// New shared crate: bin/shared-crates/rbee-http-client
pub struct RbeeHttpClient {
    client: reqwest::Client,
    retry_policy: RetryPolicy,
    circuit_breaker: CircuitBreaker,
}

impl RbeeHttpClient {
    pub fn new(config: HttpClientConfig) -> Self;
    
    // Health check helper
    pub async fn check_health(&self, url: &str) -> Result<HealthResponse>;
    
    // Shutdown helper
    pub async fn send_shutdown(&self, url: &str) -> Result<()>;
    
    // Generic request with retry
    pub async fn request<T: DeserializeOwned>(
        &self,
        method: Method,
        url: &str,
    ) -> Result<T>;
}

pub struct HttpClientConfig {
    pub timeout_secs: u64,
    pub max_retries: u32,
    pub circuit_breaker_threshold: u32,
}
```

**Usage in rbee-hive:**
```rust
use rbee_http_client::RbeeHttpClient;

let http_client = RbeeHttpClient::new(HttpClientConfig::default());

// In monitor.rs:
match http_client.check_health(&worker.url).await { ... }

// In shutdown.rs:
http_client.send_shutdown(&worker_url).await?;
```

**Benefits:**
- Consistent behavior across all HTTP calls
- Centralized retry/timeout logic
- Circuit breaker prevents cascade failures
- Connection pooling improves performance
- Easier to test with mocks

**Scope:** Medium - 200-300 LOC  
**Timeline:** 2-3 days  
**Priority:** Medium (can be done after decomposition)

---

### 10. Observability Enhancement (UNDER-UTILIZED)

**Issue:** `narration-core` exists but not fully utilized

**Current Usage:**
- Basic `tracing` macros (info!, warn!, error!)
- No structured events
- No trace ID correlation
- No span timing

**Recommendation:** 🟢 **ENHANCE observability**

**Add structured events:**
```rust
use narration_core::{trace_event, span_timing};

// Worker spawn event:
trace_event!("worker.spawn.started", {
    worker_id: &worker_id,
    model_ref: &request.model_ref,
    backend: &request.backend,
});

// Span timing:
let _span = span_timing!("model.download");
provisioner.download_model(reference, provider).await?;
// Automatically records duration when span dropped
```

**Add trace ID propagation:**
```rust
use narration_core::TraceContext;

// Extract from HTTP headers:
let trace_ctx = TraceContext::from_headers(&headers);

// Pass through call chain:
provisioner.download_model_traced(reference, provider, trace_ctx).await?;
```

**Priority:** Low (nice to have)  
**Timeline:** 1-2 days

---

## ACTION ITEMS

### 🔴 High Priority (Before Decomposition)

1. **Remove unused dependency:**
   ```toml
   # In Cargo.toml, DELETE:
   secrets-management = { path = "../shared-crates/secrets-management" }
   ```

2. **Add audit logging:**
   - Worker spawn event
   - Worker stop event  
   - Model download event
   - Estimated: 2-3 hours

3. **Add deadline propagation:**
   - HTTP middleware
   - Provisioner integration
   - Timeout enforcement
   - Estimated: 3-4 hours

### 🟡 Medium Priority (During Decomposition)

4. **Create rbee-http-client:**
   - Extract HTTP client patterns
   - Add retry/circuit-breaker
   - Refactor monitor/shutdown/workers
   - Estimated: 2-3 days
   - Can be done in parallel with decomposition

### 🟢 Low Priority (After Decomposition)

5. **Enhance observability:**
   - Add structured events
   - Add trace ID propagation
   - Add span timing
   - Estimated: 1-2 days

---

## VERIFICATION CHECKLIST

- [x] All Cargo.toml dependencies analyzed
- [x] grep search for each shared crate usage
- [x] Identified 5 actively used crates
- [x] Identified 3 unused crates (1 remove, 2 add usage)
- [x] Identified 2 missing opportunities
- [x] Provided concrete recommendations
- [x] Estimated impact and timeline

---

## APPENDIX: Search Commands Used

```bash
# Check for each shared crate usage:
grep -r "use hive_core" src/
grep -r "use model_catalog" src/
grep -r "use gpu_info" src/
grep -r "use auth_min" src/
grep -r "use secrets_management" src/  # 0 matches
grep -r "use input_validation" src/
grep -r "use audit_logging" src/  # 0 matches
grep -r "use deadline_propagation" src/  # 0 matches

# Check HTTP client patterns:
grep -r "reqwest::Client" src/
# Found in: monitor.rs, shutdown.rs, http/workers.rs

# Count usages:
grep -c "use model_catalog" src/**/*.rs
```

**Audit Complete:** 2025-10-19  
**Auditor:** TEAM-131
