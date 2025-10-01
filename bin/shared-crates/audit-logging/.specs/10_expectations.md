# Audit Logging — Consumer Expectations

**Status**: Draft  
**Purpose**: Documents what other crates expect from `audit-logging`  
**Last Updated**: 2025-10-01

---

## 0. Overview

This document catalogs the expectations and dependencies that other llama-orch crates have on `audit-logging`. It serves as a contract specification to guide implementation priorities and API stability.

**Consuming crates**:
- `orchestratord` — Authentication, task lifecycle, node management
- `pool-managerd` — Pool lifecycle, resource operations
- `worker-orcd` — VRAM operations, security incidents
- `vram-residency` — VRAM seal/verify operations (detailed in `11_worker_vram_residency.md`)
- `auth-min` — Authentication event emission
- `input-validation` — Sanitization for audit data
- All services — Security incident reporting

---

## 1. Core Type Expectations

### 1.1 `AuditLogger` Struct

**Required by**: All services

**Expected public interface**:
```rust
pub struct AuditLogger {
    // Internal state (not exposed)
}

impl AuditLogger {
    pub fn new(config: AuditConfig) -> Result<Self, AuditError>;
    pub async fn emit(&self, event: AuditEvent) -> Result<(), AuditError>;
    pub async fn query(&self, query: AuditQuery) -> Result<Vec<AuditEventEnvelope>, AuditError>;
    pub async fn verify_integrity(&self, options: VerifyOptions) -> Result<VerifyResult, AuditError>;
    pub async fn flush(&self) -> Result<(), AuditError>;
}
```

**Usage contexts**:
- **Service initialization**: Create logger at startup
- **Middleware**: Emit auth events from authentication middleware
- **Handlers**: Emit events from API handlers
- **Shutdown**: Flush buffered events on graceful shutdown

**Performance requirements**:
- MUST be async (non-blocking)
- MUST support concurrent emission from multiple tasks
- MUST buffer events (1000 events or 10MB)
- MUST batch writes (flush every 1 second or 100 events)

---

### 1.2 `AuditEvent` Enum

**Required by**: All services

**Expected variants** (see `10-event-types.md` for complete list):
```rust
pub enum AuditEvent {
    // Authentication (4 variants)
    AuthSuccess { timestamp, actor, method, path, service_id },
    AuthFailure { timestamp, attempted_user, reason, ip, path, service_id },
    TokenCreated { timestamp, actor, token_fingerprint, scope, expires_at },
    TokenRevoked { timestamp, actor, token_fingerprint, reason },
    
    // Authorization (3 variants)
    AuthorizationGranted { timestamp, actor, resource, action },
    AuthorizationDenied { timestamp, actor, resource, action, reason },
    PermissionChanged { timestamp, actor, subject, old_permissions, new_permissions },
    
    // Resource Operations (8 variants)
    PoolCreated { timestamp, actor, pool_id, model_ref, node_id, replicas, gpu_devices },
    PoolDeleted { timestamp, actor, pool_id, model_ref, node_id, reason, replicas_terminated },
    PoolModified { timestamp, actor, pool_id, changes },
    NodeRegistered { timestamp, actor, node_id, gpu_count, total_vram_gb, capabilities },
    NodeDeregistered { timestamp, actor, node_id, reason, pools_affected },
    TaskSubmitted { timestamp, actor, task_id, model_ref, prompt_length, prompt_hash, max_tokens },
    TaskCompleted { timestamp, task_id, worker_id, tokens_generated, duration_ms, result },
    TaskCanceled { timestamp, actor, task_id, reason },
    
    // VRAM Operations (6 variants - see 11_worker_vram_residency.md)
    VramSealed { timestamp, shard_id, gpu_device, vram_bytes, digest, worker_id },
    SealVerified { timestamp, shard_id, worker_id },
    SealVerificationFailed { timestamp, shard_id, reason, expected_digest, actual_digest, worker_id, severity },
    VramAllocated { timestamp, requested_bytes, allocated_bytes, available_bytes, used_bytes, gpu_device, worker_id },
    VramAllocationFailed { timestamp, requested_bytes, available_bytes, reason, gpu_device, worker_id },
    VramDeallocated { timestamp, shard_id, freed_bytes, remaining_used, gpu_device, worker_id },
    
    // Security Incidents (5 variants)
    RateLimitExceeded { timestamp, ip, endpoint, limit, actual, window_seconds },
    PathTraversalAttempt { timestamp, actor, attempted_path, endpoint },
    InvalidTokenUsed { timestamp, ip, token_prefix, endpoint },
    PolicyViolation { timestamp, policy, violation, details, severity, worker_id, action_taken },
    SuspiciousActivity { timestamp, actor, activity_type, details, risk_score },
    
    // Data Access (3 variants - GDPR)
    InferenceExecuted { timestamp, customer_id, job_id, model_ref, tokens_processed, provider_id, result },
    ModelAccessed { timestamp, customer_id, model_ref, access_type, provider_id },
    DataDeleted { timestamp, customer_id, data_types, reason },
    
    // Compliance (3 variants - platform mode)
    GdprDataAccessRequest { timestamp, customer_id, requester, scope },
    GdprDataExport { timestamp, customer_id, data_types, export_format, file_hash },
    GdprRightToErasure { timestamp, customer_id, completed_at, data_types_deleted },
}
```

**Expected traits**:
- `Debug`, `Clone` (for logging and testing)
- `Serialize`, `Deserialize` (via serde)

---

### 1.3 `AuditConfig` Struct

**Required by**: All services

**Expected public fields**:
```rust
pub struct AuditConfig {
    pub mode: AuditMode,
    pub service_id: String,
    pub rotation_policy: RotationPolicy,
    pub retention_policy: RetentionPolicy,
}

pub enum AuditMode {
    Local {
        base_dir: PathBuf,
    },
    Platform(PlatformConfig),
}

pub struct PlatformConfig {
    pub endpoint: String,
    pub provider_id: String,
    pub provider_key: Vec<u8>,
    pub batch_size: usize,
    pub flush_interval: Duration,
}

pub enum RotationPolicy {
    Daily,
    SizeLimit(usize),
    Both { daily: bool, size_limit: usize },
}

pub struct RetentionPolicy {
    pub min_retention_days: u32,
    pub archive_after_days: u32,
    pub delete_after_days: u32,
}
```

**Usage**:
- Services configure logger at startup
- Local mode for single-node deployments
- Platform mode for marketplace deployments

---

### 1.4 Supporting Types

**Required by**: All services

```rust
pub struct ActorInfo {
    pub user_id: String,           // "admin@example.com" or "token:a3f2c1"
    pub ip: Option<IpAddr>,        // Source IP address
    pub auth_method: AuthMethod,   // BearerToken, ApiKey, mTLS
    pub session_id: Option<String>, // For correlation
}

pub enum AuthMethod {
    BearerToken,
    ApiKey,
    MTls,
    Internal,  // Service-to-service
}

pub struct ResourceInfo {
    pub resource_type: String,  // "pool", "node", "job", "shard"
    pub resource_id: String,    // "pool-123", "shard-abc123"
    pub parent_id: Option<String>, // "node-1" (parent of pool)
}

pub enum AuditResult {
    Success,
    Failure { reason: String },
    PartialSuccess { details: String },
}
```

---

## 2. orchestratord Expectations

### 2.1 Authentication Middleware Integration

**Required by**: `bin/orchestratord/src/app/auth_min.rs`

**Expected usage**:
```rust
pub async fn bearer_auth_middleware(
    State(state): State<Arc<AppState>>,
    mut req: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let auth_header = req.headers().get("authorization")
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    let token = auth_min::parse_bearer(auth_header)
        .map_err(|_| StatusCode::UNAUTHORIZED)?;
    
    let expected = state.config.api_token.as_ref()
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    if !auth_min::timing_safe_eq(token.as_bytes(), expected.as_bytes()) {
        // Audit failure
        state.audit_logger.emit(AuditEvent::AuthFailure {
            timestamp: Utc::now(),
            attempted_user: None,
            reason: "invalid_token".to_string(),
            ip: extract_ip(&req),
            path: req.uri().path().to_string(),
            service_id: "orchestratord".to_string(),
        }).await.ok();  // Don't block on audit failure
        
        return Err(StatusCode::UNAUTHORIZED);
    }
    
    // Audit success
    let token_fp = auth_min::fingerprint_token(token);
    state.audit_logger.emit(AuditEvent::AuthSuccess {
        timestamp: Utc::now(),
        actor: ActorInfo {
            user_id: format!("token:{}", token_fp),
            ip: Some(extract_ip(&req)),
            auth_method: AuthMethod::BearerToken,
            session_id: None,
        },
        method: AuthMethod::BearerToken,
        path: req.uri().path().to_string(),
        service_id: "orchestratord".to_string(),
    }).await.ok();
    
    Ok(next.run(req).await)
}
```

**Expected behavior**:
- Emit `AuthSuccess` on successful authentication
- Emit `AuthFailure` on failed authentication
- Use token fingerprints (never log full tokens)
- Non-blocking emission (use `.ok()` to ignore errors)

---

### 2.2 Task Lifecycle Events

**Required by**: `bin/orchestratord/src/api/data.rs`

**Expected usage**:
```rust
pub async fn create_task(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TaskRequest>,
) -> Result<Json<TaskResponse>, ErrO> {
    // Validate and create task
    let task = state.orchestrator.submit_task(req).await?;
    
    // Audit task submission
    state.audit_logger.emit(AuditEvent::TaskSubmitted {
        timestamp: Utc::now(),
        actor: ActorInfo {
            user_id: extract_user_from_request(&state),
            ip: None,  // TODO: Extract from request
            auth_method: AuthMethod::BearerToken,
            session_id: None,
        },
        task_id: task.task_id.clone(),
        model_ref: task.model_ref.clone(),
        prompt_length: task.prompt.len(),
        prompt_hash: compute_sha256(&task.prompt),  // Never log prompt content
        max_tokens: task.max_tokens,
    }).await.ok();
    
    Ok(Json(TaskResponse { task }))
}
```

**Expected events**:
- `TaskSubmitted` — When task is created
- `TaskCompleted` — When task finishes successfully
- `TaskCanceled` — When task is canceled

**Security requirements**:
- Never log prompt content (use length and hash)
- Sanitize task_id and model_ref with `input-validation`

---

### 2.3 Node Management Events

**Required by**: `bin/orchestratord/src/api/nodes.rs`

**Expected usage**:
```rust
pub async fn register_node(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<RegisterResponse>, StatusCode> {
    // Register node
    let node = state.node_registry.register(req).await?;
    
    // Audit node registration
    state.audit_logger.emit(AuditEvent::NodeRegistered {
        timestamp: Utc::now(),
        actor: ActorInfo {
            user_id: "system".to_string(),  // Internal operation
            ip: None,
            auth_method: AuthMethod::Internal,
            session_id: None,
        },
        node_id: node.node_id.clone(),
        gpu_count: node.gpu_count,
        total_vram_gb: node.total_vram_gb,
        capabilities: node.capabilities.clone(),
    }).await.ok();
    
    Ok(Json(RegisterResponse { node }))
}
```

**Expected events**:
- `NodeRegistered` — When node joins cluster
- `NodeDeregistered` — When node leaves cluster

---

## 3. pool-managerd Expectations

### 3.1 Pool Lifecycle Events

**Required by**: `bin/pool-managerd/src/api/pools.rs`

**Expected usage**:
```rust
pub async fn create_pool(
    State(state): State<Arc<PoolState>>,
    Json(req): Json<CreatePoolRequest>,
) -> Result<Json<CreatePoolResponse>, PoolError> {
    // Create pool
    let pool = state.pool_manager.create_pool(req).await?;
    
    // Audit pool creation
    state.audit_logger.emit(AuditEvent::PoolCreated {
        timestamp: Utc::now(),
        actor: ActorInfo {
            user_id: "admin".to_string(),  // TODO: Extract from auth
            ip: None,
            auth_method: AuthMethod::BearerToken,
            session_id: None,
        },
        pool_id: pool.pool_id.clone(),
        model_ref: pool.model_ref.clone(),
        node_id: pool.node_id.clone(),
        replicas: pool.replicas,
        gpu_devices: pool.gpu_devices.clone(),
    }).await?;
    
    Ok(Json(CreatePoolResponse { pool }))
}

pub async fn delete_pool(
    State(state): State<Arc<PoolState>>,
    Path(pool_id): Path<String>,
) -> Result<StatusCode, PoolError> {
    // Get pool info before deletion
    let pool = state.pool_manager.get_pool(&pool_id).await?;
    
    // Delete pool
    state.pool_manager.delete_pool(&pool_id).await?;
    
    // Audit pool deletion
    state.audit_logger.emit(AuditEvent::PoolDeleted {
        timestamp: Utc::now(),
        actor: ActorInfo {
            user_id: "admin".to_string(),
            ip: None,
            auth_method: AuthMethod::BearerToken,
            session_id: None,
        },
        pool_id: pool.pool_id.clone(),
        model_ref: pool.model_ref.clone(),
        node_id: pool.node_id.clone(),
        reason: "user_requested".to_string(),
        replicas_terminated: pool.replicas,
    }).await?;
    
    Ok(StatusCode::NO_CONTENT)
}
```

**Expected events**:
- `PoolCreated` — When pool is created
- `PoolDeleted` — When pool is deleted
- `PoolModified` — When pool configuration changes

---

### 3.2 Authentication Events

**Required by**: `bin/pool-managerd/src/api/auth.rs`

**Expected usage**: Same pattern as orchestratord (see §2.1)

**Status**: ⚠️ pool-managerd authentication not yet implemented (per SEC-AUTH-3002)

**When implementing**:
- Add auth middleware (pattern exists in orchestratord)
- Emit `AuthSuccess` and `AuthFailure` events
- Use token fingerprints

---

## 4. worker-orcd Expectations

### 4.1 VRAM Operation Events

**Required by**: `bin/worker-orcd-crates/vram-residency`

**See**: `11_worker_vram_residency.md` for detailed requirements

**Summary**:
- `VramSealed` — When model is sealed in VRAM
- `SealVerified` — When seal verification passes
- `SealVerificationFailed` — When seal verification fails (CRITICAL)
- `VramAllocated` — When VRAM is allocated
- `VramAllocationFailed` — When VRAM allocation fails (OOM)
- `VramDeallocated` — When VRAM is freed

**Security requirements**:
- Never log VRAM pointers
- Always log digest for forensics
- Immediate flush on seal verification failure

---

### 4.2 Security Incident Events

**Required by**: All worker-orcd crates

**Expected usage**:
```rust
// Policy violation detection
if unified_memory_detected()? {
    audit_logger.emit(AuditEvent::PolicyViolation {
        timestamp: Utc::now(),
        policy: "vram_only".to_string(),
        violation: "unified_memory_detected".to_string(),
        details: "UMA enabled, cannot enforce VRAM-only policy".to_string(),
        severity: "critical".to_string(),
        worker_id: self.worker_id.clone(),
        action_taken: "worker_stopped".to_string(),
    }).await?;
    
    return Err(WorkerError::PolicyViolation);
}
```

**Expected events**:
- `PolicyViolation` — When security policy is violated
- `SuspiciousActivity` — When anomalous behavior detected

---

## 5. input-validation Integration

### 5.1 Sanitization Before Logging

**Required by**: All services

**Expected usage**:
```rust
use input_validation::sanitize_string;

// Sanitize user-controlled data before audit logging
let safe_user_id = sanitize_string(&user_id)?;
let safe_resource_id = sanitize_string(&resource_id)?;

audit_logger.emit(AuditEvent::PoolDeleted {
    actor: ActorInfo {
        user_id: safe_user_id,  // ✅ Sanitized
        ...
    },
    pool_id: safe_resource_id,  // ✅ Sanitized
    ...
}).await?;
```

**Why required**:
- Prevents log injection attacks (ANSI escape sequences, control characters)
- Prevents Unicode directional override attacks
- Ensures audit logs are machine-readable

**Integration points**:
- All `user_id` fields
- All `resource_id` fields (pool_id, node_id, task_id, shard_id)
- All `reason` and `details` fields
- Any user-supplied strings

**See**: `bin/shared-crates/input-validation/.specs/10_expectations.md` §4.5

---

## 6. auth-min Integration

### 6.1 Token Fingerprinting

**Required by**: All services with authentication

**Expected usage**:
```rust
use auth_min::fingerprint_token;

// Never log full token
let token_fp = fingerprint_token(token);

audit_logger.emit(AuditEvent::AuthSuccess {
    actor: ActorInfo {
        user_id: format!("token:{}", token_fp),  // ✅ Fingerprint only
        ...
    },
    ...
}).await?;
```

**Why required**:
- Prevents credential leakage in audit logs
- Enables correlation without exposing secrets
- Complies with security best practices

**See**: `bin/shared-crates/auth-min/README.md` — Token Fingerprinting section

---

## 7. Error Handling Expectations

### 7.1 `AuditError` Enum

**Required by**: All consuming crates

**Expected variants**:
```rust
pub enum AuditError {
    BufferFull,
    Io(std::io::Error),
    Serialization(serde_json::Error),
    InvalidChain(String),
    BrokenChain(String),
    ChecksumMismatch { file: String, expected: String, actual: String },
    MissingSignature,
    InvalidSignature { audit_id: String },
}
```

**Expected traits**:
- `Error`, `Debug`, `Display` (via thiserror)
- Structured error messages for operators

**Error handling patterns**:
```rust
// Non-critical event: Log error but continue
if let Err(e) = audit_logger.emit(event).await {
    tracing::error!(error = %e, "Failed to emit audit event");
}

// Critical event: Propagate error
audit_logger.emit(AuditEvent::SealVerificationFailed { ... }).await?;
```

---

## 8. Performance Requirements

### 8.1 Non-Blocking Emission

**Required by**: All services

**Expected behavior**:
- Audit emission MUST NOT block operations
- Use async/await for all operations
- Buffer events in memory (max 1000 events or 10MB)
- Batch writes (flush every 1 second or 100 events)

**Implementation pattern**:
```rust
// ✅ CORRECT: Non-blocking
tokio::spawn({
    let logger = self.audit_logger.clone();
    async move {
        logger.emit(event).await.ok();
    }
});

// ❌ WRONG: Blocking
self.audit_logger.emit(event).await?;  // Blocks VRAM operation
```

---

### 8.2 Graceful Degradation

**Required by**: All services

**Expected behavior**:
- If buffer is full, drop events (log warning)
- If disk is full, log error but don't crash
- If platform endpoint unreachable, buffer locally
- Flush all buffered events on graceful shutdown

---

## 9. Query API Expectations

### 9.1 Query by Actor

**Required by**: Admin tools, compliance reports

**Expected usage**:
```rust
let events = audit_logger.query(AuditQuery {
    actor: Some("admin@example.com".to_string()),
    start_time: Some(Utc::now() - Duration::days(7)),
    end_time: Some(Utc::now()),
    event_types: vec![],  // All types
    limit: 100,
}).await?;
```

---

### 9.2 Query by Event Type

**Required by**: Security monitoring, incident response

**Expected usage**:
```rust
let events = audit_logger.query(AuditQuery {
    actor: None,
    start_time: Some(Utc::now() - Duration::hours(1)),
    end_time: Some(Utc::now()),
    event_types: vec![
        "auth.failure".to_string(),
        "security.rate_limit_exceeded".to_string(),
    ],
    limit: 1000,
}).await?;
```

---

### 9.3 Query by Resource

**Required by**: Resource lifecycle tracking

**Expected usage**:
```rust
let events = audit_logger.query(AuditQuery {
    resource_id: Some("pool-123".to_string()),
    start_time: Some(Utc::now() - Duration::days(30)),
    end_time: Some(Utc::now()),
    event_types: vec![
        "pool.created".to_string(),
        "pool.deleted".to_string(),
        "pool.modified".to_string(),
    ],
    limit: 100,
}).await?;
```

---

## 10. Integrity Verification Expectations

### 10.1 Hash Chain Verification

**Required by**: Admin tools, compliance audits

**Expected usage**:
```rust
// Verify last 1000 events
let result = audit_logger.verify_integrity(VerifyOptions {
    mode: VerifyMode::LastN(1000),
}).await?;

match result {
    VerifyResult::Valid => println!("✅ Integrity verified"),
    VerifyResult::Invalid { broken_at } => {
        eprintln!("❌ Hash chain broken at event: {}", broken_at);
    }
}
```

---

### 10.2 File Checksum Verification

**Required by**: Admin tools, archival processes

**Expected usage**:
```rust
let result = audit_logger.verify_file_checksums().await?;

for file_result in result.files {
    match file_result.status {
        ChecksumStatus::Valid => {
            println!("✅ {}: valid", file_result.filename);
        }
        ChecksumStatus::Invalid { expected, actual } => {
            eprintln!("❌ {}: checksum mismatch", file_result.filename);
        }
    }
}
```

---

## 11. Configuration Expectations

### 11.1 Environment Variables

**Expected environment variables**:
```bash
# Audit log directory (local mode)
LLORCH_AUDIT_DIR=/var/lib/llorch/audit

# Platform mode endpoint
LLORCH_AUDIT_ENDPOINT=https://audit.llama-orch-platform.com

# Provider credentials (platform mode)
LLORCH_PROVIDER_ID=provider-a
LLORCH_PROVIDER_KEY_FILE=/etc/llorch/provider-key

# Retention policy
LLORCH_AUDIT_RETENTION_DAYS=2555  # 7 years (SOC2)
```

---

### 11.2 Service Initialization

**Expected pattern**:
```rust
use audit_logging::{AuditLogger, AuditConfig, AuditMode};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize audit logger
    let audit_logger = AuditLogger::new(AuditConfig {
        mode: AuditMode::Local {
            base_dir: PathBuf::from(
                std::env::var("LLORCH_AUDIT_DIR")
                    .unwrap_or("/var/lib/llorch/audit/orchestratord".to_string())
            ),
        },
        service_id: "orchestratord".to_string(),
        rotation_policy: RotationPolicy::Daily,
        retention_policy: RetentionPolicy::default(),
    })?;
    
    // Add to app state
    let state = AppState {
        audit_logger: Arc::new(audit_logger),
        ...
    };
    
    // Start server
    ...
    
    // Graceful shutdown: flush buffered events
    state.audit_logger.flush().await?;
    
    Ok(())
}
```

---

## 12. Testing & Proof Bundle Integration

### 12.1 Test Support

**Required by**: All crate test suites

**Expected test utilities**:
```rust
#[cfg(test)]
pub mod test_utils {
    pub fn mock_audit_logger() -> AuditLogger;
    pub fn mock_audit_event(event_type: &str) -> AuditEvent;
    pub fn assert_event_emitted(logger: &AuditLogger, event_type: &str);
}
```

---

### 12.2 Proof Bundle Requirements

**Required by**: `determinism-suite`

**Expected behavior**:
- Unit tests MUST emit proof bundles to `.proof_bundle/unit/<run_id>/`
- Include: test metadata, event timeline, verification results
- Respect `LLORCH_RUN_ID` and `LLORCH_PROOF_DIR`

---

## 13. Observability & Telemetry

### 13.1 Structured Logging

**Required by**: All consuming crates

**Expected log events**:
- Audit event emitted: `event_type`, `audit_id`, `service_id`
- Buffer flush: `event_count`, `duration_ms`
- Integrity verification: `result`, `events_verified`
- File rotation: `old_file`, `new_file`, `event_count`

**Log levels**:
- `info` — Normal operations (emit, flush, rotate)
- `warn` — Buffer approaching full, slow writes
- `error` — Emit failed, integrity violation, disk full

---

### 13.2 Metrics

**Required by**: Prometheus exporter

**Expected metrics**:
- `audit_events_emitted_total{service_id, event_type}` — Total events emitted
- `audit_events_dropped_total{service_id, reason}` — Events dropped (buffer full)
- `audit_buffer_size_bytes{service_id}` — Current buffer size
- `audit_flush_duration_seconds{service_id}` — Flush duration
- `audit_integrity_checks_total{service_id, result}` — Integrity checks

**Access**:
```rust
pub fn metrics(&self) -> AuditMetrics;
```

---

## 14. Security & Compliance

### 14.1 Clippy Configuration

**Required**: TIER 1 (security-critical)

**Enforced lints**:
```rust
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
```

**Rationale**: Audit logging is security-critical; any panic or UB could compromise audit trail integrity.

---

### 14.2 Sensitive Data Handling

**Required by**: All consuming crates

**MUST NOT log**:
- Full API tokens (use fingerprints)
- Raw passwords
- VRAM pointers
- Seal secret keys
- Prompt content (use length/hash)
- Customer PII (unless required for GDPR)

**MUST log**:
- Token fingerprints (first 6 hex chars of SHA-256)
- User IDs (opaque identifiers)
- Resource IDs (pool_id, shard_id)
- IP addresses (for security monitoring)
- Timestamps (UTC)
- Action outcomes (success/failure)

---

## 15. Implementation Priority

### Phase 1: M0 Essentials (Current Sprint)

1. ✅ `AuditLogger` struct with basic API
2. ✅ `AuditEvent` enum with core variants
3. ✅ Local file-based storage (append-only)
4. ✅ Async event emission with buffering
5. ⬜ Hash chain implementation
6. ⬜ Integration with `input-validation` for sanitization
7. ⬜ VRAM operation events (for vram-residency)

### Phase 2: Service Integration (Next)

8. ⬜ Wire into orchestratord authentication middleware
9. ⬜ Wire into orchestratord task handlers
10. ⬜ Wire into pool-managerd pool handlers
11. ⬜ Wire into worker-orcd VRAM operations
12. ⬜ Add structured logging for all operations

### Phase 3: Production Hardening (Post-M0)

13. ⬜ Platform mode support with HTTP client
14. ⬜ Event signatures (HMAC-SHA256 or Ed25519)
15. ⬜ Query API with filters and pagination
16. ⬜ Integrity verification API
17. ⬜ Retention policy enforcement
18. ⬜ Metrics emission

### Phase 4: Advanced Features (Future)

19. ⬜ Export to SIEM (Splunk, ELK)
20. ⬜ Real-time anomaly detection
21. ⬜ Automated compliance reports (SOC2, GDPR)
22. ⬜ Multi-region audit replication

---

## 16. Open Questions

### Q1: Actor Extraction from Request

**Question**: How should services extract actor identity from authenticated requests?  
**Options**:
- A) Add `actor` field to request extensions (set by auth middleware)
- B) Pass actor explicitly to all handlers
- C) Extract from state/context in each handler

**Recommendation**: Option A (cleanest, most ergonomic)

---

### Q2: Audit Failures in Critical Paths

**Question**: Should audit failures block critical operations (e.g., seal verification)?  
**Options**:
- A) Always propagate audit errors (strict)
- B) Log error but continue (graceful degradation)
- C) Configurable per event type

**Recommendation**: Option C (critical events propagate, non-critical log)

---

### Q3: Platform Mode Rollout

**Question**: When should platform mode be implemented?  
**Deferred**: Post-M0. Local mode sufficient for single-node deployments.

---

## 17. Breaking Changes Policy

**Pre-1.0 status**: Breaking changes allowed (per user rules).

**Expected breaking changes**:
- Add required fields to `AuditEvent` variants
- Change event type naming convention
- Refactor `AuditConfig` for platform mode
- Add signature fields to `AuditEventEnvelope`

**Migration support**: None required (pre-1.0).

---

## 18. References

**Specs**:
- `bin/shared-crates/audit-logging/.specs/00-overview.md` — Crate overview
- `bin/shared-crates/audit-logging/.specs/10-event-types.md` — Event schemas
- `bin/shared-crates/audit-logging/.specs/20-storage-and-tamper-evidence.md` — Storage requirements
- `bin/shared-crates/audit-logging/.specs/30-security-and-api.md` — Security & API
- `bin/shared-crates/audit-logging/.specs/11_worker_vram_residency.md` — VRAM audit requirements

**Related crates**:
- `bin/shared-crates/input-validation` — Sanitization for audit data
- `bin/shared-crates/auth-min` — Token fingerprinting
- `bin/shared-crates/narration-core` — Observability (distinct from audit)

**Security docs**:
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Existing vulnerabilities
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Worker-orcd security
- `.docs/security/SECURITY_OVERSEER_SUMMARY.md` — Security posture

---

**End of Expectations Document**
