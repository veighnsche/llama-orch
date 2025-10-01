# Audit Logging — Security & API Specification

**Crate**: `bin/shared-crates/audit-logging`  
**Status**: Draft  
**Last Updated**: 2025-10-01

---

## 0. Overview

This document specifies security requirements, access control, and API design for the audit-logging crate. Audit logs are security-critical and must be protected from unauthorized access, modification, and deletion.

---

## 1. Security Requirements

### 1.1 Threat Model

**Threats to Audit Logs**:

| Threat | Impact | Mitigation |
|--------|--------|------------|
| **Unauthorized access** | Privacy breach, compliance violation | Access control, encryption |
| **Tampering** | Evidence destruction, false evidence | Hash chains, signatures |
| **Deletion** | Evidence destruction | Append-only storage, backups |
| **Injection** | Log poisoning, false evidence | Input validation, sanitization |
| **Denial of service** | Audit unavailable | Buffering, async writes |
| **Exfiltration** | Sensitive data leak | Encryption, redaction |

---

### 1.2 Input Validation

**Purpose**: Prevent log injection attacks.

**Threats**:
- **ANSI escape injection**: `\x1b[31mFAKE ERROR\x1b[0m`
- **Control character injection**: `\r\n[ERROR] Fake log line`
- **Unicode directional override**: `\u202E` (right-to-left override)
- **Null byte injection**: `\0` (truncates logs)

**Mitigation**: Use `input-validation` crate:
```rust
use input_validation::sanitize_string;

// Sanitize before logging
let safe_user_id = sanitize_string(&user_id)?;
let safe_resource = sanitize_string(&resource_id)?;

audit_logger.emit(AuditEvent::PoolDeleted {
    actor: ActorInfo {
        user_id: safe_user_id,  // ✅ Sanitized
        ...
    },
    resource: ResourceInfo {
        pool_id: safe_resource,  // ✅ Sanitized
        ...
    },
    ...
}).await?;
```

**Validation Rules**:
- Remove ANSI escape sequences
- Remove control characters (except newline in structured fields)
- Remove Unicode directional overrides
- Remove null bytes
- Limit string length (max 1024 chars per field)

---

### 1.3 Sensitive Data Redaction

**Purpose**: Prevent credential and PII leakage in audit logs.

**MUST NOT log**:
- Full API tokens (use fingerprints: `token:a3f2c1`)
- Raw passwords
- VRAM pointers (`0x7f8a4c000000`)
- Seal secret keys
- Raw model bytes
- Internal memory addresses
- Prompt content (log length/hash only)
- Customer PII (unless required for GDPR)

**MUST log**:
- Token fingerprints (first 6 hex chars of SHA-256)
- User IDs (opaque identifiers)
- Resource IDs (pool_id, shard_id)
- IP addresses (for security monitoring)
- Timestamps (UTC)
- Action outcomes (success/failure)

**Example**:
```rust
// ❌ FORBIDDEN
AuditEvent::AuthSuccess {
    actor: ActorInfo {
        token: "sk_live_abc123def456...",  // Never log full token
    }
}

// ✅ CORRECT
AuditEvent::AuthSuccess {
    actor: ActorInfo {
        user_id: "token:a3f2c1",  // Fingerprint only
    }
}
```

**Token Fingerprinting**:
```rust
use sha2::{Sha256, Digest};

pub fn fingerprint_token(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let hash = hasher.finalize();
    format!("token:{}", hex::encode(&hash[..3]))  // First 6 hex chars
}
```

---

### 1.4 Access Control

**Purpose**: Restrict who can read/write audit logs.

**Principles**:
- **Write**: Only authenticated services can write
- **Read**: Only administrators can read
- **No updates**: No API for modifying existing events
- **No deletes**: No API for deleting events (except retention policy)
- **Audit access**: Audit log access is itself audited

**File Permissions** (local mode):
```bash
# Audit directory
drwx------  2 llorch-audit llorch-audit  4096 Oct  1 16:48 /var/lib/llorch/audit/

# Audit files (read-only after rotation)
-rw-------  1 llorch-audit llorch-audit  52428800 Oct  1 23:59 2025-10-01.audit
-r--------  1 llorch-audit llorch-audit        65 Oct  2 00:00 2025-10-01.audit.sha256
```

**Service User**:
- Services run as `llorch` user
- Audit files owned by `llorch-audit` user
- Services can write but not read (write-only permissions)
- Administrators can read but not write (read-only permissions)

**Platform Mode**:
- **Customers**: Query their own events only
- **Regulators**: Query for compliance (with approval)
- **Platform admins**: Full access (audited)
- **Providers**: Cannot access (security risk)

---

### 1.5 Encryption

**Purpose**: Protect audit logs at rest and in transit.

**At Rest** (local mode):
- Use encrypted filesystem (LUKS, dm-crypt)
- Or encrypt individual files (AES-256-GCM)
- Key management via systemd credentials or KMS

**At Rest** (platform mode):
- S3 server-side encryption (SSE-KMS)
- KMS key per customer for isolation
- Cross-region replication with encryption

**In Transit**:
- TLS 1.3 for HTTP communication
- mTLS for service-to-service (optional)
- Certificate pinning for platform mode

**Example** (file encryption):
```rust
use aes_gcm::{Aes256Gcm, Key, Nonce};
use aes_gcm::aead::{Aead, NewAead};

pub fn encrypt_audit_file(
    plaintext: &[u8],
    key: &[u8; 32]
) -> Result<Vec<u8>, AuditError> {
    let cipher = Aes256Gcm::new(Key::from_slice(key));
    let nonce = Nonce::from_slice(b"unique nonce");  // Use unique nonce per file
    
    cipher.encrypt(nonce, plaintext)
        .map_err(|e| AuditError::EncryptionFailed(e.to_string()))
}
```

---

### 1.6 Performance & Availability

**Purpose**: Audit logging must not block operations.

**Requirements**:
- **Non-blocking**: Async event emission
- **Buffering**: Buffer events in memory (max 1000 events or 10MB)
- **Batching**: Flush every 1 second or 100 events
- **Backpressure**: Drop events if buffer full (log warning)
- **Flush on critical**: Immediate flush for critical events (seal verification failure)
- **Flush on shutdown**: Graceful shutdown flushes all buffered events

**Implementation**:
```rust
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};

pub struct AuditLogger {
    tx: mpsc::Sender<AuditEvent>,
}

impl AuditLogger {
    pub fn new(config: AuditConfig) -> Self {
        let (tx, rx) = mpsc::channel(1000);  // Buffer 1000 events
        
        // Spawn background writer task
        tokio::spawn(audit_writer_task(rx, config));
        
        Self { tx }
    }
    
    pub async fn emit(&self, event: AuditEvent) -> Result<(), AuditError> {
        // Non-blocking send
        self.tx.send(event).await
            .map_err(|_| AuditError::BufferFull)
    }
}

async fn audit_writer_task(
    mut rx: mpsc::Receiver<AuditEvent>,
    config: AuditConfig
) {
    let mut writer = AuditFileWriter::new(config.file_path).unwrap();
    let mut buffer = Vec::new();
    let mut flush_interval = interval(Duration::from_secs(1));
    
    loop {
        tokio::select! {
            // Receive event
            Some(event) = rx.recv() => {
                buffer.push(event);
                
                // Flush if buffer full
                if buffer.len() >= 100 {
                    flush_buffer(&mut writer, &mut buffer).await;
                }
            }
            
            // Periodic flush
            _ = flush_interval.tick() => {
                if !buffer.is_empty() {
                    flush_buffer(&mut writer, &mut buffer).await;
                }
            }
        }
    }
}

async fn flush_buffer(
    writer: &mut AuditFileWriter,
    buffer: &mut Vec<AuditEvent>
) {
    for event in buffer.drain(..) {
        if let Err(e) = writer.write_event(event) {
            tracing::error!(error = %e, "Failed to write audit event");
        }
    }
}
```

---

## 2. API Design

### 2.1 Logger Initialization

**Local Mode**:
```rust
use audit_logging::{AuditLogger, AuditConfig};

let config = AuditConfig {
    mode: AuditMode::Local {
        base_dir: PathBuf::from("/var/lib/llorch/audit/orchestratord"),
    },
    service_id: "orchestratord".to_string(),
    rotation_policy: RotationPolicy::Daily,
    retention_policy: RetentionPolicy::default(),  // 7 years
};

let logger = AuditLogger::new(config)?;
```

**Platform Mode**:
```rust
use audit_logging::{AuditLogger, AuditConfig, PlatformConfig};

let config = AuditConfig {
    mode: AuditMode::Platform(PlatformConfig {
        endpoint: "https://audit.llama-orch-platform.com".to_string(),
        provider_id: "provider-a".to_string(),
        provider_key: load_provider_key()?,
        batch_size: 100,
        flush_interval: Duration::from_secs(10),
    }),
    service_id: "orchestratord".to_string(),
    rotation_policy: RotationPolicy::Daily,
    retention_policy: RetentionPolicy::default(),
};

let logger = AuditLogger::new(config)?;
```

---

### 2.2 Event Emission

**Basic Emission**:
```rust
logger.emit(AuditEvent::AuthSuccess {
    timestamp: Utc::now(),
    actor: ActorInfo {
        user_id: "admin@llorch.io".to_string(),
        ip: Some("192.168.1.100".parse()?),
        auth_method: AuthMethod::BearerToken,
        session_id: Some("sess-abc123".to_string()),
    },
    method: AuthMethod::BearerToken,
    path: "/v2/tasks".to_string(),
    service_id: "orchestratord".to_string(),
}).await?;
```

**With Error Handling**:
```rust
// Non-critical event: Log error but continue
if let Err(e) = logger.emit(event).await {
    tracing::error!(error = %e, "Failed to emit audit event");
}

// Critical event: Propagate error
logger.emit(AuditEvent::SealVerificationFailed { ... }).await?;
```

**Spawn Async** (non-blocking):
```rust
// Don't block VRAM operation
tokio::spawn({
    let logger = self.audit_logger.clone();
    async move {
        logger.emit(AuditEvent::VramAllocated { ... }).await
    }
});
```

---

### 2.3 Query API

**Query by Actor**:
```rust
use audit_logging::{AuditQuery, AuditLogger};

let events = logger.query(AuditQuery {
    actor: Some("admin@example.com".to_string()),
    start_time: Some(Utc::now() - Duration::days(7)),
    end_time: Some(Utc::now()),
    event_types: vec![],  // All types
    limit: 100,
}).await?;

for event in events {
    println!("{:?}", event);
}
```

**Query by Event Type**:
```rust
let events = logger.query(AuditQuery {
    actor: None,
    start_time: Some(Utc::now() - Duration::hours(1)),
    end_time: Some(Utc::now()),
    event_types: vec!["auth.failure".to_string(), "security.rate_limit_exceeded".to_string()],
    limit: 1000,
}).await?;
```

**Query by Resource**:
```rust
let events = logger.query(AuditQuery {
    resource_id: Some("pool-123".to_string()),
    start_time: Some(Utc::now() - Duration::days(30)),
    end_time: Some(Utc::now()),
    event_types: vec!["pool.created".to_string(), "pool.deleted".to_string()],
    limit: 100,
}).await?;
```

---

### 2.4 Integrity Verification

**Verify Hash Chain**:
```rust
// Verify last 1000 events
let result = logger.verify_integrity(VerifyOptions {
    mode: VerifyMode::LastN(1000),
}).await?;

match result {
    VerifyResult::Valid => println!("Integrity verified"),
    VerifyResult::Invalid { broken_at } => {
        eprintln!("Hash chain broken at event: {}", broken_at);
    }
}
```

**Verify File Checksums**:
```rust
let result = logger.verify_file_checksums().await?;

for file_result in result.files {
    match file_result.status {
        ChecksumStatus::Valid => {
            println!("✅ {}: valid", file_result.filename);
        }
        ChecksumStatus::Invalid { expected, actual } => {
            eprintln!("❌ {}: checksum mismatch", file_result.filename);
            eprintln!("   Expected: {}", expected);
            eprintln!("   Actual:   {}", actual);
        }
    }
}
```

**Verify Signatures** (platform mode):
```rust
let result = logger.verify_signatures(VerifyOptions {
    mode: VerifyMode::All,
}).await?;

println!("Verified {} events", result.verified_count);
println!("Invalid signatures: {}", result.invalid_count);
```

---

### 2.5 Compliance Reports

**GDPR Data Access Report**:
```rust
let report = logger.generate_gdpr_report(GdprReportRequest {
    customer_id: "customer-123".to_string(),
    start_date: Utc::now() - Duration::days(365),
    end_date: Utc::now(),
}).await?;

println!("Inference executions: {}", report.inference_count);
println!("Models accessed: {:?}", report.models_accessed);
println!("Data exports: {}", report.export_count);
```

**SOC2 Audit Report**:
```rust
let report = logger.generate_soc2_report(Soc2ReportRequest {
    start_date: Utc::now() - Duration::days(365),
    end_date: Utc::now(),
}).await?;

println!("Authentication events: {}", report.auth_events);
println!("Authorization failures: {}", report.authz_failures);
println!("Security incidents: {}", report.security_incidents);
```

---

## 3. Integration Patterns

### 3.1 orchestratord Integration

**Initialization**:
```rust
// bin/orchestratord/src/main.rs
use audit_logging::{AuditLogger, AuditConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize audit logger
    let audit_logger = AuditLogger::new(AuditConfig {
        mode: AuditMode::Local {
            base_dir: PathBuf::from("/var/lib/llorch/audit/orchestratord"),
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
}
```

**Authentication Middleware**:
```rust
// bin/orchestratord/src/app/auth_min.rs
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

---

### 3.2 pool-managerd Integration

**Pool Operations**:
```rust
// bin/pool-managerd/src/api/pools.rs
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
```

---

### 3.3 worker-orcd Integration

**VRAM Operations**:
```rust
// bin/worker-orcd-crates/vram-residency/src/lib.rs
impl VramManager {
    pub async fn seal_model(
        &mut self,
        model_bytes: &[u8],
        gpu_device: u32,
    ) -> Result<SealedShard> {
        // Allocate VRAM
        let vram_ptr = self.allocate_vram(model_bytes.len())?;
        
        // Compute digest
        let digest = compute_sha256(model_bytes);
        
        // Create sealed shard
        let shard = SealedShard {
            shard_id: generate_shard_id(),
            gpu_device,
            vram_bytes: model_bytes.len(),
            digest: digest.clone(),
            sealed_at: SystemTime::now(),
            vram_ptr,
        };
        
        // AUDIT: Model sealed
        self.audit_logger.emit(AuditEvent::VramSealed {
            timestamp: Utc::now(),
            shard_id: shard.shard_id.clone(),
            gpu_device: shard.gpu_device,
            vram_bytes: shard.vram_bytes,
            digest: digest,
            worker_id: self.worker_id.clone(),
        }).await?;
        
        Ok(shard)
    }
}
```

---

## 4. Testing

### 4.1 Unit Tests

**Hash Chain Verification**:
```rust
#[tokio::test]
async fn test_hash_chain_verification() {
    let logger = AuditLogger::new(test_config()).unwrap();
    
    // Emit events
    logger.emit(test_event_1()).await.unwrap();
    logger.emit(test_event_2()).await.unwrap();
    logger.emit(test_event_3()).await.unwrap();
    
    // Verify chain
    let result = logger.verify_integrity(VerifyOptions {
        mode: VerifyMode::All,
    }).await.unwrap();
    
    assert!(matches!(result, VerifyResult::Valid));
}
```

**Tampering Detection**:
```rust
#[tokio::test]
async fn test_tampering_detection() {
    let logger = AuditLogger::new(test_config()).unwrap();
    
    // Emit events
    logger.emit(test_event_1()).await.unwrap();
    logger.emit(test_event_2()).await.unwrap();
    
    // Tamper with file
    tamper_with_audit_file(&logger.file_path());
    
    // Verify chain (should fail)
    let result = logger.verify_integrity(VerifyOptions {
        mode: VerifyMode::All,
    }).await.unwrap();
    
    assert!(matches!(result, VerifyResult::Invalid { .. }));
}
```

---

### 4.2 Integration Tests

**End-to-End Audit Flow**:
```rust
#[tokio::test]
async fn test_e2e_audit_flow() {
    // Start orchestratord with audit logging
    let server = start_test_server().await;
    
    // Make authenticated request
    let response = client.post("/v2/tasks")
        .header("Authorization", "Bearer test-token")
        .json(&test_task_request())
        .send()
        .await
        .unwrap();
    
    assert_eq!(response.status(), 200);
    
    // Query audit logs
    let events = server.audit_logger.query(AuditQuery {
        event_types: vec!["auth.success".to_string()],
        limit: 10,
        ..Default::default()
    }).await.unwrap();
    
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event_type, "auth.success");
}
```

---

## 5. Refinement Opportunities

### 5.1 Immediate Improvements

1. **Add structured logging** for audit events (JSON format)
2. **Implement file-based storage** with append-only guarantees
3. **Add hash chain verification** on startup
4. **Integrate with input-validation** crate

### 5.2 Medium-Term Enhancements

5. **Add platform mode** with HTTP client and signature support
6. **Implement query API** with filters and pagination
7. **Add compliance reports** (GDPR, SOC2)
8. **Create admin CLI** for audit log management

### 5.3 Long-Term Vision

9. **Real-time streaming** to SIEM systems (Splunk, ELK)
10. **Anomaly detection** using ML on audit patterns
11. **Blockchain integration** for ultimate tamper evidence
12. **Multi-tenant isolation** with per-customer audit trails

---

**End of Security & API Specification**
