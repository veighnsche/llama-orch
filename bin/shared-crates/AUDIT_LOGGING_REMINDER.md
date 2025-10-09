# Audit Logging Reminder for Engineers

**Location**: `bin/shared-crates/audit-logging/`  
**Purpose**: Tamper-evident security audit logging  
**Status**: Production-ready (Security Rating: A-)

---

## ⚠️ When to Use Audit Logging

Use the `audit-logging` crate for **security-critical events**:

### ✅ MUST Use Audit Logging For:

1. **Authentication Events**
   - Login success/failure
   - Token creation/revocation
   - Session management

2. **Authorization Events**
   - Permission grants/denials
   - Permission changes
   - Access control decisions

3. **Resource Operations**
   - Pool creation/deletion/modification
   - Node registration/deregistration
   - Task submission/completion/cancellation

4. **VRAM Operations** (Security-Critical)
   - VRAM sealing
   - Seal verification
   - VRAM allocation/deallocation

5. **Security Incidents**
   - Rate limit exceeded
   - Path traversal attempts
   - Invalid token usage
   - Policy violations
   - Suspicious activity

6. **Data Access** (GDPR Compliance)
   - Inference execution
   - Model access
   - Data deletion

7. **Compliance Events** (GDPR)
   - Data access requests
   - Data export
   - Right to erasure

---

## ❌ Do NOT Use Audit Logging For:

- **Operational logs** → Use `tracing` or `narration-core`
- **Debug logs** → Use `tracing::debug!()`
- **Performance metrics** → Use `metrics` crate
- **Application state** → Use `tracing::info!()`

---

## 📝 How to Use

### 1. Add Dependency

```toml
[dependencies]
audit-logging = { path = "../shared-crates/audit-logging" }
chrono = { workspace = true }
```

### 2. Initialize Logger

```rust
use audit_logging::{AuditLogger, AuditConfig, AuditMode, RotationPolicy, RetentionPolicy};

let config = AuditConfig {
    mode: AuditMode::Local {
        base_dir: PathBuf::from("/var/log/llama-orch/audit"),
    },
    service_id: "queen-rbee".to_string(),
    rotation_policy: RotationPolicy::Daily,
    retention_policy: RetentionPolicy::Days(2555), // 7 years for compliance
};

let audit_logger = AuditLogger::new(config)?;
```

### 3. Log Events

```rust
use audit_logging::{AuditEvent, ActorInfo, AuthMethod};
use chrono::Utc;

// Authentication success
audit_logger.emit(AuditEvent::AuthSuccess {
    timestamp: Utc::now(),
    actor: ActorInfo {
        user_id: "admin@example.com".to_string(),
        ip: Some("192.168.1.1".parse()?),
        auth_method: AuthMethod::BearerToken,
        session_id: Some("session-abc123".to_string()),
    },
    method: AuthMethod::BearerToken,
    path: "/v2/tasks".to_string(),
    service_id: "queen-rbee".to_string(),
}).await?;

// Pool creation
audit_logger.emit(AuditEvent::PoolCreated {
    timestamp: Utc::now(),
    actor: ActorInfo { /* ... */ },
    pool_id: "pool-123".to_string(),
    model_ref: "meta-llama/Llama-3.1-8B".to_string(),
    node_id: "node-gpu-0".to_string(),
    replicas: 2,
    gpu_devices: vec![0, 1],
}).await?;

// VRAM sealing (security-critical)
audit_logger.emit(AuditEvent::VramSealed {
    timestamp: Utc::now(),
    shard_id: "shard-abc123".to_string(),
    gpu_device: 0,
    vram_bytes: 8_589_934_592, // 8GB
    digest: "sha256:abc123...".to_string(),
    worker_id: "worker-gpu-0".to_string(),
}).await?;
```

---

## 🔒 Security Features

### Why NOT Hand-Roll Your Own?

| Feature | audit-logging | Hand-Rolled Logging |
|---------|---------------|---------------------|
| **Tamper-Evident** | ✅ SHA-256 hash chains | ❌ No integrity protection |
| **Input Validation** | ✅ Prevents log injection | ❌ Vulnerable to attacks |
| **File Permissions** | ✅ 0600 (owner-only) | ❌ Often world-readable |
| **Disk Space Monitoring** | ✅ Fails fast when full | ❌ Silent event loss |
| **Counter Overflow** | ✅ Detects u64::MAX | ❌ Duplicate IDs |
| **SOC2 Compliant** | ✅ Yes | ❌ No |
| **GDPR Compliant** | ✅ Yes | ❌ No |
| **Test Coverage** | ✅ 85% (48 unit + 60 BDD) | ❌ Untested |
| **Security Rating** | ✅ A- (Excellent) | ❌ Unknown |

---

## 📚 Documentation

- **README**: `bin/shared-crates/audit-logging/README.md`
- **Security Audit**: `bin/shared-crates/audit-logging/SECURITY_AUDIT.md`
- **Security Verification**: `bin/shared-crates/audit-logging/.specs/21_security_verification.md`
- **Test Coverage**: `bin/shared-crates/audit-logging/TEST_COVERAGE_SUMMARY.md`
- **Robustness Fixes**: `bin/shared-crates/audit-logging/ROBUSTNESS_FIXES.md`

---

## 🎯 Integration Checklist

When adding audit logging to your crate:

- [ ] Add `audit-logging` dependency to `Cargo.toml`
- [ ] Initialize `AuditLogger` in application startup
- [ ] Identify all security-critical events
- [ ] Replace hand-rolled logging with `audit_logger.emit()`
- [ ] Add audit logging to authentication/authorization code
- [ ] Add audit logging to resource operations
- [ ] Add audit logging to security incidents
- [ ] Test with BDD scenarios (if applicable)
- [ ] Document audit events in your crate's README
- [ ] Configure retention policy for compliance

---

## 🚨 Common Mistakes to Avoid

### ❌ WRONG: Hand-Rolling Security Logs

```rust
// DON'T DO THIS!
tracing::info!("User {} authenticated successfully", user_id);
tracing::warn!("Failed login attempt from {}", ip);
println!("Pool {} created by {}", pool_id, user_id);
```

**Problems**:
- Not tamper-evident (can be modified)
- No input validation (vulnerable to log injection)
- No hash chain integrity
- Not compliance-ready

### ✅ CORRECT: Use Audit Logging

```rust
// DO THIS INSTEAD!
audit_logger.emit(AuditEvent::AuthSuccess {
    timestamp: Utc::now(),
    actor: ActorInfo { user_id, ip, auth_method, session_id },
    method: AuthMethod::BearerToken,
    path: "/v2/tasks".to_string(),
    service_id: "queen-rbee".to_string(),
}).await?;
```

**Benefits**:
- Tamper-evident (SHA-256 hash chains)
- Input validated (prevents injection)
- SOC2/GDPR compliant
- Secure file permissions

---

## 🔍 Event Types Reference

See all 32 event types in `bin/shared-crates/audit-logging/src/events.rs`:

- **Authentication**: `AuthSuccess`, `AuthFailure`, `TokenCreated`, `TokenRevoked`
- **Authorization**: `AuthorizationGranted`, `AuthorizationDenied`, `PermissionChanged`
- **Resources**: `PoolCreated`, `PoolDeleted`, `PoolModified`, `NodeRegistered`, `NodeDeregistered`, `TaskSubmitted`, `TaskCompleted`, `TaskCanceled`
- **VRAM**: `VramSealed`, `SealVerified`, `SealVerificationFailed`, `VramAllocated`, `VramAllocationFailed`, `VramDeallocated`
- **Security**: `RateLimitExceeded`, `PathTraversalAttempt`, `InvalidTokenUsed`, `PolicyViolation`, `SuspiciousActivity`
- **Data Access**: `InferenceExecuted`, `ModelAccessed`, `DataDeleted`
- **Compliance**: `GdprDataAccessRequest`, `GdprDataExport`, `GdprRightToErasure`

---

## 💡 Questions?

- **Security concerns**: See `SECURITY_AUDIT.md`
- **Integration help**: See `README.md`
- **Test examples**: See `bdd/tests/features/*.feature`
- **API reference**: Run `cargo doc --open -p audit-logging`

---

**Remember**: Security audit logging is NOT optional for production systems. Use `audit-logging` crate!
