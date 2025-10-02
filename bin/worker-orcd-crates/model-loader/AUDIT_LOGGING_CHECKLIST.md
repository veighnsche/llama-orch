# Audit Logging Integration Checklist

**Crate**: `model-loader`  
**Audit Partner**: `audit-logging`  
**Status**: ⬜ Not Yet Integrated  
**Priority**: HIGH (Security-critical events must be audited)

---

## Executive Summary

`model-loader` is a **TIER 1 security-critical crate** that validates model files before they are loaded into VRAM. It must emit audit events for:

- **Hash verification failures** (integrity violations, supply chain attacks)
- **Path traversal attempts** (active security attacks)
- **Malformed model rejections** (potential exploit attempts)

**Why audit?** These events indicate **active attacks or supply chain compromise** and are required for:
- SOC2 compliance (security incident logging)
- ISO 27001 compliance (security event records)
- Supply chain security (prove we detected tampering)
- Forensic investigation (trace attack patterns)

---

## What We MUST Audit

### 1. Hash Verification Failures (CRITICAL)

**Event**: Model hash mismatch detected

**When**: `hash::verify_hash()` returns `HashMismatch` error

**Why Audit**:
- Indicates corrupted download OR **malicious model substitution**
- Supply chain attack detection (compromised model registry)
- Compliance requirement (prove we detected tampering)
- Forensic evidence (trace attack timeline)

**Audit Event**:
```rust
AuditEvent::IntegrityViolation {
    timestamp: Utc::now(),
    resource_type: "model".to_string(),
    resource_id: sanitize_string(&canonical_path.to_string_lossy())?,
    expected_hash: expected_hash.to_string(),
    actual_hash: actual_hash.to_string(),
    severity: Severity::Critical,
    action_taken: "Model load rejected".to_string(),
}
```

**Compliance**: SOC2 CC6.1, ISO 27001 A.12.4.1

---

### 2. Path Traversal Attempts (CRITICAL)

**Event**: Path validation fails (directory traversal attempt)

**When**: `path::validate_path()` returns `PathValidationFailed` error

**Why Audit**:
- **Active attack** — someone is trying to escape allowed directory
- Security incident (requires investigation)
- Attacker behavior analysis (track IP, patterns)
- Compliance requirement (log all security violations)

**Audit Event**:
```rust
AuditEvent::PathTraversalAttempt {
    timestamp: Utc::now(),
    actor: ActorInfo {
        user_id: worker_id.clone(),  // From request context
        ip: request_ip,              // From HTTP headers
        auth_method: AuthMethod::BearerToken,
        session_id: correlation_id,
    },
    attempted_path: sanitize_string(&model_path.to_string_lossy())?,
    endpoint: "model_load".to_string(),
}
```

**Compliance**: SOC2 CC6.1, ISO 27001 A.12.4.1

---

### 3. Malformed Model Rejections (HIGH)

**Event**: GGUF validation fails (malformed model)

**When**: `gguf::validate_gguf()` returns `InvalidFormat` error

**Why Audit**:
- Could be accidental corruption (low risk)
- Could be **exploit attempt** (buffer overflow, DoS)
- Could be supply chain compromise (malicious model)
- Forensic evidence (track malformed model sources)

**Audit Event**:
```rust
AuditEvent::MalformedModelRejected {
    timestamp: Utc::now(),
    model_ref: sanitize_string(&model_path.to_string_lossy())?,
    validation_error: error.to_string(),
    severity: Severity::High,
    action_taken: "Model load rejected".to_string(),
}
```

**Compliance**: SOC2 CC6.1, ISO 27001 A.12.4.1

---

### 4. Resource Limit Violations (MEDIUM)

**Event**: Model exceeds size limits or tensor count limits

**When**: 
- `file_size > request.max_size`
- `tensor_count > MAX_TENSORS`
- `string_length > MAX_STRING_LEN`

**Why Audit**:
- Could be DoS attempt (resource exhaustion)
- Could be exploit attempt (integer overflow)
- Operational issue (model too large for VRAM)

**Audit Event**:
```rust
AuditEvent::ResourceLimitViolation {
    timestamp: Utc::now(),
    resource_type: "model".to_string(),
    limit_type: "file_size".to_string(),  // or "tensor_count", "string_length"
    limit_value: max_size,
    actual_value: file_size,
    severity: Severity::Medium,
    action_taken: "Model load rejected".to_string(),
}
```

**Compliance**: SOC2 CC6.1 (availability controls)

---

## What We Should NOT Audit

### Normal Operations (Use narration-core Instead)

- ❌ **Successful model loads** — Too verbose, not security-critical
- ❌ **GGUF validation success** — Normal operation, not an event
- ❌ **File size checks (within limits)** — Operational, not security
- ❌ **Hash verification success** — Expected behavior, not an event

**Why not?** Audit logs are for **security-critical events only**. Normal operations belong in:
- **narration-core** — For debugging and observability
- **tracing** — For structured logging
- **metrics** — For performance monitoring

**Audit log retention**: 7 years (regulatory requirement)  
**Narration log retention**: Days/weeks (operational need)

---

## Integration Steps

### Step 1: Add audit-logging Dependency

**File**: `Cargo.toml`

```toml
[dependencies]
# ... existing dependencies ...

# Audit logging for security-critical events
audit-logging = { path = "../../shared-crates/audit-logging" }

# Input validation (already present)
input-validation = { path = "../../shared-crates/input-validation" }
```

**Status**: ⬜ Not yet added

---

### Step 2: Add AuditLogger to ModelLoader

**File**: `src/loader.rs`

```rust
use audit_logging::{AuditLogger, AuditEvent, Severity, ActorInfo};
use std::sync::Arc;

pub struct ModelLoader {
    allowed_root: PathBuf,
    audit_logger: Option<Arc<AuditLogger>>,  // Optional for testing
}

impl ModelLoader {
    /// Create new model loader with default allowed root (no audit)
    pub fn new() -> Self {
        Self {
            allowed_root: PathBuf::from("/var/lib/llorch/models"),
            audit_logger: None,
        }
    }
    
    /// Create model loader with audit logging enabled
    pub fn with_audit(allowed_root: PathBuf, audit_logger: Arc<AuditLogger>) -> Self {
        Self {
            allowed_root,
            audit_logger: Some(audit_logger),
        }
    }
}
```

**Status**: ⬜ Not yet implemented

---

### Step 3: Emit Audit Events on Failures

**File**: `src/loader.rs`

#### Hash Verification Failure

```rust
// In load_and_validate()
if let Some(expected_hash) = request.expected_hash {
    match hash::verify_hash(&model_bytes, expected_hash) {
        Ok(_) => {
            // Success — no audit needed (use narration-core for observability)
            tracing::info!(
                path = ?canonical_path,
                hash = expected_hash,
                "Hash verification succeeded"
            );
        }
        Err(LoadError::HashMismatch { expected, actual }) => {
            // CRITICAL: Audit integrity violation
            if let Some(logger) = &self.audit_logger {
                logger.emit(AuditEvent::IntegrityViolation {
                    timestamp: Utc::now(),
                    resource_type: "model".to_string(),
                    resource_id: sanitize_string(&canonical_path.to_string_lossy())?,
                    expected_hash: expected.clone(),
                    actual_hash: actual.clone(),
                    severity: Severity::Critical,
                    action_taken: "Model load rejected".to_string(),
                })?;
            }
            
            tracing::error!(
                path = ?canonical_path,
                expected = expected,
                actual = actual,
                "Hash verification failed — integrity violation"
            );
            
            return Err(LoadError::HashMismatch { expected, actual });
        }
    }
}
```

#### Path Traversal Attempt

```rust
// In load_and_validate()
let canonical_path = match path::validate_path(request.model_path, &self.allowed_root) {
    Ok(path) => path,
    Err(e) => {
        // CRITICAL: Audit path traversal attempt
        if let Some(logger) = &self.audit_logger {
            logger.emit(AuditEvent::PathTraversalAttempt {
                timestamp: Utc::now(),
                actor: ActorInfo {
                    user_id: request.worker_id.clone(),  // From request context
                    ip: request.source_ip,               // From HTTP headers
                    auth_method: AuthMethod::BearerToken,
                    session_id: request.correlation_id.clone(),
                },
                attempted_path: sanitize_string(&request.model_path.to_string_lossy())?,
                endpoint: "model_load".to_string(),
            })?;
        }
        
        tracing::error!(
            path = ?request.model_path,
            error = %e,
            "Path validation failed — potential traversal attempt"
        );
        
        return Err(e);
    }
};
```

#### Malformed Model Rejection

```rust
// In load_and_validate()
if let Err(e) = gguf::validate_gguf(&model_bytes) {
    // HIGH: Audit malformed model rejection
    if let Some(logger) = &self.audit_logger {
        logger.emit(AuditEvent::MalformedModelRejected {
            timestamp: Utc::now(),
            model_ref: sanitize_string(&canonical_path.to_string_lossy())?,
            validation_error: e.to_string(),
            severity: Severity::High,
            action_taken: "Model load rejected".to_string(),
        })?;
    }
    
    tracing::error!(
        path = ?canonical_path,
        error = %e,
        "GGUF validation failed — malformed model"
    );
    
    return Err(e);
}
```

**Status**: ⬜ Not yet implemented

---

### Step 4: Add Request Context for Actor Info

**File**: `src/types.rs`

```rust
pub struct LoadRequest<'a> {
    pub model_path: &'a Path,
    pub expected_hash: Option<&'a str>,
    pub max_size: usize,
    
    // NEW: Actor context for audit logging
    pub worker_id: String,              // Who is loading the model
    pub source_ip: Option<IpAddr>,      // Where the request came from
    pub correlation_id: Option<String>, // Request correlation ID
}
```

**Status**: ⬜ Not yet implemented

---

### Step 5: Update Tests

**File**: `tests/audit_integration_tests.rs` (new file)

```rust
use model_loader::{ModelLoader, LoadRequest};
use audit_logging::{AuditLogger, AuditConfig, AuditMode};
use tempfile::TempDir;

#[test]
fn test_hash_mismatch_emits_audit_event() {
    // Setup audit logger
    let temp_dir = TempDir::new().unwrap();
    let audit_logger = Arc::new(AuditLogger::new(AuditConfig {
        mode: AuditMode::Local {
            base_dir: temp_dir.path().to_path_buf(),
        },
        service_id: "model-loader-test".to_string(),
        rotation_policy: RotationPolicy::Daily,
        retention_policy: RetentionPolicy::default(),
    }).unwrap());
    
    // Setup model loader with audit
    let loader = ModelLoader::with_audit(
        PathBuf::from("/tmp/models"),
        audit_logger.clone(),
    );
    
    // Create model with wrong hash
    let model_bytes = create_valid_gguf();
    let wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000";
    
    // Attempt load (should fail)
    let result = loader.validate_bytes(&model_bytes, Some(wrong_hash));
    assert!(matches!(result, Err(LoadError::HashMismatch { .. })));
    
    // Verify audit event was emitted
    audit_logger.flush().await.unwrap();
    
    let events = audit_logger.query(AuditQuery {
        event_types: vec!["integrity_violation"],
        ..Default::default()
    }).await.unwrap();
    
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].severity, Severity::Critical);
}

#[test]
fn test_path_traversal_emits_audit_event() {
    // Similar test for path traversal
    // ...
}

#[test]
fn test_malformed_model_emits_audit_event() {
    // Similar test for malformed model
    // ...
}
```

**Status**: ⬜ Not yet implemented

---

### Step 6: Update Documentation

**Files to update**:
- ✅ `AUDIT_LOGGING_CHECKLIST.md` (this file)
- ⬜ `README.md` — Add audit logging section
- ⬜ `.specs/00_model-loader.md` — Add audit requirements
- ⬜ `.specs/20_security.md` — Reference audit events

**Status**: Partially complete

---

## New Audit Event Types Required

These event types need to be **added to audit-logging crate**:

### 1. IntegrityViolation

```rust
/// Model integrity violation (hash mismatch)
IntegrityViolation {
    timestamp: DateTime<Utc>,
    resource_type: String,  // "model"
    resource_id: String,    // Sanitized model path
    expected_hash: String,  // Expected SHA-256 hash
    actual_hash: String,    // Actual SHA-256 hash
    severity: Severity,     // Critical
    action_taken: String,   // "Model load rejected"
}
```

### 2. MalformedModelRejected

```rust
/// Malformed model rejected (GGUF validation failed)
MalformedModelRejected {
    timestamp: DateTime<Utc>,
    model_ref: String,         // Sanitized model reference
    validation_error: String,  // Error message (sanitized)
    severity: Severity,        // High
    action_taken: String,      // "Model load rejected"
}
```

### 3. ResourceLimitViolation

```rust
/// Resource limit violation (size, tensor count, etc.)
ResourceLimitViolation {
    timestamp: DateTime<Utc>,
    resource_type: String,  // "model"
    limit_type: String,     // "file_size", "tensor_count", "string_length"
    limit_value: u64,       // Maximum allowed value
    actual_value: u64,      // Actual value
    severity: Severity,     // Medium
    action_taken: String,   // "Model load rejected"
}
```

**Status**: ⬜ Not yet added to audit-logging

---

## Compliance Mapping

### SOC2 Requirements

| Control | Requirement | Audit Event | Status |
|---------|-------------|-------------|--------|
| **CC6.1** | Security incident logging | IntegrityViolation | ⬜ |
| **CC6.1** | Security incident logging | PathTraversalAttempt | ⬜ |
| **CC6.1** | Security incident logging | MalformedModelRejected | ⬜ |
| **CC6.6** | Availability controls | ResourceLimitViolation | ⬜ |

### ISO 27001 Requirements

| Control | Requirement | Audit Event | Status |
|---------|-------------|-------------|--------|
| **A.12.4.1** | Event logging | IntegrityViolation | ⬜ |
| **A.12.4.1** | Event logging | PathTraversalAttempt | ⬜ |
| **A.12.4.1** | Event logging | MalformedModelRejected | ⬜ |

### Supply Chain Security

| Requirement | Audit Event | Status |
|-------------|-------------|--------|
| Detect model tampering | IntegrityViolation | ⬜ |
| Detect malicious models | MalformedModelRejected | ⬜ |
| Trace attack timeline | All events with timestamps | ⬜ |

---

## Security Considerations

### What We MUST Sanitize

**Before logging to audit**:
- ✅ **Model paths** — Use `input_validation::sanitize_string()`
- ✅ **Error messages** — Strip ANSI, control chars, null bytes
- ✅ **User input** — Always sanitize before logging

**Why**: Prevent log injection attacks in audit trail.

### What We MUST NOT Log

**Forbidden content**:
- ❌ **Model file contents** — Too large, not relevant
- ❌ **VRAM pointers** — Security risk
- ❌ **Full file paths** — May contain sensitive info (sanitize first)
- ❌ **Stack traces** — Use error messages only

**Why**: Audit logs must be concise and security-focused.

### Actor Information

**Required fields**:
- ✅ `worker_id` — Which worker requested the load
- ✅ `source_ip` — Where the request came from (if available)
- ✅ `correlation_id` — Request tracking across services

**How to get**:
- From HTTP request headers (`X-Worker-Id`, `X-Forwarded-For`, `X-Correlation-Id`)
- From gRPC metadata
- From request context

---

## Testing Requirements

### Unit Tests

- ⬜ Test audit event emission on hash mismatch
- ⬜ Test audit event emission on path traversal
- ⬜ Test audit event emission on malformed model
- ⬜ Test audit event emission on resource limits
- ⬜ Test no audit events on success (only narration)

### Integration Tests

- ⬜ Test audit logger integration with model-loader
- ⬜ Test audit event serialization
- ⬜ Test audit event query API
- ⬜ Test correlation ID propagation

### BDD Scenarios

- ⬜ Scenario: Hash mismatch triggers audit event
- ⬜ Scenario: Path traversal triggers audit event
- ⬜ Scenario: Malformed model triggers audit event
- ⬜ Scenario: Successful load does NOT trigger audit event

---

## Implementation Checklist

### Phase 1: Foundation (P0)

- [ ] Add `audit-logging` dependency to `Cargo.toml`
- [ ] Add `AuditLogger` field to `ModelLoader`
- [ ] Add `with_audit()` constructor
- [ ] Add actor context fields to `LoadRequest`
- [ ] Update `load_and_validate()` signature

### Phase 2: Event Emission (P0)

- [ ] Emit `IntegrityViolation` on hash mismatch
- [ ] Emit `PathTraversalAttempt` on path validation failure
- [ ] Emit `MalformedModelRejected` on GGUF validation failure
- [ ] Emit `ResourceLimitViolation` on size/tensor/string limits

### Phase 3: Testing (P0)

- [ ] Write unit tests for each audit event
- [ ] Write integration tests for audit logger
- [ ] Write BDD scenarios for audit events
- [ ] Verify no audit events on success

### Phase 4: Documentation (P1)

- [ ] Update `README.md` with audit logging section
- [ ] Update `.specs/00_model-loader.md` with audit requirements
- [ ] Update `.specs/20_security.md` with audit event mapping
- [ ] Add examples to integration guide

### Phase 5: Audit-Logging Crate Updates (P0)

- [ ] Add `IntegrityViolation` event type to `audit-logging`
- [ ] Add `MalformedModelRejected` event type to `audit-logging`
- [ ] Add `ResourceLimitViolation` event type to `audit-logging`
- [ ] Update `audit-logging` specs with new event types
- [ ] Write BDD tests for new event types in `audit-logging`

---

## Acceptance Criteria

### Definition of Done

- ✅ All P0 checklist items completed
- ✅ All tests passing (unit, integration, BDD)
- ✅ Documentation updated (README, specs)
- ✅ Code review approved
- ✅ Clippy clean (TIER 1 configuration)
- ✅ No panics, no unwrap, no expect
- ✅ Audit events verified in test environment

### Verification

```bash
# Run all tests
cargo test -p model-loader

# Run audit integration tests
cargo test -p model-loader test_audit

# Verify audit events are emitted
cargo test -p model-loader test_hash_mismatch_emits_audit_event

# Check Clippy (TIER 1)
cargo clippy -p model-loader -- -D warnings
```

---

## Timeline

**Estimated effort**: 2-3 days

- **Day 1**: Phase 1 + Phase 2 (foundation + event emission)
- **Day 2**: Phase 3 (testing)
- **Day 3**: Phase 4 + Phase 5 (documentation + audit-logging updates)

---

## Questions for audit-logging Team

1. **Event type approval**: Do the proposed event types (`IntegrityViolation`, `MalformedModelRejected`, `ResourceLimitViolation`) align with your taxonomy?

2. **Severity levels**: Are `Critical`, `High`, `Medium` appropriate for these events?

3. **Actor context**: Should we require `worker_id` and `source_ip`, or make them optional?

4. **Sanitization**: Should we use `input_validation::sanitize_string()` for all string fields, or does audit-logging handle this internally?

5. **Performance**: Is synchronous `emit()` acceptable for model-loader, or should we use async?

---

## References

- **audit-logging crate**: `bin/shared-crates/audit-logging`
- **audit-logging specs**: `bin/shared-crates/audit-logging/.specs/`
- **model-loader specs**: `.specs/00_model-loader.md`
- **Security audit**: `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` (Issue #19)

---

**Status**: 🟡 **PHASE 1 COMPLETE** — Actor context added, Phase 2 pending  
**Priority**: HIGH (Security-critical events must be audited)  
**Owner**: model-loader team + audit-logging team  
**Target**: M0 (blocking for production readiness)

---

## ✅ Phase 1 Complete (Foundation)

- ✅ Added `audit-logging` dependency to `Cargo.toml`
- ✅ Added actor context fields to `LoadRequest` (worker_id, source_ip, correlation_id)
- ✅ Added builder methods (`.with_worker_id()`, `.with_source_ip()`, `.with_correlation_id()`)

**Next**: Phase 2 - Add `AuditLogger` to `ModelLoader` and emit audit events on failures

---

**Last Updated**: 2025-10-02  
**Maintainer**: @llama-orch-maintainers
