# Audit Logging — Worker VRAM Residency Requirements

**Consumer**: `bin/worker-orcd-crates/vram-residency`  
**Status**: Draft  
**Last Updated**: 2025-10-01

---

## 0. Overview

This document specifies what `vram-residency` expects from the `audit-logging` crate for security audit trail requirements.

**Context**: `vram-residency` is a security-critical component that must maintain an immutable audit trail of all VRAM operations for compliance, forensics, and security incident investigation.

**Reference**: 
- `bin/worker-orcd-crates/vram-residency/.specs/20_security.md` — Security spec (Section 7.1)
- `bin/worker-orcd-crates/vram-residency/.specs/10_expectations.md` — Consumer expectations (Section 10.2)

---

## 1. Use Cases

### 1.1 VRAM Seal Operations

**Purpose**: Audit trail of model sealing operations.

**Event**: Model sealed in VRAM
```rust
AuditEvent::VramSealed {
    timestamp: Utc::now(),
    shard_id: "shard-abc123",
    gpu_device: 0,
    vram_bytes: 8_000_000_000,  // 8GB
    digest: "abc123def456...",
    worker_id: "worker-gpu-0",
}
```

**Why audit**: 
- Proves which models were loaded when
- Enables forensic investigation of compromised models
- Compliance requirement (know what data was processed)

---

### 1.2 Seal Verification

**Purpose**: Audit trail of seal verification attempts (success and failure).

**Event**: Seal verification passed
```rust
AuditEvent::SealVerified {
    timestamp: Utc::now(),
    shard_id: "shard-abc123",
    result: AuditResult::Success,
    worker_id: "worker-gpu-0",
}
```

**Event**: Seal verification failed (SECURITY INCIDENT)
```rust
AuditEvent::SealVerificationFailed {
    timestamp: Utc::now(),
    shard_id: "shard-abc123",
    reason: "digest_mismatch",
    expected_digest: "abc123...",
    actual_digest: "xyz789...",
    worker_id: "worker-gpu-0",
    severity: "critical",
}
```

**Why audit**: 
- Detect VRAM corruption or tampering
- Security incident investigation
- Prove integrity of inference operations

---

### 1.3 VRAM Allocation

**Purpose**: Track VRAM resource usage for capacity planning and DoS detection.

**Event**: VRAM allocation request
```rust
AuditEvent::VramAllocated {
    timestamp: Utc::now(),
    requested_bytes: 8_000_000_000,
    allocated_bytes: 8_000_000_000,
    available_bytes: 16_000_000_000,
    used_bytes: 8_000_000_000,
    gpu_device: 0,
    worker_id: "worker-gpu-0",
}
```

**Event**: VRAM allocation failed (OOM)
```rust
AuditEvent::VramAllocationFailed {
    timestamp: Utc::now(),
    requested_bytes: 16_000_000_000,
    available_bytes: 8_000_000_000,
    reason: "insufficient_vram",
    gpu_device: 0,
    worker_id: "worker-gpu-0",
}
```

**Why audit**: 
- Detect DoS attacks (repeated OOM attempts)
- Capacity planning
- Billing/usage tracking

---

### 1.4 VRAM Deallocation

**Purpose**: Track VRAM cleanup for leak detection.

**Event**: VRAM deallocated
```rust
AuditEvent::VramDeallocated {
    timestamp: Utc::now(),
    shard_id: "shard-abc123",
    freed_bytes: 8_000_000_000,
    remaining_used: 0,
    gpu_device: 0,
    worker_id: "worker-gpu-0",
}
```

**Why audit**: 
- Detect VRAM leaks
- Verify cleanup on worker shutdown
- Resource accounting

---

### 1.5 Policy Violations

**Purpose**: Audit security policy violations.

**Event**: VRAM-only policy violation detected
```rust
AuditEvent::PolicyViolation {
    timestamp: Utc::now(),
    policy: "vram_only",
    violation: "unified_memory_detected",
    details: "UMA enabled, cannot enforce VRAM-only policy",
    severity: "critical",
    worker_id: "worker-gpu-0",
    action_taken: "worker_stopped",
}
```

**Why audit**: 
- Security incident investigation
- Compliance violation tracking
- Detect configuration errors

---

## 2. Required API

### 2.1 Event Emission

**Emit audit event**:
```rust
use audit_logging::{AuditLogger, AuditEvent};

impl VramManager {
    async fn seal_model(&mut self, ...) -> Result<SealedShard> {
        // Perform sealing...
        
        // Emit audit event
        self.audit_logger.emit(AuditEvent::VramSealed {
            timestamp: Utc::now(),
            shard_id: shard.shard_id.clone(),
            gpu_device: shard.gpu_device,
            vram_bytes: shard.vram_bytes,
            digest: shard.digest.clone(),
            worker_id: self.worker_id.clone(),
        }).await?;
        
        Ok(shard)
    }
}
```

---

### 2.2 Event Types

**Required event types for vram-residency**:

```rust
pub enum AuditEvent {
    // Seal operations
    VramSealed {
        timestamp: DateTime<Utc>,
        shard_id: String,
        gpu_device: u32,
        vram_bytes: usize,
        digest: String,
        worker_id: String,
    },
    
    // Seal verification
    SealVerified {
        timestamp: DateTime<Utc>,
        shard_id: String,
        result: AuditResult,
        worker_id: String,
    },
    
    SealVerificationFailed {
        timestamp: DateTime<Utc>,
        shard_id: String,
        reason: String,
        expected_digest: String,
        actual_digest: String,
        worker_id: String,
        severity: String,
    },
    
    // VRAM allocation
    VramAllocated {
        timestamp: DateTime<Utc>,
        requested_bytes: usize,
        allocated_bytes: usize,
        available_bytes: usize,
        used_bytes: usize,
        gpu_device: u32,
        worker_id: String,
    },
    
    VramAllocationFailed {
        timestamp: DateTime<Utc>,
        requested_bytes: usize,
        available_bytes: usize,
        reason: String,
        gpu_device: u32,
        worker_id: String,
    },
    
    // VRAM deallocation
    VramDeallocated {
        timestamp: DateTime<Utc>,
        shard_id: String,
        freed_bytes: usize,
        remaining_used: usize,
        gpu_device: u32,
        worker_id: String,
    },
    
    // Policy violations
    PolicyViolation {
        timestamp: DateTime<Utc>,
        policy: String,
        violation: String,
        details: String,
        severity: String,
        worker_id: String,
        action_taken: String,
    },
}
```

---

### 2.3 Logger Initialization

**Initialize audit logger**:
```rust
use audit_logging::AuditLogger;

impl VramManager {
    pub fn new(config: &WorkerConfig) -> Result<Self> {
        // Initialize audit logger
        let audit_logger = AuditLogger::local(&config.audit_dir)?;
        
        Ok(Self {
            audit_logger,
            // ...
        })
    }
}
```

---

### 2.4 Async Support

**All audit operations MUST be async**:
```rust
impl AuditLogger {
    pub async fn emit(&self, event: AuditEvent) -> Result<(), AuditError>;
}
```

**Rationale**: 
- Audit writes should not block VRAM operations
- Buffering and batching for performance
- Network I/O for platform mode

---

## 3. Event Format

### 3.1 Standard Fields

**All events MUST include**:
```rust
pub struct AuditEventEnvelope {
    pub audit_id: String,           // Unique ID: "audit-2025-1001-164805-abc123"
    pub timestamp: DateTime<Utc>,   // ISO 8601 UTC
    pub event_type: String,         // "vram.sealed", "seal.verified", etc.
    pub worker_id: String,          // "worker-gpu-0"
    pub event: AuditEvent,          // Event-specific data
    pub signature: Option<String>,  // HMAC signature (platform mode)
}
```

---

### 3.2 Structured Logging

**Events MUST be machine-readable JSON**:
```json
{
  "audit_id": "audit-2025-1001-200532-abc123",
  "timestamp": "2025-10-01T20:05:32Z",
  "event_type": "vram.sealed",
  "worker_id": "worker-gpu-0",
  "event": {
    "shard_id": "shard-abc123",
    "gpu_device": 0,
    "vram_bytes": 8000000000,
    "digest": "abc123def456789012345678901234567890123456789012345678901234"
  }
}
```

---

## 4. Security Requirements

### 4.1 Immutability

**Audit logs MUST be append-only**:
- No updates to existing events
- No deletions (except for retention policy)
- Tamper-evident storage (hash chain)

---

### 4.2 Tamper Evidence

**Each event MUST include hash of previous event**:
```rust
pub struct AuditEventEnvelope {
    pub prev_hash: String,  // SHA-256 of previous event
    pub hash: String,       // SHA-256 of this event
}
```

**Verification**:
```rust
impl AuditLogger {
    pub fn verify_integrity(&self) -> Result<(), AuditError> {
        // Verify hash chain
        // Detect tampering
    }
}
```

---

### 4.3 Sensitive Data Redaction

**MUST NOT log**:
- VRAM pointers
- Seal secret keys
- Raw model bytes
- Internal memory addresses

**Example**:
```rust
// ❌ FORBIDDEN
AuditEvent::VramSealed {
    vram_ptr: 0x7f8a4c000000,  // Never log pointers
}

// ✅ CORRECT
AuditEvent::VramSealed {
    shard_id: "shard-abc123",  // Opaque ID only
}
```

---

### 4.4 Access Control

**Audit logs MUST be read-only**:
- Only administrators can read
- No API for updates/deletes
- Audit log access is itself audited

---

## 5. Usage in vram-residency

### 5.1 Seal Operation Audit

```rust
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

### 5.2 Seal Verification Audit

```rust
impl VramManager {
    pub async fn verify_sealed(&self, shard: &SealedShard) -> Result<()> {
        // Re-compute digest
        let current_digest = self.compute_vram_digest(shard)?;
        
        // Verify seal
        if current_digest != shard.digest {
            // AUDIT: Verification FAILED (security incident)
            self.audit_logger.emit(AuditEvent::SealVerificationFailed {
                timestamp: Utc::now(),
                shard_id: shard.shard_id.clone(),
                reason: "digest_mismatch".to_string(),
                expected_digest: shard.digest.clone(),
                actual_digest: current_digest,
                worker_id: self.worker_id.clone(),
                severity: "critical".to_string(),
            }).await?;
            
            return Err(VramError::SealVerificationFailed);
        }
        
        // AUDIT: Verification passed
        self.audit_logger.emit(AuditEvent::SealVerified {
            timestamp: Utc::now(),
            shard_id: shard.shard_id.clone(),
            result: AuditResult::Success,
            worker_id: self.worker_id.clone(),
        }).await?;
        
        Ok(())
    }
}
```

---

### 5.3 VRAM Allocation Audit

```rust
impl VramManager {
    fn allocate_vram(&mut self, size: usize) -> Result<*mut c_void> {
        let available = self.total_vram.saturating_sub(self.used_vram);
        
        if size > available {
            // AUDIT: Allocation failed
            tokio::spawn({
                let logger = self.audit_logger.clone();
                let worker_id = self.worker_id.clone();
                async move {
                    logger.emit(AuditEvent::VramAllocationFailed {
                        timestamp: Utc::now(),
                        requested_bytes: size,
                        available_bytes: available,
                        reason: "insufficient_vram".to_string(),
                        gpu_device: 0,
                        worker_id,
                    }).await
                }
            });
            
            return Err(VramError::InsufficientVram(size, available));
        }
        
        // Allocate VRAM
        let ptr = unsafe { cuda_malloc(size)? };
        self.used_vram = self.used_vram.saturating_add(size);
        
        // AUDIT: Allocation succeeded
        tokio::spawn({
            let logger = self.audit_logger.clone();
            let worker_id = self.worker_id.clone();
            let used = self.used_vram;
            let total = self.total_vram;
            async move {
                logger.emit(AuditEvent::VramAllocated {
                    timestamp: Utc::now(),
                    requested_bytes: size,
                    allocated_bytes: size,
                    available_bytes: total - used,
                    used_bytes: used,
                    gpu_device: 0,
                    worker_id,
                }).await
            }
        });
        
        Ok(ptr)
    }
}
```

---

### 5.4 Policy Violation Audit

```rust
impl VramManager {
    pub async fn enforce_vram_only_policy(&self) -> Result<()> {
        // Check for unified memory
        if unified_memory_detected()? {
            // AUDIT: Policy violation (critical)
            self.audit_logger.emit(AuditEvent::PolicyViolation {
                timestamp: Utc::now(),
                policy: "vram_only".to_string(),
                violation: "unified_memory_detected".to_string(),
                details: "UMA enabled, cannot enforce VRAM-only policy".to_string(),
                severity: "critical".to_string(),
                worker_id: self.worker_id.clone(),
                action_taken: "worker_stopped".to_string(),
            }).await?;
            
            return Err(VramError::PolicyViolation(
                "Unified memory detected"
            ));
        }
        
        Ok(())
    }
}
```

---

## 6. Performance Considerations

### 6.1 Async Emission

**Audit MUST NOT block VRAM operations**:
```rust
// ✅ CORRECT: Spawn async task
tokio::spawn({
    let logger = self.audit_logger.clone();
    async move {
        logger.emit(event).await
    }
});

// ❌ WRONG: Blocking
self.audit_logger.emit(event).await?;  // Blocks VRAM operation
```

---

### 6.2 Buffering

**Audit logger SHOULD buffer events**:
- Batch writes every 1 second or 100 events
- Flush on critical events (seal verification failure)
- Flush on worker shutdown

---

### 6.3 Error Handling

**Audit failures MUST NOT crash worker**:
```rust
// Log audit error but continue
if let Err(e) = self.audit_logger.emit(event).await {
    tracing::error!(error = %e, "Failed to emit audit event");
    // Continue operation
}
```

---

## 7. Storage

### 7.1 Local Storage

**Default location**:
```
/var/lib/llorch/audit/worker-gpu-0/
  ├─ 2025-10-01.audit       # Daily log file
  ├─ 2025-10-01.audit.sha256 # Checksum
  └─ manifest.json          # File index
```

---

### 7.2 Platform Mode

**For platform deployments**:
- Events sent to central audit service
- Provider signs events with private key
- Platform validates signatures
- Immutable storage (S3 Object Lock)

---

## 8. Dependencies

**Required crates**:
- `chrono` — Timestamps
- `serde` — Serialization
- `tokio` — Async I/O
- `sha2` — Hash chain
- `thiserror` — Error types

---

## 9. Implementation Priority

### Phase 1: M0 Essentials
1. ✅ Event types for vram-residency
2. ✅ Local file-based audit logger
3. ✅ Async event emission
4. ✅ Basic tamper evidence (hash chain)

### Phase 2: Production Hardening
5. ⬜ Platform mode support
6. ⬜ Event buffering and batching
7. ⬜ Integrity verification
8. ⬜ Query API

---

## 10. References

**Specifications**:
- `bin/worker-orcd-crates/vram-residency/.specs/20_security.md` — Security requirements (Section 7.1)
- `bin/worker-orcd-crates/vram-residency/.specs/10_expectations.md` — Consumer expectations (Section 10.2)
- `bin/shared-crates/audit-logging/README.md` — Audit logging overview

**Standards**:
- GDPR — Data processing records
- SOC2 — Audit trail requirements
- ISO 27001 — Security event logging

---

**End of Requirements Document**
