# Audit Logging & Narration Requirements for vram-residency

**Date**: 2025-10-02  
**Status**: Analysis for audit-logging and narration-core teams  
**Question**: Does vram-residency need audit logging and/or narration?

---

## Executive Summary

**Answer**: **YES, but with different priorities**

| Capability | Required? | Priority | Rationale |
|------------|-----------|----------|-----------|
| **Audit Logging** | ✅ YES | **P0 - Critical** | Security requirement (TIER 1) |
| **Narration/Tracing** | ✅ YES | **P1 - High** | Debugging & observability |
| **Structured Logging** | ✅ YES | **P1 - High** | Production diagnostics |

---

## 1. Audit Logging Requirements

### 1.1 Security Specification Mandates

**From `.specs/20_security.md`**:

- **RP-005**: VRAM deallocation MUST be tracked for audit trail
- **MS-005**: Drop implementation MUST log deallocation
- **CI-006**: Seal secret keys MUST NOT be logged (audit must filter)

### 1.2 Critical Security Events

The following events **MUST** be audited per security spec:

#### 1. VramSealed (CRITICAL)
```rust
audit_logger.emit(AuditEvent::VramSealed {
    timestamp: Utc::now(),
    shard_id: shard.shard_id.clone(),
    gpu_device: shard.gpu_device,
    vram_bytes: shard.vram_bytes,
    digest: shard.digest.clone(),
    worker_id: config.worker_id.clone(),
}).await?;
```

**Why**: Cryptographic seal creation is a trust anchor for the entire system.

#### 2. SealVerified (HIGH)
```rust
audit_logger.emit(AuditEvent::SealVerified {
    timestamp: Utc::now(),
    shard_id: shard.shard_id.clone(),
    gpu_device: shard.gpu_device,
    digest: shard.digest.clone(),
}).await?;
```

**Why**: Confirms VRAM integrity before inference execution.

#### 3. SealVerificationFailed (CRITICAL)
```rust
audit_logger.emit(AuditEvent::SealVerificationFailed {
    timestamp: Utc::now(),
    shard_id: shard.shard_id.clone(),
    reason: "digest mismatch".to_string(),
    severity: "CRITICAL",
}).await?;
```

**Why**: Indicates VRAM corruption or tampering. Worker MUST stop.

#### 4. VramDeallocated (MEDIUM)
```rust
audit_logger.emit(AuditEvent::VramDeallocated {
    timestamp: Utc::now(),
    shard_id: self.shard_id.clone(),
    freed_bytes: self.vram_bytes,
}).await?;
```

**Why**: Required for capacity tracking and leak detection.

#### 5. PolicyViolation (CRITICAL)
```rust
audit_logger.emit(AuditEvent::PolicyViolation {
    timestamp: Utc::now(),
    reason: "UMA detected on GPU".to_string(),
    severity: "CRITICAL",
}).await?;
```

**Why**: VRAM-only policy violation. Worker MUST NOT start.

### 1.3 Current Implementation

**Status**: ⚠️ **Placeholder using `tracing`**

Current code in `src/audit/events.rs`:
```rust
/// Emit VramSealed audit event
pub fn emit_vram_sealed(shard: &SealedShard) {
    tracing::info!(
        event = "VramSealed",
        shard_id = %shard.shard_id,
        gpu_device = %shard.gpu_device,
        vram_bytes = %shard.vram_bytes,
        digest = %&shard.digest[..16],
    );
}
```

**Problem**: `tracing` is NOT tamper-evident and lacks cryptographic integrity.

### 1.4 Integration Requirements

**From `audit-logging` crate** (per `.specs/31_dependency_verification.md`):

✅ Provides all required event types:
- `AuditEvent::VramSealed`
- `AuditEvent::SealVerified`
- `AuditEvent::SealVerificationFailed`
- `AuditEvent::VramDeallocated`
- `AuditEvent::PolicyViolation`

✅ Provides tamper-evident logging:
- Cryptographic integrity (HMAC chain)
- Immutable append-only log
- Structured event types

✅ Non-blocking async API:
```rust
let audit_logger = AuditLogger::new(config)?;
audit_logger.emit(event).await?; // Non-blocking
```

### 1.5 Migration Path

**Phase 1** (Current - M0): Use `tracing` placeholder
**Phase 2** (Post-M0): Integrate `audit-logging` crate

```rust
// Replace this:
pub fn emit_vram_sealed(shard: &SealedShard) {
    tracing::info!(event = "VramSealed", ...);
}

// With this:
pub async fn emit_vram_sealed(
    audit_logger: &AuditLogger,
    shard: &SealedShard
) -> Result<()> {
    audit_logger.emit(AuditEvent::VramSealed {
        timestamp: Utc::now(),
        shard_id: shard.shard_id.clone(),
        gpu_device: shard.gpu_device,
        vram_bytes: shard.vram_bytes,
        digest: shard.digest.clone(),
        worker_id: audit_logger.worker_id().clone(),
    }).await
}
```

---

## 2. Narration/Tracing Requirements

### 2.1 Observability Needs

**Purpose**: Debugging, performance analysis, production diagnostics

**Current Usage**: 16 `tracing::` call sites across the crate

#### Debug-Level Tracing (11 sites)
- CUDA allocation/deallocation
- Validation steps
- Policy enforcement checks
- Digest computation
- Key derivation

#### Info-Level Tracing (4 sites)
- CUDA context initialization
- GPU device selection
- Policy enforcement status

#### Error-Level Tracing (1 site)
- CUDA free failures in Drop

### 2.2 Critical Narration Points

#### 1. CUDA Context Initialization
```rust
tracing::info!(
    device = %device,
    name = %gpu.name,
    vram_gb = %(gpu.vram_total_bytes / 1024 / 1024 / 1024),
    "CUDA context initialized"
);
```

**Why**: Confirms GPU selection and VRAM capacity at startup.

#### 2. VRAM Allocation
```rust
tracing::debug!(
    size = %size,
    device = %self.device,
    "CUDA allocate completed"
);
```

**Why**: Track memory allocation patterns for capacity planning.

#### 3. VRAM Deallocation Errors
```rust
tracing::error!(
    size = %self.size,
    device = %self.device,
    error_code = %result,
    "CUDA free failed in Drop (non-fatal)"
);
```

**Why**: Detect VRAM leaks and CUDA driver issues.

#### 4. Policy Enforcement
```rust
tracing::info!(
    gpu_device = %gpu_device,
    "VRAM-only policy enforced: using cudaMalloc exclusively"
);
```

**Why**: Confirm VRAM-only policy is active.

### 2.3 Narration-Core Integration

**Question**: Should vram-residency use `narration-core`?

**Answer**: ✅ **YES, but not immediately**

**Rationale**:
- Current `tracing` is sufficient for M0
- `narration-core` provides richer context (spans, correlation IDs)
- Integration can happen post-M0 alongside `audit-logging`

**Benefits of narration-core**:
- Structured spans for seal/verify operations
- Request correlation across worker-orcd → vram-residency
- Performance metrics (seal latency, allocation time)
- Distributed tracing support

---

## 3. Current State Analysis

### 3.1 Dependencies

**Current `Cargo.toml`**:
```toml
[dependencies]
thiserror.workspace = true
tracing.workspace = true      # ← Basic tracing only
sha2.workspace = true
hmac.workspace = true
subtle.workspace = true
hkdf.workspace = true
gpu-info = { path = "../../shared-crates/gpu-info" }
```

**Missing**:
- ❌ `audit-logging` - Required for security compliance
- ❌ `narration-core` - Optional but recommended

### 3.2 Code Organization

```
src/
├── audit/
│   ├── mod.rs
│   └── events.rs          # ← Placeholder audit events (uses tracing)
├── cuda_ffi/
│   └── mod.rs             # ← CUDA operations (uses tracing::debug)
├── policy/
│   ├── enforcement.rs     # ← Policy checks (uses tracing::info)
│   └── validation.rs      # ← Device validation (uses tracing::info)
├── seal/
│   ├── digest.rs          # ← Digest computation (uses tracing::debug)
│   └── key_derivation.rs  # ← Key derivation (uses tracing::debug)
└── validation/
    ├── gpu_device.rs      # ← Input validation (uses tracing::debug)
    └── shard_id.rs        # ← Input validation (uses tracing::debug)
```

### 3.3 Test Coverage

**From `.specs/40_testing.md`**:

> 95% of code is GPU-agnostic (cryptography, validation, **audit**)

**Audit test requirements**:
```rust
#[test]
fn test_seal_operation_audited() {
    let (audit_logger, receiver) = create_test_audit_logger();
    let manager = VramManager::new_mock_with_audit(1024 * 1024, audit_logger)?;
    
    let shard = manager.seal_model("test", 0, &[0u8; 1024])?;
    
    // Check audit event was emitted
    let events = receiver.try_recv_all();
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], AuditEvent::VramSealed { .. }));
}
```

**Status**: ⬜ Not yet implemented (requires `audit-logging` integration)

---

## 4. Recommendations

### 4.1 For audit-logging Team

**Action Required**: ✅ **vram-residency NEEDS audit-logging**

**Priority**: **P0 - Blocking for production**

**Scope**:
1. ✅ Event types already defined in `audit-logging` crate
2. ✅ API matches vram-residency expectations
3. ⬜ Integration work needed in vram-residency

**Timeline**:
- **M0**: Use `tracing` placeholder (acceptable for alpha)
- **Post-M0**: Integrate `audit-logging` (required for production)

**Integration Points**:
```rust
// VramManager needs AuditLogger dependency
pub struct VramManager {
    allocator: Box<dyn VramAllocator>,
    audit_logger: Arc<AuditLogger>,  // ← Add this
}

// All seal/verify operations emit audit events
impl VramManager {
    pub async fn seal_model(&mut self, ...) -> Result<SealedShard> {
        // ... seal logic ...
        
        self.audit_logger.emit(AuditEvent::VramSealed {
            // ... event data ...
        }).await?;
        
        Ok(shard)
    }
}
```

### 4.2 For narration-core Team

**Action Required**: ⚠️ **vram-residency SHOULD use narration-core**

**Priority**: **P1 - Recommended for production**

**Scope**:
1. Replace basic `tracing::` calls with structured spans
2. Add correlation IDs for request tracing
3. Add performance metrics (seal latency, allocation time)

**Timeline**:
- **M0**: Current `tracing` is sufficient
- **Post-M0**: Integrate `narration-core` for richer observability

**Integration Points**:
```rust
// Replace this:
tracing::info!(
    device = %device,
    "CUDA context initialized"
);

// With this:
narration::span!("cuda_context_init", {
    device: device,
    vram_gb: gpu.vram_total_bytes / 1024 / 1024 / 1024,
}).info("CUDA context initialized");
```

### 4.3 For vram-residency Team

**Immediate Actions**:
1. ✅ Keep current `tracing` placeholder for M0
2. ⬜ Add `audit-logging` dependency post-M0
3. ⬜ Refactor `src/audit/events.rs` to use `AuditLogger`
4. ⬜ Add audit event tests (per `.specs/40_testing.md`)

**Future Actions**:
5. ⬜ Consider `narration-core` for structured observability
6. ⬜ Add performance metrics (seal latency, VRAM allocation time)
7. ⬜ Add distributed tracing support

---

## 5. Dependency Matrix

| Feature | Current | M0 Target | Production Target |
|---------|---------|-----------|-------------------|
| **Basic Logging** | ✅ `tracing` | ✅ `tracing` | ✅ `tracing` |
| **Audit Events** | ⚠️ `tracing` (placeholder) | ⚠️ `tracing` (acceptable) | ✅ `audit-logging` (required) |
| **Structured Spans** | ❌ None | ❌ None | ✅ `narration-core` (recommended) |
| **Performance Metrics** | ❌ None | ❌ None | ✅ `narration-core` (recommended) |
| **Correlation IDs** | ❌ None | ❌ None | ✅ `narration-core` (recommended) |

---

## 6. Security Implications

### 6.1 Why Audit Logging is Critical

**From security spec (TIER 1)**:

1. **Tamper Detection**: Seal verification failures indicate VRAM corruption
2. **Forensics**: Audit trail required for incident investigation
3. **Compliance**: Cryptographic integrity verification requires audit
4. **Accountability**: Track who sealed/verified which models

### 6.2 What Happens Without Audit Logging?

❌ **Cannot detect**:
- Seal forgery attempts
- VRAM corruption patterns
- Policy violation attempts
- Resource exhaustion attacks

❌ **Cannot investigate**:
- Security incidents
- Performance degradation
- VRAM leaks
- Capacity issues

❌ **Cannot prove**:
- Model integrity
- VRAM-only policy enforcement
- Cryptographic seal validity

---

## 7. Summary

### For audit-logging Team

**Question**: Does vram-residency need audit-logging?  
**Answer**: ✅ **YES - P0 Critical**

**Reason**: Security specification (TIER 1) mandates tamper-evident audit trail for:
- Seal creation (trust anchor)
- Seal verification (integrity check)
- Policy violations (security incidents)
- VRAM deallocation (capacity tracking)

**Current State**: Using `tracing` placeholder (acceptable for M0)  
**Production Requirement**: Must integrate `audit-logging` crate  
**Timeline**: Post-M0 integration

### For narration-core Team

**Question**: Does vram-residency need narration-core?  
**Answer**: ⚠️ **RECOMMENDED - P1 High**

**Reason**: Production observability requires:
- Structured spans for seal/verify operations
- Request correlation across services
- Performance metrics (latency, throughput)
- Distributed tracing support

**Current State**: Using basic `tracing` (sufficient for M0)  
**Production Recommendation**: Integrate `narration-core` for richer observability  
**Timeline**: Post-M0 enhancement

---

**Conclusion**: Both `audit-logging` (critical) and `narration-core` (recommended) are needed for production-ready vram-residency, but current `tracing` placeholder is acceptable for M0 alpha release.

---

**Document Owner**: vram-residency team  
**Stakeholders**: audit-logging team, narration-core team, worker-orcd team  
**Last Updated**: 2025-10-02
