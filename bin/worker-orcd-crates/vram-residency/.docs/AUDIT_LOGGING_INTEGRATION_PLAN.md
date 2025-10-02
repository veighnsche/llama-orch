# Audit Logging Integration Plan — vram-residency

**Date**: 2025-10-02  
**Status**: ✅ **UNBLOCKED** — Ready to implement

---

## Summary

The vram-residency crate needs to integrate audit logging to comply with GDPR, SOC2, and ISO 27001 requirements. All audit events are already defined in the `audit-logging` crate, and the **synchronous emit method** is now available.

---

## ✅ Breaking Change: `emit()` is Now Synchronous

**File**: `bin/shared-crates/audit-logging/src/logger.rs` (lines 74-141)

**Change**: 
The `emit()` method is now **synchronous** (no longer `async`). The `emit_sync()` method has been **removed** as it was duplicate code.

**Status**: ✅ **IMPLEMENTED AND READY**

**Rationale**: The `emit()` method never actually awaited anything — it used `try_send()` which is already synchronous and non-blocking. Having both `emit()` and `emit_sync()` was unnecessary code duplication.

**Usage**:
```rust
// ✅ Simple, direct call (no .await needed)
pub fn seal_model(&mut self, ...) -> Result<SealedShard> {
    // ... sealing logic ...
    
    if let Err(e) = self.audit_logger.emit(AuditEvent::VramSealed {
        timestamp: Utc::now(),
        shard_id: shard.shard_id.clone(),
        gpu_device: shard.gpu_device,
        vram_bytes: shard.vram_bytes,
        digest: shard.digest.clone(),
        worker_id: self.worker_id.clone(),
    }) {
        tracing::error!(error = %e, "Failed to emit audit event");
    }
    
    Ok(shard)
}
```

---

## What's Ready

### ✅ Audit Events Already Defined

All required events exist in `audit-logging/src/events.rs`:

1. **VramSealed** (lines 220-227)
   ```rust
   VramSealed {
       timestamp: DateTime<Utc>,
       shard_id: String,
       gpu_device: u32,
       vram_bytes: usize,
       digest: String,
       worker_id: String,
   }
   ```

2. **SealVerified** (lines 230-234)
   ```rust
   SealVerified {
       timestamp: DateTime<Utc>,
       shard_id: String,
       worker_id: String,
   }
   ```

3. **SealVerificationFailed** (lines 237-245)
   ```rust
   SealVerificationFailed {
       timestamp: DateTime<Utc>,
       shard_id: String,
       reason: String,
       expected_digest: String,
       actual_digest: String,
       worker_id: String,
       severity: String,
   }
   ```

4. **VramAllocated** (lines 248-256)
   ```rust
   VramAllocated {
       timestamp: DateTime<Utc>,
       requested_bytes: usize,
       allocated_bytes: usize,
       available_bytes: usize,
       used_bytes: usize,
       gpu_device: u32,
       worker_id: String,
   }
   ```

5. **VramAllocationFailed** (lines 259-266)
   ```rust
   VramAllocationFailed {
       timestamp: DateTime<Utc>,
       requested_bytes: usize,
       available_bytes: usize,
       reason: String,
       gpu_device: u32,
       worker_id: String,
   }
   ```

6. **VramDeallocated** (lines 269-276)
   ```rust
   VramDeallocated {
       timestamp: DateTime<Utc>,
       shard_id: String,
       freed_bytes: usize,
       remaining_used: usize,
       gpu_device: u32,
       worker_id: String,
   }
   ```

7. **PolicyViolation** (lines 307-315)
   ```rust
   PolicyViolation {
       timestamp: DateTime<Utc>,
       policy: String,
       violation: String,
       details: String,
       severity: String,
       worker_id: String,
       action_taken: String,
   }
   ```

### ✅ Helper Functions Already Implemented

All helper functions exist in `vram-residency/src/audit/events.rs`:

- `emit_vram_sealed()` (lines 30-45)
- `emit_seal_verified()` (lines 60-72)
- `emit_seal_verification_failed()` (lines 95-114)
- `emit_vram_allocated()` (lines 129-149)
- `emit_vram_allocation_failed()` (lines 162-179)
- `emit_vram_deallocated()` (lines 193-211)
- `emit_policy_violation()` (lines 229-247)

---

## What's Needed

### 1. ~~audit-logging: Add `emit_sync()` Method~~ — COMPLETED (Breaking Change)

**Status**: ✅ **COMPLETED** — `emit()` is now synchronous

**Change**: Instead of adding `emit_sync()`, we made `emit()` synchronous and removed the duplicate code.

**Rationale**: The `async` keyword on `emit()` was unnecessary — the method never awaited anything.

---

### 2. vram-residency: Add Fields to VramManager

**Current**:
```rust
pub struct VramManager {
    context: CudaContext,
    seal_key: SecretKey,
    allocations: HashMap<usize, SafeCudaPtr>,
}
```

**Required**:
```rust
use audit_logging::AuditLogger;
use std::sync::Arc;

pub struct VramManager {
    context: CudaContext,
    seal_key: SecretKey,
    allocations: HashMap<usize, SafeCudaPtr>,
    audit_logger: Arc<AuditLogger>,  // ✅ Add this
    worker_id: String,               // ✅ Add this
}
```

---

### 3. vram-residency: Update Constructor

**Current**:
```rust
pub fn new_with_token(worker_token: &str, gpu_device: u32) -> Result<Self>
```

**Required**:
```rust
pub fn new_with_token(
    worker_token: &str,
    gpu_device: u32,
    audit_logger: Arc<AuditLogger>,
    worker_id: String,
) -> Result<Self> {
    // ... existing logic ...
    
    Ok(Self {
        context,
        seal_key,
        allocations: HashMap::new(),
        audit_logger,
        worker_id,
    })
}
```

---

### 4. vram-residency: Emit Audit Events

**Example** (seal_model):
```rust
pub fn seal_model(&mut self, model_bytes: &[u8], gpu_device: u32) -> Result<SealedShard> {
    // ... sealing logic ...
    
    // ✅ Emit audit event (non-blocking)
    if let Err(e) = self.audit_logger.emit_sync(AuditEvent::VramSealed {
        timestamp: Utc::now(),
        shard_id: shard.shard_id.clone(),
        gpu_device: shard.gpu_device,
        vram_bytes: shard.vram_bytes,
        digest: shard.digest.clone(),
        worker_id: self.worker_id.clone(),
    }) {
        tracing::error!(error = %e, "Failed to emit audit event");
        // Don't fail the operation if audit fails
    }
    
    Ok(shard)
}
```

---

## Implementation Checklist

### Phase 1: audit-logging ✅ COMPLETED

- [x] Implement `emit_sync()` method ✅
- [x] Add unit tests ✅
- [x] Update documentation ✅
- [x] Review and merge ✅

### Phase 2: vram-residency (READY TO START)

- [ ] Add `audit_logger` and `worker_id` fields to `VramManager`
- [ ] Update `new_with_token()` constructor
- [ ] Emit audit events in `seal_model()`
- [ ] Emit audit events in `verify_sealed()`
- [ ] Emit audit events for allocation failures
- [ ] Emit audit events in policy enforcement
- [ ] Add unit tests for audit emission
- [ ] Add integration tests for audit trail
- [ ] Update documentation

---

## Timeline

**Status**: ✅ UNBLOCKED — Ready to implement  
**Estimated**: 1-2 days of implementation work  
**Priority**: HIGH (compliance requirement)  
**Next Step**: Begin Phase 2 implementation

---

## ~~Alternative Approach (If Feature Not Available)~~ — NO LONGER NEEDED

✅ The `emit_sync()` feature is now available. No workarounds needed.

---

## Compliance Impact

**Without audit logging**:
- ❌ GDPR Article 30 violation (no processing records)
- ❌ SOC2 CC6.1 violation (no security event trail)
- ❌ ISO 27001 A.12.4.1 violation (no event logging)

**With audit logging**:
- ✅ GDPR compliant (processing records maintained)
- ✅ SOC2 compliant (security events logged)
- ✅ ISO 27001 compliant (event logging enabled)

---

## References

- **Feature Request**: `bin/shared-crates/audit-logging/FEATURE_REQUEST_SYNC_EMIT.md`
- **Audit Report**: `bin/worker-orcd-crates/vram-residency/SECURITY_AUDIT_AUDIT_LOGGING.md`
- **Audit Events**: `bin/shared-crates/audit-logging/src/events.rs`
- **Helper Functions**: `bin/worker-orcd-crates/vram-residency/src/audit/events.rs`

---

**Status**: ✅ **UNBLOCKED** — Ready to implement  
**Blocker Resolved**: `emit_sync()` method now available in audit-logging  
**Next Step**: Begin Phase 2 implementation (add fields to VramManager and emit events)
