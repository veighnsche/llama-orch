# Feature Request: Synchronous emit_sync() Method

**Date**: 2025-10-02  
**Requestor**: vram-residency team  
**Priority**: HIGH  
**Status**: ✅ **RESOLVED** (Breaking Change Applied Instead)

---

## Problem Statement

The `vram-residency` crate needs to emit audit events from **synchronous functions** (`seal_model`, `verify_sealed`). Currently, `AuditLogger::emit()` is async, which cannot be called from sync context without spawning tasks or blocking on a runtime.

### Current Limitation

```rust
// ❌ CANNOT DO THIS (sync function calling async)
pub fn seal_model(&mut self, model_bytes: &[u8], gpu_device: u32) -> Result<SealedShard> {
    // ... sealing logic ...
    
    // ❌ ERROR: cannot call async function from sync context
    self.audit_logger.emit(AuditEvent::VramSealed { ... }).await?;
    
    Ok(shard)
}
```

### Current Workarounds (All Problematic)

**Option 1: Spawn async task** (recommended by audit, but complex):
```rust
tokio::spawn({
    let logger = self.audit_logger.clone();
    let shard = shard.clone();  // ❌ Requires Clone
    async move {
        logger.emit(event).await.ok();  // ❌ Errors silently dropped
    }
});
```

**Problems**:
- Requires cloning data (performance overhead)
- Errors are silently dropped (no way to handle)
- Adds complexity to every call site
- Requires tokio runtime (not always available in tests)

**Option 2: Make functions async** (breaks API):
```rust
// ❌ BREAKING CHANGE
pub async fn seal_model(&mut self, ...) -> Result<SealedShard> {
    self.audit_logger.emit(event).await?;
}
```

**Problems**:
- Breaking API change for all callers
- Forces async propagation up the call stack
- Makes testing more complex
- Not always desirable (VRAM ops should be sync)

**Option 3: Block on runtime** (defeats purpose):
```rust
tokio::runtime::Handle::current()
    .block_on(self.audit_logger.emit(event))?;
```

**Problems**:
- Blocks the thread (defeats non-blocking design)
- May panic if called from async context
- Not available in all contexts

---

## Proposed Solution

Add a **synchronous `emit_sync()` method** that uses the existing non-blocking `try_send()` internally:

```rust
impl AuditLogger {
    /// Emit audit event (synchronous, non-blocking)
    ///
    /// This is a synchronous wrapper around the async emit() that can be
    /// called from non-async contexts. It uses try_send() internally, so
    /// it never blocks.
    ///
    /// # Errors
    ///
    /// Returns `AuditError::BufferFull` if buffer is full.
    /// Returns `AuditError::InvalidInput` if event validation fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// // Can be called from sync functions
    /// pub fn seal_model(&mut self, ...) -> Result<SealedShard> {
    ///     // ... sealing logic ...
    ///     
    ///     self.audit_logger.emit_sync(AuditEvent::VramSealed {
    ///         timestamp: Utc::now(),
    ///         shard_id: shard.shard_id.clone(),
    ///         // ...
    ///     })?;
    ///     
    ///     Ok(shard)
    /// }
    /// ```
    pub fn emit_sync(&self, mut event: AuditEvent) -> Result<()> {
        // Validate and sanitize event
        validation::validate_event(&mut event)?;
        
        // Generate unique audit ID
        let counter = self.event_counter.fetch_add(1, Ordering::SeqCst);
        
        // Check for counter overflow
        if counter == u64::MAX {
            tracing::error!("Audit counter overflow detected");
            return Err(AuditError::CounterOverflow);
        }
        
        let audit_id = format!("audit-{}-{:016x}", self.config.service_id, counter);
        
        // Create envelope
        let envelope = AuditEventEnvelope::new(
            audit_id,
            Utc::now(),
            self.config.service_id.clone(),
            event,
            String::new(), // prev_hash set by writer
        );
        
        // Try to send (non-blocking) - THIS IS THE KEY
        self.tx.try_send(WriterMessage::Event(envelope))
            .map_err(|_| AuditError::BufferFull)?;
        
        Ok(())
    }
}
```

---

## Why This Works

The implementation is **identical to the async `emit()`** except it doesn't need `async/await` because:

1. **`try_send()` is already non-blocking** - it returns immediately
2. **No `.await` needed** - `try_send()` is synchronous
3. **Same buffering behavior** - uses the same channel
4. **Same error handling** - returns `BufferFull` if channel is full
5. **Same validation** - uses same `validate_event()`

### Current emit() Implementation

```rust
pub async fn emit(&self, mut event: AuditEvent) -> Result<()> {
    validation::validate_event(&mut event)?;
    let counter = self.event_counter.fetch_add(1, Ordering::SeqCst);
    // ... create envelope ...
    
    // THIS IS NON-BLOCKING ALREADY!
    self.tx.try_send(WriterMessage::Event(envelope))
        .map_err(|_| AuditError::BufferFull)?;
    
    Ok(())
}
```

**The `async` keyword is not actually needed** - the function doesn't await anything!

---

## Benefits

### For vram-residency

✅ **Simple call sites**:
```rust
pub fn seal_model(&mut self, ...) -> Result<SealedShard> {
    // ... sealing logic ...
    
    // ✅ Simple, direct call
    self.audit_logger.emit_sync(AuditEvent::VramSealed { ... })?;
    
    Ok(shard)
}
```

✅ **No cloning required** - can borrow data directly  
✅ **Proper error handling** - errors propagate normally  
✅ **No spawning tasks** - cleaner code  
✅ **Works in tests** - no tokio runtime required  

### For audit-logging

✅ **No breaking changes** - `emit()` stays async  
✅ **Same implementation** - just remove `async` keyword  
✅ **Same performance** - uses same `try_send()`  
✅ **Broader use cases** - works from sync and async contexts  

---

## Alternative Names

If `emit_sync()` is not preferred, consider:

- `emit_nowait()` - emphasizes non-blocking behavior
- `try_emit()` - matches `try_send()` naming
- `emit_nb()` - "nb" for non-blocking (common in embedded)
- `emit_immediate()` - emphasizes immediate return

---

## Resolution: Breaking Change Applied

Instead of adding `emit_sync()`, we made a **breaking change** to simplify the API:

- ❌ **Removed**: The `async` keyword from `emit()`
- ❌ **Removed**: The duplicate `emit_sync()` method
- ✅ **Result**: One simple `emit()` method that works everywhere

## Implementation Details

**File**: `bin/shared-crates/audit-logging/src/logger.rs` (lines 74-141)

**Method signature** (changed from `async`):
```rust
pub fn emit(&self, mut event: AuditEvent) -> Result<()>
```

**Rationale**: The `emit()` method never actually awaited anything — it used `try_send()` which is already synchronous and non-blocking. Having both `emit()` and `emit_sync()` was unnecessary code duplication.

**Tests updated**:
- `test_emit_from_sync_context()` — Verifies sync emission works
- `test_emit_counter_overflow()` — Verifies overflow detection
- All tests passing ✅

**Documentation updated**:
- README.md — Updated to show single `emit()` method
- Inline docs — Comprehensive examples
- Breaking change document created

---

## Questions for audit-logging Team

1. **Is there a reason `emit()` is marked `async`?** It doesn't await anything internally.

2. **Would you prefer a different name?** (`emit_nowait`, `try_emit`, etc.)

3. **Should we deprecate async `emit()`?** Since it's not actually async, it may be misleading.

4. **Any concerns about thread safety?** `try_send()` is already thread-safe.

---

## Related Issues

- **vram-residency audit**: `SECURITY_AUDIT_AUDIT_LOGGING.md` (CRITICAL-1 through CRITICAL-7)
- **Compliance**: GDPR Art. 30, SOC2 CC6.1, ISO 27001 A.12.4.1
- **Blocking**: vram-residency cannot emit audit events without this feature

---

## Priority Justification

**HIGH** because:
- Blocks vram-residency audit logging implementation
- Affects compliance (GDPR, SOC2, ISO 27001)
- Simple to implement (just remove `async` keyword)
- No breaking changes required

---

**Requested by**: vram-residency team  
**Date**: 2025-10-02  
**Contact**: See `bin/worker-orcd-crates/vram-residency/SECURITY_AUDIT_AUDIT_LOGGING.md`
