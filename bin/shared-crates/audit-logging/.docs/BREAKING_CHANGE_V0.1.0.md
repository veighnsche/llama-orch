# BREAKING CHANGE: emit() is now synchronous

**Date**: 2025-10-02  
**Version**: 0.1.0  
**Type**: API Breaking Change  
**Severity**: LOW (easy migration)

---

## Summary

The `emit()` method is now **synchronous** (no longer `async`). The `emit_sync()` method has been **removed** as it was duplicate code.

**Rationale**: The `emit()` method never actually awaited anything — it used `try_send()` which is already synchronous and non-blocking. Having both `emit()` and `emit_sync()` was unnecessary code duplication.

---

## What Changed

### Before (v0.0.0)

```rust
// emit() was async (but didn't actually await anything)
pub async fn emit(&self, event: AuditEvent) -> Result<()>

// emit_sync() was a duplicate
pub fn emit_sync(&self, event: AuditEvent) -> Result<()>
```

### After (v0.1.0)

```rust
// emit() is now synchronous (no duplication)
pub fn emit(&self, event: AuditEvent) -> Result<()>
```

---

## Migration Guide

### If you were using `emit()` (async)

**Before**:
```rust
audit_logger.emit(AuditEvent::AuthSuccess {
    // ...
}).await?;
```

**After** (remove `.await`):
```rust
audit_logger.emit(AuditEvent::AuthSuccess {
    // ...
})?;
```

**Migration effort**: Just remove `.await` — that's it!

### If you were using `emit_sync()`

**Before**:
```rust
audit_logger.emit_sync(AuditEvent::VramSealed {
    // ...
})?;
```

**After** (rename to `emit`):
```rust
audit_logger.emit(AuditEvent::VramSealed {
    // ...
})?;
```

**Migration effort**: Just rename `emit_sync` → `emit` — that's it!

---

## Why This Change?

### 1. **No Actual Async Behavior**

The `emit()` method never awaited anything:
```rust
pub async fn emit(&self, event: AuditEvent) -> Result<()> {
    // ... validation ...
    
    // This is synchronous and non-blocking!
    self.tx.try_send(WriterMessage::Event(envelope))
        .map_err(|_| AuditError::BufferFull)?;
    
    Ok(())
    // No .await anywhere!
}
```

The `async` keyword was misleading — it suggested blocking I/O when there was none.

### 2. **Code Duplication**

Having both `emit()` and `emit_sync()` was 100% duplicate code:
- Same implementation
- Same validation
- Same error handling
- Same performance

The only difference was the `async` keyword, which wasn't needed.

### 3. **Simpler API**

One method that works everywhere is better than two methods that do the same thing:
- ✅ Works from sync contexts
- ✅ Works from async contexts
- ✅ Works from Drop implementations
- ✅ No `.await` needed
- ✅ No confusion about which to use

---

## Security Impact

**None**. The implementation is identical:
- ✅ Same non-blocking behavior
- ✅ Same bounded buffer (1000 events)
- ✅ Same input validation
- ✅ Same counter overflow detection
- ✅ Same thread safety

---

## Performance Impact

**None**. The implementation is identical:
- ✅ Same `try_send()` call
- ✅ Same atomic counter
- ✅ Same background writer
- ✅ Zero overhead

---

## Compatibility

### Breaking Changes

- ❌ **Removed**: `emit_sync()` method
- ❌ **Changed**: `emit()` is no longer `async`

### Non-Breaking

- ✅ All event types unchanged
- ✅ All error types unchanged
- ✅ `AuditLogger::new()` unchanged
- ✅ `flush()` and `shutdown()` unchanged

---

## Migration Checklist

For each crate using `audit-logging`:

- [ ] Find all `audit_logger.emit(...).await` calls
- [ ] Remove `.await` from each call
- [ ] Find all `audit_logger.emit_sync(...)` calls
- [ ] Rename to `audit_logger.emit(...)`
- [ ] Run tests
- [ ] Done!

**Estimated time**: 5-10 minutes per crate

---

## Affected Crates

### Internal Crates (need migration)

1. **rbees-orcd** — If using audit logging
2. **pool-managerd** — If using audit logging
3. **worker-orcd** — If using audit logging
4. **vram-residency** — Will use new API (not yet integrated)

### External Consumers

**None** — audit-logging is not yet published or used externally.

---

## Testing

All tests updated and passing:
```
running 3 tests
test logger::tests::test_emit_from_sync_context ... ok
test logger::tests::test_counter_overflow_detection ... ok
test logger::tests::test_emit_counter_overflow ... ok

test result: ok. 3 passed; 0 failed; 0 ignored
```

---

## Documentation Updates

- [x] README.md updated with new API
- [x] Inline docs updated
- [x] Examples updated
- [x] Integration plan updated

---

## Rollback Plan

If needed, we can revert to v0.0.0:
```bash
git revert <this-commit>
```

But this is unlikely to be needed since:
- Migration is trivial (just remove `.await`)
- No functionality changes
- No security changes
- No performance changes

---

## Approval

**Requested by**: User (vram-residency team feedback)  
**Rationale**: Eliminate code duplication, simplify API  
**Risk**: LOW (easy migration, no functionality change)  
**Benefit**: HIGH (simpler API, less confusion)

---

**Status**: ✅ IMPLEMENTED  
**Version**: 0.1.0  
**Date**: 2025-10-02
