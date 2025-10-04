# 🎀 Narration v0.2.0 Migration Guide for FT Stories

**From**: Narration Core Team  
**Date**: 2025-10-04  
**Applies to**: All FT-001 through FT-010 stories

---

## What Changed in v0.2.0

### ✨ NEW: Builder Pattern API

**Before (v0.1.0)**:
```rust
use observability_narration_core::{narrate, NarrationFields};

narrate(NarrationFields {
    actor: "worker-orcd",
    action: "inference_start",
    target: job_id.clone(),
    correlation_id: Some(correlation_id),
    job_id: Some(job_id.clone()),
    tokens_in: Some(100),
    human: format!("Starting inference for job {}", job_id),
    ..Default::default()
});
```

**After (v0.2.0)** - 43% less code!:
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD, ACTION_INFERENCE_START};

Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_START, &job_id)
    .human(format!("Starting inference for job {}", job_id))
    .correlation_id(correlation_id)
    .job_id(&job_id)
    .tokens_in(100)
    .emit();
```

### ✨ NEW: Built-in Axum Middleware

**Before (v0.1.0)** - You had to write custom middleware:
```rust
// 50+ lines of custom middleware code...
```

**After (v0.2.0)** - Just use the built-in!:
```rust
use observability_narration_core::axum::correlation_middleware;

let app = Router::new()
    .route("/execute", post(handler))
    .layer(middleware::from_fn(correlation_middleware));  // ← Done!
```

### ✨ NEW: Level Methods

**Before (v0.1.0)**:
```rust
narrate_warn(NarrationFields { ... });
narrate_error(NarrationFields { ... });
narrate_debug(NarrationFields { ... });
```

**After (v0.2.0)**:
```rust
Narration::new(...).emit_warn();
Narration::new(...).emit_error();
Narration::new(...).emit_debug();
```

---

## Quick Conversion Guide

### Pattern 1: Basic Narration

**Old**:
```rust
narrate(NarrationFields {
    actor: "worker-orcd",
    action: "some_action",
    target: target_value.clone(),
    human: format!("Message {}", var),
    ..Default::default()
});
```

**New**:
```rust
Narration::new(ACTOR_WORKER_ORCD, "some_action", &target_value)
    .human(format!("Message {}", var))
    .emit();
```

### Pattern 2: With Correlation ID

**Old**:
```rust
narrate(NarrationFields {
    actor: "worker-orcd",
    action: "action",
    target: target.clone(),
    correlation_id: Some(correlation_id),
    job_id: Some(job_id.clone()),
    human: "Message".to_string(),
    ..Default::default()
});
```

**New**:
```rust
Narration::new(ACTOR_WORKER_ORCD, "action", &target)
    .human("Message")
    .correlation_id(correlation_id)
    .job_id(&job_id)
    .emit();
```

### Pattern 3: With Metrics

**Old**:
```rust
narrate(NarrationFields {
    actor: "worker-orcd",
    action: "inference_complete",
    target: job_id.clone(),
    tokens_out: Some(50),
    duration_ms: Some(150),
    human: format!("Completed: {} tokens", 50),
    ..Default::default()
});
```

**New**:
```rust
Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_COMPLETE, &job_id)
    .human(format!("Completed: {} tokens", 50))
    .tokens_out(50)
    .duration_ms(150)
    .emit();
```

### Pattern 4: Error Level

**Old**:
```rust
narrate_error(NarrationFields {
    actor: "worker-orcd",
    action: "spawn",
    target: "GPU0".to_string(),
    error_kind: Some("ResourceExhausted".to_string()),
    human: "Failed to spawn".to_string(),
    ..Default::default()
});
```

**New**:
```rust
Narration::new(ACTOR_WORKER_ORCD, ACTION_SPAWN, "GPU0")
    .human("Failed to spawn")
    .error_kind("ResourceExhausted")
    .emit_error();  // ← ERROR level
```

### Pattern 5: Warn Level

**Old**:
```rust
narrate_warn(NarrationFields {
    actor: "worker-orcd",
    action: "validation",
    target: job_id.clone(),
    error_kind: Some("validation_failed".to_string()),
    human: "Validation failed".to_string(),
    ..Default::default()
});
```

**New**:
```rust
Narration::new(ACTOR_WORKER_ORCD, "validation", &job_id)
    .human("Validation failed")
    .error_kind("ValidationFailed")
    .emit_warn();  // ← WARN level
```

### Pattern 6: Debug Level

**Old**:
```rust
narrate_debug(NarrationFields {
    actor: "worker-orcd",
    action: "stream_start",
    target: job_id.clone(),
    human: "Stream started".to_string(),
    ..Default::default()
});
```

**New**:
```rust
Narration::new(ACTOR_WORKER_ORCD, "stream_start", &job_id)
    .human("Stream started")
    .emit_debug();  // ← DEBUG level
```

### Pattern 7: With Cute & Story

**Old**:
```rust
narrate(NarrationFields {
    actor: "worker-orcd",
    action: "ready",
    target: "worker".to_string(),
    human: "Worker ready".to_string(),
    cute: Some("Worker is ready to help! 🎉".to_string()),
    story: Some("\"I'm ready!\" announced the worker.".to_string()),
    ..Default::default()
});
```

**New**:
```rust
Narration::new(ACTOR_WORKER_ORCD, "ready", "worker")
    .human("Worker ready")
    .cute("Worker is ready to help! 🎉")
    .story("\"I'm ready!\" announced the worker.")
    .emit();
```

---

## Key Changes

### 1. Import Changes

**Remove**:
```rust
use observability_narration_core::{narrate, narrate_warn, narrate_error, narrate_debug, NarrationFields};
```

**Add**:
```rust
use observability_narration_core::{
    Narration,
    ACTOR_WORKER_ORCD,
    ACTION_INFERENCE_START,
    ACTION_INFERENCE_COMPLETE,
    // ... other constants as needed
};
```

### 2. No More `.clone()` Spam

**Before**: `target: job_id.clone()`, `correlation_id: Some(correlation_id)`  
**After**: `&job_id`, `correlation_id` (builder handles conversion)

### 3. No More `Some()` Wrapping

**Before**: `correlation_id: Some(correlation_id)`, `tokens_out: Some(50)`  
**After**: `.correlation_id(correlation_id)`, `.tokens_out(50)`

### 4. No More `..Default::default()`

Builder handles defaults automatically!

---

## Available Constants

### Actors
- `ACTOR_ORCHESTRATORD` - "orchestratord"
- `ACTOR_POOL_MANAGERD` - "pool-managerd"
- `ACTOR_WORKER_ORCD` - "worker-orcd"
- `ACTOR_INFERENCE_ENGINE` - "inference-engine"
- `ACTOR_VRAM_RESIDENCY` - "vram-residency"

### Actions
- `ACTION_ADMISSION` - "admission"
- `ACTION_ENQUEUE` - "enqueue"
- `ACTION_DISPATCH` - "dispatch"
- `ACTION_SPAWN` - "spawn"
- `ACTION_READY_CALLBACK` - "ready_callback"
- `ACTION_HEARTBEAT_SEND` - "heartbeat_send"
- `ACTION_HEARTBEAT_RECEIVE` - "heartbeat_receive"
- `ACTION_SHUTDOWN` - "shutdown"
- `ACTION_INFERENCE_START` - "inference_start"
- `ACTION_INFERENCE_COMPLETE` - "inference_complete"
- `ACTION_INFERENCE_ERROR` - "inference_error"
- `ACTION_CANCEL` - "cancel"
- `ACTION_SEAL` - "seal"

---

## FT-004: Use Built-In Middleware!

**IMPORTANT**: FT-004 (Correlation ID Middleware) is now **much simpler**!

Instead of implementing custom middleware, just use:

```rust
use observability_narration_core::axum::correlation_middleware;

let app = Router::new()
    .route("/execute", post(execute_handler))
    .layer(middleware::from_fn(correlation_middleware));
```

The middleware automatically:
- ✅ Extracts `X-Correlation-ID` from headers
- ✅ Validates UUID v4 format
- ✅ Generates new ID if missing/invalid
- ✅ Stores in request extensions
- ✅ Adds to response headers

**No custom code needed!**

---

## Testing Changes

### Old Test Pattern
```rust
#[test]
fn test_something() {
    let adapter = CaptureAdapter::install();
    // ... test code
    let captured = adapter.captured();
    assert_eq!(captured[0].actor, "worker-orcd");
}
```

### New Test Pattern (Same!)
```rust
#[test]
#[serial(capture_adapter)]  // ← Add this!
fn test_something() {
    let adapter = CaptureAdapter::install();
    // ... test code
    adapter.assert_field("actor", ACTOR_WORKER_ORCD);  // ← Use constants!
}
```

**Only change**: Add `#[serial(capture_adapter)]` attribute for test isolation.

---

## Migration Checklist for Each Story

When implementing FT stories, update narration code to v0.2.0:

- [ ] Replace `narrate(NarrationFields { ... })` with `Narration::new(...).emit()`
- [ ] Replace `narrate_warn(...)` with `.emit_warn()`
- [ ] Replace `narrate_error(...)` with `.emit_error()`
- [ ] Replace `narrate_debug(...)` with `.emit_debug()`
- [ ] Use constants: `ACTOR_*` and `ACTION_*`
- [ ] Remove `.clone()` where possible (use `&` references)
- [ ] Remove `Some()` wrapping (builder handles it)
- [ ] Remove `..Default::default()`
- [ ] For FT-004: Use built-in `correlation_middleware` instead of custom
- [ ] Add `#[serial(capture_adapter)]` to tests

---

## Files Already Updated to v0.2.0

- ✅ FT-001-http-server-setup.md
- ✅ FT-002-execute-endpoint-skeleton.md
- ✅ FT-003-sse-streaming.md (partially)
- ✅ FT-004-correlation-id-middleware.md
- ✅ FT-006-ffi-interface-definition.md
- ✅ FT-010-cuda-context-init.md

## Files Needing Update

- ⏳ FT-005-request-validation.md
- ⏳ FT-007-rust-ffi-bindings.md
- ⏳ FT-008-error-code-system-cpp.md
- ⏳ FT-009-error-code-to-result-rust.md

---

## Questions?

See:
- `bin/shared-crates/narration-core/README.md` - Full documentation
- `bin/shared-crates/narration-core/QUICKSTART.md` - Quick start guide
- `bin/shared-crates/narration-core/tests/smoke_test.rs` - 24 usage examples
- `bin/shared-crates/narration-core/tests/e2e_axum_integration.rs` - 3 E2E examples

---

*Migration guide prepared with love and precision! 🎀*

— The Narration Core Team 💝
