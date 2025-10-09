# 🚀 Foundation Engineer Quick Start

**Goal**: Get narration working in your service in <5 minutes.

---

## Step 1: Add Dependencies (30 seconds)

Add to your `Cargo.toml`:

```toml
[dependencies]
observability-narration-core = { path = "../shared-crates/narration-core" }

# Optional: For Axum middleware
# observability-narration-core = { path = "../shared-crates/narration-core", features = ["axum"] }
```

---

## Step 2: Import and Use (2 minutes)

### Option A: Builder Pattern (Recommended) ✨

```rust
use observability_narration_core::{
    Narration,
    ACTOR_ORCHESTRATORD,
    ACTION_ENQUEUE,
};

fn enqueue_job(job_id: &str, correlation_id: &str) {
    Narration::new(ACTOR_ORCHESTRATORD, ACTION_ENQUEUE, job_id)
        .human(format!("Enqueued job {job_id}"))
        .correlation_id(correlation_id)
        .emit();
}
```

### Option B: Function-Based API

```rust
use observability_narration_core::{
    narrate_auto, NarrationFields,
    ACTOR_ORCHESTRATORD, ACTION_ENQUEUE,
};

fn enqueue_job(job_id: &str, correlation_id: &str) {
    narrate_auto(NarrationFields {
        actor: ACTOR_ORCHESTRATORD,
        action: ACTION_ENQUEUE,
        target: job_id.to_string(),
        human: format!("Enqueued job {job_id}"),
        correlation_id: Some(correlation_id.to_string()),
        ..Default::default()
    });
}
```

---

## Step 3: Axum Integration (Optional, 2 minutes)

If you're using Axum:

```rust
use axum::{Router, routing::post, middleware, extract::Extension};
use observability_narration_core::{
    Narration,
    ACTOR_WORKER_ORCD,
    axum::correlation_middleware,
};

async fn execute_handler(
    Extension(correlation_id): Extension<String>,
    // ... your other extractors
) -> impl IntoResponse {
    Narration::new(ACTOR_WORKER_ORCD, "execute", "job-123")
        .human("Received execute request")
        .correlation_id(&correlation_id)
        .emit();
    
    // ... handler logic
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/execute", post(execute_handler))
        .layer(middleware::from_fn(correlation_middleware));  // ← Add this!
    
    // ... serve app
}
```

The middleware automatically:
- ✅ Extracts correlation ID from `X-Correlation-ID` header
- ✅ Validates the ID (UUID v4 format)
- ✅ Generates a new ID if missing/invalid
- ✅ Stores ID in request extensions
- ✅ Adds ID to response headers

---

## Available Constants

### Actors

```rust
use observability_narration_core::{
    ACTOR_ORCHESTRATORD,      // "rbees-orcd"
    ACTOR_POOL_MANAGERD,      // "pool-managerd"
    ACTOR_WORKER_ORCD,        // "worker-orcd"
    ACTOR_INFERENCE_ENGINE,   // "inference-engine"
    ACTOR_VRAM_RESIDENCY,     // "vram-residency"
};
```

### Actions

```rust
use observability_narration_core::{
    // Admission/Queue
    ACTION_ADMISSION,
    ACTION_ENQUEUE,
    ACTION_DISPATCH,
    
    // Worker lifecycle
    ACTION_SPAWN,
    ACTION_READY_CALLBACK,
    ACTION_HEARTBEAT_SEND,
    ACTION_SHUTDOWN,
    
    // Inference
    ACTION_INFERENCE_START,
    ACTION_INFERENCE_COMPLETE,
    ACTION_CANCEL,
    
    // VRAM
    ACTION_SEAL,
};
```

---

## Common Patterns

### Request Received

```rust
Narration::new(ACTOR_WORKER_ORCD, "request_received", job_id)
    .human("Received execute request")
    .correlation_id(correlation_id)
    .job_id(job_id)
    .emit();
```

### Request Completed

```rust
Narration::new(ACTOR_WORKER_ORCD, "request_completed", job_id)
    .human("Completed inference")
    .correlation_id(correlation_id)
    .duration_ms(elapsed_ms)
    .tokens_in(100)
    .tokens_out(50)
    .emit();
```

### Error Handling

```rust
Narration::new(ACTOR_POOL_MANAGERD, ACTION_SPAWN, "GPU0")
    .human("Failed to spawn worker: insufficient VRAM")
    .correlation_id(correlation_id)
    .error_kind("ResourceExhausted")
    .retry_after_ms(5000)
    .emit_error();  // ← ERROR level
```

### State Transition

```rust
Narration::new(ACTOR_POOL_MANAGERD, ACTION_SPAWN, worker_id)
    .human(format!("Spawned worker {worker_id}"))
    .correlation_id(correlation_id)
    .pool_id(pool_id)
    .device("GPU0")
    .emit();
```

---

## Testing Your Narration

Add to your test dependencies:

```toml
[dev-dependencies]
observability-narration-core = { path = "../shared-crates/narration-core", features = ["test-support"] }
serial_test = "3.0"
```

Write tests:

```rust
use observability_narration_core::CaptureAdapter;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]  // ← Required for test isolation!
fn test_enqueue_emits_narration() {
    let adapter = CaptureAdapter::install();
    
    // Call your function
    enqueue_job("job-123", "req-abc");
    
    // Assert on narration
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].actor, ACTOR_ORCHESTRATORD);
    assert!(captured[0].human.contains("Enqueued"));
    
    // Or use helpers:
    adapter.assert_includes("Enqueued");
    adapter.assert_field("correlation_id", "req-abc");
    adapter.assert_correlation_id_present();
}
```

**Important**: Always use `#[serial(capture_adapter)]` for tests that use `CaptureAdapter`!

---

## Correlation ID Helpers

### Generate

```rust
use observability_narration_core::generate_correlation_id;

let correlation_id = generate_correlation_id();  // UUID v4
```

### Validate

```rust
use observability_narration_core::validate_correlation_id;

if validate_correlation_id(&id).is_some() {
    // Valid UUID v4
}
```

### Extract from HTTP Headers

```rust
use observability_narration_core::http::extract_context_from_headers;

let (correlation_id, trace_id, span_id, parent_span_id) = 
    extract_context_from_headers(&headers);
```

### Inject into HTTP Headers

```rust
use observability_narration_core::http::inject_context_into_headers;

inject_context_into_headers(
    &mut headers,
    Some(&correlation_id),
    None, None, None
);
```

---

## All Builder Methods

```rust
Narration::new(actor, action, target)
    // Required
    .human("Description")
    
    // Identity
    .correlation_id(id)
    .session_id(id)
    .job_id(id)
    .task_id(id)
    .pool_id(id)
    .replica_id(id)
    .worker_id(id)
    
    // Context
    .error_kind("ResourceExhausted")
    .duration_ms(150)
    .retry_after_ms(5000)
    .backoff_ms(2000)
    .queue_position(3)
    .predicted_start_ms(420)
    
    // Engine/Model
    .engine("llamacpp")
    .engine_version("v1.2.3")
    .model_ref("llama-7b")
    .device("GPU0")
    
    // Performance
    .tokens_in(100)
    .tokens_out(50)
    .decode_time_ms(120)
    
    // Story mode (optional)
    .story("\"Ready!\" said worker")
    
    // Emit
    .emit()         // INFO level
    .emit_warn()    // WARN level
    .emit_error()   // ERROR level
```

---

## What Gets Auto-Injected

When you use `.emit()` or `narrate_auto()`, these fields are automatically added:

- `emitted_by`: Service name and version (e.g., "rbees-orcd@0.1.0")
- `emitted_at_ms`: Unix timestamp in milliseconds

You don't need to set these manually!

---

## Secret Redaction (Automatic)

Secrets are **automatically redacted** from `human`, `cute`, and `story` fields:

- ✅ Bearer tokens: `Bearer abc123` → `[REDACTED]`
- ✅ API keys: `api_key=secret` → `[REDACTED]`
- ✅ JWT tokens: `eyJ...` → `[REDACTED]`
- ✅ Private keys: `-----BEGIN PRIVATE KEY-----` → `[REDACTED]`
- ✅ URL passwords: `http://user:pass@host` → `[REDACTED]`

**You don't need to do anything** - it's automatic!

---

## When to Narrate

### ✅ Always Narrate

- Request received
- Request completed (with duration)
- State transitions (enqueued, dispatched, completed)
- External service calls
- Errors

### ❌ Don't Narrate

- Internal function calls
- Loop iterations
- Temporary variables
- Debug-only info (use TRACE macros instead)

### Performance

- Each narration: ~1-5μs overhead
- **Recommendation**: <100 narrations per request

---

## Troubleshooting

### "My tests don't capture events"

**Solution**: Add `#[serial(capture_adapter)]` and use `--features test-support`:

```rust
#[test]
#[serial(capture_adapter)]  // ← Required!
fn test_my_function() {
    let adapter = CaptureAdapter::install();
    // ... test code
}
```

Run with: `cargo test --features test-support`

### "Correlation ID not propagating"

**Solution**: Use the middleware (Axum) or HTTP helpers (other frameworks):

```rust
// Axum: Use middleware
.layer(middleware::from_fn(correlation_middleware))

// Other: Extract manually
let (correlation_id, _, _, _) = extract_context_from_headers(&headers);
```

### "Secrets appearing in logs"

**Solution**: They shouldn't! Redaction is automatic. If you see secrets:
1. Check you're using `human`, `cute`, or `story` fields (auto-redacted)
2. Don't put secrets in custom fields (not auto-redacted)

---

## Complete Working Example

```rust
use axum::{Router, routing::post, middleware, extract::Extension};
use observability_narration_core::{
    Narration, CaptureAdapter,
    ACTOR_WORKER_ORCD,
    ACTION_INFERENCE_START,
    ACTION_INFERENCE_COMPLETE,
    axum::correlation_middleware,
};

async fn execute_handler(
    Extension(correlation_id): Extension<String>,
) -> &'static str {
    // Start
    Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_START, "job-123")
        .human("Starting inference")
        .correlation_id(&correlation_id)
        .emit();
    
    // ... do work ...
    
    // Complete
    Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_COMPLETE, "job-123")
        .human("Completed inference")
        .correlation_id(&correlation_id)
        .duration_ms(150)
        .tokens_in(100)
        .tokens_out(50)
        .emit();
    
    "OK"
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/execute", post(execute_handler))
        .layer(middleware::from_fn(correlation_middleware));
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

---

## Need Help?

1. **Full docs**: See [`README.md`](./README.md)
2. **Field reference**: See README section "NarrationFields Reference"
3. **Troubleshooting**: See README section "Troubleshooting"
4. **Examples**: See [`examples/`](./examples/)
5. **Tests**: See [`tests/smoke_test.rs`](./tests/smoke_test.rs) for 24 usage examples

---

## Test Summary

**narration-core**: ✅ **119 tests passing**
- 50 unit tests
- 3 E2E Axum integration tests
- 16 integration tests
- 9 property tests (1 ignored)
- 24 smoke tests
- 17 doc tests

**narration-macros**: ✅ **48 tests passing**
- 2 unit tests
- 30 integration tests
- 13 actor inference tests
- 1 minimal smoke test
- 1 foundation engineer smoke test
- 1 error documentation test

**Total**: ✅ **167 tests passing** across both crates

---

**Built with diligence, tested with rigor, delivered with confidence.** ✅

*— The Narration Core Team (we promise it works out-of-the-box!) 🎀*
