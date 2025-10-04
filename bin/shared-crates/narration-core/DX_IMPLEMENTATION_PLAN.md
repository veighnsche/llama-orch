# narration-core DX Improvements — Implementation Plan

**Created**: 2025-10-04  
**Owner**: Developer Experience Team  
**Target**: narration-core v0.2.0  
**Timeline**: 3 weeks

---

## Overview

Transform narration-core from **B+ (good with friction)** to **A- (developer-friendly)**.

**Current**: Verbose API, broken macro, no Axum support, 175 lines duplication  
**Target**: Builder pattern, working middleware, clear examples, maintainable code

---

## Unit 1: Merge narration-macros into narration-core

**Priority**: P0  
**Effort**: 3 hours  
**Files**: Entire `narration-macros/` directory, `narration-core/Cargo.toml`, `narration-core/src/lib.rs`  
**Blocks**: Developer confusion, crate fragmentation

### Problem

**Two separate crates cause confusion**:
- `narration-core` (v0.1.0, production-ready)
- `narration-macros` (v0.0.0, stubs only)

**Issues**:
1. Developers don't know which to import
2. Macros don't work (stubs that return original code)
3. Version mismatch (0.1.0 vs 0.0.0)
4. Premature optimization (YAGNI - macros not needed yet)
5. README advertises broken macros

### Solution

**Delete narration-macros entirely** (can't merge proc-macro into regular lib):

**Why deletion is correct**:
1. Proc-macro crates MUST be separate (Rust limitation)
2. Macros are v0.0.0 stubs that don't work
3. No users depend on them
4. README advertises broken functionality
5. YAGNI - not needed yet

**Step 1**: Delete narration-macros
```bash
rm -rf bin/shared-crates/narration-macros/
```

**Step 2**: Remove from workspace
```toml
# Cargo.toml - Remove line:
# "bin/shared-crates/narration-macros",
```

**Step 3**: Update narration-core README
```markdown
## Usage

### Basic Narration (Recommended)

\`\`\`rust
use observability_narration_core::{narrate_auto, NarrationFields};

narrate_auto(NarrationFields {
    actor: "worker-orcd",
    action: "execute",
    target: job_id.to_string(),
    human: format!("Executing job {job_id}"),
    correlation_id: Some(req_id),
    ..Default::default()
});
\`\`\`

**Note**: Procedural macros (`#[narrate]`, `#[trace_fn]`) are planned for v0.3.0.
Use function-based API for now.
```

**Step 4**: Add note about future macros
```markdown
## Roadmap

### v0.2.0 (Current)
- Builder pattern for ergonomics
- Axum middleware
- Policy guide

### v0.3.0 (Future)
- Procedural macros (`#[narrate]`, `#[trace_fn]`)
- Compile-time template expansion
- Actor inference from module path
```

### Acceptance

- [x] `narration-macros/` directory deleted
- [x] Removed from workspace `Cargo.toml`
- [ ] README updated (remove macro examples, add roadmap note)
- [x] `narration-core` compiles without errors
- [ ] Tests pass
- [ ] Single crate to import

---

## Unit 2: Add Axum Middleware

**Priority**: P0  
**Effort**: 3 hours  
**Files**: `narration-core/src/axum.rs` (NEW), `Cargo.toml`, `lib.rs`  
**Blocks**: FT-004, all Axum integrations

### Problem

Every team reimplements correlation ID middleware. No built-in support.

### Solution

**File**: `src/axum.rs`
```rust
#[cfg(feature = "axum")]
pub mod axum {
    use axum::{
        extract::Request,
        middleware::Next,
        response::Response,
        http::HeaderValue,
    };
    use crate::{generate_correlation_id, validate_correlation_id};
    
    /// Axum middleware for correlation ID extraction/generation.
    pub async fn correlation_middleware(mut req: Request, next: Next) -> Response {
        let correlation_id = req.headers()
            .get("X-Correlation-ID")
            .and_then(|v| v.to_str().ok())
            .and_then(|id| validate_correlation_id(id).map(|_| id.to_string()))
            .unwrap_or_else(|| generate_correlation_id());
        
        req.extensions_mut().insert(correlation_id.clone());
        
        let mut response = next.run(req).await;
        
        if let Ok(header_value) = HeaderValue::from_str(&correlation_id) {
            response.headers_mut().insert("X-Correlation-ID", header_value);
        }
        
        response
    }
}
```

**File**: `Cargo.toml`
```toml
[features]
axum = ["dep:axum"]

[dependencies]
axum = { version = "0.7", optional = true }
```

**File**: `lib.rs`
```rust
#[cfg(feature = "axum")]
pub mod axum;
```

### Acceptance

- [ ] Middleware compiles with `--features axum`
- [ ] Middleware extracts correlation ID from header
- [ ] Middleware generates ID if missing
- [ ] Middleware validates ID format
- [ ] Middleware stores ID in extensions
- [ ] Middleware adds ID to response headers
- [ ] Integration test with Axum router
- [ ] README example added

---

## Unit 3: Fix README `HeaderLike` Example

**Priority**: P0  
**Effort**: 15 minutes  
**Files**: `narration-core/README.md`  
**Blocks**: Developer copy-paste errors

### Problem

README shows wrong method name:
```rust
impl HeaderLike for MyRequest {
    fn get_header(&self, name: &str) -> Option<&str> {  // ← Wrong!
        self.headers.get(name).map(|v| v.as_str())
    }
}
```

Trait requires `get_str` and `insert_str`.

### Solution

Update README example:
```rust
impl HeaderLike for MyRequest {
    fn get_str(&self, name: &str) -> Option<String> {
        self.headers.get(name)?.to_str().ok().map(String::from)
    }
    
    fn insert_str(&mut self, name: &str, value: &str) {
        if let Ok(header_value) = HeaderValue::from_str(value) {
            self.headers.insert(name, header_value);
        }
    }
}
```

### Acceptance

- [ ] Example matches trait definition
- [ ] Both methods implemented
- [ ] Return types correct
- [ ] Example compiles

---

## Unit 4: Add Builder Pattern

**Priority**: P1  
**Effort**: 4 hours  
**Files**: `narration-core/src/builder.rs` (NEW), `lib.rs`  
**Blocks**: API ergonomics

### Problem

Current API requires 7 lines + boilerplate:
```rust
narrate_auto(NarrationFields {
    actor: "orchestratord",
    action: "enqueue",
    target: job_id.to_string(),
    human: format!("Enqueued job {job_id}"),
    correlation_id: Some(req_id),
    ..Default::default()
});
```

### Solution

**File**: `src/builder.rs`
```rust
pub struct Narration {
    fields: NarrationFields,
}

impl Narration {
    pub fn new(actor: &'static str, action: &'static str, target: impl Into<String>) -> Self {
        Self {
            fields: NarrationFields {
                actor,
                action,
                target: target.into(),
                human: String::new(),
                ..Default::default()
            }
        }
    }
    
    pub fn human(mut self, msg: impl Into<String>) -> Self {
        self.fields.human = msg.into();
        self
    }
    
    pub fn correlation_id(mut self, id: impl Into<String>) -> Self {
        self.fields.correlation_id = Some(id.into());
        self
    }
    
    pub fn job_id(mut self, id: impl Into<String>) -> Self {
        self.fields.job_id = Some(id.into());
        self
    }
    
    pub fn pool_id(mut self, id: impl Into<String>) -> Self {
        self.fields.pool_id = Some(id.into());
        self
    }
    
    pub fn worker_id(mut self, id: impl Into<String>) -> Self {
        self.fields.worker_id = Some(id.into());
        self
    }
    
    pub fn cute(mut self, msg: impl Into<String>) -> Self {
        self.fields.cute = Some(msg.into());
        self
    }
    
    pub fn duration_ms(mut self, ms: u64) -> Self {
        self.fields.duration_ms = Some(ms);
        self
    }
    
    pub fn error_kind(mut self, kind: impl Into<String>) -> Self {
        self.fields.error_kind = Some(kind.into());
        self
    }
    
    pub fn emit(self) {
        crate::narrate_auto(self.fields)
    }
    
    pub fn emit_warn(self) {
        crate::narrate_warn(self.fields)
    }
    
    pub fn emit_error(self) {
        crate::narrate_error(self.fields)
    }
}
```

**File**: `lib.rs`
```rust
mod builder;
pub use builder::Narration;
```

**Usage**:
```rust
Narration::new("orchestratord", "enqueue", job_id)
    .human(format!("Enqueued job {job_id}"))
    .correlation_id(req_id)
    .emit();
```

### Acceptance

- [ ] Builder compiles
- [ ] All common fields have methods
- [ ] `emit()`, `emit_warn()`, `emit_error()` work
- [ ] Unit tests for builder
- [ ] README example added
- [ ] 4 lines vs. 7 lines (43% reduction)

---

## Unit 5: Add Policy Guide to README

**Priority**: P1  
**Effort**: 2 hours  
**Files**: `narration-core/README.md`  
**Blocks**: Log spam, unclear usage

### Problem

No guidance on when to narrate. Developers will either:
- Over-narrate (log spam)
- Under-narrate (missing context)

### Solution

Add section to README:
```markdown
## When to Narrate

### Always Narrate (INFO level)

**Request lifecycle**:
- Request received (with correlation ID)
- Request completed (with duration)

**State transitions**:
- Job enqueued/dispatched/completed
- Worker spawned/ready/shutdown
- Model loaded/unloaded

**External calls**:
- HTTP requests to other services
- Database operations
- File I/O

### Narrate on Error (ERROR level)

- Validation failures
- External service errors
- Timeout/cancellation
- Resource exhaustion

### Don't Narrate

- Internal function calls
- Loop iterations
- Temporary variables
- Debug-only info (use TRACE instead)

### Performance Impact

- Each narration: ~1-5μs overhead
- Redaction: ~180ms for 200-char strings
- **Recommendation**: <100 narrations per request

### Example: Good Narration

\`\`\`rust
// ✅ Request received
Narration::new("worker-orcd", "request_received", job_id)
    .human("Received execute request")
    .correlation_id(req_id)
    .emit();

// ✅ State transition
Narration::new("worker-orcd", "inference_start", job_id)
    .human("Starting inference")
    .correlation_id(req_id)
    .emit();

// ✅ Request completed
Narration::new("worker-orcd", "request_completed", job_id)
    .human("Completed inference")
    .correlation_id(req_id)
    .duration_ms(elapsed_ms)
    .emit();
\`\`\`

### Example: Bad Narration

\`\`\`rust
// ❌ Internal function call
fn parse_token(token: &str) -> Result<Token> {
    Narration::new("worker-orcd", "parse_token", token)
        .human("Parsing token")
        .emit();  // ← Don't narrate internal functions
    // ...
}

// ❌ Loop iteration
for item in items {
    Narration::new("worker-orcd", "process_item", item.id)
        .human("Processing item")
        .emit();  // ← Don't narrate loops
}
\`\`\`
```

### Acceptance

- [ ] Policy guide added to README
- [ ] Examples show good vs. bad narration
- [ ] Performance impact documented
- [ ] Reviewed by 2+ teams

---

## Unit 6: Fix Duplicate Logic in `auto.rs`

**Priority**: P1  
**Effort**: 30 minutes  
**Files**: `narration-core/src/auto.rs`  
**Blocks**: Code quality

### Problem

`narrate_auto` calls `inject_provenance` then duplicates the same checks:
```rust
pub fn narrate_auto(mut fields: NarrationFields) {
    inject_provenance(&mut fields);  // ← Does the checks
    
    // Duplicate checks!
    if fields.emitted_by.is_none() {
        fields.emitted_by = Some(service_identity());
    }
    if fields.emitted_at_ms.is_none() {
        fields.emitted_at_ms = Some(current_timestamp_ms());
    }
    crate::narrate(fields);
}
```

### Solution

Remove duplicate checks:
```rust
pub fn narrate_auto(mut fields: NarrationFields) {
    inject_provenance(&mut fields);
    crate::narrate(fields);
}

pub fn narrate_full(mut fields: NarrationFields) {
    inject_provenance(&mut fields);
    
    let (trace_id, span_id, parent_span_id) = crate::otel::extract_otel_context();
    if fields.trace_id.is_none() {
        fields.trace_id = trace_id;
    }
    if fields.span_id.is_none() {
        fields.span_id = span_id;
    }
    if fields.parent_span_id.is_none() {
        fields.parent_span_id = parent_span_id;
    }
    
    crate::narrate(fields);
}
```

### Acceptance

- [ ] Duplicate logic removed
- [ ] Tests still pass
- [ ] Behavior unchanged
- [ ] Clippy clean

---

## Unit 7: Extract Event Emission Macro

**Priority**: P2  
**Effort**: 2 hours  
**Files**: `narration-core/src/lib.rs`  
**Blocks**: Maintainability

### Problem

175 lines of duplicated field lists (5 log levels × 35 fields):
```rust
Level::TRACE => event!(Level::TRACE, actor = fields.actor, action = fields.action, ...),
Level::DEBUG => event!(Level::DEBUG, actor = fields.actor, action = fields.action, ...),
Level::INFO => event!(Level::INFO, actor = fields.actor, action = fields.action, ...),
// ... 35 fields repeated 5 times
```

### Solution

Extract into internal macro:
```rust
macro_rules! emit_event {
    ($level:expr, $fields:expr, $human:expr, $cute:expr, $story:expr) => {
        event!(
            $level,
            actor = $fields.actor,
            action = $fields.action,
            target = %$fields.target,
            human = %$human,
            cute = $cute.as_deref(),
            story = $story.as_deref(),
            correlation_id = $fields.correlation_id.as_deref(),
            session_id = $fields.session_id.as_deref(),
            job_id = $fields.job_id.as_deref(),
            task_id = $fields.task_id.as_deref(),
            pool_id = $fields.pool_id.as_deref(),
            replica_id = $fields.replica_id.as_deref(),
            worker_id = $fields.worker_id.as_deref(),
            error_kind = $fields.error_kind.as_deref(),
            retry_after_ms = $fields.retry_after_ms,
            backoff_ms = $fields.backoff_ms,
            duration_ms = $fields.duration_ms,
            queue_position = $fields.queue_position,
            predicted_start_ms = $fields.predicted_start_ms,
            engine = $fields.engine.as_deref(),
            engine_version = $fields.engine_version.as_deref(),
            model_ref = $fields.model_ref.as_deref(),
            device = $fields.device.as_deref(),
            tokens_in = $fields.tokens_in,
            tokens_out = $fields.tokens_out,
            decode_time_ms = $fields.decode_time_ms,
            emitted_by = $fields.emitted_by.as_deref(),
            emitted_at_ms = $fields.emitted_at_ms,
            trace_id = $fields.trace_id.as_deref(),
            span_id = $fields.span_id.as_deref(),
            parent_span_id = $fields.parent_span_id.as_deref(),
            source_location = $fields.source_location.as_deref(),
        )
    };
}

pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    let Some(tracing_level) = level.to_tracing_level() else { return; };
    
    let human = redact_secrets(&fields.human, RedactionPolicy::default());
    let cute = fields.cute.as_ref().map(|c| redact_secrets(c, RedactionPolicy::default()));
    let story = fields.story.as_ref().map(|s| redact_secrets(s, RedactionPolicy::default()));
    
    match tracing_level {
        Level::TRACE => emit_event!(Level::TRACE, fields, human, cute, story),
        Level::DEBUG => emit_event!(Level::DEBUG, fields, human, cute, story),
        Level::INFO => emit_event!(Level::INFO, fields, human, cute, story),
        Level::WARN => emit_event!(Level::WARN, fields, human, cute, story),
        Level::ERROR => emit_event!(Level::ERROR, fields, human, cute, story),
    }
    
    #[cfg(any(test, feature = "test-support"))]
    {
        let mut redacted_fields = fields;
        redacted_fields.human = human.to_string();
        redacted_fields.cute = cute.map(|c| c.to_string());
        redacted_fields.story = story.map(|s| s.to_string());
        capture::notify(redacted_fields);
    }
}
```

**Before**: 400 lines  
**After**: 90 lines (78% reduction)

### Acceptance

- [ ] All tests pass
- [ ] No behavior change
- [ ] Clippy clean
- [ ] Lines reduced by >70%

---

## Unit 8: Add Complete Axum Example to README

**Priority**: P0  
**Effort**: 1 hour  
**Files**: `narration-core/README.md`  
**Blocks**: FT-004 integration

### Problem

README shows trait implementation but no complete middleware example.

### Solution

Add section after "HTTP Context Propagation":
```markdown
### Complete Axum Integration Example

\`\`\`rust
use axum::{
    Router,
    routing::post,
    middleware,
    extract::{Extension, Json},
    response::IntoResponse,
    http::StatusCode,
};
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD, ACTION_EXECUTE};

// Enable Axum middleware (requires feature = "axum")
#[cfg(feature = "axum")]
use observability_narration_core::axum::correlation_middleware;

async fn execute_handler(
    Extension(correlation_id): Extension<String>,
    Json(payload): Json<ExecuteRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    // Narrate request received
    Narration::new(ACTOR_WORKER_ORCD, ACTION_EXECUTE, &payload.job_id)
        .human("Received execute request")
        .correlation_id(&correlation_id)
        .emit();
    
    // ... handler logic
    
    Ok(StatusCode::OK)
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/execute", post(execute_handler))
        .layer(middleware::from_fn(correlation_middleware));
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
\`\`\`

**Cargo.toml**:
\`\`\`toml
[dependencies]
observability-narration-core = { path = "../narration-core", features = ["axum"] }
axum = "0.7"
tokio = { version = "1", features = ["full"] }
\`\`\`
```

### Acceptance

- [ ] Complete working example
- [ ] Shows middleware setup
- [ ] Shows handler usage
- [ ] Shows Cargo.toml config
- [ ] Example compiles

---

## Unit 9: Use Constants in README Examples

**Priority**: P2  
**Effort**: 30 minutes  
**Files**: `narration-core/README.md`  
**Blocks**: Type safety

### Problem

Constants exported but not used in examples:
```rust
// Exported
pub const ACTOR_ORCHESTRATORD: &str = "orchestratord";

// But examples use literals
actor: "orchestratord",  // ← Should use constant
```

### Solution

Update all README examples:
```rust
use observability_narration_core::{
    Narration,
    ACTOR_ORCHESTRATORD,
    ACTION_ENQUEUE,
};

Narration::new(ACTOR_ORCHESTRATORD, ACTION_ENQUEUE, job_id)
    .human("Enqueued job")
    .emit();
```

### Acceptance

- [ ] All examples use constants
- [ ] Import statements updated
- [ ] Examples compile
- [ ] Consistent pattern throughout

---

## Unit 10: Add Field Reference Table

**Priority**: P1  
**Effort**: 1 hour  
**Files**: `narration-core/README.md`  
**Blocks**: API discoverability

### Problem

`NarrationFields` has 30+ fields. No reference documentation.

### Solution

Add table after "Key Concepts":
```markdown
## NarrationFields Reference

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `actor` | `&'static str` | Service name | `"worker-orcd"` |
| `action` | `&'static str` | Action performed | `"execute"` |
| `target` | `String` | Target of action | `job_id` |
| `human` | `String` | Human description | `"Executing job"` |

### Optional Identity Fields

| Field | Type | Description |
|-------|------|-------------|
| `correlation_id` | `Option<String>` | Request tracking ID |
| `session_id` | `Option<String>` | Session identifier |
| `job_id` | `Option<String>` | Job identifier |
| `task_id` | `Option<String>` | Task identifier |
| `pool_id` | `Option<String>` | Pool identifier |
| `replica_id` | `Option<String>` | Replica identifier |
| `worker_id` | `Option<String>` | Worker identifier |

### Optional Context Fields

| Field | Type | Description |
|-------|------|-------------|
| `error_kind` | `Option<String>` | Error category |
| `duration_ms` | `Option<u64>` | Operation duration |
| `retry_after_ms` | `Option<u64>` | Retry delay |
| `backoff_ms` | `Option<u64>` | Backoff duration |
| `queue_position` | `Option<usize>` | Position in queue |

### Auto-Redacted Fields

| Field | Type | Description |
|-------|------|-------------|
| `authorization` | `Option<String>` | Auth header (auto-redacted) |
| `api_key` | `Option<String>` | API key (auto-redacted) |
| `token` | `Option<String>` | Token (auto-redacted) |
| `password` | `Option<String>` | Password (auto-redacted) |
| `secret` | `Option<String>` | Secret (auto-redacted) |

### Provenance Fields (Auto-Injected)

| Field | Type | Description |
|-------|------|-------------|
| `emitted_by` | `Option<String>` | Service@version (auto) |
| `emitted_at_ms` | `Option<u64>` | Unix timestamp (auto) |
| `trace_id` | `Option<String>` | OpenTelemetry trace ID |
| `span_id` | `Option<String>` | OpenTelemetry span ID |
```

### Acceptance

- [ ] All fields documented
- [ ] Types shown
- [ ] Auto-redacted fields marked
- [ ] Auto-injected fields marked
- [ ] Examples provided

---

## Unit 11: Add Troubleshooting Section

**Priority**: P1  
**Effort**: 1 hour  
**Files**: `narration-core/README.md`  
**Blocks**: Developer debugging

### Problem

No troubleshooting guide. Common issues not documented.

### Solution

Add section before "Contributing":
```markdown
## Troubleshooting

### Events Not Captured in Tests

**Problem**: `adapter.captured()` returns empty vec.

**Causes**:
1. Missing `#[serial(capture_adapter)]` attribute
2. `CaptureAdapter::install()` not called
3. Wrong test feature flag

**Solution**:
\`\`\`rust
use serial_test::serial;

#[test]
#[serial(capture_adapter)]  // ← Required!
fn test_narration() {
    let adapter = CaptureAdapter::install();  // ← Required!
    // ... test code
}
\`\`\`

**Run with**: `cargo test --features test-support`

---

### Correlation ID Not Propagating

**Problem**: Downstream services don't see correlation ID.

**Causes**:
1. Not extracted from request headers
2. Not injected into outgoing headers
3. Validation failed (invalid format)

**Solution**:
\`\`\`rust
// Extract from incoming request
let correlation_id = extract_context_from_headers(&req.headers()).0
    .unwrap_or_else(|| generate_correlation_id());

// Inject into outgoing request
inject_context_into_headers(
    &mut outgoing_headers,
    Some(&correlation_id),
    None, None, None
);
\`\`\`

---

### Secrets Appearing in Logs

**Problem**: Sensitive data not redacted.

**Causes**:
1. Wrong field name (must be exact: `authorization`, `api_key`, etc.)
2. Secret in `human` text (auto-redacted)
3. Secret in custom field (not auto-redacted)

**Solution**:
\`\`\`rust
// ✅ Auto-redacted (exact field name)
Narration::new("worker-orcd", "auth", "api")
    .authorization(format!("Bearer {token}"))  // ← Redacted
    .emit();

// ✅ Redacted in human text
Narration::new("worker-orcd", "auth", "api")
    .human(format!("Auth with token {token}"))  // ← Redacted
    .emit();

// ❌ NOT redacted (custom field)
Narration::new("worker-orcd", "auth", "api")
    .custom_field(format!("Bearer {token}"))  // ← NOT redacted!
    .emit();
\`\`\`

---

### Middleware Not Working

**Problem**: Correlation ID middleware not extracting ID.

**Causes**:
1. Middleware not added to router
2. Wrong header name (case-sensitive)
3. `HeaderLike` trait not implemented

**Solution**:
\`\`\`rust
// Use built-in Axum middleware (requires feature = "axum")
use observability_narration_core::axum::correlation_middleware;

let app = Router::new()
    .route("/execute", post(handler))
    .layer(middleware::from_fn(correlation_middleware));  // ← Add middleware
\`\`\`
```

### Acceptance

- [ ] 4+ common issues documented
- [ ] Causes listed
- [ ] Solutions with code examples
- [ ] Tested by developers

---

## Unit 12: Improve `narrate_auto!` Macro

**Priority**: P2  
**Effort**: 2 hours  
**Files**: `narration-core/src/auto.rs`  
**Blocks**: Macro ergonomics

### Problem

Macro doesn't eliminate boilerplate, just moves it:
```rust
narrate_auto! {
    actor: "pool-managerd",
    action: "spawn",
    target: "GPU0",
    human: "Spawning",
};
// Still requires all 4 fields, no enforcement
```

### Solution

Enforce required fields at compile time:
```rust
#[macro_export]
macro_rules! narrate_auto {
    (
        actor: $actor:expr,
        action: $action:expr,
        target: $target:expr,
        human: $human:expr
        $(, $field:ident: $value:expr)* $(,)?
    ) => {
        $crate::auto::narrate_auto($crate::NarrationFields {
            actor: $actor,
            action: $action,
            target: $target.into(),
            human: $human.into(),
            $($field: $value,)*
            ..Default::default()
        })
    };
}
```

**Benefit**: Required fields enforced, optional fields truly optional.

### Acceptance

- [ ] Required fields enforced at compile time
- [ ] Optional fields work
- [ ] Compile error if required field missing
- [ ] Tests updated
- [ ] README example updated

---

## Timeline

### Week 1: P0 (Unblock Adoption)

**Day 1-2**:
- Unit 1: Merge narration-macros into core (3h)
- Unit 2: Add Axum middleware (3h)

**Day 3-4**:
- Unit 3: Fix README example (15m)
- Unit 8: Add Axum example to README (1h)
- Unit 5: Add policy guide (2h)

**Day 5**:
- Unit 10: Add field reference (1h)
- Unit 11: Add troubleshooting (1h)

**Deliverable**: narration-core v0.2.0 (minor release - crate merge + Axum support)

### Week 2: P1 (Improve Ergonomics)

**Day 1-3**:
- Unit 4: Add builder pattern (4h)
- Unit 6: Fix duplicate logic (30m)
- Unit 9: Use constants in examples (30m)

**Day 4-5**:
- Integration testing
- README polish
- Review with 2+ teams

**Deliverable**: narration-core v0.2.1 (patch release with builder)

### Week 3: P2 (Code Quality)

**Day 1-2**:
- Unit 7: Extract event macro (2h)
- Unit 12: Improve `narrate_auto!` (2h)

**Day 3-5**:
- Performance testing
- Documentation review
- Migration guide

**Deliverable**: narration-core v0.2.2 (patch release with quality improvements)

---

## Success Metrics

**Before**:
- Crates: 2 (narration-core + narration-macros, confusing)
- API patterns: 3 (confusing)
- Lines per narration: 7 (verbose)
- Axum support: None (friction)
- Code duplication: 175 lines (unmaintainable)
- DX Score: 6.3/10 (B+)

**After**:
- Crates: 1 (narration-core with optional features, clear)
- API patterns: 1 preferred (builder pattern)
- Lines per narration: 4 (concise)
- Axum support: Built-in (smooth)
- Code duplication: 0 lines (maintainable)
- DX Score: 8.5/10 (A-)

---

## Dependencies

**Unit 1** → Blocks nothing (independent)  
**Unit 2** → Blocks Unit 8 (Axum example needs middleware)  
**Unit 3** → Blocks nothing (independent)  
**Unit 4** → Blocks Unit 9 (builder used in examples)  
**Unit 5** → Blocks nothing (independent)  
**Unit 6** → Blocks nothing (independent)  
**Unit 7** → Blocks nothing (independent)  
**Unit 8** → Requires Unit 2 (middleware must exist)  
**Unit 9** → Requires Unit 4 (builder in examples)  
**Unit 10** → Blocks nothing (independent)  
**Unit 11** → Blocks nothing (independent)  
**Unit 12** → Blocks nothing (independent)

**Critical Path**: Unit 2 → Unit 8 (Axum support)

---

## Review Checklist

After each unit:
- [ ] Tests pass (`cargo test --features test-support`)
- [ ] Clippy clean (`cargo clippy -- -D warnings`)
- [ ] Rustfmt applied (`cargo fmt`)
- [ ] README updated if API changed
- [ ] Examples compile
- [ ] Reviewed by 1+ developer

---

## Rollout Plan

**v0.2.0** (Week 1) - **BREAKING**: Crate merge:
- Merge narration-macros into narration-core
- Add Axum middleware
- Fix README examples
- Add policy guide
- **Breaking**: Remove `observability-narration-macros` dependency

**v0.2.1** (Week 2):
- Add builder pattern
- Update all examples
- Migration guide

**v0.2.2** (Week 3):
- Extract duplication
- Code quality improvements

---
Crafted with love by Developer Experience Team 🎨
