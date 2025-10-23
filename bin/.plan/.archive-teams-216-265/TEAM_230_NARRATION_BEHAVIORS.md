# TEAM-230: Narration Behavior Inventory

**Date:** Oct 22, 2025  
**Crates:** `observability-narration-core` + `narration-macros`  
**Complexity:** High  
**Status:** ✅ COMPLETE

// TEAM-230: Investigated

---

## Executive Summary

Narration system provides human-readable observability for users via SSE streams, stderr, and logs. **NOT for compliance/audit logging** (separate crate exists for that). Core pattern: job-scoped SSE channels with fail-fast security.

**Key Behaviors:**
- NarrationFactory pattern for ergonomic narration
- Job-scoped SSE routing (CRITICAL for security)
- Format string interpolation with `.context()`
- Fixed-width output format (10-char actor, 15-char action)
- Automatic task-local context propagation

---

## 1. Narration Core (`observability-narration-core`)

### 1.1 Core Architecture

**Pattern:** Factory + Builder + SSE Sink

```rust
// Factory pattern (most ergonomic)
const NARRATE: NarrationFactory = NarrationFactory::new("actor");
NARRATE.action("status").human("Message").emit();

// Builder pattern (flexible)
Narration::new("actor", "action", "target")
    .human("Message")
    .emit();
```

### 1.2 NarrationFactory

**Purpose:** Ergonomic narration with default actor baked in

**Key Methods:**
- `new(actor)` - Compile-time validation (actor ≤10 chars)
- `action(action)` - Runtime validation (action ≤15 chars)
- `with_job_id(job_id)` - Create job-scoped builder

**Compile-Time Validation:**
- Actor string ≤10 characters (enforced at compile time)
- UTF-8 character counting (not bytes)
- Panics at compile time if violated

**Usage Pattern:**
```rust
const NARRATE: NarrationFactory = NarrationFactory::new("qn-router");

// Simple usage
NARRATE.action("status").human("Ready").emit();

// Job-scoped (no need to repeat job_id)
let JOB = NARRATE.with_job_id("job-123");
JOB.action("start").human("Starting").emit();
JOB.action("complete").human("Done").emit();
```

### 1.3 Narration Builder

**Purpose:** Fluent API for constructing narration events

**Key Features:**
- Format string interpolation with `.context()`
- Chainable builder methods
- Automatic task-local context injection
- Multiple emit levels (info, warn, error, fatal)

**Context Interpolation:**
```rust
Narration::new("queen-rbee", "start", "queen")
    .context("http://localhost:8080")  // {0}
    .context("8080")                   // {1}
    .human("✅ Started on {0}, port {1}")
    .emit();
```

**Builder Methods:**
- Identity: `job_id()`, `correlation_id()`, `session_id()`, `hive_id()`, `worker_id()`
- Metrics: `duration_ms()`, `tokens_in()`, `tokens_out()`, `decode_time_ms()`
- Context: `error_kind()`, `retry_after_ms()`, `backoff_ms()`, `queue_position()`
- Engine: `engine()`, `engine_version()`, `model_ref()`, `device()`
- Formatting: `table()` - JSON to CLI table

### 1.4 SSE Sink (CRITICAL SECURITY)

**Pattern:** Job-scoped channels with fail-fast

**Security Model:**
- NO global channel (privacy hazard)
- Job-scoped channels ONLY
- Events without job_id → DROPPED (fail-fast)
- Better to lose narration than leak sensitive data

**Key Functions:**
- `create_job_channel(job_id, capacity)` - Create isolated channel
- `remove_job_channel(job_id)` - Cleanup on completion
- `send(fields)` - Route to job channel (drops if no job_id)
- `take_job_receiver(job_id)` - Get stream (can only call once)

**Integration Pattern:**
```rust
// 1. Create job channel (in job_router::create_job)
sse_sink::create_job_channel(job_id.clone(), 1000);

// 2. Emit narration with job_id
NARRATE.action("status")
    .job_id(&job_id)  // ← CRITICAL for SSE routing
    .human("Message")
    .emit();

// 3. Stream to client (in SSE endpoint)
let mut rx = sse_sink::take_job_receiver(&job_id)?;
while let Some(event) = rx.recv().await {
    println!("{}", event.formatted);
}

// 4. Cleanup
sse_sink::remove_job_channel(&job_id);
```

**Event Format:**
```rust
pub struct NarrationEvent {
    pub formatted: String,  // "[actor     ] action         : message"
    pub actor: String,
    pub action: String,
    pub human: String,
    pub job_id: Option<String>,
    // ... other fields
}
```

### 1.5 Correlation ID

**Purpose:** Request tracking across services

**Functions:**
- `generate_correlation_id()` - UUID v4
- `validate_correlation_id(id)` - Byte-level validation (<100ns)
- `from_header(value)` - Extract from HTTP header
- `propagate(id)` - Pass to downstream service

**Format:** UUID v4 (36 chars, e.g., `550e8400-e29b-41d4-a716-446655440000`)

### 1.6 Output Behavior

**Destinations:**
1. **stderr** - Always (guaranteed visibility)
2. **SSE** - If job_id present and channel exists
3. **tracing** - If subscriber configured

**Format:** Fixed-width for alignment
```
[actor     ] action         : message
 ^^^ 10ch    ^^^ 15ch
```

**No Redaction:** Users need full context for debugging (audit logging is separate)

### 1.7 Actor/Action Taxonomy

**Actors (≤10 chars):**
- `orchestratord`, `pool-managerd`, `worker-orcd`
- `👑 queen-rbee`, `👑 queen-router`
- `inference-engine`, `vram-residency`

**Actions (≤15 chars):**
- Admission: `admission`, `enqueue`, `dispatch`
- Lifecycle: `spawn`, `ready_callback`, `heartbeat_send`, `shutdown`
- Inference: `inference_start`, `inference_complete`, `cancel`
- VRAM: `vram_allocate`, `vram_deallocate`, `seal`, `verify`
- Hive: `hive_install`, `hive_start`, `hive_stop`, `hive_status`, `hive_list`

---

## 2. Narration Macros (`narration-macros`)

### 2.1 Attribute Macros

**`#[trace_fn]` - Function Tracing:**
- Auto-generates entry/exit narration
- Captures function name and arguments
- Measures elapsed time
- Handles Result types automatically

**`#[narrate(...)]` - Template Interpolation:**
- Compile-time template expansion
- Automatic actor inference from module path
- Supports `human`, `cute`, `story` fields

### 2.2 Actor Inference

**Pattern:** Extract service name from module path

```rust
extract_service_name("llama_orch::orchestratord::admission")
// → "orchestratord"
```

**Supported Services:**
- `orchestratord`, `pool_managerd`, `worker_orcd`
- `vram_residency`, `inference_engine`

---

## 3. Integration Points

### 3.1 Used By

**All Binaries:**
- `rbee-keeper` - CLI narration
- `queen-rbee` - Job routing, hive lifecycle
- `rbee-hive` - Worker lifecycle, model management
- `queen-rbee-hive-lifecycle` - Hive operations

**Usage Count:** 82 imports (23 in narration-core, 59 NARRATE macro uses)

### 3.2 Critical Patterns

**Job-Scoped Narration (Server-Side):**
```rust
// MUST include job_id for SSE routing
NARRATE.action("status")
    .job_id(&job_id)  // ← CRITICAL
    .human("Message")
    .emit();
```

**Client-Side Narration (No SSE):**
```rust
// No job_id needed (goes to stderr only)
NARRATE.action("status")
    .human("Message")
    .emit();
```

**TimeoutEnforcer Integration:**
```rust
TimeoutEnforcer::new(timeout)
    .with_job_id(&job_id)  // ← CRITICAL for SSE routing
    .enforce(future).await
```

---

## 4. Test Coverage

### 4.1 Existing Tests

**Unit Tests:**
- Builder pattern tests (basic, with IDs, with metrics)
- Factory pattern tests (basic, chainable, context interpolation)
- SSE formatting tests (formatted field matches stderr)
- Correlation ID tests (generation, validation, header parsing)

**Integration Tests:**
- Job isolation tests (SSE channels don't leak between jobs)
- Race condition tests (narration before channel creation)
- Channel cleanup tests

**BDD Tests:**
- Core narration features
- Field taxonomy validation
- Story mode behavior
- Test capture adapter

### 4.2 Test Gaps

**Missing Tests:**
- ❌ Task-local context propagation (auto-injection)
- ❌ Table formatting edge cases (nested objects, large arrays)
- ❌ Concurrent SSE channel creation/removal
- ❌ Memory leak tests (channel cleanup on job failure)
- ❌ NarrationFactory compile-time validation (actor >10 chars)
- ❌ Action runtime validation (action >15 chars)

---

## 5. Error Handling

**Fail-Fast Patterns:**
- No job_id → event DROPPED (intentional)
- Channel doesn't exist → event DROPPED (intentional)
- Channel full → event DROPPED (try_send fails silently)

**Compile-Time Errors:**
- Actor >10 chars → compile error
- Invalid const context → compile error

**Runtime Panics:**
- Action >15 chars → panic with clear message

---

## 6. Performance Characteristics

**Benchmarks Exist:** `benches/narration_benchmarks.rs`

**Key Metrics:**
- Correlation ID validation: <100ns (byte-level)
- Format string interpolation: O(n) where n = context count
- SSE routing: O(1) HashMap lookup

**Memory:**
- Per-job channel: ~capacity * sizeof(NarrationEvent)
- Default capacity: 1000 events
- Cleanup on job completion prevents leaks

---

## 7. Dependencies

**Core:**
- `tracing` - Structured logging backend
- `serde` + `serde_json` - Serialization
- `tokio` - Async runtime, MPSC channels
- `uuid` - Correlation ID generation
- `once_cell` - Global SSE broadcaster

**Optional:**
- `opentelemetry` - Cloud profile integration
- `axum` - HTTP integration

---

## 8. Critical Behaviors Summary

1. **Job-scoped SSE routing is MANDATORY** - Without job_id, events are dropped
2. **No global channel** - Privacy hazard, all channels are job-scoped
3. **Fixed-width format** - 10-char actor, 15-char action for alignment
4. **Fail-fast security** - Better to lose narration than leak data
5. **No redaction** - Users need full context (audit logging is separate)
6. **Task-local context** - Automatic job_id/correlation_id injection
7. **Format interpolation** - `.context()` + `{0}`, `{1}` in messages

---

**Handoff:** Ready for Phase 5 integration analysis  
**Next:** TEAM-231 (daemon-lifecycle)
