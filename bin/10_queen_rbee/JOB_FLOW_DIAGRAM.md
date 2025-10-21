# Job Flow Visual Diagrams

## Current Flow (Confusing)

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT REQUEST                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  POST /v1/jobs                                                  │
│  { "operation": "hive_list" }                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  http/jobs.rs::handle_create_job()                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ let job_id = registry.create_job()  ← Creates job #1     │  │
│  │ registry.set_payload(&job_id, payload)  ← Stores it      │  │
│  │ return { job_id, sse_url }                                │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Returns to client: { job_id: "job-abc", sse_url: "/v1/..." }  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CLIENT CONNECTS: GET /v1/jobs/job-abc/stream                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  http/jobs.rs::handle_stream_job()                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ execute_and_stream(                                       │  │
│  │   job_id,                                                 │  │
│  │   registry,                                               │  │
│  │   move |_job_id, payload| {  ← CONFUSING CLOSURE         │  │
│  │     let state = JobRouterState { ... };                  │  │
│  │     async move {                                          │  │
│  │       route_job(state, payload)  ← Calls router          │  │
│  │         .map(|_| ())  ← Discards result?!                │  │
│  │     }                                                     │  │
│  │   }                                                       │  │
│  │ )                                                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  job-registry crate::execute_and_stream()                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ let payload = registry.take_payload(&job_id)             │  │
│  │ tokio::spawn(executor(job_id, payload))  ← Runs closure  │  │
│  │ return stream from receiver                               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  job_router.rs::route_job()                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ let job_id = registry.create_job()  ← Creates job #2?!   │  │
│  │ let operation = parse(payload)                            │  │
│  │ match operation {                                         │  │
│  │   HiveList => handle_hive_list_job(...),                 │  │
│  │   ...                                                     │  │
│  │ }                                                         │  │
│  │ return JobResponse { job_id, sse_url }  ← Discarded!     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  operations.rs::handle_hive_list_job()                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ // Actually does the work                                 │  │
│  │ let hives = catalog.list_hives().await?;                  │  │
│  │ // Streams results back                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

PROBLEMS:
❌ Job created TWICE (once in HTTP, once in router)
❌ Confusing closure wrapping
❌ route_job() returns JobResponse but we discard it
❌ Too many layers of indirection
❌ Hard to follow the flow
```

---

## Recommended Flow (Clean & Simple)

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT REQUEST                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  POST /v1/jobs                                                  │
│  { "operation": "hive_list" }                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  http/jobs.rs::handle_create_job()  [HTTP LAYER - THIN]        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ // Just delegate to router                                │  │
│  │ let response = job_router::create_job(                   │  │
│  │     state.into(),                                         │  │
│  │     payload                                               │  │
│  │ ).await?;                                                 │  │
│  │                                                           │  │
│  │ return Json(response)                                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  job_router.rs::create_job()  [BUSINESS LOGIC]                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ let job_id = state.registry.create_job()                 │  │
│  │ state.registry.set_payload(&job_id, payload)             │  │
│  │ return JobResponse { job_id, sse_url }                   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Returns to client: { job_id: "job-abc", sse_url: "/v1/..." }  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CLIENT CONNECTS: GET /v1/jobs/job-abc/stream                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  http/jobs.rs::handle_stream_job()  [HTTP LAYER - THIN]        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ // Just delegate to router                                │  │
│  │ let stream = job_router::execute_job(                    │  │
│  │     job_id,                                               │  │
│  │     state.into()                                          │  │
│  │ ).await;                                                  │  │
│  │                                                           │  │
│  │ // Convert to SSE events                                  │  │
│  │ return Sse::new(stream.map(|d| Event::default().data(d)))│  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  job_router.rs::execute_job()  [BUSINESS LOGIC]                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ // Use helper for execution + streaming                   │  │
│  │ execute_and_stream(                                       │  │
│  │   job_id,                                                 │  │
│  │   registry,                                               │  │
│  │   |_id, payload| route_operation(payload, state)         │  │
│  │ )                                                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  job_router.rs::route_operation()  [INTERNAL]                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ let operation = serde_json::from_value(payload)?;        │  │
│  │                                                           │  │
│  │ match operation {                                         │  │
│  │   Operation::HiveList =>                                 │  │
│  │     operations::hive::list(state.hive_catalog).await,   │  │
│  │   Operation::HiveStart { id } =>                         │  │
│  │     operations::hive::start(id, state.hive_catalog).await│  │
│  │   ...                                                     │  │
│  │ }                                                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  operations/hive.rs::list()  [PURE BUSINESS LOGIC]              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ let hives = catalog.list_hives().await?;                  │  │
│  │ // Stream results back via registry                       │  │
│  │ Ok(())                                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

BENEFITS:
✅ Job created ONCE (in router)
✅ HTTP layer is thin (just delegates)
✅ Clear separation: HTTP → Router → Operations
✅ Easy to test (router has no HTTP dependencies)
✅ Easy to follow the flow
```

---

## Layer Responsibilities

### Current (Confused)

```
┌──────────────────────────────────────────────────────────┐
│ HTTP Layer (http/jobs.rs)                                │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ ❌ Creates jobs                                       │ │
│ │ ❌ Stores payloads                                    │ │
│ │ ❌ Calls execute_and_stream with complex closure     │ │
│ │ ❌ Mixes HTTP and business logic                     │ │
│ └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Router Layer (job_router.rs)                             │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ ❌ ALSO creates jobs (duplicate!)                    │ │
│ │ ✅ Parses operations                                  │ │
│ │ ✅ Dispatches to handlers                            │ │
│ │ ❌ Returns JobResponse that gets discarded           │ │
│ └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Operations Layer (operations.rs)                         │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ ✅ Actual business logic                             │ │
│ └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### Recommended (Clear)

```
┌──────────────────────────────────────────────────────────┐
│ HTTP Layer (http/jobs.rs)  [50 lines]                    │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ ✅ Converts HTTP requests to function calls          │ │
│ │ ✅ Converts responses to HTTP format                 │ │
│ │ ✅ Handles HTTP-specific errors (StatusCode)         │ │
│ │ ✅ THIN - just delegates to router                   │ │
│ └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Router Layer (job_router.rs)  [200 lines]                │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ ✅ Creates jobs (ONCE)                               │ │
│ │ ✅ Stores/retrieves payloads                         │ │
│ │ ✅ Parses operations                                  │ │
│ │ ✅ Dispatches to operation handlers                  │ │
│ │ ✅ Manages job lifecycle                             │ │
│ │ ✅ Clean public API: create_job(), execute_job()    │ │
│ └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Operations Layer (operations/*)  [100 lines each]        │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ ✅ Pure business logic                               │ │
│ │ ✅ No HTTP dependencies                              │ │
│ │ ✅ Easy to test                                      │ │
│ │ ✅ Reusable (can call from CLI, tests, etc.)        │ │
│ └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## Code Size Comparison

### Current

```
http/jobs.rs:           117 lines  (❌ Too much logic)
job_router.rs:          410 lines  (❌ Duplicate job creation)
operations.rs:          402 lines  (✅ OK)
────────────────────────────────
Total:                  929 lines
```

### Recommended

```
http/jobs.rs:            50 lines  (✅ Thin HTTP wrapper)
job_router.rs:          200 lines  (✅ Clean API)
operations/hive.rs:     100 lines  (✅ Focused)
operations/worker.rs:   100 lines  (✅ Focused)
operations/model.rs:    100 lines  (✅ Focused)
operations/infer.rs:    100 lines  (✅ Focused)
────────────────────────────────
Total:                  650 lines  (✅ 30% reduction!)
```

---

## Testing Comparison

### Current (Hard to Test)

```rust
// Can't test router without HTTP mocking
#[tokio::test]
async fn test_route_job() {
    // Need to mock HTTP state
    let state = SchedulerState { ... };
    
    // Need to create job first via HTTP?
    // Or duplicate the logic?
    
    // Hard to test in isolation
}
```

### Recommended (Easy to Test)

```rust
// Test router without HTTP
#[tokio::test]
async fn test_create_job() {
    let state = JobState { ... };
    let payload = json!({ "operation": "hive_list" });
    
    let response = job_router::create_job(state, payload).await.unwrap();
    
    assert!(response.job_id.starts_with("job-"));
}

// Test operations without router
#[tokio::test]
async fn test_list_hives() {
    let catalog = setup_test_catalog();
    
    let hives = operations::hive::list(catalog).await.unwrap();
    
    assert_eq!(hives.len(), 2);
}
```

---

## Summary

**Current Flow**: Confusing, duplicated, hard to test
**Recommended Flow**: Clean, simple, easy to test

**Key Changes**:
1. HTTP layer becomes thin (just delegates)
2. Router has clean API (`create_job`, `execute_job`)
3. Operations are pure business logic
4. Single state struct (no duplication)
5. Job created ONCE (in router)
6. Clear separation of concerns
