# TEAM-191: Operations Split & Shared Job Architecture Analysis

**Date:** 2025-10-21  
**Status:** 🔍 Analysis

## Problem Statement

Currently, all operations are in a single `rbee-operations` enum, but:
1. **Queen operations**: Handled by queen-rbee (hive management, status)
2. **Hive operations**: Should be forwarded to rbee-hive (workers, models, inference)
3. **Shared architecture**: Both use job-based architecture (POST /v1/jobs, SSE streaming)

## Current Operation Classification

### Queen-Only Operations (Handled by queen-rbee)
These operations are executed by queen-rbee and NOT forwarded to hives:

```rust
// System-wide
Status                    // Query hive-registry for all hives/workers

// Hive lifecycle management
SshTest                   // Test SSH connection before install
HiveInstall               // Install hive binary (local or remote)
HiveUninstall             // Uninstall hive
HiveUpdate                // Update hive configuration
HiveStart                 // Start hive daemon
HiveStop                  // Stop hive daemon
HiveList                  // List all hives from catalog
HiveGet                   // Get single hive details
HiveStatus                // Check hive health endpoint
```

**Total: 9 operations**

### Hive Operations (Forwarded to rbee-hive)
These operations are forwarded from queen-rbee to rbee-hive:

```rust
// Worker management
WorkerSpawn               // Spawn new worker process
WorkerList                // List all workers on hive
WorkerGet                 // Get single worker details
WorkerDelete              // Delete/stop worker

// Model management
ModelDownload             // Download model to hive
ModelList                 // List models on hive
ModelGet                  // Get model details
ModelDelete               // Delete model from hive

// Inference
Infer                     // Run inference on worker
```

**Total: 9 operations**

## Proposed Split

### Option 1: Two Separate Enums (Recommended)

```
bin/99_shared_crates/
├── rbee-operations/          # Queen operations (rbee-keeper → queen-rbee)
│   └── src/lib.rs
└── hive-operations/          # Hive operations (queen-rbee → rbee-hive)
    └── src/lib.rs
```

**Benefits:**
- Clear separation of concerns
- Queen can import both (extends hive operations)
- Type safety at compile time
- Each binary only imports what it needs

### Option 2: Single Enum with Trait (Alternative)

```rust
pub enum Operation {
    // Queen operations
    Queen(QueenOperation),
    // Hive operations (forwarded)
    Hive(HiveOperation),
}
```

**Benefits:**
- Single source of truth
- Easier to see all operations

**Drawbacks:**
- More complex pattern matching
- Harder to enforce separation

## Shared Job Architecture Components

Both queen-rbee and rbee-hive use the same job-based architecture:

### Shared Components (Candidates for new crate)

#### 1. Job Registry
**Location:** `bin/99_shared_crates/job-registry/`
**Status:** ✅ Already shared
**Used by:** queen-rbee, (will be used by rbee-hive)

```rust
pub struct JobRegistry<T> {
    jobs: RwLock<HashMap<String, T>>,
}

impl JobRegistry<T> {
    pub fn create_job(&self, payload: T) -> String;
    pub fn get_job(&self, job_id: &str) -> Option<T>;
    pub fn execute_and_stream(...) -> impl Stream;
}
```

#### 2. HTTP Job Endpoints Pattern
**Current:** Duplicated in queen-rbee
**Should be:** Shared crate

```rust
// POST /v1/jobs - Create job
pub async fn handle_create_job(
    State(state): State<SchedulerState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<JobResponse>, (StatusCode, String)>

// GET /v1/jobs/{job_id}/stream - Stream results
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<SchedulerState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
```

**Pattern:**
1. Client POSTs operation to `/v1/jobs`
2. Server creates job, returns `{job_id, sse_url}`
3. Client GETs `/v1/jobs/{job_id}/stream` for SSE results
4. Server executes job, streams narration events
5. Server sends `[DONE]` when complete

#### 3. Job Router Pattern
**Current:** Implemented in queen-rbee
**Should be:** Shared pattern/trait

```rust
pub async fn create_job(
    state: JobState,
    payload: serde_json::Value,
) -> Result<JobResponse>

pub async fn execute_job(
    job_id: String,
    state: JobState,
) -> impl Stream<Item = String>

async fn route_operation(
    payload: serde_json::Value,
    // ... state
) -> Result<()>
```

#### 4. SSE Streaming Integration
**Current:** Uses `observability-narration-core::sse_sink`
**Status:** ✅ Already shared
**Pattern:**
```rust
// Subscribe to SSE broadcaster
let mut sse_rx = sse_sink::subscribe()?;

// Stream events
loop {
    match sse_rx.recv().await {
        Ok(event) => yield Event::default().data(format!("[{}] {}", event.actor, event.human)),
        Err(_) => break,
    }
}
```

## Proposed New Shared Crate: `job-server`

### Purpose
Provide reusable job-based HTTP server components for both queen-rbee and rbee-hive.

### Structure
```
bin/99_shared_crates/job-server/
├── Cargo.toml
└── src/
    ├── lib.rs              # Re-exports
    ├── http.rs             # HTTP endpoint handlers
    ├── router.rs           # Job routing traits
    └── types.rs            # JobResponse, JobState trait
```

### API Design

```rust
// Trait for operation routing
pub trait OperationRouter: Send + Sync {
    type Operation: DeserializeOwned;
    
    async fn route_operation(
        &self,
        operation: Self::Operation,
    ) -> Result<()>;
}

// Trait for job state
pub trait JobState: Clone + Send + Sync {
    fn registry(&self) -> Arc<JobRegistry<String>>;
}

// HTTP handlers (generic over operation type)
pub async fn handle_create_job<S, O>(
    State(state): State<S>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<JobResponse>, (StatusCode, String)>
where
    S: JobState,
    O: DeserializeOwned;

pub async fn handle_stream_job<S>(
    Path(job_id): Path<String>,
    State(state): State<S>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
where
    S: JobState;
```

### Usage in queen-rbee

```rust
use job_server::{OperationRouter, JobState, handle_create_job, handle_stream_job};
use rbee_operations::QueenOperation;

struct QueenJobState {
    registry: Arc<JobRegistry<String>>,
    hive_catalog: Arc<HiveCatalog>,
    hive_registry: Arc<HiveRegistry>,
}

impl JobState for QueenJobState {
    fn registry(&self) -> Arc<JobRegistry<String>> {
        self.registry.clone()
    }
}

impl OperationRouter for QueenJobState {
    type Operation = QueenOperation;
    
    async fn route_operation(&self, op: QueenOperation) -> Result<()> {
        match op {
            QueenOperation::Status => { /* ... */ }
            QueenOperation::HiveList => { /* ... */ }
            // ...
        }
    }
}

// In router setup
Router::new()
    .route("/v1/jobs", post(handle_create_job::<QueenJobState, QueenOperation>))
    .route("/v1/jobs/{job_id}/stream", get(handle_stream_job::<QueenJobState>))
```

### Usage in rbee-hive

```rust
use job_server::{OperationRouter, JobState, handle_create_job, handle_stream_job};
use hive_operations::HiveOperation;

struct HiveJobState {
    registry: Arc<JobRegistry<String>>,
    worker_registry: Arc<WorkerRegistry>,
    model_catalog: Arc<ModelCatalog>,
}

impl JobState for HiveJobState {
    fn registry(&self) -> Arc<JobRegistry<String>> {
        self.registry.clone()
    }
}

impl OperationRouter for HiveJobState {
    type Operation = HiveOperation;
    
    async fn route_operation(&self, op: HiveOperation) -> Result<()> {
        match op {
            HiveOperation::WorkerSpawn { .. } => { /* ... */ }
            HiveOperation::ModelList => { /* ... */ }
            // ...
        }
    }
}

// Same router pattern!
Router::new()
    .route("/v1/jobs", post(handle_create_job::<HiveJobState, HiveOperation>))
    .route("/v1/jobs/{job_id}/stream", get(handle_stream_job::<HiveJobState>))
```

## Operation Forwarding Pattern

When queen-rbee receives a hive operation, it forwards to the hive:

```rust
// In queen-rbee's route_operation
match operation {
    QueenOperation::Status => { /* handle locally */ }
    QueenOperation::HiveList => { /* handle locally */ }
    
    // Forward to hive
    QueenOperation::Forward(hive_op) => {
        // 1. Get hive from catalog
        let hive = state.hive_catalog.get(&hive_op.hive_id())?;
        
        // 2. POST to hive's /v1/jobs endpoint
        let client = reqwest::Client::new();
        let response = client
            .post(format!("http://{}:{}/v1/jobs", hive.host, hive.port))
            .json(&hive_op)
            .send()
            .await?;
        
        let job_response: JobResponse = response.json().await?;
        
        // 3. Stream from hive's SSE endpoint
        let sse_stream = client
            .get(&job_response.sse_url)
            .send()
            .await?;
        
        // 4. Forward SSE events to our client
        // (narration events are automatically forwarded via sse_sink)
    }
}
```

## Implementation Plan

### Phase 1: Split Operations (TEAM-191A)
1. Create `bin/99_shared_crates/hive-operations/`
2. Move hive operations to new crate
3. Keep queen operations in `rbee-operations`
4. Update imports in queen-rbee

### Phase 2: Create job-server Crate (TEAM-191B)
1. Create `bin/99_shared_crates/job-server/`
2. Extract HTTP handlers from queen-rbee
3. Define traits (OperationRouter, JobState)
4. Add generic implementations

### Phase 3: Refactor queen-rbee (TEAM-191C)
1. Implement traits for QueenJobState
2. Replace custom HTTP handlers with job-server
3. Test all operations still work

### Phase 4: Implement rbee-hive Job Architecture (TEAM-191D)
1. Add job-registry to rbee-hive
2. Implement HiveJobState
3. Implement OperationRouter for hive operations
4. Add /v1/jobs endpoints using job-server

### Phase 5: Implement Operation Forwarding (TEAM-191E)
1. Add HTTP client to queen-rbee
2. Implement forwarding logic
3. Test queen → hive → queen flow

## Benefits

### Code Reuse
- **~200 lines** of HTTP endpoint code shared
- **~100 lines** of job routing pattern shared
- **~50 lines** of SSE streaming logic shared
- **Total: ~350 lines** eliminated duplication

### Type Safety
- Compile-time enforcement of operation separation
- Clear contract between queen and hive
- Impossible to accidentally handle wrong operation

### Consistency
- Both binaries use identical job architecture
- Same client experience (POST /v1/jobs, GET /stream)
- Same narration/SSE patterns

### Maintainability
- Fix bugs in one place
- Add features to both binaries simultaneously
- Clear separation of concerns

## Risks & Mitigations

### Risk: Over-abstraction
**Mitigation:** Keep traits simple, allow customization

### Risk: Breaking changes
**Mitigation:** Implement incrementally, test at each phase

### Risk: Performance overhead
**Mitigation:** Use zero-cost abstractions (traits, generics)

## Next Steps

1. ✅ Analyze current operations (this document)
2. 🔲 Get user approval on split approach
3. 🔲 Implement Phase 1 (split operations)
4. 🔲 Design job-server API in detail
5. 🔲 Implement Phase 2 (create job-server)
6. 🔲 Continue with phases 3-5

---

**Questions for User:**
1. Approve two-enum approach (QueenOperation + HiveOperation)?
2. Approve new `job-server` shared crate?
3. Should we include operation forwarding in initial scope?
