# Scheduling SPEC — Job Ordering & Worker Selection (SCHED-11xxx)

**Status**: Draft  
**Applies to**: `bin/orchestratord-crates/scheduling/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

The `scheduling` crate is the **complete scheduling system** for orchestratord. It owns job admission, queue management, job tracking, and worker selection. This is the centralized brain for all scheduling decisions.

**Why it exists:**
- Orchestratord is "the brain" - scheduling is its core intelligence
- Consolidate all scheduling logic in one place (admission → queue → selection → dispatch)
- Provide cohesive scheduling with queue, admission, job tracking as internal modules
- Combined decision: "Admit job → enqueue → select worker → dispatch"

**What it does:**
- **Admission** (src/admission.rs): Validate requests before enqueue (model exists, context length, budget)
- **Queue** (src/queue.rs): Bounded FIFO with Interactive/Batch priorities, reject/drop-lru policies
- **Job Tracker** (src/job_tracker.rs): Track job state (Queued → Dispatched → Executing → Completed/Failed)
- **Worker Selection** (src/scheduler.rs): Pick best worker (least-loaded, most-vram-free, round-robin)
- **Scheduling Loop** (src/scheduler.rs): Main scheduling loop (dequeue → select worker → dispatch)
- **Eviction Decisions**: Decide which workers to evict (LRU, LFU, manual)

**What it does NOT do:**
- ❌ Execute inference (workers do this)
- ❌ Spawn workers (pool managers do this via worker-lifecycle)
- ❌ Stream results (streaming crate does this)
- ❌ Track pool/worker state (pool-registry crate does this)

**Internal Modules:**
```
scheduling/
  src/
    admission.rs      # Validate requests before enqueue
    queue.rs          # Bounded FIFO with priorities
    job_tracker.rs    # Track job lifecycle state
    scheduler.rs      # Worker selection & main loop
    eviction.rs       # Eviction policy decisions
    lib.rs
```

---

## 1. Core Responsibilities

### [SCHED-11001] Admission
The crate MUST validate requests before enqueue (model exists, context length, budget).

### [SCHED-11002] Queue Management
The crate MUST implement bounded FIFO queue with Interactive/Batch priorities.

### [SCHED-11003] Job Tracking
The crate MUST track job lifecycle state transitions.

### [SCHED-11004] Job Selection
The crate MUST select next job from queue respecting priority and ordering.

### [SCHED-11005] Worker Selection
The crate MUST select best available worker for job.

### [SCHED-11006] Eviction Decisions
The crate MUST decide which workers to evict when VRAM needed.

---

## 2. Admission (src/admission.rs)

### [SCHED-11010] Admission Rules
Before enqueue, validate:
```rust
pub fn admit(request: InferenceRequest, catalog: &Catalog) -> Result<Job, AdmissionError>;
```

1. **Model exists**: Check model in catalog
2. **Context length**: Verify `prompt.tokens() <= model.context_max`
3. **Token budget**: Verify `max_tokens <= session.budget`
4. **Valid parameters**: Check temperature, seed ranges

### [SCHED-11011] Admission Errors
```rust
pub enum AdmissionError {
    ModelNotFound(String),
    ContextTooLarge { requested: usize, max: usize },
    BudgetExceeded { requested: usize, remaining: usize },
    InvalidParameter(String),
}
```

---

## 3. Queue (src/queue.rs)

### [SCHED-11020] Queue Structure
```rust
pub struct JobQueue {
    interactive: VecDeque<Job>,
    batch: VecDeque<Job>,
    capacity: usize,
    policy: QueuePolicy,
}
```

### [SCHED-11021] Enqueue
`enqueue(job: Job, priority: Priority)` MUST:
1. Check if queue is full
2. If full and policy is `Reject`: return `EnqueueError::QueueFullReject`
3. If full and policy is `DropLru`: drop oldest batch job, then enqueue
4. Add job to appropriate priority queue
5. Return `Ok(())`

### [SCHED-11022] Dequeue
`dequeue()` MUST:
1. If interactive queue not empty: pop from interactive
2. Else if batch queue not empty: pop from batch
3. Else: return `None`

### [SCHED-11023] Drop LRU Policy
When queue full with `DropLru` policy:
1. Find oldest batch job (front of queue)
2. Remove it from queue
3. Emit metrics: `tasks_dropped_total`
4. Continue with enqueue

---

## 4. Job Tracker (src/job_tracker.rs)

### [SCHED-11030] Job Structure
```rust
pub struct Job {
    pub job_id: String,
    pub session_id: String,
    pub model_ref: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub seed: Option<u64>,
    pub priority: Priority,
    pub submitted_at: DateTime<Utc>,
}
```

### [SCHED-11031] Job State
Track job lifecycle:
```rust
pub enum JobState {
    Queued,      // In queue, waiting
    Dispatched,  // Sent to worker
    Executing,   // Worker processing
    Completed,   // Finished successfully
    Failed,      // Error occurred
    Cancelled,   // User cancelled
}
```

### [SCHED-11032] State Transitions
The tracker MUST enforce valid transitions:
- Queued → Dispatched → Executing → Completed/Failed
- Queued/Dispatched/Executing → Cancelled

---

## 5. Worker Selection

### [SCHED-11040] Priority Ordering
The crate MUST process jobs by priority:
1. `interactive` priority jobs first
2. `batch` priority jobs second
3. Within same priority, use FIFO ordering

### [SCHED-11041] Starvation Prevention
The crate SHOULD prevent starvation:
- Promote long-waiting batch jobs to interactive priority
- Default: promote after 5 minutes in queue

### [SCHED-11042] Fair Scheduling
The crate SHOULD implement fair scheduling within same priority:
- Round-robin between users/sessions
- Prevent single user monopolizing queue

---

---

## 6. Worker Selection (Placement)

### [SCHED-11050] Placement Algorithms
The crate MUST support placement algorithms:

**least-loaded** (default):
- Select worker with fewest active jobs
- Tie-break: select worker on GPU with most free VRAM

**most-vram-free**:
- Select worker on GPU with most free VRAM
- Useful for memory-intensive jobs

**round-robin**:
- Cycle through workers deterministically
- Fair distribution across workers

### [SCHED-11051] Model Matching
The crate MUST only select workers that have the requested model loaded.

### [SCHED-11052] State Query
Before placement, the crate MUST query pool managers:
```
GET /v2/state
```
Get current GPU VRAM, running workers, worker URIs.

---

---

## 7. Worker Startup Decisions

### [SCHED-11060] No Worker Available
If no suitable worker exists, the crate MUST decide:

**Option 1: Start new worker**
- Query pool managers for GPU with sufficient free VRAM
- Command pool manager: `POST /v2/workers/start { model_ref, gpu_id }`
- Wait for worker to become ready (with timeout)
- Retry placement once worker available

**Option 2: Reject job**
- No pool has sufficient VRAM
- All pools are at capacity
- Return error: `POOL_UNAVAILABLE`

**Option 3: Wait and retry**
- Existing worker may free up soon
- Retry placement after delay (with backoff)

### [SCHED-11061] Startup Timeout
If worker doesn't become ready within timeout (default 60s):
- Retry once (maybe transient error)
- If second attempt fails, reject job

---

---

## 8. Scheduling Loop

### [SCHED-11070] Main Loop
The scheduler MUST run continuously:

```
loop {
    1. Check if workers available
    2. If no workers and queue not empty:
       - Attempt to start new worker (if VRAM available)
    3. Select next job from queue
    4. Select best worker for job
    5. Dispatch job to worker
    6. Update worker state (mark busy)
    7. Repeat
}
```

### [SCHED-11071] Dispatch
When dispatching job to worker:
1. Send request to worker: `POST {worker_uri}/execute`
2. Relay SSE stream to client (via streaming layer)
3. On completion, mark worker available again

---

---

## 9. Eviction Decisions

### [SCHED-11080] Eviction Policy
When VRAM is needed and all GPUs full, the crate MUST decide which worker to evict:

**LRU** (least recently used):
- Evict worker for model least recently used

**LFU** (least frequently used):
- Evict worker for model least frequently used

**Manual**:
- Only evict on explicit admin command

### [SCHED-11081] Eviction Safety
The crate MUST NOT evict workers with active jobs:
- Wait for job to complete, then evict
- Or drain worker (finish job, then stop)

---

---

## 10. Error Handling

### [SCHED-11090] Error Types
```rust
pub enum SchedulingError {
    NoWorkersAvailable,
    PoolUnavailable,
    WorkerStartupFailed,
    WorkerStartupTimeout,
    DispatchFailed,
    InsufficientVram,
}
```

### [SCHED-11091] Retry Logic
The crate MUST implement retry with backoff:
- Retry transient errors (worker unreachable, startup timeout)
- Do NOT retry permanent errors (insufficient VRAM, invalid request)

---

---

## 11. Metrics

### [SCHED-11100] Queue Metrics
The crate MUST track:
```rust
pub struct QueueMetrics {
    pub queue_depth: usize,
    pub tasks_enqueued_total: u64,
    pub tasks_rejected_total: u64,
    pub tasks_dropped_total: u64,
    pub queue_wait_time_ms: Histogram,
}
```

### [SCHED-11101] Scheduling Metrics
The crate MUST track:
- `scheduling_latency_ms` — Time to select worker
- `placement_algorithm{type}` — Which algorithm used
- `worker_starts_total` — New workers started
- `evictions_total{policy}` — Workers evicted

---

## 12. Dependencies

### [SCHED-11110] Required Crates
```toml
[dependencies]
pool-registry = { path = "../pool-registry" }
tokio = { workspace = true }
tracing = { workspace = true }
thiserror = { workspace = true }
chrono = { workspace = true }
serde = { workspace = true, features = ["derive"] }
```

---

## 13. Traceability

**Code**: `bin/orchestratord-crates/scheduling/src/`  
**Tests**: `bin/orchestratord-crates/scheduling/tests/`  
**Parent**: `bin/orchestratord/.specs/00_orchestratord.md`  
**Used by**: `orchestratord`  
**Depends on**: `pool-registry`  
**Spec IDs**: SCHED-11001 to SCHED-11110

---

**End of Specification**
