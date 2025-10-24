# Scheduler vs Registry: Architecture Clarification

**TEAM-285:** Clarifying the relationship between worker-registry and scheduler

## Your Question

> "This code confuses me because it looks a lot like a scheduler. But scheduling is done by `/bin/15_queen_rbee_crates/scheduler`. Do we have 2 places where scheduling happens?"

## Answer: NO - Single Responsibility, Clear Separation ✅

**We have ONE scheduler (scheduler crate) that USES the registry as a data source.**

---

## Architecture: Layered Responsibility

```
┌─────────────────────────────────────────────────────────┐
│                    job_router.rs                        │
│                  (Orchestration Layer)                  │
│                                                         │
│  Operation::Infer → Create JobRequest                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│              queen-rbee-scheduler                       │
│                (Scheduling Logic)                       │
│                                                         │
│  SimpleScheduler::schedule(request)                    │
│  - Business logic: Which worker?                       │
│  - Policy: Load balancing, affinity, etc.             │
│  - Returns: ScheduleResult                             │
└────────────────────────┬────────────────────────────────┘
                         │
                         │ Uses ↓
                         │
┌─────────────────────────────────────────────────────────┐
│           queen-rbee-worker-registry                    │
│              (Data Access Layer)                        │
│                                                         │
│  find_best_worker_for_model(model_id)                  │
│  - Data query: Filter workers by model                 │
│  - Data query: Filter by online + available            │
│  - Returns: WorkerInfo (data)                          │
└─────────────────────────────────────────────────────────┘
```

---

## Detailed Breakdown

### 1. Worker Registry (Data Layer)

**File:** `bin/15_queen_rbee_crates/worker-registry/src/registry.rs`

**Purpose:** Data access and filtering

**Responsibilities:**
- Store worker heartbeats (in-memory cache)
- Filter workers by various criteria
- Provide **data queries** (not decisions)

**Key Methods:**
```rust
// Simple filters (data queries)
list_all_workers() -> Vec<WorkerInfo>
list_online_workers() -> Vec<WorkerInfo>
list_available_workers() -> Vec<WorkerInfo>
find_workers_by_model(model_id) -> Vec<WorkerInfo>

// Convenience method (still just a data query)
find_best_worker_for_model(model_id) -> Option<WorkerInfo>
```

**What `find_best_worker_for_model()` does:**
```rust
pub fn find_best_worker_for_model(&self, model_id: &str) -> Option<WorkerInfo> {
    self.inner
        .get_all_heartbeats()
        .into_iter()
        .filter(|hb| {
            hb.is_recent() &&           // Online?
            hb.worker.is_available() && // Ready?
            hb.worker.serves_model(model_id) // Has model?
        })
        .map(|hb| hb.worker)
        .next() // Return FIRST match (no policy!)
}
```

**This is NOT scheduling!** It's just:
1. Filter by model
2. Filter by online
3. Filter by available
4. Return first match

**No policy decisions:**
- ❌ No load balancing
- ❌ No affinity rules
- ❌ No priority handling
- ❌ No custom routing logic

---

### 2. Scheduler (Business Logic Layer)

**File:** `bin/15_queen_rbee_crates/scheduler/src/simple.rs`

**Purpose:** Scheduling decisions and policy

**Responsibilities:**
- Implement scheduling **policy**
- Make **business decisions** about worker selection
- Handle scheduling **errors** and **fallbacks**
- Future: Load balancing, affinity, priorities

**Current Implementation (SimpleScheduler):**
```rust
impl JobScheduler for SimpleScheduler {
    async fn schedule(&self, request: JobRequest) -> Result<ScheduleResult, SchedulerError> {
        // POLICY: Use registry to find worker
        let worker = self.worker_registry
            .find_best_worker_for_model(&request.model)
            .ok_or_else(|| SchedulerError::NoWorkersAvailable)?;
        
        // POLICY: Build result with worker URL
        Ok(ScheduleResult {
            worker_id: worker.id,
            worker_url: format!("http://localhost:{}", worker.port),
            worker_port: worker.port,
            model: worker.model_id,
        })
    }
}
```

**This IS scheduling!** It:
1. Receives job request
2. Applies policy (currently: first available)
3. Handles errors (no workers available)
4. Returns routing decision

**Future policies (M2+):**
- ✅ Load balancing (round-robin, least-loaded)
- ✅ Affinity rules (same worker for same user)
- ✅ Priority queues (interactive vs batch)
- ✅ Custom Rhai scripts (programmable routing)

---

## Why This Separation?

### 1. Single Responsibility Principle ✅

**Registry:**
- "What workers are available?"
- "Which workers serve this model?"
- Pure data access, no policy

**Scheduler:**
- "Which worker should handle this job?"
- "How do I balance load?"
- Business logic and policy

### 2. Testability ✅

**Registry tests:**
- Data filtering works correctly
- Heartbeat expiration works
- Concurrent access is safe

**Scheduler tests:**
- Scheduling policy works correctly
- Error handling works
- Load balancing works (future)

### 3. Extensibility ✅

**Want a new scheduling policy?**
- ✅ Implement `JobScheduler` trait
- ✅ Use registry methods as needed
- ❌ Don't modify registry

**Example - Future RhaiScheduler:**
```rust
impl JobScheduler for RhaiScheduler {
    async fn schedule(&self, request: JobRequest) -> Result<ScheduleResult> {
        // Custom Rhai script decides policy
        let script_result = self.engine.eval(&self.script)?;
        
        // Use registry to get worker data
        let workers = self.worker_registry.find_workers_by_model(&request.model);
        
        // Apply Rhai policy to select worker
        let selected = apply_rhai_policy(workers, script_result)?;
        
        Ok(ScheduleResult { ... })
    }
}
```

---

## Is `find_best_worker_for_model()` Misnamed?

**Maybe!** It could be more accurately named:

**Current name:** `find_best_worker_for_model()`  
**Better name:** `find_first_available_worker_for_model()`

**Why "best" is misleading:**
- Implies policy decision (load balancing, optimization)
- Actually just returns first match
- No "best" logic, just filtering

**Why we keep it:**
- Historical reasons (TEAM-270)
- Convenience method for simple cases
- Future: Could add simple load balancing here

**Recommendation:**
- ✅ Keep the name for now (backward compatibility)
- ✅ Add doc comment clarifying it's "first match"
- ✅ Real "best" logic belongs in scheduler

---

## Call Flow Example

### User Request: Infer with model "llama-3-8b"

```
1. job_router.rs receives Operation::Infer
   ↓
2. Creates JobRequest { model: "llama-3-8b", ... }
   ↓
3. Calls scheduler.schedule(request)
   ↓
4. SimpleScheduler.schedule():
   - Calls worker_registry.find_best_worker_for_model("llama-3-8b")
   - Registry returns: WorkerInfo { id: "worker-123", port: 9301, ... }
   - Scheduler builds: ScheduleResult { worker_url: "http://localhost:9301", ... }
   ↓
5. job_router.rs routes request to worker URL
   ↓
6. Worker executes inference
```

**Who makes the decision?** Scheduler (step 4)  
**Who provides the data?** Registry (step 4, nested call)

---

## Summary

### ❌ We do NOT have 2 schedulers

### ✅ We have:

1. **Worker Registry** (Data Layer)
   - Stores worker state
   - Provides filtered queries
   - No policy decisions

2. **Scheduler** (Business Logic Layer)
   - Makes routing decisions
   - Implements scheduling policy
   - Uses registry as data source

### Analogy

**Registry = Database**
- "SELECT * FROM workers WHERE model = 'llama-3-8b' AND status = 'Ready'"

**Scheduler = Application Logic**
- "Given these workers, which one should I use based on my policy?"

---

## Should We Refactor?

### Option 1: Keep As-Is ✅ (Recommended)

**Pros:**
- Clear separation already exists
- Registry is just a convenience wrapper
- Scheduler is the single source of truth for policy

**Cons:**
- `find_best_worker_for_model()` name is slightly misleading

### Option 2: Remove `find_best_worker_for_model()` from Registry

**Change:**
```rust
// Remove from registry
- find_best_worker_for_model()

// Scheduler does filtering itself
impl SimpleScheduler {
    async fn schedule(&self, request: JobRequest) -> Result<ScheduleResult> {
        let workers = self.worker_registry.find_workers_by_model(&request.model);
        let worker = workers.into_iter()
            .filter(|w| w.is_available())
            .next()
            .ok_or(...)?;
        // ...
    }
}
```

**Pros:**
- Even clearer separation
- Registry has no "best" logic at all

**Cons:**
- More verbose scheduler code
- Breaks existing code (minor)

### Recommendation: Keep As-Is

The current architecture is correct. The registry method is just a convenience that doesn't violate separation of concerns.

---

## Conclusion

**You were right to question it!** The name `find_best_worker_for_model()` does sound like scheduling.

**But the architecture is correct:**
- Registry: Data queries (filtering)
- Scheduler: Policy decisions (routing)
- Clear separation of concerns

**The "best" in the name is historical and slightly misleading, but the actual implementation is just "first available" which is a simple data query, not a policy decision.**

**Real scheduling policy lives in the scheduler crate, where it belongs!** ✅
