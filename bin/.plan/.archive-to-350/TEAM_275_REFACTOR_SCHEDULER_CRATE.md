# TEAM-275 Refactor: Inference Scheduler Crate

**Date:** Oct 23, 2025  
**Status:** ✅ COMPLETE  
**Mission:** Extract scheduler into dedicated crate, pre-wire for M2 Rhai scheduler

---

## 🎯 What We Did

Refactored the simple inference scheduler into a dedicated crate that's **pre-wired for the M2 Rhai programmable scheduler**.

### Key Changes

1. **Created New Crate:** `queen-rbee-inference-scheduler`
2. **Moved Code:** From `queen-rbee/src/inference_scheduler.rs` to crate
3. **Added Abstraction:** `InferenceScheduler` trait for pluggable implementations
4. **Current Implementation:** `SimpleScheduler` (first available worker)
5. **Future Ready:** Stub for `RhaiScheduler` (M2+)

---

## 📁 New Crate Structure

```
bin/15_queen_rbee_crates/inference-scheduler/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs         - Trait definition + exports
    ├── types.rs       - Request/Response/Error types
    └── simple.rs      - SimpleScheduler implementation
```

**Total:** ~600 LOC

---

## 🏗️ Architecture

### Strategy Pattern (Pluggable Schedulers)

```
┌─────────────────────────────────────────┐
│  InferenceScheduler (trait)             │
└─────────────────┬───────────────────────┘
                  │
       ┌──────────┴──────────┐
       │                     │
┌──────▼──────┐      ┌──────▼──────────┐
│ SimpleScheduler │  │ RhaiScheduler   │
│ (M0/M1)        │  │ (M2+)           │
│ ✅ Implemented │  │ 🚧 Stub         │
└────────────────┘  └─────────────────┘
```

### Trait Definition

```rust
#[async_trait]
pub trait InferenceScheduler: Send + Sync {
    async fn schedule(
        &self,
        request: InferenceRequest
    ) -> Result<ScheduleResult, SchedulerError>;
}
```

---

## 🔧 Current Implementation (M0/M1)

### SimpleScheduler

**Algorithm:**
1. Find workers serving requested model
2. Filter by online + available status
3. Return **first match** (no load balancing)

**Usage:**
```rust
use queen_rbee_inference_scheduler::{SimpleScheduler, InferenceRequest};

let scheduler = SimpleScheduler::new(worker_registry);
let result = scheduler.schedule(request).await?;
scheduler.execute_inference(result, request, line_handler).await?;
```

---

## 🚀 Future Implementation (M2+)

### RhaiScheduler (Programmable)

**Features (M2):**
- User-written Rhai scripts for custom routing
- 40+ built-in helper functions
- YAML config support (compiles to Rhai)
- Platform mode (immutable) vs Home/Lab mode (customizable)

**Example Rhai Script:**
```rhai
// scheduler.rhai
fn schedule_job(job, pools, workers) {
    // Route large models to multi-GPU setups
    if job.model.contains("70b") {
        return workers.filter(|w| w.gpus > 1).first();
    }
    
    // Route to cheapest available
    return workers.min_by(|w| w.cost_per_token);
}
```

**Usage (M2):**
```rust
use queen_rbee_inference_scheduler::RhaiScheduler;

let scheduler = RhaiScheduler::from_file("scheduler.rhai")?;
let result = scheduler.schedule(request).await?;
```

---

## 📝 Files Created

### New Crate Files

1. **`Cargo.toml`** - Crate manifest
   - Dependencies: async-trait, thiserror, reqwest, futures
   - TODO M2: Add `rhai = "1.16"`

2. **`README.md`** - Comprehensive documentation
   - Current implementation (SimpleScheduler)
   - Future implementation (RhaiScheduler)
   - Usage examples
   - Architecture diagrams

3. **`src/lib.rs`** - Trait definition
   - `InferenceScheduler` trait
   - Re-exports for SimpleScheduler
   - Stub comment for RhaiScheduler

4. **`src/types.rs`** - Type definitions
   - `InferenceRequest` - Request with model, prompt, params
   - `ScheduleResult` - Worker selection result
   - `WorkerInferenceRequest` - Payload sent to worker
   - `WorkerJobResponse` - Worker's response
   - `SchedulerError` - Error types with thiserror

5. **`src/simple.rs`** - SimpleScheduler implementation
   - `schedule()` - Find worker
   - `execute_inference()` - Run inference and stream
   - Full HTTP + SSE streaming logic

---

## 📝 Files Modified

### queen-rbee Changes

1. **`Cargo.toml`** (+1 line)
   - Added dependency: `queen-rbee-inference-scheduler`

2. **`src/lib.rs`** (-1 line)
   - Removed `inference_scheduler` module (moved to crate)

3. **`src/job_router.rs`** (+10 LOC)
   - Updated to use `queen_rbee_inference_scheduler` crate
   - Added TODO M2 comment for RhaiScheduler
   - Cleaner separation of concerns

4. **Deleted:** `src/inference_scheduler.rs`
   - Moved to dedicated crate

---

## ✅ Benefits

### Immediate (M0/M1)

1. **Better Architecture** - Scheduler is now a pluggable component
2. **Cleaner Separation** - Queen-rbee doesn't contain scheduling logic
3. **Easier Testing** - Scheduler can be tested independently
4. **Reusable** - Other components can use the scheduler

### Future (M2+)

5. **Easy to Extend** - Just implement `InferenceScheduler` trait
6. **No Breaking Changes** - Trait abstraction allows swapping implementations
7. **Pre-Wired** - Ready for Rhai scheduler when M2 arrives
8. **Multiple Schedulers** - Can have SimpleScheduler + RhaiScheduler + others

---

## 🔍 Code Comparison

### Before (Inline in queen-rbee)

```rust
// queen-rbee/src/inference_scheduler.rs (310 LOC)
pub async fn schedule_inference(...) -> Result<()> {
    // All scheduling logic inline
}

// queen-rbee/src/job_router.rs
use crate::inference_scheduler::schedule_inference;
schedule_inference(request, registry, handler).await?;
```

### After (Dedicated Crate)

```rust
// queen-rbee-inference-scheduler/src/lib.rs
#[async_trait]
pub trait InferenceScheduler: Send + Sync {
    async fn schedule(...) -> Result<ScheduleResult, SchedulerError>;
}

// queen-rbee-inference-scheduler/src/simple.rs
impl InferenceScheduler for SimpleScheduler { ... }

// queen-rbee/src/job_router.rs
use queen_rbee_inference_scheduler::{SimpleScheduler, InferenceScheduler};
let scheduler = SimpleScheduler::new(registry);
let result = scheduler.schedule(request).await?;
scheduler.execute_inference(result, request, handler).await?;
```

---

## 🎓 Design Patterns Used

### 1. Strategy Pattern
- `InferenceScheduler` trait = Strategy interface
- `SimpleScheduler` = Concrete strategy (M0/M1)
- `RhaiScheduler` = Concrete strategy (M2+)

### 2. Dependency Injection
- Scheduler injected into job_router
- Easy to swap implementations
- Testable with mock schedulers

### 3. Separation of Concerns
- Scheduling logic → Scheduler crate
- Job routing → Queen-rbee
- Worker tracking → Worker registry

---

## 📊 LOC Breakdown

**New Crate:**
- `lib.rs`: ~100 LOC (trait + docs)
- `types.rs`: ~120 LOC (types + errors)
- `simple.rs`: ~280 LOC (implementation)
- `README.md`: ~100 LOC (docs)
- **Total: ~600 LOC**

**Queen-rbee Changes:**
- Removed: 310 LOC (old inference_scheduler.rs)
- Added: 10 LOC (updated job_router.rs)
- **Net: -300 LOC in queen-rbee**

---

## 🧪 Compilation Status

```bash
cargo check --bin queen-rbee   # ✅ PASS
cargo check --bin rbee-keeper  # ✅ PASS
cargo check --bin rbee-hive    # ✅ PASS
```

**Warnings (non-blocking):**
- Unused imports (cleanup needed)
- Unused fields (will be used in M2)

---

## 🎯 M2 Migration Path

When implementing RhaiScheduler in M2:

### Step 1: Add Rhai Dependency
```toml
# Cargo.toml
[dependencies]
rhai = "1.16"
```

### Step 2: Implement RhaiScheduler
```rust
// src/rhai.rs
pub struct RhaiScheduler {
    engine: rhai::Engine,
    script: rhai::AST,
    worker_registry: Arc<WorkerRegistry>,
}

#[async_trait]
impl InferenceScheduler for RhaiScheduler {
    async fn schedule(&self, request: InferenceRequest) 
        -> Result<ScheduleResult, SchedulerError> 
    {
        // Execute Rhai script
        let result = self.engine.call_fn::<ScheduleResult>(
            &mut scope,
            &self.script,
            "schedule_job",
            (request, workers)
        )?;
        Ok(result)
    }
}
```

### Step 3: Update queen-rbee
```rust
// job_router.rs
// M0/M1: Simple scheduler
let scheduler = SimpleScheduler::new(registry);

// M2: Rhai scheduler
let scheduler = RhaiScheduler::from_file("scheduler.rhai")?;

// Same interface!
let result = scheduler.schedule(request).await?;
```

---

## 📚 Documentation

**See:**
- `.business/stakeholders/RHAI_PROGRAMMABLE_SCHEDULER.md` - Full Rhai vision
- `bin/15_queen_rbee_crates/inference-scheduler/README.md` - Crate docs
- `bin/.plan/TEAM_275_HANDOFF.md` - Original implementation
- `bin/.plan/TEAM_275_SUMMARY.md` - Quick reference

---

## ✅ Checklist

- [x] Created `queen-rbee-inference-scheduler` crate
- [x] Defined `InferenceScheduler` trait
- [x] Implemented `SimpleScheduler`
- [x] Added comprehensive types and errors
- [x] Moved code from queen-rbee to crate
- [x] Updated queen-rbee to use crate
- [x] Added TODO M2 comments for RhaiScheduler
- [x] Wrote comprehensive README
- [x] All binaries compile successfully
- [x] Pre-wired for M2 Rhai scheduler

---

## 🎉 Result

**The scheduler is now:**
- ✅ A dedicated, reusable crate
- ✅ Pluggable via trait abstraction
- ✅ Pre-wired for M2 Rhai scheduler
- ✅ Cleaner architecture
- ✅ Easier to test and extend

**Ready for M2 Rhai implementation! 🚀**

---

**TEAM-275 refactor complete! Scheduler crate ready for future! 🎯**
