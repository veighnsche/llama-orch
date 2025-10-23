# TEAM-273 to TEAM-275: Final Implementation Phases

**Phases:** 7-9 of 9  
**Teams:** 273 (Job Router Integration), 274 (HTTP Testing), 275 (Mode 3)

---

## TEAM-273: Hive Job Router Integration (Phase 7)

**Estimated Effort:** 16-20 hours  
**Prerequisites:** TEAM-272 complete

### Mission

Verify all 8 operations are properly wired up in job_router.rs. Remove all TODO markers. Ensure clean compilation.

### Key Deliverables

1. ✅ All TODO markers removed from job_router.rs
2. ✅ All 8 operations calling real functions
3. ✅ Proper error handling for all operations
4. ✅ Clean compilation with no warnings
5. ✅ Integration smoke tests

### Checklist

**Operations to verify:**

- [ ] WorkerSpawn - Calls worker_spawner.spawn_worker()
- [ ] WorkerList - Calls worker_registry.list()
- [ ] WorkerGet - Calls worker_registry.get()
- [ ] WorkerDelete - Calls worker_registry.remove() + process kill
- [ ] ModelDownload - Calls model_provisioner.download_model()
- [ ] ModelList - Calls model_catalog.list()
- [ ] ModelGet - Calls model_catalog.get()
- [ ] ModelDelete - Calls model_catalog.remove() + file deletion

**State verification:**

```rust
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>,
    pub model_provisioner: Arc<ModelProvisioner>,
    pub worker_registry: Arc<WorkerRegistry>,
    pub worker_spawner: Arc<WorkerSpawner>,
}
```

**main.rs initialization:**

```rust
// TEAM-273: Initialize all hive state
let model_catalog = Arc::new(ModelCatalog::new());
let model_provisioner = Arc::new(ModelProvisioner::new(model_catalog.clone())?);
let worker_registry = Arc::new(WorkerRegistry::new());
let worker_spawner = Arc::new(WorkerSpawner::new(worker_registry.clone())?);

let job_state = http::jobs::HiveState {
    registry: job_registry,
    model_catalog,
    model_provisioner,
    worker_registry,
    worker_spawner,
};
```

### Acceptance Criteria

- [ ] All TODO markers removed
- [ ] All operations implemented
- [ ] `cargo check --bin rbee-hive` passes with NO warnings
- [ ] `cargo build --bin rbee-hive` succeeds
- [ ] Manual smoke test: rbee-hive starts and responds to /health

### Testing Commands

```bash
# Clean build
cargo clean
cargo build --bin rbee-hive

# Start hive
cargo run --bin rbee-hive -- --port 8600

# Test health endpoint
curl http://localhost:8600/health
# Expected: "ok"

# Test capabilities
curl http://localhost:8600/capabilities
# Expected: JSON with devices
```

---

## TEAM-274: HTTP Mode Testing & Validation (Phase 8)

**Estimated Effort:** 16-24 hours  
**Prerequisites:** TEAM-273 complete

### Mission

Validate all operations work correctly via HTTP. Establish performance baselines. Document known limitations.

### Key Deliverables

1. ✅ Integration tests for all 8 operations
2. ✅ End-to-end workflow tests
3. ✅ Performance benchmarks
4. ✅ Known limitations documented
5. ✅ Test report

### Test Plan

#### Test 1: Model Operations

```bash
# Start hive
cargo run --bin rbee-hive -- --port 8600

# Test ModelList (should be empty)
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "model_list", "hive_id": "localhost"}'

# Get job_id from response, then:
curl http://localhost:8600/v1/jobs/{job_id}/stream

# Expected: SSE stream with "No models found"

# Test ModelDownload
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "model_download", "hive_id": "localhost", "model": "test-model"}'

# Expected: Download progress events

# Test ModelGet
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "model_get", "hive_id": "localhost", "id": "test-model"}'

# Expected: Model details JSON

# Test ModelDelete
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "model_delete", "hive_id": "localhost", "id": "test-model"}'

# Expected: Deletion confirmation
```

#### Test 2: Worker Operations

```bash
# Test WorkerList (should be empty)
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "worker_list", "hive_id": "localhost"}'

# Test WorkerSpawn
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "worker_spawn", "hive_id": "localhost", "model": "test-model", "worker": "worker-1", "device": "CPU-0"}'

# Expected: Worker spawn events (may fail if worker binary missing)

# Test WorkerGet
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "worker_get", "hive_id": "localhost", "id": "worker-1"}'

# Test WorkerDelete
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "worker_delete", "hive_id": "localhost", "id": "worker-1"}'
```

#### Test 3: Error Handling

```bash
# Test getting non-existent model
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "model_get", "hive_id": "localhost", "id": "nonexistent"}'

# Expected: Error narration event

# Test deleting non-existent worker
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "worker_delete", "hive_id": "localhost", "id": "nonexistent"}'

# Expected: Error narration event
```

### Performance Benchmarks

Measure operation latency:

```bash
# Benchmark ModelList (100 iterations)
time for i in {1..100}; do
  curl -s -X POST http://localhost:8600/v1/jobs \
    -H "Content-Type: application/json" \
    -d '{"operation": "model_list", "hive_id": "localhost"}' > /dev/null
done

# Expected: ~1.1ms per operation (HTTP overhead)
```

### Known Limitations to Document

1. **Worker binary missing** - WorkerSpawn will fail without actual worker binary
2. **Model download stub** - Downloads don't actually fetch from HuggingFace yet
3. **Process cleanup** - WorkerDelete may not kill process properly
4. **File deletion** - ModelDelete removes from catalog but may not delete files

### Acceptance Criteria

- [ ] All 8 operations tested manually
- [ ] Error handling verified
- [ ] Performance baselines measured
- [ ] Known limitations documented
- [ ] Test report created (TEAM_274_TEST_REPORT.md)

---

## TEAM-275: Mode 3 (Integrated) Implementation (Phase 9)

**Estimated Effort:** 30-58 hours  
**Prerequisites:** TEAM-274 complete, HTTP mode validated

### Mission

Implement Mode 3 (Integrated) for localhost operations. Achieve 110x speedup by calling rbee-hive crates directly instead of via HTTP.

### Key Deliverables

1. ✅ IntegratedHive struct
2. ✅ execute_integrated() function
3. ✅ Updated forward_to_hive() routing
4. ✅ Feature flag working
5. ✅ Performance benchmarks showing speedup
6. ✅ No breaking changes to HTTP mode

### Implementation Guide

#### Step 1: Add Optional Dependencies

**File:** `bin/10_queen_rbee/Cargo.toml`

```toml
[dependencies]
# TEAM-275: Optional dependencies for integrated mode
rbee-hive-worker-lifecycle = { path = "../25_rbee_hive_crates/worker-lifecycle", optional = true }
rbee-hive-model-catalog = { path = "../25_rbee_hive_crates/model-catalog", optional = true }
rbee-hive-model-provisioner = { path = "../25_rbee_hive_crates/model-provisioner", optional = true }

[features]
local-hive = [
    "rbee-hive-worker-lifecycle",
    "rbee-hive-model-catalog",
    "rbee-hive-model-provisioner",
]
```

#### Step 2: Create IntegratedHive Struct

**File:** `bin/10_queen_rbee/src/integrated_hive.rs`

```rust
// TEAM-275: Integrated hive for Mode 3
#![cfg(feature = "local-hive")]

use anyhow::Result;
use rbee_hive_model_catalog::ModelCatalog;
use rbee_hive_model_provisioner::ModelProvisioner;
use rbee_hive_worker_lifecycle::{WorkerRegistry, WorkerSpawner};
use std::sync::Arc;

pub struct IntegratedHive {
    pub model_catalog: Arc<ModelCatalog>,
    pub model_provisioner: Arc<ModelProvisioner>,
    pub worker_registry: Arc<WorkerRegistry>,
    pub worker_spawner: Arc<WorkerSpawner>,
}

impl IntegratedHive {
    pub fn new() -> Result<Self> {
        let model_catalog = Arc::new(ModelCatalog::new());
        let model_provisioner = Arc::new(ModelProvisioner::new(model_catalog.clone())?);
        let worker_registry = Arc::new(WorkerRegistry::new());
        let worker_spawner = Arc::new(WorkerSpawner::new(worker_registry.clone())?);
        
        Ok(Self {
            model_catalog,
            model_provisioner,
            worker_registry,
            worker_spawner,
        })
    }
}
```

#### Step 3: Implement execute_integrated()

**File:** `bin/10_queen_rbee/src/hive_forwarder.rs`

```rust
// TEAM-275: Integrated mode execution
#[cfg(feature = "local-hive")]
async fn execute_integrated(
    job_id: &str,
    operation: Operation,
    integrated_hive: Arc<IntegratedHive>,
) -> Result<()> {
    use observability_narration_core::NarrationFactory;
    const NARRATE: NarrationFactory = NarrationFactory::new("qn-integrated");
    
    match operation {
        Operation::WorkerSpawn { hive_id, model, worker, device } => {
            NARRATE
                .action("integrated_spawn")
                .job_id(job_id)
                .context(&worker)
                .human("🚀 [Integrated] Spawning worker '{}'")
                .emit();
            
            integrated_hive
                .worker_spawner
                .spawn_worker(job_id, &worker, &model, &device)
                .await?;
        }
        
        Operation::WorkerList { .. } => {
            let workers = integrated_hive.worker_registry.list();
            
            NARRATE
                .action("integrated_list")
                .job_id(job_id)
                .context(workers.len().to_string())
                .human("📋 [Integrated] Found {} worker(s)")
                .emit();
            
            for worker in &workers {
                NARRATE
                    .action("worker_entry")
                    .job_id(job_id)
                    .context(&worker.id)
                    .context(&worker.model_id)
                    .human("  {} | {}")
                    .emit();
            }
        }
        
        Operation::WorkerGet { id, .. } => {
            let worker = integrated_hive.worker_registry.get(&id)?;
            let json = serde_json::to_string_pretty(&worker)?;
            
            NARRATE
                .action("integrated_get")
                .job_id(job_id)
                .human(&json)
                .emit();
        }
        
        Operation::WorkerDelete { id, .. } => {
            integrated_hive.worker_registry.remove(&id)?;
            
            NARRATE
                .action("integrated_delete")
                .job_id(job_id)
                .context(&id)
                .human("✅ [Integrated] Worker deleted: {}")
                .emit();
        }
        
        Operation::ModelDownload { model, .. } => {
            integrated_hive
                .model_provisioner
                .download_model(job_id, &model)
                .await?;
        }
        
        Operation::ModelList { .. } => {
            let models = integrated_hive.model_catalog.list();
            
            NARRATE
                .action("integrated_model_list")
                .job_id(job_id)
                .context(models.len().to_string())
                .human("📋 [Integrated] Found {} model(s)")
                .emit();
            
            for model in &models {
                NARRATE
                    .action("model_entry")
                    .job_id(job_id)
                    .context(&model.id)
                    .human("  {}")
                    .emit();
            }
        }
        
        Operation::ModelGet { id, .. } => {
            let model = integrated_hive.model_catalog.get(&id)?;
            let json = serde_json::to_string_pretty(&model)?;
            
            NARRATE
                .action("integrated_model_get")
                .job_id(job_id)
                .human(&json)
                .emit();
        }
        
        Operation::ModelDelete { id, .. } => {
            integrated_hive.model_catalog.remove(&id)?;
            
            NARRATE
                .action("integrated_model_delete")
                .job_id(job_id)
                .context(&id)
                .human("✅ [Integrated] Model deleted: {}")
                .emit();
        }
        
        _ => {
            return Err(anyhow::anyhow!(
                "Operation {:?} should not be forwarded to integrated hive",
                operation
            ));
        }
    }
    
    Ok(())
}
```

#### Step 4: Update forward_to_hive()

```rust
// TEAM-275: Updated routing with Mode 3
pub async fn forward_to_hive(
    job_id: &str,
    operation: Operation,
    config: Arc<QueenConfig>,
    #[cfg(feature = "local-hive")]
    integrated_hive: Option<Arc<IntegratedHive>>,
) -> Result<()> {
    let hive_id = operation.hive_id().unwrap_or("localhost");
    let operation_name = operation.name();
    
    let is_localhost = hive_id == "localhost";
    let has_integrated = cfg!(feature = "local-hive");
    
    let mode = if is_localhost && has_integrated {
        "integrated"
    } else if is_localhost {
        "localhost-http"
    } else {
        "remote-http"
    };
    
    NARRATE
        .action("forward_start")
        .job_id(job_id)
        .context(operation_name)
        .context(&hive_id)
        .context(mode)
        .human("Forwarding {} operation to hive '{}' (mode: {})")
        .emit();
    
    // TEAM-275: Mode 3 implementation
    #[cfg(feature = "local-hive")]
    if is_localhost && has_integrated {
        if let Some(hive) = integrated_hive {
            NARRATE
                .action("forward_integrated")
                .job_id(job_id)
                .human("⚡ Using integrated mode (direct function calls)")
                .emit();
            
            return execute_integrated(job_id, operation, hive).await;
        }
    }
    
    // Fallback to HTTP mode (existing code)
    stream_from_hive(job_id, operation, config).await
}
```

#### Step 5: Initialize in main.rs

```rust
// TEAM-275: Initialize integrated hive if feature enabled
#[cfg(feature = "local-hive")]
let integrated_hive = {
    NARRATE
        .action("integrated_init")
        .human("🔧 Initializing integrated hive state")
        .emit();
    
    Some(Arc::new(IntegratedHive::new()?))
};

#[cfg(not(feature = "local-hive"))]
let integrated_hive = None;

let job_state = http::SchedulerState {
    registry: job_server,
    config: config.clone(),
    hive_registry: worker_registry.clone(),
    integrated_hive: integrated_hive.clone(),
};
```

### Performance Testing

```bash
# Build with Mode 3
cargo build --bin queen-rbee --features local-hive

# Benchmark WorkerList (100 iterations)
# HTTP mode:
time for i in {1..100}; do
  rbee worker list --hive localhost > /dev/null
done
# Expected: ~110ms (1.1ms per op)

# Integrated mode:
time for i in {1..100}; do
  rbee worker list --hive localhost > /dev/null
done
# Expected: ~1ms (0.01ms per op)

# Speedup: 110x ✅
```

### Acceptance Criteria

- [ ] IntegratedHive struct implemented
- [ ] execute_integrated() function working
- [ ] forward_to_hive() routing updated
- [ ] Feature flag working correctly
- [ ] `cargo build --features local-hive` succeeds
- [ ] All 8 operations work in integrated mode
- [ ] Performance benchmarks show 100x+ speedup
- [ ] HTTP mode still works (no breaking changes)
- [ ] Documentation updated

### Documentation Updates

Update these files:

1. **QUEEN_TO_HIVE_COMMUNICATION_MODES.md**
   - Change Mode 3 status from "BLOCKED" to "✅ IMPLEMENTED"
   - Add performance benchmarks
   - Update implementation status table

2. **README.md**
   - Add Mode 3 feature flag documentation
   - Add build instructions with `--features local-hive`

3. **CHANGELOG.md**
   - Add entry for Mode 3 implementation

---

## Summary

**TEAM-273:** Job router integration (16-20h)  
**TEAM-274:** HTTP testing & validation (16-24h)  
**TEAM-275:** Mode 3 implementation (30-58h)

**Total:** 62-102 hours for final phases

**Grand Total (All 9 Phases):** 260-376 hours (6-9 weeks)

---

## 🎉 Success!

When TEAM-275 completes:

- ✅ All 8 operations working via HTTP
- ✅ Mode 3 (Integrated) working for localhost
- ✅ 110x speedup achieved
- ✅ No breaking changes
- ✅ Full test coverage
- ✅ Complete documentation

**The rbee-hive implementation is DONE! 🐝**
