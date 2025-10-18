# 🔌 BDD Wiring Opportunities - Product Code Ready

**Date:** 2025-10-11  
**Team:** TEAM-080  
**Purpose:** Identify stub step functions where product code ALREADY EXISTS and just needs wiring

---

## Executive Summary

**Found 25+ step functions that are stubs BUT product code is ready:**

- ✅ **queen-rbee WorkerRegistry** - 8 methods ready
- ✅ **model-catalog** - 5 methods ready  
- ✅ **DownloadTracker** - 4 methods ready
- ✅ **BeehiveRegistry** - 5 methods ready
- ✅ **SSH operations** - 2 methods ready

**Total: ~24 functions can be wired immediately (no new product code needed)**

---

## Category 1: WorkerRegistry Operations (8 functions)

### Product Code Available

**File:** `/bin/queen-rbee/src/worker_registry.rs`

```rust
impl WorkerRegistry {
    pub async fn register(&self, worker: WorkerInfo)
    pub async fn update_state(&self, worker_id: &str, state: WorkerState) -> bool
    pub async fn get(&self, worker_id: &str) -> Option<WorkerInfo>
    pub async fn list(&self) -> Vec<WorkerInfo>
    pub async fn remove(&self, worker_id: &str) -> bool
    pub async fn clear(&self)
    pub async fn count(&self) -> usize
    pub async fn list_workers(&self) -> Result<Vec<WorkerInfoExtended>>
}
```

### Stub Functions Ready to Wire

#### 1. State Transition Testing

**File:** `test-harness/bdd/src/steps/concurrency.rs`

```rust
#[given(expr = "worker-001 is transitioning from {string} to {string}")]
pub async fn given_worker_transitioning(world: &mut World, from: String, to: String) {
    // TEAM-079: Simulate state transition
    tracing::info!("TEAM-079: Worker transitioning from {} to {}", from, to);
    world.last_action = Some(format!("transitioning_{}_{}", from, to));
}
```

**Can wire to:**
```rust
#[given(expr = "worker-001 is transitioning from {string} to {string}")]
pub async fn given_worker_transitioning(world: &mut World, from: String, to: String) {
    // TEAM-080: Wire to real WorkerRegistry state transitions
    let registry = world.queen_registry.as_ref().expect("Registry not initialized").inner();
    
    // Set initial state
    let from_state = match from.as_str() {
        "idle" => WorkerState::Idle,
        "busy" => WorkerState::Busy,
        "loading" => WorkerState::Loading,
        _ => panic!("Unknown state: {}", from),
    };
    
    // Register worker with initial state
    let worker = WorkerInfo {
        id: "worker-001".to_string(),
        url: "http://localhost:8081".to_string(),
        model_ref: "test-model".to_string(),
        backend: "cpu".to_string(),
        device: 0,
        state: from_state,
        slots_total: 4,
        slots_available: 4,
        vram_bytes: None,
        node_name: "test-node".to_string(),
    };
    registry.register(worker).await;
    
    // Start transition to new state (spawn task to simulate async transition)
    let to_state = match to.as_str() {
        "idle" => WorkerState::Idle,
        "busy" => WorkerState::Busy,
        "loading" => WorkerState::Loading,
        _ => panic!("Unknown state: {}", to),
    };
    
    let reg = registry.clone();
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        reg.update_state("worker-001", to_state).await;
    });
    
    tracing::info!("TEAM-080: Worker-001 transitioning {} -> {}", from, to);
    world.last_action = Some(format!("transitioning_{}_{}", from, to));
}
```

#### 2. Concurrent State Updates

**File:** `test-harness/bdd/src/steps/concurrency.rs`

```rust
#[when(expr = "request-A updates state to {string} at T+{int}ms")]
pub async fn when_request_a_updates(world: &mut World, state: String, time: u32) {
    // TEAM-079: Simulate concurrent state update A
    tracing::info!("TEAM-079: Request-A updates to {} at T+{}ms", state, time);
    world.last_action = Some(format!("request_a_{}_{}", state, time));
}

#[when(expr = "request-B updates state to {string} at T+{int}ms")]
pub async fn when_request_b_updates(world: &mut World, state: String, time: u32) {
    // TEAM-079: Simulate concurrent state update B
    tracing::info!("TEAM-079: Request-B updates to {} at T+{}ms", state, time);
    world.last_action = Some(format!("request_b_{}_{}", state, time));
}
```

**Can wire to:**
```rust
#[when(expr = "request-A updates state to {string} at T+{int}ms")]
pub async fn when_request_a_updates(world: &mut World, state: String, time: u32) {
    // TEAM-080: Spawn concurrent state update A
    let registry = world.queen_registry.as_ref().expect("Registry not initialized").inner().clone();
    let new_state = match state.as_str() {
        "idle" => WorkerState::Idle,
        "busy" => WorkerState::Busy,
        "loading" => WorkerState::Loading,
        _ => panic!("Unknown state: {}", state),
    };
    
    let handle = tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(time as u64)).await;
        registry.update_state("worker-001", new_state).await
    });
    
    world.concurrent_handles.push(handle);
    tracing::info!("TEAM-080: Request-A updating to {} at T+{}ms", state, time);
}

#[when(expr = "request-B updates state to {string} at T+{int}ms")]
pub async fn when_request_b_updates(world: &mut World, state: String, time: u32) {
    // TEAM-080: Spawn concurrent state update B
    let registry = world.queen_registry.as_ref().expect("Registry not initialized").inner().clone();
    let new_state = match state.as_str() {
        "idle" => WorkerState::Idle,
        "busy" => WorkerState::Busy,
        "loading" => WorkerState::Loading,
        _ => panic!("Unknown state: {}", state),
    };
    
    let handle = tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(time as u64)).await;
        registry.update_state("worker-001", new_state).await
    });
    
    world.concurrent_handles.push(handle);
    tracing::info!("TEAM-080: Request-B updating to {} at T+{}ms", state, time);
}
```

**Note:** Need to add `concurrent_handles: Vec<tokio::task::JoinHandle<bool>>` to `World` struct.

---

## Category 2: Model Catalog Operations (5 functions)

### Product Code Available

**File:** `/bin/shared-crates/model-catalog/src/lib.rs`

```rust
impl ModelCatalog {
    pub async fn init(&self) -> Result<()>
    pub async fn find_model(&self, reference: &str, provider: &str) -> Result<Option<ModelInfo>>
    pub async fn register_model(&self, model: &ModelInfo) -> Result<()>
    pub async fn remove_model(&self, reference: &str, provider: &str) -> Result<()>
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>>
}
```

### Stub Functions Ready to Wire

#### 1. Concurrent Catalog Registration

**File:** `test-harness/bdd/src/steps/concurrency.rs`

```rust
#[when(expr = "all {int} attempt to register in catalog")]
pub async fn when_concurrent_catalog_register(world: &mut World, count: usize) {
    // TEAM-079: Test concurrent catalog registration
    tracing::info!("TEAM-079: {} instances registering in catalog", count);
    world.last_action = Some(format!("catalog_register_{}", count));
}
```

**Can wire to:**
```rust
#[when(expr = "all {int} attempt to register in catalog")]
pub async fn when_concurrent_catalog_register(world: &mut World, count: usize) {
    // TEAM-080: Test concurrent catalog INSERT with real SQLite
    let catalog_path = world.model_catalog_path.as_ref().expect("Catalog not initialized");
    
    let mut handles = vec![];
    for i in 0..count {
        let path = catalog_path.clone();
        let handle = tokio::spawn(async move {
            let catalog = ModelCatalog::new(path.to_string_lossy().to_string());
            let model = ModelInfo {
                reference: "tinyllama-q4".to_string(),
                provider: "hf".to_string(),
                local_path: format!("/tmp/model_{}.gguf", i),
                size_bytes: 1000000,
                downloaded_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64,
            };
            catalog.register_model(&model).await
        });
        handles.push(handle);
    }
    
    // Collect results
    world.concurrent_results.clear();
    for handle in handles {
        let result = handle.await.unwrap();
        world.concurrent_results.push(result);
    }
    
    tracing::info!("TEAM-080: {} catalog registrations attempted", count);
}
```

**Note:** This tests SQLite's UNIQUE constraint handling on concurrent INSERTs.

---

## Category 3: Download Tracker Operations (4 functions)

### Product Code Available

**File:** `/bin/rbee-hive/src/download_tracker.rs`

```rust
impl DownloadTracker {
    pub async fn start_download(&self) -> String
    pub async fn send_progress(&self, download_id: &str, event: DownloadEvent) -> Result<()>
    pub async fn subscribe(&self, download_id: &str) -> Option<broadcast::Receiver<DownloadEvent>>
    pub async fn complete_download(&self, download_id: &str)
}
```

### Stub Functions Ready to Wire

#### 1. Download Progress Tracking

**File:** `test-harness/bdd/src/steps/concurrency.rs`

```rust
#[given(expr = "{int} rbee-hive instances are downloading {string}")]
pub async fn given_multiple_downloads(world: &mut World, count: usize, model: String) {
    // TEAM-079: Simulate concurrent downloads
    tracing::info!("TEAM-079: {} instances downloading {}", count, model);
    world.last_action = Some(format!("concurrent_downloads_{}_{}", count, model));
}
```

**Can wire to:**
```rust
#[given(expr = "{int} rbee-hive instances are downloading {string}")]
pub async fn given_multiple_downloads(world: &mut World, count: usize, model: String) {
    // TEAM-080: Start multiple concurrent downloads with real DownloadTracker
    use rbee_hive::DownloadTracker;
    
    let tracker = DownloadTracker::new();
    let mut download_ids = vec![];
    
    for _ in 0..count {
        let download_id = tracker.start_download().await;
        download_ids.push(download_id);
    }
    
    // Store tracker and IDs in world for later steps
    world.download_tracker = Some(tracker);
    world.download_ids = download_ids;
    
    tracing::info!("TEAM-080: Started {} concurrent downloads for {}", count, model);
    world.last_action = Some(format!("concurrent_downloads_{}_{}", count, model));
}
```

**Note:** Need to add to `World`:
```rust
pub download_tracker: Option<rbee_hive::DownloadTracker>,
pub download_ids: Vec<String>,
```

#### 2. Download Completion

**File:** `test-harness/bdd/src/steps/concurrency.rs`

```rust
#[when(expr = "all {int} complete download simultaneously")]
pub async fn when_concurrent_download_complete(world: &mut World, count: usize) {
    // TEAM-079: Simulate simultaneous download completion
    tracing::info!("TEAM-079: {} downloads complete simultaneously", count);
    world.last_action = Some(format!("downloads_complete_{}", count));
}
```

**Can wire to:**
```rust
#[when(expr = "all {int} complete download simultaneously")]
pub async fn when_concurrent_download_complete(world: &mut World, count: usize) {
    // TEAM-080: Send completion events to all downloads
    use rbee_hive::DownloadEvent;
    
    let tracker = world.download_tracker.as_ref().expect("Tracker not initialized");
    let download_ids = &world.download_ids;
    
    let mut handles = vec![];
    for download_id in download_ids.iter().take(count) {
        let tracker_clone = tracker.clone(); // Assuming Clone is implemented
        let id = download_id.clone();
        let handle = tokio::spawn(async move {
            tracker_clone.send_progress(&id, DownloadEvent::Complete {
                local_path: format!("/tmp/{}.gguf", id),
            }).await
        });
        handles.push(handle);
    }
    
    // Wait for all completions
    for handle in handles {
        handle.await.unwrap().unwrap();
    }
    
    tracing::info!("TEAM-080: {} downloads completed", count);
    world.last_action = Some(format!("downloads_complete_{}", count));
}
```

---

## Category 4: Failure Recovery (6 functions)

### Product Code Available

**File:** `/bin/queen-rbee/src/worker_registry.rs`

```rust
impl WorkerRegistry {
    pub async fn remove(&self, worker_id: &str) -> bool  // For crash cleanup
    pub async fn get(&self, worker_id: &str) -> Option<WorkerInfo>  // For checking state
}
```

### Stub Functions Ready to Wire

#### 1. Worker Crash Simulation

**File:** `test-harness/bdd/src/steps/failure_recovery.rs`

```rust
#[given(expr = "worker-001 is processing inference request {string}")]
pub async fn given_worker_processing_request(world: &mut World, request_id: String) {
    // TEAM-079: Set up worker processing state
    tracing::info!("TEAM-079: Worker processing request {}", request_id);
    world.last_action = Some(format!("processing_{}", request_id));
}

#[when(expr = "worker-001 crashes unexpectedly")]
pub async fn when_worker_crashes(world: &mut World) {
    // TEAM-079: Simulate worker crash
    tracing::info!("TEAM-079: Worker-001 crashed");
    world.last_action = Some("worker_crashed".to_string());
}
```

**Can wire to:**
```rust
#[given(expr = "worker-001 is processing inference request {string}")]
pub async fn given_worker_processing_request(world: &mut World, request_id: String) {
    // TEAM-080: Register worker in Busy state
    let registry = world.queen_registry.as_ref().expect("Registry not initialized").inner();
    
    let worker = WorkerInfo {
        id: "worker-001".to_string(),
        url: "http://localhost:8081".to_string(),
        model_ref: "test-model".to_string(),
        backend: "cpu".to_string(),
        device: 0,
        state: WorkerState::Busy,  // Processing state
        slots_total: 4,
        slots_available: 3,  // 1 slot in use
        vram_bytes: None,
        node_name: "test-node".to_string(),
    };
    registry.register(worker).await;
    
    world.active_request_id = Some(request_id.clone());
    tracing::info!("TEAM-080: Worker-001 processing request {}", request_id);
}

#[when(expr = "worker-001 crashes unexpectedly")]
pub async fn when_worker_crashes(world: &mut World) {
    // TEAM-080: Remove worker from registry (simulates crash detection)
    let registry = world.queen_registry.as_ref().expect("Registry not initialized").inner();
    
    let removed = registry.remove("worker-001").await;
    assert!(removed, "Worker should exist before crash");
    
    tracing::info!("TEAM-080: Worker-001 crashed and removed from registry");
    world.last_action = Some("worker_crashed".to_string());
}
```

#### 2. Backup Worker Availability

**File:** `test-harness/bdd/src/steps/failure_recovery.rs`

```rust
#[given(expr = "worker-002 is available with same model")]
pub async fn given_worker_002_available(world: &mut World) {
    // TEAM-079: Set up backup worker
    tracing::info!("TEAM-079: Worker-002 available");
    world.last_action = Some("worker_002_available".to_string());
}
```

**Can wire to:**
```rust
#[given(expr = "worker-002 is available with same model")]
pub async fn given_worker_002_available(world: &mut World) {
    // TEAM-080: Register backup worker with same model
    let registry = world.queen_registry.as_ref().expect("Registry not initialized").inner();
    
    let worker = WorkerInfo {
        id: "worker-002".to_string(),
        url: "http://localhost:8082".to_string(),
        model_ref: "test-model".to_string(),  // Same model as worker-001
        backend: "cpu".to_string(),
        device: 0,
        state: WorkerState::Idle,  // Available for work
        slots_total: 4,
        slots_available: 4,
        vram_bytes: None,
        node_name: "test-node".to_string(),
    };
    registry.register(worker).await;
    
    tracing::info!("TEAM-080: Worker-002 available with same model");
    world.last_action = Some("worker_002_available".to_string());
}
```

---

## Category 5: BeehiveRegistry Operations (5 functions)

### Product Code Available

**File:** `/bin/queen-rbee/src/beehive_registry.rs`

```rust
impl BeehiveRegistry {
    pub async fn add_node(&self, node: BeehiveNode) -> Result<()>
    pub async fn get_node(&self, node_name: &str) -> Result<Option<BeehiveNode>>
    pub async fn list_nodes(&self) -> Result<Vec<BeehiveNode>>
    pub async fn remove_node(&self, node_name: &str) -> Result<bool>
    pub async fn update_status(&self, node_name: &str, status: &str, last_seen: i64) -> Result<()>
}
```

### Stub Functions Ready to Wire

#### 1. Multiple rbee-hive Instances

**File:** `test-harness/bdd/src/steps/failure_recovery.rs`

```rust
#[given(expr = "{int} workers are running")]
pub async fn given_workers_running(world: &mut World, count: usize) {
    // TEAM-079: Set up multiple workers
    tracing::info!("TEAM-079: {} workers running", count);
    world.last_action = Some(format!("workers_running_{}", count));
}
```

**Can wire to:**
```rust
#[given(expr = "{int} workers are running")]
pub async fn given_workers_running(world: &mut World, count: usize) {
    // TEAM-080: Register multiple workers in queen-rbee registry
    let registry = world.queen_registry.as_ref().expect("Registry not initialized").inner();
    
    for i in 0..count {
        let worker = WorkerInfo {
            id: format!("worker-{:03}", i + 1),
            url: format!("http://localhost:{}", 8081 + i),
            model_ref: "test-model".to_string(),
            backend: "cpu".to_string(),
            device: 0,
            state: WorkerState::Idle,
            slots_total: 4,
            slots_available: 4,
            vram_bytes: None,
            node_name: format!("node-{}", i + 1),
        };
        registry.register(worker).await;
    }
    
    tracing::info!("TEAM-080: {} workers registered and running", count);
    world.last_action = Some(format!("workers_running_{}", count));
}
```

---

## Summary Table

| Category | Stub Functions | Product Code Ready | Effort |
|----------|----------------|-------------------|--------|
| WorkerRegistry | 8 | ✅ Yes | 2-3 hours |
| ModelCatalog | 5 | ✅ Yes | 1-2 hours |
| DownloadTracker | 4 | ✅ Yes | 1-2 hours |
| Failure Recovery | 6 | ✅ Yes | 2 hours |
| BeehiveRegistry | 5 | ✅ Yes | 1 hour |
| **TOTAL** | **28** | **✅ All Ready** | **7-10 hours** |

---

## World Struct Updates Needed

To support the new wiring, add these fields to `World`:

```rust
// In test-harness/bdd/src/steps/world.rs

pub struct World {
    // ... existing fields ...
    
    // TEAM-080: For concurrent operations
    pub concurrent_handles: Vec<tokio::task::JoinHandle<bool>>,
    
    // TEAM-080: For download tracking
    pub download_tracker: Option<Arc<DownloadTracker>>,
    pub download_ids: Vec<String>,
    
    // TEAM-080: For request tracking
    pub active_request_id: Option<String>,
}
```

---

## Implementation Priority

### Phase 1: High Value (4 hours)
1. **WorkerRegistry state transitions** (2 hours)
   - Concurrent state updates
   - State transition testing
   - Most scenarios depend on this

2. **Failure recovery basics** (2 hours)
   - Worker crash simulation
   - Backup worker setup
   - Critical for Gap-F1

### Phase 2: Medium Value (3 hours)
3. **DownloadTracker** (2 hours)
   - Concurrent download tracking
   - Progress events
   - Needed for Gap-C5

4. **ModelCatalog concurrency** (1 hour)
   - Concurrent INSERT testing
   - Needed for Gap-C3

### Phase 3: Low Value (3 hours)
5. **BeehiveRegistry** (1 hour)
   - Multi-node setup
   - Nice to have

6. **Remaining stubs** (2 hours)
   - Edge cases
   - Less critical scenarios

---

## Next Steps

1. **Update World struct** with new fields (30 minutes)
2. **Wire Phase 1 functions** (4 hours)
3. **Test compilation** (30 minutes)
4. **Run BDD tests** to verify wiring (1 hour)
5. **Document progress** (30 minutes)

**Total: ~6.5 hours to wire high-value functions**

---

## Key Insight

**The product code is MORE COMPLETE than the BDD tests!**

- queen-rbee WorkerRegistry: ✅ Fully implemented
- model-catalog: ✅ Fully implemented
- DownloadTracker: ✅ Fully implemented
- BeehiveRegistry: ✅ Fully implemented

**The gap is in the TEST WIRING, not the product code.**

This is actually GOOD - it means we can wire tests quickly without waiting for product development.

---

**Created by:** TEAM-080  
**Date:** 2025-10-11  
**Status:** Ready for implementation  
**Estimated effort:** 7-10 hours for all 28 functions
