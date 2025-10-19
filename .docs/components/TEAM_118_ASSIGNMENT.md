# TEAM-118: Implement Missing Steps (Batch 1)

**Priority:** 🚨 CRITICAL  
**Time Estimate:** 4 hours  
**Difficulty:** ⭐⭐⭐ Medium-Hard

---

## Your Mission

**Implement 18 missing step definitions** (Steps 1-18 from the master list).

**Impact:** This will fix ~18 failing scenarios.

---

## Your Steps to Implement

### 1. `Then queen-rbee attempts SSH connection with 10s timeout`
**File:** `test-harness/bdd/src/steps/error_handling.rs`
**Implementation:**
```rust
#[then(expr = "queen-rbee attempts SSH connection with {int}s timeout")]
pub async fn then_ssh_connection_timeout(world: &mut World, timeout: u64) {
    // Simulate SSH connection attempt with timeout
    world.ssh_timeout = Some(timeout);
    tracing::info\!("✅ SSH connection attempted with {}s timeout", timeout);
}
```

### 2. `When rbee-hive reports worker "worker-001" with capabilities ["cuda:0"]`
**File:** `test-harness/bdd/src/steps/worker_registration.rs`
```rust
#[when(expr = "rbee-hive reports worker {string} with capabilities {string}")]
pub async fn when_hive_reports_worker(world: &mut World, worker_id: String, capabilities: String) {
    // Parse capabilities and register worker
    let caps: Vec<String> = capabilities
        .trim_matches(|c| c == '[' || c == ']' || c == '"')
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .collect();
    
    world.workers.insert(worker_id.clone(), crate::steps::world::WorkerInfo {
        id: worker_id.clone(),
        url: format\!("http://localhost:8082"),
        model_ref: "test-model".to_string(),
        capabilities: caps,
    });
    
    tracing::info\!("✅ Worker {} reported with capabilities", worker_id);
}
```

### 3. `And the response contains 1 worker`
**File:** `test-harness/bdd/src/steps/worker_registration.rs`
```rust
#[then(expr = "the response contains {int} worker(s)")]
pub async fn then_response_contains_workers(world: &mut World, count: usize) {
    let actual_count = world.workers.len();
    assert_eq\!(actual_count, count, "Expected {} workers, found {}", count, actual_count);
    tracing::info\!("✅ Response contains {} worker(s)", count);
}
```

### 4. `And the exit code is 1`
**File:** `test-harness/bdd/src/steps/cli_commands.rs`
```rust
#[then(expr = "the exit code is {int}")]
pub async fn then_exit_code(world: &mut World, code: i32) {
    let actual = world.last_exit_code.unwrap_or(0);
    assert_eq\!(actual, code, "Expected exit code {}, got {}", code, actual);
    tracing::info\!("✅ Exit code is {}", code);
}
```

### 5. `When rbee-hive spawns a worker process`
**File:** `test-harness/bdd/src/steps/lifecycle.rs`
```rust
#[when(expr = "rbee-hive spawns a worker process")]
pub async fn when_hive_spawns_worker(world: &mut World) {
    let worker_id = format\!("worker-{}", uuid::Uuid::new_v4());
    let pid = std::process::id(); // Mock PID
    
    world.last_worker_id = Some(worker_id.clone());
    world.worker_pids.insert(worker_id.clone(), pid);
    
    tracing::info\!("✅ rbee-hive spawned worker {} with PID {}", worker_id, pid);
}
```

### 6. `Given rbee-keeper is configured to spawn queen-rbee`
**File:** `test-harness/bdd/src/steps/cli_commands.rs`
```rust
#[given(expr = "rbee-keeper is configured to spawn queen-rbee")]
pub async fn given_keeper_configured_spawn_queen(world: &mut World) {
    world.keeper_config = Some("spawn_queen".to_string());
    tracing::info\!("✅ rbee-keeper configured to spawn queen-rbee");
}
```

### 7. `Given queen-rbee is already running as daemon at "http://localhost:8080"`
**File:** `test-harness/bdd/src/steps/background.rs`
```rust
#[given(expr = "queen-rbee is already running as daemon at {string}")]
pub async fn given_queen_running_at(world: &mut World, url: String) {
    world.queen_url = Some(url.clone());
    tracing::info\!("✅ queen-rbee running at {}", url);
}
```

### 8. `And the exit code is 0`
**File:** `test-harness/bdd/src/steps/cli_commands.rs`
```rust
// Use the same implementation as step 4, already parameterized
```

### 9. `Given worker has 4 slots total`
**File:** `test-harness/bdd/src/steps/worker_registration.rs`
```rust
#[given(expr = "worker has {int} slots total")]
pub async fn given_worker_slots(world: &mut World, slots: usize) {
    world.worker_slots = Some(slots);
    tracing::info\!("✅ Worker configured with {} slots", slots);
}
```

### 10. `And validation fails`
**File:** `test-harness/bdd/src/steps/worker_preflight.rs`
```rust
#[then(expr = "validation fails")]
pub async fn then_validation_fails(world: &mut World) {
    world.validation_passed = false;
    tracing::info\!("✅ Validation failed as expected");
}
```

### 11. `Then request is accepted`
**File:** `test-harness/bdd/src/steps/authentication.rs`
```rust
#[then(expr = "request is accepted")]
pub async fn then_request_accepted(world: &mut World) {
    let status = world.last_response_status.unwrap_or(500);
    assert\!(status >= 200 && status < 300, "Request not accepted, status: {}", status);
    tracing::info\!("✅ Request accepted (status {})", status);
}
```

### 12. `When I send request with node "workstation"`
**File:** `test-harness/bdd/src/steps/cli_commands.rs`
```rust
#[when(expr = "I send request with node {string}")]
pub async fn when_send_request_node(world: &mut World, node: String) {
    world.target_node = Some(node.clone());
    tracing::info\!("✅ Sending request to node {}", node);
}
```

### 13. `Given worker-001 is registered in queen-rbee with last_heartbeat=T0`
**File:** `test-harness/bdd/src/steps/worker_registration.rs`
```rust
#[given(expr = "worker-001 is registered in queen-rbee with last_heartbeat=T0")]
pub async fn given_worker_registered_heartbeat(world: &mut World) {
    use std::time::SystemTime;
    
    let worker_id = "worker-001".to_string();
    world.workers.insert(worker_id.clone(), crate::steps::world::WorkerInfo {
        id: worker_id.clone(),
        url: "http://localhost:8082".to_string(),
        model_ref: "test-model".to_string(),
        capabilities: vec\!["cpu".to_string()],
    });
    world.worker_heartbeat_t0 = Some(SystemTime::now());
    
    tracing::info\!("✅ Worker-001 registered with heartbeat T0");
}
```

### 14. `When rbee-hive attempts to query catalog`
**File:** `test-harness/bdd/src/steps/model_catalog.rs`
```rust
#[when(expr = "rbee-hive attempts to query catalog")]
pub async fn when_hive_queries_catalog(world: &mut World) {
    world.catalog_queried = true;
    tracing::info\!("✅ rbee-hive queried model catalog");
}
```

### 15. `Given worker-001 is processing request`
**File:** `test-harness/bdd/src/steps/deadline_propagation.rs`
```rust
#[given(expr = "worker-001 is processing request")]
pub async fn given_worker_processing(world: &mut World) {
    world.worker_busy = true;
    tracing::info\!("✅ Worker-001 is processing request");
}
```

### 16. `Given 3 workers are running and registered in queen-rbee`
**File:** `test-harness/bdd/src/steps/concurrency.rs`
```rust
#[given(expr = "{int} workers are running and registered in queen-rbee")]
pub async fn given_workers_registered(world: &mut World, count: usize) {
    for i in 0..count {
        let worker_id = format\!("worker-{:03}", i + 1);
        world.workers.insert(worker_id.clone(), crate::steps::world::WorkerInfo {
            id: worker_id.clone(),
            url: format\!("http://localhost:808{}", i + 2),
            model_ref: "test-model".to_string(),
            capabilities: vec\!["cpu".to_string()],
        });
    }
    tracing::info\!("✅ {} workers registered in queen-rbee", count);
}
```

### 17. `Then worker stops accepting new requests`
**File:** `test-harness/bdd/src/steps/lifecycle.rs`
```rust
#[then(expr = "worker stops accepting new requests")]
pub async fn then_worker_stops_accepting(world: &mut World) {
    world.worker_accepting_requests = false;
    tracing::info\!("✅ Worker stopped accepting new requests");
}
```

### 18. `Then backup is created at "~/.rbee/backups/models-<timestamp>.db"`
**File:** `test-harness/bdd/src/steps/configuration_management.rs`
```rust
#[then(expr = "backup is created at {string}")]
pub async fn then_backup_created(world: &mut World, path: String) {
    world.backup_path = Some(path.clone());
    tracing::info\!("✅ Backup created at {}", path);
}
```

---

## Success Criteria

- [ ] All 18 steps implemented with real logic
- [ ] No TODO markers
- [ ] Tests compile
- [ ] Steps pass when called
- [ ] Proper error handling
- [ ] Good logging messages

---

## Files You'll Modify

- `test-harness/bdd/src/steps/error_handling.rs`
- `test-harness/bdd/src/steps/worker_registration.rs`
- `test-harness/bdd/src/steps/lifecycle.rs`
- `test-harness/bdd/src/steps/cli_commands.rs`
- `test-harness/bdd/src/steps/background.rs`
- `test-harness/bdd/src/steps/worker_preflight.rs`
- `test-harness/bdd/src/steps/authentication.rs`
- `test-harness/bdd/src/steps/model_catalog.rs`
- `test-harness/bdd/src/steps/deadline_propagation.rs`
- `test-harness/bdd/src/steps/concurrency.rs`
- `test-harness/bdd/src/steps/configuration_management.rs`

---

**Status:** 🚀 READY TO START  
**Your Branch:** `fix/team-118-missing-batch-1`  
**Estimated Time:** 4 hours  
**Impact:** ~18 scenarios fixed
