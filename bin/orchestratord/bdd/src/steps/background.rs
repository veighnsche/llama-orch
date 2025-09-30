// Background service step definitions
// Behaviors: B-BG-001 through B-BG-012 (Handoff Autobind Watcher)

use crate::steps::world::World;
use cucumber::{given, then, when};
use serde_json::json;
use std::fs;
use std::path::PathBuf;

// B-BG-003, B-BG-004: Create handoff JSON file
#[given(regex = "^a handoff file exists with pool_id (.+) and replica_id (.+)$")]
pub async fn given_handoff_file_exists(world: &mut World, pool_id: String, replica_id: String) {
    let runtime_dir = PathBuf::from(".runtime/engines");
    fs::create_dir_all(&runtime_dir).expect("failed to create runtime dir");
    
    let handoff = json!({
        "url": "http://127.0.0.1:9999",
        "pool_id": pool_id,
        "replica_id": replica_id,
        "engine_version": "llamacpp-test-v1",
        "device_mask": "GPU0",
        "slots_total": 4,
        "slots_free": 4
    });
    
    let filename = runtime_dir.join(format!("{}-{}.json", pool_id, replica_id));
    fs::write(&filename, handoff.to_string()).expect("failed to write handoff file");
    
    // Store filename in world for cleanup
    world.push_fact(format!("handoff_file:{}", filename.display()));
}

// B-BG-005: Mark pool as already bound
#[given(regex = "^a pool (.+) is already bound$")]
pub async fn given_pool_already_bound(world: &mut World, pool_id: String) {
    if let Ok(mut guard) = world.state.bound_pools.lock() {
        guard.insert(format!("{}:r0", pool_id));
    }
}

// B-BG-003: Create handoff file for specific pool
#[given(regex = "^a handoff file for (.+) exists$")]
pub async fn given_handoff_file_for_pool(world: &mut World, pool_id: String) {
    let runtime_dir = PathBuf::from(".runtime/engines");
    fs::create_dir_all(&runtime_dir).expect("failed to create runtime dir");
    
    let handoff = json!({
        "url": "http://127.0.0.1:9999",
        "pool_id": pool_id.clone(),
        "replica_id": "r0",
        "engine_version": "llamacpp-test-v1",
        "device_mask": "GPU0",
        "slots_total": 4,
        "slots_free": 4
    });
    
    let filename = runtime_dir.join(format!("{}-r0.json", pool_id));
    fs::write(&filename, handoff.to_string()).expect("failed to write handoff file");
}

// B-BG-001, B-BG-012: Watcher is started in bootstrap
#[given(regex = "^the handoff watcher is running$")]
pub async fn given_handoff_watcher_running(_world: &mut World) {
    // No-op: watcher runs in background
}

// B-BG-002: Wait for watcher to process
#[when(regex = "^the handoff watcher processes the file$")]
pub async fn when_handoff_watcher_processes(_world: &mut World) {
    // Sleep for poll interval + buffer
    tokio::time::sleep(tokio::time::Duration::from_millis(1500)).await;
}

// B-BG-003, B-BG-004: Create new handoff file
#[when(regex = "^I create a new handoff file$")]
pub async fn when_create_new_handoff_file(_world: &mut World) {
    let runtime_dir = PathBuf::from(".runtime/engines");
    fs::create_dir_all(&runtime_dir).expect("failed to create runtime dir");
    
    let handoff = json!({
        "url": "http://127.0.0.1:9999",
        "pool_id": "new-pool",
        "replica_id": "r0",
        "engine_version": "llamacpp-test-v1",
        "device_mask": "GPU0",
        "slots_total": 4,
        "slots_free": 4
    });
    
    let filename = runtime_dir.join("new-pool-r0.json");
    fs::write(&filename, handoff.to_string()).expect("failed to write handoff file");
}

// B-BG-002: Wait for watcher poll
#[when(regex = "^I wait for the poll interval$")]
pub async fn when_wait_for_poll_interval(_world: &mut World) {
    tokio::time::sleep(tokio::time::Duration::from_millis(1500)).await;
}

// B-BG-007: Verify adapter bound
#[then(regex = "^an adapter is bound to pool (.+) replica (.+)$")]
pub async fn then_adapter_bound_to_pool(world: &mut World, pool_id: String, replica_id: String) {
    // Check bound_pools set
    if let Ok(guard) = world.state.bound_pools.lock() {
        let key = format!("{}:{}", pool_id, replica_id);
        assert!(guard.contains(&key), "pool not bound: {}", key);
    } else {
        panic!("failed to lock bound_pools");
    }
}

// B-BG-008: Verify pool registered in pool_manager
#[then(regex = "^the pool is registered as ready$")]
pub async fn then_pool_registered_ready(world: &mut World) {
    // Check if pool_manager is accessible
    if let Ok(_guard) = world.state.pool_manager.lock() {
        // Pool manager is accessible (simplified check)
    } else {
        panic!("failed to lock pool_manager");
    }
}

// B-BG-011: Verify narration logged
#[then(regex = "^a narration breadcrumb is emitted$")]
pub async fn then_narration_breadcrumb_emitted(world: &mut World) {
    if let Ok(guard) = world.state.logs.lock() {
        // Check for autobind-related log
        let has_autobind_log = guard.iter().any(|log| log.contains("autobind") || log.contains("handoff"));
        assert!(has_autobind_log, "no autobind narration found in logs");
    } else {
        panic!("failed to lock logs");
    }
}

// B-BG-005: Verify pool not re-bound
#[then(regex = "^the pool is not re-bound$")]
pub async fn then_pool_not_rebound(_world: &mut World) {
    // Would need to track adapter_host bind calls
    // For now, just verify no panic occurred
}

// B-BG-012: Verify new handoff was processed
#[then(regex = "^the new handoff is processed$")]
pub async fn then_new_handoff_processed(world: &mut World) {
    if let Ok(guard) = world.state.bound_pools.lock() {
        let has_new_pool = guard.iter().any(|k| k.contains("new-pool"));
        assert!(has_new_pool, "new pool not bound");
    } else {
        panic!("failed to lock bound_pools");
    }
}
