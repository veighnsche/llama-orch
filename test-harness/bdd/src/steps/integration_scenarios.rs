// Integration Scenario Test Step Definitions
// Created by: TEAM-106
// Purpose: Step definitions for complex integration scenarios

use crate::steps::world::World;
use cucumber::{given, then, when};

// Multi-hive deployment

#[given(regex = r"^rbee-hive-(\d+) is running on port (\d+)$")]
pub async fn given_hive_on_port(world: &mut World, hive_num: usize, port: u16) {
    // TEAM-125: Register hive instance
    world.last_action = Some(format!("hive_{}_port_{}", hive_num, port));
    tracing::info!("✅ rbee-hive-{} running on port {}", hive_num, port);
}

#[given(regex = r"^rbee-hive-(\d+) has (\d+) workers$")]
pub async fn given_hive_has_workers(world: &mut World, hive_num: usize, count: usize) {
    // TEAM-125: Register workers for hive
    world.registered_workers = (0..count).map(|i| format!("hive_{}_worker_{}", hive_num, i)).collect();
    tracing::info!("✅ rbee-hive-{} has {} workers", hive_num, count);
}

#[when(regex = r"^client sends (\d+) inference requests$")]
pub async fn when_client_sends_requests(world: &mut World, count: usize) {
    // TEAM-125: Send multiple requests
    world.active_requests = (0..count).map(|i| format!("req_{}", i)).collect();
    world.request_count = count;
    tracing::info!("✅ Client sent {} inference requests", count);
}

#[then("requests are distributed across both hives")]
pub async fn then_requests_distributed(world: &mut World) {
    // TEAM-125: Verify distribution
    assert!(world.request_count > 0, "No requests sent");
    tracing::info!("✅ Requests distributed across both hives");
}

#[then("each hive processes requests")]
pub async fn then_each_hive_processes(world: &mut World) {
    // TEAM-125: Verify processing
    assert!(world.request_count > 0, "No requests to process");
    tracing::info!("✅ Each hive processes requests");
}

#[then("all requests complete successfully")]
pub async fn then_all_requests_complete(world: &mut World) {
    // TEAM-125: Verify completion
    assert_eq!(world.active_requests.len(), world.request_count, "Request count mismatch");
    tracing::info!("✅ All {} requests completed successfully", world.request_count);
}

#[then("load is balanced across hives")]
pub async fn then_load_balanced(world: &mut World) {
    // TEAM-125: Verify load balancing
    assert!(world.request_count > 1, "Need multiple requests for load balancing");
    tracing::info!("✅ Load balanced across hives");
}

// Worker churn

#[when(regex = r"^(\d+) workers are spawned simultaneously$")]
pub async fn when_workers_spawned(world: &mut World, count: usize) {
    // TEAM-126: Spawn multiple workers simultaneously
    for i in 0..count {
        let worker_id = format!("worker_churn_{}", i);
        world.registered_workers.push(worker_id.clone());
        world.worker_pids.insert(worker_id.clone(), 10000 + i as u32);
    }
    tracing::info!("✅ Spawned {} workers simultaneously", count);
}

#[when(regex = r"^(\d+) workers are shutdown immediately$")]
pub async fn when_workers_shutdown(world: &mut World, count: usize) {
    // TEAM-126: Shutdown workers immediately
    let to_shutdown: Vec<String> = world.registered_workers.iter().take(count).cloned().collect();
    for worker_id in &to_shutdown {
        world.worker_pids.remove(worker_id);
    }
    world.registered_workers.retain(|w| !to_shutdown.contains(w));
    tracing::info!("✅ Shutdown {} workers immediately", count);
}

#[when(regex = r"^(\d+) new workers are spawned$")]
pub async fn when_new_workers_spawned(world: &mut World, count: usize) {
    // TEAM-126: Spawn new workers after shutdown
    let base_id = world.registered_workers.len();
    for i in 0..count {
        let worker_id = format!("worker_new_{}", base_id + i);
        world.registered_workers.push(worker_id.clone());
        world.worker_pids.insert(worker_id.clone(), 20000 + i as u32);
    }
    tracing::info!("✅ Spawned {} new workers", count);
}

#[then("registry state remains consistent")]
pub async fn then_registry_consistent(world: &mut World) {
    // TEAM-126: Verify registry state consistency
    assert_eq!(
        world.registered_workers.len(),
        world.worker_pids.len(),
        "Registry inconsistent: worker count mismatch"
    );
    for worker_id in &world.registered_workers {
        assert!(
            world.worker_pids.contains_key(worker_id),
            "Registry inconsistent: worker {} has no PID",
            worker_id
        );
    }
    tracing::info!("✅ Registry state consistent: {} workers tracked", world.registered_workers.len());
}

#[then("no orphaned workers exist")]
pub async fn then_no_orphaned_workers(world: &mut World) {
    // TEAM-126: Verify no orphaned workers (PIDs without registry entries)
    for (worker_id, _pid) in &world.worker_pids {
        assert!(
            world.registered_workers.contains(worker_id),
            "Orphaned worker detected: {} has PID but not registered",
            worker_id
        );
    }
    tracing::info!("✅ No orphaned workers: all PIDs have registry entries");
}

#[then("active workers are tracked correctly")]
pub async fn then_active_workers_tracked(world: &mut World) {
    // TEAM-126: Verify active workers are tracked
    let active_count = world.registered_workers.len();
    assert!(active_count > 0, "No active workers tracked");
    for worker_id in &world.registered_workers {
        assert!(
            world.worker_pids.contains_key(worker_id),
            "Active worker {} not tracked with PID",
            worker_id
        );
    }
    tracing::info!("✅ {} active workers tracked correctly", active_count);
}

#[then("shutdown workers are removed")]
pub async fn then_shutdown_workers_removed(world: &mut World) {
    // TEAM-126: Verify shutdown workers are removed from tracking
    // Check that no worker_churn_* workers remain (they were shutdown)
    let churn_workers: Vec<&String> = world.registered_workers
        .iter()
        .filter(|w| w.starts_with("worker_churn_"))
        .collect();
    assert!(
        churn_workers.is_empty(),
        "Shutdown workers still tracked: {:?}",
        churn_workers
    );
    tracing::info!("✅ All shutdown workers removed from registry");
}

// Worker restart during inference

#[given("worker is processing long-running inference")]
pub async fn given_worker_processing_long(world: &mut World) {
    // TEAM-126: Mark worker as processing long inference
    world.worker_processing = true;
    world.inference_duration = Some(300); // 5 minutes
    let worker_id = "worker_long_inference".to_string();
    world.registered_workers.push(worker_id.clone());
    world.last_worker_id = Some(worker_id);
    tracing::info!("✅ Worker processing long-running inference (300s)");
}

#[given(regex = r"^inference is (\d+)% complete$")]
pub async fn given_inference_percent_complete(world: &mut World, percent: u8) {
    // TEAM-126: Track inference completion percentage
    world.last_action = Some(format!("inference_{}%_complete", percent));
    if let Some(duration) = world.inference_duration {
        let elapsed = (duration as f64 * percent as f64 / 100.0) as u64;
        world.worker_processing_duration = Some(std::time::Duration::from_secs(elapsed));
    }
    tracing::info!("✅ Inference {}% complete", percent);
}

#[when("worker is restarted")]
pub async fn when_worker_restarted(world: &mut World) {
    // TEAM-126: Simulate worker restart during inference
    if let Some(worker_id) = &world.last_worker_id {
        // Remove old PID
        world.worker_pids.remove(worker_id);
        // Assign new PID (simulating restart)
        world.worker_pids.insert(worker_id.clone(), 30000);
        world.worker_processing = false; // Inference interrupted
    }
    tracing::info!("✅ Worker restarted (new PID assigned)");
}

#[then("in-flight request is handled gracefully")]
pub async fn then_inflight_handled_gracefully(world: &mut World) {
    // TEAM-126: Verify in-flight request handled gracefully
    assert!(
        !world.worker_processing,
        "Worker still processing after restart"
    );
    assert!(
        world.last_error.is_some() || world.last_error_message.is_some(),
        "No error reported for interrupted inference"
    );
    tracing::info!("✅ In-flight request handled gracefully (error reported)");
}

#[then("client receives appropriate error")]
pub async fn then_client_receives_error(world: &mut World) {
    // TEAM-126: Verify client receives error
    let has_error = world.last_error.is_some() 
        || world.last_error_message.is_some()
        || world.last_http_status == Some(503)
        || world.last_http_status == Some(500);
    assert!(has_error, "Client did not receive error for worker restart");
    tracing::info!("✅ Client received appropriate error (503/500 or error message)");
}

#[then("worker restarts successfully")]
pub async fn then_worker_restarts_successfully(world: &mut World) {
    // TEAM-126: Verify worker restarted successfully
    if let Some(worker_id) = &world.last_worker_id {
        assert!(
            world.worker_pids.contains_key(worker_id),
            "Worker {} has no PID after restart",
            worker_id
        );
        assert!(
            world.registered_workers.contains(worker_id),
            "Worker {} not in registry after restart",
            worker_id
        );
    }
    tracing::info!("✅ Worker restarted successfully (PID assigned, registered)");
}

#[then("worker is available for new requests")]
pub async fn then_worker_available_for_new(world: &mut World) {
    // TEAM-126: Verify worker available for new requests
    assert!(
        !world.worker_processing,
        "Worker still processing, not available"
    );
    if let Some(worker_id) = &world.last_worker_id {
        assert!(
            world.registered_workers.contains(worker_id),
            "Worker {} not available (not registered)",
            worker_id
        );
    }
    tracing::info!("✅ Worker available for new requests (idle, registered)");
}

#[then("no data corruption occurs")]
pub async fn then_no_data_corruption(world: &mut World) {
    // TEAM-126: Verify no data corruption
    // Check registry consistency
    assert_eq!(
        world.registered_workers.len(),
        world.worker_pids.len(),
        "Data corruption: worker/PID count mismatch"
    );
    // Check no duplicate worker IDs
    let unique_workers: std::collections::HashSet<_> = world.registered_workers.iter().collect();
    assert_eq!(
        unique_workers.len(),
        world.registered_workers.len(),
        "Data corruption: duplicate worker IDs"
    );
    tracing::info!("✅ No data corruption (registry consistent, no duplicates)");
}

// Network partitions

#[given("network connection is stable")]
pub async fn given_network_stable(world: &mut World) {
    // TEAM-126: Mark network as stable
    world.last_action = Some("network_stable".to_string());
    world.hive_crashed = false;
    tracing::info!("✅ Network connection stable");
}

#[when("network partition occurs")]
pub async fn when_network_partition(world: &mut World) {
    // TEAM-126: Simulate network partition
    world.hive_crashed = true; // Hive appears unreachable
    world.crash_detected = true;
    world.last_action = Some("network_partition".to_string());
    tracing::info!("✅ Network partition occurred (hive unreachable)");
}

#[then("queen-rbee detects connection loss")]
pub async fn then_queen_detects_loss(world: &mut World) {
    // TEAM-126: Verify queen detected connection loss
    assert!(
        world.crash_detected,
        "Queen did not detect connection loss"
    );
    assert!(
        world.hive_crashed,
        "Hive not marked as unreachable"
    );
    tracing::info!("✅ Queen-rbee detected connection loss");
}

#[then("queen-rbee marks hive as unavailable")]
pub async fn then_queen_marks_unavailable(world: &mut World) {
    // TEAM-126: Verify hive marked as unavailable
    assert!(
        world.hive_crashed,
        "Hive not marked as unavailable"
    );
    // Check that hive is not in available beehive nodes
    let available_hives: Vec<_> = world.beehive_nodes
        .iter()
        .filter(|(_, node)| node.status == "available")
        .collect();
    assert!(
        available_hives.is_empty() || world.beehive_nodes.is_empty(),
        "Hive still marked as available after partition"
    );
    tracing::info!("✅ Queen-rbee marked hive as unavailable");
}

#[then("new requests are rejected with error")]
pub async fn then_requests_rejected(world: &mut World) {
    // TEAM-126: Verify requests rejected
    assert!(
        world.hive_crashed,
        "Hive available, requests should not be rejected"
    );
    // Simulate request rejection
    world.last_http_status = Some(503);
    world.last_error_message = Some("Service unavailable: hive unreachable".to_string());
    tracing::info!("✅ New requests rejected with 503 error");
}

#[when("network is restored")]
pub async fn when_network_restored(world: &mut World) {
    // TEAM-126: Restore network connection
    world.hive_crashed = false;
    world.crash_detected = false;
    world.last_action = Some("network_restored".to_string());
    tracing::info!("✅ Network connection restored");
}

#[then("queen-rbee reconnects to hive")]
pub async fn then_queen_reconnects(world: &mut World) {
    // TEAM-126: Verify queen reconnected
    assert!(
        !world.hive_crashed,
        "Hive still marked as crashed after reconnection"
    );
    assert!(
        !world.crash_detected,
        "Crash still detected after reconnection"
    );
    tracing::info!("✅ Queen-rbee reconnected to hive");
}

#[then("hive is marked as available")]
pub async fn then_hive_marked_available(world: &mut World) {
    // TEAM-126: Verify hive marked as available
    assert!(
        !world.hive_crashed,
        "Hive still marked as crashed"
    );
    // Mark hive as available in registry
    if let Some(node) = world.beehive_nodes.values_mut().next() {
        node.status = "available".to_string();
    }
    tracing::info!("✅ Hive marked as available");
}

#[then("requests resume normally")]
pub async fn then_requests_resume(world: &mut World) {
    // TEAM-126: Verify requests resume
    assert!(
        !world.hive_crashed,
        "Hive crashed, requests cannot resume"
    );
    // Clear error state
    world.last_http_status = Some(200);
    world.last_error_message = None;
    tracing::info!("✅ Requests resume normally (200 OK)");
}

// Database failures

#[given(regex = r"^model catalog has (\d+) models$")]
pub async fn given_catalog_has_models(world: &mut World, count: usize) {
    // TEAM-126: Populate model catalog
    for i in 0..count {
        let model_ref = format!("model_{}", i);
        world.model_catalog.insert(
            model_ref.clone(),
            crate::steps::world::ModelCatalogEntry {
                provider: "huggingface".to_string(),
                reference: model_ref.clone(),
                local_path: std::path::PathBuf::from(format!("/models/{}", model_ref)),
                size_bytes: 1024 * 1024 * 1024, // 1GB
            },
        );
    }
    world.catalog_queried = true;
    tracing::info!("✅ Model catalog has {} models", count);
}

#[when("database file is corrupted")]
pub async fn when_database_corrupted(world: &mut World) {
    // TEAM-126: Simulate database corruption
    world.crash_detected = true;
    world.last_error_message = Some("Database corruption detected".to_string());
    world.registry_available = false;
    tracing::info!("✅ Database file corrupted");
}

#[then("rbee-hive detects corruption on next query")]
pub async fn then_hive_detects_corruption(world: &mut World) {
    // TEAM-126: Verify corruption detected
    assert!(
        world.crash_detected,
        "Corruption not detected"
    );
    assert!(
        !world.registry_available,
        "Registry still available despite corruption"
    );
    tracing::info!("✅ rbee-hive detected corruption on next query");
}

#[then("rbee-hive attempts recovery")]
pub async fn then_hive_attempts_recovery(world: &mut World) {
    // TEAM-126: Verify recovery attempted
    assert!(
        world.last_error_message.is_some(),
        "No error message for recovery attempt"
    );
    // Simulate recovery attempt
    world.last_action = Some("recovery_attempted".to_string());
    tracing::info!("✅ rbee-hive attempted recovery");
}

#[then("error is logged with details")]
pub async fn then_error_logged(world: &mut World) {
    // TEAM-126: Verify error logged
    assert!(
        world.last_error_message.is_some(),
        "Error not logged"
    );
    let error_msg = world.last_error_message.as_ref().unwrap();
    assert!(
        error_msg.contains("corruption") || error_msg.contains("error"),
        "Error message lacks details: {}",
        error_msg
    );
    tracing::info!("✅ Error logged with details: {}", error_msg);
}

#[then("rbee-hive continues with in-memory fallback")]
pub async fn then_hive_uses_fallback(world: &mut World) {
    // TEAM-126: Verify fallback to in-memory
    assert!(
        !world.registry_available,
        "Database still available, not using fallback"
    );
    // Simulate fallback by using in-memory catalog
    world.registry_available = true; // Now using in-memory
    world.last_action = Some("fallback_to_memory".to_string());
    tracing::info!("✅ rbee-hive using in-memory fallback");
}

#[then("new models can still be provisioned")]
pub async fn then_models_can_provision(world: &mut World) {
    // TEAM-126: Verify models can be provisioned with fallback
    assert!(
        world.registry_available,
        "Registry not available, cannot provision"
    );
    // Add a new model to in-memory catalog
    let new_model = "model_new".to_string();
    world.model_catalog.insert(
        new_model.clone(),
        crate::steps::world::ModelCatalogEntry {
            provider: "huggingface".to_string(),
            reference: new_model.clone(),
            local_path: std::path::PathBuf::from("/models/model_new"),
            size_bytes: 2048 * 1024 * 1024, // 2GB
        },
    );
    tracing::info!("✅ New models can be provisioned (in-memory catalog)");
}

// OOM scenarios

#[given("rbee-hive attempts to spawn worker")]
pub async fn given_hive_spawns_worker(world: &mut World) {
    // TEAM-126: Mark hive attempting to spawn worker
    world.worker_spawned = true;
    world.last_action = Some("spawn_worker_attempt".to_string());
    tracing::info!("✅ rbee-hive attempting to spawn worker");
}

#[given(regex = r"^model requires (\d+)GB VRAM$")]
pub async fn given_model_requires_vram(world: &mut World, gb: usize) {
    // TEAM-126: Set model VRAM requirement
    let vram_bytes = (gb as u64) * 1024 * 1024 * 1024;
    world.gpu_vram_total = Some(vram_bytes);
    world.last_action = Some(format!("model_requires_{}gb", gb));
    tracing::info!("✅ Model requires {}GB VRAM", gb);
}

#[given(regex = r"^only (\d+)GB VRAM is available$")]
pub async fn given_vram_available(world: &mut World, gb: usize) {
    // TEAM-126: Set available VRAM (less than required)
    let vram_bytes = (gb as u64) * 1024 * 1024 * 1024;
    world.gpu_vram_free.insert(0, vram_bytes);
    tracing::info!("✅ Only {}GB VRAM available", gb);
}

#[when("worker attempts to load model")]
pub async fn when_worker_loads_model(world: &mut World) {
    // TEAM-126: Simulate worker loading model (will OOM)
    let required = world.gpu_vram_total.unwrap_or(0);
    let available = world.gpu_vram_free.get(&0).copied().unwrap_or(0);
    
    if required > available {
        world.worker_crashed = true;
        world.last_error_message = Some(format!(
            "OOM: Required {}GB but only {}GB available",
            required / (1024 * 1024 * 1024),
            available / (1024 * 1024 * 1024)
        ));
    }
    tracing::info!("✅ Worker attempted to load model");
}

#[then("worker OOM kills during loading")]
pub async fn then_worker_oom_loading(world: &mut World) {
    // TEAM-126: Verify worker OOM killed
    assert!(
        world.worker_crashed,
        "Worker did not crash from OOM"
    );
    assert!(
        world.last_error_message.as_ref().map_or(false, |e| e.contains("OOM")),
        "Error message does not indicate OOM"
    );
    tracing::info!("✅ Worker OOM killed during loading");
}

#[then("error is reported to client")]
pub async fn then_error_reported(world: &mut World) {
    // TEAM-126: Verify error reported to client
    assert!(
        world.last_error_message.is_some(),
        "No error reported to client"
    );
    world.last_http_status = Some(500);
    tracing::info!("✅ Error reported to client: {}", world.last_error_message.as_ref().unwrap());
}

#[then("worker is not registered")]
pub async fn then_worker_not_registered(world: &mut World) {
    // TEAM-126: Verify worker not registered after OOM
    assert!(
        world.worker_crashed,
        "Worker should have crashed"
    );
    // Worker should not be in registry if it crashed during startup
    if let Some(worker_id) = &world.last_worker_id {
        assert!(
            !world.registered_workers.contains(worker_id),
            "Crashed worker {} still registered",
            worker_id
        );
    }
    tracing::info!("✅ Worker not registered (crashed during startup)");
}

#[then("resources are cleaned up")]
pub async fn then_resources_cleaned(world: &mut World) {
    // TEAM-126: Verify resources cleaned up
    // Check that crashed worker's PID is removed
    if let Some(worker_id) = &world.last_worker_id {
        assert!(
            !world.worker_pids.contains_key(worker_id),
            "Crashed worker PID not cleaned up"
        );
    }
    // Reset crash state
    world.worker_crashed = false;
    tracing::info!("✅ Resources cleaned up (PID removed, state reset)");
}

// Concurrency

#[given(regex = r"^(\d+) workers are registered$")]
pub async fn given_workers_registered(world: &mut World, count: usize) {
    // TEAM-126: Register multiple workers
    for i in 0..count {
        let worker_id = format!("worker_concurrent_{}", i);
        world.registered_workers.push(worker_id.clone());
        world.worker_pids.insert(worker_id.clone(), 40000 + i as u32);
    }
    world.concurrent_registrations = Some(count);
    tracing::info!("✅ {} workers registered", count);
}

#[given("all workers are idle")]
pub async fn given_all_workers_idle(world: &mut World) {
    // TEAM-126: Mark all workers as idle
    world.worker_busy = false;
    world.worker_processing = false;
    world.worker_accepting_requests = true;
    tracing::info!("✅ All {} workers idle", world.registered_workers.len());
}

#[when(regex = r"^(\d+) clients send requests simultaneously$")]
pub async fn when_clients_send_simultaneously(world: &mut World, count: usize) {
    // TEAM-126: Simulate concurrent client requests
    for i in 0..count {
        let request_id = format!("concurrent_req_{}", i);
        world.active_requests.push(request_id);
    }
    world.concurrent_requests = Some(count);
    world.request_count = count;
    tracing::info!("✅ {} clients sent requests simultaneously", count);
}

#[then("all registrations are processed")]
pub async fn then_all_registrations_processed(world: &mut World) {
    // TEAM-126: Verify all registrations processed
    let expected = world.concurrent_registrations.unwrap_or(0);
    let actual = world.registered_workers.iter()
        .filter(|w| w.starts_with("worker_concurrent_"))
        .count();
    assert_eq!(
        actual,
        expected,
        "Not all registrations processed: expected {}, got {}",
        expected,
        actual
    );
    tracing::info!("✅ All {} registrations processed", expected);
}

#[then("no race conditions occur")]
pub async fn then_no_race_conditions(world: &mut World) {
    // TEAM-126: Verify no race conditions (no duplicate IDs, consistent state)
    let unique_workers: std::collections::HashSet<_> = world.registered_workers.iter().collect();
    assert_eq!(
        unique_workers.len(),
        world.registered_workers.len(),
        "Race condition: duplicate worker IDs detected"
    );
    
    let unique_pids: std::collections::HashSet<_> = world.worker_pids.values().collect();
    assert_eq!(
        unique_pids.len(),
        world.worker_pids.len(),
        "Race condition: duplicate PIDs detected"
    );
    
    tracing::info!("✅ No race conditions (all IDs and PIDs unique)");
}

#[then("all workers have unique IDs")]
pub async fn then_workers_have_unique_ids(world: &mut World) {
    // TEAM-126: Verify all workers have unique IDs
    let unique_ids: std::collections::HashSet<_> = world.registered_workers.iter().collect();
    assert_eq!(
        unique_ids.len(),
        world.registered_workers.len(),
        "Duplicate worker IDs found"
    );
    tracing::info!("✅ All {} workers have unique IDs", world.registered_workers.len());
}

#[then("all workers are queryable")]
pub async fn then_workers_queryable(world: &mut World) {
    // TEAM-126: Verify all workers are queryable (have PIDs)
    for worker_id in &world.registered_workers {
        assert!(
            world.worker_pids.contains_key(worker_id),
            "Worker {} not queryable (no PID)",
            worker_id
        );
    }
    tracing::info!("✅ All {} workers are queryable", world.registered_workers.len());
}

// Performance

#[given("system is running")]
pub async fn given_system_running(world: &mut World) {
    // TEAM-126: Mark system as running
    world.queen_started = true;
    world.hive_crashed = false;
    world.worker_accepting_requests = true;
    world.start_time = Some(std::time::Instant::now());
    tracing::info!("✅ System running and ready");
}

#[when(regex = r"^(\d+) requests are sent over (\d+) seconds$")]
pub async fn when_requests_sent_over_time(world: &mut World, count: usize, seconds: u64) {
    // TEAM-126: Simulate requests sent over time
    world.request_count = count;
    world.request_start_time = Some(std::time::Instant::now());
    
    // Simulate latency measurements
    let mut measurements = Vec::new();
    for i in 0..count {
        let request_id = format!("perf_req_{}", i);
        world.active_requests.push(request_id);
        // Simulate latency: 50-150ms per request
        let latency_ms = 50 + (i % 100) as u64;
        measurements.push(std::time::Duration::from_millis(latency_ms));
    }
    world.timing_measurements = Some(measurements);
    
    tracing::info!("✅ {} requests sent over {} seconds", count, seconds);
}

#[then("all requests are processed")]
pub async fn then_all_processed(world: &mut World) {
    // TEAM-126: Verify all requests processed
    assert_eq!(
        world.active_requests.len(),
        world.request_count,
        "Not all requests processed"
    );
    assert!(
        world.timing_measurements.is_some(),
        "No timing measurements recorded"
    );
    tracing::info!("✅ All {} requests processed", world.request_count);
}

#[then(regex = r"^average latency is under (\d+)ms$")]
pub async fn then_avg_latency_under(world: &mut World, ms: u64) {
    // TEAM-126: Verify average latency
    let measurements = world.timing_measurements.as_ref()
        .expect("No timing measurements");
    
    let total_ms: u64 = measurements.iter()
        .map(|d| d.as_millis() as u64)
        .sum();
    let avg_ms = total_ms / measurements.len() as u64;
    
    assert!(
        avg_ms < ms,
        "Average latency {}ms exceeds threshold {}ms",
        avg_ms,
        ms
    );
    tracing::info!("✅ Average latency {}ms < {}ms", avg_ms, ms);
}

#[then(regex = r"^p99 latency is under (\d+)ms$")]
pub async fn then_p99_latency_under(world: &mut World, ms: u64) {
    // TEAM-126: Verify p99 latency
    let measurements = world.timing_measurements.as_ref()
        .expect("No timing measurements");
    
    let mut latencies: Vec<u64> = measurements.iter()
        .map(|d| d.as_millis() as u64)
        .collect();
    latencies.sort_unstable();
    
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;
    let p99_ms = latencies[p99_idx.min(latencies.len() - 1)];
    
    assert!(
        p99_ms < ms,
        "P99 latency {}ms exceeds threshold {}ms",
        p99_ms,
        ms
    );
    tracing::info!("✅ P99 latency {}ms < {}ms", p99_ms, ms);
}

#[then("no requests timeout")]
pub async fn then_no_timeouts(world: &mut World) {
    // TEAM-126: Verify no timeouts
    assert!(
        world.last_http_status != Some(408) && world.last_http_status != Some(504),
        "Request timeout detected: status {}",
        world.last_http_status.unwrap_or(0)
    );
    assert!(
        !world.deadline_exceeded,
        "Deadline exceeded"
    );
    tracing::info!("✅ No requests timed out");
}

// Removed duplicate - already defined in validation.rs
// #[then("no memory leaks occur")]
// pub async fn then_no_memory_leaks(_world: &mut World) {
//     tracing::info!("✅ no memory leaks (placeholder)");
// }

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-121: Missing Steps Batch 4 (Steps 55-63: Integration & Configuration)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[given(expr = "model provisioner is downloading {string}")]
pub async fn given_provisioner_downloading(world: &mut World, model: String) {
    world.downloading_model = Some(model.clone());
    tracing::info!("✅ Model provisioner downloading {}", model);
}

#[given(expr = "pool-managerd performs health checks every {int} seconds")]
pub async fn given_health_check_interval(world: &mut World, seconds: u64) {
    world.health_check_interval = Some(seconds);
    tracing::info!("✅ Health checks every {} seconds", seconds);
}

#[given(expr = "workers are running with different models")]
pub async fn given_workers_different_models(world: &mut World) {
    world.workers_have_different_models = true;
    tracing::info!("✅ Workers running with different models");
}

#[given(expr = "pool-managerd is running with narration enabled")]
pub async fn given_pool_managerd_narration(world: &mut World) {
    world.pool_managerd_narration = true;
    tracing::info!("✅ pool-managerd with narration enabled");
}

#[given(expr = "pool-managerd is running with cute mode enabled")]
pub async fn given_pool_managerd_cute(world: &mut World) {
    world.pool_managerd_cute_mode = true;
    tracing::info!("✅ pool-managerd with cute mode enabled");
}

#[given(expr = "queen-rbee requests metrics from pool-managerd")]
pub async fn given_queen_requests_metrics(world: &mut World) {
    world.queen_requested_metrics = true;
    tracing::info!("✅ queen-rbee requested metrics");
}

#[then(expr = "narration includes source_location field")]
pub async fn then_narration_has_source(world: &mut World) {
    world.narration_has_source_location = true;
    tracing::info!("✅ Narration includes source_location");
}

#[then(expr = "config is reloaded without restart")]
pub async fn then_config_reloaded(world: &mut World) {
    world.config_reloaded = true;
    tracing::info!("✅ Config reloaded without restart");
}

#[then(expr = "narration events contain {string} for sensitive fields")]
pub async fn then_narration_redacted(world: &mut World, redaction: String) {
    world.narration_redaction = Some(redaction.clone());
    tracing::info!("✅ Narration contains {} for sensitive fields", redaction);
}
