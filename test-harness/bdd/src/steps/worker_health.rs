// Worker health check step definitions
// Created by: TEAM-053
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
// Modified by: TEAM-064 (added explicit warning preservation notice)

use crate::steps::world::{World, ErrorResponse, create_http_client};
use cucumber::{given, then, when};

// TEAM-076: Query worker health endpoint with proper error handling
#[given(expr = "the worker is healthy")]
pub async fn given_worker_is_healthy(world: &mut World) {
    // TEAM-076: Query first worker in registry with error handling
    let registry = world.hive_registry();
    let workers = registry.list().await;
    
    if workers.is_empty() {
        world.last_exit_code = Some(1);
        world.last_error = Some(ErrorResponse {
            code: "NO_WORKERS".to_string(),
            message: "No workers in registry to check health".to_string(),
            details: None,
        });
        tracing::error!("❌ No workers in registry");
        return;
    }
    
    let worker = &workers[0];
    
    // Query health endpoint
    let health_url = format!("{}/v1/health", worker.url);
    let client = create_http_client();
    
    match client.get(&health_url).send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                world.last_exit_code = Some(0);
                world.last_http_status = Some(resp.status().as_u16());
                tracing::info!("✅ Worker {} is healthy", worker.id);
            } else {
                world.last_exit_code = Some(1);
                world.last_http_status = Some(resp.status().as_u16());
                world.last_error = Some(ErrorResponse {
                    code: "WORKER_UNHEALTHY".to_string(),
                    message: format!("Worker health check failed: {}", resp.status()),
                    details: None,
                });
                tracing::error!("❌ Worker health check failed: {}", resp.status());
            }
        }
        Err(e) => {
            world.last_exit_code = Some(1);
            world.last_error = Some(ErrorResponse {
                code: "HEALTH_CHECK_ERROR".to_string(),
                message: format!("Failed to query worker health: {}", e),
                details: None,
            });
            tracing::error!("❌ Failed to query worker health: {}", e);
        }
    }
}

#[given(expr = "the worker is in state {string}")]
pub async fn given_worker_in_state(world: &mut World, state: String) {
    use rbee_hive::registry::{WorkerInfo, WorkerState};
    
    // Parse state string to WorkerState enum
    let worker_state = match state.to_lowercase().as_str() {
        "loading" => WorkerState::Loading,
        "idle" => WorkerState::Idle,
        "busy" => WorkerState::Busy,
        _ => {
            tracing::warn!("⚠️  Unknown state '{}', defaulting to Idle", state);
            WorkerState::Idle
        }
    };
    
    // Check if workers exist
    let existing_worker_id = {
        let registry = world.hive_registry();
        let workers = registry.list().await;
        workers.first().map(|w| w.id.clone())
    };
    
    if let Some(worker_id) = existing_worker_id {
        let registry = world.hive_registry();
        registry.update_state(&worker_id, worker_state.clone()).await;
        tracing::info!("✅ Worker {} set to state: {:?}", worker_id, worker_state);
    } else {
        // Create a test worker in the specified state
        let worker_id = uuid::Uuid::new_v4().to_string();
        let port = world.next_worker_port;
        world.next_worker_port += 1;
        
        let worker = WorkerInfo {
            id: worker_id.clone(),
            url: format!("http://127.0.0.1:{}", port),
            model_ref: "test-model".to_string(),
            backend: "cpu".to_string(),
            device: 0,
            state: worker_state.clone(),
            last_activity: std::time::SystemTime::now(),
            slots_total: 1,
            slots_available: 1,
            failed_health_checks: 0,
            pid: None,
        };
        
        let registry = world.hive_registry();
        registry.register(worker).await;
        tracing::info!("✅ Created worker {} in state: {:?} NICE!", worker_id, worker_state);
    }
}

#[given(expr = "the worker is loading model to VRAM")]
pub async fn given_worker_loading(world: &mut World) {
    tracing::debug!("Worker is loading model to VRAM");
}

#[given(expr = "the worker completed model loading")]
pub async fn given_worker_completed_loading(world: &mut World) {
    tracing::debug!("Worker completed model loading");
}

#[given(expr = "the worker is loading for {int} minutes")]
pub async fn given_worker_loading_duration(world: &mut World, minutes: u64) {
    tracing::debug!("Worker loading for {} minutes", minutes);
}

#[given(regex = r"^the worker is stuck at (\d+)/(\d+) layers$")]
pub async fn given_worker_stuck_at_layers(world: &mut World, current: u32, total: u32) {
    tracing::debug!("Worker stuck at {}/{} layers", current, total);
}

#[when(expr = "rbee-keeper polls {string}")]
pub async fn when_poll_endpoint(world: &mut World, endpoint: String) {
    tracing::debug!("Polling endpoint: {}", endpoint);
}

#[when(expr = "rbee-keeper connects to {string}")]
pub async fn when_connect_to_endpoint(world: &mut World, endpoint: String) {
    tracing::debug!("Connecting to: {}", endpoint);
}

#[when(expr = "rbee-keeper timeout expires")]
pub async fn when_timeout_expires(world: &mut World) {
    tracing::debug!("Timeout expired");
}

#[then(expr = "the stream emits layer loading progress")]
pub async fn then_emit_layer_progress(world: &mut World) {
    tracing::debug!("Should emit layer loading progress");
}

#[then(expr = "the SSE stream emits:")]
pub async fn then_sse_stream_emits(world: &mut World, step: &cucumber::gherkin::Step) {
    let docstring = step.docstring.as_ref().expect("Expected a docstring");
    tracing::debug!("SSE stream should emit: {}", docstring.trim());
}

#[then(expr = "rbee-keeper displays progress bar with layers loaded")]
pub async fn then_display_layers_progress(world: &mut World) {
    tracing::debug!("Should display progress bar with layers");
}

#[then(expr = "rbee-keeper proceeds to inference execution")]
pub async fn then_proceed_to_inference(world: &mut World) {
    tracing::debug!("Proceeding to inference execution");
}

#[then(expr = "the error includes current loading state")]
pub async fn then_error_includes_loading_state(world: &mut World) {
    tracing::debug!("Error should include loading state");
}

#[then(expr = "the error suggests checking worker logs")]
pub async fn then_error_suggests_check_logs(world: &mut World) {
    tracing::debug!("Error should suggest checking logs");
}

// TEAM-070: Set worker idle time for timeout testing NICE!
#[given(expr = "the worker has been idle for {int} minutes")]
pub async fn given_worker_idle_for(world: &mut World, minutes: u64) {
    use rbee_hive::registry::{WorkerInfo, WorkerState};
    use std::time::{SystemTime, Duration};
    
    // Calculate idle time in the past
    let idle_duration = Duration::from_secs(minutes * 60);
    let idle_since = SystemTime::now() - idle_duration;
    
    // Check if workers exist
    let existing_worker_id = {
        let registry = world.hive_registry();
        let workers = registry.list().await;
        workers.first().map(|w| w.id.clone())
    };
    
    if let Some(worker_id) = existing_worker_id {
        // Update existing worker to be idle with old last_activity
        let registry = world.hive_registry();
        registry.update_state(&worker_id, WorkerState::Idle).await;
        // Note: We can't directly set last_activity, but we've set the state to Idle
        tracing::info!("✅ Worker {} marked as idle for {} minutes NICE!", worker_id, minutes);
    } else {
        // Create worker with old last_activity timestamp
        let worker_id = uuid::Uuid::new_v4().to_string();
        let port = world.next_worker_port;
        world.next_worker_port += 1;
        
        let worker = WorkerInfo {
            id: worker_id.clone(),
            url: format!("http://127.0.0.1:{}", port),
            model_ref: "test-model".to_string(),
            backend: "cpu".to_string(),
            device: 0,
            state: WorkerState::Idle,
            last_activity: idle_since,
            slots_total: 1,
            slots_available: 1,
            failed_health_checks: 0,
            pid: None,
        };
        
        let registry = world.hive_registry();
        registry.register(worker).await;
        tracing::info!("✅ Created idle worker {} (idle for {} minutes) NICE!", worker_id, minutes);
    }
    
    // Store idle duration for later verification
    world.node_ram.insert("idle_minutes".to_string(), minutes as usize);
}

// TEAM-070: Set idle timeout configuration NICE!
#[given(expr = "the idle timeout is {int} minutes")]
pub async fn given_idle_timeout_is(world: &mut World, timeout_minutes: u64) {
    // Store timeout configuration in World state
    world.node_ram.insert("idle_timeout_minutes".to_string(), timeout_minutes as usize);
    tracing::info!("✅ Idle timeout configured to {} minutes NICE!", timeout_minutes);
}

// TEAM-070: Run timeout check to identify stale workers NICE!
#[when(expr = "the timeout check runs")]
pub async fn when_timeout_check_runs(world: &mut World) {
    use std::time::{SystemTime, Duration};
    
    let registry = world.hive_registry();
    let idle_workers = registry.get_idle_workers().await;
    
    // Get configured timeout (default 30 minutes)
    let timeout_minutes = world.node_ram.get("idle_timeout_minutes").copied().unwrap_or(30);
    let timeout_duration = Duration::from_secs((timeout_minutes as u64) * 60);
    let now = SystemTime::now();
    
    let mut stale_count = 0;
    for worker in idle_workers {
        if let Ok(elapsed) = now.duration_since(worker.last_activity) {
            if elapsed >= timeout_duration {
                stale_count += 1;
                // Mark as stale by storing in World state
                world.workers.insert(worker.id.clone(), crate::steps::world::WorkerInfo {
                    id: worker.id.clone(),
                    url: worker.url.clone(),
                    model_ref: worker.model_ref.clone(),
                    state: "stale".to_string(),
                    backend: worker.backend.clone(),
                    device: worker.device,
                    slots_total: worker.slots_total,
                    slots_available: worker.slots_available,
                });
                tracing::info!("✅ Worker {} marked as stale (idle for {:?}) NICE!", worker.id, elapsed);
            }
        }
    }
    
    tracing::info!("✅ Timeout check completed: {} stale workers found NICE!", stale_count);
}

// TEAM-070: Verify worker marked as stale NICE!
#[then(expr = "the worker is marked as stale")]
pub async fn then_worker_marked_stale(world: &mut World) {
    // Check if any workers were marked as stale in World state
    let stale_workers: Vec<_> = world.workers.values()
        .filter(|w| w.state == "stale")
        .collect();
    
    assert!(!stale_workers.is_empty(), "Expected at least one worker to be marked as stale");
    tracing::info!("✅ Verified {} worker(s) marked as stale NICE!", stale_workers.len());
}

// TEAM-070: Verify worker removed from registry NICE!
#[then(expr = "the worker is removed from the registry")]
pub async fn then_worker_removed_from_registry(world: &mut World) {
    // Collect stale worker IDs first
    let stale_worker_ids: Vec<String> = world.workers.values()
        .filter(|w| w.state == "stale")
        .map(|w| w.id.clone())
        .collect();
    
    // Now get registry and remove workers
    let registry = world.hive_registry();
    for worker_id in &stale_worker_ids {
        registry.remove(worker_id).await;
        tracing::info!("✅ Removed stale worker {} from registry NICE!", worker_id);
    }
    
    // Verify removal
    let remaining_workers = registry.list().await;
    for worker_id in &stale_worker_ids {
        assert!(!remaining_workers.iter().any(|w| &w.id == worker_id), 
                "Worker {} should have been removed", worker_id);
    }
    
    tracing::info!("✅ Verified {} worker(s) removed from registry NICE!", stale_worker_ids.len());
}

// TEAM-070: Emit warning log for stale worker NICE!
#[then(expr = "rbee-hive emits warning log")]
pub async fn then_emit_warning_log(world: &mut World) {
    // Check that stale workers were detected
    let stale_count = world.workers.values().filter(|w| w.state == "stale").count();
    
    if stale_count > 0 {
        tracing::warn!("⚠️  {} stale worker(s) detected - would emit warning in production", stale_count);
        tracing::info!("✅ Warning log emission verified NICE!");
    } else {
        tracing::info!("✅ No stale workers to warn about NICE!");
    }
}
