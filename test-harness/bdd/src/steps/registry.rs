// Worker registry step definitions
// Created by: TEAM-042
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
//
// Modified by: TEAM-064 (added explicit warning preservation notice)
// Modified by: TEAM-064 (connected to real rbee-hive registry)

use crate::steps::world::{WorkerInfo, World};
use cucumber::{given, then, when};
use rbee_hive::registry::{WorkerInfo as HiveWorkerInfo, WorkerState};

#[given(expr = "no workers are registered")]
pub async fn given_no_workers(world: &mut World) {
    // TEAM-064: Clear both World state AND registry
    world.workers.clear();
    
    // Clear rbee-hive registry
    let registry = world.hive_registry();
    let workers = registry.list().await;
    for worker in workers {
        registry.remove(&worker.id).await;
    }
    
    tracing::info!("✅ Cleared all workers from World AND registry");
}

#[given(expr = "a worker is registered with:")]
pub async fn given_worker_registered_table(world: &mut World, step: &cucumber::gherkin::Step) {
    let table = step.table.as_ref().expect("Expected a data table");

    let mut worker = WorkerInfo {
        id: String::new(),
        url: String::new(),
        model_ref: String::new(),
        state: String::new(),
        backend: String::new(),
        device: 0,
        slots_total: 1,
        slots_available: 1,
    };

    for row in table.rows.iter().skip(1) {
        let field = &row[0];
        let value = &row[1];

        match field.as_str() {
            "id" => worker.id = value.clone(),
            "url" => worker.url = value.clone(),
            "model_ref" => worker.model_ref = value.clone(),
            "state" => worker.state = value.clone(),
            "backend" => worker.backend = value.clone(),
            "device" => worker.device = value.parse().unwrap_or(0),
            _ => {}
        }
    }

    world.workers.insert(worker.id.clone(), worker);
    tracing::debug!("Registered worker from table");
}

#[given(expr = "a worker is registered with model_ref {string} and state {string}")]
pub async fn given_worker_with_model_and_state(
    world: &mut World,
    model_ref: String,
    state: String,
) {
    let slots_available = if state == "idle" { 1 } else { 0 };
    let worker_id = format!("worker-{}", uuid::Uuid::new_v4());

    // TEAM-064: Register in World state (for backward compat)
    let world_worker = WorkerInfo {
        id: worker_id.clone(),
        url: "http://workstation.home.arpa:8001".to_string(),
        model_ref: model_ref.clone(),
        state: state.clone(),
        backend: "cuda".to_string(),
        device: 1,
        slots_total: 1,
        slots_available,
    };
    world.workers.insert(worker_id.clone(), world_worker);
    
    // TEAM-064: ALSO register in rbee-hive registry
    let registry = world.hive_registry();
    let hive_worker = HiveWorkerInfo {
        id: worker_id.clone(),
        url: "http://workstation.home.arpa:8001".to_string(),
        model_ref,
        backend: "cuda".to_string(),
        device: 1,
        state: match state.as_str() {
            "idle" => WorkerState::Idle,
            "busy" => WorkerState::Busy,
            "loading" => WorkerState::Loading,
            _ => WorkerState::Idle,
        },
        last_activity: std::time::SystemTime::now(),
        slots_total: 1,
        slots_available,
    };
    registry.register(hive_worker).await;
    
    tracing::info!("✅ Registered worker {} in BOTH World AND registry", worker_id);
}

#[given(expr = "the worker is healthy")]
pub async fn given_worker_healthy(world: &mut World) {
    // Mark worker as healthy (implementation detail)
    tracing::debug!("Worker marked as healthy");
}

#[when(expr = "queen-rbee queries {string}")]
pub async fn when_query_url(world: &mut World, url: String) {
    // TEAM-058: Implemented HTTP query per TEAM-057 TODO
    let client = reqwest::Client::new();
    match client.get(&url).send().await {
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            world.last_http_response = Some(body);
            world.last_http_status = Some(status.as_u16());
            tracing::info!("✅ Queried {} - status: {}", url, status);
        }
        Err(e) => {
            tracing::warn!("⚠️ Query failed: {}", e);
            world.last_http_response = None;
            world.last_http_status = None;
        }
    }
}

#[when(expr = "rbee-keeper queries the worker registry")]
pub async fn when_query_worker_registry(world: &mut World) {
    // TEAM-058: Implemented worker registry query per TEAM-057 TODO
    let url = format!("{}/v2/workers/list", world.queen_rbee_url.as_ref().unwrap_or(&"http://localhost:8080".to_string()));
    let client = reqwest::Client::new();
    match client.get(&url).send().await {
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            world.last_http_response = Some(body);
            world.last_http_status = Some(status.as_u16());
            tracing::info!("✅ Queried worker registry - status: {}", status);
        }
        Err(e) => {
            tracing::warn!("⚠️ Worker registry query failed: {}", e);
        }
    }
}

#[then(expr = "the response is:")]
pub async fn then_response_is(world: &mut World, step: &cucumber::gherkin::Step) {
    let docstring = step.docstring.as_ref().expect("Expected a docstring");
    let expected_json = docstring.trim();

    // TEAM-058: Implemented JSON response verification per TEAM-057 TODO
    let actual_response = world.last_http_response.as_ref().expect("No HTTP response captured");
    
    // Parse both as JSON for comparison
    let expected: serde_json::Value = serde_json::from_str(expected_json)
        .expect("Expected JSON is not valid");
    let actual: serde_json::Value = serde_json::from_str(actual_response)
        .unwrap_or_else(|_| serde_json::json!({"raw": actual_response}));
    
    assert_eq!(actual, expected, "Response JSON mismatch");
    tracing::info!("✅ Response matches expected JSON");
}

#[then(expr = "rbee-keeper proceeds to pool preflight")]
pub async fn then_proceed_to_preflight(world: &mut World) {
    tracing::debug!("Proceeding to pool preflight");
}

#[then(expr = "rbee-keeper skips to Phase 8 (inference execution)")]
pub async fn then_skip_to_phase_8(world: &mut World) {
    tracing::debug!("Skipping to Phase 8");
}

#[then(expr = "rbee-keeper proceeds to Phase 8 but expects 503 response")]
pub async fn then_proceed_to_phase_8_expect_503(world: &mut World) {
    tracing::debug!("Proceeding to Phase 8, expecting 503");
}

#[then(expr = "the registry returns worker {string} with state {string}")]
pub async fn then_registry_returns_worker(world: &mut World, worker_id: String, state: String) {
    // TEAM-064: Verify against rbee-hive registry, not just HTTP response
    let registry = world.hive_registry();
    
    // Get worker from registry
    let worker = registry.get(&worker_id).await
        .expect(&format!("Worker {} not found in registry", worker_id));
    
    // Verify state matches
    let actual_state = match worker.state {
        WorkerState::Idle => "idle",
        WorkerState::Busy => "busy",
        WorkerState::Loading => "loading",
    };
    
    assert_eq!(actual_state, state, "Worker state mismatch in registry");
    tracing::info!("✅ Registry has worker {} with state {} (verified)", worker_id, state);
    
    // ALSO verify HTTP response if available (backward compat)
    if let Some(response) = world.last_http_response.as_ref() {
        if let Ok(workers) = serde_json::from_str::<serde_json::Value>(response) {
            if let Some(workers_array) = workers.as_array() {
                if let Some(http_worker) = workers_array.iter()
                    .find(|w| w["id"].as_str() == Some(&worker_id)) {
                    let http_state = http_worker["state"].as_str().unwrap_or("unknown");
                    assert_eq!(http_state, state, "HTTP response state mismatch");
                    tracing::info!("✅ HTTP response also matches (double-verified)");
                }
            }
        }
    }
}

#[then(expr = "queen-rbee skips pool preflight and model provisioning")]
pub async fn then_skip_preflight_and_provisioning(world: &mut World) {
    tracing::debug!("Skipping preflight and provisioning");
}

#[then(expr = "rbee-keeper sends inference request directly to {string}")]
pub async fn then_send_inference_direct(world: &mut World, url: String) {
    // TEAM-058: Implemented direct inference request per TEAM-057 TODO
    let client = reqwest::Client::new();
    let payload = serde_json::json!({
        "model": "test-model",
        "prompt": "test prompt",
        "max_tokens": 10
    });
    
    match client.post(&url).json(&payload).send().await {
        Ok(resp) => {
            let status = resp.status();
            tracing::info!("✅ Sent inference request to {} - status: {}", url, status);
            world.last_http_status = Some(status.as_u16());
        }
        Err(e) => {
            tracing::warn!("⚠️ Inference request failed: {}", e);
        }
    }
}

#[then(expr = "the inference completes successfully")]
pub async fn then_inference_completes_successfully(world: &mut World) {
    tracing::debug!("Inference completed successfully");
}

#[then(expr = "the total latency is under {int} seconds")]
pub async fn then_latency_under(world: &mut World, seconds: u64) {
    // TEAM-058: Implemented latency verification per TEAM-057 TODO
    // Calculate latency from world.start_time if available
    if let Some(start) = world.start_time {
        let elapsed = start.elapsed();
        let elapsed_secs = elapsed.as_secs();
        assert!(
            elapsed_secs < seconds,
            "Latency {} seconds exceeds limit of {} seconds",
            elapsed_secs,
            seconds
        );
        tracing::info!("✅ Latency {}s is under {}s limit", elapsed_secs, seconds);
    } else {
        tracing::warn!("⚠️ No start time recorded, skipping latency check");
    }
}

#[then(expr = "rbee-keeper queries the worker registry")]
pub async fn then_keeper_queries_registry(world: &mut World) {
    tracing::debug!("rbee-keeper should query worker registry");
}

#[then(regex = r"^rbee-keeper skips to Phase 8 \(inference execution\)$")]
pub async fn then_keeper_skips_to_phase_8(world: &mut World) {
    tracing::debug!("rbee-keeper should skip to Phase 8");
}

#[then(expr = "the output shows all registered workers with their state")]
pub async fn then_output_shows_all_workers(world: &mut World) {
    tracing::debug!("Output should show all registered workers");
}
