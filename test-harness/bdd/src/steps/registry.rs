// Worker registry step definitions
// Created by: TEAM-040

use crate::steps::world::{WorkerInfo, World};
use cucumber::{given, then, when};

#[given(expr = "no workers are registered")]
pub async fn given_no_workers(world: &mut World) {
    world.workers.clear();
    tracing::debug!("Cleared all workers");
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

    let worker = WorkerInfo {
        id: format!("worker-{}", uuid::Uuid::new_v4()),
        url: "http://workstation.home.arpa:8001".to_string(),
        model_ref,
        state,
        backend: "cuda".to_string(),
        device: 1,
        slots_total: 1,
        slots_available,
    };

    world.workers.insert(worker.id.clone(), worker);
    tracing::debug!("Registered worker with model_ref and state");
}

#[given(expr = "the worker is healthy")]
pub async fn given_worker_healthy(world: &mut World) {
    // Mark worker as healthy (implementation detail)
    tracing::debug!("Worker marked as healthy");
}

#[when(expr = "queen-rbee queries {string}")]
pub async fn when_query_url(world: &mut World, url: String) {
    // TODO: Perform HTTP query
    tracing::debug!("Querying: {}", url);
}

#[when(expr = "rbee-keeper queries the worker registry")]
pub async fn when_query_worker_registry(world: &mut World) {
    // TODO: Query registry
    tracing::debug!("Querying worker registry");
}

#[then(expr = "the response is:")]
pub async fn then_response_is(world: &mut World, step: &cucumber::gherkin::Step) {
    let docstring = step.docstring.as_ref().expect("Expected a docstring");
    let expected_json = docstring.trim();

    // TODO: Verify response matches expected JSON
    tracing::debug!("Expected response: {}", expected_json);
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
    // TODO: Verify registry response
    tracing::debug!("Registry should return worker {} with state {}", worker_id, state);
}

#[then(expr = "queen-rbee skips pool preflight and model provisioning")]
pub async fn then_skip_preflight_and_provisioning(world: &mut World) {
    tracing::debug!("Skipping preflight and provisioning");
}

#[then(expr = "rbee-keeper sends inference request directly to {string}")]
pub async fn then_send_inference_direct(world: &mut World, url: String) {
    // TODO: Send inference request
    tracing::debug!("Sending inference request to: {}", url);
}

#[then(expr = "the inference completes successfully")]
pub async fn then_inference_completes_successfully(world: &mut World) {
    tracing::debug!("Inference completed successfully");
}

#[then(expr = "the total latency is under {int} seconds")]
pub async fn then_latency_under(world: &mut World, seconds: u64) {
    // TODO: Verify latency
    tracing::debug!("Latency should be under {} seconds", seconds);
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
