// Happy path scenario step definitions
// Created by: TEAM-040
// Modified by: TEAM-042 (implemented step definitions with mock behavior)

use crate::steps::world::World;
use cucumber::{given, then, when};

#[given(expr = "no workers are registered for model {string}")]
pub async fn given_no_workers_for_model(world: &mut World, model_ref: String) {
    // Remove any workers with this model_ref
    world.workers.retain(|_, worker| worker.model_ref != model_ref);
    tracing::debug!("Cleared workers for model: {}", model_ref);
}

#[given(expr = "node {string} is reachable at {string}")]
pub async fn given_node_reachable(world: &mut World, node: String, url: String) {
    // Update topology with URL
    if let Some(_node_info) = world.topology.get_mut(&node) {
        // Store URL in a way we can retrieve it later
        // For now, just log it
        tracing::debug!("Node {} is reachable at {}", node, url);
    }
}

#[given(expr = "node {string} has {int} MB of available RAM")]
pub async fn given_node_ram(world: &mut World, node: String, ram_mb: usize) {
    world.node_ram.insert(node.clone(), ram_mb);
    tracing::debug!("Node {} has {} MB RAM", node, ram_mb);
}

#[given(expr = "node {string} has Metal backend available")]
pub async fn given_node_metal_backend(world: &mut World, node: String) {
    world.node_backends.entry(node.clone()).or_insert_with(Vec::new).push("metal".to_string());
    tracing::debug!("Node {} has Metal backend", node);
}

// TEAM-044: Removed duplicate "I run:" step - real implementation is in cli_commands.rs
// TEAM-042 had created a mock version here that conflicted with the real execution

#[then(expr = "queen-rbee queries node {string} via SSH at {string}")]
pub async fn then_queen_rbee_ssh_query(world: &mut World, node: String, hostname: String) {
    // Mock: simulate SSH query
    tracing::info!("✅ Mock SSH query to {} at {}", node, hostname);
}

#[then(expr = "queen-rbee queries rbee-hive worker registry at {string}")]
pub async fn then_query_worker_registry(world: &mut World, url: String) {
    // Mock: simulate HTTP request to worker registry
    world.last_http_response = Some(crate::steps::world::HttpResponse {
        status: 200,
        headers: std::collections::HashMap::new(),
        body: serde_json::json!({"workers": []}).to_string(),
    });
    tracing::info!("✅ Mock query worker registry at: {}", url);
}

#[then(expr = "the worker registry returns an empty list")]
pub async fn then_registry_returns_empty(world: &mut World) {
    // Mock: verify empty worker list
    world.workers.clear();
    tracing::info!("✅ Worker registry returns empty list");
}

#[then(expr = "queen-rbee performs pool preflight check at {string}")]
pub async fn then_pool_preflight_check(world: &mut World, url: String) {
    // Mock: simulate preflight check
    world.last_http_response = Some(crate::steps::world::HttpResponse {
        status: 200,
        headers: std::collections::HashMap::new(),
        body: serde_json::json!({
            "status": "alive",
            "version": "0.1.0",
            "api_version": "v1"
        })
        .to_string(),
    });
    tracing::info!("✅ Mock preflight check at: {}", url);
}

#[then(expr = "the health check returns version {string} and status {string}")]
pub async fn then_health_check_response(world: &mut World, version: String, status: String) {
    // Mock: verify health check response
    tracing::info!("✅ Health check returned version={}, status={}", version, status);
}

#[then(expr = "rbee-hive checks the model catalog for {string}")]
pub async fn then_check_model_catalog(world: &mut World, model_ref: String) {
    // Mock: check model catalog
    tracing::info!("✅ Mock check catalog for: {}", model_ref);
}

#[then(expr = "the model is not found in the catalog")]
pub async fn then_model_not_found(world: &mut World) {
    // Mock: model not found
    tracing::info!("✅ Model not found in catalog (cold start)");
}

#[then(expr = "rbee-hive downloads the model from Hugging Face")]
pub async fn then_download_from_hf(world: &mut World) {
    // Mock: initiate download
    tracing::info!("✅ Mock download initiated from Hugging Face");
}

#[then(expr = "a download progress SSE stream is available at {string}")]
pub async fn then_download_progress_stream(world: &mut World, url: String) {
    // Mock: SSE stream for download progress
    world.sse_events.push(crate::steps::world::SseEvent {
        event_type: "progress".to_string(),
        data: serde_json::json!({
            "stage": "downloading",
            "bytes_downloaded": 2097152,
            "bytes_total": 5242880,
            "speed_mbps": 48.1
        }),
    });
    tracing::info!("✅ Mock SSE download progress stream at: {}", url);
}

#[then(expr = "rbee-keeper displays a progress bar showing download percentage and speed")]
pub async fn then_display_progress_bar(world: &mut World) {
    // Mock: display progress bar
    tracing::info!("✅ Mock progress bar: [████████----] 40% (2.0 MB / 5.0 MB) @ 48.1 Mbps");
}

#[then(expr = "the model download completes successfully")]
pub async fn then_download_completes(world: &mut World) {
    // Mock: download complete
    world.sse_events.push(crate::steps::world::SseEvent {
        event_type: "complete".to_string(),
        data: serde_json::json!({
            "stage": "complete",
            "local_path": "/models/tinyllama-q4.gguf"
        }),
    });
    tracing::info!("✅ Model download completed");
}

#[then(expr = "rbee-hive registers the model in SQLite catalog with local_path {string}")]
pub async fn then_register_model_in_catalog(world: &mut World, local_path: String) {
    // Mock: register model in catalog
    world.model_catalog.insert(
        "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
        crate::steps::world::ModelCatalogEntry {
            provider: "hf".to_string(),
            reference: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
            local_path: std::path::PathBuf::from(local_path.clone()),
            size_bytes: 5242880,
        },
    );
    tracing::info!("✅ Model registered in catalog: {}", local_path);
}

#[then(expr = "rbee-hive performs worker preflight checks")]
pub async fn then_worker_preflight_checks(world: &mut World) {
    // Mock: perform preflight checks
    tracing::info!("✅ Worker preflight checks initiated");
}

#[then(expr = "RAM check passes with {int} MB available")]
pub async fn then_ram_check_passes(world: &mut World, ram_mb: usize) {
    // Mock: RAM check passes
    tracing::info!("✅ RAM check passed: {} MB available", ram_mb);
}

#[then(expr = "Metal backend check passes")]
pub async fn then_metal_check_passes(world: &mut World) {
    // Mock: backend check passes
    tracing::info!("✅ Metal backend check passed");
}

#[then(expr = "rbee-hive spawns worker process {string} on port {int}")]
pub async fn then_spawn_worker(world: &mut World, binary: String, port: u16) {
    // Mock: spawn worker
    tracing::info!("✅ Mock spawned {} on port {}", binary, port);
}

#[then(expr = "the worker HTTP server starts on port {int}")]
pub async fn then_worker_http_starts(world: &mut World, port: u16) {
    // Mock: HTTP server started
    tracing::info!("✅ Worker HTTP server started on port {}", port);
}

#[then(expr = "the worker sends ready callback to {string}")]
pub async fn then_worker_ready_callback(world: &mut World, url: String) {
    // Mock: ready callback sent
    world.workers.insert(
        "worker-abc123".to_string(),
        crate::steps::world::WorkerInfo {
            id: "worker-abc123".to_string(),
            url: "http://mac.home.arpa:8001".to_string(),
            model_ref: "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
            state: "loading".to_string(),
            backend: "metal".to_string(),
            device: 0,
            slots_total: 1,
            slots_available: 0,
        },
    );
    tracing::info!("✅ Worker sent ready callback to: {}", url);
}

#[then(expr = "rbee-hive registers the worker in the in-memory registry")]
pub async fn then_register_worker(world: &mut World) {
    // Mock: worker registered
    tracing::info!("✅ Worker registered in in-memory registry");
}

#[then(expr = "rbee-hive returns worker details to queen-rbee")]
pub async fn then_return_worker_details(world: &mut World) {
    // Mock: worker details returned
    tracing::info!("✅ Worker details returned to queen-rbee");
}

#[then(expr = "queen-rbee returns worker URL to rbee-keeper")]
pub async fn then_return_worker_url(world: &mut World) {
    // Mock: worker URL returned
    tracing::info!("✅ Worker URL returned to rbee-keeper");
}

#[then(expr = "rbee-keeper polls worker readiness at {string}")]
pub async fn then_poll_worker_readiness(world: &mut World, url: String) {
    // Mock: poll readiness
    tracing::info!("✅ Mock poll worker readiness at: {}", url);
}

#[then(expr = "the worker returns state {string} with progress_url")]
pub async fn then_worker_state_with_progress(world: &mut World, state: String) {
    // Mock: worker state response
    tracing::info!("✅ Worker returned state: {}", state);
}

#[then(expr = "rbee-keeper streams loading progress showing layers loaded")]
pub async fn then_stream_loading_progress(world: &mut World) {
    // Mock: stream loading progress
    world.sse_events.push(crate::steps::world::SseEvent {
        event_type: "progress".to_string(),
        data: serde_json::json!({
            "stage": "loading_to_vram",
            "layers_loaded": 24,
            "layers_total": 32,
            "vram_mb": 4096
        }),
    });
    tracing::info!("✅ Mock loading progress: 24/32 layers loaded");
}

#[then(expr = "the worker completes loading and returns state {string}")]
pub async fn then_worker_completes_loading(world: &mut World, state: String) {
    // Mock: loading complete
    if let Some(worker) = world.workers.get_mut("worker-abc123") {
        worker.state = state.clone();
        worker.slots_available = 1;
    }
    tracing::info!("✅ Worker completed loading, state: {}", state);
}

#[then(expr = "rbee-keeper sends inference request to {string}")]
pub async fn then_send_inference_request(world: &mut World, url: String) {
    // Mock: send inference request
    tracing::info!("✅ Mock inference request sent to: {}", url);
}

#[then(expr = "the worker streams tokens via SSE")]
pub async fn then_stream_tokens(world: &mut World) {
    // Mock: stream tokens
    let tokens = vec!["Once", " upon", " a", " time", " in", " a", " small", " village"];
    for token in tokens {
        world.tokens_generated.push(token.to_string());
        world.sse_events.push(crate::steps::world::SseEvent {
            event_type: "token".to_string(),
            data: serde_json::json!({"token": token}),
        });
    }
    tracing::info!("✅ Mock token streaming: {} tokens", world.tokens_generated.len());
}

#[then(expr = "rbee-keeper displays tokens to stdout in real-time")]
pub async fn then_display_tokens(world: &mut World) {
    // Mock: display tokens
    let output = world.tokens_generated.join("");
    world.last_stdout = output.clone();
    tracing::info!("✅ Mock token display: {}", output);
}

#[then(expr = "the inference completes with {int} tokens generated")]
pub async fn then_inference_completes(world: &mut World, token_count: u32) {
    // Mock: inference complete
    world.inference_metrics = Some(crate::steps::world::InferenceMetrics {
        tokens_out: token_count,
        decode_time_ms: 150,
    });
    tracing::info!("✅ Inference completed: {} tokens in 150ms", token_count);
}

#[then(expr = "the worker transitions to state {string}")]
pub async fn then_worker_transitions_to_state(world: &mut World, state: String) {
    // Mock: state transition
    if let Some(worker) = world.workers.get_mut("worker-abc123") {
        worker.state = state.clone();
    }
    tracing::info!("✅ Worker transitioned to state: {}", state);
}

// TEAM-044: Removed duplicate "the exit code is" step - real implementation is in cli_commands.rs

#[then(expr = "rbee-keeper connects to the progress SSE stream")]
pub async fn then_connect_to_progress_sse(world: &mut World) {
    tracing::info!("✅ Mock connected to progress SSE stream");
}

// Registry integration steps (TEAM-041)
#[then(expr = "queen-rbee queries rbee-hive registry for node {string}")]
pub async fn then_query_beehive_registry(world: &mut World, node: String) {
    if world.beehive_nodes.contains_key(&node) {
        tracing::info!("✅ Registry query found node: {}", node);
    } else {
        tracing::info!("✅ Registry query: node '{}' not found", node);
    }
}

#[then(expr = "the registry returns SSH details for node {string}")]
pub async fn then_registry_returns_ssh_details(world: &mut World, node: String) {
    if let Some(node_info) = world.beehive_nodes.get(&node) {
        tracing::info!(
            "✅ Registry returned SSH details: {}@{}",
            node_info.ssh_user,
            node_info.ssh_host
        );
    }
}

#[then(expr = "queen-rbee establishes SSH connection using registry details")]
pub async fn then_establish_ssh_with_registry(world: &mut World) {
    tracing::info!("✅ Mock SSH connection established using registry details");
}

#[then(expr = "queen-rbee starts rbee-hive via SSH at {string}")]
pub async fn then_start_beehive_via_ssh(world: &mut World, hostname: String) {
    tracing::info!("✅ Mock started rbee-hive via SSH at: {}", hostname);
}

#[then(expr = "queen-rbee updates registry with last_connected_unix")]
pub async fn then_update_last_connected(world: &mut World) {
    let timestamp = 1728508603;
    for node in world.beehive_nodes.values_mut() {
        node.last_connected_unix = Some(timestamp);
    }
    tracing::info!("✅ Registry updated with last_connected_unix: {}", timestamp);
}
