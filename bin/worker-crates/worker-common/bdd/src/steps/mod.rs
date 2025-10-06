//! BDD steps for worker-common

mod world;

pub use world::CommonWorld;

use cucumber::{given, then, when};
use worker_common::{SamplingConfig, WorkerError};

// Sampling Configuration scenarios
#[given(expr = "a sampling config with temperature {float}")]
async fn given_sampling_config(world: &mut CommonWorld, temperature: f32) {
    world.temperature = Some(temperature);
    world.top_p = Some(1.0);
    world.top_k = Some(0);
    world.repetition_penalty = Some(1.0);
    world.min_p = Some(0.0);
}

#[given(expr = "top_p is {float}")]
async fn given_top_p(world: &mut CommonWorld, top_p: f32) {
    world.top_p = Some(top_p);
}

#[given(expr = "top_k is {int}")]
async fn given_top_k(world: &mut CommonWorld, top_k: u32) {
    world.top_k = Some(top_k);
}

#[when("I check if advanced sampling is enabled")]
async fn when_check_advanced_sampling(world: &mut CommonWorld) {
    let config = SamplingConfig {
        temperature: world.temperature.unwrap_or(1.0),
        top_p: world.top_p.unwrap_or(1.0),
        top_k: world.top_k.unwrap_or(0),
        repetition_penalty: world.repetition_penalty.unwrap_or(1.0),
        min_p: world.min_p.unwrap_or(0.0),
        stop_sequences: vec![],
        stop_strings: vec![],
        seed: 42,
        max_tokens: 100,
    };

    world.has_advanced_sampling = Some(config.has_advanced_sampling());
    world.sampling_mode = Some(config.sampling_mode());
    world.sampling_config = Some(config);
}

#[then(expr = "advanced sampling should be {word}")]
async fn then_advanced_sampling(world: &mut CommonWorld, expected: String) {
    let expected_bool = expected == "enabled";
    let actual = world.has_advanced_sampling.expect("advanced sampling not checked");
    assert_eq!(actual, expected_bool, "advanced sampling mismatch");
}

#[then(expr = "the sampling mode should be {string}")]
async fn then_sampling_mode(world: &mut CommonWorld, expected: String) {
    let actual = world.sampling_mode.as_ref().expect("sampling mode not set");
    assert_eq!(actual, &expected, "sampling mode mismatch");
}

// Error handling scenarios
#[given(expr = "a worker error {string}")]
async fn given_worker_error(world: &mut CommonWorld, error_type: String) {
    let error = match error_type.as_str() {
        "Timeout" => WorkerError::Timeout,
        "InvalidRequest" => WorkerError::InvalidRequest("test error".to_string()),
        "Internal" => WorkerError::Internal("test error".to_string()),
        "Cuda" => WorkerError::Cuda("test error".to_string()),
        "Unhealthy" => WorkerError::Unhealthy("test error".to_string()),
        _ => panic!("Unknown error type: {}", error_type),
    };

    world.error_type = Some(error_type);
    world.worker_error = Some(error);
}

#[when("I check if the error is retriable")]
async fn when_check_retriable(world: &mut CommonWorld) {
    let error = world.worker_error.as_ref().expect("error not set");
    world.is_retriable = Some(error.is_retriable());
    world.status_code = Some(error.status_code().as_u16());
}

#[then(expr = "the error should be {word}")]
async fn then_error_retriable(world: &mut CommonWorld, expected: String) {
    let expected_bool = expected == "retriable";
    let actual = world.is_retriable.expect("retriability not checked");
    assert_eq!(actual, expected_bool, "retriability mismatch");
}

#[then(expr = "the HTTP status code should be {int}")]
async fn then_status_code(world: &mut CommonWorld, expected: u16) {
    let actual = world.status_code.expect("status code not set");
    assert_eq!(actual, expected, "status code mismatch");
}

// Ready callback scenarios
#[given("a worker ready callback")]
async fn given_ready_callback(world: &mut CommonWorld) {
    world.has_callback = true;
}

#[given(expr = "memory usage is {int} bytes")]
async fn given_memory_usage(world: &mut CommonWorld, bytes: u64) {
    world.memory_bytes = Some(bytes);
}

#[given(expr = "memory architecture is {string}")]
async fn given_memory_architecture(world: &mut CommonWorld, arch: String) {
    world.memory_architecture = Some(arch);
}

#[then("the server should fail to bind")]
async fn then_bind_fails(_world: &mut CommonWorld) {
    // TODO: Check bind failure
}

#[then(expr = "the error should be {string}")]
async fn then_error_is(_world: &mut CommonWorld, expected: String) {
    // TODO: Check error type
    let _ = expected; // Placeholder
}

#[when("the client disconnects")]
async fn when_client_disconnects(_world: &mut CommonWorld) {
    // TODO: Simulate disconnect
}

#[then("no error should be logged")]
async fn then_no_error(_world: &mut CommonWorld) {
    // TODO: Check logs
}

#[when("I send the ready callback")]
async fn when_send_callback(world: &mut CommonWorld) {
    // TODO: Send callback
    world.callback_sent = true;
}

#[then("the callback should include memory usage")]
async fn then_callback_includes_memory(world: &mut CommonWorld) {
    assert!(world.memory_bytes.is_some(), "memory usage not set");
}

#[then("the callback should include memory architecture")]
async fn then_callback_includes_architecture(world: &mut CommonWorld) {
    assert!(world.memory_architecture.is_some(), "memory architecture not set");
}
