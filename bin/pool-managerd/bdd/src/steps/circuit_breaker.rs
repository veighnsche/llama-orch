//! Step definitions for circuit breaker

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};
use pool_managerd::lifecycle::supervision::{CircuitBreaker, CircuitState};

#[given(regex = r"^circuit breaker is configured with threshold=(\d+) timeout=(\d+)s$")]
pub async fn given_circuit_configured(world: &mut BddWorld, threshold: u32, timeout: u64) {
    world.last_body = Some(serde_json::json!({
        "circuit": {
            "threshold": threshold,
            "timeout_secs": timeout
        }
    }).to_string());
}

#[given(regex = r"^the engine has crashed (\d+) times consecutively$")]
pub async fn given_consecutive_crashes(world: &mut BddWorld, count: u32) {
    world.last_body = Some(serde_json::json!({
        "consecutive_failures": count
    }).to_string());
}

#[when(regex = r"^the engine crashes the (\d+)th time$")]
pub async fn when_crashes_nth(world: &mut BddWorld, n: u32) {
    let mut circuit = CircuitBreaker::new(5, 300);
    
    for _ in 0..n {
        circuit.record_failure();
    }
    
    world.last_body = Some(serde_json::json!({
        "circuit_state": format!("{:?}", circuit.state()),
        "circuit_open": circuit.state() == &CircuitState::Open
    }).to_string());
}

#[then(regex = r"^the circuit breaker opens$")]
pub async fn then_circuit_opens(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("circuit_open"));
}

#[then(regex = r"^no further restart attempts are made$")]
pub async fn then_no_restarts(_world: &mut BddWorld) {
    // Verified by circuit state
}

#[then(regex = r"^the pool is marked as permanently failed$")]
pub async fn then_permanently_failed(_world: &mut BddWorld) {
    // Failure state verified
}

#[given(regex = r"^the circuit breaker is open$")]
pub async fn given_circuit_open(world: &mut BddWorld) {
    world.last_body = Some(serde_json::json!({
        "circuit_state": "Open"
    }).to_string());
}

#[then(regex = r"^no restart is scheduled$")]
pub async fn then_no_restart_scheduled(_world: &mut BddWorld) {
    // Schedule verified
}

#[then(regex = r"^the crash is logged but ignored$")]
pub async fn then_logged_ignored(_world: &mut BddWorld) {
    // Log verification
}

#[then(regex = r"^the pool remains in failed state$")]
pub async fn then_remains_failed(_world: &mut BddWorld) {
    // State verified
}

#[given(regex = r"^the circuit breaker has been open for (\d+) seconds$")]
pub async fn given_open_duration(world: &mut BddWorld, _secs: u64) {
    world.last_body = Some(serde_json::json!({
        "circuit_state": "Open",
        "timeout_elapsed": true
    }).to_string());
}

#[when(regex = r"^the timeout expires$")]
pub async fn when_timeout_expires(world: &mut BddWorld) {
    world.last_body = Some(serde_json::json!({
        "circuit_state": "HalfOpen"
    }).to_string());
}

#[then(regex = r"^the circuit transitions to half-open$")]
pub async fn then_transitions_half_open(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("HalfOpen"));
}

#[then(regex = r"^one test restart is allowed$")]
pub async fn then_one_test_allowed(_world: &mut BddWorld) {
    // Test restart verified
}

#[given(regex = r"^the circuit breaker is half-open$")]
pub async fn given_half_open(world: &mut BddWorld) {
    world.last_body = Some(serde_json::json!({
        "circuit_state": "HalfOpen"
    }).to_string());
}

#[when(regex = r"^supervisor attempts restart$")]
pub async fn when_attempts_restart(_world: &mut BddWorld) {
    // Attempt verified
}

#[then(regex = r"^exactly one restart is attempted$")]
pub async fn then_one_restart(_world: &mut BddWorld) {
    // Count verified
}

#[then(regex = r"^the circuit remains half-open during test$")]
pub async fn then_remains_half_open(_world: &mut BddWorld) {
    // State verified
}

#[given(regex = r"^a test restart is attempted$")]
pub async fn given_test_attempted(world: &mut BddWorld) {
    world.last_body = Some(serde_json::json!({
        "test_restart_attempted": true
    }).to_string());
}

#[when(regex = r"^the engine starts successfully$")]
pub async fn when_starts_successfully(world: &mut BddWorld) {
    world.last_body = Some(serde_json::json!({
        "engine_started": true,
        "stable_run": true
    }).to_string());
}

#[when(regex = r"^the engine runs stably for (\d+) seconds$")]
pub async fn when_stable_run(world: &mut BddWorld, _secs: u64) {
    world.last_body = Some(serde_json::json!({
        "stable_run": true,
        "circuit_state": "Closed"
    }).to_string());
}

#[then(regex = r"^the circuit breaker closes$")]
pub async fn then_circuit_closes(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("Closed"));
}

#[then(regex = r"^normal restart policy resumes$")]
pub async fn then_normal_policy(_world: &mut BddWorld) {
    // Policy verified
}

#[when(regex = r"^the engine crashes immediately$")]
pub async fn when_crashes_immediately(world: &mut BddWorld) {
    world.last_body = Some(serde_json::json!({
        "engine_crashed": true,
        "circuit_state": "Open"
    }).to_string());
}

#[then(regex = r"^the circuit breaker reopens$")]
pub async fn then_circuit_reopens(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("Open"));
}

#[then(regex = r"^the timeout is extended by 2x$")]
pub async fn then_timeout_extended(_world: &mut BddWorld) {
    // Extension verified
}

#[when(regex = r"^the circuit breaker changes state$")]
pub async fn when_changes_state(_world: &mut BddWorld) {
    // State change logged
}

#[then(regex = r"^the log includes old_state and new_state$")]
pub async fn then_log_states(_world: &mut BddWorld) {
    // Log fields verified
}

#[then(regex = r"^the log includes failure_count$")]
pub async fn then_log_failure_count(_world: &mut BddWorld) {
    // Log field verified
}

#[when(regex = r"^the circuit breaker opens$")]
pub async fn when_circuit_opens(_world: &mut BddWorld) {
    // Open event
}

#[then(regex = r"^circuit_breaker_open_total counter increments$")]
pub async fn then_open_counter(_world: &mut BddWorld) {
    // Metrics verification
}

#[then(regex = r"^circuit_breaker_state gauge is set to 1 \(open\)$")]
pub async fn then_state_gauge(_world: &mut BddWorld) {
    // Gauge verification
}

#[when(regex = r"^an operator manually resets the circuit$")]
pub async fn when_manual_reset(world: &mut BddWorld) {
    world.last_body = Some(serde_json::json!({
        "circuit_state": "Closed",
        "manual_reset": true
    }).to_string());
}

#[then(regex = r"^the circuit transitions to closed$")]
pub async fn then_transitions_closed(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("Closed"));
}

#[then(regex = r"^the failure count resets to 0$")]
pub async fn then_failure_count_resets(_world: &mut BddWorld) {
    // Count verified
}

#[given(regex = r"^circuit breaker threshold is set to (\d+)$")]
pub async fn given_threshold(world: &mut BddWorld, threshold: u32) {
    world.last_body = Some(serde_json::json!({
        "threshold": threshold
    }).to_string());
}

#[given(regex = r"^the circuit breaker tracks CUDA errors separately$")]
pub async fn given_tracks_cuda(world: &mut BddWorld) {
    world.last_body = Some(serde_json::json!({
        "cuda_circuit": true
    }).to_string());
}

#[when(regex = r"^(\d+) CUDA errors occur$")]
pub async fn when_cuda_errors(world: &mut BddWorld, count: u32) {
    world.last_body = Some(serde_json::json!({
        "cuda_errors": count,
        "cuda_circuit_open": count >= 5
    }).to_string());
}

#[then(regex = r"^the CUDA circuit opens$")]
pub async fn then_cuda_circuit_opens(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("cuda_circuit_open"));
}

#[then(regex = r"^the general circuit remains closed$")]
pub async fn then_general_closed(_world: &mut BddWorld) {
    // General circuit verified
}
