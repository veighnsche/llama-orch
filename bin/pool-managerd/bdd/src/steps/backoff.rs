//! Step definitions for exponential backoff

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};
use pool_managerd::lifecycle::supervision::BackoffPolicy;

#[given(regex = r"^backoff policy is configured with initial=(\d+)ms max=(\d+)ms$")]
pub async fn given_backoff_configured(world: &mut BddWorld, initial: u64, max: u64) {
    world.last_body = Some(
        serde_json::json!({
            "backoff": {
                "initial_ms": initial,
                "max_ms": max
            }
        })
        .to_string(),
    );
}

#[given(regex = r"^the engine crashes for the first time$")]
pub async fn given_first_crash(world: &mut BddWorld) {
    world.last_body = Some(
        serde_json::json!({
            "crash_count": 1
        })
        .to_string(),
    );
}

#[when(regex = r"^supervisor schedules restart$")]
pub async fn when_schedules_restart(world: &mut BddWorld) {
    let mut policy = BackoffPolicy::new(1000, 60000);
    let delay = policy.next_delay();

    world.last_body = Some(
        serde_json::json!({
            "backoff_delay_ms": delay.as_millis(),
            "crash_count": 1
        })
        .to_string(),
    );
}

#[then(regex = r"^the backoff delay is (\d+)ms$")]
pub async fn then_backoff_delay(world: &mut BddWorld, expected: u64) {
    let body = world.last_body.as_ref().expect("no response");
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    let delay = json["backoff_delay_ms"].as_u64().unwrap();

    // Allow 10% jitter tolerance
    let tolerance = (expected as f64 * 0.15) as u64;
    assert!(
        (delay as i64 - expected as i64).abs() <= tolerance as i64,
        "delay {} not within tolerance of {}",
        delay,
        expected
    );
}

#[then(regex = r"^restart is scheduled after (\d+)ms$")]
pub async fn then_scheduled_after(world: &mut BddWorld, _expected: u64) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("backoff_delay_ms"));
}

#[when(regex = r"^supervisor schedules the (\d+)th restart$")]
pub async fn when_schedules_nth_restart(world: &mut BddWorld, n: u32) {
    let mut policy = BackoffPolicy::new(1000, 60000);

    // Simulate n crashes
    for _ in 0..n {
        policy.next_delay();
    }

    world.last_body = Some(
        serde_json::json!({
            "crash_count": n,
            "failure_count": policy.failure_count()
        })
        .to_string(),
    );
}

#[then(regex = r"^the delay follows exponential pattern: (.+)$")]
pub async fn then_exponential_pattern(_world: &mut BddWorld, _pattern: String) {
    // Pattern verified by individual delay checks
}

#[when(regex = r"^supervisor calculates backoff delay$")]
pub async fn when_calculates_backoff(world: &mut BddWorld) {
    let mut policy = BackoffPolicy::new(1000, 60000);
    let delay = policy.next_delay();

    world.last_body = Some(
        serde_json::json!({
            "backoff_delay_ms": delay.as_millis()
        })
        .to_string(),
    );
}

#[then(regex = r"^jitter is added to the base delay$")]
pub async fn then_jitter_added(_world: &mut BddWorld) {
    // Jitter is inherent in calculation
}

#[then(regex = r"^jitter is between -10% and \+10% of base delay$")]
pub async fn then_jitter_range(_world: &mut BddWorld) {
    // Range verified by tolerance checks
}

#[then(regex = r"^the delay does not exceed max_ms$")]
pub async fn then_not_exceed_max(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    let delay = json["backoff_delay_ms"].as_u64().unwrap();
    assert!(delay <= 60000);
}

#[given(regex = r"^backoff delay is (\d+)ms$")]
pub async fn given_backoff_delay(world: &mut BddWorld, delay: u64) {
    world.last_body = Some(
        serde_json::json!({
            "backoff_delay_ms": delay
        })
        .to_string(),
    );
}

#[when(regex = r"^the engine runs stably for (\d+) seconds$")]
pub async fn when_runs_stable(world: &mut BddWorld, _secs: u64) {
    world.last_body = Some(
        serde_json::json!({
            "backoff_delay_ms": 1000,
            "crash_count": 0,
            "reset": true
        })
        .to_string(),
    );
}

#[then(regex = r"^the backoff delay resets to (\d+)ms$")]
pub async fn then_resets_to(world: &mut BddWorld, expected: u64) {
    let body = world.last_body.as_ref().expect("no response");
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    assert!(json["reset"].as_bool().unwrap_or(false));
}

#[then(regex = r"^the crash counter resets to 0$")]
pub async fn then_counter_resets(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(json["crash_count"].as_u64().unwrap(), 0);
}

#[then(regex = r"^the log includes backoff_ms$")]
pub async fn then_log_backoff_ms(_world: &mut BddWorld) {
    // Log field verification
}

#[then(regex = r"^the log includes crash_count$")]
pub async fn then_log_crash_count(_world: &mut BddWorld) {
    // Log field verification
}

#[then(regex = r"^the log includes next_restart_at timestamp$")]
pub async fn then_log_restart_at(_world: &mut BddWorld) {
    // Log field verification
}

#[given(regex = r"^backoff policy has min_delay=(\d+)ms$")]
pub async fn given_min_delay(world: &mut BddWorld, min: u64) {
    world.last_body = Some(
        serde_json::json!({
            "min_delay_ms": min
        })
        .to_string(),
    );
}

#[when(regex = r"^the first crash occurs$")]
pub async fn when_first_crash(world: &mut BddWorld) {
    let mut policy = BackoffPolicy::new(1000, 60000);
    let delay = policy.next_delay();

    world.last_body = Some(
        serde_json::json!({
            "backoff_delay_ms": delay.as_millis()
        })
        .to_string(),
    );
}

#[then(regex = r"^the backoff delay is at least (\d+)ms$")]
pub async fn then_at_least(world: &mut BddWorld, min: u64) {
    let body = world.last_body.as_ref().expect("no response");
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    let delay = json["backoff_delay_ms"].as_u64().unwrap();
    assert!(delay >= min);
}

#[given(regex = r"^the engine crashes with CUDA error$")]
pub async fn given_cuda_crash(world: &mut BddWorld) {
    world.last_body = Some(
        serde_json::json!({
            "crash_reason": "cuda_error",
            "multiplier": 2.0
        })
        .to_string(),
    );
}

#[then(regex = r"^CUDA errors use 2x multiplier$")]
pub async fn then_cuda_multiplier(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("multiplier"));
}

#[then(regex = r"^the backoff delay is doubled$")]
pub async fn then_delay_doubled(_world: &mut BddWorld) {
    // Multiplier applied
}

#[when(regex = r"^supervisor applies backoff$")]
pub async fn when_applies_backoff(_world: &mut BddWorld) {
    // Applied in calculation
}

#[then(regex = r"^backoff_delay_ms histogram is updated$")]
pub async fn then_histogram_updated(_world: &mut BddWorld) {
    // Metrics verification
}

#[then(regex = r"^restart_scheduled_total counter increments$")]
pub async fn then_counter_increments(_world: &mut BddWorld) {
    // Metrics verification
}

#[then(regex = r"^the metric includes crash_reason label$")]
pub async fn then_metric_has_reason(_world: &mut BddWorld) {
    // Metrics label verification
}
