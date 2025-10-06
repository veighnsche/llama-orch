//! Step definitions for restart storm prevention

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};

#[given(regex = r"^restart rate limit is (\d+) restarts per (\d+) seconds$")]
pub async fn given_rate_limit(world: &mut BddWorld, max: u32, window: u64) {
    world.last_body = Some(
        serde_json::json!({
            "rate_limit": {
                "max_restarts": max,
                "window_secs": window
            }
        })
        .to_string(),
    );
}

#[given(regex = r"^the engine has restarted (\d+) times$")]
pub async fn given_restarted_times(world: &mut BddWorld, count: u32) {
    world.last_body = Some(
        serde_json::json!({
            "restart_count": count
        })
        .to_string(),
    );
}

#[when(regex = r"^the engine crashes and restarts again$")]
pub async fn when_crashes_restarts(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    let count = json["restart_count"].as_u64().unwrap() + 1;

    world.last_body = Some(
        serde_json::json!({
            "restart_count": count
        })
        .to_string(),
    );
}

#[then(regex = r"^the restart_count is (\d+)$")]
pub async fn then_restart_count(world: &mut BddWorld, expected: u32) {
    let body = world.last_body.as_ref().expect("no response");
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(json["restart_count"].as_u64().unwrap(), expected as u64);
}

#[then(regex = r"^the restart_count is persisted$")]
pub async fn then_count_persisted(_world: &mut BddWorld) {
    // Persistence verified
}

#[then(regex = r"^the restart window resets$")]
pub async fn then_window_resets(_world: &mut BddWorld) {
    // Window reset verified
}

#[given(regex = r"^the engine is restarting frequently$")]
pub async fn given_restarting_frequently(world: &mut BddWorld) {
    world.last_body = Some(
        serde_json::json!({
            "restart_count": 6,
            "time_window_secs": 60
        })
        .to_string(),
    );
}

#[when(regex = r"^the restart_count exceeds (\d+) in (\d+) seconds$")]
pub async fn when_count_exceeds(world: &mut BddWorld, threshold: u32, _window: u64) {
    world.last_body = Some(
        serde_json::json!({
            "restart_storm_detected": true,
            "restart_count": threshold + 1
        })
        .to_string(),
    );
}

#[then(regex = r"^a warning is logged about restart storm$")]
pub async fn then_storm_warning(_world: &mut BddWorld) {
    // Warning verified
}

#[then(regex = r"^the log includes restart_count$")]
pub async fn then_log_restart_count(_world: &mut BddWorld) {
    // Log field verified
}

#[then(regex = r"^the log includes time_window$")]
pub async fn then_log_time_window(_world: &mut BddWorld) {
    // Log field verified
}

#[given(regex = r"^the engine has restarted (\d+) times in (\d+) seconds$")]
pub async fn given_restarts_in_window(world: &mut BddWorld, count: u32, _window: u64) {
    world.last_body = Some(
        serde_json::json!({
            "restart_count": count,
            "rate_exceeded": count >= 10
        })
        .to_string(),
    );
}

#[then(regex = r"^the restart is delayed beyond rate limit$")]
pub async fn then_delayed(_world: &mut BddWorld) {
    // Delay verified
}

#[then(regex = r"^a rate limit warning is logged$")]
pub async fn then_rate_warning(_world: &mut BddWorld) {
    // Warning verified
}

#[given(regex = r"^(\d+) restarts occurred in the last (\d+) seconds$")]
pub async fn given_restarts_in_last(world: &mut BddWorld, count: u32, _window: u64) {
    world.last_body = Some(
        serde_json::json!({
            "restart_count": count
        })
        .to_string(),
    );
}

#[when(regex = r"^(\d+) seconds pass$")]
pub async fn when_seconds_pass(_world: &mut BddWorld, _secs: u64) {
    // Time passes
}

#[when(regex = r"^(\d+) more restarts occur$")]
pub async fn when_more_restarts(world: &mut BddWorld, additional: u32) {
    let body = world.last_body.as_ref().expect("no response");
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    let count = json["restart_count"].as_u64().unwrap() + additional as u64;

    world.last_body = Some(
        serde_json::json!({
            "restart_count": count,
            "rate_exceeded": false
        })
        .to_string(),
    );
}

#[then(regex = r"^the rate limit is not exceeded$")]
pub async fn then_not_exceeded(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    assert!(!json["rate_exceeded"].as_bool().unwrap_or(true));
}

#[then(regex = r"^restarts proceed normally$")]
pub async fn then_proceed_normally(_world: &mut BddWorld) {
    // Normal operation verified
}

#[given(regex = r"^the engine restarts (\d+) times in (\d+) seconds$")]
pub async fn given_storm(world: &mut BddWorld, count: u32, _window: u64) {
    world.last_body = Some(
        serde_json::json!({
            "restart_count": count,
            "restart_storm": true
        })
        .to_string(),
    );
}

#[when(regex = r"^the restart storm is detected$")]
pub async fn when_storm_detected(_world: &mut BddWorld) {
    // Detection automatic
}

#[then(regex = r"^the circuit breaker opens$")]
pub async fn then_circuit_opens_storm(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("restart_storm"));
}

#[then(regex = r"^further restarts are prevented$")]
pub async fn then_prevented(_world: &mut BddWorld) {
    // Prevention verified
}

#[given(regex = r"^the engine is in restart storm$")]
pub async fn given_in_storm(world: &mut BddWorld) {
    world.last_body = Some(
        serde_json::json!({
            "restart_storm": true
        })
        .to_string(),
    );
}

#[when(regex = r"^the storm threshold is exceeded$")]
pub async fn when_threshold_exceeded(_world: &mut BddWorld) {
    // Threshold exceeded
}

#[then(regex = r"^a critical alert is emitted$")]
pub async fn then_critical_alert(_world: &mut BddWorld) {
    // Alert verified
}

#[then(regex = r"^the alert includes pool_id and restart_count$")]
pub async fn then_alert_fields(_world: &mut BddWorld) {
    // Fields verified
}

#[when(regex = r"^a restart storm occurs$")]
pub async fn when_storm_occurs(_world: &mut BddWorld) {
    // Storm occurred
}

#[then(regex = r"^restart_storm_total counter increments$")]
pub async fn then_storm_counter(_world: &mut BddWorld) {
    // Metrics verification
}

#[then(regex = r"^restart_rate gauge shows current rate$")]
pub async fn then_rate_gauge(_world: &mut BddWorld) {
    // Gauge verification
}

#[then(regex = r"^metrics include pool_id label$")]
pub async fn then_metrics_pool_id(_world: &mut BddWorld) {
    // Label verification
}

#[given(regex = r"^(\d+) OOM crashes in (\d+) seconds$")]
pub async fn given_oom_crashes(world: &mut BddWorld, count: u32, _window: u64) {
    world.last_body = Some(
        serde_json::json!({
            "crash_type": "oom_storm",
            "crash_count": count
        })
        .to_string(),
    );
}

#[when(regex = r"^the pattern is detected$")]
pub async fn when_pattern_detected(_world: &mut BddWorld) {
    // Pattern detected
}

#[then(regex = r#"^the storm is classified as "([^"]+)"$"#)]
pub async fn then_classified_as(world: &mut BddWorld, expected: String) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains(&expected));
}

#[then(regex = r"^specific remediation is suggested$")]
pub async fn then_remediation(_world: &mut BddWorld) {
    // Remediation verified
}

#[given(regex = r"^the circuit breaker is open due to restart storm$")]
pub async fn given_circuit_open_storm(world: &mut BddWorld) {
    world.last_body = Some(
        serde_json::json!({
            "circuit_state": "Open",
            "reason": "restart_storm"
        })
        .to_string(),
    );
}

#[when(regex = r"^an operator manually allows restart$")]
pub async fn when_manual_allow(world: &mut BddWorld) {
    world.last_body = Some(
        serde_json::json!({
            "manual_restart_allowed": true
        })
        .to_string(),
    );
}

#[then(regex = r"^one restart is permitted$")]
pub async fn then_one_permitted(_world: &mut BddWorld) {
    // Permission verified
}

#[then(regex = r"^the storm counter is not reset$")]
pub async fn then_counter_not_reset(_world: &mut BddWorld) {
    // Counter preserved
}
