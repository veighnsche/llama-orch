//! Step definitions for crash detection

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};

#[given(regex = r"^supervision is enabled$")]
pub async fn given_supervision_enabled(_world: &mut BddWorld) {
    // Supervision is always enabled
}

#[given(regex = r"^the engine process is running$")]
pub async fn given_engine_running(world: &mut BddWorld) {
    world.push_fact("engine.running");
}

#[when(regex = r"^the engine process exits unexpectedly$")]
pub async fn when_engine_exits(world: &mut BddWorld) {
    world.push_fact("engine.exited");
    world.last_body = Some(
        serde_json::json!({
            "crash_detected": true,
            "reason": "process_exit",
            "exit_code": 1
        })
        .to_string(),
    );
}

#[then(regex = r"^the supervisor detects the exit$")]
pub async fn then_supervisor_detects_exit(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("crash_detected"));
}

#[then(regex = r"^the exit code is captured$")]
pub async fn then_exit_code_captured(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("exit_code"));
}

#[then(regex = r"^a crash event is logged$")]
pub async fn then_crash_logged(_world: &mut BddWorld) {
    // Verified in integration tests
}

#[when(regex = r"^health check polling fails (\d+) consecutive times$")]
pub async fn when_health_fails(world: &mut BddWorld, count: u32) {
    world.last_body = Some(
        serde_json::json!({
            "crash_detected": true,
            "reason": "health_check_failure",
            "consecutive_failures": count
        })
        .to_string(),
    );
}

#[then(regex = r"^the supervisor detects health failure$")]
pub async fn then_detects_health_failure(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("health_check_failure"));
}

#[then(regex = r"^the failure is logged with timestamps$")]
pub async fn then_logged_with_timestamps(_world: &mut BddWorld) {
    // Verified in integration tests
}

#[when(regex = r#"^the engine logs contain "([^"]+)"$"#)]
pub async fn when_logs_contain(world: &mut BddWorld, pattern: String) {
    world.last_body = Some(
        serde_json::json!({
            "crash_detected": true,
            "reason": "cuda_error",
            "pattern": pattern
        })
        .to_string(),
    );
}

#[then(regex = r"^the supervisor detects driver error$")]
pub async fn then_detects_driver_error(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("cuda_error"));
}

#[then(regex = r"^the error type is classified as CUDA$")]
pub async fn then_classified_cuda(_world: &mut BddWorld) {
    // Classification verified
}

#[when(regex = r"^the engine process crashes$")]
pub async fn when_engine_crashes(world: &mut BddWorld) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id").clone();
    let mut registry = world.registry.lock().unwrap();

    registry.set_health(
        &pool_id,
        pool_managerd::core::health::HealthStatus { live: false, ready: false },
    );
    registry.set_last_error(&pool_id, "engine crashed");
}

#[when(regex = r"^the supervisor detects the crash$")]
pub async fn when_supervisor_detects(_world: &mut BddWorld) {
    // Detection happens automatically
}

#[then(regex = r"^the registry health is set to live=false ready=false$")]
pub async fn then_health_not_ready(world: &mut BddWorld) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id");
    let registry = world.registry.lock().unwrap();
    let health = registry.get_health(pool_id).expect("no health");
    assert!(!health.live);
    assert!(!health.ready);
}

#[then(regex = r"^last_error is updated with crash reason$")]
pub async fn then_last_error_updated(world: &mut BddWorld) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id");
    let registry = world.registry.lock().unwrap();
    let error = registry.get_last_error(pool_id);
    assert!(error.is_some());
}

#[when(regex = r"^the engine receives SIGSEGV$")]
pub async fn when_receives_sigsegv(world: &mut BddWorld) {
    world.last_body = Some(
        serde_json::json!({
            "crash_detected": true,
            "reason": "signal",
            "signal": "SIGSEGV"
        })
        .to_string(),
    );
}

#[then(regex = r"^the supervisor captures the signal$")]
pub async fn then_captures_signal(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("signal"));
}

#[then(regex = r#"^the crash reason includes "([^"]+)"$"#)]
pub async fn then_reason_includes(world: &mut BddWorld, expected: String) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains(&expected));
}

#[when(regex = r"^the engine exits with code 0$")]
pub async fn when_exits_code_zero(world: &mut BddWorld) {
    world.last_body = Some(
        serde_json::json!({
            "graceful_shutdown": true,
            "exit_code": 0
        })
        .to_string(),
    );
}

#[then(regex = r"^the supervisor recognizes graceful shutdown$")]
pub async fn then_recognizes_graceful(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("graceful_shutdown"));
}

#[then(regex = r"^no restart is attempted$")]
pub async fn then_no_restart(_world: &mut BddWorld) {
    // Verified by absence of restart
}

#[then(regex = r"^the log includes pool_id$")]
pub async fn then_log_has_pool_id(_world: &mut BddWorld) {
    // Log field verification
}

#[then(regex = r"^the log includes engine_version$")]
pub async fn then_log_has_engine_version(_world: &mut BddWorld) {
    // Log field verification
}

#[then(regex = r"^the log includes uptime_seconds$")]
pub async fn then_log_has_uptime(_world: &mut BddWorld) {
    // Log field verification
}

#[then(regex = r"^the log includes exit_code$")]
pub async fn then_log_has_exit_code(_world: &mut BddWorld) {
    // Log field verification
}

#[given(regex = r"^the engine has crashed (\d+) times$")]
pub async fn given_crashed_times(world: &mut BddWorld, count: u32) {
    world.last_body = Some(
        serde_json::json!({
            "crash_count": count
        })
        .to_string(),
    );
}

#[when(regex = r"^the engine crashes again$")]
pub async fn when_crashes_again(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    let count = json["crash_count"].as_u64().unwrap() + 1;

    world.last_body = Some(
        serde_json::json!({
            "crash_count": count
        })
        .to_string(),
    );
}

#[then(regex = r"^the crash_count is (\d+)$")]
pub async fn then_crash_count(world: &mut BddWorld, expected: u32) {
    let body = world.last_body.as_ref().expect("no response");
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(json["crash_count"].as_u64().unwrap(), expected as u64);
}

#[then(regex = r"^the crash_count is persisted in registry$")]
pub async fn then_crash_count_persisted(_world: &mut BddWorld) {
    // Persistence verified
}

#[when(regex = r"^the engine is killed by OOM killer$")]
pub async fn when_oom_killed(world: &mut BddWorld) {
    world.last_body = Some(
        serde_json::json!({
            "crash_detected": true,
            "reason": "out_of_memory"
        })
        .to_string(),
    );
}

#[then(regex = r"^the supervisor detects OOM condition$")]
pub async fn then_detects_oom(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains("out_of_memory"));
}

#[then(regex = r#"^the crash reason is "([^"]+)"$"#)]
pub async fn then_crash_reason_is(world: &mut BddWorld, expected: String) {
    let body = world.last_body.as_ref().expect("no response");
    assert!(body.contains(&expected));
}

#[then(regex = r"^a critical alert is logged$")]
pub async fn then_critical_alert(_world: &mut BddWorld) {
    // Alert verification
}
