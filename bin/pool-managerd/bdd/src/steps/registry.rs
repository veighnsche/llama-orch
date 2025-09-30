//! Step definitions for registry operations

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};
use pool_managerd::core::health::HealthStatus;

#[given(regex = r"^an empty registry$")]
pub async fn given_empty_registry(world: &mut BddWorld) {
    world.registry = std::sync::Arc::new(std::sync::Mutex::new(
        pool_managerd::core::registry::Registry::new()
    ));
}

#[given(regex = r#"^a pool \"([^\"]+)\" is registered$"#)]
pub async fn given_pool_registered(world: &mut BddWorld, pool_id: String) {
    let mut registry = world.registry.lock().unwrap();
    registry.register(&pool_id);
    world.pool_id = Some(pool_id);
}

#[given(regex = r#"^no pool \"([^\"]+)\" exists$"#)]
pub async fn given_pool_not_exists(world: &mut BddWorld, pool_id: String) {
    let registry = world.registry.lock().unwrap();
    assert!(registry.get_health(&pool_id).is_none());
}

#[given(regex = r"^the pool has health live=(\w+) ready=(\w+)$")]
pub async fn given_pool_health(world: &mut BddWorld, live: String, ready: String) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set").clone();
    let mut registry = world.registry.lock().unwrap();
    
    let live_bool = live == "true";
    let ready_bool = ready == "true";
    
    registry.set_health(&pool_id, HealthStatus {
        live: live_bool,
        ready: ready_bool,
    });
}

#[given(regex = r"^the pool has active_leases (\d+)$")]
pub async fn given_pool_leases(world: &mut BddWorld, count: i32) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set").clone();
    let mut registry = world.registry.lock().unwrap();
    
    for _ in 0..count {
        registry.allocate_lease(&pool_id);
    }
}

#[given(regex = r#"^the pool has engine_version \"([^\"]+)\"$"#)]
pub async fn given_pool_engine_version(world: &mut BddWorld, version: String) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set").clone();
    let mut registry = world.registry.lock().unwrap();
    registry.set_engine_version(&pool_id, version);
}

#[given(regex = r"^the pool has no engine_version set$")]
pub async fn given_pool_no_engine_version(world: &mut BddWorld) {
    // Default state - no action needed
}

#[when(regex = r#"^I set health for pool \"([^\"]+)\" to live=(\w+) ready=(\w+)$"#)]
pub async fn when_set_health(world: &mut BddWorld, pool_id: String, live: String, ready: String) {
    let mut registry = world.registry.lock().unwrap();
    let live_bool = live == "true";
    let ready_bool = ready == "true";
    
    registry.set_health(&pool_id, HealthStatus {
        live: live_bool,
        ready: ready_bool,
    });
}

#[when(regex = r#"^I get health for pool \"([^\"]+)\"$"#)]
pub async fn when_get_health(world: &mut BddWorld, pool_id: String) {
    let registry = world.registry.lock().unwrap();
    let health = registry.get_health(&pool_id);
    
    world.last_body = Some(serde_json::json!({
        "health": health
    }).to_string());
}

#[then(regex = r"^the health status is live=(\w+) ready=(\w+)$")]
pub async fn then_health_status(world: &mut BddWorld, live: String, ready: String) {
    let body = world.last_body.as_ref().expect("no response body");
    let json: serde_json::Value = serde_json::from_str(body).expect("invalid json");
    let health = json.get("health").expect("missing health field");
    
    let live_bool = live == "true";
    let ready_bool = ready == "true";
    
    assert_eq!(health["live"].as_bool().unwrap(), live_bool);
    assert_eq!(health["ready"].as_bool().unwrap(), ready_bool);
}

#[then(regex = r"^the result is None$")]
pub async fn then_result_is_none(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response body");
    let json: serde_json::Value = serde_json::from_str(body).expect("invalid json");
    let health = json.get("health").expect("missing health field");
    assert!(health.is_null());
}

// Lease steps
#[when(regex = r#"^I allocate a lease for pool \"([^\"]+)\"$"#)]
pub async fn when_allocate_lease(world: &mut BddWorld, pool_id: String) {
    let mut registry = world.registry.lock().unwrap();
    registry.allocate_lease(&pool_id);
}

#[when(regex = r#"^I release a lease for pool \"([^\"]+)\"$"#)]
pub async fn when_release_lease(world: &mut BddWorld, pool_id: String) {
    let mut registry = world.registry.lock().unwrap();
    registry.release_lease(&pool_id);
}

#[when(regex = r#"^I get active_leases for pool \"([^\"]+)\"$"#)]
pub async fn when_get_active_leases(world: &mut BddWorld, pool_id: String) {
    let registry = world.registry.lock().unwrap();
    let count = registry.get_active_leases(&pool_id);
    world.last_body = Some(serde_json::json!({ "count": count }).to_string());
}

#[then(regex = r"^the active_leases count is (\d+)$")]
pub async fn then_leases_count(world: &mut BddWorld, expected: i32) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set");
    let registry = world.registry.lock().unwrap();
    let actual = registry.get_active_leases(pool_id);
    assert_eq!(actual, expected);
}

#[then(regex = r#"^the active_leases count for \"([^\"]+)\" is (\d+)$"#)]
pub async fn then_leases_count_for_pool(world: &mut BddWorld, pool_id: String, expected: i32) {
    let registry = world.registry.lock().unwrap();
    let actual = registry.get_active_leases(&pool_id);
    assert_eq!(actual, expected);
}

#[given(regex = r"^the pool has (\d+) active leases$")]
pub async fn given_pool_has_leases(world: &mut BddWorld, count: i32) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set").clone();
    let mut registry = world.registry.lock().unwrap();
    
    for _ in 0..count {
        registry.allocate_lease(&pool_id);
    }
}

// Handoff steps
#[given(regex = r"^a handoff JSON with all fields:$")]
pub async fn given_handoff_complete(world: &mut BddWorld, json_str: String) {
    let json: serde_json::Value = serde_json::from_str(&json_str).expect("invalid json");
    world.handoff_json = Some(json);
}

#[given(regex = r#"^a pool \"([^\"]+)\" with last_error \"([^\"]+)\"$"#)]
pub async fn given_pool_with_error(world: &mut BddWorld, pool_id: String, error: String) {
    let mut registry = world.registry.lock().unwrap();
    registry.register(&pool_id);
    registry.set_last_error(&pool_id, error);
    world.pool_id = Some(pool_id);
}

#[given(regex = r#"^a handoff JSON with engine_version \"([^\"]+)\"$"#)]
pub async fn given_handoff_minimal(world: &mut BddWorld, version: String) {
    world.handoff_json = Some(serde_json::json!({
        "engine_version": version
    }));
}

#[given(regex = r#"^a handoff JSON with only engine_version \"([^\"]+)\"$"#)]
pub async fn given_handoff_only_version(world: &mut BddWorld, version: String) {
    world.handoff_json = Some(serde_json::json!({
        "engine_version": version
    }));
}

#[given(regex = r"^a handoff JSON with:$")]
pub async fn given_handoff_partial(world: &mut BddWorld, json_str: String) {
    let json: serde_json::Value = serde_json::from_str(&json_str).expect("invalid json");
    world.handoff_json = Some(json);
}

#[when(regex = r#"^I call register_ready_from_handoff for pool \"([^\"]+)\"$"#)]
pub async fn when_register_handoff(world: &mut BddWorld, pool_id: String) {
    let handoff = world.handoff_json.as_ref().expect("no handoff json set");
    let mut registry = world.registry.lock().unwrap();
    registry.register_ready_from_handoff(&pool_id, handoff);
    world.pool_id = Some(pool_id);
}

#[then(regex = r"^the pool health is live=(\w+) ready=(\w+)$")]
pub async fn then_pool_health(world: &mut BddWorld, live: String, ready: String) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set");
    let registry = world.registry.lock().unwrap();
    let health = registry.get_health(pool_id).expect("pool not found");
    
    let live_bool = live == "true";
    let ready_bool = ready == "true";
    
    assert_eq!(health.live, live_bool);
    assert_eq!(health.ready, ready_bool);
}

#[then(regex = r#"^the pool engine_version is \"([^\"]+)\"$"#)]
pub async fn then_pool_engine_version(world: &mut BddWorld, expected: String) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set");
    let registry = world.registry.lock().unwrap();
    let actual = registry.get_engine_version(pool_id).expect("no engine_version");
    assert_eq!(actual, expected);
}

#[then(regex = r#"^the pool device_mask is \"([^\"]+)\"$"#)]
pub async fn then_pool_device_mask(world: &mut BddWorld, expected: String) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set");
    let registry = world.registry.lock().unwrap();
    let actual = registry.get_device_mask(pool_id).expect("no device_mask");
    assert_eq!(actual, expected);
}

#[then(regex = r"^the pool slots_total is (\d+)$")]
pub async fn then_pool_slots_total(world: &mut BddWorld, expected: i32) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set");
    let registry = world.registry.lock().unwrap();
    let actual = registry.get_slots_total(pool_id).expect("no slots_total");
    assert_eq!(actual, expected);
}

#[then(regex = r"^the pool slots_free is (\d+)$")]
pub async fn then_pool_slots_free(world: &mut BddWorld, expected: i32) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set");
    let registry = world.registry.lock().unwrap();
    let actual = registry.get_slots_free(pool_id).expect("no slots_free");
    assert_eq!(actual, expected);
}

#[then(regex = r"^the pool last_error is cleared$")]
pub async fn then_pool_error_cleared(world: &mut BddWorld) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set");
    let registry = world.registry.lock().unwrap();
    let error = registry.get_last_error(pool_id);
    assert!(error.is_none(), "last_error should be None");
}

#[then(regex = r"^the pool last_error is None$")]
pub async fn then_pool_error_none(world: &mut BddWorld) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set");
    let registry = world.registry.lock().unwrap();
    let error = registry.get_last_error(pool_id);
    assert!(error.is_none());
}

#[then(regex = r"^the pool heartbeat is set$")]
pub async fn then_pool_heartbeat_set(world: &mut BddWorld) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set");
    let registry = world.registry.lock().unwrap();
    let heartbeat = registry.get_heartbeat(pool_id);
    assert!(heartbeat.is_some(), "heartbeat should be set");
}

#[then(regex = r"^the pool device_mask is None$")]
pub async fn then_pool_device_mask_none(world: &mut BddWorld) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set");
    let registry = world.registry.lock().unwrap();
    let mask = registry.get_device_mask(pool_id);
    assert!(mask.is_none());
}

#[then(regex = r"^the pool slots_total is None$")]
pub async fn then_pool_slots_total_none(world: &mut BddWorld) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set");
    let registry = world.registry.lock().unwrap();
    let slots = registry.get_slots_total(pool_id);
    assert!(slots.is_none());
}

#[then(regex = r"^the pool slots_free is None$")]
pub async fn then_pool_slots_free_none(world: &mut BddWorld) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set");
    let registry = world.registry.lock().unwrap();
    let slots = registry.get_slots_free(pool_id);
    assert!(slots.is_none());
}

#[then(regex = r"^the pool heartbeat is within (\d+)ms of current time$")]
pub async fn then_pool_heartbeat_recent(world: &mut BddWorld, threshold_ms: i64) {
    let pool_id = world.pool_id.as_ref().expect("no pool_id set");
    let registry = world.registry.lock().unwrap();
    let heartbeat = registry.get_heartbeat(pool_id).expect("no heartbeat");
    
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    
    let diff = (now - heartbeat).abs();
    assert!(diff <= threshold_ms, "heartbeat too old: {}ms", diff);
}
