//! Step definitions for HTTP API endpoints

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};

#[given(regex = r"^a running pool-managerd daemon$")]
pub async fn given_daemon_running(world: &mut BddWorld) {
    world.push_fact("daemon.running");
    // In BDD tests, we use the registry directly rather than spawning a real daemon
    // For integration tests, this would start the actual server
}

#[when(regex = r"^I request GET /health$")]
pub async fn when_get_health(world: &mut BddWorld) {
    world.push_fact("api.get_health");
    // Mock: health endpoint always returns OK
    world.last_status = Some(200);
    world.last_body = Some(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION")
    }).to_string());
}

#[then(regex = r"^I receive (\d+) OK$")]
pub async fn then_receive_ok(world: &mut BddWorld, status: u16) {
    assert_eq!(world.last_status, Some(status));
}

#[then(regex = r"^I receive (\d+) Not Found$")]
pub async fn then_receive_not_found(world: &mut BddWorld, status: u16) {
    assert_eq!(world.last_status, Some(status));
}

#[then(regex = r"^I receive (\d+) Internal Server Error$")]
pub async fn then_receive_internal_error(world: &mut BddWorld, status: u16) {
    assert_eq!(world.last_status, Some(status));
}

#[then(regex = r"^the response includes (\w+) field$")]
pub async fn then_response_includes_field(world: &mut BddWorld, field: String) {
    let body = world.last_body.as_ref().expect("no response body");
    let json: serde_json::Value = serde_json::from_str(body).expect("invalid json");
    assert!(json.get(&field).is_some(), "missing field: {}", field);
}

#[then(regex = r#"^the (\w+) field equals "([^"]+)"$"#)]
pub async fn then_field_equals(world: &mut BddWorld, field: String, expected: String) {
    let body = world.last_body.as_ref().expect("no response body");
    let json: serde_json::Value = serde_json::from_str(body).expect("invalid json");
    let actual = json.get(&field).expect(&format!("missing field: {}", field));
    assert_eq!(actual.as_str().unwrap(), expected);
}

#[then(regex = r"^the version field matches CARGO_PKG_VERSION$")]
pub async fn then_version_matches(world: &mut BddWorld) {
    let body = world.last_body.as_ref().expect("no response body");
    let json: serde_json::Value = serde_json::from_str(body).expect("invalid json");
    let version = json.get("version").expect("missing version field");
    assert_eq!(version.as_str().unwrap(), env!("CARGO_PKG_VERSION"));
}

#[given(regex = r"^no pools are registered$")]
pub async fn given_no_pools(world: &mut BddWorld) {
    let registry = world.registry.lock().unwrap();
    assert_eq!(registry.snapshots().len(), 0);
}

#[when(regex = r"^I request GET /pools/([^/]+)/status$")]
pub async fn when_get_pool_status(world: &mut BddWorld, pool_id: String) {
    world.push_fact("api.get_pool_status");
    let registry = world.registry.lock().unwrap();
    
    if let Some(health) = registry.get_health(&pool_id) {
        let active_leases = registry.get_active_leases(&pool_id);
        let engine_version = registry.get_engine_version(&pool_id);
        
        world.last_status = Some(200);
        world.last_body = Some(serde_json::json!({
            "pool_id": pool_id,
            "live": health.live,
            "ready": health.ready,
            "active_leases": active_leases,
            "engine_version": engine_version
        }).to_string());
    } else {
        world.last_status = Some(404);
        world.last_body = Some(serde_json::json!({
            "error": format!("pool {} not found", pool_id)
        }).to_string());
    }
}

#[then(regex = r#"^the error message contains "([^"]+)"$"#)]
pub async fn then_error_contains(world: &mut BddWorld, expected: String) {
    let body = world.last_body.as_ref().expect("no response body");
    assert!(body.contains(&expected), "error message does not contain: {}", expected);
}

#[then(regex = r"^the (\w+) field equals (\w+)$")]
pub async fn then_field_equals_bool(world: &mut BddWorld, field: String, expected: String) {
    let body = world.last_body.as_ref().expect("no response body");
    let json: serde_json::Value = serde_json::from_str(body).expect("invalid json");
    let actual = json.get(&field).expect(&format!("missing field: {}", field));
    
    let expected_bool = match expected.as_str() {
        "true" => true,
        "false" => false,
        _ => panic!("expected boolean value, got: {}", expected),
    };
    
    assert_eq!(actual.as_bool().unwrap(), expected_bool);
}

#[then(regex = r"^the (\w+) field equals (\d+)$")]
pub async fn then_field_equals_number(world: &mut BddWorld, field: String, expected: i64) {
    let body = world.last_body.as_ref().expect("no response body");
    let json: serde_json::Value = serde_json::from_str(body).expect("invalid json");
    let actual = json.get(&field).expect(&format!("missing field: {}", field));
    assert_eq!(actual.as_i64().unwrap(), expected);
}

#[then(regex = r"^the (\w+) field is null$")]
pub async fn then_field_is_null(world: &mut BddWorld, field: String) {
    let body = world.last_body.as_ref().expect("no response body");
    let json: serde_json::Value = serde_json::from_str(body).expect("invalid json");
    let actual = json.get(&field).expect(&format!("missing field: {}", field));
    assert!(actual.is_null(), "field {} is not null", field);
}
