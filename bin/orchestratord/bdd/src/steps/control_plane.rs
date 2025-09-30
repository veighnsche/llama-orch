use crate::steps::world::World;
use cucumber::{given, then, when};
use http::Method;
use serde_json::Value;

#[given(regex = r"^a Control Plane API endpoint$")]
pub async fn given_control_plane_endpoint(_world: &mut World) {}

#[given(regex = r"^a pool id$")]
pub async fn given_pool_id(_world: &mut World) {}

#[when(regex = r"^I request pool health$")]
pub async fn when_request_pool_health(world: &mut World) {
    world.push_fact("cp.health");
    let _ = world.http_call(Method::GET, "/v2/pools/pool0/health", None).await;
}

#[then(regex = r"^I receive 200 with liveness readiness draining and metrics$")]
pub async fn then_health_200_fields(world: &mut World) {
    assert_eq!(world.last_status, Some(http::StatusCode::OK));
    let body = world.last_body.as_ref().expect("expected pool health body");
    let v: Value = serde_json::from_str(body).expect("parse pool health");
    for k in ["live", "ready", "draining", "metrics"] {
        assert!(v.get(k).is_some(), "missing {} field", k);
    }
}

#[when(regex = r"^I request pool drain with deadline_ms$")]
pub async fn when_request_pool_drain(world: &mut World) {
    world.push_fact("cp.drain");
    let body = serde_json::json!({ "deadline_ms": 1000 });
    let _ = world.http_call(Method::POST, "/v2/pools/pool0/drain", Some(body)).await;
}

#[then(regex = r"^draining begins$")]
pub async fn then_draining_begins(world: &mut World) {
    assert_eq!(world.last_status, Some(http::StatusCode::ACCEPTED));
}

#[when(regex = r"^I request pool reload with new model_ref$")]
pub async fn when_request_pool_reload(world: &mut World) {
    world.push_fact("cp.reload");
    let body = serde_json::json!({ "new_model_ref": "good" });
    let _ = world.http_call(Method::POST, "/v2/pools/pool0/reload", Some(body)).await;
}

#[then(regex = r"^reload succeeds and is atomic$")]
pub async fn then_reload_succeeds_atomic(world: &mut World) {
    assert_eq!(world.last_status, Some(http::StatusCode::OK));
}

#[then(regex = r"^reload fails and rolls back atomically$")]
pub async fn then_reload_fails_rollback_atomic(world: &mut World) {
    // Trigger a failure by calling reload with a bad model ref
    let body = serde_json::json!({ "new_model_ref": "bad" });
    let _ = world.http_call(Method::POST, "/v2/pools/pool0/reload", Some(body)).await;
    assert_eq!(world.last_status, Some(http::StatusCode::CONFLICT));
}

#[when(regex = r"^I request capabilities$")]
pub async fn when_request_capabilities(world: &mut World) {
    world.push_fact("cp.capabilities");
    let _ = world.http_call(http::Method::GET, "/v2/meta/capabilities", None).await;
}

#[then(regex = r"^I receive capabilities with engines and API version$")]
pub async fn then_capabilities_with_engines_and_version(world: &mut World) {
    assert_eq!(world.last_status, Some(http::StatusCode::OK));
    let body = world.last_body.as_ref().expect("expected capabilities body");
    let v: Value = serde_json::from_str(body).expect("parse capabilities");
    assert!(v.get("api_version").and_then(|x| x.as_str()).is_some());
    assert!(v.get("engines").and_then(|x| x.as_array()).is_some());
}
