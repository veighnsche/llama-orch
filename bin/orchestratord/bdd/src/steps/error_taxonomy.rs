use crate::steps::world::World;
use cucumber::{then, when};
use http::Method;

#[when(regex = r"^I trigger INVALID_PARAMS$")]
pub async fn when_trigger_invalid_params(world: &mut World) {
    world.push_fact("err.invalid_params");
    let body = serde_json::json!({
        "task_id": "t-bad",
        "session_id": "s-0",
        "workload": "completion",
        "model_ref": "model0",
        "engine": "llamacpp",
        "ctx": -1, // sentinel
        "priority": "interactive",
        "max_tokens": 1,
        "deadline_ms": 1000
    });
    let _ = world.http_call(Method::POST, "/v2/tasks", Some(body)).await;
}

#[then(regex = r"^I receive 400 with correlation id and error envelope code INVALID_PARAMS$")]
pub async fn then_400_corr_invalid_params(world: &mut World) {
    assert_eq!(world.last_status, Some(http::StatusCode::BAD_REQUEST));
    assert!(world.corr_id.is_some());
    let v: serde_json::Value = serde_json::from_str(world.last_body.as_ref().unwrap()).unwrap();
    assert_eq!(v["code"], "INVALID_PARAMS");
}

#[when(regex = r"^I trigger POOL_UNAVAILABLE$")]
pub async fn when_trigger_pool_unavailable(world: &mut World) {
    world.push_fact("err.pool_unavailable");
    let body = serde_json::json!({
        "task_id": "t-unavail",
        "session_id": "s-0",
        "workload": "completion",
        "model_ref": "pool-unavailable", // sentinel
        "engine": "llamacpp",
        "ctx": 0,
        "priority": "interactive",
        "max_tokens": 1,
        "deadline_ms": 1000
    });
    let _ = world.http_call(Method::POST, "/v1/tasks", Some(body)).await;
}

#[then(regex = r"^I receive 503 with correlation id and error envelope code POOL_UNAVAILABLE$")]
pub async fn then_503_corr_pool_unavailable(world: &mut World) {
    assert_eq!(world.last_status, Some(http::StatusCode::SERVICE_UNAVAILABLE));
    assert!(world.corr_id.is_some());
    let v: serde_json::Value = serde_json::from_str(world.last_body.as_ref().unwrap()).unwrap();
    assert_eq!(v["code"], "POOL_UNAVAILABLE");
}

#[when(regex = r"^I trigger INTERNAL error$")]
pub async fn when_trigger_internal_error(world: &mut World) {
    world.push_fact("err.internal");
    let body = serde_json::json!({
        "task_id": "t-int",
        "session_id": "s-0",
        "workload": "completion",
        "model_ref": "model0",
        "engine": "llamacpp",
        "ctx": 0,
        "priority": "interactive",
        "max_tokens": 1,
        "deadline_ms": 1000,
        "prompt": "cause-internal" // sentinel
    });
    let _ = world.http_call(Method::POST, "/v1/tasks", Some(body)).await;
}

#[then(regex = r"^I receive 500 with correlation id and error envelope code INTERNAL$")]
pub async fn then_500_corr_internal(world: &mut World) {
    assert_eq!(world.last_status, Some(http::StatusCode::INTERNAL_SERVER_ERROR));
    assert!(world.corr_id.is_some());
    let v: serde_json::Value = serde_json::from_str(world.last_body.as_ref().unwrap()).unwrap();
    assert_eq!(v["code"], "INTERNAL");
}

#[then(regex = r"^error envelope includes engine when applicable$")]
pub async fn then_error_envelope_includes_engine(world: &mut World) {
    let v: serde_json::Value = serde_json::from_str(world.last_body.as_ref().unwrap()).unwrap();
    assert!(v.get("engine").is_some());
}
