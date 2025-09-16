use crate::steps::world::World;
use cucumber::{then, when};
use http::Method;
use serde_json::json;

#[when(regex = r"^I set model state Deprecated with deadline_ms$")]
pub async fn when_set_state_deprecated_with_deadline(world: &mut World) {
    world.push_fact("lifecycle.deprecate");
    let body = json!({
        "state": "Deprecated",
        "deadline_ms": 60000,
        "model_id": "model0"
    });
    let _ = world
        .http_call(Method::POST, "/v1/models/state", Some(body))
        .await;
}

#[then(regex = r"^new sessions are blocked with MODEL_DEPRECATED$")]
pub async fn then_new_sessions_blocked_model_deprecated(world: &mut World) {
    // Try to enqueue and expect 403 MODEL_DEPRECATED
    let task = json!({
        "task_id": "t-dep",
        "session_id": "s-0",
        "workload": "completion",
        "model_ref": "model0",
        "engine": "llamacpp",
        "ctx": 0,
        "priority": "interactive",
        "max_tokens": 1,
        "deadline_ms": 1000
    });
    let _ = world.http_call(Method::POST, "/v1/tasks", Some(task)).await;
    assert_eq!(world.last_status, Some(http::StatusCode::FORBIDDEN));
    let v: serde_json::Value = serde_json::from_str(world.last_body.as_ref().unwrap()).unwrap();
    assert_eq!(v["code"], "MODEL_DEPRECATED");
}

#[when(regex = r"^I set model state Retired$")]
pub async fn when_set_state_retired(world: &mut World) {
    world.push_fact("lifecycle.retire");
    let body = json!({
        "state": "Retired",
        "model_id": "model0"
    });
    let _ = world
        .http_call(Method::POST, "/v1/models/state", Some(body))
        .await;
}

#[then(regex = r"^pools unload and archives retained$")]
pub async fn then_pools_unload_archives_retained(world: &mut World) {
    let logs = world.state.logs.lock().unwrap();
    let line = logs
        .iter()
        .rev()
        .find(|l| l.contains("\"event\":\"retire\""))
        .expect("no retire log event found");
    assert!(
        line.contains("\"pools_unloaded\":true"),
        "retire log missing pools_unloaded: {}",
        line
    );
    assert!(
        line.contains("\"archives_retained\":true"),
        "retire log missing archives_retained: {}",
        line
    );
}

#[then(regex = r"^model_state gauge is exported$")]
pub async fn then_model_state_gauge_exported(world: &mut World) {
    let _ = world.http_call(Method::GET, "/metrics", None).await;
    let text = world.last_body.as_ref().expect("metrics text");
    assert!(
        text.contains("# TYPE model_state "),
        "model_state gauge not exported"
    );
}
