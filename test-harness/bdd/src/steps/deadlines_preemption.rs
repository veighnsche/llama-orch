use crate::steps::world::World;
use cucumber::{given, then};
use http::Method;

#[given(regex = r"^a task with infeasible deadline$")]
pub async fn given_task_with_infeasible_deadline(world: &mut World) {
    world.push_fact("deadline.infeasible");
}

#[then(regex = r"^I receive error code DEADLINE_UNMET$")]
pub async fn then_deadline_unmet_error(world: &mut World) {
    let body = serde_json::json!({
        "task_id": "t-deadline",
        "session_id": "s-0",
        "workload": "completion",
        "model_ref": "model0",
        "engine": "llamacpp",
        "ctx": 0,
        "priority": "interactive",
        "max_tokens": 1,
        "deadline_ms": 0
    });
    let _ = world.http_call(Method::POST, "/v1/tasks", Some(body)).await;
    let v: serde_json::Value = serde_json::from_str(world.last_body.as_ref().unwrap()).unwrap();
    assert_eq!(v["code"], "DEADLINE_UNMET");
}

#[then(regex = r"^SSE metrics include on_time_probability$")]
pub async fn then_sse_metrics_include_on_time_probability(world: &mut World) {
    let _ = world.http_call(Method::GET, "/v1/tasks/t-0/stream", None).await;
    let body = world.last_body.as_ref().expect("sse body");
    assert!(
        body.contains("on_time_probability"),
        "missing on_time_probability in SSE metrics frame"
    );
}
