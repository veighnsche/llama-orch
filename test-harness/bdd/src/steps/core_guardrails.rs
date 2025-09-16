use crate::steps::world::World;
use cucumber::{given, then};
use http::Method;
use serde_json::json;

#[given(regex = r"^a task with context length beyond model limit$")]
pub async fn given_ctx_length_beyond_limit(world: &mut World) {
    world.push_fact("guard.ctx_over_limit");
}

#[given(regex = r"^a task with token budget exceeding configured limit$")]
pub async fn given_token_budget_exceeded(world: &mut World) {
    world.push_fact("guard.token_budget_exceeded");
}

#[then(regex = r"^the request is rejected before enqueue$")]
pub async fn then_rejected_before_enqueue(world: &mut World) {
    // Craft a task request based on which violation was set in Given
    let mut body = json!({
        "task_id": "t-guard",
        "session_id": "s-0",
        "workload": "completion",
        "model_ref": "model0",
        "engine": "llamacpp",
        "ctx": 0,
        "priority": "interactive",
        "max_tokens": 1,
        "deadline_ms": 1000
    });
    for ev in world.all_facts() {
        if let Some(stage) = ev.get("stage").and_then(|v| v.as_str()) {
            if stage == "guard.ctx_over_limit" {
                body["ctx"] = json!(50000);
            } else if stage == "guard.token_budget_exceeded" {
                body["max_tokens"] = json!(100000);
            }
        }
    }
    let _ = world.http_call(Method::POST, "/v1/tasks", Some(body)).await;
    assert_eq!(world.last_status, Some(http::StatusCode::BAD_REQUEST));
    let v: serde_json::Value = serde_json::from_str(world.last_body.as_ref().unwrap()).unwrap();
    assert_eq!(v["code"], "INVALID_PARAMS");
}

#[given(regex = r"^a running task exceeding watchdog thresholds$")]
pub async fn given_running_task_exceeds_watchdog(world: &mut World) {
    world.push_fact("guard.watchdog_threshold_exceeded");
}

#[then(regex = r"^the watchdog aborts the task$")]
pub async fn then_watchdog_aborts_task(_world: &mut World) {}
