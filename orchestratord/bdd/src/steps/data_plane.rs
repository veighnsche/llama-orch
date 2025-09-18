use crate::steps::world::World;
use cucumber::{given, then, when};
use http::Method;
use serde_json::json;

#[given(regex = r"^an OrchQueue API endpoint$")]
pub async fn given_api_endpoint(_world: &mut World) {}

#[when(regex = r"^I enqueue a completion task with valid payload$")]
pub async fn when_enqueue_valid_completion(world: &mut World) {
    world.push_fact("enqueue.valid");
    let task_id = "t-0".to_string();
    let body = json!({
        "task_id": task_id,
        "session_id": "s-0",
        "workload": "completion",
        "model_ref": "model0",
        "engine": "llamacpp",
        "ctx": 0,
        "priority": "interactive",
        "max_tokens": 1,
        "deadline_ms": 1000
    });
    world.task_id = Some("t-0".into());
    let _ = world.http_call(Method::POST, "/v1/tasks", Some(body)).await;
}

#[then(regex = r"^I receive 202 Accepted with correlation id$")]
pub async fn then_accepted_with_corr(world: &mut World) {
    assert_eq!(world.last_status, Some(http::StatusCode::ACCEPTED));
    assert!(world.corr_id.is_some(), "missing X-Correlation-Id header");
}

#[when(regex = r"^I stream task events$")]
pub async fn when_stream_events(world: &mut World) {
    world.push_fact("sse.start");
    let id = world.task_id.clone().unwrap_or_else(|| "t-0".into());
    let path = format!("/v1/tasks/{}/stream", id);
    let _ = world.http_call(Method::GET, &path, None).await;
}

// Alias to support scenarios that use "And I stream task events" after a Then
#[then(regex = r"^I stream task events$")]
pub async fn then_stream_events(world: &mut World) {
    when_stream_events(world).await;
}

#[then(regex = r"^I receive SSE events started, token, end$")]
pub async fn then_sse_started_token_end(world: &mut World) {
    let body = world
        .last_body
        .as_ref()
        .expect("expected SSE body from previous call");
    let a = body.find("event: started").unwrap_or(usize::MAX);
    let b = body.find("event: token").unwrap_or(usize::MAX);
    let c = body.find("event: end").unwrap_or(usize::MAX);
    assert!(a < b && b < c, "events not in order: {}", body);
}

#[then(regex = r"^I receive SSE metrics frames$")]
pub async fn then_sse_metrics_frames(world: &mut World) {
    let body = world.last_body.as_ref().expect("expected SSE body");
    assert!(
        body.contains("event: metrics"),
        "missing metrics event in SSE: {}",
        body
    );
}

#[then(regex = r"^started includes queue_position and predicted_start_ms$")]
pub async fn then_started_includes_queue_eta(world: &mut World) {
    // Prefer checking the live SSE response body if present
    if let Some(body) = world.last_body.as_ref() {
        if let Some(start_idx) = body.find("event: started") {
            let data_prefix = "data: ";
            let data_line = body[start_idx..].lines().nth(1).unwrap_or("");
            assert!(
                data_line.starts_with(data_prefix),
                "no data line after started: {}",
                data_line
            );
            let json_str = data_line.trim_start_matches(data_prefix).trim();
            let v: serde_json::Value = serde_json::from_str(json_str).expect("parse started data json");
            assert!(v.get("queue_position").is_some(), "started missing queue_position");
            assert!(v.get("predicted_start_ms").is_some(), "started missing predicted_start_ms");
            return;
        }
    }

    // Fall back to persisted transcript artifact (rendered by streaming service)
    let guard = world.state.artifacts.lock().unwrap();
    let mut found = false;
    for (_id, doc) in guard.iter() {
        if let Some(events) = doc.get("events").and_then(|e| e.as_array()) {
            for ev in events {
                if ev.get("type").and_then(|t| t.as_str()) == Some("started") {
                    if let Some(data) = ev.get("data") {
                        if data.get("queue_position").is_some()
                            && data.get("predicted_start_ms").is_some()
                        {
                            found = true;
                            break;
                        }
                    }
                }
            }
        }
        if found { break; }
    }
    assert!(found, "no started event");
}

#[then(regex = r"^SSE event ordering is per stream$")]
pub async fn then_sse_ordering_per_stream(world: &mut World) {
    let body = world.last_body.as_ref().expect("expected SSE body");
    let a = body.find("event: started").unwrap_or(usize::MAX);
    let b = body.find("event: token").unwrap_or(usize::MAX);
    let m = body.find("event: metrics").unwrap_or(usize::MAX);
    let c = body.find("event: end").unwrap_or(usize::MAX);
    assert!(a < b && b < m && m < c, "SSE order incorrect: {}", body);
}

#[given(regex = r"^queue full policy is reject$")]
pub async fn given_queue_policy_reject(world: &mut World) {
    world.push_fact("queue.policy.reject");
}

#[given(regex = r"^queue full policy is drop-lru$")]
pub async fn given_queue_policy_drop_lru(world: &mut World) {
    world.push_fact("queue.policy.drop_lru");
}

#[given(regex = r"^an OrchQueue API endpoint under load$")]
pub async fn given_api_under_load(_world: &mut World) {}

#[when(regex = r"^I enqueue a task beyond capacity$")]
pub async fn when_enqueue_beyond_capacity(world: &mut World) {
    world.push_fact("enqueue.beyond_capacity");
    let body = json!({
        "task_id": "t-over",
        "session_id": "s-0",
        "workload": "completion",
        "model_ref": "model0",
        "engine": "llamacpp",
        "ctx": 0,
        "priority": "interactive",
        "max_tokens": 1,
        "deadline_ms": 1000,
        // Sentinel to trigger 429 in handler glue
        "expected_tokens": 1000000
    });
    let _ = world.http_call(Method::POST, "/v1/tasks", Some(body)).await;
}

#[then(regex = r"^I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id$")]
pub async fn then_backpressure_headers(world: &mut World) {
    assert_eq!(world.last_status, Some(http::StatusCode::TOO_MANY_REQUESTS));
    let headers = world.last_headers.as_ref().expect("expected headers");
    assert!(headers.get("Retry-After").is_some());
    assert!(headers.get("X-Backoff-Ms").is_some());
    assert!(headers.get("X-Correlation-Id").is_some());
}

#[then(regex = r"^the error body includes policy_label retriable and retry_after_ms$")]
pub async fn then_error_body_advisory_fields(world: &mut World) {
    let body = world.last_body.as_ref().expect("expected body");
    let v: serde_json::Value = serde_json::from_str(body).expect("parse 429 body");
    assert!(v.get("policy_label").is_some());
    assert!(v.get("retriable").is_some());
    assert!(v.get("retry_after_ms").is_some());
}

#[given(regex = r"^an existing queued task$")]
pub async fn given_existing_queued_task(_world: &mut World) {}

#[when(regex = r"^I cancel the task$")]
pub async fn when_cancel_task(world: &mut World) {
    world.push_fact("cancel");
    let id = world.task_id.clone().unwrap_or_else(|| "t-0".into());
    let path = format!("/v1/tasks/{}/cancel", id);
    let _ = world.http_call(Method::POST, &path, None).await;
}

#[then(regex = r"^I receive 204 No Content with correlation id$")]
pub async fn then_no_content_with_corr(world: &mut World) {
    assert_eq!(world.last_status, Some(http::StatusCode::NO_CONTENT));
    assert!(world.corr_id.is_some());
}

#[given(regex = r"^a session id$")]
pub async fn given_session_id(_world: &mut World) {}

#[when(regex = r"^I query the session$")]
pub async fn when_query_session(world: &mut World) {
    world.push_fact("session.get");
    let id = world.task_id.clone().unwrap_or_else(|| "s-0".into());
    let path = format!("/v1/sessions/{}", id);
    let _ = world.http_call(Method::GET, &path, None).await;
}

#[then(regex = r"^I receive session info with ttl_ms_remaining turns kv_bytes kv_warmth$")]
pub async fn then_session_info_fields(world: &mut World) {
    let body = world.last_body.as_ref().expect("expected session body");
    let v: serde_json::Value = serde_json::from_str(body).expect("parse session info");
    for k in ["ttl_ms_remaining", "turns", "kv_bytes", "kv_warmth"] {
        assert!(v.get(k).is_some(), "missing {} in session info", k);
    }
}

#[when(regex = r"^I delete the session$")]
pub async fn when_delete_session(world: &mut World) {
    world.push_fact("session.delete");
    let id = world.task_id.clone().unwrap_or_else(|| "s-0".into());
    let path = format!("/v1/sessions/{}", id);
    let _ = world.http_call(Method::DELETE, &path, None).await;
}

// Alias to support scenarios that use Then/And for deletion
#[then(regex = r"^I delete the session$")]
pub async fn then_delete_session(world: &mut World) {
    when_delete_session(world).await;
}
