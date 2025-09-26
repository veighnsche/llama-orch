use crate::steps::world::World;
use axum::body::{to_bytes, Body};
use cucumber::{given, then, when};
use http::Method;
use http::Request;
use serde_json::json;
use tokio::time::{sleep, Duration};
use tower::util::ServiceExt as _;

#[given(regex = r"^an OrchQueue API endpoint$")]
pub async fn given_api_endpoint(_world: &mut World) {}

#[then(regex = r"^no further token events are emitted$")]
pub async fn then_no_further_token_events(world: &mut World) {
    let body = world.last_body.as_ref().expect("expected SSE body");
    let count = body.matches("event: token").count();
    assert_eq!(count, 1, "expected exactly one token event, got {}: {}", count, body);
    assert!(!body.contains("event: metrics"), "expected no metrics event after cancel: {}", body);
}

// Aliases to support scenarios that use And/Then with the same text
#[then(regex = r"^I enqueue a completion task with valid payload$")]
pub async fn then_enqueue_valid_completion(world: &mut World) {
    when_enqueue_valid_completion(world).await;
}

#[given(regex = r"^I enqueue a completion task with valid payload$")]
pub async fn given_enqueue_valid_completion(world: &mut World) {
    when_enqueue_valid_completion(world).await;
}

#[then(regex = r"^budget headers are present$")]
pub async fn then_budget_headers_present(world: &mut World) {
    let headers = world.last_headers.as_ref().expect("expected headers on last response");
    for k in ["X-Budget-Tokens-Remaining", "X-Budget-Time-Remaining-Ms", "X-Budget-Cost-Remaining"]
    {
        assert!(headers.get(k).is_some(), "missing header {}", k);
    }
}

#[then(regex = r"^SSE transcript artifact exists with events started token metrics end$")]
pub async fn then_sse_transcript_artifact_exists(world: &mut World) {
    let guard = world.state.artifacts.lock().unwrap();
    let mut found = false;
    for (_id, doc) in guard.iter() {
        if let Some(events) = doc.get("events").and_then(|e| e.as_array()) {
            let mut got_started = false;
            let mut got_token = false;
            let mut got_metrics = false;
            let mut got_end = false;
            for ev in events {
                if let Some(t) = ev.get("type").and_then(|t| t.as_str()) {
                    match t {
                        "started" => got_started = true,
                        "token" => got_token = true,
                        "metrics" => got_metrics = true,
                        "end" => got_end = true,
                        _ => {}
                    }
                }
            }
            if got_started && got_token && got_metrics && got_end {
                found = true;
                break;
            }
        }
    }
    assert!(found, "expected persisted SSE transcript artifact with all events");
}

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

#[when(regex = r"^I enqueue a task way beyond capacity$")]
pub async fn when_enqueue_way_beyond_capacity(world: &mut World) {
    world.push_fact("enqueue.way_beyond_capacity");
    let body = json!({
        "task_id": "t-over2",
        "session_id": "s-0",
        "workload": "completion",
        "model_ref": "model0",
        "engine": "llamacpp",
        "ctx": 0,
        "priority": "interactive",
        "max_tokens": 1,
        "deadline_ms": 1000,
        // Sentinel to trigger drop-lru 429 in handler glue
        "expected_tokens": 2000000
    });
    let _ = world.http_call(Method::POST, "/v1/tasks", Some(body)).await;
}

#[then(regex = r"^error envelope code is ADMISSION_REJECT$")]
pub async fn then_error_code_is_admission_reject(world: &mut World) {
    let body = world.last_body.as_ref().expect("expected body");
    let v: serde_json::Value = serde_json::from_str(body).expect("parse body");
    assert_eq!(v["code"], "ADMISSION_REJECT");
}

#[then(regex = r"^error envelope code is QUEUE_FULL_DROP_LRU$")]
pub async fn then_error_code_is_drop_lru(world: &mut World) {
    let body = world.last_body.as_ref().expect("expected body");
    let v: serde_json::Value = serde_json::from_str(body).expect("parse body");
    assert_eq!(v["code"], "QUEUE_FULL_DROP_LRU");
}

#[then(regex = r"^I receive 202 Accepted with correlation id$")]
pub async fn then_accepted_with_corr(world: &mut World) {
    assert_eq!(world.last_status, Some(http::StatusCode::ACCEPTED));
    assert!(world.corr_id.is_some(), "missing X-Correlation-Id header");
}

#[given(regex = r"^I receive 202 Accepted with correlation id$")]
pub async fn given_accepted_with_corr(world: &mut World) {
    then_accepted_with_corr(world).await;
}

#[when(regex = r"^I receive 202 Accepted with correlation id$")]
pub async fn when_accepted_with_corr(world: &mut World) {
    then_accepted_with_corr(world).await;
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
    let body = world.last_body.as_ref().expect("expected SSE body from previous call");
    let a = body.find("event: started").unwrap_or(usize::MAX);
    let b = body.find("event: token").unwrap_or(usize::MAX);
    let c = body.find("event: end").unwrap_or(usize::MAX);
    assert!(a < b && b < c, "events not in order: {}", body);
}

#[then(regex = r"^I receive SSE metrics frames$")]
pub async fn then_sse_metrics_frames(world: &mut World) {
    let body = world.last_body.as_ref().expect("expected SSE body");
    assert!(body.contains("event: metrics"), "missing metrics event in SSE: {}", body);
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
            let v: serde_json::Value =
                serde_json::from_str(json_str).expect("parse started data json");
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
        if found {
            break;
        }
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

#[when(regex = r"^I stream task events while canceling mid-stream$")]
pub async fn when_stream_while_cancel_mid(world: &mut World) {
    world.push_fact("sse.cancel.mid");
    let id = world.task_id.clone().unwrap_or_else(|| "t-0".into());

    // Build app router with shared state
    let app = orchestratord::app::router::build_router(world.state.clone());

    // Spawn a cancel request after a short delay so it lands between tokens
    let cancel_app = app.clone();
    let cancel_key = world.api_key.clone();
    let cancel_path = format!("/v1/tasks/{}/cancel", id.clone());
    tokio::spawn(async move {
        sleep(Duration::from_millis(10)).await;
        let mut req = Request::builder().method(http::Method::POST).uri(cancel_path);
        if let Some(key) = cancel_key {
            req = req.header("X-API-Key", key);
        }
        let req = req.body(Body::empty()).unwrap();
        let _ = cancel_app.oneshot(req).await;
    });

    // Now start the stream request
    let stream_path = format!("/v1/tasks/{}/stream", id);
    let mut req = Request::builder().method(http::Method::GET).uri(stream_path);
    if let Some(key) = &world.api_key {
        req = req.header("X-API-Key", key);
    }
    let req = req.body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.expect("stream oneshot resp");

    let status = resp.status();
    let headers_out = resp.headers().clone();
    let body_bytes = to_bytes(resp.into_body(), 1_048_576).await.unwrap_or_default();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap_or_default();

    world.corr_id =
        headers_out.get("X-Correlation-Id").and_then(|v| v.to_str().ok()).map(|s| s.to_string());
    world.last_status = Some(status);
    world.last_headers = Some(headers_out);
    world.last_body = Some(body_str);
}
