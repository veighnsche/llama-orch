use axum::{body::Body, http::Request, Router};
use http::StatusCode;
use orchestratord::app::router::build_router;
use orchestratord::state::AppState;
use serde_json::json;
use tower::util::ServiceExt as _; // for Router::oneshot

fn mk_app_with_env(cap: usize, policy: &str) -> Router {
    std::env::set_var("ORCHD_ADMISSION_CAPACITY", cap.to_string());
    std::env::set_var("ORCHD_ADMISSION_POLICY", policy);
    build_router(AppState::new())
}

fn task_body(task_id: &str) -> String {
    let body = json!({
        "task_id": task_id,
        "session_id": "s1",
        "workload": "completion",
        "model_ref": "m:stub",
        "engine": "llamacpp",
        "ctx": 1,
        "priority": "interactive",
        "max_tokens": 8,
        "deadline_ms": 1000
    });
    body.to_string()
}

#[tokio::test]
async fn post_tasks_happy_returns_202_and_body() {
    let app = mk_app_with_env(8, "reject");
    let req = Request::builder()
        .method(http::Method::POST)
        .uri("/v2/tasks")
        .header("X-API-Key", "valid")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(task_body("t-1")))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    // Budget headers must be present
    let h = resp.headers();
    assert!(h.contains_key("X-Budget-Tokens-Remaining"));
    assert!(h.contains_key("X-Budget-Time-Remaining-Ms"));
    assert!(h.contains_key("X-Budget-Cost-Remaining"));
    // Body has admission fields
    let body_bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(v["task_id"], "t-1");
    assert!(v.get("queue_position").is_some());
    assert!(v.get("predicted_start_ms").is_some());
}

#[tokio::test]
async fn post_tasks_queue_full_returns_429_with_headers() {
    let app = mk_app_with_env(1, "reject");
    // First is accepted
    let req1 = Request::builder()
        .method(http::Method::POST)
        .uri("/v2/tasks")
        .header("X-API-Key", "valid")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(task_body("t-a")))
        .unwrap();
    let resp1 = app.clone().oneshot(req1).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::ACCEPTED);

    // Second should hit reject policy -> 429
    let req2 = Request::builder()
        .method(http::Method::POST)
        .uri("/v2/tasks")
        .header("X-API-Key", "valid")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(task_body("t-b")))
        .unwrap();
    let resp2 = app.clone().oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::TOO_MANY_REQUESTS);
    let headers = resp2.headers().clone();
    assert!(headers.contains_key("X-Correlation-Id"));
    assert_eq!(headers.get("Retry-After").unwrap(), "1");
    assert_eq!(headers.get("X-Backoff-Ms").unwrap(), "1000");
}

#[tokio::test]
async fn cancel_then_stream_yields_no_tokens() {
    let app = mk_app_with_env(8, "reject");
    // Issue cancel first
    let cancel = Request::builder()
        .method(http::Method::POST)
        .uri("/v2/tasks/t-cancel/cancel")
        .header("X-API-Key", "valid")
        .body(Body::empty())
        .unwrap();
    let cancel_resp = app.clone().oneshot(cancel).await.unwrap();
    assert_eq!(cancel_resp.status(), StatusCode::NO_CONTENT);

    // Now stream
    let stream = Request::builder()
        .method(http::Method::GET)
        .uri("/v2/tasks/t-cancel/events")
        .header("X-API-Key", "valid")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(stream).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 16384).await.unwrap();
    let s = String::from_utf8(body.to_vec()).unwrap();
    assert!(s.contains("event: started"));
    assert!(s.contains("event: end"));
    assert!(!s.contains("event: token"));
    assert!(!s.contains("event: metrics"));
}
