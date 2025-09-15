use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn pact_stub_happy_path() {
    let server = MockServer::start().await;

    // POST /v1/tasks → 202 AdmissionResponse
    Mock::given(method("POST"))
        .and(path("/v1/tasks"))
        .respond_with(
            ResponseTemplate::new(202)
                .insert_header("content-type", "application/json")
                .set_body_json(json!({
                    "task_id": "11111111-1111-4111-8111-111111111111",
                    "queue_position": 3,
                    "predicted_start_ms": 420,
                    "backoff_ms": 0
                })),
        )
        .mount(&server)
        .await;

    // Cancel → 204
    Mock::given(method("POST"))
        .and(path(
            "/v1/tasks/11111111-1111-4111-8111-111111111111/cancel",
        ))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;

    // Session GET → 200
    Mock::given(method("GET"))
        .and(path("/v1/sessions/22222222-2222-4222-8222-222222222222"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(json!({
                    "ttl_ms_remaining": 600000,
                    "turns": 1,
                    "kv_bytes": 0,
                    "kv_warmth": false
                })),
        )
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/tasks", server.uri()))
        .json(&json!({
            "task_id": "11111111-1111-4111-8111-111111111111",
            "session_id": "22222222-2222-4222-8222-222222222222",
            "workload": "completion",
            "model_ref": "sha256:abc",
            "engine": "llamacpp",
            "ctx": 8192,
            "priority": "interactive",
            "max_tokens": 64,
            "deadline_ms": 30000
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 202);
    let admission: serde_json::Value = resp.json().await.unwrap();
    assert!(admission.get("task_id").is_some());
    assert!(admission.get("queue_position").is_some());
    assert!(admission.get("predicted_start_ms").is_some());
    assert!(admission.get("backoff_ms").is_some());
}

#[tokio::test]
async fn pact_stub_sse_and_backpressure_headers() {
    let server = MockServer::start().await;

    // GET /v1/tasks/:id/stream → SSE transcript
    Mock::given(method("GET"))
        .and(path("/v1/tasks/11111111-1111-4111-8111-111111111111/stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(
                    "event: started\n\n\
                     data: {\"queue_position\":3,\"predicted_start_ms\":420}\n\n\
                     event: token\n\n\
                     data: {\"t\":\"Hello\",\"i\":0}\n\n\
                     event: end\n\n\
                     data: {\"tokens_out\":1,\"decode_ms\":100}\n\n",
                ),
        )
        .mount(&server)
        .await;

    // POST /v1/tasks → 429 with backpressure headers
    Mock::given(method("POST"))
        .and(path("/v1/tasks"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("content-type", "application/json")
                .insert_header("Retry-After", "1")
                .insert_header("X-Backoff-Ms", "1000")
                .set_body_json(json!({
                    "code": "QUEUE_FULL_DROP_LRU",
                    "message": "queue full",
                    "engine": "llamacpp"
                })),
        )
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let sse = client
        .get(format!(
            "{}/v1/tasks/11111111-1111-4111-8111-111111111111/stream",
            server.uri()
        ))
        .header("Accept", "text/event-stream")
        .send()
        .await
        .unwrap();
    assert_eq!(sse.status(), 200);
    let body = sse.text().await.unwrap();
    // Basic shape checks for 3 events
    assert!(body.contains("event: started"));
    assert!(body.contains("event: token"));
    assert!(body.contains("event: end"));

    // Backpressure 429
    let resp = client
        .post(format!("{}/v1/tasks", server.uri()))
        .json(&json!({"task_id":"x","session_id":"y"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 429);
    let retry_after = resp.headers().get("Retry-After").unwrap();
    let backoff_ms = resp.headers().get("X-Backoff-Ms").unwrap();
    assert_eq!(retry_after.to_str().unwrap(), "1");
    assert_eq!(backoff_ms.to_str().unwrap(), "1000");
}
