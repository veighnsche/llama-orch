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
