use serde_json::json;
use std::{fs, path::PathBuf};

fn pacts_dir() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf();
    root.join("contracts/pacts")
}

#[test]
fn pact_enqueue_accept_and_cancel_and_session_endpoints() {
    let out = pacts_dir();
    let _ = fs::create_dir_all(&out);

    let pact = json!({
        "pactSpecification": { "version": "3.0.0" },
        "consumer": { "name": "cli-consumer" },
        "provider": { "name": "orchestratord" },
        "interactions": [
            {
                "description": "enqueue task accepted",
                "request": {
                    "method": "POST",
                    "path": "/v1/tasks",
                    "headers": { "content-type": "application/json" },
                    "body": {
                        "task_id": "11111111-1111-4111-8111-111111111111",
                        "session_id": "22222222-2222-4222-8222-222222222222",
                        "workload": "completion",
                        "model_ref": "sha256:abc",
                        "engine": "llamacpp",
                        "ctx": 8192,
                        "priority": "interactive",
                        "max_tokens": 64,
                        "deadline_ms": 30000
                    }
                },
                "response": {
                    "status": 202,
                    "headers": { "content-type": "application/json" },
                    "body": {
                        "task_id": "11111111-1111-4111-8111-111111111111",
                        "queue_position": 3,
                        "predicted_start_ms": 420,
                        "backoff_ms": 0
                    }
                }
            },
            {
                "description": "cancel task no content",
                "request": { "method": "POST", "path": "/v1/tasks/11111111-1111-4111-8111-111111111111/cancel" },
                "response": { "status": 204 }
            },
            {
                "description": "get session info",
                "request": { "method": "GET", "path": "/v1/sessions/22222222-2222-4222-8222-222222222222" },
                "response": {
                    "status": 200,
                    "headers": { "content-type": "application/json" },
                    "body": { "ttl_ms_remaining": 600000, "turns": 1, "kv_bytes": 0, "kv_warmth": false }
                }
            },
            {
                "description": "delete session",
                "request": { "method": "DELETE", "path": "/v1/sessions/22222222-2222-4222-8222-222222222222" },
                "response": { "status": 204 }
            },
            {
                "description": "enqueue task queue full",
                "request": {
                    "method": "POST",
                    "path": "/v1/tasks",
                    "headers": { "content-type": "application/json" },
                    "body": {
                        "task_id": "33333333-3333-4333-8333-333333333333",
                        "session_id": "44444444-4444-4444-8444-444444444444",
                        "workload": "completion",
                        "model_ref": "sha256:abc",
                        "engine": "llamacpp",
                        "ctx": 8192,
                        "priority": "batch",
                        "max_tokens": 64,
                        "deadline_ms": 30000
                    }
                },
                "response": {
                    "status": 429,
                    "headers": { "content-type": "application/json", "Retry-After": "1", "X-Backoff-Ms": "1000" },
                    "body": { "code": "QUEUE_FULL_DROP_LRU", "message": "queue full", "engine": "llamacpp" }
                }
            }
        ]
    });

    let path = out.join("cli-consumer-orchestratord.json");
    fs::write(&path, serde_json::to_string_pretty(&pact).unwrap()).unwrap();

    // Basic shape assertion on the accepted response body
    let first = &pact["interactions"][0]["response"]["body"];
    assert!(first.get("task_id").is_some());
    assert!(first.get("queue_position").is_some());
    assert!(first.get("predicted_start_ms").is_some());
    assert!(first.get("backoff_ms").is_some());
}
