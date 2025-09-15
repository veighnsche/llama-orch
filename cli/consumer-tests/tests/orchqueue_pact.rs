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
                        "seed": 123456789,
                        "determinism": "strict",
                        "sampler_profile_version": "v1",
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
                "description": "stream task events (SSE placeholder)",
                "request": {
                    "method": "GET",
                    "path": "/v1/tasks/11111111-1111-4111-8111-111111111111/stream",
                    "headers": { "Accept": "text/event-stream" }
                },
                "response": {
                    "status": 200,
                    "headers": { "content-type": "text/event-stream" },
                    "body": "event: started\n\ndata: {\"queue_position\":3,\"predicted_start_ms\":420}\n\n\nevent: token\n\ndata: {\"t\":\"Hello\",\"i\":0}\n\n\nevent: end\n\ndata: {\"tokens_out\":1,\"decode_ms\":100}\n\n"
                }
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
            },
            {
                "description": "admission reject (policy)",
                "request": {
                    "method": "POST",
                    "path": "/v1/tasks",
                    "headers": { "content-type": "application/json" },
                    "body": {
                        "task_id": "55555555-5555-4555-8555-555555555555",
                        "session_id": "66666666-6666-4666-8666-666666666666",
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
                    "status": 429,
                    "headers": { "content-type": "application/json", "Retry-After": "2", "X-Backoff-Ms": "2000" },
                    "body": { "code": "ADMISSION_REJECT", "message": "reject policy", "engine": "llamacpp" }
                }
            },
            {
                "description": "invalid params",
                "request": {
                    "method": "POST",
                    "path": "/v1/tasks",
                    "headers": { "content-type": "application/json" },
                    "body": {
                        "task_id": "77777777-7777-4777-8777-777777777777",
                        "session_id": "88888888-8888-4888-8888-888888888888",
                        "workload": "completion",
                        "model_ref": "sha256:abc",
                        "engine": "llamacpp",
                        "ctx": 0,
                        "priority": "interactive",
                        "max_tokens": 0,
                        "deadline_ms": 30000
                    }
                },
                "response": {
                    "status": 400,
                    "headers": { "content-type": "application/json" },
                    "body": { "code": "INVALID_PARAMS", "message": "bad request", "engine": "llamacpp" }
                }
            },
            {
                "description": "pool unready",
                "request": { "method": "POST", "path": "/v1/tasks", "headers": { "content-type": "application/json" }, "body": {"task_id": "99999999-9999-4999-8999-999999999999", "session_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "workload": "completion", "model_ref": "sha256:abc", "engine": "llamacpp", "ctx": 8192, "priority": "interactive", "max_tokens": 64, "deadline_ms": 30000 } },
                "response": { "status": 503, "headers": { "content-type": "application/json" }, "body": { "code": "POOL_UNREADY", "message": "pool not ready", "engine": "llamacpp" } }
            },
            {
                "description": "pool unavailable",
                "request": { "method": "POST", "path": "/v1/tasks", "headers": { "content-type": "application/json" }, "body": {"task_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "session_id": "cccccccc-cccc-4ccc-8ccc-cccccccccccc", "workload": "completion", "model_ref": "sha256:abc", "engine": "llamacpp", "ctx": 8192, "priority": "interactive", "max_tokens": 64, "deadline_ms": 30000 } },
                "response": { "status": 503, "headers": { "content-type": "application/json" }, "body": { "code": "POOL_UNAVAILABLE", "message": "unavailable", "engine": "llamacpp" } }
            },
            {
                "description": "replica exhausted",
                "request": { "method": "POST", "path": "/v1/tasks", "headers": { "content-type": "application/json" }, "body": {"task_id": "dddddddd-dddd-4ddd-8ddd-dddddddddddd", "session_id": "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee", "workload": "completion", "model_ref": "sha256:abc", "engine": "llamacpp", "ctx": 8192, "priority": "interactive", "max_tokens": 64, "deadline_ms": 30000 } },
                "response": { "status": 503, "headers": { "content-type": "application/json" }, "body": { "code": "REPLICA_EXHAUSTED", "message": "exhausted", "engine": "llamacpp" } }
            },
            {
                "description": "decode timeout",
                "request": { "method": "POST", "path": "/v1/tasks", "headers": { "content-type": "application/json" }, "body": {"task_id": "ffffffff-ffff-4fff-8fff-ffffffffffff", "session_id": "12121212-1212-4121-8121-121212121212", "workload": "completion", "model_ref": "sha256:abc", "engine": "llamacpp", "ctx": 8192, "priority": "interactive", "max_tokens": 64, "deadline_ms": 30000 } },
                "response": { "status": 500, "headers": { "content-type": "application/json" }, "body": { "code": "DECODE_TIMEOUT", "message": "timeout", "engine": "llamacpp" } }
            },
            {
                "description": "worker reset",
                "request": { "method": "POST", "path": "/v1/tasks", "headers": { "content-type": "application/json" }, "body": {"task_id": "13131313-1313-4131-8131-131313131313", "session_id": "14141414-1414-4141-8141-141414141414", "workload": "completion", "model_ref": "sha256:abc", "engine": "llamacpp", "ctx": 8192, "priority": "interactive", "max_tokens": 64, "deadline_ms": 30000 } },
                "response": { "status": 500, "headers": { "content-type": "application/json" }, "body": { "code": "WORKER_RESET", "message": "reset", "engine": "llamacpp" } }
            },
            {
                "description": "internal error",
                "request": { "method": "POST", "path": "/v1/tasks", "headers": { "content-type": "application/json" }, "body": {"task_id": "15151515-1515-4151-8151-151515151515", "session_id": "16161616-1616-4161-8161-161616161616", "workload": "completion", "model_ref": "sha256:abc", "engine": "llamacpp", "ctx": 8192, "priority": "interactive", "max_tokens": 64, "deadline_ms": 30000 } },
                "response": { "status": 500, "headers": { "content-type": "application/json" }, "body": { "code": "INTERNAL", "message": "internal", "engine": "llamacpp" } }
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
