use api::{Engine, Priority, Workload};
use axum::{routing::post, Router};
use contracts_api_types as api;
use futures::StreamExt;
use tokio::task::JoinHandle;
use worker_adapters_adapter_api::WorkerAdapter;
use worker_adapters_llamacpp_http::LlamaCppHttpAdapter;

async fn start_stub_server(body: String) -> (u16, JoinHandle<()>) {
    // TODO(OwnerB-LLAMACPP-TEST-STUB): Stub Axum server returns a pre-canned SSE body.
    // Why: Enables deterministic unit/integration tests for ordering/indices without
    // requiring a real llama.cpp binary in CI. Future work: simulate chunked SSE with
    // hyper::Body streaming and add cancel-on-disconnect coverage.
    let app = Router::new().route(
        "/completion",
        post(move || {
            let body = body.clone();
            async move { body.clone() }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (port, handle)
}

fn mk_request(seed: i64, max_tokens: i32) -> api::TaskRequest {
    api::TaskRequest {
        task_id: "t1".into(),
        session_id: "s1".into(),
        workload: Workload::Completion,
        model_ref: "m1".into(),
        engine: Engine::Llamacpp,
        ctx: 4096,
        priority: Priority::Interactive,
        seed: Some(seed),
        determinism: None,
        sampler_profile_version: None,
        prompt: Some("Hello".into()),
        inputs: None,
        max_tokens,
        deadline_ms: 5_000,
        expected_tokens: None,
        kv_hint: None,
        placement: None,
    }
}

#[tokio::test]
async fn stream_order_and_indices() {
    // TODO(OwnerB-LLAMACPP-TEST-STUB): This test asserts ordering/indices against a stub
    // server. Extend to chunked bodies and intermittent metrics frames, and add retries
    // classification tests using http-util helpers.
    // SSE body with started, two tokens, end
    let body = "event: started\n\
                data: {\"ts\":1}\n\
                \n\
                event: token\n\
                data: {\"i\":0,\"t\":\"He\"}\n\
                \n\
                event: token\n\
                data: {\"i\":1,\"t\":\"llo\"}\n\
                \n\
                event: end\n\
                data: {\"tokens_out\":2}\n"
        .to_string();
    let (port, _srv) = start_stub_server(body).await;

    let adapter = LlamaCppHttpAdapter::new(format!("http://127.0.0.1:{}", port));
    let req = mk_request(42, 2);
    let mut stream = adapter.submit(req).expect("stream");

    let mut kinds = Vec::new();
    let mut indices = Vec::new();
    let mut text = String::new();
    while let Some(ev) = stream.next().await {
        let ev = ev.expect("adapter event");
        kinds.push(ev.kind.clone());
        if ev.kind == "token" {
            let i = ev.data.get("i").and_then(|x| x.as_u64()).unwrap() as usize;
            let t = ev.data.get("t").and_then(|x| x.as_str()).unwrap().to_string();
            indices.push(i);
            text.push_str(&t);
        }
    }
    // order: started before token, end after token
    let a = kinds.iter().position(|k| k == "started").unwrap();
    let b = kinds.iter().position(|k| k == "token").unwrap();
    let c = kinds.iter().position(|k| k == "end").unwrap();
    assert!(a < b && b < c);
    // indices increasing
    for w in indices.windows(2) {
        assert!(w[0] < w[1]);
    }
    assert_eq!(text, "Hello");
}

#[tokio::test]
async fn determinism_with_seed() {
    // TODO(OwnerB-LLAMACPP-TEST-DETERMINISM): This verifies identical outputs for identical
    // seeds against the stub server. Future: point at a real llama.cpp instance configured
    // with deterministic flags (parallel=1, no-cont-batching) and assert byte-exact tokens.
    let body = "event: started\n\
                data: {}\n\
                \n\
                event: token\n\
                data: {\"i\":0,\"t\":\"A\"}\n\
                \n\
                event: token\n\
                data: {\"i\":1,\"t\":\"B\"}\n\
                \n\
                event: end\n\
                data: {}\n"
        .to_string();
    let (port, _srv) = start_stub_server(body).await;

    let adapter = LlamaCppHttpAdapter::new(format!("http://127.0.0.1:{}", port));
    let req1 = mk_request(123, 2);
    let req2 = mk_request(123, 2);

    let mut s1 = adapter.submit(req1).expect("stream1");
    let mut s2 = adapter.submit(req2).expect("stream2");

    let mut out1 = String::new();
    while let Some(ev) = s1.next().await {
        if let Ok(e) = ev {
            if e.kind == "token" {
                out1.push_str(e.data.get("t").and_then(|x| x.as_str()).unwrap());
            }
        }
    }
    let mut out2 = String::new();
    while let Some(ev) = s2.next().await {
        if let Ok(e) = ev {
            if e.kind == "token" {
                out2.push_str(e.data.get("t").and_then(|x| x.as_str()).unwrap());
            }
        }
    }

    assert_eq!(out1, out2);
}
