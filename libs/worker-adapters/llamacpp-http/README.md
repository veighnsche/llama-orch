# llamacpp-http

**llama.cpp HTTP server adapter**

`libs/worker-adapters/llamacpp-http` — WorkerAdapter implementation for llama.cpp HTTP server.

---

## What This Adapter Does

llamacpp-http provides **llama.cpp integration** for llama-orch:

- **HTTP API mapping** — Maps orchestrator requests to llama.cpp `/completion` endpoint
- **SSE streaming** — Streams tokens via Server-Sent Events
- **Deterministic sampling** — Supports seed-based determinism
- **KV cache** — Leverages llama.cpp's slot-based KV cache
- **Health checks** — Monitors engine availability

**Engine**: llama.cpp HTTP server (default port: 8081)

---

## Usage

### Create Adapter

```rust
use worker_adapters_llamacpp_http::LlamaCppHttpAdapter;

let adapter = LlamaCppHttpAdapter::new("http://localhost:8081");
```

### Submit Task

```rust
use worker_adapters_adapter_api::{WorkerAdapter, TaskRequest};

let task = TaskRequest {
    job_id: "job-123".to_string(),
    model: "llama-3.1-8b-instruct".to_string(),
    prompt: "Hello, world!".to_string(),
    max_tokens: 100,
    temperature: Some(0.0),
    seed: Some(42),
    session_id: None,
};

let mut stream = adapter.submit(task).await?;

while let Some(event) = stream.receiver.recv().await {
    match event {
        TokenEvent::Started { engine_version } => {
            println!("Started: {}", engine_version);
        }
        TokenEvent::Token { text, index } => {
            print!("{}", text);
        }
        TokenEvent::End { metrics } => {
            println!("\nDone: {} tokens", metrics.tokens_generated);
        }
        TokenEvent::Error { error } => {
            eprintln!("Error: {}", error);
        }
    }
}
```

---

## llama.cpp API Mapping

### Completion Endpoint

**Orchestrator Request** → **llama.cpp Request**

```json
{
  "prompt": "Hello, world!",
  "n_predict": 100,
  "temperature": 0.0,
  "top_p": 1.0,
  "seed": 42,
  "stream": true
}
```

**llama.cpp Response** (SSE):

```
data: {"content":"Hello","stop":false}

data: {"content":" there","stop":false}

data: {"content":"!","stop":true}
```

---

## Deterministic Sampling

For deterministic output:

1. Set `temperature: 0.0` (greedy decoding)
2. Set `top_p: 1.0` (no nucleus sampling)
3. Provide `seed` value
4. Use identical model weights and llama.cpp version

```rust
let task = TaskRequest {
    temperature: Some(0.0),
    seed: Some(42),
    // ... other fields
};
```

---

## Health Check

```rust
let health = adapter.health().await?;

match health.state {
    HealthState::Healthy => println!("Engine is healthy"),
    HealthState::Degraded => println!("Engine is degraded"),
    HealthState::Unhealthy => println!("Engine is unhealthy"),
}
```

---

## Configuration

### Environment Variables (Optional)

For orchestratord integration:

- `ORCHD_LLAMACPP_URL` — llama.cpp base URL (e.g., `http://localhost:8081`)
- `ORCHD_LLAMACPP_POOL` — Pool ID (default: `default`)
- `ORCHD_LLAMACPP_REPLICA` — Replica ID (default: `r0`)

### Bearer Token

```rust
use worker_adapters_http_util::auth::with_bearer;

let client = reqwest::Client::new();
let request = client.post("http://localhost:8081/completion");
let request = with_bearer(request, "my-secret-token");
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p worker-adapters-llamacpp-http -- --nocapture

# Run specific test
cargo test -p worker-adapters-llamacpp-http -- test_submit --nocapture
```

### Integration Tests

Integration tests use a mock Axum server to simulate llama.cpp SSE responses:

```rust
#[tokio::test]
async fn test_streaming() {
    let mock_server = MockServer::start().await;
    
    Mock::given(method("POST"))
        .and(path("/completion"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_string("data: {\"content\":\"Hello\"}\n\n"))
        .mount(&mock_server)
        .await;
    
    let adapter = LlamaCppHttpAdapter::new(&mock_server.uri());
    // Test streaming
}
```

---

## Dependencies

### Internal

- `worker-adapters-adapter-api` — WorkerAdapter trait
- `worker-adapters-http-util` — HTTP client, retry, streaming

### External

- `reqwest` — HTTP client
- `tokio` — Async runtime
- `serde` — Serialization
- `async-trait` — Async trait support

---

## llama.cpp Features

### Supported

- **Streaming** — SSE token streaming
- **Deterministic sampling** — Seed-based reproducibility
- **Greedy decoding** — temperature=0
- **KV cache** — Slot-based caching
- **Multiple slots** — Concurrent requests

### Not Yet Supported

- **Session continuation** — Requires session_id mapping
- **Embeddings** — Different endpoint
- **Multi-modal** — Image inputs

---

## Specifications

Implements requirements from:
- ORCH-3054 (Adapter registry)
- ORCH-3055 (Adapter dispatch)
- ORCH-3056 (Adapter lifecycle)
- ORCH-3057 (Health checks)
- ORCH-3058 (Error handling)

See `.specs/00_llama-orch.md` for full requirements.

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
