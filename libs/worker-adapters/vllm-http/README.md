# vllm-http

**vLLM HTTP server adapter**

`libs/worker-adapters/vllm-http` — WorkerAdapter implementation for vLLM HTTP server.

---

## What This Adapter Does

vllm-http provides **vLLM integration** for llama-orch:

- **OpenAI-compatible API** — Maps to vLLM's `/v1/completions` endpoint
- **SSE streaming** — Streams tokens via Server-Sent Events
- **Continuous batching** — Leverages vLLM's continuous batching
- **PagedAttention** — Benefits from vLLM's memory-efficient attention
- **High throughput** — Optimized for serving workloads

**Engine**: vLLM HTTP server (default port: 8000)

---

## Usage

### Create Adapter

```rust
use worker_adapters_vllm_http::VllmHttpAdapter;

let adapter = VllmHttpAdapter::new("http://localhost:8000");
```

### Submit Task

```rust
use worker_adapters_adapter_api::{WorkerAdapter, TaskRequest};

let task = TaskRequest {
    job_id: "job-123".to_string(),
    model: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
    prompt: "Hello, world!".to_string(),
    max_tokens: 100,
    temperature: Some(0.7),
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

## vLLM API Mapping

### Completions Endpoint

**Orchestrator Request** → **vLLM Request**

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "Hello, world!",
  "max_tokens": 100,
  "temperature": 0.7,
  "seed": 42,
  "stream": true
}
```

**vLLM Response** (SSE):

```
data: {"choices":[{"text":"Hello","index":0}]}

data: {"choices":[{"text":" there","index":0}]}

data: {"choices":[{"text":"!","index":0,"finish_reason":"stop"}]}

data: [DONE]
```

---

## vLLM Features

### Supported

- **Streaming** — SSE token streaming
- **Continuous batching** — Automatic request batching
- **PagedAttention** — Memory-efficient KV cache
- **Tensor parallelism** — Multi-GPU support
- **OpenAI compatibility** — Standard API format

### Not Yet Supported

- **Prefix caching** — Shared prompt prefixes
- **Speculative decoding** — Draft model acceleration
- **Multi-modal** — Vision models

---

## Configuration

### Environment Variables (Optional)

For orchestratord integration:

- `ORCHD_VLLM_URL` — vLLM base URL (e.g., `http://localhost:8000`)
- `ORCHD_VLLM_POOL` — Pool ID (default: `default`)
- `ORCHD_VLLM_REPLICA` — Replica ID (default: `r0`)

### Bearer Token

```rust
use worker_adapters_http_util::auth::with_bearer;

let client = reqwest::Client::new();
let request = client.post("http://localhost:8000/v1/completions");
let request = with_bearer(request, "my-secret-token");
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

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p worker-adapters-vllm-http -- --nocapture

# Run specific test
cargo test -p worker-adapters-vllm-http -- test_submit --nocapture
```

### Integration Tests

Integration tests use a mock server to simulate vLLM SSE responses:

```rust
#[tokio::test]
async fn test_streaming() {
    let mock_server = MockServer::start().await;
    
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_string("data: {\"choices\":[{\"text\":\"Hello\"}]}\n\n"))
        .mount(&mock_server)
        .await;
    
    let adapter = VllmHttpAdapter::new(&mock_server.uri());
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
