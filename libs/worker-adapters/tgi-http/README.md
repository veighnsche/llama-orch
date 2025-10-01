# tgi-http

**Text Generation Inference (TGI) HTTP server adapter**

`libs/worker-adapters/tgi-http` — WorkerAdapter implementation for Hugging Face TGI server.

---

## What This Adapter Does

tgi-http provides **TGI integration** for llama-orch:

- **TGI native API** — Maps to TGI's `/generate_stream` endpoint
- **SSE streaming** — Streams tokens via Server-Sent Events
- **Flash Attention** — Leverages TGI's optimized attention
- **Tensor parallelism** — Multi-GPU support
- **Production-ready** — Battle-tested serving infrastructure

**Engine**: Text Generation Inference server (default port: 3000)

---

## Usage

### Create Adapter

```rust
use worker_adapters_tgi_http::TgiHttpAdapter;

let adapter = TgiHttpAdapter::new("http://localhost:3000");
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

## TGI API Mapping

### Generate Stream Endpoint

**Orchestrator Request** → **TGI Request**

```json
{
  "inputs": "Hello, world!",
  "parameters": {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "seed": 42,
    "do_sample": true
  },
  "stream": true
}
```

**TGI Response** (SSE):

```
data: {"token":{"id":12345,"text":"Hello","logprob":-0.5}}

data: {"token":{"id":12346,"text":" there","logprob":-0.3}}

data: {"token":{"id":12347,"text":"!","logprob":-0.2},"generated_text":"Hello there!"}
```

---

## TGI Features

### Supported

- **Streaming** — SSE token streaming
- **Flash Attention** — Optimized attention kernels
- **Tensor parallelism** — Multi-GPU inference
- **Quantization** — GPTQ, AWQ support
- **Token details** — Logprobs, token IDs

### Not Yet Supported

- **Guided generation** — JSON schema constraints
- **Grammar-based sampling** — Structured outputs
- **Watermarking** — Text watermarking

---

## Configuration

### Environment Variables (Optional)

For orchestratord integration:

- `ORCHD_TGI_URL` — TGI base URL (e.g., `http://localhost:3000`)
- `ORCHD_TGI_POOL` — Pool ID (default: `default`)
- `ORCHD_TGI_REPLICA` — Replica ID (default: `r0`)

### Bearer Token

```rust
use worker_adapters_http_util::auth::with_bearer;

let client = reqwest::Client::new();
let request = client.post("http://localhost:3000/generate_stream");
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
cargo test -p worker-adapters-tgi-http -- --nocapture

# Run specific test
cargo test -p worker-adapters-tgi-http -- test_submit --nocapture
```

### Integration Tests

Integration tests use a mock server to simulate TGI SSE responses:

```rust
#[tokio::test]
async fn test_streaming() {
    let mock_server = MockServer::start().await;
    
    Mock::given(method("POST"))
        .and(path("/generate_stream"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_string("data: {\"token\":{\"text\":\"Hello\"}}\n\n"))
        .mount(&mock_server)
        .await;
    
    let adapter = TgiHttpAdapter::new(&mock_server.uri());
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
