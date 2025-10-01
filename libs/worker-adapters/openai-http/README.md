# openai-http

**OpenAI-compatible HTTP API adapter**

`libs/worker-adapters/openai-http` — WorkerAdapter implementation for OpenAI-compatible HTTP endpoints.

---

## What This Adapter Does

openai-http provides **OpenAI API compatibility** for llama-orch:

- **Standard API format** — Uses OpenAI `/v1/completions` format
- **SSE streaming** — Streams tokens via Server-Sent Events
- **Wide compatibility** — Works with any OpenAI-compatible server
- **Drop-in replacement** — Compatible with OpenAI client libraries
- **Flexible** — Can target OpenAI, Azure OpenAI, or local servers

**Engine**: Any OpenAI-compatible HTTP server

---

## Usage

### Create Adapter

```rust
use worker_adapters_openai_http::OpenAiHttpAdapter;

// Local OpenAI-compatible server
let adapter = OpenAiHttpAdapter::new("http://localhost:8000");

// Or OpenAI API
let adapter = OpenAiHttpAdapter::new("https://api.openai.com");
```

### Submit Task

```rust
use worker_adapters_adapter_api::{WorkerAdapter, TaskRequest};

let task = TaskRequest {
    job_id: "job-123".to_string(),
    model: "gpt-3.5-turbo".to_string(),
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

## OpenAI API Mapping

### Completions Endpoint

**Orchestrator Request** → **OpenAI Request**

```json
{
  "model": "gpt-3.5-turbo",
  "prompt": "Hello, world!",
  "max_tokens": 100,
  "temperature": 0.7,
  "seed": 42,
  "stream": true
}
```

**OpenAI Response** (SSE):

```
data: {"id":"cmpl-123","choices":[{"text":"Hello","index":0}]}

data: {"id":"cmpl-123","choices":[{"text":" there","index":0}]}

data: {"id":"cmpl-123","choices":[{"text":"!","index":0,"finish_reason":"stop"}]}

data: [DONE]
```

---

## Compatible Servers

### Supported

- **OpenAI API** — Official OpenAI endpoints
- **Azure OpenAI** — Microsoft Azure OpenAI Service
- **vLLM** — With OpenAI compatibility mode
- **Text Generation WebUI** — OpenAI-compatible mode
- **LocalAI** — OpenAI-compatible local server
- **Ollama** — With OpenAI compatibility

### Configuration Examples

#### OpenAI API

```rust
let adapter = OpenAiHttpAdapter::new("https://api.openai.com");
// Set API key via bearer token
```

#### Azure OpenAI

```rust
let adapter = OpenAiHttpAdapter::new("https://your-resource.openai.azure.com");
// Set API key and deployment name
```

#### Local vLLM

```rust
let adapter = OpenAiHttpAdapter::new("http://localhost:8000");
// No authentication required for local
```

---

## Configuration

### Environment Variables (Optional)

For orchestratord integration:

- `ORCHD_OPENAI_URL` — OpenAI-compatible base URL
- `ORCHD_OPENAI_POOL` — Pool ID (default: `default`)
- `ORCHD_OPENAI_REPLICA` — Replica ID (default: `r0`)
- `ORCHD_OPENAI_API_KEY` — API key for authentication

### Bearer Token

```rust
use worker_adapters_http_util::auth::with_bearer;

let client = reqwest::Client::new();
let request = client.post("https://api.openai.com/v1/completions");
let request = with_bearer(request, "sk-...");
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
cargo test -p worker-adapters-openai-http -- --nocapture

# Run specific test
cargo test -p worker-adapters-openai-http -- test_submit --nocapture
```

### Integration Tests

Integration tests use a mock server to simulate OpenAI SSE responses:

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
    
    let adapter = OpenAiHttpAdapter::new(&mock_server.uri());
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
