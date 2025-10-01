# triton

**NVIDIA Triton Inference Server adapter**

`libs/worker-adapters/triton` — WorkerAdapter implementation for NVIDIA Triton Inference Server.

---

## What This Adapter Does

triton provides **Triton integration** for llama-orch:

- **gRPC protocol** — Uses Triton's gRPC inference API
- **Multi-model serving** — Supports multiple models on one server
- **Dynamic batching** — Automatic request batching
- **Model ensembles** — Pipeline multiple models
- **Enterprise-grade** — Production-ready serving platform

**Engine**: NVIDIA Triton Inference Server (default port: 8000 HTTP, 8001 gRPC)

---

## Usage

### Create Adapter

```rust
use worker_adapters_triton::TritonAdapter;

let adapter = TritonAdapter::new("http://localhost:8001");
```

### Submit Task

```rust
use worker_adapters_adapter_api::{WorkerAdapter, TaskRequest};

let task = TaskRequest {
    job_id: "job-123".to_string(),
    model: "llama-3.1-8b-instruct".to_string(),
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

## Triton API Mapping

### ModelInfer gRPC

**Orchestrator Request** → **Triton Request**

```protobuf
model_name: "llama-3.1-8b-instruct"
inputs {
  name: "prompt"
  datatype: "BYTES"
  shape: [1]
  contents {
    bytes_contents: "Hello, world!"
  }
}
parameters {
  key: "max_tokens"
  value {
    int64_param: 100
  }
}
```

**Triton Response**:

```protobuf
outputs {
  name: "text"
  datatype: "BYTES"
  shape: [1]
  contents {
    bytes_contents: "Hello there!"
  }
}
```

---

## Triton Features

### Supported

- **gRPC protocol** — Binary protocol for efficiency
- **Dynamic batching** — Automatic request batching
- **Multi-model** — Multiple models per server
- **Model versioning** — Version management
- **Health checks** — Server and model health

### Not Yet Supported

- **Streaming** — Requires custom backend implementation
- **Model ensembles** — Pipeline orchestration
- **Sequence batching** — Stateful models

---

## Configuration

### Environment Variables (Optional)

For orchestratord integration:

- `ORCHD_TRITON_URL` — Triton gRPC URL (e.g., `http://localhost:8001`)
- `ORCHD_TRITON_POOL` — Pool ID (default: `default`)
- `ORCHD_TRITON_REPLICA` — Replica ID (default: `r0`)

### Authentication

Triton typically uses mTLS for authentication in production:

```rust
// Configure TLS client certificates
let tls_config = ClientTlsConfig::new()
    .ca_certificate(ca_cert)
    .identity(client_cert, client_key);
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
cargo test -p worker-adapters-triton -- --nocapture

# Run specific test
cargo test -p worker-adapters-triton -- test_submit --nocapture
```

### Integration Tests

Integration tests use a mock gRPC server:

```rust
#[tokio::test]
async fn test_inference() {
    let mock_server = MockTritonServer::start().await;
    
    mock_server
        .expect_model_infer()
        .with_model("llama-3.1-8b-instruct")
        .respond_with_text("Hello there!")
        .await;
    
    let adapter = TritonAdapter::new(&mock_server.uri());
    // Test inference
}
```

---

## Dependencies

### Internal

- `worker-adapters-adapter-api` — WorkerAdapter trait

### External

- `tonic` — gRPC client
- `prost` — Protocol Buffers
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
