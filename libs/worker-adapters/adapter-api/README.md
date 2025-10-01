# adapter-api

**Shared WorkerAdapter trait and types for engine adapters**

`libs/worker-adapters/adapter-api` — Common trait definition and types used by all worker adapters.

---

## What This Library Does

adapter-api provides the **shared interface** for all worker adapters:

- **WorkerAdapter trait** — Common interface all adapters must implement
- **Shared types** — TaskRequest, TokenStream, HealthStatus, EngineProps
- **Error types** — WorkerError taxonomy
- **No implementation** — Pure trait and types, no business logic

**Used by**: All worker adapters (llamacpp-http, vllm-http, tgi-http, mock, etc.)

---

## WorkerAdapter Trait

```rust
use async_trait::async_trait;

#[async_trait]
pub trait WorkerAdapter: Send + Sync {
    /// Health check
    async fn health(&self) -> Result<HealthStatus, WorkerError>;
    
    /// Get engine properties (version, capabilities)
    async fn props(&self) -> Result<EngineProps, WorkerError>;
    
    /// Submit task and stream tokens
    async fn submit(&self, task: TaskRequest) -> Result<TokenStream, WorkerError>;
    
    /// Cancel running task
    async fn cancel(&self, job_id: &str) -> Result<(), WorkerError>;
    
    /// Get engine version
    async fn engine_version(&self) -> Result<String, WorkerError>;
}
```

---

## Key Types

### TaskRequest

```rust
pub struct TaskRequest {
    pub job_id: String,
    pub model: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,
    pub session_id: Option<String>,
}
```

### TokenStream

```rust
pub struct TokenStream {
    pub receiver: mpsc::Receiver<TokenEvent>,
}

pub enum TokenEvent {
    Started { engine_version: String },
    Token { text: String, index: usize },
    End { metrics: Metrics },
    Error { error: WorkerError },
}
```

### HealthStatus

```rust
pub struct HealthStatus {
    pub state: HealthState,
    pub message: Option<String>,
}

pub enum HealthState {
    Healthy,
    Degraded,
    Unhealthy,
}
```

### EngineProps

```rust
pub struct EngineProps {
    pub engine_type: String,
    pub engine_version: String,
    pub supports_streaming: bool,
    pub supports_sessions: bool,
    pub max_batch_size: Option<usize>,
}
```

### WorkerError

```rust
pub enum WorkerError {
    Timeout,
    EngineUnavailable,
    InvalidRequest(String),
    CapacityExceeded,
    StreamingError(String),
    Unknown(String),
}
```

---

## Usage Example

```rust
use worker_adapters_adapter_api::{WorkerAdapter, TaskRequest, TokenEvent};
use async_trait::async_trait;

pub struct MyAdapter {
    endpoint: String,
}

#[async_trait]
impl WorkerAdapter for MyAdapter {
    async fn health(&self) -> Result<HealthStatus, WorkerError> {
        // Check engine health
        Ok(HealthStatus {
            state: HealthState::Healthy,
            message: None,
        })
    }
    
    async fn props(&self) -> Result<EngineProps, WorkerError> {
        Ok(EngineProps {
            engine_type: "my-engine".to_string(),
            engine_version: "1.0.0".to_string(),
            supports_streaming: true,
            supports_sessions: false,
            max_batch_size: Some(32),
        })
    }
    
    async fn submit(&self, task: TaskRequest) -> Result<TokenStream, WorkerError> {
        // Create channel
        let (tx, rx) = mpsc::channel(100);
        
        // Spawn task to stream tokens
        tokio::spawn(async move {
            tx.send(TokenEvent::Started {
                engine_version: "1.0.0".to_string(),
            }).await.ok();
            
            tx.send(TokenEvent::Token {
                text: "hello".to_string(),
                index: 0,
            }).await.ok();
            
            tx.send(TokenEvent::End {
                metrics: Metrics::default(),
            }).await.ok();
        });
        
        Ok(TokenStream { receiver: rx })
    }
    
    async fn cancel(&self, job_id: &str) -> Result<(), WorkerError> {
        // Cancel task
        Ok(())
    }
    
    async fn engine_version(&self) -> Result<String, WorkerError> {
        Ok("1.0.0".to_string())
    }
}
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p worker-adapters-adapter-api -- --nocapture
```

---

## Dependencies

### Internal

- None (this is a foundational trait library)

### External

- `async-trait` — Async trait support
- `tokio` — Async runtime (mpsc channels)
- `serde` — Serialization
- `thiserror` — Error types

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
