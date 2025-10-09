# api-types

**Shared API types and contracts for HTTP endpoints**

`contracts/api-types` — Rust types for rbees-orcd HTTP API (requests, responses, enums).

---

## What This Library Does

api-types provides **API contracts** for llama-orch:

- **Request types** — Enqueue, dispatch, session management
- **Response types** — Job status, pool status, health checks
- **Shared enums** — JobState, PoolState, ErrorCode
- **Serialization** — serde-based JSON serialization
- **OpenAPI generation** — Types annotated for schema generation

**Used by**: rbees-orcd, clients, test harness

---

## Key Types

### EnqueueRequest

```rust
use api_types::EnqueueRequest;

let request = EnqueueRequest {
    prompt: "Hello, world!".to_string(),
    model: "llama-3.1-8b-instruct".to_string(),
    max_tokens: 100,
    temperature: Some(0.7),
    seed: Some(42),
    session_id: None,
};
```

### EnqueueResponse

```rust
use api_types::EnqueueResponse;

let response = EnqueueResponse {
    job_id: "job-123".to_string(),
    status: JobState::Queued,
};
```

### JobStatusResponse

```rust
use api_types::{JobStatusResponse, JobState};

let response = JobStatusResponse {
    job_id: "job-123".to_string(),
    state: JobState::Running,
    pool_id: Some("default".to_string()),
    replica_id: Some("r0".to_string()),
    tokens_generated: Some(42),
    error: None,
};
```

### JobState

```rust
use api_types::JobState;

pub enum JobState {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}
```

---

## OpenAPI Generation

Types are annotated with `schemars` for OpenAPI schema generation:

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EnqueueRequest {
    /// The prompt to generate from
    pub prompt: String,
    
    /// Model identifier
    pub model: String,
    
    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    
    /// Temperature for sampling (0.0 = greedy)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}
```

Generate OpenAPI schema:

```bash
cargo xtask regen-openapi
```

Output: `contracts/openapi/rbees-orcd.yaml`

---

## Usage

### In rbees-orcd

```rust
use api_types::{EnqueueRequest, EnqueueResponse, JobState};
use axum::{Json, extract::State};

async fn enqueue(
    State(state): State<AppState>,
    Json(req): Json<EnqueueRequest>,
) -> Json<EnqueueResponse> {
    let job_id = state.orchestrator.enqueue(req).await?;
    
    Json(EnqueueResponse {
        job_id,
        status: JobState::Queued,
    })
}
```

### In Clients

```rust
use api_types::{EnqueueRequest, EnqueueResponse};

let client = reqwest::Client::new();
let request = EnqueueRequest {
    prompt: "Hello, world!".to_string(),
    model: "llama-3.1-8b-instruct".to_string(),
    max_tokens: 100,
    temperature: Some(0.7),
    seed: Some(42),
    session_id: None,
};

let response: EnqueueResponse = client
    .post("http://localhost:8080/v1/enqueue")
    .json(&request)
    .send()
    .await?
    .json()
    .await?;

println!("Job ID: {}", response.job_id);
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p contracts-api-types -- --nocapture

# Test serialization
cargo test -p contracts-api-types -- test_serde --nocapture
```

### Schema Validation

```bash
# Regenerate OpenAPI schema
cargo xtask regen-openapi

# Validate schema
cargo run -p tools-openapi-client -- validate
```

---

## Dependencies

### Internal

- None (foundational contract library)

### External

- `serde` — Serialization
- `serde_json` — JSON format
- `schemars` — JSON Schema generation

---

## Regenerating Artifacts

### OpenAPI Schema

```bash
# Regenerate OpenAPI schema from types
cargo xtask regen-openapi

# Output: contracts/openapi/rbees-orcd.yaml
```

### JSON Schema

```bash
# Regenerate JSON Schema
cargo xtask regen-schema

# Output: contracts/config-schema/schema.json
```

---

## Specifications

Implements requirements from:
- ORCH-3044 (API types)
- ORCH-3030 (HTTP endpoints)

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
