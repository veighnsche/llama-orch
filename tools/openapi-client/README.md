# openapi-client

**Generated HTTP client from OpenAPI specifications**

`tools/openapi-client` — Type-safe Rust client generated from OpenAPI YAML files.

---

## What This Tool Does

openapi-client provides **generated API client** for llama-orch:

- **Type-safe** — Generated from OpenAPI specs
- **Async** — Built on reqwest and tokio
- **Validated** — Request/response validation
- **Documented** — API docs from OpenAPI descriptions
- **Idempotent** — Deterministic code generation

**Used by**: Tests, CLI tools, external clients

---

## Usage

### Create Client

```rust
use openapi_client::Client;

let client = Client::new("http://localhost:8080");
```

### Enqueue Job

```rust
use openapi_client::{Client, EnqueueRequest};

let client = Client::new("http://localhost:8080");

let request = EnqueueRequest {
    prompt: "Hello, world!".to_string(),
    model: "llama-3.1-8b-instruct".to_string(),
    max_tokens: 100,
    seed: Some(42),
    ..Default::default()
};

let response = client.enqueue(request).await?;
println!("Job ID: {}", response.job_id);
```

### Get Job Status

```rust
let status = client.get_job_status("job-123").await?;
println!("State: {:?}", status.state);
println!("Tokens: {:?}", status.tokens_generated);
```

### List Pools

```rust
let pools = client.list_pools().await?;
for pool in pools {
    println!("Pool: {} ({})", pool.id, pool.state);
}
```

---

## Generation

### Generate Client

```bash
# Regenerate client from OpenAPI specs
cargo xtask regen-openapi
```

This generates:
- `src/lib.rs` — Client struct and methods
- `src/types.rs` — Request/response types
- `src/error.rs` — Error types

### Source Files

OpenAPI specifications:
- `contracts/openapi/rbees-orcd.yaml` — Orchestrator API
- `contracts/openapi/pool-managerd.yaml` — Pool manager API

---

## API Operations

### Orchestrator

- **POST /v1/enqueue** — Enqueue job
- **GET /v1/jobs/{id}** — Get job status
- **DELETE /v1/jobs/{id}** — Cancel job
- **GET /v1/pools** — List pools
- **GET /v1/pools/{id}** — Get pool status
- **GET /health** — Health check

### Pool Manager

- **POST /provision** — Provision engine
- **DELETE /pools/{id}** — Stop pool
- **GET /health** — Health check

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p tools-openapi-client -- --nocapture
```

### Integration Tests

```bash
# Test against running orchestrator
ORCHD_URL=http://localhost:8080 \
  cargo test -p tools-openapi-client -- test_integration --nocapture
```

---

## Dependencies

### Internal

- `contracts/api-types` — Shared types

### External

- `reqwest` — HTTP client
- `serde` — Serialization
- `serde_json` — JSON format
- `tokio` — Async runtime

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
