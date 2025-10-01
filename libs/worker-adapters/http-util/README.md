# http-util

**Shared HTTP utilities for worker adapters**

`libs/worker-adapters/http-util` — HTTP client, retry logic, streaming decode, and secret redaction for all adapters.

---

## What This Library Does

http-util provides **shared HTTP infrastructure** for worker adapters:

- **HTTP client** — HTTP/2 preferred, rustls TLS, connection pooling
- **Timeouts** — Per-request timeout enforcement
- **Retry logic** — Exponential backoff with jitter for transient failures
- **Streaming decode** — Low-allocation SSE/NDJSON token stream parsing
- **Secret redaction** — Authorization headers redacted in logs
- **Bearer auth** — Helper to inject `Authorization: Bearer` tokens

**Used by**: All HTTP-based adapters (llamacpp-http, vllm-http, tgi-http, openai-http)

---

## Key APIs

### HTTP Client

```rust
use worker_adapters_http_util::{HttpClientConfig, make_client};

let config = HttpClientConfig {
    timeout_secs: 30,
    connect_timeout_secs: 5,
    pool_idle_timeout_secs: 90,
};

let client = make_client(&config);
```

### Retry Logic

```rust
use worker_adapters_http_util::{with_retries, RetryPolicy, RetryError};

let policy = RetryPolicy {
    max_attempts: 3,
    base_delay_ms: 100,
    max_delay_ms: 5000,
    jitter: true,
    seed: None,
};

let result = with_retries(|attempt| async move {
    // Your operation
    if attempt < 3 {
        Err(RetryError::Retriable(anyhow::anyhow!("transient error")))
    } else {
        Ok(42)
    }
}, policy).await?;
```

### Bearer Auth

```rust
use worker_adapters_http_util::auth::with_bearer;

let client = reqwest::Client::new();
let request = client.get("https://api.example.com/v1/completions");
let request = with_bearer(request, "my-secret-token");
```

### Streaming Decode

```rust
use worker_adapters_http_util::streaming::decode_sse;

let body = "event: token\ndata: {\"text\":\"hello\"}\n\n";
let events = decode_sse(body)?;
```

---

## Retry Policy

### Retriable Errors

- **HTTP 429** (Too Many Requests)
- **HTTP 5xx** (Server errors)
- **Connection errors** (timeout, refused, reset)

### Non-Retriable Errors

- **HTTP 4xx** (except 429) — Client errors
- **HTTP 2xx** — Success (no retry needed)

### Backoff Formula

```
delay = min(base_delay * 2^attempt, max_delay) + jitter
```

Example with `base_delay=100ms`, `max_delay=5000ms`:
- Attempt 1: 100ms + jitter
- Attempt 2: 200ms + jitter
- Attempt 3: 400ms + jitter
- Attempt 4: 800ms + jitter
- Attempt 5: 1600ms + jitter
- Attempt 6: 3200ms + jitter
- Attempt 7: 5000ms + jitter (capped)

---

## Secret Redaction

### Header Redaction

```rust
use worker_adapters_http_util::redact::redact_headers;

let headers = vec![
    ("Authorization", "Bearer secret-token"),
    ("Content-Type", "application/json"),
];

let redacted = redact_headers(&headers);
// Output: [("Authorization", "[REDACTED]"), ("Content-Type", "application/json")]
```

### Log Line Redaction

```rust
use worker_adapters_http_util::redact::redact_line;

let log = "Authorization: Bearer abc123";
let redacted = redact_line(log);
// Output: "Authorization: [REDACTED]"
```

---

## Usage Examples

### Complete HTTP Request with Retry

```rust
use worker_adapters_http_util::{make_client, with_retries, RetryPolicy, RetryError};
use worker_adapters_http_util::auth::with_bearer;

#[tokio::main]
async fn main() -> Result<()> {
    // Create client
    let config = HttpClientConfig::default();
    let client = make_client(&config);
    
    // Define retry policy
    let policy = RetryPolicy::default();
    
    // Make request with retries
    let result = with_retries(|_attempt| async {
        let request = client.get("https://api.example.com/health");
        let request = with_bearer(request, "my-token");
        
        let response = request
            .send()
            .await
            .map_err(|e| RetryError::Retriable(e.into()))?;
        
        if response.status().is_success() {
            Ok(response.text().await?)
        } else {
            Err(RetryError::Retriable(anyhow::anyhow!("HTTP {}", response.status())))
        }
    }, policy).await?;
    
    println!("Response: {}", result);
    Ok(())
}
```

### SSE Stream Parsing

```rust
use worker_adapters_http_util::streaming::decode_sse;

let sse_body = r#"
event: started
data: {"engine":"llama.cpp"}

event: token
data: {"text":"Hello","index":0}

event: token
data: {"text":" world","index":1}

event: end
data: {"tokens":2}
"#;

let events = decode_sse(sse_body)?;
for event in events {
    println!("Event: {:?}", event);
}
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p worker-adapters-http-util -- --nocapture

# Run specific test
cargo test -p worker-adapters-http-util -- test_retry_backoff --nocapture
```

### Integration Tests

Use `wiremock` to test retry behavior:

```rust
use wiremock::{MockServer, Mock, ResponseTemplate};
use wiremock::matchers::{method, path};

#[tokio::test]
async fn test_retry_on_500() {
    let mock_server = MockServer::start().await;
    
    Mock::given(method("GET"))
        .and(path("/api"))
        .respond_with(ResponseTemplate::new(500))
        .expect(3)
        .mount(&mock_server)
        .await;
    
    // Test retry logic
}
```

---

## Dependencies

### Internal

- None (foundational utility library)

### External

- `reqwest` — HTTP client
- `tokio` — Async runtime
- `rustls` — TLS implementation
- `serde` — Serialization
- `anyhow` — Error handling

---

## Specifications

Implements requirements from:
- ORCH-3054, ORCH-3055, ORCH-3056, ORCH-3057, ORCH-3058

See `.specs/00_llama-orch.md` for full requirements.

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
