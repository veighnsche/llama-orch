# [FT-002] POST /execute Endpoint (Skeleton)

**Team**: Foundation  
**Sprint**: Week 1  
**Size**: M (2 days)  
**Owner**: [Rust Lead]  
**Status**: Backlog  
**Priority**: P0

---

## User Story

As a client, I want to submit inference requests, so that the worker can process them.

---

## Acceptance Criteria

- [ ] Endpoint accepts POST /execute with JSON body
- [ ] Request schema: `{job_id, prompt, max_tokens, temperature, seed}`
- [ ] Request validation:
  - [ ] job_id: non-empty string
  - [ ] prompt: non-empty, max 32768 chars
  - [ ] max_tokens: 1-2048
  - [ ] temperature: 0.0-2.0
  - [ ] seed: valid u64
- [ ] Returns 400 with error message if validation fails
- [ ] Returns 202 Accepted for valid requests (no actual inference yet)
- [ ] X-Correlation-Id middleware attaches correlation ID to logs
- [ ] Unit test: validation edge cases (empty prompt, temp=3.0, etc.)

---

## Definition of Done

- [ ] Code reviewed and merged
- [ ] Unit tests pass (>80% coverage)
- [ ] Integration test: POST /execute returns 202
- [ ] No warnings (rustfmt, clippy)
- [ ] API documentation: request/response schemas
- [ ] Demoed in Friday demo

---

## Dependencies

**Depends on**: FT-001 (needs server running)  
**Blocks**: FT-003 (SSE streaming needs execute endpoint)

---

## Technical Notes

### Request Schema

```rust
#[derive(Debug, Deserialize, Validate)]
struct ExecuteRequest {
    #[validate(length(min = 1))]
    job_id: String,
    
    #[validate(length(min = 1, max = 32768))]
    prompt: String,
    
    #[validate(range(min = 1, max = 2048))]
    max_tokens: u32,
    
    #[validate(range(min = 0.0, max = 2.0))]
    temperature: f32,
    
    seed: u64,
}
```

### Handler

```rust
async fn execute_handler(
    Extension(state): Extension<Arc<WorkerState>>,
    Json(req): Json<ExecuteRequest>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    // Validate
    req.validate().map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(ErrorResponse {
            code: "INVALID_REQUEST".to_string(),
            message: e.to_string(),
        }))
    })?;
    
    // TODO: Queue for inference (Week 3-4)
    
    Ok(StatusCode::ACCEPTED)
}
```

### Correlation ID Middleware

```rust
async fn correlation_id_middleware(
    mut req: Request<Body>,
    next: Next<Body>,
) -> Response {
    let correlation_id = req.headers()
        .get("X-Correlation-Id")
        .and_then(|v| v.to_str().ok())
        .map(String::from)
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    
    req.extensions_mut().insert(CorrelationId(correlation_id.clone()));
    
    let mut response = next.run(req).await;
    response.headers_mut().insert(
        "X-Correlation-Id",
        correlation_id.parse().unwrap(),
    );
    response
}
```

### Spec References

- M0-W-1300: POST /execute
- M0-W-1302: Request validation

---

## Progress Log

**YYYY-MM-DD**: Story created
