# FT-004: Correlation ID Middleware

**Team**: Foundation-Alpha  
**Sprint**: Sprint 1 - HTTP Foundation  
**Size**: S (1 day)  
**Days**: 5 - 5  
**Spec Ref**: WORK-3040 (implied), narration-core logging

---

## Story Description

Implement middleware that extracts or generates correlation IDs for request tracing. This enables end-to-end request tracking across logs and SSE events.

---

## Acceptance Criteria

- [ ] Middleware extracts `X-Correlation-ID` header from incoming requests
- [ ] If header missing, generates UUID v4 as correlation ID
- [ ] Correlation ID stored in request extensions for handler access
- [ ] All log statements include correlation ID in structured fields
- [ ] SSE events include correlation ID in metadata (optional field)
- [ ] Unit tests validate ID extraction and generation
- [ ] Integration tests validate ID propagation through request lifecycle
- [ ] Middleware runs before all route handlers
- [ ] Correlation ID format validated (UUID v4 or compatible string)

---

## Dependencies

### Upstream (Blocks This Story)
- FT-001: HTTP server infrastructure (Expected completion: Day 1)

### Downstream (This Story Blocks)
- FT-050: Narration-core logging needs correlation IDs for event tracking

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/src/http/middleware/correlation.rs` - Correlation ID middleware
- `bin/worker-orcd/src/http/middleware/mod.rs` - Middleware module exports
- `bin/worker-orcd/src/http/routes.rs` - Wire middleware to router
- `bin/worker-orcd/src/types/correlation.rs` - CorrelationId type

### Key Interfaces
```rust
use axum::{
    extract::Request,
    middleware::Next,
    response::Response,
};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct CorrelationId(String);

impl CorrelationId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
    
    pub fn from_header(value: &str) -> Option<Self> {
        // Validate format (UUID or alphanumeric string)
        if Self::is_valid(value) {
            Some(Self(value.to_string()))
        } else {
            None
        }
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
    
    fn is_valid(s: &str) -> bool {
        // Accept UUID v4 or alphanumeric strings (1-64 chars)
        s.len() >= 1 && s.len() <= 64 && s.chars().all(|c| c.is_alphanumeric() || c == '-')
    }
}

pub async fn correlation_middleware(
    mut req: Request,
    next: Next,
) -> Response {
    // Extract or generate correlation ID
    let correlation_id = req
        .headers()
        .get("X-Correlation-ID")
        .and_then(|v| v.to_str().ok())
        .and_then(CorrelationId::from_header)
        .unwrap_or_else(CorrelationId::new);
    
    // Store in request extensions
    req.extensions_mut().insert(correlation_id.clone());
    
    // Add to response headers
    let mut response = next.run(req).await;
    response.headers_mut().insert(
        "X-Correlation-ID",
        correlation_id.as_str().parse().unwrap(),
    );
    
    response
}
```

### Implementation Notes
- Use `axum::middleware::from_fn` to register middleware
- Correlation ID stored in `Request::extensions()` for handler access
- Response includes `X-Correlation-ID` header for client tracking
- Log correlation ID with every tracing statement: `tracing::info!(correlation_id = %id, "message")`
- Validation accepts UUID v4 format or alphanumeric strings (1-64 chars)
- Invalid correlation IDs in headers are ignored (generate new ID)
- Middleware should be first in chain (before logging, auth, etc.)

---

## Testing Strategy

### Unit Tests
- Test CorrelationId::new() generates valid UUID v4
- Test CorrelationId::from_header() accepts valid UUID
- Test CorrelationId::from_header() rejects invalid formats
- Test CorrelationId::is_valid() validates format constraints
- Test middleware extracts ID from header
- Test middleware generates ID when header missing

### Integration Tests
- Test request with X-Correlation-ID header preserves ID
- Test request without header generates new ID
- Test response includes X-Correlation-ID header
- Test correlation ID accessible in handler via extensions
- Test invalid header value triggers ID generation
- Test correlation ID appears in logs

### Manual Verification
1. Start server: `cargo run -- --port 8080`
2. Request with ID: `curl -H "X-Correlation-ID: test-123" http://localhost:8080/health`
3. Verify response header: `X-Correlation-ID: test-123`
4. Request without ID: `curl http://localhost:8080/health`
5. Verify response has generated UUID
6. Check logs for correlation_id field

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (6+ tests)
- [ ] Integration tests passing (6+ tests)
- [ ] Documentation updated (middleware docs, CorrelationId docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §7 HTTP API (implied correlation tracking)
- Related Stories: FT-001 (server), FT-050 (narration-core logging)
- Axum Middleware: https://docs.rs/axum/latest/axum/middleware/

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
