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

---

## 🎀 Narration Opportunities (v0.1.0)

**From**: Narration-Core Team  
**Updated**: 2025-10-04 (v0.1.0 - Production Ready)

### Critical Events to Narrate

#### 1. Correlation ID Extracted (DEBUG level) 🔍
```rust
use observability_narration_core::narrate_debug;

narrate_debug(NarrationFields {
    actor: "worker-orcd",
    action: "correlation_extract",
    target: "middleware".to_string(),
    correlation_id: Some(correlation_id.clone()),
    human: format!("Extracted correlation ID: {}", correlation_id),
    ..Default::default()
});
```

**Note**: Use DEBUG level since this happens on every request

#### 2. Correlation ID Generated (INFO level) ✅
```rust
use observability_narration_core::{narrate, NarrationFields};

narrate(NarrationFields {
    actor: "worker-orcd",
    action: "correlation_generate",
    target: "middleware".to_string(),
    correlation_id: Some(correlation_id.clone()),
    human: format!("Generated new correlation ID: {}", correlation_id),
    ..Default::default()
});
```

**Cute mode** (optional):
```rust
cute: Some(format!("Worker created a new tracking tag! 🏷️ {}", correlation_id))
```

#### 3. Invalid Correlation ID Rejected (WARN level) ⚠️
```rust
use observability_narration_core::narrate_warn;

narrate_warn(NarrationFields {
    actor: "worker-orcd",
    action: "correlation_validate",
    target: "middleware".to_string(),
    correlation_id: Some(new_correlation_id.clone()),
    error_kind: Some("invalid_correlation_id".to_string()),
    human: format!("Rejected invalid correlation ID '{}', generated new: {}", invalid_id, new_correlation_id),
    ..Default::default()
});
```

### Using Built-in HTTP Context Propagation (NEW) 🌐

**narration-core v0.1.0** provides built-in helpers:

```rust
use observability_narration_core::http::{extract_context_from_headers, inject_context_into_headers};
use observability_narration_core::{generate_correlation_id, validate_correlation_id};

// Extract from incoming request
let context = extract_context_from_headers(&headers);
let correlation_id = context.correlation_id
    .and_then(|id| validate_correlation_id(&id).map(|_| id))
    .unwrap_or_else(|| generate_correlation_id());

// Inject into outgoing response
let mut response_context = NarrationContext {
    correlation_id: Some(correlation_id.clone()),
    trace_id: context.trace_id,
    span_id: Some(generate_span_id()),
};
inject_context_into_headers(&response_context, &mut response_headers);
```

### Testing with CaptureAdapter

```rust
use observability_narration_core::CaptureAdapter;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]
fn test_correlation_id_middleware() {
    let adapter = CaptureAdapter::install();
    
    // Request without correlation ID
    let response = client.get("/health").send().await?;
    
    // Assert correlation ID was generated
    adapter.assert_includes("Generated new correlation ID");
    adapter.assert_correlation_id_present();
    
    // Verify response header contains correlation ID
    let correlation_id = response.headers()
        .get("X-Correlation-Id")
        .unwrap();
    assert!(validate_correlation_id(correlation_id).is_some());
}
```

### Why This Matters

**Correlation ID middleware** is critical for:
- 🔗 **Request tracing** across orchestrator → worker → engine
- 🐛 **Debugging** multi-service workflows
- 📊 **Metrics aggregation** by request
- 🚨 **Alerting** on request failures
- 📈 **Performance tracking** end-to-end

**IMPORTANT**: This middleware enables ALL other narration by providing correlation IDs. Every subsequent narration event should include the `correlation_id` field!

### New in v0.1.0
- ✅ **Built-in HTTP helpers** (`extract_context_from_headers`, `inject_context_into_headers`)
- ✅ **Fast validation** (<100ns per correlation ID)
- ✅ **7 logging levels** (DEBUG for extract, INFO for generate, WARN for invalid)
- ✅ **Test assertions** (`assert_correlation_id_present()`)
- ✅ **Property tests** for correlation ID invariants

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04  
**Narration Updated**: 2025-10-04 (v0.1.0)

---
Planned by Project Management Team 📋  
*Narration guidance updated by Narration-Core Team 🎀*

---

## 🔍 Testing Team Requirements

**From**: Testing Team (Pre-Development Audit)

### Unit Testing Requirements
- **Test CorrelationId::new() generates valid UUID v4** (format validation)
- **Test CorrelationId::from_header() accepts valid UUID** (parsing)
- **Test CorrelationId::from_header() accepts alphanumeric strings** (1-64 chars)
- **Test CorrelationId::from_header() rejects invalid formats** (special chars, too long)
- **Test CorrelationId::is_valid() validates format constraints** (length, charset)
- **Test middleware extracts ID from header** (X-Correlation-ID present)
- **Test middleware generates ID when header missing** (UUID v4)
- **Property test**: All valid UUID v4 formats accepted

### Integration Testing Requirements
- **Test request with X-Correlation-ID header preserves ID** (passthrough)
- **Test request without header generates new ID** (auto-generation)
- **Test response includes X-Correlation-ID header** (echo back)
- **Test correlation ID accessible in handler via extensions** (request context)
- **Test invalid header value triggers ID generation** (fallback)
- **Test correlation ID appears in logs** (structured logging)
- **Test correlation ID propagates through middleware chain** (ordering)

### BDD Testing Requirements (VERY IMPORTANT)
- **Scenario**: Client provides correlation ID
  - Given a request with X-Correlation-ID header "test-123"
  - When the request is processed
  - Then the response should include X-Correlation-ID "test-123"
  - And logs should include correlation_id "test-123"
- **Scenario**: Client does not provide correlation ID
  - Given a request without X-Correlation-ID header
  - When the request is processed
  - Then the response should include a generated UUID v4
  - And logs should include the generated correlation_id
- **Scenario**: Invalid correlation ID rejected
  - Given a request with X-Correlation-ID "invalid@#$%"
  - When the request is processed
  - Then a new UUID v4 should be generated
  - And the response should include the new ID

### Critical Paths to Test
- Correlation ID extraction from header
- UUID v4 generation when missing
- ID storage in request extensions
- ID propagation to response headers
- ID inclusion in all log statements

### Edge Cases
- Empty X-Correlation-ID header
- Very long correlation ID (>64 chars)
- Unicode characters in correlation ID
- Multiple X-Correlation-ID headers
- Correlation ID with null bytes

---
Test opportunities identified by Testing Team 🔍
