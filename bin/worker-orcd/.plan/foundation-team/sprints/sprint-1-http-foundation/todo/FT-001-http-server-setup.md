# FT-001: HTTP Server Setup

**Team**: Foundation-Alpha  
**Sprint**: Sprint 1 - HTTP Foundation  
**Size**: S (1 day)  
**Days**: 1 - 1  
**Spec Ref**: M0-W-1110, WORK-3010

---

## Story Description

Initialize Axum HTTP server with tokio runtime, bind to configurable address, and implement basic health endpoint. This is the foundation for all HTTP communication with the worker.

---

## Acceptance Criteria

- [ ] Axum HTTP server initialized with tokio runtime (multi-threaded)
- [ ] Server binds to address from `WORKER_ADDR` env var (default: `127.0.0.1:8080`)
- [ ] `/health` endpoint returns 200 OK with `{"status": "healthy"}`
- [ ] Server logs startup with `tracing::info` including bind address
- [ ] Unit test validates `/health` endpoint response structure
- [ ] Integration test validates server startup and graceful shutdown
- [ ] Error handling for bind failures (port already in use, permission denied)
- [ ] Graceful shutdown on SIGTERM/SIGINT with cleanup
- [ ] Server state struct holds bind address and shutdown channel

---

## Dependencies

### Upstream (Blocks This Story)
- None (first story in Foundation team)

### Downstream (This Story Blocks)
- FT-002: POST /execute endpoint needs server infrastructure
- FT-004: Correlation ID middleware needs server routing
- FT-005: Request validation needs server routing

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/src/http/server.rs` - Server initialization and lifecycle
- `bin/worker-orcd/src/http/routes.rs` - Route definitions
- `bin/worker-orcd/src/http/health.rs` - Health endpoint handler
- `bin/worker-orcd/src/http/mod.rs` - HTTP module exports
- `bin/worker-orcd/Cargo.toml` - Add dependencies: axum, tokio, tower, serde_json

### Key Interfaces
```rust
use axum::{Router, routing::{get, post}};
use std::net::SocketAddr;
use tokio::sync::broadcast;

pub struct HttpServer {
    addr: SocketAddr,
    shutdown_tx: broadcast::Sender<()>,
}

impl HttpServer {
    /// Create new HTTP server bound to address
    pub async fn new(addr: SocketAddr) -> Result<Self, ServerError>;
    
    /// Run server until shutdown signal received
    pub async fn run(self) -> Result<(), ServerError>;
    
    /// Trigger graceful shutdown
    pub fn shutdown(&self) -> Result<(), ServerError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("Failed to bind to {addr}: {source}")]
    BindFailed {
        addr: SocketAddr,
        source: std::io::Error,
    },
    #[error("Server runtime error: {0}")]
    Runtime(String),
}
```

### Implementation Notes
- Use `tokio::runtime::Builder::new_multi_thread()` for runtime
- Bind address format: `IP:PORT` (e.g., `127.0.0.1:8080`, `0.0.0.0:3000`)
- Health endpoint returns JSON: `{"status": "healthy"}` with `Content-Type: application/json`
- Shutdown channel uses `tokio::sync::broadcast` for multi-subscriber support
- Log bind address at INFO level: `"HTTP server listening on {addr}"`
- Handle SIGTERM/SIGINT via `tokio::signal::ctrl_c()`
- Server should complete graceful shutdown within 5 seconds

---

## Testing Strategy

### Unit Tests
- Test health endpoint handler returns correct JSON structure
- Test health endpoint returns 200 status code
- Test ServerError variants format correctly

### Integration Tests
- Test server starts successfully on available port
- Test server binds to custom address from env var
- Test server returns 503 after shutdown initiated
- Test graceful shutdown completes within timeout
- Test bind failure when port already in use (spawn two servers)
- Test bind failure with invalid address format

### Manual Verification
1. Start server: `cargo run -- --port 8080`
2. Curl health: `curl http://localhost:8080/health`
3. Verify JSON response: `{"status":"healthy"}`
4. Send SIGTERM: `kill -TERM <pid>`
5. Verify graceful shutdown in logs

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (3+ tests)
- [ ] Integration tests passing (4+ tests)
- [ ] Documentation updated (module-level docs in server.rs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §5.2 Initialization Sequence (M0-W-1110)
- Spec: `bin/.specs/01_M0_worker_orcd.md` §7.3 Health Endpoint (M0-W-1320)
- Related Stories: FT-002 (execute endpoint), FT-004 (middleware)
- Axum Docs: https://docs.rs/axum/latest/axum/

---

## 🎀 Narration Opportunities (v0.2.0)

**From**: Narration-Core Team  
**Updated**: 2025-10-04 (v0.2.0 - Production Ready with Builder Pattern & Axum Middleware)

### Critical Events to Narrate

#### 1. Server Startup (INFO level) ✅
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD, ACTION_SPAWN};

// NEW v0.2.0: Builder pattern (43% less boilerplate!)
Narration::new(ACTOR_WORKER_ORCD, ACTION_SPAWN, "http-server")
    .human(format!("HTTP server listening on {}", addr))
    .emit();  // Auto-injects: emitted_by, emitted_at_ms
```

**Cute mode** (optional):
```rust
Narration::new(ACTOR_WORKER_ORCD, ACTION_SPAWN, "http-server")
    .human(format!("HTTP server listening on {}", addr))
    .cute(format!("Worker woke up and opened the door at {}! 🏠✨", addr))
    .emit();
```

#### 2. Health Check Requests (DEBUG level) 🔍
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD};

// NEW v0.2.0: Builder with debug level
Narration::new(ACTOR_WORKER_ORCD, "health_check", "health")
    .human("Health check requested")
    .correlation_id(correlation_id)
    .emit_debug();  // ← DEBUG level to avoid log spam
```

**Note**: Use `.emit_debug()` to avoid log spam from health checks

#### 3. Server Shutdown (WARN level) ⚠️
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD, ACTION_SHUTDOWN};

// NEW v0.2.0: Builder with warn level
Narration::new(ACTOR_WORKER_ORCD, ACTION_SHUTDOWN, "http-server")
    .human("HTTP server shutting down gracefully")
    .duration_ms(shutdown_duration_ms)
    .story("\"Time to rest,\" whispered the worker, closing the door gently. 👋")
    .emit_warn();  // ← WARN level
```

#### 4. Bind Failures (ERROR level) 🚨
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD, ACTION_SPAWN};

// NEW v0.2.0: Builder with error level
Narration::new(ACTOR_WORKER_ORCD, ACTION_SPAWN, "http-server")
    .human(format!("Failed to bind to {}: {}", addr, error))
    .error_kind("BindFailed")
    .emit_error();  // ← ERROR level
```

### Testing with CaptureAdapter

```rust
use observability_narration_core::CaptureAdapter;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]
fn test_server_startup_narration() {
    let adapter = CaptureAdapter::install();
    
    // Start server (emits narration)
    let server = HttpServer::new("127.0.0.1:8080".parse().unwrap()).await?;
    
    // Assert narration captured
    adapter.assert_includes("HTTP server listening");
    adapter.assert_field("actor", "worker-orcd");
    adapter.assert_field("action", "spawn");
    adapter.assert_provenance_present();  // NEW in v0.2.0
}
```

### HTTP Context Propagation (NEW) 🌐

Extract correlation IDs from incoming requests:

```rust
use observability_narration_core::http::{extract_context_from_headers, HeaderLike};

// In your health endpoint
async fn health_handler(headers: HeaderMap) -> Json<HealthResponse> {
    let context = extract_context_from_headers(&headers);
    
    if let Some(correlation_id) = context.correlation_id {
        narrate_debug(NarrationFields {
            actor: "worker-orcd",
            action: "health_check",
            target: "health".to_string(),
            correlation_id: Some(correlation_id),
            human: "Health check requested".to_string(),
            ..Default::default()
        });
    }
    
    Json(HealthResponse { status: "healthy" })
}
```

### Why This Matters

**Server lifecycle events** are critical for:
- 🔍 Debugging deployment issues
- 📊 Tracking worker availability
- 🚨 Alerting on bind failures
- 📈 Measuring startup/shutdown times
- 🔗 Correlating requests across services

### New in v0.2.0
- ✅ **Builder pattern API** - 43% less boilerplate (7 lines → 4 lines)
- ✅ **Axum middleware** - built-in correlation ID handling
- ✅ **All constants** - ACTOR_*, ACTION_* for type safety
- ✅ **Level methods** - `.emit()`, `.emit_warn()`, `.emit_error()`, `.emit_debug()`
- ✅ **Auto-injection** - automatic emitted_by, emitted_at_ms
- ✅ **119 tests passing** - including 24 smoke tests + 3 E2E tests

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04  
**Narration Updated**: 2025-10-04 (v0.2.0)

---
Planned by Project Management Team 📋  
*Narration guidance updated by Narration-Core Team 🎀*

---

## 🔍 Testing Team Requirements

**From**: Testing Team (Pre-Development Audit)

### Unit Testing Requirements
- **Test health endpoint response structure** (JSON format validation)
- **Test health endpoint status code** (200 OK)
- **Test ServerError variants** (all error types format correctly)
- **Test bind address parsing** (valid and invalid formats)
- **Test shutdown signal handling** (SIGTERM/SIGINT)
- **Property test**: Server state transitions (startup → running → shutdown)

### Integration Testing Requirements
- **Test server startup on available port** (dynamic port allocation)
- **Test server binds to custom address from env var** (WORKER_ADDR)
- **Test graceful shutdown completes within timeout** (5 seconds max)
- **Test bind failure when port already in use** (spawn two servers, second fails)
- **Test bind failure with invalid address** (malformed IP:PORT)
- **Test concurrent health checks** (multiple requests during startup)

### BDD Testing Requirements (VERY IMPORTANT)
- **Scenario**: Server starts successfully
  - Given no server is running
  - When I start the server on port 8080
  - Then the server should bind successfully
  - And the health endpoint should return 200 OK
- **Scenario**: Server handles graceful shutdown
  - Given a running server
  - When I send SIGTERM
  - Then the server should complete in-flight requests
  - And shutdown within 5 seconds
- **Scenario**: Server rejects invalid bind address
  - Given an invalid address format
  - When I attempt to start the server
  - Then the server should fail with BindFailed error
  - And log the error with bind address

### Critical Paths to Test
- Server startup sequence (tokio runtime → bind → route registration)
- Health endpoint availability immediately after startup
- Graceful shutdown with in-flight request handling
- Error propagation from bind failures to caller

### Edge Cases
- Port 0 (OS-assigned port)
- IPv6 addresses
- Localhost vs 0.0.0.0 binding
- Shutdown during startup
- Multiple shutdown signals

---
Test opportunities identified by Testing Team 🔍
