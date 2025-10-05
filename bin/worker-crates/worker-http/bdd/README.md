# worker-http BDD Tests

Behavior-Driven Development tests for HTTP server and SSE streaming.

## Running Tests

```bash
# Run all BDD tests
cargo run --bin bdd-runner

# Run specific feature
cargo run --bin bdd-runner -- tests/features/server_lifecycle.feature
```

## Features

- **Server Lifecycle** — Verify server startup and shutdown
  - Bind to port
  - Graceful shutdown
  - Bind failure handling

- **SSE Streaming** — Verify token streaming behavior
  - Multiple token events
  - Client disconnect handling
  - Clean stream closure

- **Request Validation** — Verify request validation
  - Required headers
  - Content-Type validation
  - Correlation ID handling

## Critical Behaviors

These tests verify **critical HTTP contract behaviors** that must work correctly:

1. **Graceful shutdown** prevents dropped requests
2. **SSE streaming** enables real-time inference
3. **Request validation** prevents invalid requests

**Consequence of undertesting**: Dropped requests, broken streaming, security issues.
