# [FT-001] HTTP Server Setup

**Team**: Foundation  
**Sprint**: Week 1  
**Size**: M (2 days)  
**Owner**: [Rust Lead]  
**Status**: Backlog  
**Priority**: P0 (Critical Path)

---

## User Story

As a worker operator, I want a basic HTTP server running, so that I can make requests to the worker.

---

## Acceptance Criteria

- [ ] Axum server starts on port specified via CLI flag `--port`
- [ ] Server binds to `0.0.0.0` (all interfaces)
- [ ] GET /health endpoint returns 200 OK with `{"status":"starting"}`
- [ ] Server logs startup message: "Worker starting on port {port}"
- [ ] Server handles SIGTERM gracefully (logs "Shutting down", exits with code 0)
- [ ] Integration test: `curl http://localhost:8080/health` succeeds

---

## Definition of Done

- [ ] Code written and reviewed
- [ ] Unit tests pass (>80% coverage for server setup)
- [ ] Integration test: server starts and responds to /health
- [ ] No compiler warnings (rustfmt, clippy)
- [ ] Documentation: README updated with "How to Run" section
- [ ] Demoed in Friday demo (show server starting, health check)

---

## Dependencies

**Depends on**: None (first story)  
**Blocks**: FT-002 (POST /execute needs server running)

---

## Technical Notes

### Implementation

**File**: `bin/worker-orcd/src/main.rs`

```rust
use axum::{routing::get, Router};
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "8080")]
    port: u16,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    
    let app = Router::new()
        .route("/health", get(health_handler));
    
    let addr = format!("0.0.0.0:{}", args.port);
    tracing::info!("Worker starting on port {}", args.port);
    
    axum::Server::bind(&addr.parse().unwrap())
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn health_handler() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({"status": "starting"}))
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.unwrap();
    tracing::info!("Shutting down");
}
```

### Testing

**Integration test**: `tests/integration/http_server.rs`

```rust
#[tokio::test]
async fn test_server_starts() {
    let server = spawn_server(8081).await;
    let response = reqwest::get("http://localhost:8081/health").await.unwrap();
    assert_eq!(response.status(), 200);
    server.shutdown().await;
}
```

### Spec References

- M0-W-1100: Command-line interface
- M0-W-1320: GET /health endpoint

---

## Progress Log

**YYYY-MM-DD**: Story created, ready for Week 1 sprint
