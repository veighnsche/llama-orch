# worker-registry

Shared worker registry for tracking workers across pool nodes.

## Purpose

This crate provides a SQLite-backed worker registry that is shared between:
- `queen-rbee` - Orchestrator daemon (M1+)
- `rbee-keeper` - Orchestrator CLI

## Features

- SQLite-backed persistence
- Worker tracking by node and model
- State management (idle, ready, busy, loading)
- Health check timestamp tracking
- Thread-safe async operations

## Usage

```rust
use worker_registry::{WorkerRegistry, WorkerInfo};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create registry
    let registry = WorkerRegistry::new("~/.rbee/workers.db".to_string());
    registry.init().await?;

    // Find existing worker
    if let Some(worker) = registry.find_worker("mac", "hf:model").await? {
        println!("Found worker: {}", worker.url);
    }

    // Register new worker
    let worker = WorkerInfo {
        id: "worker-123".to_string(),
        node: "mac".to_string(),
        url: "http://mac.home.arpa:8081".to_string(),
        model_ref: "hf:model".to_string(),
        state: "idle".to_string(),
        last_health_check_unix: 1234567890,
    };
    registry.register_worker(&worker).await?;

    Ok(())
}
```

## Schema

```sql
CREATE TABLE workers (
    id TEXT PRIMARY KEY,
    node TEXT NOT NULL,
    url TEXT NOT NULL,
    model_ref TEXT NOT NULL,
    state TEXT NOT NULL,
    last_health_check_unix INTEGER NOT NULL
);
```

## Created by

TEAM-027
