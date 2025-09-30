//! pool-managerd daemon entrypoint.
//!
//! Standalone daemon that manages engine lifecycle (spawn, health, supervision).
//! Exposes HTTP API for orchestratord to call.
//!
//! Responsibilities:
//! 1. Spawn engines from PreparedEngine (via POST /pools/{id}/preload)
//! 2. Monitor health and update registry
//! 3. Supervise processes with backoff on crash
//! 4. Expose HTTP API for orchestratord

use pool_managerd::api::routes::{create_router, AppState};
use pool_managerd::core::registry::Registry;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Create shared registry
    let registry = Arc::new(Mutex::new(Registry::new()));

    // Create app state
    let state = AppState {
        registry: registry.clone(),
    };

    // Create router
    let app = create_router(state);

    // Bind address from env or default
    let addr = std::env::var("POOL_MANAGERD_ADDR").unwrap_or_else(|_| "127.0.0.1:9200".to_string());
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!("pool-managerd listening on {}", addr);

    // Start server
    axum::serve(listener, app).await?;

    Ok(())
}
