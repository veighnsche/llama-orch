//! Daemon command - start HTTP pool manager
//!
//! Per test-001-mvp.md Phase 2: Pool Preflight
//! Runs persistent HTTP server for worker management
//!
//! Created by: TEAM-027

use anyhow::Result;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::http::{create_router, HttpServer};
use crate::monitor::health_monitor_loop;
use crate::registry::WorkerRegistry;
use crate::timeout::idle_timeout_loop;

pub async fn handle(addr: String) -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rbee_hive=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting rbee-hive daemon");

    // Parse address
    let addr: SocketAddr = addr.parse()?;
    tracing::info!("Binding to {}", addr);

    // Create worker registry
    let registry = Arc::new(WorkerRegistry::new());
    tracing::info!("Worker registry initialized");

    // Create router
    let router = create_router(registry.clone());

    // Create HTTP server
    let server = HttpServer::new(addr, router).await?;
    tracing::info!("HTTP server ready at http://{}", addr);

    // TEAM-027: Spawn background tasks
    // Per test-001-mvp.md Phase 5 (health monitor) and EC10 (idle timeout)
    let registry_clone = registry.clone();
    tokio::spawn(async move {
        health_monitor_loop(registry_clone).await;
    });
    tracing::info!("Health monitor loop started (30s interval)");

    let registry_clone = registry.clone();
    tokio::spawn(async move {
        idle_timeout_loop(registry_clone).await;
    });
    tracing::info!("Idle timeout loop started (5min threshold)");

    // Run server
    server.run().await?;

    Ok(())
}
