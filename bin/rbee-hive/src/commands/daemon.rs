//! Daemon command - start HTTP pool manager
//!
//! Per test-001-mvp.md Phase 2: Pool Preflight
//! Runs persistent HTTP server for worker management
//!
//! TEAM-030: Worker registry is ephemeral (in-memory), model catalog is persistent (SQLite)
//! TEAM-034: Added download tracker for SSE streaming
//!
//! Created by: TEAM-027
//! Modified by: TEAM-029, TEAM-030, TEAM-034

use anyhow::Result;
use model_catalog::ModelCatalog;
use rbee_hive::download_tracker::DownloadTracker;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::http::{create_router, HttpServer};
use crate::monitor::health_monitor_loop;
use crate::provisioner::ModelProvisioner;
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

    // Create worker registry (TEAM-030: In-memory, ephemeral)
    let registry = Arc::new(WorkerRegistry::new());
    tracing::info!("Worker registry initialized (in-memory, ephemeral)");

    // TEAM-029: Initialize model catalog (SQLite, persistent)
    let model_catalog_path =
        dirs::home_dir().unwrap_or_default().join(".rbee/models.db").to_string_lossy().to_string();
    let model_catalog = Arc::new(ModelCatalog::new(model_catalog_path));
    model_catalog.init().await?;
    tracing::info!("Model catalog initialized (SQLite, persistent)");

    // TEAM-029: Initialize model provisioner
    let model_base_dir =
        std::env::var("RBEE_MODEL_BASE_DIR").unwrap_or_else(|_| ".test-models".to_string());
    tracing::info!("Model provisioner initialized (base_dir: {})", model_base_dir);
    let provisioner = Arc::new(ModelProvisioner::new(model_base_dir.into()));

    // TEAM-034: Initialize download tracker for SSE streaming
    let download_tracker = Arc::new(DownloadTracker::new());
    tracing::info!("Download tracker initialized");

    // TEAM-102: Load API token for authentication
    // TODO: Replace with secrets-management file-based loading
    let expected_token = std::env::var("LLORCH_API_TOKEN")
        .unwrap_or_else(|_| {
            tracing::info!("⚠️  LLORCH_API_TOKEN not set - using dev mode (no auth)");
            String::new()
        });
    
    if !expected_token.is_empty() {
        tracing::info!("✅ API token loaded (authentication enabled)");
    }

    // Create router
    // TEAM-102: Added expected_token for authentication
    let router = create_router(
        registry.clone(),
        model_catalog.clone(),
        provisioner.clone(),
        download_tracker.clone(),
        addr,
        expected_token, // TEAM-102
    );

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

    // TEAM-030: Setup graceful shutdown handler
    let registry_shutdown = registry.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Shutdown signal received - cleaning up workers");
        shutdown_all_workers(registry_shutdown).await;
    });

    // Run server
    server.run().await?;

    Ok(())
}

/// Shutdown all workers (TEAM-030: Cascading shutdown)
/// TEAM-101: Enhanced with force-kill capability
async fn shutdown_all_workers(registry: Arc<WorkerRegistry>) {
    let workers = registry.list().await;
    tracing::info!("Shutting down {} workers", workers.len());

    for worker in workers {
        tracing::info!("Sending shutdown to worker {}", worker.id);

        // Try to gracefully shutdown worker via HTTP
        if let Err(e) = shutdown_worker(&worker.url).await {
            tracing::warn!("Failed to shutdown worker {} gracefully: {}", worker.id, e);
        }
        
        // TEAM-101: Force-kill if worker doesn't respond
        if let Some(pid) = worker.pid {
            force_kill_worker_if_needed(pid, &worker.id).await;
        }
    }

    // Clear registry
    registry.clear().await;
    tracing::info!("All workers shutdown complete");
}

/// Shutdown a single worker via HTTP
async fn shutdown_worker(worker_url: &str) -> Result<()> {
    let client = reqwest::Client::new();

    // Try POST /v1/shutdown endpoint
    let response = client
        .post(format!("{}/v1/shutdown", worker_url))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await?;

    if response.status().is_success() {
        tracing::info!("Worker at {} acknowledged shutdown", worker_url);
    }

    Ok(())
}

/// TEAM-101: Force-kill worker if it doesn't respond to graceful shutdown
/// Implements SIGTERM → wait 10s → SIGKILL sequence
async fn force_kill_worker_if_needed(pid: u32, worker_id: &str) {
    use sysinfo::{System, Pid, Signal};
    
    // Wait 10 seconds for graceful shutdown
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;
    
    // Check if process still exists
    let mut sys = System::new();
    sys.refresh_processes();
    
    let pid_obj = Pid::from_u32(pid);
    if let Some(process) = sys.process(pid_obj) {
        tracing::warn!(
            worker_id = worker_id,
            pid = pid,
            "Worker did not respond to graceful shutdown, sending SIGKILL"
        );
        
        // Send SIGKILL
        if process.kill_with(Signal::Kill).is_some() {
            tracing::info!(
                worker_id = worker_id,
                pid = pid,
                signal = "SIGKILL",
                "TEAM-101: Force-killed worker process"
            );
        } else {
            tracing::error!(
                worker_id = worker_id,
                pid = pid,
                "Failed to send SIGKILL to worker process"
            );
        }
    } else {
        tracing::info!(
            worker_id = worker_id,
            pid = pid,
            "Worker process exited gracefully"
        );
    }
}
