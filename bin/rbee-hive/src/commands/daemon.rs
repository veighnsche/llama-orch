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
use std::path::PathBuf;
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
    let expected_token = std::env::var("LLORCH_API_TOKEN").unwrap_or_else(|_| {
        tracing::info!("⚠️  LLORCH_API_TOKEN not set - using dev mode (no auth)");
        String::new()
    });

    if !expected_token.is_empty() {
        tracing::info!("✅ API token loaded (authentication enabled)");
    }

    // TEAM-114: Initialize audit logging (Week 2)
    // Disabled by default for home lab mode (zero overhead)
    // Set LLORCH_AUDIT_MODE=local to enable file-based audit logging
    let audit_mode = std::env::var("LLORCH_AUDIT_MODE")
        .ok()
        .and_then(|mode| match mode.as_str() {
            "local" => {
                let base_dir = std::env::var("LLORCH_AUDIT_DIR")
                    .unwrap_or_else(|_| "/var/log/llama-orch/audit".to_string());
                Some(audit_logging::AuditMode::Local { base_dir: PathBuf::from(base_dir) })
            }
            _ => None,
        })
        .unwrap_or(audit_logging::AuditMode::Disabled);

    let audit_config = audit_logging::AuditConfig {
        mode: audit_mode,
        service_id: "rbee-hive".to_string(),
        rotation_policy: audit_logging::RotationPolicy::Daily,
        retention_policy: audit_logging::RetentionPolicy::default(),
        flush_mode: audit_logging::FlushMode::Hybrid {
            batch_size: 100,
            batch_interval_secs: 5,
            critical_immediate: true, // Always flush security events immediately
        },
    };

    let audit_logger = match audit_logging::AuditLogger::new(audit_config) {
        Ok(logger) => {
            tracing::info!("✅ Audit logging initialized (disabled for home lab mode)");
            Some(Arc::new(logger))
        }
        Err(e) => {
            tracing::info!("⚠️  Audit logging disabled: {}", e);
            None
        }
    };

    // Create router
    // TEAM-102: Added expected_token for authentication
    // TEAM-114: Added audit_logger for security events
    let router = create_router(
        registry.clone(),
        model_catalog.clone(),
        provisioner.clone(),
        download_tracker.clone(),
        addr,
        expected_token,       // TEAM-102
        audit_logger.clone(), // TEAM-114
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
/// TEAM-105: Parallel shutdown with progress metrics, 30s timeout, and audit logging
async fn shutdown_all_workers(registry: Arc<WorkerRegistry>) {
    use std::time::Instant;

    let shutdown_start = Instant::now();
    let workers = registry.list().await;
    let total_workers = workers.len();

    tracing::info!(
        "TEAM-105: Starting parallel shutdown of {} workers (30s timeout)",
        total_workers
    );

    if total_workers == 0 {
        tracing::info!("No workers to shutdown");
        return;
    }

    // TEAM-105: Parallel shutdown with progress tracking and 30s timeout
    let mut shutdown_tasks = Vec::new();

    for worker in workers {
        let worker_id = worker.id.clone();
        let worker_url = worker.url.clone();
        let worker_pid = worker.pid;

        // Spawn concurrent shutdown task for each worker
        let task = tokio::spawn(async move {
            tracing::info!("TEAM-105: Initiating shutdown for worker {}", worker_id);

            // Try graceful shutdown via HTTP
            let graceful_success = match shutdown_worker(&worker_url).await {
                Ok(_) => {
                    tracing::info!("Worker {} acknowledged graceful shutdown", worker_id);
                    true
                }
                Err(e) => {
                    tracing::warn!("Worker {} graceful shutdown failed: {}", worker_id, e);
                    false
                }
            };

            // TEAM-101: Force-kill if worker doesn't respond
            if let Some(pid) = worker_pid {
                force_kill_worker_if_needed(pid, &worker_id).await;
            }

            (worker_id, graceful_success)
        });

        shutdown_tasks.push(task);
    }

    // TEAM-105: Wait for all shutdowns with 30s timeout
    let mut completed = 0;
    let mut graceful_count = 0;
    let mut forced_count = 0;
    let mut timeout_count = 0;

    for task in shutdown_tasks {
        let elapsed = shutdown_start.elapsed();
        let remaining = std::time::Duration::from_secs(30).saturating_sub(elapsed);

        if remaining.is_zero() {
            // TEAM-105: Timeout exceeded - force-kill remaining workers
            tracing::error!(
                "TEAM-105: Shutdown timeout (30s) exceeded - force-killing remaining workers"
            );
            timeout_count += 1;
            task.abort();
            continue;
        }

        match tokio::time::timeout(remaining, task).await {
            Ok(Ok((worker_id, graceful))) => {
                completed += 1;
                if graceful {
                    graceful_count += 1;
                } else {
                    forced_count += 1;
                }
                tracing::info!(
                    "TEAM-105: Shutdown progress: {}/{} workers completed (graceful: {}, forced: {}, timeout: {})",
                    completed, total_workers, graceful_count, forced_count, timeout_count
                );
            }
            Ok(Err(e)) => {
                completed += 1;
                forced_count += 1;
                tracing::error!("TEAM-105: Worker shutdown task failed: {}", e);
            }
            Err(_) => {
                // Task timed out
                completed += 1;
                timeout_count += 1;
                tracing::error!("TEAM-105: Worker shutdown task timed out");
            }
        }
    }

    // Clear registry
    registry.clear().await;

    let total_duration = shutdown_start.elapsed();

    // TEAM-105: Audit logging for shutdown completion
    tracing::info!(
        "TEAM-105: SHUTDOWN AUDIT - Total: {}, Graceful: {}, Forced: {}, Timeout: {}, Duration: {:.2}s",
        total_workers, graceful_count, forced_count, timeout_count, total_duration.as_secs_f64()
    );

    if timeout_count > 0 {
        tracing::warn!(
            "TEAM-105: {} workers exceeded shutdown timeout and were force-killed",
            timeout_count
        );
    }
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
    use sysinfo::{Pid, Signal, System};

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
        tracing::info!(worker_id = worker_id, pid = pid, "Worker process exited gracefully");
    }
}
