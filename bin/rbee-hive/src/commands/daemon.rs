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
use crate::shutdown::{shutdown_all_workers, ShutdownConfig};
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

    // TEAM-116: Setup graceful shutdown handler with new orchestration
    let registry_shutdown = registry.clone();
    let shutdown_config = ShutdownConfig::default();
    
    tokio::select! {
        result = server.run() => {
            result?;
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Shutdown signal received - initiating graceful shutdown");
            let metrics = shutdown_all_workers(registry_shutdown, shutdown_config).await;
            
            // Emit shutdown metrics
            tracing::info!(
                "rbee_hive_shutdown_duration_seconds = {:.2}",
                metrics.duration_secs
            );
            tracing::info!(
                "rbee_hive_workers_graceful_shutdown_total = {}",
                metrics.graceful_shutdown
            );
            tracing::info!(
                "rbee_hive_workers_force_killed_total = {}",
                metrics.force_killed
            );
        }
    }

    Ok(())
}

// TEAM-116: Old shutdown functions removed - now using shutdown module
