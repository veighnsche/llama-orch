//! GET /metrics endpoint - Prometheus metrics
//!
//! Exposes Prometheus-compatible metrics about the worker pool.
//!
//! Created by: TEAM-104

use crate::http::routes::AppState;
use axum::{extract::State, http::StatusCode};
use tracing::debug;

/// Handle GET /metrics
///
/// Returns Prometheus-compatible metrics in text format
///
/// # Metrics Exposed
/// - `rbee_hive_workers_total{state}` - Total workers by state
/// - `rbee_hive_workers_failed_health_checks` - Workers with failed health checks
/// - `rbee_hive_workers_restart_count` - Total restart count
/// - `rbee_hive_models_downloaded_total` - Total models downloaded
/// - `rbee_hive_download_active` - Currently active downloads
pub async fn handle_metrics(State(state): State<AppState>) -> Result<String, (StatusCode, String)> {
    debug!("Metrics requested");

    // Update metrics from current state
    rbee_hive::metrics::update_worker_metrics(state.registry.clone()).await;
    rbee_hive::metrics::update_download_metrics(state.download_tracker.clone()).await;

    // Render metrics
    rbee_hive::metrics::render_metrics().map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to render metrics: {}", e))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provisioner::ModelProvisioner;
    use crate::registry::WorkerRegistry;
    use model_catalog::ModelCatalog;
    use rbee_hive::download_tracker::DownloadTracker;
    use std::path::PathBuf;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let registry = Arc::new(WorkerRegistry::new());
        let catalog = Arc::new(ModelCatalog::new(":memory:".to_string()));
        let provisioner = Arc::new(ModelProvisioner::new(PathBuf::from("/tmp")));
        let download_tracker = Arc::new(DownloadTracker::new());
        let addr: std::net::SocketAddr = "127.0.0.1:9200".parse().unwrap();
        let expected_token = "test-token".to_string();

        let state = AppState {
            registry,
            model_catalog: catalog,
            provisioner,
            download_tracker,
            server_addr: addr,
            expected_token,
        };

        let result = handle_metrics(State(state)).await;
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.contains("rbee_hive"));
    }
}
