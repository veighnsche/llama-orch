// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Good Prometheus integration

// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Prometheus metrics implementation

//! Prometheus metrics for rbee-hive
//!
//! Exposes metrics about worker pool state, health, and operations.
//!
//! # Metrics Exposed
//! - `rbee_hive_workers_total{state}` - Total workers by state (idle, busy, loading)
//! - `rbee_hive_workers_failed_health_checks` - Workers with failed health checks
//! - `rbee_hive_workers_restart_count` - Total restart count across all workers
//! - `rbee_hive_models_downloaded_total` - Total models downloaded
//! - `rbee_hive_download_active` - Currently active downloads
//!
//! # Update Frequency
//! Metrics are updated on-demand when the `/metrics` endpoint is scraped.
//! Prometheus typically scrapes every 15-30 seconds.
//!
//! # Usage
//! ```rust,no_run
//! use rbee_hive::metrics;
//! use std::sync::Arc;
//!
//! // Update metrics before rendering
//! metrics::update_worker_metrics(registry.clone()).await;
//! let metrics_text = metrics::render_metrics()?;
//! ```
//!
//! Created by: TEAM-104

use lazy_static::lazy_static;
use prometheus::{
    opts, register_gauge_vec, register_int_counter, register_int_gauge, Encoder, GaugeVec,
    IntCounter, IntGauge, TextEncoder,
};
use std::sync::Arc;

lazy_static! {
    /// Total workers by state
    pub static ref WORKERS_BY_STATE: GaugeVec = register_gauge_vec!(
        opts!("rbee_hive_workers_total", "Total workers by state"),
        &["state"]
    )
    .expect("Failed to register workers_by_state metric");

    /// Workers with failed health checks
    pub static ref WORKERS_FAILED_HEALTH: IntGauge = register_int_gauge!(
        opts!(
            "rbee_hive_workers_failed_health_checks",
            "Workers with failed health checks"
        )
    )
    .expect("Failed to register workers_failed_health metric");

    /// Total restart count across all workers
    pub static ref WORKERS_RESTART_COUNT: IntGauge = register_int_gauge!(
        opts!(
            "rbee_hive_workers_restart_count",
            "Total restart count across all workers"
        )
    )
    .expect("Failed to register workers_restart_count metric");

    /// Total models downloaded
    pub static ref MODELS_DOWNLOADED_TOTAL: IntCounter = register_int_counter!(
        opts!(
            "rbee_hive_models_downloaded_total",
            "Total models downloaded"
        )
    )
    .expect("Failed to register models_downloaded_total metric");

    /// Currently active downloads
    pub static ref DOWNLOADS_ACTIVE: IntGauge = register_int_gauge!(
        opts!("rbee_hive_download_active", "Currently active downloads")
    )
    .expect("Failed to register downloads_active metric");

    /// TEAM-114: Worker restart failures total
    pub static ref WORKER_RESTART_FAILURES_TOTAL: IntCounter = register_int_counter!(
        opts!(
            "rbee_hive_worker_restart_failures_total",
            "Total worker restart failures"
        )
    )
    .expect("Failed to register worker_restart_failures_total metric");

    /// TEAM-114: Circuit breaker activations total
    pub static ref CIRCUIT_BREAKER_ACTIVATIONS_TOTAL: IntCounter = register_int_counter!(
        opts!(
            "rbee_hive_circuit_breaker_activations_total",
            "Total circuit breaker activations"
        )
    )
    .expect("Failed to register circuit_breaker_activations_total metric");
}

/// Update worker metrics from registry state
///
/// # Arguments
/// * `registry` - Worker registry to collect metrics from
pub async fn update_worker_metrics(registry: Arc<crate::registry::WorkerRegistry>) {
    use crate::registry::WorkerState;

    let workers = registry.list().await;

    // Reset state counters
    WORKERS_BY_STATE.with_label_values(&["idle"]).set(0.0);
    WORKERS_BY_STATE.with_label_values(&["busy"]).set(0.0);
    WORKERS_BY_STATE.with_label_values(&["loading"]).set(0.0);

    let mut failed_health_count = 0;
    let mut total_restart_count = 0;

    // Count workers by state
    for worker in workers {
        let state_label = match worker.state {
            WorkerState::Idle => "idle",
            WorkerState::Busy => "busy",
            WorkerState::Loading => "loading",
        };
        WORKERS_BY_STATE.with_label_values(&[state_label]).inc();

        if worker.failed_health_checks > 0 {
            failed_health_count += 1;
        }

        total_restart_count += worker.restart_count as i64;
    }

    WORKERS_FAILED_HEALTH.set(failed_health_count);
    WORKERS_RESTART_COUNT.set(total_restart_count);
}

/// Update download metrics from download tracker
///
/// NOTE: TEAM-104: DownloadTracker doesn't expose list_active() method yet.
/// For now, this is a placeholder. Downloads are tracked via DOWNLOADS_ACTIVE
/// counter which is incremented/decremented in the download endpoints.
///
/// # Arguments
/// * `_download_tracker` - Download tracker (unused for now)
pub async fn update_download_metrics<T>(_download_tracker: Arc<T>) {
    // TEAM-104: Placeholder - download metrics are updated directly in endpoints
    // via DOWNLOADS_ACTIVE.inc() / DOWNLOADS_ACTIVE.dec()
}

/// Render metrics in Prometheus text format
///
/// # Returns
/// Metrics as a string in Prometheus exposition format
pub fn render_metrics() -> Result<String, Box<dyn std::error::Error>> {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;
    Ok(String::from_utf8(buffer)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_registration() {
        // Metrics should be registered without panicking
        let _ = &*WORKERS_BY_STATE;
        let _ = &*WORKERS_FAILED_HEALTH;
        let _ = &*WORKERS_RESTART_COUNT;
        let _ = &*MODELS_DOWNLOADED_TOTAL;
        let _ = &*DOWNLOADS_ACTIVE;
        let _ = &*WORKER_RESTART_FAILURES_TOTAL; // TEAM-114
        let _ = &*CIRCUIT_BREAKER_ACTIVATIONS_TOTAL; // TEAM-114
    }

    #[test]
    fn test_render_metrics() {
        // Ensure metrics are registered by accessing them
        let _ = &*WORKERS_BY_STATE;
        let _ = &*WORKERS_FAILED_HEALTH;
        let _ = &*WORKERS_RESTART_COUNT;
        let _ = &*MODELS_DOWNLOADED_TOTAL;
        let _ = &*DOWNLOADS_ACTIVE;
        let _ = &*WORKER_RESTART_FAILURES_TOTAL; // TEAM-114
        let _ = &*CIRCUIT_BREAKER_ACTIVATIONS_TOTAL; // TEAM-114

        // Should be able to render metrics
        let result = render_metrics();
        assert!(result.is_ok());
        let metrics_text = result.unwrap();
        // Metrics should not be empty
        assert!(!metrics_text.is_empty(), "Metrics output should not be empty");
    }

    #[tokio::test]
    async fn test_update_worker_metrics() {
        use crate::registry::WorkerRegistry;
        let registry = Arc::new(WorkerRegistry::new());

        // Should not panic with empty registry
        update_worker_metrics(registry).await;
    }

    #[tokio::test]
    async fn test_update_download_metrics() {
        use crate::download_tracker::DownloadTracker;
        let tracker = Arc::new(DownloadTracker::new());

        // Should not panic with empty tracker
        update_download_metrics(tracker).await;
    }
}
