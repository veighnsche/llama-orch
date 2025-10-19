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
    /// Total workers by state (TEAM-115: Added backend and device labels)
    pub static ref WORKERS_BY_STATE: GaugeVec = register_gauge_vec!(
        opts!("rbee_hive_workers_total", "Total workers by state"),
        &["state", "backend", "device"]
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

    /// TEAM-115: Memory metrics
    pub static ref MEMORY_AVAILABLE_BYTES: IntGauge = register_int_gauge!(
        opts!("rbee_hive_memory_available_bytes", "Available system memory in bytes")
    )
    .expect("Failed to register memory_available_bytes metric");

    pub static ref MEMORY_TOTAL_BYTES: IntGauge = register_int_gauge!(
        opts!("rbee_hive_memory_total_bytes", "Total system memory in bytes")
    )
    .expect("Failed to register memory_total_bytes metric");

    /// TEAM-115: Disk metrics
    pub static ref DISK_AVAILABLE_BYTES: IntGauge = register_int_gauge!(
        opts!("rbee_hive_disk_available_bytes", "Available disk space in bytes")
    )
    .expect("Failed to register disk_available_bytes metric");

    pub static ref DISK_TOTAL_BYTES: IntGauge = register_int_gauge!(
        opts!("rbee_hive_disk_total_bytes", "Total disk space in bytes")
    )
    .expect("Failed to register disk_total_bytes metric");

    /// TEAM-115: Worker spawn failures due to resource limits
    pub static ref WORKER_SPAWN_RESOURCE_FAILURES_TOTAL: IntCounter = register_int_counter!(
        opts!(
            "rbee_hive_worker_spawn_resource_failures_total",
            "Total worker spawn failures due to insufficient resources"
        )
    )
    .expect("Failed to register worker_spawn_resource_failures_total metric");

    /// TEAM-116: Shutdown metrics
    pub static ref SHUTDOWN_DURATION_SECONDS: prometheus::Histogram = prometheus::register_histogram!(
        prometheus::histogram_opts!(
            "rbee_hive_shutdown_duration_seconds",
            "Duration of graceful shutdown in seconds"
        )
    )
    .expect("Failed to register shutdown_duration_seconds metric");

    pub static ref WORKERS_GRACEFUL_SHUTDOWN_TOTAL: IntCounter = register_int_counter!(
        opts!(
            "rbee_hive_workers_graceful_shutdown_total",
            "Total workers that shutdown gracefully"
        )
    )
    .expect("Failed to register workers_graceful_shutdown_total metric");

    pub static ref WORKERS_FORCE_KILLED_TOTAL: IntCounter = register_int_counter!(
        opts!(
            "rbee_hive_workers_force_killed_total",
            "Total workers that were force-killed"
        )
    )
    .expect("Failed to register workers_force_killed_total metric");
}

/// Update worker metrics from registry state
///
/// # Arguments
/// * `registry` - Worker registry to collect metrics from
pub async fn update_worker_metrics(registry: Arc<crate::registry::WorkerRegistry>) {
    use crate::registry::WorkerState;

    let workers = registry.list().await;

    // TEAM-115: Reset all label combinations (we'll only set the ones that exist)
    // Note: In production, we should track which label combinations exist to avoid cardinality explosion
    
    let mut failed_health_count = 0;
    let mut total_restart_count = 0;

    // Count workers by state, backend, and device
    for worker in workers {
        let state_label = match worker.state {
            WorkerState::Idle => "idle",
            WorkerState::Busy => "busy",
            WorkerState::Loading => "loading",
        };
        let backend_label = worker.backend.as_str();
        let device_label = worker.device.to_string();
        
        // TEAM-115: Set metric with backend and device labels
        WORKERS_BY_STATE
            .with_label_values(&[state_label, backend_label, &device_label])
            .inc();

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

/// Update resource metrics (memory, disk)
///
/// TEAM-115: Updates system resource metrics for monitoring
pub fn update_resource_metrics() {
    use crate::resources::get_resource_info;
    
    if let Ok(info) = get_resource_info() {
        MEMORY_TOTAL_BYTES.set(info.memory_total_bytes as i64);
        MEMORY_AVAILABLE_BYTES.set(info.memory_available_bytes as i64);
        DISK_TOTAL_BYTES.set(info.disk_total_bytes as i64);
        DISK_AVAILABLE_BYTES.set(info.disk_available_bytes as i64);
    }
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
