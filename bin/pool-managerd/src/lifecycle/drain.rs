//! Drain lifecycle for graceful pool shutdown.
//!
//! Spec: OC-POOL-3010, ORCH-3031, OC-CTRL-2002
//! Implements graceful drain with deadline and force-stop escalation.

use anyhow::Result;
use std::time::{Duration, Instant};

use crate::lifecycle::preload;
use crate::registry::Registry;

/// Drain request with deadline
#[derive(Debug, Clone)]
pub struct DrainRequest {
    pub pool_id: String,
    pub deadline_ms: u64,
}

impl DrainRequest {
    pub fn new(pool_id: impl Into<String>, deadline_ms: u64) -> Self {
        Self { pool_id: pool_id.into(), deadline_ms }
    }
}

/// Drain outcome
#[derive(Debug, Clone)]
pub struct DrainOutcome {
    pub pool_id: String,
    pub force_stopped: bool,
    pub duration_ms: u64,
    pub final_lease_count: i32,
}

/// Execute drain: set draining flag, wait for leases to drain, stop engine
pub fn execute_drain(req: DrainRequest, registry: &mut Registry) -> Result<DrainOutcome> {
    let start = Instant::now();
    let deadline = start + Duration::from_millis(req.deadline_ms);

    tracing::info!(
        pool_id = %req.pool_id,
        deadline_ms = req.deadline_ms,
        "starting drain"
    );

    // Set draining flag - refuses new lease allocations
    registry.set_draining(&req.pool_id, true);

    // Wait for active leases to drain
    let mut force_stopped = false;
    loop {
        let active_leases = registry.get_active_leases(&req.pool_id);

        if active_leases == 0 {
            tracing::info!(
                pool_id = %req.pool_id,
                "all leases drained naturally"
            );
            break;
        }

        if Instant::now() >= deadline {
            tracing::warn!(
                pool_id = %req.pool_id,
                active_leases = active_leases,
                "drain deadline exceeded, force-stopping"
            );
            force_stopped = true;
            break;
        }

        // Poll every 100ms
        std::thread::sleep(Duration::from_millis(100));
    }

    // Stop the engine process
    if let Err(e) = preload::stop_pool(&req.pool_id) {
        tracing::error!(
            pool_id = %req.pool_id,
            error = %e,
            "failed to stop engine during drain"
        );
        // Continue - we still mark as drained
    }

    // Update registry health to not ready
    registry.set_health(&req.pool_id, crate::health::HealthStatus { live: false, ready: false });

    let duration_ms = start.elapsed().as_millis() as u64;
    let final_lease_count = registry.get_active_leases(&req.pool_id);

    tracing::info!(
        pool_id = %req.pool_id,
        duration_ms = duration_ms,
        force_stopped = force_stopped,
        final_lease_count = final_lease_count,
        "drain completed"
    );

    Ok(DrainOutcome { pool_id: req.pool_id, force_stopped, duration_ms, final_lease_count })
}
