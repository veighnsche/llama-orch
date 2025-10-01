//! Reload lifecycle for atomic model swaps.
//!
//! Spec: ORCH-3031, ORCH-3038, OC-CTRL-2003

use anyhow::Result;

use crate::lifecycle::{drain, preload};
use crate::registry::Registry;
// TODO: Remove PreparedEngine - migrating to worker-orcd
// use provisioners_engine_provisioner::PreparedEngine;

/// Reload request
#[derive(Debug, Clone)]
pub struct ReloadRequest {
    pub pool_id: String,
    pub new_model_ref: String,
    pub drain_deadline_ms: u64,
}

impl ReloadRequest {
    pub fn new(
        pool_id: impl Into<String>,
        new_model_ref: impl Into<String>,
        drain_deadline_ms: u64,
    ) -> Self {
        Self { pool_id: pool_id.into(), new_model_ref: new_model_ref.into(), drain_deadline_ms }
    }
}

/// Reload outcome
#[derive(Debug, Clone)]
pub struct ReloadOutcome {
    pub pool_id: String,
    pub success: bool,
    pub rolled_back: bool,
    pub new_engine_version: Option<String>,
    pub duration_ms: u64,
}

// TODO: Remove this function - migrating to worker-orcd
/*
/// Execute reload: drain → stage model → restart engine → health check → ready or rollback
pub fn execute_reload(
    req: ReloadRequest,
    registry: &mut Registry,
    new_prepared: PreparedEngine,
) -> Result<ReloadOutcome> {
    let start = std::time::Instant::now();

    tracing::info!(
        pool_id = %req.pool_id,
        new_model_ref = %req.new_model_ref,
        "starting reload"
    );

    // Save old state for rollback
    let old_engine_version = registry.get_engine_version(&req.pool_id);
    let old_health = registry.get_health(&req.pool_id);

    // Step 1: Drain the pool
    let drain_req = drain::DrainRequest::new(&req.pool_id, req.drain_deadline_ms);
    let drain_outcome = drain::execute_drain(drain_req, registry)?;

    if drain_outcome.force_stopped {
        tracing::warn!(
            pool_id = %req.pool_id,
            "drain force-stopped, proceeding with reload"
        );
    }

    // Step 2: Stage new model (assumed already done via model-provisioner)
    // In real implementation, would call model-provisioner::ensure_present here

    // Step 3: Start new engine
    match preload::execute(new_prepared.clone(), registry) {
        Ok(outcome) => {
            // Success - new engine is running and ready
            tracing::info!(
                pool_id = %req.pool_id,
                new_engine_version = %outcome.pool_id,
                "reload succeeded"
            );

            let duration_ms = start.elapsed().as_millis() as u64;

            Ok(ReloadOutcome {
                pool_id: req.pool_id,
                success: true,
                rolled_back: false,
                new_engine_version: Some(new_prepared.engine_version.clone()),
                duration_ms,
            })
        }
        Err(e) => {
            // Failure - rollback
            tracing::error!(
                pool_id = %req.pool_id,
                error = %e,
                "reload failed, rolling back"
            );

            // Restore old state
            if let Some(old_health) = old_health {
                registry.set_health(&req.pool_id, old_health);
            }
            if let Some(old_version) = old_engine_version {
                registry.set_engine_version(&req.pool_id, old_version);
            }

            let duration_ms = start.elapsed().as_millis() as u64;

            Ok(ReloadOutcome {
                pool_id: req.pool_id,
                success: false,
                rolled_back: true,
                new_engine_version: None,
                duration_ms,
            })
        }
    }
}
*/
