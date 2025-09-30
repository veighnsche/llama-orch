//! Drain and reload orchestrations (planning-only).
//!
//! TODO: Implement drain/reload lifecycle for atomic model swaps.
//! Spec: ORCH-3031, ORCH-3038 (drain/reload atomic and reversible; reload success toggles Ready, failure rolls back).
//! Checklist: CHECKLIST.md "Supervision & Lifecycle" → "Draining & reload: stop accepting new leases; wait for in-flight or force stop on deadline."
//! Usage: Called by orchestratord control API (POST /v1/pools/{id}/drain, POST /v1/pools/{id}/reload).
//! Expected API:
//!   - `DrainRequest::new(pool_id, deadline_ms) -> Self`
//!   - `ReloadRequest::new(pool_id, new_model_ref, new_engine_version) -> Self`
//!   - `execute_drain(req: DrainRequest, registry: &mut Registry) -> Result<DrainOutcome>`
//!   - `execute_reload(req: ReloadRequest, registry: &mut Registry, provisioner: &dyn Provisioner) -> Result<ReloadOutcome>`
//! Integration: Calls registry.set_draining(true), waits for active_leases → 0, then stops engine.
//! Reload: drain → stage new model → restart engine → health check → flip ready=true or rollback.
//! Tests: Integration test for drain/reload cycles with deadlines (CHECKLIST.md "Testing Strategy").

#[derive(Debug, Clone)]
pub struct DrainRequest;

#[derive(Debug, Clone)]
pub struct ReloadRequest;
