//! Leases (planning-only).
//!
//! TODO: Implement lease tracking for active inference slots.
//! Spec: ORCH-3004 (bounded FIFO queue), ORCH-3010 (dispatch to Ready replicas with slots_free > 0).
//! Checklist: CHECKLIST.md "Capacity & Leases" → "Increment/decrement atomically; never negative (tests exist)."
//! Usage: Registry already has allocate_lease/release_lease; this module could provide richer lease metadata.
//! Expected API (if expanded beyond registry counters):
//!   - `Lease::new(lease_id, job_id, pool_id, allocated_at) -> Self`
//!   - `Lease::release(&self) -> LeaseId`
//!   - Optional: track per-lease metrics (tokens_in, tokens_out, duration) for capacity estimation.
//! Integration: Registry.allocate_lease/release_lease already implement atomic counters (registry.rs:408-454).
//! Decision: If registry counters are sufficient, this module can be deleted. Otherwise, expand for per-lease metadata.
//! Tests: Unit tests exist (OC-POOL-3007: leases never negative, registry.rs:483-494).
//! TODO: Decide if this module is needed or if registry counters are sufficient. If not needed, delete.

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LeaseId(String);

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Lease;
