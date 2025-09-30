//! Backoff and circuit breaker policies (planning-only).
//!
//! TODO: Implement exponential backoff with jitter for engine restart supervision.
//! Spec: ORCH-3038 (driver/CUDA errors → restart with backoff), ORCH-3040 (circuit breakers).
//! Checklist: CHECKLIST.md "Supervision & Lifecycle" → "Backoff policy: exponential with jitter; max backoff cap; reset on stable run."
//! Usage: Called by supervision module (not yet implemented) when engine crashes or health checks fail.
//! Expected API:
//!   - `BackoffPolicy::new(initial_ms, max_ms, jitter_factor) -> Self`
//!   - `BackoffPolicy::next_delay(&mut self) -> Duration` (exponential increment)
//!   - `BackoffPolicy::reset(&mut self)` (on stable run)
//!   - Optional: circuit breaker state machine (open/half-open/closed) with failure threshold.
//! Tests: BDD step "restart storms are bounded by circuit breaker" (test-harness/bdd/src/steps/pool_manager.rs:48).

#[derive(Debug, Clone, Copy)]
pub struct BackoffPolicy {
    pub initial_ms: u64,
    pub max_ms: u64,
}
