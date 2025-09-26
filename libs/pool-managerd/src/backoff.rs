//! Backoff and circuit breaker policies (planning-only).

#[derive(Debug, Clone, Copy)]
pub struct BackoffPolicy {
    pub initial_ms: u64,
    pub max_ms: u64,
}
