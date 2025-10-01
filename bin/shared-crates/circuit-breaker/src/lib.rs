//! circuit-breaker — Circuit breaker pattern for service resilience
//!
//! Stops calling failing services temporarily, prevents cascading failures, allows recovery.
//!
//! # Key Responsibilities
//!
//! - Track failure rate per downstream service
//! - Open circuit after N consecutive failures
//! - Half-open state (test if service recovered)
//! - Close circuit when service healthy again
//! - Fail fast (return error immediately when circuit open)
//!
//! # Example
//!
//! ```rust
//! use circuit_breaker::{CircuitBreaker, CircuitState};
//!
//! let breaker = CircuitBreaker::new("pool-manager", 5); // Open after 5 failures
//!
//! match breaker.call(|| make_request()).await {
//!     Ok(response) => { /* Success */ }
//!     Err(e) if breaker.is_open() => { /* Circuit open, fail fast */ }
//!     Err(e) => { /* Request failed, circuit may open */ }
//! }
//! ```

// High-importance crate: TIER 2 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::integer_arithmetic)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::missing_errors_doc)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CircuitError {
    #[error("circuit open")]
    CircuitOpen,
    #[error("request failed: {0}")]
    RequestFailed(String),
}

pub type Result<T> = std::result::Result<T, CircuitError>;

/// Circuit state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

struct CircuitData {
    state: CircuitState,
    failure_count: u32,
    last_failure: Option<Instant>,
    opened_at: Option<Instant>,
}

/// Circuit breaker
pub struct CircuitBreaker {
    name: String,
    threshold: u32,
    timeout: Duration,
    data: Arc<Mutex<CircuitData>>,
}

impl CircuitBreaker {
    pub fn new(name: &str, threshold: u32) -> Self {
        Self {
            name: name.to_string(),
            threshold,
            timeout: Duration::from_secs(30),
            data: Arc::new(Mutex::new(CircuitData {
                state: CircuitState::Closed,
                failure_count: 0,
                last_failure: None,
                opened_at: None,
            })),
        }
    }
    
    pub fn is_open(&self) -> bool {
        let data = self.data.lock().ok();
        data.map(|d| d.state == CircuitState::Open).unwrap_or(false)
    }
    
    pub fn record_success(&self) {
        if let Ok(mut data) = self.data.lock() {
            data.failure_count = 0;
            data.state = CircuitState::Closed;
            tracing::debug!(circuit = %self.name, "Circuit closed");
        }
    }
    
    pub fn record_failure(&self) {
        if let Ok(mut data) = self.data.lock() {
            data.failure_count = data.failure_count.saturating_add(1);
            data.last_failure = Some(Instant::now());
            
            if data.failure_count >= self.threshold {
                data.state = CircuitState::Open;
                data.opened_at = Some(Instant::now());
                tracing::warn!(
                    circuit = %self.name,
                    failures = %data.failure_count,
                    "Circuit opened"
                );
            }
        }
    }
    
    pub fn try_reset(&self) -> bool {
        if let Ok(mut data) = self.data.lock() {
            if data.state == CircuitState::Open {
                if let Some(opened_at) = data.opened_at {
                    if opened_at.elapsed() >= self.timeout {
                        data.state = CircuitState::HalfOpen;
                        tracing::info!(circuit = %self.name, "Circuit half-open");
                        return true;
                    }
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_circuit_breaker() {
        let breaker = CircuitBreaker::new("test", 3);
        
        assert!(!breaker.is_open());
        
        // Record failures
        breaker.record_failure();
        breaker.record_failure();
        assert!(!breaker.is_open());
        
        breaker.record_failure();
        assert!(breaker.is_open()); // Should open after 3 failures
        
        // Success closes circuit
        breaker.record_success();
        assert!(!breaker.is_open());
    }
}
