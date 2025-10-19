//! Worker restart policy
//!
//! Implements exponential backoff and circuit breaker for worker restarts.
//!
//! TEAM-114: Week 2 - Worker lifecycle features
//!
//! # Restart Policy
//! - Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, max 60s
//! - Max restart attempts: 3 (configurable)
//! - Circuit breaker: Stop after N failures in M minutes
//! - Jitter: ±20% to prevent thundering herd

use std::time::{Duration, SystemTime};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RestartError {
    #[error("max restart attempts exceeded ({0})")]
    MaxAttemptsExceeded(u32),
    #[error("circuit breaker open: {0} failures in {1} seconds")]
    CircuitBreakerOpen(u32, u64),
    #[error("restart too soon: {0}s remaining")]
    BackoffNotElapsed(u64),
}

pub type Result<T> = std::result::Result<T, RestartError>;

/// Restart policy configuration
#[derive(Debug, Clone)]
pub struct RestartPolicy {
    /// Maximum restart attempts before giving up
    pub max_attempts: u32,
    /// Base backoff delay in seconds (doubled each attempt)
    pub base_backoff_secs: u64,
    /// Maximum backoff delay in seconds
    pub max_backoff_secs: u64,
    /// Circuit breaker: max failures in time window
    pub circuit_breaker_threshold: u32,
    /// Circuit breaker: time window in seconds
    pub circuit_breaker_window_secs: u64,
    /// Add jitter to backoff (±20%)
    pub jitter_enabled: bool,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_backoff_secs: 1,
            max_backoff_secs: 60,
            circuit_breaker_threshold: 5,
            circuit_breaker_window_secs: 300, // 5 minutes
            jitter_enabled: true,
        }
    }
}

impl RestartPolicy {
    /// Calculate backoff delay for given attempt number
    ///
    /// # Arguments
    /// * `attempt` - Restart attempt number (0-indexed)
    ///
    /// # Returns
    /// Duration to wait before restart
    pub fn calculate_backoff(&self, attempt: u32) -> Duration {
        // Exponential backoff: base * 2^attempt
        let delay_secs = self.base_backoff_secs * 2u64.pow(attempt);
        let delay_secs = delay_secs.min(self.max_backoff_secs);

        // Add jitter (±20%) to prevent thundering herd
        let delay_secs = if self.jitter_enabled {
            let jitter = (delay_secs as f64 * 0.2) as u64;
            let jitter_offset = (rand::random::<f64>() * 2.0 - 1.0) * jitter as f64;
            ((delay_secs as f64 + jitter_offset).max(0.0)) as u64
        } else {
            delay_secs
        };

        Duration::from_secs(delay_secs)
    }

    /// Check if restart is allowed based on policy
    ///
    /// # Arguments
    /// * `restart_count` - Current restart count
    /// * `last_restart` - Last restart timestamp
    ///
    /// # Returns
    /// Ok(backoff_duration) if restart allowed, Err otherwise
    pub fn check_restart_allowed(
        &self,
        restart_count: u32,
        last_restart: Option<SystemTime>,
    ) -> Result<Duration> {
        // Check max attempts
        if restart_count >= self.max_attempts {
            return Err(RestartError::MaxAttemptsExceeded(self.max_attempts));
        }

        // Calculate required backoff
        let backoff = self.calculate_backoff(restart_count);

        // Check if backoff elapsed
        if let Some(last) = last_restart {
            let elapsed = SystemTime::now().duration_since(last).unwrap_or(Duration::ZERO);

            if elapsed < backoff {
                let remaining = (backoff - elapsed).as_secs();
                return Err(RestartError::BackoffNotElapsed(remaining));
            }
        }

        Ok(backoff)
    }

    /// Check circuit breaker status
    ///
    /// # Arguments
    /// * `failures` - List of failure timestamps
    ///
    /// # Returns
    /// Ok if circuit closed, Err if circuit open
    pub fn check_circuit_breaker(&self, failures: &[SystemTime]) -> Result<()> {
        let window_start = SystemTime::now()
            .checked_sub(Duration::from_secs(self.circuit_breaker_window_secs))
            .unwrap_or(SystemTime::UNIX_EPOCH);

        let recent_failures = failures.iter().filter(|&&t| t >= window_start).count() as u32;

        if recent_failures >= self.circuit_breaker_threshold {
            return Err(RestartError::CircuitBreakerOpen(
                recent_failures,
                self.circuit_breaker_window_secs,
            ));
        }

        Ok(())
    }

    /// Reset restart count if worker has been stable
    ///
    /// # Arguments
    /// * `last_restart` - Last restart timestamp
    /// * `stability_threshold_secs` - Seconds of uptime to consider stable (default: 300 = 5 min)
    ///
    /// # Returns
    /// true if restart count should be reset
    pub fn should_reset_count(
        &self,
        last_restart: Option<SystemTime>,
        stability_threshold_secs: u64,
    ) -> bool {
        if let Some(last) = last_restart {
            let uptime = SystemTime::now().duration_since(last).unwrap_or(Duration::ZERO);

            uptime.as_secs() >= stability_threshold_secs
        } else {
            false
        }
    }
}

/// Circuit breaker state tracker
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Recent failure timestamps
    failures: Vec<SystemTime>,
    /// Policy configuration
    policy: RestartPolicy,
}

impl CircuitBreaker {
    /// Create new circuit breaker
    pub fn new(policy: RestartPolicy) -> Self {
        Self { failures: Vec::new(), policy }
    }

    /// Record a failure
    pub fn record_failure(&mut self) {
        self.failures.push(SystemTime::now());

        // Trim old failures outside the window
        let window_start = SystemTime::now()
            .checked_sub(Duration::from_secs(self.policy.circuit_breaker_window_secs))
            .unwrap_or(SystemTime::UNIX_EPOCH);

        self.failures.retain(|&t| t >= window_start);
    }

    /// Check if circuit is open (too many failures)
    pub fn is_open(&self) -> bool {
        self.policy.check_circuit_breaker(&self.failures).is_err()
    }

    /// Reset circuit breaker
    pub fn reset(&mut self) {
        self.failures.clear();
    }

    /// Get recent failure count
    pub fn failure_count(&self) -> usize {
        self.failures.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_backoff() {
        let policy = RestartPolicy::default();

        // Attempt 0: 1s
        let backoff = policy.calculate_backoff(0);
        assert!(backoff.as_secs() >= 0 && backoff.as_secs() <= 2); // With jitter

        // Attempt 1: 2s
        let backoff = policy.calculate_backoff(1);
        assert!(backoff.as_secs() >= 1 && backoff.as_secs() <= 3);

        // Attempt 2: 4s
        let backoff = policy.calculate_backoff(2);
        assert!(backoff.as_secs() >= 3 && backoff.as_secs() <= 5);

        // Attempt 10: should cap at max_backoff_secs (60s)
        let backoff = policy.calculate_backoff(10);
        assert!(backoff.as_secs() <= 72); // 60s + 20% jitter
    }

    #[test]
    fn test_max_attempts() {
        let policy = RestartPolicy::default();

        // Should allow restarts up to max_attempts
        assert!(policy.check_restart_allowed(0, None).is_ok());
        assert!(policy.check_restart_allowed(1, None).is_ok());
        assert!(policy.check_restart_allowed(2, None).is_ok());

        // Should reject after max_attempts
        assert!(policy.check_restart_allowed(3, None).is_err());
    }

    #[test]
    fn test_circuit_breaker() {
        let policy = RestartPolicy {
            circuit_breaker_threshold: 3,
            circuit_breaker_window_secs: 60,
            ..Default::default()
        };

        let now = SystemTime::now();
        let failures = vec![now, now, now]; // 3 failures

        // Should trip circuit breaker
        assert!(policy.check_circuit_breaker(&failures).is_err());

        // Should allow with fewer failures
        let failures = vec![now, now]; // 2 failures
        assert!(policy.check_circuit_breaker(&failures).is_ok());
    }

    #[test]
    fn test_should_reset_count() {
        let policy = RestartPolicy::default();

        // No last restart - should not reset
        assert!(!policy.should_reset_count(None, 300));

        // Recent restart - should not reset
        let recent = SystemTime::now();
        assert!(!policy.should_reset_count(Some(recent), 300));

        // Old restart - should reset
        let old = SystemTime::now().checked_sub(Duration::from_secs(400)).unwrap();
        assert!(policy.should_reset_count(Some(old), 300));
    }
}
