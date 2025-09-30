//! Supervision, backoff, and circuit breaker for engine lifecycle.
//!
//! Spec: OC-POOL-3010, OC-POOL-3011, ORCH-3038, ORCH-3040

use std::time::{Duration, Instant};
use rand::Rng;

/// Crash reason classification
#[derive(Debug, Clone, PartialEq)]
pub enum CrashReason {
    ProcessExit(i32),
    HealthCheckFailure,
    CudaError,
    OutOfMemory,
    Signal(String),
}

/// Crash event
#[derive(Debug, Clone)]
pub struct CrashEvent {
    pub pool_id: String,
    pub reason: CrashReason,
    pub timestamp: Instant,
    pub uptime_seconds: u64,
}

/// Exponential backoff policy with jitter
#[derive(Debug, Clone)]
pub struct BackoffPolicy {
    pub initial_ms: u64,
    pub max_ms: u64,
    pub jitter_factor: f64,
    current_failures: u32,
    last_restart: Option<Instant>,
}

impl BackoffPolicy {
    pub fn new(initial_ms: u64, max_ms: u64) -> Self {
        Self {
            initial_ms,
            max_ms,
            jitter_factor: 0.1,
            current_failures: 0,
            last_restart: None,
        }
    }

    /// Calculate next delay with exponential backoff and jitter
    pub fn next_delay(&mut self) -> Duration {
        self.current_failures += 1;
        
        // Exponential: 2^n * initial_ms
        let base_delay = if self.current_failures == 1 {
            self.initial_ms
        } else {
            let exp = (self.current_failures - 1).min(10); // Cap exponent
            (2u64.pow(exp) * self.initial_ms).min(self.max_ms)
        };
        
        // Add jitter: -10% to +10%
        let mut rng = rand::thread_rng();
        let jitter = rng.gen_range(-self.jitter_factor..=self.jitter_factor);
        let jittered = (base_delay as f64 * (1.0 + jitter)) as u64;
        
        self.last_restart = Some(Instant::now());
        Duration::from_millis(jittered.min(self.max_ms))
    }

    /// Reset backoff after stable run
    pub fn reset(&mut self) {
        self.current_failures = 0;
        self.last_restart = None;
    }

    /// Check if stable run period has elapsed
    pub fn is_stable(&self, stable_period_secs: u64) -> bool {
        if let Some(last) = self.last_restart {
            last.elapsed().as_secs() >= stable_period_secs
        } else {
            false
        }
    }

    pub fn failure_count(&self) -> u32 {
        self.current_failures
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker for restart storms
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    state: CircuitState,
    failure_threshold: u32,
    consecutive_failures: u32,
    timeout_secs: u64,
    opened_at: Option<Instant>,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, timeout_secs: u64) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_threshold,
            consecutive_failures: 0,
            timeout_secs,
            opened_at: None,
        }
    }

    /// Record a failure
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        
        if self.state == CircuitState::Closed 
            && self.consecutive_failures >= self.failure_threshold {
            self.open();
        } else if self.state == CircuitState::HalfOpen {
            self.open();
        }
    }

    /// Record a success
    pub fn record_success(&mut self) {
        if self.state == CircuitState::HalfOpen {
            self.close();
        }
        self.consecutive_failures = 0;
    }

    /// Open the circuit
    fn open(&mut self) {
        self.state = CircuitState::Open;
        self.opened_at = Some(Instant::now());
        
        tracing::warn!(
            consecutive_failures = self.consecutive_failures,
            "circuit breaker opened"
        );
    }

    /// Close the circuit
    fn close(&mut self) {
        self.state = CircuitState::Closed;
        self.consecutive_failures = 0;
        self.opened_at = None;
        
        tracing::info!("circuit breaker closed");
    }

    /// Check if timeout has elapsed and transition to half-open
    pub fn check_timeout(&mut self) {
        if self.state == CircuitState::Open {
            if let Some(opened) = self.opened_at {
                if opened.elapsed().as_secs() >= self.timeout_secs {
                    self.state = CircuitState::HalfOpen;
                    tracing::info!("circuit breaker transitioned to half-open");
                }
            }
        }
    }

    /// Check if restart is allowed
    pub fn allows_restart(&self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true,
            CircuitState::Open => false,
        }
    }

    pub fn state(&self) -> &CircuitState {
        &self.state
    }

    pub fn reset(&mut self) {
        self.close();
    }
}

/// Restart rate limiter with sliding window
#[derive(Debug, Clone)]
pub struct RestartRateLimiter {
    max_restarts: u32,
    window_secs: u64,
    restart_times: Vec<Instant>,
}

impl RestartRateLimiter {
    pub fn new(max_restarts: u32, window_secs: u64) -> Self {
        Self {
            max_restarts,
            window_secs,
            restart_times: Vec::new(),
        }
    }

    /// Record a restart
    pub fn record_restart(&mut self) {
        let now = Instant::now();
        self.restart_times.push(now);
        self.cleanup_old_restarts();
    }

    /// Remove restarts outside the window
    fn cleanup_old_restarts(&mut self) {
        let cutoff = Instant::now() - Duration::from_secs(self.window_secs);
        self.restart_times.retain(|&t| t > cutoff);
    }

    /// Check if rate limit is exceeded
    pub fn is_rate_exceeded(&mut self) -> bool {
        self.cleanup_old_restarts();
        self.restart_times.len() as u32 >= self.max_restarts
    }

    /// Get current restart count in window
    pub fn restart_count(&mut self) -> u32 {
        self.cleanup_old_restarts();
        self.restart_times.len() as u32
    }

    /// Detect restart storm
    pub fn is_storm(&mut self) -> bool {
        self.is_rate_exceeded()
    }

    pub fn reset(&mut self) {
        self.restart_times.clear();
    }
}
