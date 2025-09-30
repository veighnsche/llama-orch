//! Unit tests for supervision, backoff, and circuit breaker

use pool_managerd::lifecycle::supervision::{
    BackoffPolicy, CircuitBreaker, CircuitState, RestartRateLimiter, CrashReason, CrashEvent,
};
use std::time::Duration;

#[test]
fn test_backoff_first_delay() {
    let mut policy = BackoffPolicy::new(1000, 60000);
    let delay = policy.next_delay();
    
    // First delay should be ~1000ms with jitter
    let delay_ms = delay.as_millis() as u64;
    assert!(delay_ms >= 900 && delay_ms <= 1100, "delay {} not in range", delay_ms);
    assert_eq!(policy.failure_count(), 1);
}

#[test]
fn test_backoff_exponential_progression() {
    let mut policy = BackoffPolicy::new(1000, 60000);
    
    let delays: Vec<u64> = (0..5).map(|_| {
        policy.next_delay().as_millis() as u64
    }).collect();
    
    // Should roughly follow: 1000, 2000, 4000, 8000, 16000
    // Allow jitter tolerance of 15%
    assert!(delays[0] >= 900 && delays[0] <= 1100);
    assert!(delays[1] >= 1800 && delays[1] <= 2200);
    assert!(delays[2] >= 3600 && delays[2] <= 4400);
    assert!(delays[3] >= 7200 && delays[3] <= 8800);
    assert!(delays[4] >= 14400 && delays[4] <= 17600);
}

#[test]
fn test_backoff_max_cap() {
    let mut policy = BackoffPolicy::new(1000, 5000);
    
    // Simulate many failures
    for _ in 0..20 {
        policy.next_delay();
    }
    
    let delay = policy.next_delay();
    assert!(delay.as_millis() <= 5000);
}

#[test]
fn test_backoff_reset() {
    let mut policy = BackoffPolicy::new(1000, 60000);
    
    policy.next_delay();
    policy.next_delay();
    assert_eq!(policy.failure_count(), 2);
    
    policy.reset();
    assert_eq!(policy.failure_count(), 0);
    
    let delay = policy.next_delay();
    assert!(delay.as_millis() >= 900 && delay.as_millis() <= 1100);
}

#[test]
fn test_circuit_breaker_opens_after_threshold() {
    let mut circuit = CircuitBreaker::new(5, 300);
    
    assert_eq!(circuit.state(), &CircuitState::Closed);
    assert!(circuit.allows_restart());
    
    for _ in 0..4 {
        circuit.record_failure();
        assert_eq!(circuit.state(), &CircuitState::Closed);
    }
    
    circuit.record_failure();
    assert_eq!(circuit.state(), &CircuitState::Open);
    assert!(!circuit.allows_restart());
}

#[test]
fn test_circuit_breaker_half_open_to_closed() {
    let mut circuit = CircuitBreaker::new(5, 0); // Zero timeout for immediate transition
    
    for _ in 0..5 {
        circuit.record_failure();
    }
    assert_eq!(circuit.state(), &CircuitState::Open);
    
    circuit.check_timeout();
    assert_eq!(circuit.state(), &CircuitState::HalfOpen);
    
    circuit.record_success();
    assert_eq!(circuit.state(), &CircuitState::Closed);
}

#[test]
fn test_circuit_breaker_half_open_reopens_on_failure() {
    let mut circuit = CircuitBreaker::new(5, 0);
    
    for _ in 0..5 {
        circuit.record_failure();
    }
    circuit.check_timeout();
    assert_eq!(circuit.state(), &CircuitState::HalfOpen);
    
    circuit.record_failure();
    assert_eq!(circuit.state(), &CircuitState::Open);
}

#[test]
fn test_circuit_breaker_manual_reset() {
    let mut circuit = CircuitBreaker::new(5, 300);
    
    for _ in 0..5 {
        circuit.record_failure();
    }
    assert_eq!(circuit.state(), &CircuitState::Open);
    
    circuit.reset();
    assert_eq!(circuit.state(), &CircuitState::Closed);
    assert!(circuit.allows_restart());
}

#[test]
fn test_restart_rate_limiter_sliding_window() {
    let mut limiter = RestartRateLimiter::new(10, 60);
    
    for _ in 0..5 {
        limiter.record_restart();
    }
    
    assert_eq!(limiter.restart_count(), 5);
    assert!(!limiter.is_rate_exceeded());
    
    for _ in 0..5 {
        limiter.record_restart();
    }
    
    assert_eq!(limiter.restart_count(), 10);
    assert!(limiter.is_rate_exceeded());
}

#[test]
fn test_restart_rate_limiter_cleanup() {
    let mut limiter = RestartRateLimiter::new(10, 1); // 1 second window
    
    limiter.record_restart();
    assert_eq!(limiter.restart_count(), 1);
    
    std::thread::sleep(Duration::from_millis(1100));
    
    assert_eq!(limiter.restart_count(), 0);
}

#[test]
fn test_restart_rate_limiter_storm_detection() {
    let mut limiter = RestartRateLimiter::new(5, 60);
    
    for _ in 0..5 {
        limiter.record_restart();
    }
    
    assert!(limiter.is_storm());
}

#[test]
fn test_restart_rate_limiter_reset() {
    let mut limiter = RestartRateLimiter::new(10, 60);
    
    for _ in 0..5 {
        limiter.record_restart();
    }
    
    assert_eq!(limiter.restart_count(), 5);
    
    limiter.reset();
    assert_eq!(limiter.restart_count(), 0);
}

#[test]
fn test_crash_reason_classification() {
    let reasons = vec![
        CrashReason::ProcessExit(1),
        CrashReason::HealthCheckFailure,
        CrashReason::CudaError,
        CrashReason::OutOfMemory,
        CrashReason::Signal("SIGSEGV".to_string()),
    ];
    
    for reason in reasons {
        assert!(format!("{:?}", reason).len() > 0);
    }
}
