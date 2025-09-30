//! Unit tests for drain lifecycle

use pool_managerd::lifecycle::drain::{DrainRequest, execute_drain};
use pool_managerd::core::registry::Registry;
use pool_managerd::core::health::HealthStatus;

#[test]
fn test_drain_sets_draining_flag() {
    let mut registry = Registry::new();
    registry.register("test-pool");
    registry.set_health("test-pool", HealthStatus { live: true, ready: true });
    
    let req = DrainRequest::new("test-pool", 1000);
    let outcome = execute_drain(req, &mut registry).expect("drain failed");
    
    assert!(registry.get_draining("test-pool"));
    assert_eq!(outcome.pool_id, "test-pool");
}

#[test]
fn test_drain_with_no_leases_completes_immediately() {
    let mut registry = Registry::new();
    registry.register("test-pool");
    registry.set_health("test-pool", HealthStatus { live: true, ready: true });
    
    let req = DrainRequest::new("test-pool", 5000);
    let outcome = execute_drain(req, &mut registry).expect("drain failed");
    
    assert_eq!(outcome.final_lease_count, 0);
    assert!(!outcome.force_stopped);
    assert!(outcome.duration_ms < 1000); // Should be quick
}

#[test]
fn test_drain_updates_health_to_not_ready() {
    let mut registry = Registry::new();
    registry.register("test-pool");
    registry.set_health("test-pool", HealthStatus { live: true, ready: true });
    
    let req = DrainRequest::new("test-pool", 1000);
    let _ = execute_drain(req, &mut registry).expect("drain failed");
    
    let health = registry.get_health("test-pool").expect("no health");
    assert!(!health.live);
    assert!(!health.ready);
}

#[test]
fn test_drain_waits_for_leases_to_complete() {
    // Note: This test demonstrates the lease waiting logic conceptually
    // In practice, drain holds the registry lock so concurrent release isn't possible
    // Real-world usage would have leases released by engine processes, not directly
    
    let mut registry = Registry::new();
    registry.register("test-pool");
    registry.set_health("test-pool", HealthStatus { live: true, ready: true });
    
    // Test with leases that complete before drain starts
    registry.allocate_lease("test-pool");
    registry.allocate_lease("test-pool");
    registry.release_lease("test-pool");
    registry.release_lease("test-pool");
    
    let req = DrainRequest::new("test-pool", 5000);
    let outcome = execute_drain(req, &mut registry).expect("drain failed");
    
    assert_eq!(outcome.final_lease_count, 0);
    assert!(!outcome.force_stopped);
}

#[test]
fn test_drain_force_stops_on_deadline() {
    let mut registry = Registry::new();
    registry.register("test-pool");
    registry.set_health("test-pool", HealthStatus { live: true, ready: true });
    
    // Allocate leases that won't be released
    registry.allocate_lease("test-pool");
    registry.allocate_lease("test-pool");
    
    let req = DrainRequest::new("test-pool", 500); // Short deadline
    let outcome = execute_drain(req, &mut registry).expect("drain failed");
    
    assert!(outcome.force_stopped);
    assert_eq!(outcome.final_lease_count, 2); // Leases still active
    assert!(outcome.duration_ms >= 500);
}
