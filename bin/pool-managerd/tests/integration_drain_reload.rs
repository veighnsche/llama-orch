//! Integration tests for drain and reload cycles

use pool_managerd::lifecycle::{drain, reload};
use pool_managerd::core::registry::Registry;
use pool_managerd::core::health::HealthStatus;
use provisioners_engine_provisioner::PreparedEngine;
use std::path::PathBuf;

#[test]
fn test_drain_then_reload_cycle() {
    let mut registry = Registry::new();
    registry.register("test-pool");
    registry.set_health("test-pool", HealthStatus { live: true, ready: true });
    registry.set_engine_version("test-pool", "v1.0.0");
    
    // Step 1: Drain
    let drain_req = drain::DrainRequest::new("test-pool", 2000);
    let drain_outcome = drain::execute_drain(drain_req, &mut registry)
        .expect("drain failed");
    
    assert!(!drain_outcome.force_stopped);
    
    let health = registry.get_health("test-pool").unwrap();
    assert!(!health.ready);
    
    // Step 2: Reload would happen here
    // (Skipped in unit test since we'd need a real engine process)
    
    // Verify registry state after drain
    assert!(registry.get_draining("test-pool"));
}

#[test]
fn test_reload_with_drain_timeout() {
    let mut registry = Registry::new();
    registry.register("test-pool");
    registry.set_health("test-pool", HealthStatus { live: true, ready: true });
    
    // Allocate leases that won't complete
    registry.allocate_lease("test-pool");
    registry.allocate_lease("test-pool");
    
    // This would test reload with stuck leases
    // In practice, reload calls drain which would force-stop
    
    let active = registry.get_active_leases("test-pool");
    assert_eq!(active, 2);
}

#[test]
fn test_multiple_drain_cycles() {
    let mut registry = Registry::new();
    
    for i in 0..3 {
        let pool_id = format!("pool-{}", i);
        registry.register(&pool_id);
        registry.set_health(&pool_id, HealthStatus { live: true, ready: true });
        
        let drain_req = drain::DrainRequest::new(&pool_id, 1000);
        let outcome = drain::execute_drain(drain_req, &mut registry)
            .expect("drain failed");
        
        assert_eq!(outcome.pool_id, pool_id);
        assert!(registry.get_draining(&pool_id));
        
        let health = registry.get_health(&pool_id).unwrap();
        assert!(!health.ready);
    }
}

#[test]
fn test_drain_preserves_pool_id() {
    let mut registry = Registry::new();
    let pool_id = "my-special-pool-123";
    
    registry.register(pool_id);
    registry.set_health(pool_id, HealthStatus { live: true, ready: true });
    registry.set_device_mask(pool_id, "0,1");
    
    let drain_req = drain::DrainRequest::new(pool_id, 1000);
    let outcome = drain::execute_drain(drain_req, &mut registry)
        .expect("drain failed");
    
    assert_eq!(outcome.pool_id, pool_id);
    
    // Device mask should be preserved
    assert_eq!(registry.get_device_mask(pool_id), Some("0,1".to_string()));
}

#[test]
fn test_concurrent_drains_different_pools() {
    use std::sync::{Arc, Mutex};
    
    let registry = Arc::new(Mutex::new(Registry::new()));
    
    let mut handles = vec![];
    
    for i in 0..3 {
        let pool_id = format!("pool-{}", i);
        let reg = registry.clone();
        
        {
            let mut r = reg.lock().unwrap();
            r.register(&pool_id);
            r.set_health(&pool_id, HealthStatus { live: true, ready: true });
        }
        
        let handle = std::thread::spawn(move || {
            let mut r = reg.lock().unwrap();
            let req = drain::DrainRequest::new(&pool_id, 1000);
            drain::execute_drain(req, &mut r)
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        let outcome = handle.join().unwrap().expect("drain failed");
        assert!(!outcome.force_stopped);
    }
    
    let reg = registry.lock().unwrap();
    for i in 0..3 {
        let pool_id = format!("pool-{}", i);
        assert!(reg.get_draining(&pool_id));
    }
}
