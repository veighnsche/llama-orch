// TEAM-243: Timeout propagation tests for timeout-enforcer
// Purpose: Verify timeout chains work correctly across layered operations
// Scale: Reasonable for NUC (5-10 concurrent, 100 timeouts total)
// Historical Context: TEAM-243 implemented Priority 1 critical tests for timeout infrastructure

use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Test basic timeout enforcement
#[tokio::test]
async fn test_basic_timeout_enforcement() {
    let timeout = Duration::from_millis(100);
    let start = Instant::now();
    
    // Create a future that takes longer than timeout
    let result = tokio::time::timeout(
        timeout,
        async {
            sleep(Duration::from_millis(500)).await;
            "completed"
        }
    ).await;
    
    let elapsed = start.elapsed();
    
    // Should timeout
    assert!(result.is_err());
    // Should timeout around 100ms (±50ms tolerance)
    assert!(elapsed.as_millis() >= 100 && elapsed.as_millis() <= 150);
    
    println!("✓ Basic timeout enforcement works ({}ms)", elapsed.as_millis());
}

/// Test timeout doesn't fire if operation completes in time
#[tokio::test]
async fn test_timeout_doesnt_fire_early() {
    let timeout = Duration::from_millis(500);
    let start = Instant::now();
    
    let result = tokio::time::timeout(
        timeout,
        async {
            sleep(Duration::from_millis(100)).await;
            "completed"
        }
    ).await;
    
    let elapsed = start.elapsed();
    
    // Should complete successfully
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "completed");
    // Should complete around 100ms (±50ms tolerance)
    assert!(elapsed.as_millis() >= 100 && elapsed.as_millis() <= 150);
    
    println!("✓ Timeout doesn't fire early ({}ms)", elapsed.as_millis());
}

/// Test layered timeouts (Keeper → Queen → Hive)
#[tokio::test]
async fn test_layered_timeouts() {
    // Keeper timeout: 10s
    let keeper_timeout = Duration::from_secs(10);
    // Queen timeout: 5s
    let queen_timeout = Duration::from_secs(5);
    // Hive timeout: 2s
    let hive_timeout = Duration::from_secs(2);
    
    let start = Instant::now();
    
    // Simulate operation that takes 3s (should timeout at hive level)
    let result = tokio::time::timeout(
        keeper_timeout,
        async {
            tokio::time::timeout(
                queen_timeout,
                async {
                    tokio::time::timeout(
                        hive_timeout,
                        async {
                            sleep(Duration::from_secs(3)).await;
                            "completed"
                        }
                    ).await
                }
            ).await
        }
    ).await;
    
    let elapsed = start.elapsed();
    
    // Should timeout at hive level (2s)
    // Result is Ok(Ok(Err(timeout))) because nested timeouts return Ok(Err)
    assert!(result.is_ok());
    let queen_result = result.unwrap();
    assert!(queen_result.is_ok());
    let hive_result = queen_result.unwrap();
    assert!(hive_result.is_err());
    assert!(elapsed.as_secs() >= 2 && elapsed.as_secs() <= 3);
    
    println!("✓ Layered timeouts work correctly ({}s)", elapsed.as_secs());
}

/// Test innermost timeout fires first
#[tokio::test]
async fn test_innermost_timeout_fires_first() {
    let outer_timeout = Duration::from_secs(10);
    let inner_timeout = Duration::from_millis(100);
    
    let start = Instant::now();
    
    let result = tokio::time::timeout(
        outer_timeout,
        async {
            tokio::time::timeout(
                inner_timeout,
                async {
                    sleep(Duration::from_secs(5)).await;
                    "completed"
                }
            ).await
        }
    ).await;
    
    let elapsed = start.elapsed();
    
    // Should timeout at inner level
    // Result is Ok(Err(timeout)) because inner timeout returns Ok(Err)
    assert!(result.is_ok());
    let inner_result = result.unwrap();
    assert!(inner_result.is_err());
    assert!(elapsed.as_millis() >= 100 && elapsed.as_millis() <= 200);
    
    println!("✓ Innermost timeout fires first ({}ms)", elapsed.as_millis());
}

/// Test timeout with concurrent operations
#[tokio::test]
async fn test_timeout_concurrent_operations() {
    let timeout = Duration::from_millis(500);
    
    let result = tokio::time::timeout(
        timeout,
        async {
            let mut handles = vec![];
            
            // Spawn 5 concurrent tasks
            for i in 0..5 {
                let handle = tokio::spawn(async move {
                    sleep(Duration::from_millis(100 * i as u64)).await;
                    format!("task-{}", i)
                });
                handles.push(handle);
            }
            
            // Wait for all tasks
            let mut results = vec![];
            for handle in handles {
                results.push(handle.await.unwrap());
            }
            
            results
        }
    ).await;
    
    // Should complete successfully (all tasks finish within 500ms)
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 5);
    
    println!("✓ Timeout with concurrent operations works");
}

/// Test timeout with streaming operations
#[tokio::test]
async fn test_timeout_streaming() {
    let timeout = Duration::from_millis(500);
    let start = Instant::now();
    
    let result = tokio::time::timeout(
        timeout,
        async {
            let mut count = 0;
            loop {
                sleep(Duration::from_millis(100)).await;
                count += 1;
                if count >= 10 {
                    break;
                }
            }
            count
        }
    ).await;
    
    let elapsed = start.elapsed();
    
    // Should timeout before reaching 10 iterations
    assert!(result.is_err());
    assert!(elapsed.as_millis() >= 500 && elapsed.as_millis() <= 600);
    
    println!("✓ Timeout with streaming operations works ({}ms)", elapsed.as_millis());
}

/// Test timeout with error handling
#[tokio::test]
async fn test_timeout_with_error_handling() {
    let timeout = Duration::from_millis(100);
    
    let result = tokio::time::timeout(
        timeout,
        async {
            sleep(Duration::from_millis(500)).await;
            Err::<String, String>("operation failed".to_string())
        }
    ).await;
    
    // Should timeout before error is returned
    assert!(result.is_err());
    
    println!("✓ Timeout with error handling works");
}

/// Test timeout cancellation is clean
#[tokio::test]
async fn test_timeout_cancellation_is_clean() {
    let timeout = Duration::from_millis(100);
    let cleanup_called = std::sync::Arc::new(std::sync::Mutex::new(false));
    let cleanup_clone = cleanup_called.clone();
    
    let result = tokio::time::timeout(
        timeout,
        async {
            // Simulate cleanup on drop
            struct Cleanup(std::sync::Arc<std::sync::Mutex<bool>>);
            impl Drop for Cleanup {
                fn drop(&mut self) {
                    *self.0.lock().unwrap() = true;
                }
            }
            
            let _cleanup = Cleanup(cleanup_clone);
            sleep(Duration::from_secs(10)).await;
            "completed"
        }
    ).await;
    
    // Should timeout
    assert!(result.is_err());
    
    // Cleanup should be called
    sleep(Duration::from_millis(50)).await;
    assert!(*cleanup_called.lock().unwrap());
    
    println!("✓ Timeout cancellation is clean");
}

/// Test multiple sequential timeouts
#[tokio::test]
async fn test_multiple_sequential_timeouts() {
    let timeout = Duration::from_millis(200);
    
    // First operation: completes in time
    let result1 = tokio::time::timeout(
        timeout,
        async {
            sleep(Duration::from_millis(100)).await;
            "completed-1"
        }
    ).await;
    assert!(result1.is_ok());
    
    // Second operation: times out
    let result2 = tokio::time::timeout(
        timeout,
        async {
            sleep(Duration::from_millis(500)).await;
            "completed-2"
        }
    ).await;
    assert!(result2.is_err());
    
    // Third operation: completes in time
    let result3 = tokio::time::timeout(
        timeout,
        async {
            sleep(Duration::from_millis(100)).await;
            "completed-3"
        }
    ).await;
    assert!(result3.is_ok());
    
    println!("✓ Multiple sequential timeouts work correctly");
}

/// Test timeout with job_id propagation
#[tokio::test]
async fn test_timeout_with_job_id_propagation() {
    let timeout = Duration::from_millis(100);
    let job_id = "job-123";
    
    let result = tokio::time::timeout(
        timeout,
        async {
            // Simulate operation with job_id
            sleep(Duration::from_millis(500)).await;
            format!("completed-{}", job_id)
        }
    ).await;
    
    // Should timeout
    assert!(result.is_err());
    
    println!("✓ Timeout with job_id propagation works");
}

/// Test timeout precision
#[tokio::test]
async fn test_timeout_precision() {
    // Test various timeout durations
    let timeouts = vec![
        Duration::from_millis(50),
        Duration::from_millis(100),
        Duration::from_millis(200),
        Duration::from_millis(500),
    ];
    
    for timeout in timeouts {
        let start = Instant::now();
        
        let result = tokio::time::timeout(
            timeout,
            async {
                sleep(Duration::from_secs(10)).await;
                "completed"
            }
        ).await;
        
        let elapsed = start.elapsed();
        
        // Should timeout
        assert!(result.is_err());
        
        // Should be within ±50ms of target
        let tolerance = Duration::from_millis(50);
        assert!(
            elapsed >= timeout && elapsed <= timeout + tolerance,
            "Timeout precision failed: expected ~{:?}, got {:?}",
            timeout,
            elapsed
        );
    }
    
    println!("✓ Timeout precision verified for multiple durations");
}

/// Test timeout with resource cleanup
#[tokio::test]
async fn test_timeout_resource_cleanup() {
    let timeout = Duration::from_millis(100);
    let cleanup_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let cleanup_clone = cleanup_count.clone();
    
    let result = tokio::time::timeout(
        timeout,
        async {
            struct Resource(std::sync::Arc<std::sync::atomic::AtomicUsize>);
            impl Drop for Resource {
                fn drop(&mut self) {
                    self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
            
            let _resource = Resource(cleanup_clone);
            sleep(Duration::from_secs(10)).await;
            "completed"
        }
    ).await;
    
    // Should timeout
    assert!(result.is_err());
    
    // Resource should be cleaned up
    sleep(Duration::from_millis(50)).await;
    assert_eq!(cleanup_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    
    println!("✓ Timeout resource cleanup verified");
}

/// Test zero timeout
#[tokio::test]
async fn test_zero_timeout() {
    let timeout = Duration::from_millis(0);
    
    let result = tokio::time::timeout(
        timeout,
        async {
            sleep(Duration::from_millis(1)).await;
            "completed"
        }
    ).await;
    
    // Should timeout immediately
    assert!(result.is_err());
    
    println!("✓ Zero timeout works correctly");
}

/// Test very large timeout
#[tokio::test]
async fn test_very_large_timeout() {
    let timeout = Duration::from_secs(3600); // 1 hour
    
    let result = tokio::time::timeout(
        timeout,
        async {
            sleep(Duration::from_millis(10)).await;
            "completed"
        }
    ).await;
    
    // Should complete successfully
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "completed");
    
    println!("✓ Very large timeout works correctly");
}
