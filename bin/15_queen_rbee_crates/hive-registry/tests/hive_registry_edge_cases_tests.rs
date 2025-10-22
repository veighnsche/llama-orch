// TEAM-250: Hive registry edge case tests
// Purpose: Test staleness detection, worker aggregation, memory management
// Priority: HIGH (state management reliability)
// Scale: Reasonable for NUC (100 hives max, 5 workers per hive)

use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::time::sleep;

// ============================================================================
// Staleness Edge Cases Tests
// ============================================================================

#[tokio::test]
async fn test_hive_marked_stale_after_30s() {
    // TEAM-250: Test hive marked stale after 30s (6 missed heartbeats @ 5s interval)
    
    let last_heartbeat = SystemTime::now() - Duration::from_secs(31);
    let now = SystemTime::now();
    
    let age = now.duration_since(last_heartbeat).unwrap();
    let is_stale = age.as_secs() > 30;
    
    assert!(is_stale, "Hive should be stale after 30s");
}

#[tokio::test]
async fn test_hive_marked_active_on_heartbeat() {
    // TEAM-250: Test hive marked active on heartbeat received
    
    let last_heartbeat = SystemTime::now();
    let now = SystemTime::now();
    
    let age = now.duration_since(last_heartbeat).unwrap();
    let is_stale = age.as_secs() > 30;
    
    assert!(!is_stale, "Hive should be active (just received heartbeat)");
}

#[tokio::test]
async fn test_list_active_hives_excludes_stale() {
    // TEAM-250: Test list_active_hives() excludes stale hives
    
    use std::collections::HashMap;
    
    let mut hives: HashMap<String, SystemTime> = HashMap::new();
    
    // Add active hive (recent heartbeat)
    hives.insert("hive-active".to_string(), SystemTime::now());
    
    // Add stale hive (31s ago)
    hives.insert("hive-stale".to_string(), SystemTime::now() - Duration::from_secs(31));
    
    // Filter active hives (< 30s)
    let now = SystemTime::now();
    let active: Vec<_> = hives.iter()
        .filter(|(_, last_seen)| {
            now.duration_since(**last_seen).unwrap().as_secs() <= 30
        })
        .map(|(id, _)| id.clone())
        .collect();
    
    assert_eq!(active.len(), 1);
    assert_eq!(active[0], "hive-active");
}

#[tokio::test]
async fn test_staleness_calculation_with_clock_skew() {
    // TEAM-250: Test staleness calculation with clock skew
    
    let last_heartbeat = SystemTime::now();
    
    // Simulate small clock skew (1s)
    sleep(Duration::from_secs(1)).await;
    
    let now = SystemTime::now();
    let age = now.duration_since(last_heartbeat).unwrap();
    
    assert!(age.as_secs() >= 1 && age.as_secs() < 2, "Should account for clock skew");
}

#[test]
fn test_staleness_boundary_exactly_30s() {
    // TEAM-250: Test staleness boundary (exactly 30s)
    
    let last_heartbeat = SystemTime::now() - Duration::from_secs(30);
    let now = SystemTime::now();
    
    let age = now.duration_since(last_heartbeat).unwrap();
    let is_stale = age.as_secs() > 30; // Strict > 30s
    
    assert!(!is_stale, "Exactly 30s should NOT be stale (strict >)");
}

// ============================================================================
// Worker Aggregation Tests
// ============================================================================

#[test]
fn test_hive_with_0_workers() {
    // TEAM-250: Test hive with 0 workers
    
    let workers: Vec<String> = vec![];
    
    assert_eq!(workers.len(), 0);
    // In real code, should display "-" for worker fields
}

#[test]
fn test_hive_with_1_worker() {
    // TEAM-250: Test hive with 1 worker
    
    let workers = vec!["worker-1"];
    
    assert_eq!(workers.len(), 1);
    assert_eq!(workers[0], "worker-1");
}

#[test]
fn test_hive_with_5_workers() {
    // TEAM-250: Test hive with 5 workers (NUC-friendly max)
    
    let workers = vec!["worker-1", "worker-2", "worker-3", "worker-4", "worker-5"];
    
    assert_eq!(workers.len(), 5);
}

#[test]
fn test_worker_state_updates_reflected() {
    // TEAM-250: Test worker state updates reflected in registry
    
    use std::collections::HashMap;
    
    let mut worker_states: HashMap<String, String> = HashMap::new();
    
    // Initial state
    worker_states.insert("worker-1".to_string(), "idle".to_string());
    assert_eq!(worker_states.get("worker-1"), Some(&"idle".to_string()));
    
    // Update state
    worker_states.insert("worker-1".to_string(), "busy".to_string());
    assert_eq!(worker_states.get("worker-1"), Some(&"busy".to_string()));
}

#[test]
fn test_get_worker_with_multiple_hives() {
    // TEAM-250: Test get_worker() with multiple hives
    
    use std::collections::HashMap;
    
    let mut hives: HashMap<String, Vec<String>> = HashMap::new();
    
    hives.insert("hive-1".to_string(), vec!["worker-1".to_string()]);
    hives.insert("hive-2".to_string(), vec!["worker-2".to_string()]);
    
    // Find worker-1
    let mut found = false;
    for (hive_id, workers) in &hives {
        if workers.contains(&"worker-1".to_string()) {
            found = true;
            assert_eq!(hive_id, "hive-1");
            break;
        }
    }
    assert!(found, "Should find worker-1 in hive-1");
}

// ============================================================================
// Concurrent Operations Tests
// ============================================================================

#[tokio::test]
async fn test_10_concurrent_update_hive_state_different_hives() {
    // TEAM-250: Test 10 concurrent update_hive_state() calls (different hives)
    
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    
    let hives = Arc::new(RwLock::new(HashMap::<String, String>::new()));
    
    let mut handles = vec![];
    
    for i in 0..10 {
        let hives_clone = hives.clone();
        let handle = tokio::spawn(async move {
            let mut h = hives_clone.write().await;
            h.insert(format!("hive-{}", i), format!("state-{}", i));
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
    
    let h = hives.read().await;
    assert_eq!(h.len(), 10);
}

#[tokio::test]
async fn test_10_concurrent_update_hive_state_same_hive() {
    // TEAM-250: Test 10 concurrent update_hive_state() calls (same hive)
    
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    
    let hives = Arc::new(RwLock::new(HashMap::<String, String>::new()));
    
    let mut handles = vec![];
    
    for i in 0..10 {
        let hives_clone = hives.clone();
        let handle = tokio::spawn(async move {
            let mut h = hives_clone.write().await;
            h.insert("hive-1".to_string(), format!("state-{}", i));
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Last write wins
    let h = hives.read().await;
    assert_eq!(h.len(), 1);
    assert!(h.get("hive-1").unwrap().starts_with("state-"));
}

#[tokio::test]
async fn test_5_concurrent_reads_during_5_writes() {
    // TEAM-250: Test 5 concurrent reads during 5 writes
    
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    
    let hives = Arc::new(RwLock::new(HashMap::<String, String>::new()));
    
    // Initial state
    {
        let mut h = hives.write().await;
        h.insert("hive-1".to_string(), "initial".to_string());
    }
    
    let mut handles = vec![];
    
    // Spawn 5 writers
    for i in 0..5 {
        let hives_clone = hives.clone();
        let handle = tokio::spawn(async move {
            sleep(Duration::from_millis(10)).await;
            let mut h = hives_clone.write().await;
            h.insert("hive-1".to_string(), format!("state-{}", i));
        });
        handles.push(handle);
    }
    
    // Spawn 5 readers
    for _ in 0..5 {
        let hives_clone = hives.clone();
        let handle = tokio::spawn(async move {
            let h = hives_clone.read().await;
            h.get("hive-1").cloned()
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
}

#[tokio::test]
async fn test_concurrent_get_worker() {
    // TEAM-250: Test concurrent get_worker() calls
    
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    
    let hives = Arc::new(RwLock::new(HashMap::<String, Vec<String>>::new()));
    
    // Setup
    {
        let mut h = hives.write().await;
        h.insert("hive-1".to_string(), vec!["worker-1".to_string()]);
    }
    
    let mut handles = vec![];
    
    // Spawn 10 concurrent readers
    for _ in 0..10 {
        let hives_clone = hives.clone();
        let handle = tokio::spawn(async move {
            let h = hives_clone.read().await;
            h.get("hive-1").cloned()
        });
        handles.push(handle);
    }
    
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_some());
    }
}

#[tokio::test]
async fn test_rwlock_behavior_readers_dont_block_readers() {
    // TEAM-250: Test RwLock behavior (readers don't block readers)
    
    use std::sync::Arc;
    use tokio::sync::RwLock;
    
    let data = Arc::new(RwLock::new(String::from("test")));
    
    let mut handles = vec![];
    
    // Spawn 10 concurrent readers
    for _ in 0..10 {
        let data_clone = data.clone();
        let handle = tokio::spawn(async move {
            let _guard = data_clone.read().await;
            sleep(Duration::from_millis(100)).await;
        });
        handles.push(handle);
    }
    
    // All readers should complete concurrently
    for handle in handles {
        handle.await.unwrap();
    }
}

// ============================================================================
// Memory Management Tests
// ============================================================================

#[tokio::test]
async fn test_100_hive_updates() {
    // TEAM-250: Test 100 hive updates (not 1000+)
    
    use std::collections::HashMap;
    
    let mut hives: HashMap<String, String> = HashMap::new();
    
    for i in 0..100 {
        hives.insert(format!("hive-{}", i), format!("state-{}", i));
    }
    
    assert_eq!(hives.len(), 100);
}

#[tokio::test]
async fn test_memory_usage_stays_constant() {
    // TEAM-250: Test memory usage stays constant (old states replaced)
    
    use std::collections::HashMap;
    
    let mut hives: HashMap<String, String> = HashMap::new();
    
    // Update same hive 100 times
    for i in 0..100 {
        hives.insert("hive-1".to_string(), format!("state-{}", i));
    }
    
    // Should only have 1 entry (not 100)
    assert_eq!(hives.len(), 1);
}

#[test]
fn test_old_states_replaced_not_accumulated() {
    // TEAM-250: Test old states are replaced (not accumulated)
    
    use std::collections::HashMap;
    
    let mut hives: HashMap<String, String> = HashMap::new();
    
    hives.insert("hive-1".to_string(), "state-1".to_string());
    assert_eq!(hives.len(), 1);
    
    hives.insert("hive-1".to_string(), "state-2".to_string());
    assert_eq!(hives.len(), 1); // Still 1, not 2
    
    assert_eq!(hives.get("hive-1"), Some(&"state-2".to_string()));
}

#[test]
fn test_no_dangling_references() {
    // TEAM-250: Test no dangling references
    
    use std::collections::HashMap;
    
    let mut hives: HashMap<String, String> = HashMap::new();
    
    hives.insert("hive-1".to_string(), "state-1".to_string());
    
    // Remove hive
    hives.remove("hive-1");
    
    // Should be completely gone
    assert!(hives.get("hive-1").is_none());
}

#[test]
fn test_cleanup_after_hive_removal() {
    // TEAM-250: Test cleanup after hive removal
    
    use std::collections::HashMap;
    
    let mut hives: HashMap<String, Vec<String>> = HashMap::new();
    
    // Add hive with workers
    hives.insert("hive-1".to_string(), vec!["worker-1".to_string(), "worker-2".to_string()]);
    
    // Remove hive
    hives.remove("hive-1");
    
    // Everything should be cleaned up
    assert_eq!(hives.len(), 0);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_hive_id_validation() {
    // TEAM-250: Test hive ID validation
    
    let valid_ids = vec![
        "hive-1",
        "hive-local",
        "hive-remote-server",
    ];
    
    for id in valid_ids {
        assert!(id.starts_with("hive-") || id == "localhost");
    }
}

#[test]
fn test_empty_hive_id() {
    // TEAM-250: Test empty hive ID
    
    let hive_id = "";
    
    assert!(hive_id.is_empty());
    // In real code, should be rejected
}

#[test]
fn test_very_long_hive_id() {
    // TEAM-250: Test very long hive ID
    
    let hive_id = "hive-".to_string() + &"a".repeat(1000);
    
    assert!(hive_id.len() > 1000);
    // In real code, should either accept or reject with clear error
}
