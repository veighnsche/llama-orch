// TEAM-XXX: Telemetry pipeline testing - HiveRegistry worker storage tests
//!
//! Tests for worker telemetry storage and scheduling queries
//!
//! Coverage:
//! - Worker storage (update_workers, get_workers)
//! - Scheduling queries (find_idle, find_by_model, find_by_capacity)
//! - Thread safety (concurrent access)
//! - Stale worker cleanup

use queen_rbee_hive_registry::HiveRegistry;
use rbee_hive_monitor::ProcessStats;
use std::sync::Arc;
use std::thread;

// ============================================================================
// WORKER STORAGE TESTS
// ============================================================================

#[test]
fn test_update_workers_stores_correctly() {
    // GIVEN: Empty registry
    let registry = HiveRegistry::new();

    // WHEN: Store workers for hive
    let workers = vec![
        create_test_worker(1001, "llm", "8080", 0.0, 0, Some("llama-3.2-1b")),
        create_test_worker(1002, "llm", "8081", 85.0, 8192, Some("llama-3.2-3b")),
    ];

    registry.update_workers("localhost", workers.clone());

    // THEN: Workers stored
    let stored = registry.get_workers("localhost");
    assert!(stored.is_some());
    let stored_workers = stored.unwrap();
    assert_eq!(stored_workers.len(), 2);
    assert_eq!(stored_workers[0].pid, 1001);
    assert_eq!(stored_workers[1].pid, 1002);
}

#[test]
fn test_get_workers_returns_stored() {
    // GIVEN: Registry with workers
    let registry = HiveRegistry::new();
    let workers = vec![
        create_test_worker(2001, "llm", "8080", 0.0, 0, None),
    ];
    registry.update_workers("hive-1", workers);

    // WHEN: Get workers for existing hive
    let result = registry.get_workers("hive-1");

    // THEN: Returns workers
    assert!(result.is_some());
    assert_eq!(result.unwrap().len(), 1);

    // WHEN: Get workers for non-existent hive
    let result = registry.get_workers("nonexistent");

    // THEN: Returns None
    assert!(result.is_none());
}

#[test]
fn test_get_all_workers_flattens() {
    // GIVEN: Multiple hives with workers
    let registry = HiveRegistry::new();

    registry.update_workers(
        "hive-1",
        vec![
            create_test_worker(3001, "llm", "8080", 0.0, 0, None),
            create_test_worker(3002, "llm", "8081", 0.0, 0, None),
        ],
    );

    registry.update_workers(
        "hive-2",
        vec![
            create_test_worker(4001, "vllm", "9000", 0.0, 0, None),
        ],
    );

    // WHEN: Get all workers
    let all = registry.get_all_workers();

    // THEN: Flattens across hives
    assert_eq!(all.len(), 3);
    let pids: Vec<u32> = all.iter().map(|w| w.pid).collect();
    assert!(pids.contains(&3001));
    assert!(pids.contains(&3002));
    assert!(pids.contains(&4001));
}

#[test]
fn test_update_workers_replaces_existing() {
    // GIVEN: Registry with workers
    let registry = HiveRegistry::new();
    registry.update_workers(
        "localhost",
        vec![create_test_worker(5001, "llm", "8080", 0.0, 0, None)],
    );

    // WHEN: Update with new workers
    registry.update_workers(
        "localhost",
        vec![
            create_test_worker(5002, "llm", "8081", 0.0, 0, None),
            create_test_worker(5003, "llm", "8082", 0.0, 0, None),
        ],
    );

    // THEN: Old workers replaced
    let workers = registry.get_workers("localhost").unwrap();
    assert_eq!(workers.len(), 2);
    assert!(!workers.iter().any(|w| w.pid == 5001));
    assert!(workers.iter().any(|w| w.pid == 5002));
    assert!(workers.iter().any(|w| w.pid == 5003));
}

// ============================================================================
// SCHEDULING QUERY TESTS
// ============================================================================

#[test]
fn test_find_idle_workers_filters() {
    // GIVEN: Mix of idle and busy workers
    let registry = HiveRegistry::new();
    registry.update_workers(
        "localhost",
        vec![
            create_test_worker(6001, "llm", "8080", 0.0, 0, None), // Idle
            create_test_worker(6002, "llm", "8081", 85.0, 8192, None), // Busy
            create_test_worker(6003, "llm", "8082", 0.0, 4096, None), // Idle
        ],
    );

    // WHEN: Find idle workers
    let idle = registry.find_idle_workers();

    // THEN: Only idle workers returned
    assert_eq!(idle.len(), 2);
    assert!(idle.iter().all(|w| w.gpu_util_pct == 0.0));
    let pids: Vec<u32> = idle.iter().map(|w| w.pid).collect();
    assert!(pids.contains(&6001));
    assert!(pids.contains(&6003));
}

#[test]
fn test_find_workers_with_model_matches() {
    // GIVEN: Workers with different models
    let registry = HiveRegistry::new();
    registry.update_workers(
        "localhost",
        vec![
            create_test_worker(7001, "llm", "8080", 0.0, 0, Some("llama-3.2-1b")),
            create_test_worker(7002, "llm", "8081", 0.0, 0, Some("llama-3.2-3b")),
            create_test_worker(7003, "llm", "8082", 0.0, 0, Some("llama-3.2-1b")),
            create_test_worker(7004, "llm", "8083", 0.0, 0, None), // No model
        ],
    );

    // WHEN: Find workers with specific model
    let with_1b = registry.find_workers_with_model("llama-3.2-1b");

    // THEN: Only matching workers returned
    assert_eq!(with_1b.len(), 2);
    assert!(with_1b.iter().all(|w| w.model.as_deref() == Some("llama-3.2-1b")));

    // WHEN: Find workers with different model
    let with_3b = registry.find_workers_with_model("llama-3.2-3b");

    // THEN: Returns correct worker
    assert_eq!(with_3b.len(), 1);
    assert_eq!(with_3b[0].pid, 7002);

    // WHEN: Find workers with non-existent model
    let with_none = registry.find_workers_with_model("nonexistent");

    // THEN: Returns empty
    assert_eq!(with_none.len(), 0);
}

#[test]
fn test_find_workers_with_capacity_checks_vram() {
    // GIVEN: Workers with different VRAM usage
    let registry = HiveRegistry::new();
    registry.update_workers(
        "localhost",
        vec![
            create_test_worker(8001, "llm", "8080", 0.0, 0, None),       // 0 MB used
            create_test_worker(8002, "llm", "8081", 85.0, 8192, None),   // 8 GB used
            create_test_worker(8003, "llm", "8082", 50.0, 16384, None),  // 16 GB used
            create_test_worker(8004, "llm", "8083", 90.0, 20480, None),  // 20 GB used
        ],
    );

    // WHEN: Find workers with 4GB capacity needed
    let with_4gb = registry.find_workers_with_capacity(4096);

    // THEN: Only workers with enough free VRAM
    // Total VRAM assumed: 24576 MB (24 GB)
    // Worker 8001: 0 + 4096 = 4096 < 24576 ✓
    // Worker 8002: 8192 + 4096 = 12288 < 24576 ✓
    // Worker 8003: 16384 + 4096 = 20480 < 24576 ✓
    // Worker 8004: 20480 + 4096 = 24576 NOT < 24576 ✗
    assert_eq!(with_4gb.len(), 3);
    let pids: Vec<u32> = with_4gb.iter().map(|w| w.pid).collect();
    assert!(pids.contains(&8001));
    assert!(pids.contains(&8002));
    assert!(pids.contains(&8003));
    assert!(!pids.contains(&8004));

    // WHEN: Find workers with 8GB capacity needed
    let with_8gb = registry.find_workers_with_capacity(8192);

    // THEN: Only workers with 8GB+ free
    // Worker 8001: 0 + 8192 = 8192 < 24576 ✓
    // Worker 8002: 8192 + 8192 = 16384 < 24576 ✓
    // Worker 8003: 16384 + 8192 = 24576 NOT < 24576 ✗
    // Worker 8004: 20480 + 8192 = 28672 NOT < 24576 ✗
    assert_eq!(with_8gb.len(), 2);
}

// ============================================================================
// THREAD SAFETY TESTS
// ============================================================================

#[test]
fn test_update_workers_thread_safe() {
    // GIVEN: Shared registry
    let registry = Arc::new(HiveRegistry::new());

    // WHEN: Multiple threads update workers
    let mut handles = vec![];

    for i in 0..10 {
        let reg = Arc::clone(&registry);
        let handle = thread::spawn(move || {
            let workers = vec![
                create_test_worker((i * 100) as u32, "llm", "8080", 0.0, 0, None),
            ];
            reg.update_workers(&format!("hive-{}", i), workers);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // THEN: All updates succeeded
    let all = registry.get_all_workers();
    assert_eq!(all.len(), 10, "All updates should succeed");
}

#[test]
fn test_concurrent_read_write() {
    // GIVEN: Registry with initial workers
    let registry = Arc::new(HiveRegistry::new());
    registry.update_workers(
        "localhost",
        vec![create_test_worker(9001, "llm", "8080", 0.0, 0, None)],
    );

    // WHEN: Concurrent reads and writes
    let mut handles = vec![];

    // Writer thread
    let reg = Arc::clone(&registry);
    let write_handle = thread::spawn(move || {
        for i in 0..100 {
            let workers = vec![
                create_test_worker((9000 + i) as u32, "llm", "8080", 0.0, 0, None),
            ];
            reg.update_workers("localhost", workers);
        }
    });
    handles.push(write_handle);

    // Reader threads
    for _ in 0..5 {
        let reg = Arc::clone(&registry);
        let read_handle = thread::spawn(move || {
            for _ in 0..100 {
                let _ = reg.get_workers("localhost");
                let _ = reg.get_all_workers();
                let _ = reg.find_idle_workers();
            }
        });
        handles.push(read_handle);
    }

    // THEN: No panics or deadlocks
    for handle in handles {
        handle.join().unwrap();
    }
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_empty_workers_array() {
    // GIVEN: Registry
    let registry = HiveRegistry::new();

    // WHEN: Update with empty workers
    registry.update_workers("localhost", vec![]);

    // THEN: Stored as empty
    let workers = registry.get_workers("localhost");
    assert!(workers.is_some());
    assert_eq!(workers.unwrap().len(), 0);
}

#[test]
fn test_multiple_hives_isolated() {
    // GIVEN: Multiple hives
    let registry = HiveRegistry::new();

    registry.update_workers(
        "hive-1",
        vec![create_test_worker(10001, "llm", "8080", 0.0, 0, None)],
    );

    registry.update_workers(
        "hive-2",
        vec![create_test_worker(10002, "vllm", "9000", 0.0, 0, None)],
    );

    // WHEN: Get workers for each hive
    let hive1_workers = registry.get_workers("hive-1").unwrap();
    let hive2_workers = registry.get_workers("hive-2").unwrap();

    // THEN: Hives isolated
    assert_eq!(hive1_workers.len(), 1);
    assert_eq!(hive2_workers.len(), 1);
    assert_eq!(hive1_workers[0].pid, 10001);
    assert_eq!(hive2_workers[0].pid, 10002);
}

#[test]
fn test_scheduling_on_empty_registry() {
    // GIVEN: Empty registry
    let registry = HiveRegistry::new();

    // WHEN: Run scheduling queries
    let idle = registry.find_idle_workers();
    let with_model = registry.find_workers_with_model("test");
    let with_capacity = registry.find_workers_with_capacity(4096);

    // THEN: All return empty (not error)
    assert_eq!(idle.len(), 0);
    assert_eq!(with_model.len(), 0);
    assert_eq!(with_capacity.len(), 0);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn create_test_worker(
    pid: u32,
    group: &str,
    instance: &str,
    gpu_util_pct: f64,
    vram_mb: u64,
    model: Option<&str>,
) -> ProcessStats {
    ProcessStats {
        pid,
        group: group.to_string(),
        instance: instance.to_string(),
        cpu_pct: 0.0,
        rss_mb: 1024,
        io_r_mb_s: 0.0,
        io_w_mb_s: 0.0,
        uptime_s: 100,
        gpu_util_pct,
        vram_mb,
        total_vram_mb: 24576, // TEAM-364: Default 24GB for tests
        model: model.map(|s| s.to_string()),
    }
}
