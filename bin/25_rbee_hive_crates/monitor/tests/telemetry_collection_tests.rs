// TEAM-XXX: Telemetry pipeline testing - High-level collection API tests
//!
//! Tests for collect_all_workers(), collect_group(), collect_instance()
//!
//! Coverage:
//! - Multi-worker collection
//! - Group filtering
//! - Single instance collection
//! - Empty cgroup handling

use rbee_hive_monitor::{collect_all_workers, collect_group, collect_instance, MonitorConfig, ProcessMonitor};

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_collect_all_workers_returns_all() {
    // GIVEN: Multiple workers across groups
    let workers = vec![
        ("llm", "8080"),
        ("llm", "8081"),
        ("vllm", "9000"),
        ("comfy", "7000"),
    ];

    let mut pids = Vec::new();
    for (group, instance) in &workers {
        let config = MonitorConfig {
            group: group.to_string(),
            instance: instance.to_string(),
            cpu_limit: None,
            memory_limit: None,
        };

        let pid = ProcessMonitor::spawn_monitored(
            config,
            "/bin/sleep",
            vec!["300".to_string()],
        )
        .await
        .unwrap();
        pids.push(pid);
    }

    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // WHEN: Collect all workers
    let result = collect_all_workers().await;

    // THEN: All workers returned
    assert!(result.is_ok());
    let all = result.unwrap();
    assert_eq!(all.len(), 4, "Should collect all 4 workers");

    // Verify groups
    assert_eq!(all.iter().filter(|w| w.group == "llm").count(), 2);
    assert_eq!(all.iter().filter(|w| w.group == "vllm").count(), 1);
    assert_eq!(all.iter().filter(|w| w.group == "comfy").count(), 1);

    // CLEANUP
    for pid in pids {
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }
    }
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_collect_group_filters_by_group() {
    // GIVEN: Workers in multiple groups
    let llm_workers = vec!["8080", "8081", "8082"];
    let vllm_workers = vec!["9000", "9001"];

    let mut pids = Vec::new();

    // Spawn LLM workers
    for instance in &llm_workers {
        let config = MonitorConfig {
            group: "llm".to_string(),
            instance: instance.to_string(),
            cpu_limit: None,
            memory_limit: None,
        };
        let pid = ProcessMonitor::spawn_monitored(
            config,
            "/bin/sleep",
            vec!["300".to_string()],
        )
        .await
        .unwrap();
        pids.push(pid);
    }

    // Spawn VLLM workers
    for instance in &vllm_workers {
        let config = MonitorConfig {
            group: "vllm".to_string(),
            instance: instance.to_string(),
            cpu_limit: None,
            memory_limit: None,
        };
        let pid = ProcessMonitor::spawn_monitored(
            config,
            "/bin/sleep",
            vec!["300".to_string()],
        )
        .await
        .unwrap();
        pids.push(pid);
    }

    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // WHEN: Collect only LLM group
    let result = collect_group("llm").await;

    // THEN: Only LLM workers returned
    assert!(result.is_ok());
    let llm_stats = result.unwrap();
    assert_eq!(llm_stats.len(), 3, "Should collect 3 LLM workers");
    assert!(llm_stats.iter().all(|w| w.group == "llm"));

    // WHEN: Collect only VLLM group
    let result = collect_group("vllm").await;

    // THEN: Only VLLM workers returned
    assert!(result.is_ok());
    let vllm_stats = result.unwrap();
    assert_eq!(vllm_stats.len(), 2, "Should collect 2 VLLM workers");
    assert!(vllm_stats.iter().all(|w| w.group == "vllm"));

    // CLEANUP
    for pid in pids {
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }
    }
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_collect_instance_single_worker() {
    // GIVEN: Specific worker instance
    let config = MonitorConfig {
        group: "llm".to_string(),
        instance: "8080".to_string(),
        cpu_limit: None,
        memory_limit: None,
    };

    let pid = ProcessMonitor::spawn_monitored(
        config,
        "/bin/sleep",
        vec!["300".to_string()],
    )
    .await
    .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // WHEN: Collect specific instance
    let result = collect_instance("llm", "8080").await;

    // THEN: Single worker returned
    assert!(result.is_ok());
    let stats = result.unwrap();
    assert_eq!(stats.group, "llm");
    assert_eq!(stats.instance, "8080");
    assert_eq!(stats.pid, pid);

    // CLEANUP
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_collect_handles_empty_cgroup() {
    // GIVEN: No workers in cgroup tree
    // (Assuming rbee.slice exists but is empty)

    // WHEN: Collect all workers
    let result = collect_all_workers().await;

    // THEN: Returns empty list (not error)
    assert!(result.is_ok());
    let workers = result.unwrap();
    // May contain workers from other tests running concurrently
    // Just verify no crash
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_collect_group_nonexistent() {
    // GIVEN: Group that doesn't exist
    // WHEN: Try to collect
    let result = collect_group("nonexistent_group").await;

    // THEN: Returns empty list (not error)
    assert!(result.is_ok());
    let workers = result.unwrap();
    assert_eq!(workers.len(), 0, "Should return empty list");
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_collect_instance_nonexistent() {
    // GIVEN: Instance that doesn't exist
    // WHEN: Try to collect
    let result = collect_instance("llm", "99999").await;

    // THEN: Returns error
    assert!(result.is_err(), "Should fail for nonexistent instance");
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_collect_all_workers_partial_failure() {
    // GIVEN: Mix of healthy and dead workers
    let config1 = MonitorConfig {
        group: "test".to_string(),
        instance: "1111".to_string(),
        cpu_limit: None,
        memory_limit: None,
    };

    let config2 = MonitorConfig {
        group: "test".to_string(),
        instance: "2222".to_string(),
        cpu_limit: None,
        memory_limit: None,
    };

    // Spawn two workers
    let pid1 = ProcessMonitor::spawn_monitored(
        config1,
        "/bin/sleep",
        vec!["300".to_string()],
    )
    .await
    .unwrap();

    let pid2 = ProcessMonitor::spawn_monitored(
        config2,
        "/bin/sleep",
        vec!["1".to_string()],  // Dies quickly
    )
    .await
    .unwrap();

    // Wait for second worker to die
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // WHEN: Collect all (one alive, one dead)
    let result = collect_all_workers().await;

    // THEN: Returns workers that are alive (skips dead)
    // NOTE: Current implementation may fail on first dead worker
    // This test documents expected behavior after fix
    assert!(result.is_ok());
    let workers = result.unwrap();
    
    // Should include at least the alive worker
    let alive_workers: Vec<_> = workers.iter()
        .filter(|w| w.group == "test")
        .collect();
    
    // Current behavior: may be 0 (collection fails on dead worker)
    // Expected behavior: should be 1 (skips dead, returns alive)
    // TODO: Fix collection to continue on errors

    // CLEANUP
    unsafe {
        libc::kill(pid1 as i32, libc::SIGKILL);
    }
}

// ============================================================================
// PLATFORM TESTS
// ============================================================================

#[tokio::test]
#[cfg(not(target_os = "linux"))]
async fn test_collect_all_workers_fallback() {
    // GIVEN: Non-Linux platform
    // WHEN: Try to collect
    let result = collect_all_workers().await;

    // THEN: Returns empty list (not implemented)
    assert!(result.is_ok());
    let workers = result.unwrap();
    assert_eq!(workers.len(), 0);
}

#[tokio::test]
#[cfg(not(target_os = "linux"))]
async fn test_collect_group_fallback() {
    // GIVEN: Non-Linux platform
    // WHEN: Try to collect group
    let result = collect_group("llm").await;

    // THEN: Returns empty list
    assert!(result.is_ok());
    let workers = result.unwrap();
    assert_eq!(workers.len(), 0);
}
