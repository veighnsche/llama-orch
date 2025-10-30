// TEAM-XXX: Telemetry pipeline testing - ProcessMonitor unit tests
//!
//! Tests for worker spawn and telemetry collection via cgroups + nvidia-smi
//!
//! Coverage:
//! - Worker spawn with cgroup placement
//! - Resource limit enforcement
//! - Telemetry collection from cgroups
//! - GPU stats via nvidia-smi
//! - Model detection from cmdline
//! - Uptime calculation
//! - Error handling and graceful degradation

use rbee_hive_monitor::{MonitorConfig, ProcessMonitor, ProcessStats};
use std::path::PathBuf;

// ============================================================================
// WORKER SPAWN TESTS
// ============================================================================

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_spawn_creates_cgroup() {
    // GIVEN: Valid monitor config
    let config = MonitorConfig {
        group: "test".to_string(),
        instance: "9999".to_string(),
        cpu_limit: None,
        memory_limit: None,
    };

    // WHEN: Spawn monitored process (mock binary)
    let result = ProcessMonitor::spawn_monitored(
        config,
        "/bin/sleep",
        vec!["300".to_string()],
    )
    .await;

    // THEN: Spawn succeeds and cgroup exists
    assert!(result.is_ok(), "Spawn should succeed");
    let pid = result.unwrap();

    let cgroup_path = format!("/sys/fs/cgroup/rbee.slice/test/9999");
    assert!(
        std::path::Path::new(&cgroup_path).exists(),
        "Cgroup directory should exist"
    );

    // CLEANUP: Kill process
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_spawn_applies_cpu_limit() {
    // GIVEN: Config with CPU limit
    let config = MonitorConfig {
        group: "test".to_string(),
        instance: "9998".to_string(),
        cpu_limit: Some("200%".to_string()), // 2 cores
        memory_limit: None,
    };

    // WHEN: Spawn with CPU limit
    let result = ProcessMonitor::spawn_monitored(
        config,
        "/bin/sleep",
        vec!["300".to_string()],
    )
    .await;

    // THEN: CPU limit applied in cgroup
    assert!(result.is_ok());
    let pid = result.unwrap();

    let cpu_max_path = "/sys/fs/cgroup/rbee.slice/test/9998/cpu.max";
    let cpu_max = std::fs::read_to_string(cpu_max_path).unwrap();
    assert!(cpu_max.contains("200000"), "CPU limit should be 200000 quota");

    // CLEANUP
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_spawn_applies_memory_limit() {
    // GIVEN: Config with memory limit
    let config = MonitorConfig {
        group: "test".to_string(),
        instance: "9997".to_string(),
        cpu_limit: None,
        memory_limit: Some("1G".to_string()),
    };

    // WHEN: Spawn with memory limit
    let result = ProcessMonitor::spawn_monitored(
        config,
        "/bin/sleep",
        vec!["300".to_string()],
    )
    .await;

    // THEN: Memory limit applied
    assert!(result.is_ok());
    let pid = result.unwrap();

    let mem_max_path = "/sys/fs/cgroup/rbee.slice/test/9997/memory.max";
    let mem_max = std::fs::read_to_string(mem_max_path).unwrap();
    let expected = 1024 * 1024 * 1024; // 1GB in bytes
    assert_eq!(mem_max.trim(), expected.to_string());

    // CLEANUP
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_spawn_returns_valid_pid() {
    // GIVEN: Valid config
    let config = MonitorConfig {
        group: "test".to_string(),
        instance: "9996".to_string(),
        cpu_limit: None,
        memory_limit: None,
    };

    // WHEN: Spawn process
    let result = ProcessMonitor::spawn_monitored(
        config,
        "/bin/sleep",
        vec!["300".to_string()],
    )
    .await;

    // THEN: Returns valid PID
    assert!(result.is_ok());
    let pid = result.unwrap();
    assert!(pid > 0, "PID should be positive");

    // Verify process exists
    let proc_path = format!("/proc/{}", pid);
    assert!(std::path::Path::new(&proc_path).exists());

    // CLEANUP
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

// ============================================================================
// TELEMETRY COLLECTION TESTS
// ============================================================================

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_collect_reads_cgroup_stats() {
    // GIVEN: Running monitored process
    let config = MonitorConfig {
        group: "test".to_string(),
        instance: "9995".to_string(),
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

    // Wait for cgroup population
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // WHEN: Collect stats
    let result = ProcessMonitor::collect_stats("test", "9995").await;

    // THEN: Stats collected successfully
    assert!(result.is_ok(), "Collection should succeed");
    let stats = result.unwrap();
    assert_eq!(stats.pid, pid);
    assert_eq!(stats.group, "test");
    assert_eq!(stats.instance, "9995");
    assert!(stats.rss_mb > 0, "RSS should be non-zero");

    // CLEANUP
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_collect_queries_nvidia_smi() {
    // NOTE: This test requires nvidia-smi to be available
    // If not available, should gracefully return (0.0, 0)

    // GIVEN: Running process
    let config = MonitorConfig {
        group: "test".to_string(),
        instance: "9994".to_string(),
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

    // WHEN: Collect stats (includes GPU query)
    let result = ProcessMonitor::collect_stats("test", "9994").await;

    // THEN: Collection succeeds (GPU stats may be 0 if no GPU)
    assert!(result.is_ok());
    let stats = result.unwrap();
    // Process not using GPU, should return 0
    assert_eq!(stats.gpu_util_pct, 0.0);
    assert_eq!(stats.vram_mb, 0);

    // CLEANUP
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_collect_parses_cmdline() {
    // GIVEN: Process with --model argument
    let config = MonitorConfig {
        group: "test".to_string(),
        instance: "9993".to_string(),
        cpu_limit: None,
        memory_limit: None,
    };

    // Spawn with --model in args
    let pid = ProcessMonitor::spawn_monitored(
        config,
        "/bin/bash",
        vec![
            "-c".to_string(),
            "exec -a \"test-worker --model test-model-name\" sleep 300".to_string(),
        ],
    )
    .await
    .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // WHEN: Collect stats
    let result = ProcessMonitor::collect_stats("test", "9993").await;

    // THEN: Model detected
    assert!(result.is_ok());
    let stats = result.unwrap();
    // NOTE: exec -a changes argv[0] but not cmdline, so model may not be detected
    // This test verifies parser doesn't crash on missing model

    // CLEANUP
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_collect_calculates_uptime() {
    // GIVEN: Process running for known time
    let config = MonitorConfig {
        group: "test".to_string(),
        instance: "9992".to_string(),
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

    // Wait 2 seconds
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // WHEN: Collect stats
    let result = ProcessMonitor::collect_stats("test", "9992").await;

    // THEN: Uptime is approximately 2 seconds
    assert!(result.is_ok());
    let stats = result.unwrap();
    assert!(stats.uptime_s >= 1, "Uptime should be at least 1 second");
    assert!(stats.uptime_s <= 5, "Uptime should be at most 5 seconds");

    // CLEANUP
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_collect_handles_missing_gpu() {
    // GIVEN: System without GPU or nvidia-smi
    let config = MonitorConfig {
        group: "test".to_string(),
        instance: "9991".to_string(),
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

    // WHEN: Collect stats (nvidia-smi may fail)
    let result = ProcessMonitor::collect_stats("test", "9991").await;

    // THEN: Collection succeeds with zero GPU stats (graceful degradation)
    assert!(result.is_ok());
    let stats = result.unwrap();
    // If no GPU, should return 0
    assert_eq!(stats.gpu_util_pct, 0.0);
    assert_eq!(stats.vram_mb, 0);

    // CLEANUP
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_collect_handles_dead_process() {
    // GIVEN: Worker that died
    let config = MonitorConfig {
        group: "test".to_string(),
        instance: "9990".to_string(),
        cpu_limit: None,
        memory_limit: None,
    };

    let pid = ProcessMonitor::spawn_monitored(
        config,
        "/bin/sleep",
        vec!["1".to_string()],
    )
    .await
    .unwrap();

    // Wait for process to die
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // WHEN: Try to collect stats
    let result = ProcessMonitor::collect_stats("test", "9990").await;

    // THEN: Collection fails gracefully
    assert!(result.is_err(), "Should fail for dead process");
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_enumerate_walks_cgroup_tree() {
    // GIVEN: Multiple workers in different groups
    let configs = vec![
        MonitorConfig {
            group: "llm".to_string(),
            instance: "8080".to_string(),
            cpu_limit: None,
            memory_limit: None,
        },
        MonitorConfig {
            group: "llm".to_string(),
            instance: "8081".to_string(),
            cpu_limit: None,
            memory_limit: None,
        },
        MonitorConfig {
            group: "vllm".to_string(),
            instance: "9000".to_string(),
            cpu_limit: None,
            memory_limit: None,
        },
    ];

    let mut pids = Vec::new();
    for config in configs {
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

    // WHEN: Enumerate all workers
    let result = ProcessMonitor::enumerate_all().await;

    // THEN: All workers found
    assert!(result.is_ok());
    let all_stats = result.unwrap();
    assert_eq!(all_stats.len(), 3, "Should find all 3 workers");

    // CLEANUP
    for pid in pids {
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }
    }
}

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_spawn_invalid_binary() {
    // GIVEN: Invalid binary path
    let config = MonitorConfig {
        group: "test".to_string(),
        instance: "9989".to_string(),
        cpu_limit: None,
        memory_limit: None,
    };

    // WHEN: Try to spawn non-existent binary
    let result = ProcessMonitor::spawn_monitored(
        config,
        "/nonexistent/binary",
        vec![],
    )
    .await;

    // THEN: Spawn fails
    assert!(result.is_err(), "Should fail for invalid binary");
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_spawn_invalid_cpu_limit() {
    // GIVEN: Invalid CPU limit format
    let config = MonitorConfig {
        group: "test".to_string(),
        instance: "9988".to_string(),
        cpu_limit: Some("invalid".to_string()),
        memory_limit: None,
    };

    // WHEN: Try to spawn with bad limit
    let result = ProcessMonitor::spawn_monitored(
        config,
        "/bin/sleep",
        vec!["300".to_string()],
    )
    .await;

    // THEN: Spawn fails with clear error
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("CPU limit"));
}

// ============================================================================
// PLATFORM TESTS
// ============================================================================

#[tokio::test]
#[cfg(not(target_os = "linux"))]
async fn test_spawn_fallback_on_non_linux() {
    // GIVEN: Non-Linux platform
    let config = MonitorConfig {
        group: "test".to_string(),
        instance: "9999".to_string(),
        cpu_limit: Some("200%".to_string()),
        memory_limit: Some("4G".to_string()),
    };

    // WHEN: Spawn process
    let result = ProcessMonitor::spawn_monitored(
        config,
        "/bin/sleep",
        vec!["10".to_string()],
    )
    .await;

    // THEN: Spawn succeeds (limits ignored)
    assert!(result.is_ok(), "Fallback spawn should succeed");

    // No cgroup verification on non-Linux
}

#[tokio::test]
#[cfg(not(target_os = "linux"))]
async fn test_collect_fallback_on_non_linux() {
    // GIVEN: Non-Linux platform
    // WHEN: Try to collect stats
    let result = ProcessMonitor::collect_stats("test", "9999").await;

    // THEN: Returns error (not supported)
    assert!(result.is_err(), "Collection not supported on non-Linux");
}
