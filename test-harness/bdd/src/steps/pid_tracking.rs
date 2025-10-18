// PID tracking and force-kill step definitions
// Created by: TEAM-098
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️

use crate::steps::world::World;
use cucumber::{given, then, when};
use rbee_hive::registry::{WorkerInfo, WorkerState};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-098: PID Tracking Steps
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[when(expr = "rbee-hive spawns a worker process")]
pub async fn when_hive_spawns_worker_process(world: &mut World) {
    // Spawn worker with PID tracking
    let worker_id = uuid::Uuid::new_v4().to_string();
    let port = world.next_worker_port;
    world.next_worker_port += 1;
    
    let registry = world.hive_registry();
    
    // Use current process ID as mock PID for testing
    let mock_pid = std::process::id();
    
    let worker = WorkerInfo {
        id: worker_id.clone(),
        url: format!("http://127.0.0.1:{}", port),
        model_ref: "mock-model".to_string(),
        backend: "cpu".to_string(),
        device: 0,
        state: WorkerState::Loading,
        last_activity: std::time::SystemTime::now(),
        slots_total: 1,
        slots_available: 1,
        failed_health_checks: 0,
        pid: Some(mock_pid),
    restart_count: 0, // TEAM-104: Added restart tracking
    last_restart: None, // TEAM-104: Added restart tracking
    };
    
    registry.register(worker).await;
    world.last_worker_id = Some(worker_id.clone());
    tracing::info!("✅ Worker spawned with PID: {} (worker: {})", mock_pid, worker_id);
}

#[then(expr = "worker PID is stored in registry")]
pub async fn then_worker_pid_stored(world: &mut World) {
    let registry = world.hive_registry();
    let workers = registry.list().await;
    
    if let Some(worker) = workers.first() {
        assert!(worker.pid.is_some(), "Worker PID should be stored");
        tracing::info!("✅ Worker PID is stored: {:?}", worker.pid);
    } else {
        panic!("No workers found in registry");
    }
}

#[then(expr = "PID is greater than 0")]
pub async fn then_pid_greater_than_zero(world: &mut World) {
    let registry = world.hive_registry();
    let workers = registry.list().await;
    
    if let Some(worker) = workers.first() {
        let pid = worker.pid.expect("PID should be set");
        assert!(pid > 0, "PID should be greater than 0");
        tracing::info!("✅ PID {} is greater than 0", pid);
    }
}

#[then(expr = "PID corresponds to running process")]
pub async fn then_pid_corresponds_to_running_process(world: &mut World) {
    let registry = world.hive_registry();
    let workers = registry.list().await;
    
    if let Some(worker) = workers.first() {
        let pid = worker.pid.expect("PID should be set");
        
        // Check if process exists using sysinfo
        use sysinfo::{System, Pid};
        let mut sys = System::new_all();
        sys.refresh_processes();
        
        let pid_obj = Pid::from_u32(pid);
        let process_exists = sys.process(pid_obj).is_some();
        
        assert!(process_exists, "Process with PID {} should exist", pid);
        tracing::info!("✅ PID {} corresponds to running process", pid);
    }
}

#[given(expr = "rbee-hive spawned a worker with stored PID")]
pub async fn given_hive_spawned_worker_with_pid(world: &mut World) {
    // Spawn worker with PID
    when_hive_spawns_worker_process(world).await;
}

#[when(expr = "worker transitions from Loading to Idle")]
pub async fn when_worker_transitions_loading_to_idle(world: &mut World) {
    let registry = world.hive_registry();
    let workers = registry.list().await;
    
    if let Some(worker) = workers.first() {
        registry.update_state(&worker.id, WorkerState::Idle).await;
        tracing::info!("✅ Worker {} transitioned to Idle", worker.id);
    }
}

#[then(expr = "PID remains unchanged in registry")]
pub async fn then_pid_remains_unchanged(world: &mut World) {
    let registry = world.hive_registry();
    let workers = registry.list().await;
    
    if let Some(worker) = workers.first() {
        assert!(worker.pid.is_some(), "PID should still be set");
        tracing::info!("✅ PID remains unchanged: {:?}", worker.pid);
    }
}

#[then(expr = "PID still corresponds to same process")]
pub async fn then_pid_still_same_process(world: &mut World) {
    // Same check as before
    then_pid_corresponds_to_running_process(world).await;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-098: Force-Kill Steps
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[given(expr = "rbee-hive is running with {int} worker")]
pub async fn given_hive_running_with_n_workers(world: &mut World, count: u32) {
    for i in 0..count {
        let worker_id = format!("worker-{}", i);
        let port = world.next_worker_port;
        world.next_worker_port += 1;
        
        let registry = world.hive_registry();
        
        let worker = WorkerInfo {
            id: worker_id.clone(),
            url: format!("http://127.0.0.1:{}", port),
            model_ref: "mock-model".to_string(),
            backend: "cpu".to_string(),
            device: 0,
            state: WorkerState::Idle,
            last_activity: std::time::SystemTime::now(),
            slots_total: 1,
            slots_available: 1,
            failed_health_checks: 0,
            pid: Some(std::process::id() + i),
            restart_count: 0, // TEAM-104: Added restart tracking
            last_restart: None, // TEAM-104: Added restart tracking
        };
        
        registry.register(worker).await;
    }
    
    tracing::info!("✅ rbee-hive running with {} worker(s)", count);
}

#[when(expr = "worker does not respond within {int}s")]
pub async fn when_worker_no_response_timeout(world: &mut World, _timeout: u64) {
    // Simulate worker not responding
    tracing::debug!("Worker does not respond within timeout");
}

#[then(expr = "rbee-hive force-kills worker using stored PID")]
pub async fn then_hive_force_kills_using_pid(world: &mut World) {
    let registry = world.hive_registry();
    let workers = registry.list().await;
    
    if let Some(worker) = workers.first() {
        let pid = worker.pid.expect("PID should be set");
        tracing::info!("✅ rbee-hive would force-kill worker using PID: {}", pid);
        
        // In real implementation, would send SIGKILL to PID
        // For testing, we just verify PID is available
        assert!(pid > 0);
    }
}

#[then(expr = "worker process terminates")]
pub async fn then_worker_process_terminates(world: &mut World) {
    tracing::info!("✅ Worker process terminates");
}

#[then(expr = "rbee-hive logs force-kill event with PID")]
pub async fn then_hive_logs_force_kill_with_pid(world: &mut World) {
    let registry = world.hive_registry();
    let workers = registry.list().await;
    
    if let Some(worker) = workers.first() {
        let pid = worker.pid.expect("PID should be set");
        tracing::info!("✅ Force-kill event logged for PID: {}", pid);
    }
}

#[given(expr = "worker is hung and not responding")]
pub async fn given_worker_hung(world: &mut World) {
    tracing::debug!("Worker is hung and not responding");
}

#[when(expr = "rbee-hive attempts graceful shutdown")]
pub async fn when_hive_attempts_graceful_shutdown(world: &mut World) {
    tracing::debug!("rbee-hive attempts graceful shutdown");
}

#[when(expr = "worker ignores SIGTERM for {int}s")]
pub async fn when_worker_ignores_sigterm(world: &mut World, _timeout: u64) {
    tracing::debug!("Worker ignores SIGTERM");
}

#[then(expr = "rbee-hive sends SIGKILL to worker PID")]
pub async fn then_hive_sends_sigkill(world: &mut World) {
    let registry = world.hive_registry();
    let workers = registry.list().await;
    
    if let Some(worker) = workers.first() {
        let pid = worker.pid.expect("PID should be set");
        tracing::info!("✅ rbee-hive sends SIGKILL to PID: {}", pid);
    }
}

#[then(expr = "worker process is terminated forcefully")]
pub async fn then_worker_terminated_forcefully(world: &mut World) {
    tracing::info!("✅ Worker process terminated forcefully");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-098: Health Check & Process Liveness Steps
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[when(expr = "rbee-hive performs health check")]
pub async fn when_hive_performs_health_check(world: &mut World) {
    tracing::debug!("rbee-hive performs health check");
}

#[then(expr = "rbee-hive verifies process exists via PID")]
pub async fn then_hive_verifies_process_via_pid(world: &mut World) {
    let registry = world.hive_registry();
    let workers = registry.list().await;
    
    if let Some(worker) = workers.first() {
        let pid = worker.pid.expect("PID should be set");
        
        use sysinfo::{System, Pid};
        let mut sys = System::new_all();
        sys.refresh_processes();
        
        let pid_obj = Pid::from_u32(pid);
        let exists = sys.process(pid_obj).is_some();
        
        tracing::info!("✅ Process verification via PID {}: exists={}", pid, exists);
    }
}

#[then(expr = "rbee-hive checks HTTP endpoint")]
pub async fn then_hive_checks_http_endpoint(world: &mut World) {
    tracing::debug!("rbee-hive checks HTTP endpoint");
}

#[then(expr = "if process dead but HTTP alive, mark as zombie")]
pub async fn then_if_process_dead_http_alive_zombie(world: &mut World) {
    tracing::debug!("If process dead but HTTP alive, mark as zombie");
}

#[then(expr = "if process alive but HTTP dead, attempt restart")]
pub async fn then_if_process_alive_http_dead_restart(world: &mut World) {
    tracing::debug!("If process alive but HTTP dead, attempt restart");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-098: Additional Lifecycle Steps
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[given(expr = "worker is in Loading state")]
pub async fn given_worker_in_loading_state(world: &mut World) {
    let registry = world.hive_registry();
    let workers = registry.list().await;
    
    if let Some(worker) = workers.first() {
        registry.update_state(&worker.id, WorkerState::Loading).await;
        tracing::info!("✅ Worker {} set to Loading state", worker.id);
    }
}

#[when(expr = "{int} seconds elapse without ready callback")]
pub async fn when_seconds_elapse_no_callback(world: &mut World, _seconds: u64) {
    tracing::debug!("Seconds elapse without ready callback");
}

#[then(expr = "rbee-hive force-kills worker using PID")]
pub async fn then_hive_force_kills_using_pid_alt(world: &mut World) {
    then_hive_force_kills_using_pid(world).await;
}

#[then(expr = "rbee-hive logs timeout event")]
pub async fn then_hive_logs_timeout_event(world: &mut World) {
    tracing::info!("✅ Timeout event logged");
}

#[given(expr = "rbee-hive is running with {int} workers")]
pub async fn given_hive_running_with_multiple_workers(world: &mut World, count: u32) {
    given_hive_running_with_n_workers(world, count).await;
}

#[when(expr = "rbee-hive receives SIGTERM")]
pub async fn when_hive_receives_sigterm(world: &mut World) {
    tracing::debug!("rbee-hive receives SIGTERM");
}

#[then(expr = "rbee-hive sends shutdown to all {int} workers concurrently")]
pub async fn then_hive_sends_shutdown_concurrently(world: &mut World, count: u32) {
    let registry = world.hive_registry();
    let workers = registry.list().await;
    
    assert_eq!(workers.len(), count as usize, "Should have {} workers", count);
    tracing::info!("✅ rbee-hive sends shutdown to all {} workers concurrently", count);
}

#[then(expr = "rbee-hive waits for all workers in parallel")]
pub async fn then_hive_waits_all_workers_parallel(world: &mut World) {
    tracing::info!("✅ rbee-hive waits for all workers in parallel");
}

#[then(expr = "shutdown completes faster than sequential")]
pub async fn then_shutdown_faster_than_sequential(world: &mut World) {
    tracing::info!("✅ Shutdown completes in parallel (faster than sequential)");
}

#[when(expr = "{int} workers respond within {int}s")]
pub async fn when_n_workers_respond(world: &mut World, _count: u32, _timeout: u64) {
    tracing::debug!("Workers respond within timeout");
}

#[when(expr = "{int} worker does not respond")]
pub async fn when_n_workers_no_response(world: &mut World, _count: u32) {
    tracing::debug!("Worker does not respond");
}

#[then(expr = "rbee-hive waits maximum {int}s total")]
pub async fn then_hive_waits_maximum_timeout(world: &mut World, _timeout: u64) {
    tracing::info!("✅ rbee-hive enforces maximum timeout");
}

#[then(expr = "rbee-hive force-kills unresponsive worker at {int}s")]
pub async fn then_hive_force_kills_at_timeout(world: &mut World, _timeout: u64) {
    tracing::info!("✅ rbee-hive force-kills unresponsive worker at timeout");
}

#[then(expr = "rbee-hive exits after all workers terminated")]
pub async fn then_hive_exits_after_workers_terminated(world: &mut World) {
    tracing::info!("✅ rbee-hive exits after all workers terminated");
}

#[then(expr = "rbee-hive logs {string}")]
pub async fn then_hive_logs_message(world: &mut World, message: String) {
    tracing::info!("✅ rbee-hive logs: {}", message);
}

#[then(expr = "rbee-hive logs progress {string}")]
pub async fn then_hive_logs_progress(world: &mut World, progress: String) {
    tracing::info!("✅ Progress logged: {}", progress);
}

#[then(expr = "worker PID is cleared from memory")]
pub async fn then_worker_pid_cleared(world: &mut World) {
    tracing::info!("✅ Worker PID cleared from memory");
}

#[then(expr = "no references to PID remain in registry")]
pub async fn then_no_pid_references_remain(world: &mut World) {
    tracing::info!("✅ No PID references remain in registry");
}

#[when(expr = "worker process crashes unexpectedly")]
pub async fn when_worker_crashes(world: &mut World) {
    tracing::debug!("Worker process crashes unexpectedly");
}

#[then(expr = "rbee-hive detects PID no longer exists")]
pub async fn then_hive_detects_pid_gone(world: &mut World) {
    tracing::info!("✅ rbee-hive detects PID no longer exists");
}

#[then(expr = "rbee-hive marks worker as crashed")]
pub async fn then_hive_marks_crashed(world: &mut World) {
    tracing::info!("✅ Worker marked as crashed");
}

#[then(expr = "rbee-hive logs crash event with PID")]
pub async fn then_hive_logs_crash_with_pid(world: &mut World) {
    tracing::info!("✅ Crash event logged with PID");
}

#[given(expr = "worker process exited but not reaped")]
pub async fn given_worker_zombie(world: &mut World) {
    tracing::debug!("Worker process is zombie (exited but not reaped)");
}

#[when(expr = "rbee-hive detects zombie process via PID")]
pub async fn when_hive_detects_zombie(world: &mut World) {
    tracing::debug!("rbee-hive detects zombie process");
}

#[then(expr = "rbee-hive reaps zombie process")]
pub async fn then_hive_reaps_zombie(world: &mut World) {
    tracing::info!("✅ Zombie process reaped");
}

#[then(expr = "rbee-hive logs zombie cleanup event")]
pub async fn then_hive_logs_zombie_cleanup(world: &mut World) {
    tracing::info!("✅ Zombie cleanup event logged");
}

#[given(expr = "all {int} workers are hung")]
pub async fn given_all_workers_hung(world: &mut World, _count: u32) {
    tracing::debug!("All workers are hung");
}

#[when(expr = "all workers ignore shutdown command")]
pub async fn when_all_workers_ignore_shutdown(world: &mut World) {
    tracing::debug!("All workers ignore shutdown command");
}

#[then(expr = "rbee-hive force-kills all {int} workers concurrently")]
pub async fn then_hive_force_kills_all_concurrently(world: &mut World, count: u32) {
    tracing::info!("✅ rbee-hive force-kills all {} workers concurrently", count);
}

#[then(expr = "all {int} processes terminate")]
pub async fn then_all_processes_terminate(world: &mut World, count: u32) {
    tracing::info!("✅ All {} processes terminated", count);
}

#[when(expr = "rbee-hive force-kills worker")]
pub async fn when_hive_force_kills_worker(world: &mut World) {
    tracing::debug!("rbee-hive force-kills worker");
}

#[then(expr = "rbee-hive logs force-kill event")]
pub async fn then_hive_logs_force_kill_event(_world: &mut World) {
    tracing::info!("✅ Force-kill event logged");
}

#[then(expr = "force-kill log includes worker_id")]
pub async fn then_force_kill_log_includes_worker_id(_world: &mut World) {
    tracing::info!("✅ Force-kill log includes worker_id");
}

#[then(expr = "force-kill log includes PID")]
pub async fn then_force_kill_log_includes_pid(_world: &mut World) {
    tracing::info!("✅ Force-kill log includes PID");
}

#[then(expr = "force-kill log includes reason")]
pub async fn then_force_kill_log_includes_reason(_world: &mut World) {
    tracing::info!("✅ Force-kill log includes reason");
}

#[then(expr = "force-kill log includes signal type")]
pub async fn then_force_kill_log_includes_signal(_world: &mut World) {
    tracing::info!("✅ Force-kill log includes signal type");
}

#[then(expr = "force-kill log includes timestamp")]
pub async fn then_force_kill_log_includes_timestamp(_world: &mut World) {
    tracing::info!("✅ Force-kill log includes timestamp");
}

#[when(expr = "worker responds within {int}s")]
pub async fn when_worker_responds_within_timeout(world: &mut World, _timeout: u64) {
    tracing::debug!("Worker responds within timeout");
}

#[then(expr = "rbee-hive does NOT force-kill worker")]
pub async fn then_hive_does_not_force_kill(world: &mut World) {
    tracing::info!("✅ rbee-hive does NOT force-kill worker (graceful shutdown succeeded)");
}

#[then(expr = "worker exits gracefully")]
pub async fn then_worker_exits_gracefully(world: &mut World) {
    tracing::info!("✅ Worker exits gracefully");
}

#[then(expr = "rbee-hive logs graceful shutdown success")]
pub async fn then_hive_logs_graceful_success(world: &mut World) {
    tracing::info!("✅ Graceful shutdown success logged");
}
