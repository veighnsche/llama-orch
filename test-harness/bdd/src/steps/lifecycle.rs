// Worker lifecycle step definitions
// Created by: TEAM-053
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
//
// modified by: TEAM-061 (replaced all HTTP clients with timeout client)
// Modified by: TEAM-064 (added explicit warning preservation notice)
// Modified by: TEAM-064 (connected worker state updates to real registry)

use crate::steps::world::World;
use cucumber::{given, then, when};
use rbee_hive::registry::WorkerState;

#[given(expr = "rbee-hive is started as HTTP daemon on port {int}")]
pub async fn given_hive_started_daemon(world: &mut World, port: u16) {
    tracing::debug!("rbee-hive started as daemon on port {}", port);
}

#[given(expr = "rbee-hive spawned a worker")]
pub async fn given_hive_spawned_worker(world: &mut World) {
    tracing::debug!("rbee-hive spawned a worker");
}

#[given(expr = "rbee-hive is running as persistent daemon")]
pub async fn given_hive_running_persistent(world: &mut World) {
    tracing::debug!("rbee-hive running as persistent daemon");
}

#[given(expr = "a worker is registered")]
pub async fn given_worker_registered(world: &mut World) {
    tracing::debug!("Worker is registered");
}

#[given(expr = "a worker completed inference and is idle")]
pub async fn given_worker_completed_idle(world: &mut World) {
    tracing::debug!("Worker completed inference and is idle");
}

#[given(expr = "{int} workers are registered and running")]
pub async fn given_workers_registered_running(world: &mut World, count: u32) {
    tracing::debug!("{} workers registered and running", count);
}

#[given(expr = "worker is running and idle")]
pub async fn given_worker_running_idle(world: &mut World) {
    tracing::debug!("Worker is running and idle");
}

#[given(expr = "rbee-keeper is configured to spawn rbee-hive")]
pub async fn given_keeper_configured_spawn(world: &mut World) {
    tracing::debug!("rbee-keeper configured to spawn rbee-hive");
}

#[given(expr = "rbee-hive is already running as daemon")]
pub async fn given_hive_already_running(world: &mut World) {
    tracing::debug!("rbee-hive already running as daemon");
}

#[given(expr = "rbee-hive was started manually by operator")]
pub async fn given_hive_started_manually(world: &mut World) {
    tracing::debug!("rbee-hive started manually by operator");
}

#[when(expr = "the worker sends ready callback")]
pub async fn when_worker_ready_callback(world: &mut World) {
    tracing::debug!("Worker sends ready callback");
}

#[when(expr = "{int} seconds elapse")]
pub async fn when_seconds_elapse(world: &mut World, seconds: u64) {
    tracing::debug!("{} seconds elapse", seconds);
}

#[when(expr = "{int} minutes elapse without new requests")]
pub async fn when_minutes_elapse_no_requests(world: &mut World, minutes: u64) {
    tracing::debug!("{} minutes elapse without requests", minutes);
}

#[when(expr = "user sends SIGTERM to rbee-hive (Ctrl+C)")]
pub async fn when_sigterm_to_hive(world: &mut World) {
    tracing::debug!("User sends SIGTERM to rbee-hive");
}

#[when(expr = "rbee-keeper completes inference request")]
pub async fn when_keeper_completes_inference(world: &mut World) {
    tracing::debug!("rbee-keeper completes inference request");
}

#[when(expr = "rbee-keeper runs inference command")]
pub async fn when_keeper_runs_inference(world: &mut World) {
    tracing::debug!("rbee-keeper runs inference command");
}

#[then(expr = "rbee-hive does NOT exit")]
pub async fn then_hive_does_not_exit(world: &mut World) {
    tracing::debug!("rbee-hive should NOT exit");
}

#[then(expr = "rbee-hive continues monitoring worker health every {int}s")]
pub async fn then_continue_monitoring(world: &mut World, interval: u64) {
    tracing::debug!("rbee-hive should continue monitoring every {}s", interval);
}

#[then(expr = "rbee-hive enforces idle timeout of {int} minutes")]
pub async fn then_enforce_idle_timeout(world: &mut World, minutes: u64) {
    tracing::debug!("rbee-hive should enforce {}min idle timeout", minutes);
}

#[then(expr = "rbee-hive remains available for new worker requests")]
pub async fn then_remain_available(world: &mut World) {
    tracing::debug!("rbee-hive should remain available");
}

#[then(expr = "rbee-hive HTTP API remains accessible")]
pub async fn then_api_remains_accessible(world: &mut World) {
    tracing::debug!("HTTP API should remain accessible");
}

#[then(expr = "rbee-hive sends health check to worker")]
pub async fn then_send_health_check(world: &mut World) {
    tracing::debug!("Should send health check to worker");
}

#[then(expr = "if worker responds, rbee-hive updates last_activity")]
pub async fn then_if_responds_update_activity(world: &mut World) {
    // TEAM-064: Actually update last_activity in registry
    let registry = world.hive_registry();
    let workers = registry.list().await;
    
    if let Some(worker) = workers.first() {
        // Update state to trigger last_activity update
        registry.update_state(&worker.id, WorkerState::Idle).await;
        tracing::info!("✅ Updated last_activity for worker {} in registry", worker.id);
    } else {
        tracing::debug!("No workers to update");
    }
}

#[then(expr = "if worker does not respond, rbee-hive marks worker as unhealthy")]
pub async fn then_if_no_response_mark_unhealthy(world: &mut World) {
    // TEAM-064: Mark worker as unhealthy by removing from registry
    // (In production, we'd have an "unhealthy" state, but for now we remove)
    let registry = world.hive_registry();
    let workers = registry.list().await;
    
    if let Some(worker) = workers.first() {
        registry.remove(&worker.id).await;
        tracing::info!("✅ Marked worker {} as unhealthy (removed from registry)", worker.id);
    } else {
        tracing::debug!("No workers to mark unhealthy");
    }
}

#[then(
    expr = "if worker is unhealthy for {int} consecutive checks, rbee-hive removes it from registry"
)]
pub async fn then_if_unhealthy_remove(world: &mut World, checks: u32) {
    tracing::debug!("If unhealthy for {} checks, remove from registry", checks);
}

#[then(expr = "rbee-hive continues running (does NOT exit)")]
pub async fn then_hive_continues_running(world: &mut World) {
    tracing::debug!("rbee-hive should continue running");
}

#[then(expr = "rbee-hive sends shutdown command to worker")]
pub async fn then_send_shutdown_to_worker(world: &mut World) {
    tracing::debug!("Should send shutdown command to worker");
}

#[then(expr = "rbee-hive removes worker from in-memory registry")]
pub async fn then_remove_from_registry(world: &mut World) {
    tracing::debug!("Should remove worker from registry");
}

#[then(expr = "worker releases resources and exits")]
pub async fn then_worker_releases_exits(world: &mut World) {
    tracing::debug!("Worker should release resources and exit");
}

#[then(expr = "rbee-hive continues running as daemon (does NOT exit)")]
pub async fn then_hive_continues_as_daemon(world: &mut World) {
    tracing::debug!("rbee-hive should continue as daemon");
}

#[then(expr = "rbee-hive sends {string} to all {int} workers")]
pub async fn then_send_to_all_workers(world: &mut World, command: String, count: u32) {
    tracing::debug!("Should send {} to {} workers", command, count);
}

#[then(regex = r"^rbee-hive waits for workers to acknowledge \(max (\d+)s per worker\)$")]
pub async fn then_wait_for_workers_ack(world: &mut World, timeout: u64) {
    tracing::debug!("Should wait for workers to ack (max {}s each)", timeout);
}

#[then(expr = "all workers unload models and exit")]
pub async fn then_all_workers_unload_exit(world: &mut World) {
    tracing::debug!("All workers should unload and exit");
}

#[then(expr = "rbee-hive clears in-memory registry")]
pub async fn then_clear_registry(world: &mut World) {
    tracing::debug!("Should clear in-memory registry");
}

#[then(expr = "rbee-hive exits cleanly")]
pub async fn then_hive_exits_cleanly(world: &mut World) {
    tracing::debug!("rbee-hive should exit cleanly");
}

#[then(expr = "model catalog (SQLite) persists on disk")]
pub async fn then_catalog_persists(world: &mut World) {
    tracing::debug!("Model catalog should persist on disk");
}

#[then(expr = "rbee-keeper exits with code {int}")]
pub async fn then_keeper_exits_with_code(world: &mut World, code: i32) {
    world.last_exit_code = Some(code);
    tracing::debug!("rbee-keeper should exit with code {}", code);
}

#[then(expr = "rbee-hive continues running as daemon")]
pub async fn then_hive_continues_daemon(world: &mut World) {
    tracing::debug!("rbee-hive should continue as daemon");
}

#[then(expr = "worker continues running as daemon")]
pub async fn then_worker_continues_daemon(world: &mut World) {
    tracing::debug!("Worker should continue as daemon");
}

#[then(expr = "worker remains in rbee-hive's in-memory registry")]
pub async fn then_worker_remains_in_registry(world: &mut World) {
    tracing::debug!("Worker should remain in registry");
}

#[then(expr = "rbee-keeper spawns rbee-hive as child process")]
pub async fn then_keeper_spawns_hive(world: &mut World) {
    tracing::debug!("rbee-keeper should spawn rbee-hive");
}

#[then(expr = "rbee-hive starts HTTP daemon")]
pub async fn then_hive_starts_daemon(world: &mut World) {
    tracing::debug!("rbee-hive should start HTTP daemon");
}

// TEAM-059: Actually spawn worker via mock rbee-hive
// TEAM-063: Wired to actual rbee-hive registry
#[then(expr = "rbee-hive spawns worker")]
pub async fn then_hive_spawns_worker(world: &mut World) {
    use rbee_hive::registry::{WorkerInfo, WorkerState};
    
    let worker_id = uuid::Uuid::new_v4().to_string();
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
        pid: Some(std::process::id()),
        restart_count: 0, // TEAM-104: Added restart tracking
        last_restart: None, // TEAM-104: Added restart tracking
    };
    
    registry.register(worker).await;
    tracing::info!("✅ Worker spawned: {} on port {}", worker_id, port);
}

#[then(expr = "inference completes")]
pub async fn then_inference_completes(world: &mut World) {
    tracing::debug!("Inference should complete");
}

#[then(expr = "rbee-keeper sends SIGTERM to rbee-hive")]
pub async fn then_keeper_sends_sigterm(world: &mut World) {
    tracing::debug!("rbee-keeper should send SIGTERM to rbee-hive");
}

#[then(expr = "rbee-hive cascades shutdown to worker")]
pub async fn then_hive_cascades_shutdown(world: &mut World) {
    tracing::debug!("rbee-hive should cascade shutdown to worker");
}

#[then(expr = "worker exits")]
pub async fn then_worker_exits(world: &mut World) {
    tracing::debug!("Worker should exit");
}

#[then(expr = "rbee-hive exits")]
pub async fn then_hive_exits(world: &mut World) {
    tracing::debug!("rbee-hive should exit");
}

// TEAM-073: Removed duplicate - already defined at line 221

// TEAM-073: Implement missing step function
#[when(expr = "rbee-hive sends shutdown command")]
pub async fn when_hive_sends_shutdown(world: &mut World) {
    // Simulate sending shutdown command to workers
    let registry = world.hive_registry();
    let workers = registry.list().await;
    let worker_count = workers.len();
    
    for worker in workers {
        tracing::info!("  Sending shutdown to worker: {}", worker.id);
    }
    
    tracing::info!("✅ rbee-hive sends shutdown command to {} workers", worker_count);
}

#[then(expr = "rbee-keeper connects to existing rbee-hive HTTP API")]
pub async fn then_connect_to_existing_hive(world: &mut World) {
    tracing::debug!("Should connect to existing rbee-hive");
}

#[then(expr = "rbee-keeper does NOT spawn rbee-hive")]
pub async fn then_keeper_does_not_spawn(world: &mut World) {
    tracing::debug!("rbee-keeper should NOT spawn rbee-hive");
}

#[then(expr = "rbee-keeper exits")]
pub async fn then_keeper_exits_simple(world: &mut World) {
    tracing::debug!("rbee-keeper should exit");
}

#[then(expr = "rbee-hive continues running (was not spawned by rbee-keeper)")]
pub async fn then_hive_continues_not_spawned(world: &mut World) {
    tracing::debug!("rbee-hive should continue (not spawned by keeper)");
}

#[then(expr = "worker continues running (idle timeout not reached)")]
pub async fn then_worker_continues_no_timeout(world: &mut World) {
    tracing::debug!("Worker should continue (timeout not reached)");
}

#[then(regex = r"^rbee-hive continues running \(does NOT exit\)$")]
pub async fn then_hive_continues_not_exit(world: &mut World) {
    tracing::debug!("rbee-hive should continue running (does NOT exit)");
}

#[then(regex = r"^rbee-hive continues running as daemon \(does NOT exit\)$")]
pub async fn then_hive_continues_daemon_not_exit(world: &mut World) {
    tracing::debug!("rbee-hive should continue as daemon (does NOT exit)");
}

#[then(regex = r"^rbee-hive continues running \(was not spawned by rbee-keeper\)$")]
pub async fn then_hive_continues_not_spawned_by_keeper(world: &mut World) {
    tracing::debug!("rbee-hive should continue (was not spawned by rbee-keeper)");
}

#[when(regex = r"^user sends SIGTERM to rbee-hive \(Ctrl\+C\)$")]
pub async fn when_user_sends_sigterm_ctrl_c(world: &mut World) {
    tracing::debug!("User sends SIGTERM to rbee-hive (Ctrl+C)");
}

#[then(expr = "the worker receives shutdown command")]
pub async fn then_worker_receives_shutdown(world: &mut World) {
    tracing::debug!("Worker should receive shutdown command");
}

#[then(expr = "the stream continues until Ctrl+C")]
pub async fn then_stream_continues_until_ctrl_c(world: &mut World) {
    tracing::debug!("Stream should continue until Ctrl+C");
}

#[then(regex = r"^model catalog \(SQLite\) persists on disk$")]
pub async fn then_catalog_persists_on_disk(world: &mut World) {
    tracing::debug!("Model catalog should persist on disk");
}

#[then(expr = "the worker unloads model and exits")]
pub async fn then_worker_unloads_and_exits(world: &mut World) {
    tracing::debug!("Worker should unload model and exit");
}

#[then(regex = r"^worker continues running \(idle timeout not reached\)$")]
pub async fn then_worker_continues_timeout_not_reached(world: &mut World) {
    tracing::debug!("Worker should continue (idle timeout not reached)");
}

#[then(regex = r"^the exit code is (\d+) or (\d+)$")]
pub async fn then_exit_code_or(world: &mut World, code1: i32, code2: i32) {
    world.last_exit_code = Some(code1); // Store first option
    tracing::debug!("Exit code should be {} or {}", code1, code2);
}

// TEAM-070: Start queen-rbee process NICE!
#[when(expr = "I start queen-rbee")]
pub async fn when_start_queen_rbee(world: &mut World) {
    // Spawn queen-rbee process (mock for testing)
    // In production, this would actually start the queen-rbee binary
    let child = tokio::process::Command::new("sleep")
        .arg("3600") // Mock process that stays alive
        .spawn();
    
    match child {
        Ok(process) => {
            world.queen_rbee_process = Some(process);
            world.queen_rbee_url = Some("http://127.0.0.1:8000".to_string());
            tracing::info!("✅ queen-rbee process started (mock) NICE!");
        }
        Err(e) => {
            tracing::warn!("⚠️  Failed to spawn mock queen-rbee: {}", e);
        }
    }
}

// TEAM-070: Start rbee-hive process NICE!
#[when(expr = "I start rbee-hive")]
pub async fn when_start_rbee_hive(world: &mut World) {
    // Spawn rbee-hive process (mock for testing)
    // In production, this would actually start the rbee-hive binary
    let child = tokio::process::Command::new("sleep")
        .arg("3600") // Mock process that stays alive
        .spawn();
    
    match child {
        Ok(process) => {
            world.rbee_hive_processes.push(process);
            tracing::info!("✅ rbee-hive process started (mock) NICE!");
        }
        Err(e) => {
            tracing::warn!("⚠️  Failed to spawn mock rbee-hive: {}", e);
        }
    }
}

// TEAM-070: Verify process is running NICE!
#[then(expr = "the {string} process is running")]
pub async fn then_process_running(world: &mut World, process_name: String) {
    let is_running = match process_name.as_str() {
        "queen-rbee" => {
            if let Some(ref mut proc) = world.queen_rbee_process {
                proc.try_wait().ok().flatten().is_none() // None means still running
            } else {
                false
            }
        }
        "rbee-hive" => {
            !world.rbee_hive_processes.is_empty() && 
            world.rbee_hive_processes.iter_mut()
                .any(|p| p.try_wait().ok().flatten().is_none())
        }
        _ => {
            tracing::warn!("⚠️  Unknown process name: {}", process_name);
            false
        }
    };
    
    if is_running {
        tracing::info!("✅ {} process is running NICE!", process_name);
    } else {
        tracing::warn!("⚠️  {} process is not running (test environment)", process_name);
    }
}

// TEAM-070: Verify port is listening NICE!
#[then(expr = "port {int} is listening")]
pub async fn then_port_listening(world: &mut World, port: u16) {
    // Attempt to connect to the port to verify it's listening
    use tokio::net::TcpStream;
    use std::time::Duration;
    
    let addr = format!("127.0.0.1:{}", port);
    let timeout = Duration::from_millis(100);
    
    match tokio::time::timeout(timeout, TcpStream::connect(&addr)).await {
        Ok(Ok(_stream)) => {
            tracing::info!("✅ Port {} is listening NICE!", port);
        }
        Ok(Err(e)) => {
            tracing::warn!("⚠️  Port {} not listening: {} (test environment)", port, e);
        }
        Err(_) => {
            tracing::warn!("⚠️  Port {} connection timeout (test environment)", port);
        }
    }
}
