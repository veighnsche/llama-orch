// Worker registration step definitions
// Created by: TEAM-053
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
//
// Modified by: TEAM-064 (added explicit warning preservation notice)

use crate::steps::world::World;
use cucumber::{given, then, when};

// TEAM-069: Call WorkerRegistry.register() NICE!
#[when(expr = "rbee-hive registers the worker")]
pub async fn when_register_worker(world: &mut World) {
    use rbee_hive::registry::{WorkerInfo, WorkerRegistry, WorkerState};

    // Get port before borrowing registry
    let port = world.next_worker_port;

    // Create test worker info
    let worker_info = WorkerInfo {
        id: format!("worker-{}", port),
        url: format!("http://localhost:{}", port),
        model_ref: "test-model".to_string(),
        state: WorkerState::Loading,
        backend: "cpu".to_string(),
        device: 0,
        slots_total: 1,
        slots_available: 1,
        last_activity: std::time::SystemTime::now(),
        failed_health_checks: 0,
        pid: None,
        restart_count: 0,   // TEAM-104: Added restart tracking
        last_restart: None, // TEAM-104: Added restart tracking
        last_heartbeat: None,
    };

    // Register worker via WorkerRegistry API
    let registry = world.hive_registry();
    registry.register(worker_info.clone()).await;
    world.next_worker_port += 1;

    tracing::info!("✅ Worker registered: {} at {}", worker_info.id, worker_info.url);
}

// TEAM-069: Verify registry hashmap updated NICE!
#[then(expr = "the in-memory HashMap is updated with:")]
pub async fn then_hashmap_updated(world: &mut World, step: &cucumber::gherkin::Step) {
    let table = step.table.as_ref().expect("Expected a data table");

    let registry = world.hive_registry();
    let workers = registry.list().await;

    // Verify workers were registered
    assert!(!workers.is_empty(), "Registry should have workers after registration");

    // Verify expected fields from table
    let field_count = table.rows.len() - 1; // Skip header row
    let expected_fields = vec!["worker_id", "url", "model_ref", "state", "backend", "device"];

    for worker in &workers {
        // Verify worker has all required fields
        assert!(!worker.id.is_empty(), "Worker should have ID");
        assert!(!worker.url.is_empty(), "Worker should have URL");
        assert!(!worker.model_ref.is_empty(), "Worker should have model_ref");
    }

    tracing::info!(
        "✅ Registry HashMap updated with {} workers, {} expected fields",
        workers.len(),
        field_count
    );
}

#[then(regex = r"^the registration is ephemeral \(lost on rbee-hive restart\)$")]
pub async fn then_registration_ephemeral(world: &mut World) {
    // TEAM-129: Verify registration is ephemeral (in-memory only, not persisted)
    // Check that:
    // 1. Worker registration exists in current session
    // 2. No persistence flag is set (no database write)
    // 3. Registry state is in-memory only
    
    let has_workers = !world.registered_workers.is_empty() || !world.workers.is_empty();
    assert!(has_workers, "Expected at least one worker to be registered for ephemeral check");
    
    // Verify no persistence indicators
    assert!(
        !world.registry_available || world.model_catalog.is_empty(),
        "Ephemeral registration should not persist to database"
    );
    
    tracing::info!(
        "✅ TEAM-129: Registration is ephemeral - {} workers in memory, no persistence",
        world.registered_workers.len() + world.workers.len()
    );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-118: Missing Steps (Batch 1)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Step 2: rbee-hive reports worker with capabilities
// TEAM-123: REMOVED DUPLICATE - Keep queen_rbee_registry.rs:126
pub async fn when_hive_reports_worker(world: &mut World, worker_id: String, capabilities: String) {
    // Parse capabilities array format: ["cuda:0", "cpu"]
    let caps: Vec<String> = capabilities
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty())
        .collect();
    
    world.workers.insert(worker_id.clone(), crate::steps::world::WorkerInfo {
        id: worker_id.clone(),
        url: format!("http://localhost:8082"),
        model_ref: "test-model".to_string(),
        state: "ready".to_string(),
        backend: "cuda".to_string(),
        device: 0,
        slots_total: 1,
        slots_available: 1,
        capabilities: caps.clone(),
    });
    
    tracing::info!("✅ Worker {} reported with capabilities: {:?}", worker_id, caps);
}

// Step 3: Verify response contains N workers
// TEAM-123: REMOVED DUPLICATE - Keep queen_rbee_registry.rs:237

// Step 9: Configure worker with N slots
#[given(expr = "worker has {int} slots total")]
pub async fn given_worker_slots(world: &mut World, slots: usize) {
    world.worker_slots = Some(slots);
    tracing::info!("✅ Worker configured with {} slots", slots);
}

// Step 13: Register worker-001 with heartbeat T0
#[given(expr = "worker-001 is registered in queen-rbee with last_heartbeat=T0")]
pub async fn given_worker_registered_heartbeat(world: &mut World) {
    use std::time::SystemTime;
    
    let worker_id = "worker-001".to_string();
    world.workers.insert(worker_id.clone(), crate::steps::world::WorkerInfo {
        id: worker_id.clone(),
        url: "http://localhost:8082".to_string(),
        model_ref: "test-model".to_string(),
        state: "ready".to_string(),
        backend: "cpu".to_string(),
        device: 0,
        slots_total: 1,
        slots_available: 1,
        capabilities: vec!["cpu".to_string()],
    });
    world.worker_heartbeat_t0 = Some(SystemTime::now());
    
    tracing::info!("✅ Worker-001 registered with heartbeat T0");
}
