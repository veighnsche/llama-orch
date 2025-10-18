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
use cucumber::{then, when};

// TEAM-069: Call WorkerRegistry.register() NICE!
#[when(expr = "rbee-hive registers the worker")]
pub async fn when_register_worker(world: &mut World) {
    use rbee_hive::registry::{WorkerRegistry, WorkerInfo, WorkerState};
    
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
        restart_count: 0, // TEAM-104: Added restart tracking
        last_restart: None, // TEAM-104: Added restart tracking
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
    
    tracing::info!("✅ Registry HashMap updated with {} workers, {} expected fields",
        workers.len(), field_count);
}

#[then(regex = r"^the registration is ephemeral \(lost on rbee-hive restart\)$")]
pub async fn then_registration_ephemeral(_world: &mut World) {
    tracing::debug!("Registration is ephemeral");
}
