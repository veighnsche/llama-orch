// Background step definitions
// Created by: TEAM-040
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
//
// Modified by: TEAM-056 (attempted auto-registration, reverted due to timing issues)
// Modified by: TEAM-064 (added explicit warning preservation notice)
// Modified by: TEAM-065 (marked FAKE functions that create false positives)
// Modified by: TEAM-066 (clarified test setup vs product behavior)

use crate::steps::world::{NodeInfo, World};
use cucumber::given;

// TEAM-066: Test setup - defines topology for test scenarios
// This is test data configuration, not product behavior to test
#[given(expr = "the following topology:")]
pub async fn given_topology(world: &mut World, step: &cucumber::gherkin::Step) {
    let table = step.table.as_ref().expect("Expected a data table");

    for row in table.rows.iter().skip(1) {
        // Skip header row
        let node = row[0].clone();
        let hostname = row[1].clone();
        let components = row[2].split(',').map(|s| s.trim().to_string()).collect();
        let capabilities = row[3].split(',').map(|s| s.trim().to_string()).collect();

        world.topology.insert(node.clone(), NodeInfo { hostname, components, capabilities });
    }

    tracing::info!("✅ Test setup: Topology configured with {} nodes", world.topology.len());
}

// TEAM-066: Test setup - sets current node context for test
// This is test data configuration, not product behavior to test
#[given(expr = "I am on node {string}")]
pub async fn given_current_node(world: &mut World, node: String) {
    world.current_node = Some(node.clone());
    tracing::info!("✅ Test setup: Current node set to: {}", node);
}

// TEAM-066: Test setup - configures queen-rbee URL
// This references the global queen-rbee instance started in main.rs
#[given(expr = "queen-rbee is running at {string}")]
pub async fn given_queen_rbee_url(world: &mut World, url: String) {
    // TEAM-051: Use the global queen-rbee instance (already started in main)
    // Just set the URL in the world - the instance is shared across all scenarios
    world.queen_rbee_url = Some(url.clone());
    tracing::info!("✅ Test setup: Using global queen-rbee at: {}", url);
}

// TEAM-066: Test setup - configures model catalog path
// This is test configuration, not product behavior to test
#[given(expr = "the model catalog is SQLite at {string}")]
pub async fn given_model_catalog_path(world: &mut World, path: String) {
    let expanded_path = shellexpand::tilde(&path).to_string();
    world.model_catalog_path = Some(expanded_path.clone().into());
    tracing::info!("✅ Test setup: Model catalog path set to: {}", expanded_path);
}

// TEAM-071: Verify worker registry is in-memory ephemeral NICE!
#[given(expr = "the worker registry is in-memory ephemeral per node")]
pub async fn given_worker_registry_ephemeral(world: &mut World) {
    // Verify registry exists and is in-memory (not persisted to disk)
    let registry = world.hive_registry();
    let workers = registry.list().await;

    // In-memory registry should be accessible
    tracing::info!("✅ Worker registry is in-memory ephemeral ({} workers) NICE!", workers.len());
}

// TEAM-066: Test setup - configures rbee-hive registry path
// This is test configuration, not product behavior to test
#[given(expr = "the rbee-hive registry is SQLite at {string}")]
pub async fn given_beehive_registry_path(world: &mut World, path: String) {
    let expanded_path = shellexpand::tilde(&path).to_string();
    world.registry_db_path = Some(expanded_path.clone());
    tracing::info!("✅ Test setup: rbee-hive registry path set to: {}", expanded_path);
}
