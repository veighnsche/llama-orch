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

use crate::steps::world::{NodeInfo, World};
use cucumber::given;

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

    tracing::debug!("Topology configured with {} nodes", world.topology.len());
}

#[given(expr = "I am on node {string}")]
pub async fn given_current_node(world: &mut World, node: String) {
    world.current_node = Some(node.clone());
    tracing::debug!("Current node set to: {}", node);
}

#[given(expr = "queen-rbee is running at {string}")]
pub async fn given_queen_rbee_url(world: &mut World, url: String) {
    // TEAM-051: Use the global queen-rbee instance (already started in main)
    // Just set the URL in the world - the instance is shared across all scenarios
    world.queen_rbee_url = Some(url.clone());
    tracing::debug!("Using global queen-rbee at: {}", url);
}

#[given(expr = "the model catalog is SQLite at {string}")]
pub async fn given_model_catalog_path(world: &mut World, path: String) {
    let expanded_path = shellexpand::tilde(&path).to_string();
    world.model_catalog_path = Some(expanded_path.clone().into());
    tracing::debug!("Model catalog path set to: {}", expanded_path);
}

#[given(expr = "the worker registry is in-memory ephemeral per node")]
pub async fn given_worker_registry_ephemeral(_world: &mut World) {
    // This is a documentation step - no state change needed
    tracing::debug!("Worker registry configured as in-memory ephemeral");
}

#[given(expr = "the rbee-hive registry is SQLite at {string}")]
pub async fn given_beehive_registry_path(world: &mut World, path: String) {
    let expanded_path = shellexpand::tilde(&path).to_string();
    world.registry_db_path = Some(expanded_path.clone());
    tracing::debug!("rbee-hive registry path set to: {}", expanded_path);
}
