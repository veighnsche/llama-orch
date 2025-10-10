// rbee-hive Registry setup step definitions
// Created by: TEAM-041

use cucumber::{given, when, then};
use crate::steps::world::World;

// Setup scenarios
#[given(expr = "queen-rbee is running")]
pub async fn given_queen_rbee_running(world: &mut World) {
    // Assumes queen-rbee is already running (for setup scenarios)
    // Uses default URL from Background if not specified
    if world.queen_rbee_url.is_none() {
        world.queen_rbee_url = Some("http://localhost:8080".to_string());
    }
    tracing::debug!("queen-rbee is running at: {:?}", world.queen_rbee_url);
}

#[given(expr = "the rbee-hive registry is empty")]
pub async fn given_registry_empty(world: &mut World) {
    world.beehive_nodes.clear();
    tracing::debug!("rbee-hive registry cleared");
}

#[given(expr = "node {string} is registered in rbee-hive registry")]
pub async fn given_node_in_registry(world: &mut World, node: String) {
    // Add a basic node entry
    world.beehive_nodes.insert(
        node.clone(),
        crate::steps::world::BeehiveNode {
            node_name: node.clone(),
            ssh_host: format!("{}.home.arpa", node),
            ssh_port: 22,
            ssh_user: "vince".to_string(),
            ssh_key_path: Some("/home/vince/.ssh/id_ed25519".to_string()),
            git_repo_url: "https://github.com/user/llama-orch.git".to_string(),
            git_branch: "main".to_string(),
            install_path: "/home/vince/rbee".to_string(),
            last_connected_unix: None,
            status: "reachable".to_string(),
        },
    );
    tracing::debug!("Node {} added to registry", node);
}

#[given(expr = "node {string} is registered in rbee-hive registry with SSH details")]
pub async fn given_node_in_registry_with_ssh(world: &mut World, node: String) {
    given_node_in_registry(world, node).await;
}

#[given(expr = "multiple nodes are registered in rbee-hive registry")]
pub async fn given_multiple_nodes_in_registry(world: &mut World) {
    given_node_in_registry(world, "mac".to_string()).await;
    given_node_in_registry(world, "workstation".to_string()).await;
    tracing::debug!("Multiple nodes registered");
}

#[given(expr = "the rbee-hive registry does not contain node {string}")]
pub async fn given_node_not_in_registry(world: &mut World, node: String) {
    world.beehive_nodes.remove(&node);
    tracing::debug!("Node {} removed from registry", node);
}

// Then steps for registry operations
#[then(expr = "rbee-keeper sends request to queen-rbee at {string}")]
pub async fn then_request_to_queen_rbee_registry(world: &mut World, url: String) {
    tracing::debug!("Should send request to: {}", url);
}

#[then(expr = "queen-rbee validates SSH connection with:")]
pub async fn then_validate_ssh_connection(world: &mut World, step: &cucumber::gherkin::Step) {
    let docstring = step.docstring.as_ref().expect("Expected a docstring");
    tracing::debug!("Should validate SSH with: {}", docstring.trim());
}

#[then(expr = "the SSH connection succeeds")]
pub async fn then_ssh_connection_succeeds(world: &mut World) {
    tracing::debug!("SSH connection should succeed");
}

#[then(expr = "the SSH connection fails with timeout")]
pub async fn then_ssh_connection_fails(world: &mut World) {
    tracing::debug!("SSH connection should fail");
}

#[then(expr = "queen-rbee saves node to rbee-hive registry:")]
pub async fn then_save_node_to_registry(world: &mut World, step: &cucumber::gherkin::Step) {
    let table = step.table.as_ref().expect("Expected a table");
    tracing::debug!("Should save node with {} fields", table.rows.len() - 1);
}

#[then(expr = "queen-rbee does NOT save node to registry")]
pub async fn then_do_not_save_node(world: &mut World) {
    tracing::debug!("Should NOT save node to registry");
}

#[then(expr = "rbee-keeper displays:")]
pub async fn then_display_output(world: &mut World, step: &cucumber::gherkin::Step) {
    let docstring = step.docstring.as_ref().expect("Expected a docstring");
    tracing::debug!("Should display: {}", docstring.trim());
}

#[then(expr = "queen-rbee loads SSH details from registry")]
pub async fn then_load_ssh_details(world: &mut World) {
    tracing::debug!("Should load SSH details from registry");
}

#[then(expr = "queen-rbee executes installation via SSH:")]
pub async fn then_execute_installation(world: &mut World, step: &cucumber::gherkin::Step) {
    let docstring = step.docstring.as_ref().expect("Expected a docstring");
    tracing::debug!("Should execute installation: {}", docstring.trim());
}

#[then(expr = "queen-rbee removes node from registry")]
pub async fn then_remove_node_from_registry(world: &mut World) {
    tracing::debug!("Should remove node from registry");
}

#[then(expr = "the query returns no results")]
pub async fn then_query_returns_no_results(world: &mut World) {
    tracing::debug!("Query should return no results");
}

#[then(expr = "queen-rbee attempts SSH connection")]
pub async fn then_attempt_ssh_connection(world: &mut World) {
    tracing::debug!("Should attempt SSH connection");
}
