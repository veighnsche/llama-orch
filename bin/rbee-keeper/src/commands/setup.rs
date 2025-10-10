//! Setup command handlers
//!
//! Created by: TEAM-043
//!
//! Manages rbee-hive node registry through queen-rbee

use anyhow::{Context, Result};
use colored::Colorize;
use serde::{Deserialize, Serialize};

use crate::cli::SetupAction;

const QUEEN_RBEE_URL: &str = "http://localhost:8080";

#[derive(Debug, Serialize)]
struct AddNodeRequest {
    node_name: String,
    ssh_host: String,
    ssh_port: u16,
    ssh_user: String,
    ssh_key_path: Option<String>,
    git_repo_url: String,
    git_branch: String,
    install_path: String,
}

#[derive(Debug, Deserialize)]
struct AddNodeResponse {
    success: bool,
    message: String,
    node_name: String,
}

#[derive(Debug, Deserialize)]
struct BeehiveNode {
    node_name: String,
    ssh_host: String,
    ssh_port: u16,
    ssh_user: String,
    ssh_key_path: Option<String>,
    git_repo_url: String,
    git_branch: String,
    install_path: String,
    last_connected_unix: Option<i64>,
    status: String,
}

#[derive(Debug, Deserialize)]
struct ListNodesResponse {
    nodes: Vec<BeehiveNode>,
}

#[derive(Debug, Deserialize)]
struct RemoveNodeResponse {
    success: bool,
    message: String,
}

pub async fn handle(action: SetupAction) -> Result<()> {
    match action {
        SetupAction::AddNode {
            name,
            ssh_host,
            ssh_port,
            ssh_user,
            ssh_key,
            git_repo,
            git_branch,
            install_path,
        } => {
            handle_add_node(
                name,
                ssh_host,
                ssh_port,
                ssh_user,
                ssh_key,
                git_repo,
                git_branch,
                install_path,
            )
            .await
        }
        SetupAction::ListNodes => handle_list_nodes().await,
        SetupAction::RemoveNode { name } => handle_remove_node(name).await,
        SetupAction::Install { node } => handle_install(node).await,
    }
}

async fn handle_add_node(
    name: String,
    ssh_host: String,
    ssh_port: u16,
    ssh_user: String,
    ssh_key: Option<String>,
    git_repo: String,
    git_branch: String,
    install_path: String,
) -> Result<()> {
    println!("{} Adding node '{}' to registry...", "[queen-rbee]".cyan(), name);

    let client = reqwest::Client::new();
    let url = format!("{}/v2/registry/beehives/add", QUEEN_RBEE_URL);

    let request = AddNodeRequest {
        node_name: name.clone(),
        ssh_host: ssh_host.clone(),
        ssh_port,
        ssh_user: ssh_user.clone(),
        ssh_key_path: ssh_key,
        git_repo_url: git_repo,
        git_branch,
        install_path,
    };

    println!("{} 🔌 Testing SSH connection to {}", "[queen-rbee]".cyan(), ssh_host);

    let response = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .context("Failed to send request to queen-rbee")?;

    let status = response.status();
    let result: AddNodeResponse =
        response.json().await.context("Failed to parse response from queen-rbee")?;

    if status.is_success() && result.success {
        println!(
            "{} ✅ SSH connection successful! Node '{}' saved to registry",
            "[queen-rbee]".cyan(),
            name
        );
        Ok(())
    } else {
        // TEAM-047: Fixed exit code - use anyhow::bail instead of std::process::exit
        println!("{} ❌ SSH connection failed: {}", "[queen-rbee]".cyan(), result.message);
        anyhow::bail!("SSH connection failed: {}", result.message)
    }
}

async fn handle_list_nodes() -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("{}/v2/registry/beehives/list", QUEEN_RBEE_URL);

    let response = client.get(&url).send().await.context("Failed to send request to queen-rbee")?;

    let result: ListNodesResponse =
        response.json().await.context("Failed to parse response from queen-rbee")?;

    if result.nodes.is_empty() {
        println!("{}", "No nodes registered".yellow());
        return Ok(());
    }

    println!("\n{}", "Registered rbee-hive nodes:".bold());
    println!("{}", "─".repeat(80));

    for node in result.nodes {
        println!("\n{}: {}", "Node".bold(), node.node_name.cyan());
        println!("  SSH: {}@{}:{}", node.ssh_user, node.ssh_host, node.ssh_port);
        if let Some(key) = node.ssh_key_path {
            println!("  Key: {}", key);
        }
        println!("  Git: {} ({})", node.git_repo_url, node.git_branch);
        println!("  Install: {}", node.install_path);
        println!(
            "  Status: {}",
            match node.status.as_str() {
                "reachable" => node.status.green(),
                "offline" => node.status.red(),
                _ => node.status.yellow(),
            }
        );
        if let Some(ts) = node.last_connected_unix {
            let dt = chrono::DateTime::from_timestamp(ts, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or_else(|| "unknown".to_string());
            println!("  Last connected: {}", dt);
        }
    }

    println!();
    Ok(())
}

async fn handle_remove_node(name: String) -> Result<()> {
    println!("{} Removing node '{}' from registry...", "[queen-rbee]".cyan(), name);

    let client = reqwest::Client::new();
    let url = format!("{}/v2/registry/beehives/remove", QUEEN_RBEE_URL);

    let response = client
        .post(&url)
        .json(&serde_json::json!({ "node_name": name }))
        .send()
        .await
        .context("Failed to send request to queen-rbee")?;

    let result: RemoveNodeResponse =
        response.json().await.context("Failed to parse response from queen-rbee")?;

    if result.success {
        println!("{} ✅ Node '{}' removed successfully", "[queen-rbee]".cyan(), name);
        Ok(())
    } else {
        // TEAM-047: Fixed exit code - use anyhow::bail instead of std::process::exit
        println!("{} ❌ {}", "[queen-rbee]".cyan(), result.message);
        anyhow::bail!("{}", result.message)
    }
}

async fn handle_install(node: String) -> Result<()> {
    println!("{} Installing rbee-hive on node '{}'...", "[queen-rbee]".cyan(), node);

    // First, get node details from registry
    let client = reqwest::Client::new();
    let url = format!("{}/v2/registry/beehives/list", QUEEN_RBEE_URL);

    let response =
        client.get(&url).send().await.context("Failed to get node list from queen-rbee")?;

    let result: ListNodesResponse =
        response.json().await.context("Failed to parse response from queen-rbee")?;

    let node_info = result
        .nodes
        .into_iter()
        .find(|n| n.node_name == node)
        .context(format!("Node '{}' not found in registry", node))?;

    println!(
        "{} 📡 Connecting to {}@{}",
        "[queen-rbee]".cyan(),
        node_info.ssh_user,
        node_info.ssh_host
    );

    // Execute installation commands via SSH
    let install_script = format!(
        r#"
cd {} && \
git clone {} . || git pull && \
git checkout {} && \
cargo build --release --bin rbee-hive
"#,
        node_info.install_path, node_info.git_repo_url, node_info.git_branch
    );

    println!("{} 🔧 Installing rbee-hive...", "[queen-rbee]".cyan());
    println!("{}", install_script.trim());

    // TODO: Actually execute SSH command
    // For now, just print what would be executed
    println!("{} ✅ Installation complete (simulated)", "[queen-rbee]".cyan());

    Ok(())
}
