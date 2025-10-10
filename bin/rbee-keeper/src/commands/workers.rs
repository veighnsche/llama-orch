//! Worker management commands
//!
//! Created by: TEAM-046

use anyhow::Result;
use colored::Colorize;
use serde::Deserialize;

use crate::cli::WorkersAction;

/// Handle workers command
///
/// TEAM-046: Implements worker management via queen-rbee
pub async fn handle(action: WorkersAction) -> Result<()> {
    match action {
        WorkersAction::List => list_workers().await,
        WorkersAction::Health { node } => check_health(node).await,
        WorkersAction::Shutdown { id } => shutdown_worker(id).await,
    }
}

#[derive(Deserialize)]
struct WorkerInfo {
    worker_id: String,
    node: String,
    state: String,
    model_ref: Option<String>,
    url: String,
}

#[derive(Deserialize)]
struct WorkersListResponse {
    workers: Vec<WorkerInfo>,
}

/// List all registered workers
async fn list_workers() -> Result<()> {
    println!("{}", "=== Registered Workers ===".cyan().bold());
    println!();

    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8080";

    let response = client
        .get(format!("{}/v2/workers/list", queen_url))
        .send()
        .await?;

    if !response.status().is_success() {
        anyhow::bail!("Failed to list workers: HTTP {}", response.status());
    }

    let workers_list: WorkersListResponse = response.json().await?;

    if workers_list.workers.is_empty() {
        println!("{}", "No workers registered".yellow());
        return Ok(());
    }

    for worker in workers_list.workers {
        println!("{} {}", "Worker ID:".bold(), worker.worker_id);
        println!("  Node:  {}", worker.node);
        println!("  State: {}", worker.state);
        if let Some(model) = worker.model_ref {
            println!("  Model: {}", model);
        }
        println!("  URL:   {}", worker.url);
        println!();
    }

    Ok(())
}

#[derive(Deserialize)]
struct HealthResponse {
    status: String,
    workers: Vec<WorkerHealthInfo>,
}

#[derive(Deserialize)]
struct WorkerHealthInfo {
    worker_id: String,
    state: String,
    ready: bool,
}

/// Check worker health on a specific node
async fn check_health(node: String) -> Result<()> {
    println!("{}", format!("=== Worker Health on {} ===", node).cyan().bold());
    println!();

    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8080";

    let response = client
        .get(format!("{}/v2/workers/health?node={}", queen_url, node))
        .send()
        .await?;

    if !response.status().is_success() {
        anyhow::bail!("Failed to check health: HTTP {}", response.status());
    }

    let health: HealthResponse = response.json().await?;

    println!("{} {}", "Status:".bold(), health.status);
    println!();

    if health.workers.is_empty() {
        println!("{}", "No workers on this node".yellow());
        return Ok(());
    }

    for worker in health.workers {
        let ready_icon = if worker.ready { "✓".green() } else { "✗".red() };
        println!("{} {} - {} (ready: {})", ready_icon, worker.worker_id, worker.state, worker.ready);
    }

    Ok(())
}

/// Shutdown a specific worker
async fn shutdown_worker(id: String) -> Result<()> {
    println!("{}", format!("=== Shutting Down Worker {} ===", id).cyan().bold());
    println!();

    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8080";

    let response = client
        .post(format!("{}/v2/workers/shutdown", queen_url))
        .json(&serde_json::json!({
            "worker_id": id
        }))
        .send()
        .await?;

    if !response.status().is_success() {
        anyhow::bail!("Failed to shutdown worker: HTTP {}", response.status());
    }

    println!("{} Worker shutdown command sent", "✓".green());
    println!("Worker will unload model and exit gracefully");

    Ok(())
}
