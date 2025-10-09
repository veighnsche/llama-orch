//! Model management commands
//!
//! Created by: TEAM-022

use crate::cli::ModelsAction;
use anyhow::Result;
use colored::Colorize;
use pool_core::catalog::{ModelCatalog, ModelEntry};
use std::path::PathBuf;

pub fn handle(action: ModelsAction) -> Result<()> {
    match action {
        ModelsAction::Download { model } => download(model),
        ModelsAction::List => list(),
        ModelsAction::Catalog => catalog(),
        ModelsAction::Register { id, name, repo, architecture } => {
            register(id, name, repo, architecture)
        }
        ModelsAction::Unregister { id } => unregister(id),
    }
}

fn download(_model: String) -> Result<()> {
    println!("{}", "Model download not yet implemented".yellow());
    println!("This will be implemented in CP3");
    Ok(())
}

fn list() -> Result<()> {
    println!("{}", "Model list not yet implemented".yellow());
    println!("This will be implemented in CP2");
    Ok(())
}

fn catalog() -> Result<()> {
    let catalog_path = PathBuf::from(".test-models/catalog.json");

    let catalog = match ModelCatalog::load(&catalog_path) {
        Ok(c) => c,
        Err(_) => {
            println!(
                "{}",
                "No catalog found. Create one with 'llorch-pool models register'".yellow()
            );
            return Ok(());
        }
    };

    println!();
    println!("{}", format!("Model Catalog for {}", catalog.pool_id).bold());
    println!("{}", "=".repeat(80));
    println!(
        "{:<15} {:<30} {:<12} {:<10}",
        "ID".bold(),
        "Name".bold(),
        "Downloaded".bold(),
        "Size".bold()
    );
    println!("{}", "-".repeat(80));

    for model in &catalog.models {
        let status = if model.downloaded { "✅".green() } else { "❌".red() };
        println!("{:<15} {:<30} {:<12} {:.1} GB", model.id, model.name, status, model.size_gb);
    }

    println!("{}", "=".repeat(80));
    println!("Total models: {}\n", catalog.models.len());

    Ok(())
}

fn register(id: String, name: String, repo: String, architecture: String) -> Result<()> {
    let catalog_path = PathBuf::from(".test-models/catalog.json");

    let mut catalog = ModelCatalog::load(&catalog_path).unwrap_or_else(|_| {
        let pool_id = hostname::get().unwrap().to_string_lossy().to_string();
        ModelCatalog::new(pool_id)
    });

    let entry = ModelEntry {
        id: id.clone(),
        name,
        path: PathBuf::from(format!(".test-models/{}", id)),
        format: "safetensors".to_string(),
        size_gb: 0.0, // Will be updated after download
        architecture,
        downloaded: false,
        backends: vec!["cpu".to_string(), "metal".to_string(), "cuda".to_string()],
        metadata: serde_json::json!({
            "repo": repo,
        }),
    };

    catalog.add_model(entry)?;
    catalog.save(&catalog_path)?;

    println!("{}", format!("✅ Model {} registered", id).green());

    Ok(())
}

fn unregister(id: String) -> Result<()> {
    let catalog_path = PathBuf::from(".test-models/catalog.json");
    let mut catalog = ModelCatalog::load(&catalog_path)?;

    catalog.remove_model(&id)?;
    catalog.save(&catalog_path)?;

    println!("{}", format!("✅ Model {} unregistered", id).green());

    Ok(())
}
