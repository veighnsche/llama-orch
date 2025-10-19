//! Model management commands
//!
//! Created by: TEAM-022
//! Refactored by: TEAM-022 (using indicatif for progress)

use crate::cli::ModelsAction;
use anyhow::Result;
use colored::Colorize;
use hive_core::catalog::{ModelCatalog, ModelEntry};
use indicatif::{ProgressBar, ProgressStyle};
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

fn download(model_id: String) -> Result<()> {
    // TEAM-022: CP3 - Model download implementation
    let catalog_path = PathBuf::from(".test-models/catalog.json");
    let mut catalog = ModelCatalog::load(&catalog_path)?;

    // Find model in catalog and extract needed info
    let (repo, model_name, model_path) = {
        let model = catalog
            .find_model(&model_id)
            .ok_or_else(|| anyhow::anyhow!("Model {} not in catalog", model_id))?;

        if model.downloaded {
            println!("{}", format!("✅ Model {} already downloaded", model_id).green());
            return Ok(());
        }

        let repo = model.metadata["repo"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("No repo in metadata"))?
            .to_string();

        (repo, model.name.clone(), model.path.clone())
    };

    // Create progress spinner
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(ProgressStyle::default_spinner().template("{spinner:.cyan} {msg}").unwrap());
    spinner.set_message(format!("📥 Downloading {} from {}", model_name, repo));

    println!("   Target: {}", model_path.display());

    // Create target directory
    std::fs::create_dir_all(&model_path)?;

    // TEAM-023: DEPRECATED - huggingface-cli is deprecated, use `hf` CLI instead!
    // TODO: Replace with: Command::new("hf").args(["download", ...])
    // Install with: pip install huggingface_hub[cli]
    // The `hf` command is the modern replacement for `huggingface-cli`
    let status = std::process::Command::new("hf")
        .args([
            "download",
            &repo,
            "--include",
            "*.safetensors",
            "*.json",
            "tokenizer.model",
            "--local-dir",
            model_path.to_str().unwrap(),
        ])
        .status()?;

    spinner.finish_and_clear();

    if !status.success() {
        anyhow::bail!("Download failed for {}", model_id);
    }

    // Calculate actual size
    let size_bytes = calculate_dir_size(&model_path)?;
    let size_gb = size_bytes as f64 / 1_000_000_000.0;

    // Update catalog
    let model = catalog
        .find_model_mut(&model_id)
        .ok_or_else(|| anyhow::anyhow!("Model {} not in catalog", model_id))?;
    model.downloaded = true;
    model.size_gb = size_gb;

    catalog.save(&catalog_path)?;

    println!(
        "{}",
        format!("✅ Model {} downloaded successfully ({:.1} GB)", model_id, size_gb).green()
    );

    Ok(())
}

fn calculate_dir_size(path: &std::path::Path) -> Result<u64> {
    let mut total = 0;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            total += metadata.len();
        } else if metadata.is_dir() {
            total += calculate_dir_size(&entry.path())?;
        }
    }
    Ok(total)
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
                "No catalog found. Create one with 'rbee-hive models register'".yellow()
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
