use clap::Parser;
use provisioners_engine_provisioner::{provider_for, EngineProvisioner};
use std::path::PathBuf;

/// Engine Provisioner CLI — start engines per config (MVP)
#[derive(Parser, Debug)]
#[command(author, version, about, long_about=None)]
struct Args {
    /// Path to config file (YAML or JSON) conforming to contracts/config-schema
    #[arg(short, long)]
    config: PathBuf,

    /// Optional pool id to provision; if omitted, provisions the first llama.cpp pool
    #[arg(short, long)]
    pool: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let bytes = std::fs::read(&args.config)?;

    // Try YAML first, then JSON
    let cfg: contracts_config_schema::Config = match serde_yaml::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => serde_json::from_slice(&bytes)?,
    };

    let pool = if let Some(pid) = args.pool.as_deref() {
        cfg.pools
            .iter()
            .find(|p| p.id == pid)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("pool id '{}' not found in config", pid))?
    } else {
        cfg.pools
            .iter()
            .find(|p| matches!(p.engine, contracts_config_schema::Engine::Llamacpp))
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no llama.cpp pool found in config"))?
    };

    let prov = provider_for(&pool)?;
    prov.ensure(&pool)?;
    Ok(())
}
