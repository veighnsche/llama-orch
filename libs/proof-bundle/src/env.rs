use anyhow::{Context, Result};
use std::env;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Resolve the base directory for proof bundles.
pub fn proof_base_dir() -> Result<PathBuf> {
    if let Ok(dir) = env::var("LLORCH_PROOF_DIR") {
        return Ok(PathBuf::from(dir));
    }
    // Default: current working directory (Cargo runs tests with CWD at the crate root)
    let cwd = std::env::current_dir().context("resolve current_dir")?;
    Ok(cwd.join(".proof_bundle"))
}

/// Resolve the run id, honoring LLORCH_RUN_ID or generating a timestamp(-sha8) id.
pub fn resolve_run_id() -> String {
    if let Ok(id) = env::var("LLORCH_RUN_ID") {
        return id;
    }
    let ts = epoch_seconds();
    if let Some(sha8) = resolve_sha8() {
        format!("{}-{}", ts, sha8)
    } else {
        ts
    }
}

fn epoch_seconds() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    format!("{}", secs)
}

fn resolve_sha8() -> Option<String> {
    // Prefer explicit envs often set by CI
    let cand = env::var("GIT_SHA")
        .ok()
        .or_else(|| env::var("CI_COMMIT_SHA").ok())
        .or_else(|| env::var("GITHUB_SHA").ok());
    cand.map(|s| s.chars().take(8).collect())
}
