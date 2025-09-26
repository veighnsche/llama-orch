use anyhow::{Context, Result};
use std::env;
use std::path::PathBuf;

/// Resolve the base directory for proof bundles.
pub fn proof_base_dir() -> Result<PathBuf> {
    if let Ok(dir) = env::var("LLORCH_PROOF_DIR") {
        return Ok(PathBuf::from(dir));
    }
    // Default: current working directory (Cargo runs tests with CWD at the crate root)
    let cwd = std::env::current_dir().context("resolve current_dir")?;
    Ok(cwd.join(".proof_bundle"))
}
