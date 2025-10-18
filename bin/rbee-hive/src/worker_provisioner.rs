// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - No security issues found

//! Worker Provisioner Module
//!
//! Created by: TEAM-079
//!
//! Manages worker binary building and catalog.
//! Builds worker binaries from git with cargo and tracks them in SQLite.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct WorkerProvisioner {
    workspace_root: PathBuf,
}

impl WorkerProvisioner {
    /// Create a new worker provisioner
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
    }

    /// Build a worker binary with specified features
    pub fn build_worker(
        &self,
        worker_type: &str,
        features: &[String],
    ) -> Result<PathBuf> {
        tracing::info!("Building worker {} with features {:?}", worker_type, features);

        let mut cmd = Command::new("cargo");
        cmd.arg("build")
            .arg("--release")
            .arg("--bin")
            .arg(worker_type)
            .current_dir(&self.workspace_root);

        if !features.is_empty() {
            cmd.arg("--features").arg(features.join(","));
        }

        let output = cmd.output()
            .context("Failed to execute cargo build")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Build failed: {}", stderr);
        }

        let binary_path = self.workspace_root
            .join("target/release")
            .join(worker_type);

        Ok(binary_path)
    }

    /// Verify a binary is executable
    pub fn verify_binary(&self, path: &Path) -> Result<()> {
        if !path.exists() {
            anyhow::bail!("Binary not found: {:?}", path);
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = std::fs::metadata(path)
                .context("Failed to read binary metadata")?;
            let permissions = metadata.permissions();
            if permissions.mode() & 0o111 == 0 {
                anyhow::bail!("Binary is not executable");
            }
        }

        Ok(())
    }

    /// Get the expected binary path for a worker type
    pub fn binary_path(&self, worker_type: &str) -> PathBuf {
        self.workspace_root
            .join("target/release")
            .join(worker_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_provisioner_creation() {
        let root = PathBuf::from("/tmp/test");
        let provisioner = WorkerProvisioner::new(root.clone());
        assert_eq!(provisioner.workspace_root, root);
    }

    #[test]
    fn test_binary_path() {
        let root = PathBuf::from("/tmp/test");
        let provisioner = WorkerProvisioner::new(root);
        let path = provisioner.binary_path("llm-worker-rbee");
        assert_eq!(path, PathBuf::from("/tmp/test/target/release/llm-worker-rbee"));
    }
}
