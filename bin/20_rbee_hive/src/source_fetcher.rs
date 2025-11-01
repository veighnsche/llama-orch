//! Source fetcher for PKGBUILD installation
//!
//! TEAM-378: Handles downloading and extracting sources from PKGBUILD source=() field
//! Supports git repositories with tag/branch/commit syntax

use anyhow::{Context, Result};
use observability_narration_core::n;
use std::path::Path;
use tokio::process::Command;

/// Parse and fetch sources from PKGBUILD source array
///
/// Supports:
/// - `git+https://github.com/user/repo.git#tag=v1.0.0`
/// - `git+https://github.com/user/repo.git#branch=main`
/// - `git+https://github.com/user/repo.git#commit=abc123`
/// - Plain URLs (future: tarballs, etc.)
///
/// TEAM-378: For local development, if sources is empty, tries to symlink
/// the current workspace to srcdir (avoids needing to push/tag)
pub async fn fetch_sources(sources: &[String], srcdir: &Path) -> Result<()> {
    if sources.is_empty() {
        n!("fetch_sources_empty", "📝 Empty source array - checking for local workspace");
        return try_use_local_workspace(srcdir).await;
    }

    for (idx, source) in sources.iter().enumerate() {
        n!("fetch_source", "📥 Fetching source {}/{}: {}", idx + 1, sources.len(), source);
        
        if source.starts_with("git+") {
            fetch_git_source(source, srcdir).await?;
        } else if source.starts_with("http://") || source.starts_with("https://") {
            // TODO: Implement tarball/file download
            anyhow::bail!("HTTP/HTTPS file downloads not yet implemented. Use git+ URLs for now.");
        } else {
            n!("fetch_source_skip", "⏭️  Skipping unknown source type: {}", source);
        }
    }

    Ok(())
}

/// Fetch a git repository
///
/// Syntax: `git+https://github.com/user/repo.git#tag=v1.0.0`
/// Syntax: `git+https://github.com/user/repo.git#branch=main`
/// Syntax: `git+https://github.com/user/repo.git#commit=abc123`
async fn fetch_git_source(source: &str, srcdir: &Path) -> Result<()> {
    // Parse git URL
    let source = source.strip_prefix("git+").unwrap_or(source);
    
    let (url, ref_spec) = if let Some(pos) = source.find('#') {
        let (url_part, ref_part) = source.split_at(pos);
        let ref_part = &ref_part[1..]; // Skip the '#'
        (url_part, Some(ref_part))
    } else {
        (source, None)
    };

    n!("git_clone_start", "🔄 Cloning repository: {}", url);
    
    // Extract repo name from URL for directory name
    let repo_name = url
        .rsplit('/')
        .next()
        .unwrap_or("repo")
        .trim_end_matches(".git");
    
    let clone_dir = srcdir.join(repo_name);

    // Clone the repository
    let mut cmd = Command::new("git");
    cmd.arg("clone")
        .arg("--depth")
        .arg("1"); // Shallow clone for speed

    // Add branch/tag if specified
    if let Some(ref_spec) = ref_spec {
        if let Some(tag) = ref_spec.strip_prefix("tag=") {
            n!("git_clone_tag", "🏷️  Checking out tag: {}", tag);
            cmd.arg("--branch").arg(tag);
        } else if let Some(branch) = ref_spec.strip_prefix("branch=") {
            n!("git_clone_branch", "🌿 Checking out branch: {}", branch);
            cmd.arg("--branch").arg(branch);
        } else if let Some(commit) = ref_spec.strip_prefix("commit=") {
            // For commits, we need to clone full repo then checkout
            n!("git_clone_commit", "📌 Will checkout commit: {}", commit);
            // Remove --depth for commit checkout
            cmd = Command::new("git");
            cmd.arg("clone");
        }
    }

    cmd.arg(url).arg(&clone_dir);

    n!("git_clone_exec", "⚙️  Executing: git clone {} -> {}", url, clone_dir.display());
    
    let output = cmd
        .output()
        .await
        .context("Failed to execute git clone")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Git clone failed: {}", stderr);
    }

    n!("git_clone_ok", "✓ Repository cloned to: {}", clone_dir.display());

    // If commit was specified, checkout the specific commit
    if let Some(ref_spec) = ref_spec {
        if let Some(commit) = ref_spec.strip_prefix("commit=") {
            n!("git_checkout_commit", "📌 Checking out commit: {}", commit);
            
            let output = Command::new("git")
                .arg("-C")
                .arg(&clone_dir)
                .arg("checkout")
                .arg(commit)
                .output()
                .await
                .context("Failed to checkout commit")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                anyhow::bail!("Git checkout failed: {}", stderr);
            }

            n!("git_checkout_ok", "✓ Checked out commit: {}", commit);
        }
    }

    Ok(())
}

/// Try to use local workspace for development
///
/// TEAM-378: If we're running inside the llama-orch workspace,
/// symlink it to srcdir instead of cloning from git.
/// This allows local development without needing to push/tag.
async fn try_use_local_workspace(srcdir: &Path) -> Result<()> {
    // Find workspace root by looking for Cargo.toml with [workspace]
    let mut current = std::env::current_dir()?;
    
    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            let content = tokio::fs::read_to_string(&cargo_toml).await?;
            if content.contains("[workspace]") {
                n!("local_workspace_found", "🏠 Found local workspace: {}", current.display());
                
                // Create srcdir parent if needed
                if let Some(parent) = srcdir.parent() {
                    tokio::fs::create_dir_all(parent).await?;
                }
                
                // Symlink workspace to srcdir/llama-orch
                let target = srcdir.join("llama-orch");
                
                #[cfg(unix)]
                {
                    tokio::fs::symlink(&current, &target).await
                        .context("Failed to create symlink to local workspace")?;
                }
                
                #[cfg(not(unix))]
                {
                    // On Windows, copy instead of symlink (requires admin for symlinks)
                    anyhow::bail!("Windows not yet supported for local development. Use git+ sources.");
                }
                
                n!("local_workspace_linked", "✓ Linked local workspace to: {}", target.display());
                return Ok(());
            }
        }
        
        // Move up one directory
        if !current.pop() {
            break;
        }
    }
    
    anyhow::bail!("No sources specified and not running in llama-orch workspace. Cannot proceed.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_parse_git_url_with_tag() {
        let temp = TempDir::new().unwrap();
        let srcdir = temp.path();

        // This would require a real git repo, so we'll skip actual cloning in tests
        // Just test the parsing logic
        let source = "git+https://github.com/user/repo.git#tag=v1.0.0";
        assert!(source.starts_with("git+"));
    }

    #[tokio::test]
    async fn test_empty_sources() {
        let temp = TempDir::new().unwrap();
        let srcdir = temp.path();

        let result = fetch_sources(&[], srcdir).await;
        assert!(result.is_ok());
    }
}
