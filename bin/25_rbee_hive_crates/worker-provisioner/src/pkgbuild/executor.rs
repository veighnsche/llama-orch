// PKGBUILD executor for worker installation
// Executes build and package functions from parsed PKGBUILD files
// TEAM-402: Migrated to worker-provisioner crate

use super::parser::PkgBuild;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio_util::sync::CancellationToken; // TEAM-388: For cancellable builds

/// PKGBUILD execution errors
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    /// IO error during script execution
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Build function failed with non-zero exit code
    #[error("Build failed with exit code {0}")]
    BuildFailed(i32),

    /// Package function failed with non-zero exit code
    #[error("Package failed with exit code {0}")]
    PackageFailed(i32),

    /// PKGBUILD does not contain a build() function
    #[error("Missing build function in PKGBUILD")]
    MissingBuildFunction,

    /// PKGBUILD does not contain a package() function
    #[error("Missing package function in PKGBUILD")]
    MissingPackageFunction,

    /// Source directory does not exist
    #[error("Source directory not found: {0}")]
    SourceDirNotFound(String),
}

/// PKGBUILD executor
///
/// Executes build() and package() functions from PKGBUILD files.
/// Runs commands in a shell environment with proper variables set.
#[derive(Debug)]
pub struct PkgBuildExecutor {
    /// Source directory (where source code is extracted)
    srcdir: PathBuf,

    /// Package directory (where files are installed)
    pkgdir: PathBuf,

    /// Working directory (where build happens)
    workdir: PathBuf,
}

impl PkgBuildExecutor {
    /// Create a new executor
    ///
    /// # Arguments
    /// * `srcdir` - Source directory (e.g., /tmp/build/src)
    /// * `pkgdir` - Package directory (e.g., /tmp/build/pkg)
    /// * `workdir` - Working directory (e.g., /tmp/build)
    pub fn new(srcdir: PathBuf, pkgdir: PathBuf, workdir: PathBuf) -> Self {
        Self { srcdir, pkgdir, workdir }
    }

    /// Execute the build() function from PKGBUILD
    ///
    /// Streams output line-by-line to the callback function.
    ///
    /// # Arguments
    /// * `pkgbuild` - Parsed PKGBUILD
    /// * `output_callback` - Called for each line of output
    pub async fn build<F>(
        &self,
        pkgbuild: &PkgBuild,
        mut output_callback: F,
    ) -> Result<(), ExecutionError>
    where
        F: FnMut(&str),
    {
        let build_fn = pkgbuild.build_fn.as_ref().ok_or(ExecutionError::MissingBuildFunction)?;

        output_callback(&format!("==> Building {} {}...", pkgbuild.pkgname, pkgbuild.pkgver));

        // Create shell script with PKGBUILD variables
        let script = self.create_build_script(pkgbuild, build_fn);

        // Execute script
        self.execute_script(&script, "build", &mut output_callback).await?;

        output_callback(&format!("==> Build complete: {}", pkgbuild.pkgname));
        Ok(())
    }

    /// Execute the build() function from PKGBUILD with cancellation support
    ///
    /// TEAM-388: Cancellable version of build() - checks cancellation token periodically
    /// This is critical for cargo builds which can take minutes.
    ///
    /// # Arguments
    /// * `pkgbuild` - Parsed PKGBUILD
    /// * `cancel_token` - Cancellation token to check
    /// * `output_callback` - Called for each line of output
    pub async fn build_with_cancellation<F>(
        &self,
        pkgbuild: &PkgBuild,
        cancel_token: CancellationToken,
        mut output_callback: F,
    ) -> Result<(), ExecutionError>
    where
        F: FnMut(&str),
    {
        let build_fn = pkgbuild.build_fn.as_ref().ok_or(ExecutionError::MissingBuildFunction)?;

        output_callback(&format!(
            "==> Building {} {} (cancellable)...",
            pkgbuild.pkgname, pkgbuild.pkgver
        ));

        // Create shell script with PKGBUILD variables
        let script = self.create_build_script(pkgbuild, build_fn);

        // Execute script with cancellation support
        self.execute_script_with_cancellation(&script, "build", cancel_token, &mut output_callback)
            .await?;

        output_callback(&format!("==> Build complete: {}", pkgbuild.pkgname));
        Ok(())
    }

    /// Execute the package() function from PKGBUILD
    ///
    /// Streams output line-by-line to the callback function.
    ///
    /// # Arguments
    /// * `pkgbuild` - Parsed PKGBUILD
    /// * `output_callback` - Called for each line of output
    pub async fn package<F>(
        &self,
        pkgbuild: &PkgBuild,
        mut output_callback: F,
    ) -> Result<(), ExecutionError>
    where
        F: FnMut(&str),
    {
        let package_fn =
            pkgbuild.package_fn.as_ref().ok_or(ExecutionError::MissingPackageFunction)?;

        output_callback(&format!("==> Packaging {} {}...", pkgbuild.pkgname, pkgbuild.pkgver));

        // Create shell script with PKGBUILD variables
        let script = self.create_package_script(pkgbuild, package_fn);

        // Execute script
        self.execute_script(&script, "package", &mut output_callback).await?;

        output_callback(&format!("==> Package complete: {}", pkgbuild.pkgname));
        Ok(())
    }

    /// Execute both build() and package() functions
    ///
    /// Convenience method that runs the full build pipeline.
    pub async fn build_and_package<F>(
        &self,
        pkgbuild: &PkgBuild,
        mut output_callback: F,
    ) -> Result<(), ExecutionError>
    where
        F: FnMut(&str),
    {
        self.build(pkgbuild, &mut output_callback).await?;
        self.package(pkgbuild, &mut output_callback).await?;
        Ok(())
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // PRIVATE HELPERS
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Create a shell script for the build() function
    fn create_build_script(&self, pkgbuild: &PkgBuild, build_fn: &str) -> String {
        format!(
            r#"#!/bin/bash
set -e  # Exit on error
set -u  # Exit on undefined variable

# PKGBUILD variables
export pkgname="{pkgname}"
export pkgver="{pkgver}"
export pkgrel="{pkgrel}"
export srcdir="{srcdir}"
export pkgdir="{pkgdir}"

# Build function
{build_fn}
"#,
            pkgname = pkgbuild.pkgname,
            pkgver = pkgbuild.pkgver,
            pkgrel = pkgbuild.pkgrel,
            srcdir = self.srcdir.display(),
            pkgdir = self.pkgdir.display(),
            build_fn = build_fn,
        )
    }

    /// Create a shell script for the package() function
    fn create_package_script(&self, pkgbuild: &PkgBuild, package_fn: &str) -> String {
        format!(
            r#"#!/bin/bash
set -e  # Exit on error
set -u  # Exit on undefined variable

# PKGBUILD variables
export pkgname="{pkgname}"
export pkgver="{pkgver}"
export pkgrel="{pkgrel}"
export srcdir="{srcdir}"
export pkgdir="{pkgdir}"

# Package function
{package_fn}
"#,
            pkgname = pkgbuild.pkgname,
            pkgver = pkgbuild.pkgver,
            pkgrel = pkgbuild.pkgrel,
            srcdir = self.srcdir.display(),
            pkgdir = self.pkgdir.display(),
            package_fn = package_fn,
        )
    }

    /// Execute a shell script and stream output with cancellation support
    ///
    /// TEAM-388: Cancellable version that kills the child process when cancelled
    async fn execute_script_with_cancellation<F>(
        &self,
        script: &str,
        phase: &str,
        cancel_token: CancellationToken,
        output_callback: &mut F,
    ) -> Result<(), ExecutionError>
    where
        F: FnMut(&str),
    {
        // Write script to temp file
        let script_path = self.workdir.join(format!("{}.sh", phase));
        tokio::fs::write(&script_path, script).await?;

        // Make executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = tokio::fs::metadata(&script_path).await?.permissions();
            perms.set_mode(0o755);
            tokio::fs::set_permissions(&script_path, perms).await?;
        }

        // Execute script
        // TEAM-388: Create new process group so we can kill all subprocesses (including cargo)
        let mut cmd = Command::new("bash");
        cmd.arg(&script_path)
            .current_dir(&self.workdir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // TEAM-388: On Unix, create a new process group
        #[cfg(unix)]
        {
            cmd.process_group(0); // 0 = create new process group with PID as PGID
        }

        let mut child = cmd.spawn()?;

        let stdout = child.stdout.take().expect("Failed to capture stdout");
        let stderr = child.stderr.take().expect("Failed to capture stderr");

        // Stream output in REAL-TIME through callback
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let tx_stdout = tx.clone();
        let stdout_task = tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();

            while let Ok(Some(line)) = lines.next_line().await {
                let _ = tx_stdout.send(line);
            }
        });

        let tx_stderr = tx.clone();
        let stderr_task = tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();

            while let Ok(Some(line)) = lines.next_line().await {
                if line.to_lowercase().contains("error:") || line.to_lowercase().contains("failed")
                {
                    let _ = tx_stderr.send(format!("ERROR: {}", line));
                } else {
                    let _ = tx_stderr.send(line);
                }
            }
        });

        // Drop original tx so channel closes when tasks finish
        drop(tx);

        // TEAM-388: Stream output and check for cancellation
        loop {
            tokio::select! {
                // Check for cancellation
                _ = cancel_token.cancelled() => {
                    output_callback("==> Build cancelled, killing process group...");

                    // TEAM-421: Kill the entire process group (bash + cargo + all subprocesses)
                    // Using nix crate for safe Unix syscalls (no unsafe blocks needed)
                    #[cfg(unix)]
                    {
                        use nix::sys::signal::{killpg, Signal};
                        use nix::unistd::Pid;

                        if let Some(pid) = child.id() {
                            output_callback(&format!("==> Killing process group PGID: {}", pid));

                            // Send SIGTERM to process group (safe wrapper)
                            if let Err(e) = killpg(Pid::from_raw(pid as i32), Signal::SIGTERM) {
                                output_callback(&format!("==> Warning: SIGTERM failed: {}", e));
                            } else {
                                output_callback("==> SIGTERM sent to process group");
                            }

                            // Give it a moment to terminate gracefully
                            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

                            // If still running, send SIGKILL
                            if let Err(e) = killpg(Pid::from_raw(pid as i32), Signal::SIGKILL) {
                                output_callback(&format!("==> Warning: SIGKILL failed: {}", e));
                            } else {
                                output_callback("==> SIGKILL sent to process group");
                            }
                        }
                    }

                    // Also try tokio's kill (for the parent process)
                    let _ = child.kill().await;

                    // Wait for tasks to complete
                    let _ = tokio::join!(stdout_task, stderr_task);
                    return Err(ExecutionError::BuildFailed(-1));
                }
                // Receive output
                line = rx.recv() => {
                    match line {
                        Some(line) => output_callback(&line),
                        None => break, // Channel closed, process finished
                    }
                }
            }
        }

        // Wait for tasks to complete
        let _ = tokio::join!(stdout_task, stderr_task);

        // Wait for completion
        let status = child.wait().await?;

        if !status.success() {
            let code = status.code().unwrap_or(-1);
            return Err(match phase {
                "build" => ExecutionError::BuildFailed(code),
                "package" => ExecutionError::PackageFailed(code),
                _ => ExecutionError::BuildFailed(code),
            });
        }

        Ok(())
    }

    /// Execute a shell script and stream output
    async fn execute_script<F>(
        &self,
        script: &str,
        phase: &str,
        output_callback: &mut F,
    ) -> Result<(), ExecutionError>
    where
        F: FnMut(&str),
    {
        // Write script to temp file
        let script_path = self.workdir.join(format!("{}.sh", phase));
        tokio::fs::write(&script_path, script).await?;

        // Make executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = tokio::fs::metadata(&script_path).await?.permissions();
            perms.set_mode(0o755);
            tokio::fs::set_permissions(&script_path, perms).await?;
        }

        // Execute script
        let mut child = Command::new("bash")
            .arg(&script_path)
            .current_dir(&self.workdir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdout = child.stdout.take().expect("Failed to capture stdout");
        let stderr = child.stderr.take().expect("Failed to capture stderr");

        // TEAM-378: Stream output in REAL-TIME through callback
        // We need to use channels to send output from tasks back to main thread
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let tx_stdout = tx.clone();
        let stdout_task = tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();

            while let Ok(Some(line)) = lines.next_line().await {
                let _ = tx_stdout.send(line);
            }
        });

        let tx_stderr = tx.clone();
        let stderr_task = tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();

            // TEAM-384: Don't prefix normal cargo output with "ERROR:"
            // Cargo uses stderr for progress output, not just errors
            while let Ok(Some(line)) = lines.next_line().await {
                // Only prefix actual error lines (contain "error:" or "failed")
                if line.to_lowercase().contains("error:") || line.to_lowercase().contains("failed")
                {
                    let _ = tx_stderr.send(format!("ERROR: {}", line));
                } else {
                    // Normal cargo output (e.g., "Compiling X")
                    let _ = tx_stderr.send(line);
                }
            }
        });

        // Drop original tx so channel closes when tasks finish
        drop(tx);

        // Stream output in real-time as it arrives
        while let Some(line) = rx.recv().await {
            output_callback(&line);
        }

        // Wait for tasks to complete
        let _ = tokio::join!(stdout_task, stderr_task);

        // Wait for completion
        let status = child.wait().await?;

        if !status.success() {
            let code = status.code().unwrap_or(-1);
            return Err(match phase {
                "build" => ExecutionError::BuildFailed(code),
                "package" => ExecutionError::PackageFailed(code),
                _ => ExecutionError::BuildFailed(code),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Script generation with proper variable substitution
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[tokio::test]
    async fn test_build_script_contains_all_pkgbuild_variables() {
        let temp = TempDir::new().unwrap();
        let srcdir = temp.path().join("src");
        let pkgdir = temp.path().join("pkg");
        let workdir = temp.path().to_path_buf();

        let executor = PkgBuildExecutor::new(srcdir.clone(), pkgdir.clone(), workdir);

        let pkgbuild = PkgBuild {
            pkgname: "test-worker".to_string(),
            pkgver: "2.5.1".to_string(),
            pkgrel: "3".to_string(),
            pkgdesc: String::new(),
            arch: vec![],
            license: vec![],
            depends: vec![],
            makedepends: vec![],
            source: vec![],
            source_x86_64: vec![],
            source_aarch64: vec![],
            sha256sums: vec![],
            sha256sums_x86_64: vec![],
            sha256sums_aarch64: vec![],
            noextract: vec![],
            build_fn: Some("echo 'test'".to_string()),
            package_fn: None,
            variables: Default::default(),
        };

        let script = executor.create_build_script(&pkgbuild, pkgbuild.build_fn.as_ref().unwrap());

        // Verify all PKGBUILD variables are set
        assert!(script.contains("pkgname=\"test-worker\""));
        assert!(script.contains("pkgver=\"2.5.1\""));
        assert!(script.contains("pkgrel=\"3\""));
        assert!(script.contains(&format!("srcdir=\"{}\"", srcdir.display())));
        assert!(script.contains(&format!("pkgdir=\"{}\"", pkgdir.display())));

        // Verify bash safety flags
        assert!(script.contains("set -e")); // Exit on error
        assert!(script.contains("set -u")); // Exit on undefined variable
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Successful build execution streams output
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[tokio::test]
    async fn test_successful_build_streams_output_to_callback() {
        let temp = TempDir::new().unwrap();
        let srcdir = temp.path().join("src");
        let pkgdir = temp.path().join("pkg");
        let workdir = temp.path().to_path_buf();

        tokio::fs::create_dir_all(&srcdir).await.unwrap();
        tokio::fs::create_dir_all(&pkgdir).await.unwrap();

        let executor = PkgBuildExecutor::new(srcdir, pkgdir, workdir);

        let pkgbuild = PkgBuild {
            pkgname: "test-pkg".to_string(),
            pkgver: "1.0.0".to_string(),
            pkgrel: "1".to_string(),
            pkgdesc: String::new(),
            arch: vec![],
            license: vec![],
            depends: vec![],
            makedepends: vec![],
            source: vec![],
            source_x86_64: vec![],
            source_aarch64: vec![],
            sha256sums: vec![],
            sha256sums_x86_64: vec![],
            sha256sums_aarch64: vec![],
            noextract: vec![],
            build_fn: Some(
                r#"
                echo "Step 1: Downloading dependencies"
                echo "Step 2: Compiling source"
                echo "Step 3: Running tests"
            "#
                .to_string(),
            ),
            package_fn: None,
            variables: Default::default(),
        };

        let mut output = Vec::new();
        let result = executor
            .build(&pkgbuild, |line| {
                output.push(line.to_string());
            })
            .await;

        assert!(result.is_ok(), "Build should succeed");

        // Verify all output lines were captured
        assert!(output.iter().any(|line| line.contains("Step 1")));
        assert!(output.iter().any(|line| line.contains("Step 2")));
        assert!(output.iter().any(|line| line.contains("Step 3")));

        // Verify progress messages
        assert!(output.iter().any(|line| line.contains("Building test-pkg")));
        assert!(output.iter().any(|line| line.contains("Build complete")));
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Failed build returns error with exit code
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[tokio::test]
    async fn test_build_failure_returns_error_with_exit_code() {
        let temp = TempDir::new().unwrap();
        let srcdir = temp.path().join("src");
        let pkgdir = temp.path().join("pkg");
        let workdir = temp.path().to_path_buf();

        tokio::fs::create_dir_all(&srcdir).await.unwrap();
        tokio::fs::create_dir_all(&pkgdir).await.unwrap();

        let executor = PkgBuildExecutor::new(srcdir, pkgdir, workdir);

        let pkgbuild = PkgBuild {
            pkgname: "failing-pkg".to_string(),
            pkgver: "1.0.0".to_string(),
            pkgrel: "1".to_string(),
            pkgdesc: String::new(),
            arch: vec![],
            license: vec![],
            depends: vec![],
            makedepends: vec![],
            source: vec![],
            source_x86_64: vec![],
            source_aarch64: vec![],
            sha256sums: vec![],
            sha256sums_x86_64: vec![],
            sha256sums_aarch64: vec![],
            noextract: vec![],
            build_fn: Some(
                r#"
                echo "Starting build..."
                exit 42
            "#
                .to_string(),
            ),
            package_fn: None,
            variables: Default::default(),
        };

        let mut output = Vec::new();
        let result = executor
            .build(&pkgbuild, |line| {
                output.push(line.to_string());
            })
            .await;

        assert!(result.is_err(), "Build should fail");

        match result.unwrap_err() {
            ExecutionError::BuildFailed(code) => {
                assert_eq!(code, 42, "Should return correct exit code");
            }
            _ => panic!("Expected BuildFailed error"),
        }

        // Verify output was still captured before failure
        assert!(output.iter().any(|line| line.contains("Starting build")));
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Package function can access build artifacts
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[tokio::test]
    async fn test_package_function_accesses_build_artifacts() {
        let temp = TempDir::new().unwrap();
        let srcdir = temp.path().join("src");
        let pkgdir = temp.path().join("pkg");
        let workdir = temp.path().to_path_buf();

        tokio::fs::create_dir_all(&srcdir).await.unwrap();
        tokio::fs::create_dir_all(&pkgdir).await.unwrap();

        let executor = PkgBuildExecutor::new(srcdir.clone(), pkgdir.clone(), workdir);

        let pkgbuild = PkgBuild {
            pkgname: "artifact-test".to_string(),
            pkgver: "1.0.0".to_string(),
            pkgrel: "1".to_string(),
            pkgdesc: String::new(),
            arch: vec![],
            license: vec![],
            depends: vec![],
            makedepends: vec![],
            source: vec![],
            source_x86_64: vec![],
            source_aarch64: vec![],
            sha256sums: vec![],
            sha256sums_x86_64: vec![],
            sha256sums_aarch64: vec![],
            noextract: vec![],
            build_fn: Some(format!(
                r#"
                echo "Building binary..."
                echo "fake-binary" > {}/test-binary
                "#,
                srcdir.display()
            )),
            package_fn: Some(format!(
                r#"
                echo "Installing binary..."
                mkdir -p {}/usr/local/bin
                cp {}/test-binary {}/usr/local/bin/
                echo "Binary installed"
                "#,
                pkgdir.display(),
                srcdir.display(),
                pkgdir.display()
            )),
            variables: Default::default(),
        };

        let mut output = Vec::new();

        // Build
        executor
            .build(&pkgbuild, |line| {
                output.push(line.to_string());
            })
            .await
            .unwrap();

        // Package
        executor
            .package(&pkgbuild, |line| {
                output.push(line.to_string());
            })
            .await
            .unwrap();

        // Verify binary was "installed"
        let installed_binary = pkgdir.join("usr/local/bin/test-binary");
        assert!(installed_binary.exists(), "Binary should be installed to pkgdir");

        let content = tokio::fs::read_to_string(&installed_binary).await.unwrap();
        assert_eq!(content.trim(), "fake-binary");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: build_and_package runs full pipeline
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[tokio::test]
    async fn test_build_and_package_runs_full_pipeline() {
        let temp = TempDir::new().unwrap();
        let srcdir = temp.path().join("src");
        let pkgdir = temp.path().join("pkg");
        let workdir = temp.path().to_path_buf();

        tokio::fs::create_dir_all(&srcdir).await.unwrap();
        tokio::fs::create_dir_all(&pkgdir).await.unwrap();

        let executor = PkgBuildExecutor::new(srcdir, pkgdir, workdir);

        let pkgbuild = PkgBuild {
            pkgname: "full-pipeline".to_string(),
            pkgver: "1.0.0".to_string(),
            pkgrel: "1".to_string(),
            pkgdesc: String::new(),
            arch: vec![],
            license: vec![],
            depends: vec![],
            makedepends: vec![],
            source: vec![],
            source_x86_64: vec![],
            source_aarch64: vec![],
            sha256sums: vec![],
            sha256sums_x86_64: vec![],
            sha256sums_aarch64: vec![],
            noextract: vec![],
            build_fn: Some("echo 'BUILD_PHASE'".to_string()),
            package_fn: Some("echo 'PACKAGE_PHASE'".to_string()),
            variables: Default::default(),
        };

        let mut output = Vec::new();
        let result = executor
            .build_and_package(&pkgbuild, |line| {
                output.push(line.to_string());
            })
            .await;

        assert!(result.is_ok());

        // Verify both phases ran
        assert!(output.iter().any(|line| line.contains("BUILD_PHASE")));
        assert!(output.iter().any(|line| line.contains("PACKAGE_PHASE")));

        // Verify phase markers
        assert!(output.iter().any(|line| line.contains("Building full-pipeline")));
        assert!(output.iter().any(|line| line.contains("Packaging full-pipeline")));
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Missing build function returns error
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[tokio::test]
    async fn test_missing_build_function_returns_error() {
        let temp = TempDir::new().unwrap();
        let srcdir = temp.path().join("src");
        let pkgdir = temp.path().join("pkg");
        let workdir = temp.path().to_path_buf();

        let executor = PkgBuildExecutor::new(srcdir, pkgdir, workdir);

        let pkgbuild = PkgBuild {
            pkgname: "no-build".to_string(),
            pkgver: "1.0.0".to_string(),
            pkgrel: "1".to_string(),
            pkgdesc: String::new(),
            arch: vec![],
            license: vec![],
            depends: vec![],
            makedepends: vec![],
            source: vec![],
            source_x86_64: vec![],
            source_aarch64: vec![],
            sha256sums: vec![],
            sha256sums_x86_64: vec![],
            sha256sums_aarch64: vec![],
            noextract: vec![],
            build_fn: None, // Missing!
            package_fn: Some("echo 'test'".to_string()),
            variables: Default::default(),
        };

        let result = executor.build(&pkgbuild, |_| {}).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ExecutionError::MissingBuildFunction));
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Missing package function returns error
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[tokio::test]
    async fn test_missing_package_function_returns_error() {
        let temp = TempDir::new().unwrap();
        let srcdir = temp.path().join("src");
        let pkgdir = temp.path().join("pkg");
        let workdir = temp.path().to_path_buf();

        let executor = PkgBuildExecutor::new(srcdir, pkgdir, workdir);

        let pkgbuild = PkgBuild {
            pkgname: "no-package".to_string(),
            pkgver: "1.0.0".to_string(),
            pkgrel: "1".to_string(),
            pkgdesc: String::new(),
            arch: vec![],
            license: vec![],
            depends: vec![],
            makedepends: vec![],
            source: vec![],
            source_x86_64: vec![],
            source_aarch64: vec![],
            sha256sums: vec![],
            sha256sums_x86_64: vec![],
            sha256sums_aarch64: vec![],
            noextract: vec![],
            build_fn: Some("echo 'test'".to_string()),
            package_fn: None, // Missing!
            variables: Default::default(),
        };

        let result = executor.package(&pkgbuild, |_| {}).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ExecutionError::MissingPackageFunction));
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Stderr is captured and prefixed with ERROR
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[tokio::test]
    async fn test_stderr_is_captured_with_error_prefix() {
        let temp = TempDir::new().unwrap();
        let srcdir = temp.path().join("src");
        let pkgdir = temp.path().join("pkg");
        let workdir = temp.path().to_path_buf();

        tokio::fs::create_dir_all(&srcdir).await.unwrap();
        tokio::fs::create_dir_all(&pkgdir).await.unwrap();

        let executor = PkgBuildExecutor::new(srcdir, pkgdir, workdir);

        let pkgbuild = PkgBuild {
            pkgname: "stderr-test".to_string(),
            pkgver: "1.0.0".to_string(),
            pkgrel: "1".to_string(),
            pkgdesc: String::new(),
            arch: vec![],
            license: vec![],
            depends: vec![],
            makedepends: vec![],
            source: vec![],
            source_x86_64: vec![],
            source_aarch64: vec![],
            sha256sums: vec![],
            sha256sums_x86_64: vec![],
            sha256sums_aarch64: vec![],
            noextract: vec![],
            build_fn: Some(
                r#"
                echo "Normal output"
                echo "Warning message" >&2
                echo "More output"
            "#
                .to_string(),
            ),
            package_fn: None,
            variables: Default::default(),
        };

        let mut output = Vec::new();
        executor
            .build(&pkgbuild, |line| {
                output.push(line.to_string());
            })
            .await
            .unwrap();

        // Verify stdout captured
        assert!(output.iter().any(|line| line.contains("Normal output")));
        assert!(output.iter().any(|line| line.contains("More output")));

        // Verify stderr captured without ERROR prefix (TEAM-384: only prefix actual errors)
        assert!(output.iter().any(|line| line.contains("Warning message")));
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Build failure in pipeline stops before package
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[tokio::test]
    async fn test_build_failure_prevents_package_phase() {
        let temp = TempDir::new().unwrap();
        let srcdir = temp.path().join("src");
        let pkgdir = temp.path().join("pkg");
        let workdir = temp.path().to_path_buf();

        tokio::fs::create_dir_all(&srcdir).await.unwrap();
        tokio::fs::create_dir_all(&pkgdir).await.unwrap();

        let executor = PkgBuildExecutor::new(srcdir, pkgdir, workdir);

        let pkgbuild = PkgBuild {
            pkgname: "fail-early".to_string(),
            pkgver: "1.0.0".to_string(),
            pkgrel: "1".to_string(),
            pkgdesc: String::new(),
            arch: vec![],
            license: vec![],
            depends: vec![],
            makedepends: vec![],
            source: vec![],
            source_x86_64: vec![],
            source_aarch64: vec![],
            sha256sums: vec![],
            sha256sums_x86_64: vec![],
            sha256sums_aarch64: vec![],
            noextract: vec![],
            build_fn: Some("exit 1".to_string()),
            package_fn: Some("echo 'SHOULD_NOT_RUN'".to_string()),
            variables: Default::default(),
        };

        let mut output = Vec::new();
        let result = executor
            .build_and_package(&pkgbuild, |line| {
                output.push(line.to_string());
            })
            .await;

        assert!(result.is_err());

        // Verify package phase never ran
        assert!(!output.iter().any(|line| line.contains("SHOULD_NOT_RUN")));
        assert!(!output.iter().any(|line| line.contains("Packaging")));
    }
}
