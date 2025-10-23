//! Queen-rbee command handlers
//!
//! TEAM-276: Extracted from main.rs

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use queen_lifecycle::ensure_queen_running;

use crate::cli::QueenAction;

const NARRATE: NarrationFactory = NarrationFactory::new("keeper");

pub async fn handle_queen(action: QueenAction, queen_url: &str) -> Result<()> {
    match action {
        QueenAction::Start => handle_start(queen_url).await,
        QueenAction::Stop => handle_stop(queen_url).await,
        QueenAction::Status => handle_status(queen_url).await,
        QueenAction::Rebuild { with_local_hive } => handle_rebuild(with_local_hive).await,
        QueenAction::Info => handle_info(queen_url).await,
        QueenAction::Install { binary } => handle_install(binary).await,
        QueenAction::Uninstall => handle_uninstall(queen_url).await,
    }
}

async fn handle_start(queen_url: &str) -> Result<()> {
    let queen_handle = ensure_queen_running(queen_url).await?;
    NARRATE
        .action("queen_start")
        .context(queen_handle.base_url())
        .human("✅ Queen started on {}")
        .emit();
    std::mem::forget(queen_handle);
    Ok(())
}

async fn handle_stop(queen_url: &str) -> Result<()> {
    let client = reqwest::Client::new();

    // First check if queen is running
    let health_check = client.get(format!("{}/health", queen_url)).send().await;

    let is_running = matches!(health_check, Ok(response) if response.status().is_success());

    if !is_running {
        NARRATE.action("queen_stop").human("⚠️  Queen not running").emit();
        return Ok(());
    }

    // Queen is running, send shutdown request
    let shutdown_client = reqwest::Client::builder()
        .timeout(tokio::time::Duration::from_secs(30))
        .build()?;

    match shutdown_client.post(format!("{}/v1/shutdown", queen_url)).send().await {
        Ok(_) => {
            NARRATE.action("queen_stop").human("✅ Queen stopped").emit();
            Ok(())
        }
        Err(e) => {
            // Connection closed/reset is expected - queen shuts down before responding
            if e.is_connect() || e.to_string().contains("connection closed") {
                NARRATE.action("queen_stop").human("✅ Queen stopped").emit();
                Ok(())
            } else {
                // Unexpected error
                NARRATE
                    .action("queen_stop")
                    .context(e.to_string())
                    .human("⚠️  Failed to stop queen: {}")
                    .error_kind("shutdown_failed")
                    .emit();
                Err(e.into())
            }
        }
    }
}

async fn handle_status(queen_url: &str) -> Result<()> {
    // TEAM-186: Check queen-rbee health endpoint
    let client = reqwest::Client::builder()
        .timeout(tokio::time::Duration::from_secs(5))
        .build()?;

    match client.get(format!("{}/health", queen_url)).send().await {
        Ok(response) if response.status().is_success() => {
            NARRATE
                .action("queen_status")
                .context(queen_url)
                .human("✅ Queen is running on {}")
                .emit();

            // Try to get more details from the response
            if let Ok(body) = response.text().await {
                println!("Status: {}", body);
            }
            Ok(())
        }
        Ok(response) => {
            NARRATE
                .action("queen_status")
                .context(response.status().to_string())
                .human("⚠️  Queen responded with status: {}")
                .emit();
            Ok(())
        }
        Err(_) => {
            NARRATE
                .action("queen_status")
                .context(queen_url)
                .human("❌ Queen is not running on {}")
                .emit();
            Ok(())
        }
    }
}

async fn handle_rebuild(with_local_hive: bool) -> Result<()> {
    // TEAM-262: Added queen rebuild command for local-hive optimization
    // TEAM-263: Implemented actual build logic
    NARRATE.action("queen_rebuild").human("🔨 Rebuilding queen-rbee...").emit();

    // Determine build command
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("build").arg("--release").arg("--bin").arg("queen-rbee");

    if with_local_hive {
        NARRATE
            .action("queen_rebuild")
            .human("✨ Building with integrated local hive (50-100x faster localhost)...")
            .emit();
        cmd.arg("--features").arg("local-hive");
    } else {
        NARRATE
            .action("queen_rebuild")
            .human("📡 Building distributed queen (remote hives only)...")
            .emit();
    }

    // Execute build
    NARRATE
        .action("queen_rebuild")
        .human("⏳ Running cargo build (this may take a few minutes)...")
        .emit();

    let output = cmd.output()?;

    if output.status.success() {
        NARRATE.action("queen_rebuild").human("✅ Build successful!").emit();

        // Show binary location
        let binary_path = "target/release/queen-rbee";
        NARRATE
            .action("queen_rebuild")
            .context(binary_path)
            .human("📦 Binary available at: {}")
            .emit();

        if with_local_hive {
            NARRATE
                .action("queen_rebuild")
                .human("💡 Restart queen to use the new binary with local-hive feature")
                .emit();
        }
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        NARRATE
            .action("queen_rebuild")
            .context(stderr.to_string())
            .human("❌ Build failed: {}")
            .error_kind("build_failed")
            .emit();
        anyhow::bail!("Build failed");
    }
}

async fn handle_info(queen_url: &str) -> Result<()> {
    // TEAM-262: Query queen's /v1/build-info endpoint
    let client = reqwest::Client::builder()
        .timeout(tokio::time::Duration::from_secs(5))
        .build()?;

    match client.get(format!("{}/v1/build-info", queen_url)).send().await {
        Ok(response) if response.status().is_success() => {
            NARRATE.action("queen_info").human("📋 Queen build information:").emit();
            if let Ok(body) = response.text().await {
                println!("{}", body);
            }
            Ok(())
        }
        Err(_) => {
            NARRATE
                .action("queen_info")
                .human("❌ Queen is not running or /v1/build-info not available")
                .emit();
            Ok(())
        }
        _ => {
            NARRATE.action("queen_info").human("⚠️  Failed to get build info").emit();
            Ok(())
        }
    }
}

async fn handle_install(binary: Option<String>) -> Result<()> {
    // TEAM-262: Install queen binary
    // TEAM-263: Implemented install logic
    NARRATE.action("queen_install").human("📦 Installing queen-rbee...").emit();

    // Resolve binary path
    let source_path = if let Some(path) = binary {
        NARRATE
            .action("queen_install")
            .context(&path)
            .human("Using provided binary: {}")
            .emit();
        std::path::PathBuf::from(path)
    } else {
        // Auto-detect: try target/release first, then target/debug
        let release_path = std::path::PathBuf::from("target/release/queen-rbee");
        let debug_path = std::path::PathBuf::from("target/debug/queen-rbee");

        if release_path.exists() {
            NARRATE
                .action("queen_install")
                .context("target/release/queen-rbee")
                .human("Found binary: {}")
                .emit();
            release_path
        } else if debug_path.exists() {
            NARRATE
                .action("queen_install")
                .context("target/debug/queen-rbee")
                .human("Found binary: {}")
                .emit();
            debug_path
        } else {
            NARRATE
                .action("queen_install")
                .human("❌ No binary found. Run 'rbee-keeper queen rebuild' first")
                .error_kind("binary_not_found")
                .emit();
            anyhow::bail!("Binary not found");
        }
    };

    // Verify binary exists
    if !source_path.exists() {
        NARRATE
            .action("queen_install")
            .context(source_path.display().to_string())
            .human("❌ Binary not found: {}")
            .error_kind("binary_not_found")
            .emit();
        anyhow::bail!("Binary not found");
    }

    // Determine install location (~/.local/bin/queen-rbee)
    let home = std::env::var("HOME")?;
    let install_dir = std::path::PathBuf::from(format!("{}/.local/bin", home));
    let install_path = install_dir.join("queen-rbee");

    // Create install directory if needed
    std::fs::create_dir_all(&install_dir)?;

    // Copy binary
    NARRATE
        .action("queen_install")
        .context(install_path.display().to_string())
        .human("📋 Installing to: {}")
        .emit();

    std::fs::copy(&source_path, &install_path)?;

    // Make executable (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&install_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&install_path, perms)?;
    }

    NARRATE.action("queen_install").human("✅ Queen installed successfully!").emit();
    NARRATE
        .action("queen_install")
        .context(install_path.display().to_string())
        .human("📍 Binary location: {}")
        .emit();
    NARRATE
        .action("queen_install")
        .human("💡 Make sure ~/.local/bin is in your PATH")
        .emit();

    Ok(())
}

async fn handle_uninstall(queen_url: &str) -> Result<()> {
    // TEAM-262: Uninstall queen binary
    // TEAM-263: Implemented uninstall logic
    NARRATE.action("queen_uninstall").human("🗑️  Uninstalling queen-rbee...").emit();

    // Determine install location
    let home = std::env::var("HOME")?;
    let install_path = std::path::PathBuf::from(format!("{}/.local/bin/queen-rbee", home));

    // Check if binary exists
    if !install_path.exists() {
        NARRATE
            .action("queen_uninstall")
            .context(install_path.display().to_string())
            .human("⚠️  Queen not installed at: {}")
            .emit();
        return Ok(());
    }

    // Check if queen is running
    let client = reqwest::Client::builder()
        .timeout(tokio::time::Duration::from_secs(2))
        .build()?;

    let is_running = matches!(
        client.get(format!("{}/health", queen_url)).send().await,
        Ok(response) if response.status().is_success()
    );

    if is_running {
        NARRATE
            .action("queen_uninstall")
            .human("⚠️  Queen is currently running. Stop it first with: rbee-keeper queen stop")
            .emit();
        anyhow::bail!("Queen is running");
    }

    // Remove binary
    std::fs::remove_file(&install_path)?;

    NARRATE
        .action("queen_uninstall")
        .human("✅ Queen uninstalled successfully!")
        .emit();
    NARRATE
        .action("queen_uninstall")
        .context(install_path.display().to_string())
        .human("🗑️  Removed: {}")
        .emit();

    Ok(())
}
