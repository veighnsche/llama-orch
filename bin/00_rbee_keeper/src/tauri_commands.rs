//! Tauri commands for the rbee-keeper GUI
//!
//! TEAM-293: Created Tauri command wrappers for all CLI operations
//! TEAM-297: Updated to use specta v2 for proper TypeScript type generation
//! TEAM-334: Cleaned up - only keeping ssh_list, rest to be re-implemented later
//!
//! Currently only exposes SSH config parsing. Other commands will be added
//! as the architecture stabilizes.

use crate::ssh_resolver;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use super::*;
    use specta_typescript::Typescript;
    use tauri_specta::{collect_commands, Builder};

    #[test]
    fn export_typescript_bindings() {
        // TEAM-297: Test that exports TypeScript bindings
        // TEAM-333: Updated to use ssh_list command
        // TEAM-335: Added queen lifecycle commands
        // TEAM-336: Include NarrationEvent type for frontend + test command
        // TEAM-338: RULE ZERO FIX - Use DaemonStatus from daemon-lifecycle (deleted QueenStatus/HiveStatus)
        //
        // NOTE: NarrationEvent is NOT a tauri-specta Event - it's emitted from a
        // custom tracing layer using Tauri's Emitter trait. We export it as an
        // extra type so TypeScript can listen to "narration" events with proper typing.
        use crate::tracing_init::NarrationEvent;

        let builder = Builder::<tauri::Wry>::new()
            .commands(collect_commands![
                test_narration,
                ssh_list,
                get_installed_hives,
                ssh_open_config,
                queen_status,
                queen_start,
                queen_stop,
                queen_install,
                queen_rebuild,
                queen_uninstall,
                hive_start,
                hive_stop,
                hive_status,
                hive_install,
                hive_uninstall,
                hive_rebuild,
            ])
            .typ::<NarrationEvent>()
            .typ::<lifecycle_local::DaemonStatus>();

        builder
            .export(Typescript::default(), "ui/src/generated/bindings.ts")
            .expect("Failed to export typescript bindings");
    }
}

// ============================================================================
// SSH TYPES
// ============================================================================

/// SSH target from ~/.ssh/config
///
/// TEAM-333: Type for SSH config entries with specta support for TypeScript bindings
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
pub struct SshTarget {
    /// Host alias from SSH config
    pub host: String,
    /// Host subtitle (optional)
    pub host_subtitle: Option<String>,
    /// Hostname (IP or domain)
    pub hostname: String,
    /// SSH username
    pub user: String,
    /// SSH port
    pub port: u16,
    /// Connection status
    pub status: SshTargetStatus,
}

/// SSH target connection status
///
/// TEAM-333: Status enum for SSH targets
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "lowercase")]
pub enum SshTargetStatus {
    Online,
    Offline,
    Unknown,
}

// ============================================================================
// QUEEN COMMANDS
// ============================================================================

/// Get queen-rbee daemon status
/// TEAM-338: Returns structured status (isRunning, isInstalled)
/// TEAM-338: RULE ZERO FIX - Use DaemonStatus directly (deleted QueenStatus duplicate)
#[tauri::command]
#[specta::specta]
pub async fn queen_status() -> Result<lifecycle_local::DaemonStatus, String> {
    use crate::Config;
    use lifecycle_local::{check_daemon_health, SshConfig};

    let config = Config::load().map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();

    // Check status (running + installed)
    let health_url = format!("{}/health", queen_url);
    let ssh_config = SshConfig::localhost(); // Queen is always localhost

    Ok(check_daemon_health(&health_url, "queen-rbee", &ssh_config).await)
}

/// Start queen-rbee daemon on localhost
/// TEAM-335: Thin wrapper around handle_queen() - business logic in handlers/queen.rs
#[tauri::command]
#[specta::specta]
pub async fn queen_start() -> Result<String, String> {
    use crate::cli::QueenAction;
    use crate::handlers::handle_queen;
    use crate::Config;
    use observability_narration_core::n;

    n!("queen_start", "🚀 Starting queen from Tauri GUI");

    let config = Config::load().map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();

    handle_queen(QueenAction::Start, &queen_url)
        .await
        .map(|_| "Queen started successfully".to_string())
        .map_err(|e| format!("{}", e))
}

/// Stop queen-rbee daemon
/// TEAM-335: Thin wrapper around handle_queen()
#[tauri::command]
#[specta::specta]
pub async fn queen_stop() -> Result<String, String> {
    use crate::cli::QueenAction;
    use crate::handlers::handle_queen;
    use crate::Config;

    let config = Config::load().map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();

    handle_queen(QueenAction::Stop, &queen_url)
        .await
        .map(|_| "Queen stopped successfully".to_string())
        .map_err(|e| format!("{}", e))
}

/// Install queen-rbee binary
/// TEAM-335: Thin wrapper around handle_queen()
#[tauri::command]
#[specta::specta]
pub async fn queen_install(binary: Option<String>) -> Result<String, String> {
    use crate::cli::QueenAction;
    use crate::handlers::handle_queen;
    use crate::Config;

    let config = Config::load().map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();

    handle_queen(QueenAction::Install { binary }, &queen_url)
        .await
        .map(|_| "Queen installed successfully".to_string())
        .map_err(|e| format!("{}", e))
}

/// Rebuild queen-rbee from source
/// TEAM-335: Thin wrapper around handle_queen()
#[tauri::command]
#[specta::specta]
pub async fn queen_rebuild(_with_local_hive: bool) -> Result<String, String> {
    use crate::cli::QueenAction;
    use crate::handlers::handle_queen;
    use crate::Config;

    let config = Config::load().map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();

    handle_queen(QueenAction::Rebuild, &queen_url)
        .await
        .map(|_| "Queen rebuilt successfully".to_string())
        .map_err(|e| format!("{}", e))
}

/// Uninstall queen-rbee binary
/// TEAM-335: Thin wrapper around handle_queen()
#[tauri::command]
#[specta::specta]
pub async fn queen_uninstall() -> Result<String, String> {
    use crate::cli::QueenAction;
    use crate::handlers::handle_queen;
    use crate::Config;

    let config = Config::load().map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();

    handle_queen(QueenAction::Uninstall, &queen_url)
        .await
        .map(|_| "Queen uninstalled successfully".to_string())
        .map_err(|e| format!("{}", e))
}

// ============================================================================
// TEST COMMANDS
// ============================================================================

/// Test narration event emission
/// TEAM-336: Debug command to verify narration pipeline works
#[tauri::command]
#[specta::specta]
pub async fn test_narration() -> Result<String, String> {
    use observability_narration_core::n;

    n!("test_narration", "🎯 Test narration event from Tauri command");
    tracing::info!("This is a tracing::info! event");
    tracing::warn!("This is a tracing::warn! event");
    tracing::error!("This is a tracing::error! event");

    Ok("Narration test events emitted - check the panel!".to_string())
}

// ============================================================================
// HIVE COMMANDS
// ============================================================================

/// Start rbee-hive daemon
/// TEAM-338: Thin wrapper around handle_hive()
#[tauri::command]
#[specta::specta]
pub async fn hive_start(alias: String) -> Result<String, String> {
    use crate::cli::HiveAction;
    use crate::handlers::handle_hive;
    use crate::Config;

    let config = Config::load().map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();

    handle_hive(HiveAction::Start { alias, port: None }, &queen_url)
        .await
        .map(|_| "Hive started successfully".to_string())
        .map_err(|e| format!("{}", e))
}

/// Stop rbee-hive daemon
/// TEAM-338: Thin wrapper around handle_hive()
#[tauri::command]
#[specta::specta]
pub async fn hive_stop(alias: String) -> Result<String, String> {
    use crate::cli::HiveAction;
    use crate::handlers::handle_hive;
    use crate::Config;

    let config = Config::load().map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();

    handle_hive(HiveAction::Stop { alias, port: None }, &queen_url)
        .await
        .map(|_| "Hive stopped successfully".to_string())
        .map_err(|e| format!("{}", e))
}

/// Check rbee-hive status
/// TEAM-338: Returns structured status (isRunning, isInstalled)
/// TEAM-338: RULE ZERO FIX - Use DaemonStatus directly (deleted HiveStatus duplicate)
/// TEAM-342: Added narration for visibility in UI
#[tauri::command]
#[specta::specta]
pub async fn hive_status(alias: String) -> Result<lifecycle_ssh::DaemonStatus, String> {
    use crate::ssh_resolver::resolve_ssh_config;
    use lifecycle_ssh::check_daemon_health;
    use observability_narration_core::n;

    // TEAM-342: Narrate status check start
    n!("hive_status_check", "🔍 Checking status for hive '{}'", alias);

    // Resolve SSH config for this hive (localhost or ~/.ssh/config)
    let ssh = resolve_ssh_config(&alias)
        .map_err(|e| format!("Failed to resolve SSH config for '{}': {}", alias, e))?;

    // Check status (running + installed)
    let health_url = format!("http://{}:7835/health", ssh.hostname);

    let status = check_daemon_health(&health_url, "rbee-hive", &ssh).await;

    // TEAM-342: Narrate status result
    if status.is_running {
        n!("hive_status_running", "✅ Hive '{}' is running", alias);
    } else if status.is_installed {
        n!("hive_status_stopped", "⏸️  Hive '{}' is installed but not running", alias);
    } else {
        n!("hive_status_not_installed", "❌ Hive '{}' is not installed", alias);
    }

    Ok(status)
}

/// Install rbee-hive binary
/// TEAM-338: Thin wrapper around handle_hive()
#[tauri::command]
#[specta::specta]
pub async fn hive_install(alias: String) -> Result<String, String> {
    use crate::cli::HiveAction;
    use crate::handlers::handle_hive;
    use crate::Config;

    let config = Config::load().map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();

    handle_hive(HiveAction::Install { alias }, &queen_url)
        .await
        .map(|_| "Hive installed successfully".to_string())
        .map_err(|e| format!("{}", e))
}

/// Uninstall rbee-hive binary
/// TEAM-338: Thin wrapper around handle_hive()
#[tauri::command]
#[specta::specta]
pub async fn hive_uninstall(alias: String) -> Result<String, String> {
    use crate::cli::HiveAction;
    use crate::handlers::handle_hive;
    use crate::Config;

    let config = Config::load().map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();

    handle_hive(HiveAction::Uninstall { alias }, &queen_url)
        .await
        .map(|_| "Hive uninstalled successfully".to_string())
        .map_err(|e| format!("{}", e))
}

/// Rebuild rbee-hive from source
/// TEAM-338: Thin wrapper around handle_hive()
#[tauri::command]
#[specta::specta]
pub async fn hive_rebuild(alias: String) -> Result<String, String> {
    use crate::cli::HiveAction;
    use crate::handlers::handle_hive;
    use crate::Config;

    let config = Config::load().map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();

    handle_hive(HiveAction::Rebuild { alias }, &queen_url)
        .await
        .map(|_| "Hive rebuilt successfully".to_string())
        .map_err(|e| format!("{}", e))
}

// ============================================================================
// SSH COMMANDS
// ============================================================================

/// Open SSH config file in default text editor
/// TEAM-338: Opens ~/.ssh/config with system default editor
#[tauri::command]
#[specta::specta]
pub async fn ssh_open_config() -> Result<String, String> {
    use observability_narration_core::n;
    use std::process::Command;

    n!("ssh_open_config", "Opening SSH config in default editor");

    let home =
        std::env::var("HOME").map_err(|_| "HOME environment variable not set".to_string())?;
    let ssh_config_path = std::path::PathBuf::from(home).join(".ssh/config");

    // Create .ssh directory if it doesn't exist
    if let Some(parent) = ssh_config_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create .ssh directory: {}", e))?;
    }

    // Create empty config file if it doesn't exist
    if !ssh_config_path.exists() {
        std::fs::write(&ssh_config_path, "")
            .map_err(|e| format!("Failed to create SSH config file: {}", e))?;
    }

    // Open with default editor (xdg-open on Linux, open on macOS, notepad on Windows)
    #[cfg(target_os = "linux")]
    let status = Command::new("xdg-open")
        .arg(&ssh_config_path)
        .spawn()
        .map_err(|e| format!("Failed to open editor: {}", e))?;

    #[cfg(target_os = "macos")]
    let status = Command::new("open")
        .arg(&ssh_config_path)
        .spawn()
        .map_err(|e| format!("Failed to open editor: {}", e))?;

    #[cfg(target_os = "windows")]
    let status = Command::new("notepad.exe")
        .arg(&ssh_config_path)
        .spawn()
        .map_err(|e| format!("Failed to open editor: {}", e))?;

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    return Err("Unsupported operating system".to_string());

    drop(status); // Don't wait for editor to close

    Ok(format!("Opened SSH config: {}", ssh_config_path.display()))
}

#[tauri::command]
#[specta::specta]
pub async fn ssh_list() -> Result<Vec<SshTarget>, String> {
    // TEAM-333: Parse ~/.ssh/config and return list of SSH targets
    // TEAM-333: Deduplicate by hostname, keep shortest host alias
    use observability_narration_core::n;
    use std::collections::HashMap;

    n!("ssh_list", "Reading SSH config");

    // Get SSH config path
    let home =
        std::env::var("HOME").map_err(|_| "HOME environment variable not set".to_string())?;
    let ssh_config_path = std::path::PathBuf::from(home).join(".ssh/config");

    // Parse SSH config
    let hosts = ssh_resolver::parse_ssh_config(&ssh_config_path)
        .map_err(|e| format!("Failed to parse SSH config: {}", e))?;

    // TEAM-333: Deduplicate by hostname - collect all aliases first, then pick shortest
    let mut by_hostname: HashMap<String, Vec<String>> = HashMap::new();
    let mut configs_map: HashMap<String, lifecycle_ssh::SshConfig> = HashMap::new();

    // First pass: collect all aliases for each unique hostname
    for (host, config) in hosts {
        let key = format!("{}:{}@{}", config.hostname, config.port, config.user);
        by_hostname.entry(key.clone()).or_insert_with(Vec::new).push(host);
        configs_map.insert(key, config);
    }

    // Second pass: for each hostname, pick shortest alias and use others as subtitle
    let mut targets: Vec<SshTarget> = Vec::new();
    for (key, mut aliases) in by_hostname {
        // Sort aliases by length (shortest first)
        aliases.sort_by_key(|a| a.len());

        let config = configs_map.get(&key).unwrap();
        let primary = aliases[0].clone();
        let subtitle = if aliases.len() > 1 { Some(aliases[1..].join(", ")) } else { None };

        targets.push(SshTarget {
            host: primary,
            host_subtitle: subtitle,
            hostname: config.hostname.clone(),
            user: config.user.clone(),
            port: config.port,
            status: SshTargetStatus::Unknown,
        });
    }

    // Sort by host name
    targets.sort_by(|a, b| a.host.cmp(&b.host));

    // TEAM-360: Add localhost as an available target
    // Localhost is always available for installation
    targets.insert(0, SshTarget {
        host: "localhost".to_string(),
        host_subtitle: Some("This machine".to_string()),
        hostname: "localhost".to_string(),
        user: std::env::var("USER").unwrap_or_else(|_| "user".to_string()),
        port: 22,
        status: SshTargetStatus::Unknown,
    });

    n!("ssh_list", "Found {} unique SSH targets (including localhost)", targets.len());

    Ok(targets)
}

/// TEAM-367: Get list of all installed hives (checks actual status from backend)
/// Returns list of hive IDs that are actually installed
#[tauri::command]
#[specta::specta]
pub async fn get_installed_hives() -> Result<Vec<String>, String> {
    use observability_narration_core::n;
    
    n!("get_installed_hives", "Checking which hives are installed");
    
    let mut installed = Vec::new();
    
    // Check localhost
    match hive_status("localhost".to_string()).await {
        Ok(status) => {
            if status.is_installed {
                installed.push("localhost".to_string());
                n!("get_installed_hives", "localhost is installed");
            }
        }
        Err(e) => {
            n!("get_installed_hives", "Failed to check localhost: {}", e);
        }
    }
    
    // Check all SSH targets
    match ssh_list().await {
        Ok(targets) => {
            for target in targets {
                if target.host == "localhost" {
                    continue; // Already checked
                }
                
                match hive_status(target.host.clone()).await {
                    Ok(status) => {
                        if status.is_installed {
                            installed.push(target.host.clone());
                            n!("get_installed_hives", "{} is installed", target.host);
                        }
                    }
                    Err(e) => {
                        n!("get_installed_hives", "Failed to check {}: {}", target.host, e);
                    }
                }
            }
        }
        Err(e) => {
            n!("get_installed_hives", "Failed to get SSH targets: {}", e);
        }
    }
    
    n!("get_installed_hives", "Found {} installed hives", installed.len());
    Ok(installed)
}
